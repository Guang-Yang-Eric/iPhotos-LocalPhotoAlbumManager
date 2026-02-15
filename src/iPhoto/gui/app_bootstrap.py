"""Application bootstrap — extracts DI resolution and ViewModel creation from MainCoordinator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from iPhoto.di.container import DependencyContainer
from iPhoto.events.bus import EventBus
from iPhoto.application.services.album_service import AlbumService
from iPhoto.application.services.asset_service import AssetService
from iPhoto.domain.repositories import IAssetRepository
from iPhoto.infrastructure.services.thumbnail_cache_service import ThumbnailCacheService
from iPhoto.gui.viewmodels.asset_list_viewmodel import AssetListViewModel
from iPhoto.gui.viewmodels.asset_data_source import AssetDataSource

if TYPE_CHECKING:
    from iPhoto.appctx import AppContext


class AppBootstrap:
    """Resolve services from the DI container and create ViewModels.

    Extracted from ``MainCoordinator.__init__`` to keep the coordinator
    focused on page-level orchestration (≤200 lines goal).
    """

    def __init__(self, container: DependencyContainer, context: "AppContext") -> None:
        self._logger = logging.getLogger(__name__)

        if container is None:
            raise RuntimeError("DependencyContainer is required for AppBootstrap")

        # Resolve core services
        self.event_bus: EventBus = container.resolve(EventBus)
        self.album_service: AlbumService = container.resolve(AlbumService)
        self.asset_service: AssetService = container.resolve(AssetService)
        self.asset_repo: IAssetRepository = container.resolve(IAssetRepository)

        # ViewModel setup
        lib_root = context.library.root()
        self.asset_data_source = AssetDataSource(self.asset_repo, lib_root)

        cache_root = Path.home() / ".iPhoto" / "cache" / "thumbs"
        if lib_root:
            cache_root = lib_root / ".iPhoto" / "cache" / "thumbs"

        self.thumbnail_service = ThumbnailCacheService(cache_root)
        self.asset_list_vm = AssetListViewModel(
            self.asset_data_source, self.thumbnail_service
        )
