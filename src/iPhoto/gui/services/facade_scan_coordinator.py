"""Coordinator encapsulating scan-related facade logic."""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

from ...config import DEFAULT_INCLUDE, DEFAULT_EXCLUDE
from ...models.album import Album
from ...utils.logging import get_logger

if TYPE_CHECKING:
    from ...library.manager import LibraryManager
    from .library_update_service import LibraryUpdateService


class FacadeScanCoordinator:
    """Coordinate album scanning operations on behalf of the facade."""

    def __init__(
        self,
        *,
        library_manager_getter: Callable[[], Optional["LibraryManager"]],
        library_update_service: "LibraryUpdateService",
        require_album_fn: Callable[[], Optional[Album]],
    ) -> None:
        self._library_manager_getter = library_manager_getter
        self._library_update_service = library_update_service
        self._require_album_fn = require_album_fn
        self._logger = get_logger()

    def rescan_current(self) -> List[dict]:
        """Rescan the active album synchronously."""

        album = self._require_album_fn()
        if album is None:
            return []
        return self._library_update_service.rescan_album(album)

    def rescan_current_async(self) -> None:
        """Start a background rescan for the active album."""

        album = self._require_album_fn()
        if album is None:
            return

        library = self._library_manager_getter()
        if library:
            filters = album.manifest.get("filters", {}) if isinstance(album.manifest, dict) else {}
            include = filters.get("include", DEFAULT_INCLUDE)
            exclude = filters.get("exclude", DEFAULT_EXCLUDE)

            library.start_scanning(album.root, include, exclude)
        else:
            self._library_update_service.rescan_album_async(album)

    def cancel_active_scans(self) -> None:
        """Request cancellation of any in-flight scan operations."""

        library = self._library_manager_getter()
        if library is not None:
            try:
                library.stop_scanning()
                library.pause_watcher()
            except RuntimeError:
                self._logger.warning("Failed to stop active scan during shutdown", exc_info=True)

        self._library_update_service.cancel_active_scan()

    def pair_live_current(self) -> List[dict]:
        """Rebuild Live Photo pairings for the active album."""

        album = self._require_album_fn()
        if album is None:
            return []
        return self._library_update_service.pair_live(album)
