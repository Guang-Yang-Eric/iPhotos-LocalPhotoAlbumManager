"""GUI entry point for the iPhoto desktop application."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def _setup_tooltip_palette(app: QApplication) -> None:
    """Configure an opaque tooltip palette to prevent translucent artefacts."""

    tooltip_palette = QPalette(app.palette())

    def _resolved_colour(source: QColor, fallback: QColor) -> QColor:
        if not source.isValid():
            return QColor(fallback)
        resolved = QColor(source)
        resolved.setAlpha(255)
        return resolved

    base_colour = _resolved_colour(
        tooltip_palette.color(QPalette.ColorRole.Window), QColor("#eef3f6")
    )
    text_colour = _resolved_colour(
        tooltip_palette.color(QPalette.ColorRole.WindowText), QColor(Qt.GlobalColor.black)
    )

    if abs(base_colour.lightness() - text_colour.lightness()) < 40:
        base_colour = QColor("#eef3f6")
        text_colour = QColor(Qt.GlobalColor.black)

    tooltip_palette.setColor(QPalette.ColorRole.ToolTipBase, base_colour)
    tooltip_palette.setColor(QPalette.ColorRole.ToolTipText, text_colour)
    app.setPalette(tooltip_palette, "QToolTip")


def _build_container(context):
    """Build and populate the DI container (imported lazily)."""

    from iPhoto.di.container import DependencyContainer
    from iPhoto.events.bus import EventBus
    from iPhoto.infrastructure.db.pool import ConnectionPool
    from iPhoto.domain.repositories import IAlbumRepository, IAssetRepository
    from iPhoto.infrastructure.repositories.sqlite_album_repository import SQLiteAlbumRepository
    from iPhoto.infrastructure.repositories.sqlite_asset_repository import SQLiteAssetRepository
    from iPhoto.infrastructure.services.metadata_provider import ExifToolMetadataProvider
    from iPhoto.infrastructure.services.thumbnail_generator import PillowThumbnailGenerator
    from iPhoto.application.interfaces import IMetadataProvider, IThumbnailGenerator
    from iPhoto.application.use_cases.open_album import OpenAlbumUseCase
    from iPhoto.application.use_cases.scan_album import ScanAlbumUseCase
    from iPhoto.application.use_cases.pair_live_photos import PairLivePhotosUseCase
    from iPhoto.application.services.album_service import AlbumService
    from iPhoto.application.services.asset_service import AssetService

    container = DependencyContainer()

    # 1. Event Bus
    container.register_singleton(EventBus)

    # 2. Database Connection Pool
    db_path = Path.home() / ".iPhoto" / "global_index.db"
    if context.library.root():
        db_path = context.library.root() / ".iPhoto" / "global_index.db"

    db_path.parent.mkdir(parents=True, exist_ok=True)

    pool = ConnectionPool(db_path)
    container.register_instance(ConnectionPool, pool)

    # 3. Repositories
    container.register_factory(IAlbumRepository, lambda: SQLiteAlbumRepository(pool), singleton=True)
    container.register_factory(IAssetRepository, lambda: SQLiteAssetRepository(pool), singleton=True)

    # 4. Infrastructure Services
    container.register_singleton(IMetadataProvider, ExifToolMetadataProvider)
    container.register_singleton(IThumbnailGenerator, PillowThumbnailGenerator)

    # 5. Services & Use Cases
    album_repo = container.resolve(IAlbumRepository)
    asset_repo = container.resolve(IAssetRepository)
    event_bus = container.resolve(EventBus)
    metadata_provider = container.resolve(IMetadataProvider)
    thumbnail_generator = container.resolve(IThumbnailGenerator)

    open_uc = OpenAlbumUseCase(album_repo, asset_repo, event_bus)
    scan_uc = ScanAlbumUseCase(album_repo, asset_repo, event_bus, metadata_provider, thumbnail_generator)
    pair_uc = PairLivePhotosUseCase(asset_repo, event_bus)

    container.register_factory(AlbumService, lambda: AlbumService(open_uc, scan_uc, pair_uc), singleton=True)
    container.register_factory(AssetService, lambda: AssetService(asset_repo), singleton=True)

    return container


def main(argv: list[str] | None = None) -> int:
    """Launch the Qt application and return the exit code."""

    arguments = list(sys.argv if argv is None else argv)
    app = QApplication(arguments)

    _setup_tooltip_palette(app)

    # --- Lazy imports: defer heavy modules until after QApplication is created ---
    from iPhoto.appctx import AppContext
    from iPhoto.gui.ui.main_window import MainWindow
    from iPhoto.gui.coordinators.main_coordinator import MainCoordinator

    context = AppContext()
    container = _build_container(context)

    # --- Phase 4: Coordinator Wiring ---
    window = MainWindow(context)

    coordinator = MainCoordinator(window, context, container)

    window.set_coordinator(coordinator)

    coordinator.start()
    window.show()

    # Allow opening an album directly via argv[1].
    if len(arguments) > 1:
        coordinator.open_album_from_path(Path(arguments[1]))
    else:
        window.ui.sidebar.select_all_photos(emit_signal=True)

    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual launch
    raise SystemExit(main())
