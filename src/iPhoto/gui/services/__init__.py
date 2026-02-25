"""Service layer bridging the GUI facade with domain-specific workflows."""

_SERVICE_NAMES = {
    "AlbumMetadataService",
    "AssetImportService",
    "AssetMoveService",
    "DeletionService",
    "LibraryUpdateService",
    "MoveOperationResult",
    "RestorationService",
}


def __getattr__(name):
    if name in _SERVICE_NAMES:
        if name == "AlbumMetadataService":
            from .album_metadata_service import AlbumMetadataService
            return AlbumMetadataService
        elif name == "AssetImportService":
            from .asset_import_service import AssetImportService
            return AssetImportService
        elif name == "AssetMoveService":
            from .asset_move_service import AssetMoveService
            return AssetMoveService
        elif name == "DeletionService":
            from .deletion_service import DeletionService
            return DeletionService
        elif name in ("LibraryUpdateService", "MoveOperationResult"):
            from .library_update_service import LibraryUpdateService, MoveOperationResult
            return {"LibraryUpdateService": LibraryUpdateService, "MoveOperationResult": MoveOperationResult}[name]
        elif name == "RestorationService":
            from .restoration_service import RestorationService
            return RestorationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AlbumMetadataService",
    "AssetImportService",
    "AssetMoveService",
    "DeletionService",
    "LibraryUpdateService",
    "MoveOperationResult",
    "RestorationService",
]
