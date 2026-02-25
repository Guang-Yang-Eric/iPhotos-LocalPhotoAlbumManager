"""Background tasks and workers."""

from __future__ import annotations

_LAZY_EXPORTS = {
    "AssetLoaderWorker": ".asset_loader_worker",
    "EditSidebarPreviewWorker": ".edit_sidebar_preview_worker",
    "ImageLoadWorker": ".image_load_worker",
    "ImportSignals": ".import_worker",
    "ImportWorker": ".import_worker",
    "IncrementalRefreshSignals": ".incremental_refresh_worker",
    "IncrementalRefreshWorker": ".incremental_refresh_worker",
    "MoveSignals": ".move_worker",
    "MoveWorker": ".move_worker",
    "PreviewRenderSignals": ".preview_render_worker",
    "PreviewRenderWorker": ".preview_render_worker",
    "ThumbnailGeneratorWorker": ".thumbnail_generator_worker",
    "ThumbnailLoader": ".thumbnail_loader",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib
        module = importlib.import_module(_LAZY_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS.keys())
