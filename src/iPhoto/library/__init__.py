"""Basic Library management helpers."""

from .tree import AlbumNode


def __getattr__(name):
    if name in ("GeotaggedAsset", "LibraryManager"):
        from .manager import GeotaggedAsset, LibraryManager
        _exports = {"GeotaggedAsset": GeotaggedAsset, "LibraryManager": LibraryManager}
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AlbumNode", "GeotaggedAsset", "LibraryManager"]
