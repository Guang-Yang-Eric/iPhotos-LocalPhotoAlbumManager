"""Expose Qt models used by the GUI."""

from .roles import Roles
from .edit_session import EditSession

_MODEL_NAMES = {"AlbumTreeModel", "AlbumTreeRole", "NodeType"}


def __getattr__(name):
    if name in _MODEL_NAMES:
        from .album_tree_model import AlbumTreeModel, AlbumTreeRole, NodeType
        _exports = {
            "AlbumTreeModel": AlbumTreeModel,
            "AlbumTreeRole": AlbumTreeRole,
            "NodeType": NodeType,
        }
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AlbumTreeModel",
    "AlbumTreeRole",
    "NodeType",
    "Roles",
    "EditSession",
]
