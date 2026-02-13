"""Fullscreen/backdrop helpers for GL image viewer."""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor


def resolve_viewer_surface_color(widget: Any) -> QColor:
    """Resolve the preferred viewer backdrop colour with a safe fallback."""

    try:
        from ...palette import viewer_surface_color  # type: ignore

        return QColor(viewer_surface_color(widget))
    except Exception:
        return QColor(0, 0, 0)
