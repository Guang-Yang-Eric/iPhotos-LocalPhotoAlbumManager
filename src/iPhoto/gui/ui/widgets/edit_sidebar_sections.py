"""Reusable section UI helpers for EditSidebar."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QWidget


def build_separator(parent: QWidget) -> QFrame:
    """Return a subtle divider separating adjacent section headers."""

    separator = QFrame(parent)
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setFrameShadow(QFrame.Shadow.Plain)
    separator.setStyleSheet("QFrame { background-color: palette(mid); }")
    separator.setFixedHeight(1)
    return separator

