"""Tone mapping helpers powering the non-destructive edit pipeline.

This module has been refactored into a modular package structure under
iPhoto.core.filters for improved maintainability. This file now serves
as a compatibility layer, re-exporting the main API.
"""

from __future__ import annotations

from .light_resolver import LIGHT_KEYS


def __getattr__(name):
    if name == "apply_adjustments":
        from .filters import apply_adjustments
        return apply_adjustments
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LIGHT_KEYS", "apply_adjustments"]
