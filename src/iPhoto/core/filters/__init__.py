"""Modular image filtering package for non-destructive photo editing.

This package provides image adjustment functionality through a clean separation
of concerns:
- algorithms: Pure mathematical functions for image processing
- executors: Different implementation strategies (JIT, Pillow, NumPy, fallback)
- utils: Platform-specific utilities
"""

from __future__ import annotations


def __getattr__(name):
    if name == "apply_adjustments":
        from .facade import apply_adjustments
        return apply_adjustments
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["apply_adjustments"]
