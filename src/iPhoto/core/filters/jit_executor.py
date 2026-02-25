"""JIT-accelerated image adjustment executor using Numba.

This module provides the fastest execution path for image adjustments,
using Numba JIT compilation to process pixels directly in the QImage buffer.
It gracefully falls back to AOT compiled extensions or NumPy vectorization
when runtime JIT is unavailable (e.g., in Nuitka packaged builds).
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

from PySide6.QtGui import QImage

from .utils import _resolve_pixel_buffer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _resolve_executors():
    """Lazily resolve the best available adjustment kernels.

    Returns (apply_adjustments_fast, apply_color_adjustments_inplace).
    """

    _IN_BUILD_MODE = os.environ.get("IPHOTO_BUILD_AOT") == "1"

    # 1. Try AOT
    if not _IN_BUILD_MODE:
        try:
            from . import _jit_compiled

            logger.info("Loaded AOT-compiled image filters.")
            return (
                _jit_compiled._apply_adjustments_fast,
                _jit_compiled._apply_color_adjustments_inplace,
            )
        except ImportError:
            logger.debug("AOT compiled module not found.")

    # 2. Try JIT
    _IS_COMPILED = (
        "__compiled__" in globals()
        or hasattr(sys, "frozen")
        or hasattr(sys, "_MEIPASS")
    )

    if not _IS_COMPILED:
        try:
            from numba import jit  # noqa: F401

            logger.debug("Using runtime Numba JIT compilation.")
            from .jit_kernels import (
                _apply_adjustments_fast,
                _apply_color_adjustments_inplace,
            )
            return _apply_adjustments_fast, _apply_color_adjustments_inplace
        except ImportError:
            logger.debug("Numba is not available; falling back to NumPy implementation.")

    # 3. Fallback: NumPy Vectorization
    logger.warning(
        "AOT module missing and runtime JIT unavailable. "
        "Falling back to slower NumPy implementation.",
    )

    from .numpy_executor import (
        apply_adjustments_buffer as _apply_adjustments_numpy,
        apply_color_adjustments_inplace_buffer as _apply_color_adjustments_numpy,
    )
    return _apply_adjustments_numpy, _apply_color_adjustments_numpy


def apply_adjustments_fast_qimage(
    image: QImage,
    width: int,
    height: int,
    bytes_per_line: int,
    exposure_term: float,
    brightness_term: float,
    brilliance_strength: float,
    highlights: float,
    shadows: float,
    contrast_factor: float,
    black_point: float,
    saturation: float,
    vibrance: float,
    cast: float,
    gain_r: float,
    gain_g: float,
    gain_b: float,
    apply_bw: bool,
    bw_intensity: float,
    bw_neutrals: float,
    bw_tone: float,
    bw_grain: float,
) -> None:
    """Mutate ``image`` in-place using the available adjustment kernel (AOT, JIT, or NumPy)."""

    import numpy as np

    view, buffer_guard = _resolve_pixel_buffer(image)
    buffer_handle = buffer_guard
    _ = buffer_handle

    if getattr(view, "readonly", False):
        raise BufferError("QImage pixel buffer is read-only")

    if width <= 0 or height <= 0:
        return

    expected_size = bytes_per_line * height
    buffer = np.frombuffer(view, dtype=np.uint8, count=expected_size)
    if buffer.size < expected_size:
        raise BufferError("QImage pixel buffer is smaller than expected")

    apply_color = abs(saturation) > 1e-6 or abs(vibrance) > 1e-6 or cast > 1e-6

    _apply_adjustments_fast, _ = _resolve_executors()
    if _apply_adjustments_fast is None:
        logger.error("No image adjustment kernel available.")
        return

    _apply_adjustments_fast(
        buffer,
        width,
        height,
        bytes_per_line,
        exposure_term,
        brightness_term,
        brilliance_strength,
        highlights,
        shadows,
        contrast_factor,
        black_point,
        saturation,
        vibrance,
        cast,
        gain_r,
        gain_g,
        gain_b,
        apply_color,
        apply_bw,
        bw_intensity,
        bw_neutrals,
        bw_tone,
        bw_grain,
    )


def apply_color_adjustments_inplace_qimage(
    image: QImage,
    saturation: float,
    vibrance: float,
    cast: float,
    gain_r: float,
    gain_g: float,
    gain_b: float,
) -> None:
    """Apply only color adjustments to a QImage in-place."""
    if image.isNull():
        return

    apply_color = abs(saturation) > 1e-6 or abs(vibrance) > 1e-6 or cast > 1e-6
    if not apply_color:
        return

    import numpy as np

    view, guard = _resolve_pixel_buffer(image)
    buffer_handle = guard
    _ = buffer_handle

    if getattr(view, "readonly", False):
        raise BufferError("QImage pixel buffer is read-only")

    width = image.width()
    height = image.height()
    bytes_per_line = image.bytesPerLine()

    if width <= 0 or height <= 0:
        return

    expected_size = bytes_per_line * height
    buffer = np.frombuffer(view, dtype=np.uint8, count=expected_size)
    if buffer.size < expected_size:
        raise BufferError("QImage pixel buffer is smaller than expected")

    _, _apply_color_adjustments_inplace = _resolve_executors()
    if _apply_color_adjustments_inplace is None:
        logger.error(
            "No color adjustment kernel available. "
            "This may be due to missing dependencies (e.g., Numba), "
            "the AOT module not being compiled, or an unsupported environment. "
            "Please ensure all dependencies are installed and the AOT module is built if required."
        )
        return

    _apply_color_adjustments_inplace(
        buffer,
        width,
        height,
        bytes_per_line,
        saturation,
        vibrance,
        cast,
        gain_r,
        gain_g,
        gain_b,
    )
