"""Helpers for loading Qt image primitives with Pillow fallbacks."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional
import logging

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QImageReader, QPixmap

from .deps import load_pillow, load_rawpy
from ..media_classifier import RAW_EXTENSIONS

_PILLOW = load_pillow()
if _PILLOW is not None:  # pragma: no branch - import guard
    _Image = _PILLOW.Image
    _ImageOps = _PILLOW.ImageOps
    _ImageQt = _PILLOW.ImageQt
else:  # pragma: no cover - executed when Pillow is unavailable
    _Image = None  # type: ignore[assignment]
    _ImageOps = None  # type: ignore[assignment]
    _ImageQt = None  # type: ignore[assignment]

# Type checking import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from PIL import Image

_LOGGER = logging.getLogger(__name__)


def _is_raw_file(source: Path) -> bool:
    """Return ``True`` when *source* has a known camera-RAW extension."""
    return source.suffix.lower() in RAW_EXTENSIONS


def _load_raw_with_rawpy(
    source: Path, target: QSize | None = None
) -> Optional[QImage]:
    """Decode a camera-RAW file via :mod:`rawpy` and return a :class:`QImage`.

    ``rawpy`` wraps LibRaw which provides full-resolution demosaicked output for
    every major RAW format.  When a *target* size is supplied the decoded frame
    is optionally downscaled to save memory, but the aspect ratio is always
    preserved so the playback viewer never displays black borders or stretched
    pixels.
    """
    rp_support = load_rawpy()
    if rp_support is None:
        return None

    rawpy = rp_support.rawpy
    try:
        with rawpy.imread(str(source)) as raw:
            # ``postprocess`` demosaics + white-balances the sensor data and
            # returns an ``(H, W, 3)`` uint8 RGB array at full sensor resolution.
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False)
    except Exception:
        _LOGGER.debug("rawpy failed to decode %s", source, exc_info=True)
        return None

    h, w, channels = rgb.shape
    if channels < 3 or h == 0 or w == 0:
        return None

    # Build a QImage from the raw RGB buffer.  The data must remain alive
    # while the QImage references it, so we call ``.copy()`` immediately.
    bytes_per_line = w * 3
    image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

    if target is not None and target.isValid() and not target.isEmpty():
        src_size = image.size()
        scaled_target = src_size.scaled(target, Qt.AspectRatioMode.KeepAspectRatio)
        if (
            scaled_target.width() < src_size.width()
            or scaled_target.height() < src_size.height()
        ):
            image = image.scaled(
                scaled_target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

    return image


def load_qimage(source: Path, target: QSize | None = None) -> Optional[QImage]:
    """Return a :class:`QImage` for *source* with optional scaling."""

    if not source.exists():
        _LOGGER.debug("Skipping image load for missing path: %s", source)
        return None

    # RAW camera files need a dedicated decoder.  ``rawpy`` (backed by
    # LibRaw) provides full-resolution demosaicked frames that neither Qt's
    # built-in readers nor Pillow can deliver.
    if _is_raw_file(source):
        raw_image = _load_raw_with_rawpy(source, target)
        if raw_image is not None:
            return raw_image
        # Fall through to the standard pipeline so embedded previews are
        # still available when rawpy is missing.

    # ``QImageReader`` is most efficient when it can stream directly from the
    # filename because many formats (JPEG, HEIC, etc.) expose fast-paths for
    # downscaling during decode.  Reading the bytes eagerly would defeat those
    # optimisations, so we prefer to hand the path to Qt and only fall back to
    # Pillow if decoding fails entirely.
    reader = QImageReader(str(source))
    # Qt maintains a process-wide image cache that is enabled by default.
    # Large libraries can end up decoding hundreds of images during a single
    # browsing session which would otherwise accumulate in that cache.  The
    # additional allocations not only increase peak memory usage but can also
    # hold operating system file handles open.  Older PySide6 builds do not
    # expose ``setCacheEnabled`` though, so we guard the call to keep the code
    # compatible with those runtimes while still disabling the cache whenever
    # the API is available.
    disable_cache = getattr(reader, "setCacheEnabled", None)
    if callable(disable_cache):
        disable_cache(False)
    reader.setAutoTransform(True)
    if target is not None and target.isValid() and not target.isEmpty():
        original_size = reader.size()
        if original_size.isValid() and not original_size.isEmpty():
            # ``QImageReader.setScaledSize`` always interprets the requested
            # dimensions literally, even when that would distort the image.  We
            # pre-compute a size that preserves the source aspect ratio so the
            # decoder performs a proportional downscale rather than stretching to
            # fill the viewport bounds supplied by the caller.
            scaled_target = original_size.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            # Only request scaling when the destination is genuinely smaller; this
            # avoids unnecessary interpolation for thumbnails that are already
            # below the desired output resolution.
            if (
                scaled_target.width() < original_size.width()
                or scaled_target.height() < original_size.height()
            ):
                reader.setScaledSize(scaled_target)
        else:
            # Some formats only disclose their intrinsic size during ``read``. In
            # those cases we skip ``setScaledSize`` entirely to avoid guessing an
            # aspect ratio that might be wildly incorrect.  The caller will still
            # downscale the resulting pixmap once Qt reports the true dimensions.
            pass
    image = reader.read()
    if not image.isNull():
        return image
    return _load_with_pillow(source, target)


def load_qpixmap(source: Path, target: QSize | None = None) -> Optional[QPixmap]:
    """Return a :class:`QPixmap` for *source*, falling back to Pillow when required."""

    image = load_qimage(source, target)
    if image is None or image.isNull():
        return None
    pixmap = QPixmap.fromImage(image)
    if pixmap.isNull():
        return None
    return pixmap


def qimage_from_bytes(data: bytes) -> Optional[QImage]:
    """Return a :class:`QImage` decoded from JPEG/PNG *data*."""

    image = QImage()
    if image.loadFromData(data):
        return image
    if image.loadFromData(data, "JPEG"):
        return image
    if image.loadFromData(data, "JPG"):
        return image
    if image.loadFromData(data, "PNG"):
        return image
    if _Image is None or _ImageOps is None or _ImageQt is None:
        return None
    try:
        with _Image.open(BytesIO(data)) as img:  # type: ignore[union-attr]
            img = _ImageOps.exif_transpose(img)
            qt_image = _ImageQt(img.convert("RGBA"))
    except Exception:
        _LOGGER.exception("Pillow failed to decode image bytes in qimage_from_bytes")
        return None
    return QImage(qt_image)


def qimage_from_pil(image: "Image.Image") -> Optional[QImage]:
    """Return a :class:`QImage` from a PIL Image."""
    if _ImageQt is None:
        return None
    try:
        qt_image = _ImageQt(image.convert("RGBA"))
        return QImage(qt_image)
    except Exception:
        _LOGGER.exception("Failed to convert PIL image to QImage")
        return None


def _load_with_pillow(source: Path, target: QSize | None = None) -> Optional[QImage]:
    if _Image is None or _ImageOps is None or _ImageQt is None:
        return None
    try:
        with _Image.open(source) as img:  # type: ignore[attr-defined]
            img = _ImageOps.exif_transpose(img)  # type: ignore[attr-defined]
            if target is not None and target.isValid() and not target.isEmpty():
                resample = getattr(_Image, "Resampling", _Image)
                resample_filter = getattr(resample, "LANCZOS", _Image.BICUBIC)
                img.thumbnail((target.width(), target.height()), resample_filter)
            qt_image = _ImageQt(img.convert("RGBA"))  # type: ignore[attr-defined]
    except Exception:
        _LOGGER.exception("Pillow failed to load image from %s", source)
        return None
    return QImage(qt_image)


def generate_micro_thumbnail(source: Path) -> Optional[bytes]:
    """Generate a 16x16 (max dimension) JPEG thumbnail bytes for the given image.

    This function loads the image using Pillow, scales it down maintaining aspect ratio
    such that the longest side is 16 pixels, and encodes it as a JPEG.
    For RAW camera files, ``rawpy`` is used as the primary decoder.
    """
    if _Image is None or _ImageOps is None:
        return None

    if not source.exists():
        return None

    # RAW files: decode via rawpy first, then resize with Pillow
    if _is_raw_file(source):
        return _generate_raw_micro_thumbnail(source)

    try:
        with _Image.open(source) as img:  # type: ignore[attr-defined]
            # Optimization: Use draft mode for JPEG images to speed up loading
            # We request 64x64 to have enough headroom for high-quality downscaling
            # to the target 16x16 size while avoiding loading the full resolution image.
            if img.format == "JPEG":
                img.draft("RGB", (64, 64))

            # Scale to 16px max dimension
            target_size = (16, 16)
            resample = getattr(_Image, "Resampling", _Image)
            # Use BICUBIC instead of LANCZOS for speed; quality difference is negligible at 16x16
            resample_filter = getattr(resample, "BICUBIC", _Image.BICUBIC)
            # Optimization: Call thumbnail() BEFORE exif_transpose().
            # thumbnail() reduces the image dimensions in-place (often triggering the load).
            # If we transpose first (which creates a copy), we might be allocating a full-res
            # rotated copy of a large image (e.g. 20MP PNG), which is slow and memory-heavy.
            # By thumbnailing first, we only transpose a tiny 16x16 image.
            # This ordering is safe for any rectangular target box: thumbnail() fits the image
            # into a bounding box while preserving aspect ratio, and exif_transpose() only
            # rotates/flips the image (swapping width/height), so the final dimensions match
            # regardless of the order.
            img.thumbnail(target_size, resample_filter)

            # Handle orientation
            img = _ImageOps.exif_transpose(img)  # type: ignore[attr-defined]

            # Convert to RGB to ensure JPEG compatibility (drop alpha if present)
            # We convert AFTER resizing to avoid expensive RGB conversion on full-res images (e.g. RGBA PNGs)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save to bytes
            output = BytesIO()
            img.save(output, format="JPEG", quality=75)
            return output.getvalue()
    except Exception:
        _LOGGER.debug("Failed to generate micro thumbnail for %s", source, exc_info=True)
        return None


def _generate_raw_micro_thumbnail(source: Path) -> Optional[bytes]:
    """Generate a 16×16 JPEG thumbnail from a RAW file via ``rawpy``."""

    rp_support = load_rawpy()
    if rp_support is None or _Image is None:
        return None

    rawpy = rp_support.rawpy
    try:
        with rawpy.imread(str(source)) as raw:
            # Use half_size=True to decode at half sensor resolution — much
            # faster than full demosaic and more than adequate for a 16px
            # thumbnail.
            rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=False)
    except Exception:
        _LOGGER.debug("rawpy micro-thumbnail failed for %s", source, exc_info=True)
        return None

    try:
        img = _Image.fromarray(rgb)
        img.thumbnail((16, 16), _Image.Resampling.BICUBIC if hasattr(_Image, "Resampling") else _Image.BICUBIC)
        if img.mode != "RGB":
            img = img.convert("RGB")
        output = BytesIO()
        img.save(output, format="JPEG", quality=75)
        return output.getvalue()
    except Exception:
        _LOGGER.debug("Failed to encode RAW micro thumbnail for %s", source, exc_info=True)
        return None
