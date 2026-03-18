"""Utilities for extracting representative video frames for thumbnails."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING

from PySide6.QtCore import QSize
from PySide6.QtGui import QImage

from ....config import THUMBNAIL_SEEK_GUARD_SEC
from ....errors import ExternalToolError
from ....utils.ffmpeg import extract_video_frame, extract_frame_with_pyav, probe_video_rotation
from ....utils import image_loader

if TYPE_CHECKING:
    from PIL import Image as _PILImage


def _apply_pil_rotation(img: "_PILImage.Image", rotation_cw: int) -> "_PILImage.Image":
    """Return *img* rotated clockwise by *rotation_cw* degrees (multiples of 90 only).

    Uses :meth:`~PIL.Image.Image.transpose` for an exact, lossless pixel
    rearrangement.  Non-multiples of 90 and zero return the image unchanged.
    """
    from PIL import Image  # local import to avoid mandatory top-level dependency

    rotation_cw = rotation_cw % 360
    if rotation_cw == 90:
        return img.transpose(Image.Transpose.ROTATE_270)
    if rotation_cw == 180:
        return img.transpose(Image.Transpose.ROTATE_180)
    if rotation_cw == 270:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img


def grab_video_frame(
    path: Path,
    size: QSize,
    *,
    still_image_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> Optional[QImage]:
    """Return a decoded frame for *path* scaled to *size*.

    The Display Matrix rotation stored in the video's stream metadata is
    applied so the returned :class:`QImage` is correctly oriented on all
    platforms.  ``ffmpeg`` is invoked with ``-noautorotate`` to ensure
    that the raw coded frame is returned without any implicit rotation,
    and this function then applies the rotation explicitly using PIL.
    """

    target_size = (max(size.width(), 1), max(size.height(), 1))

    # Probe rotation once per file — applies to every frame in the container.
    rotation_cw, _, _ = probe_video_rotation(path)

    for target in _seek_targets(still_image_time, duration):
        # 1. Try PyAV first (direct memory access, faster)
        pil_image = extract_frame_with_pyav(path, at=target, scale=target_size)
        if pil_image is not None:
            if rotation_cw:
                pil_image = _apply_pil_rotation(pil_image, rotation_cw)
            return image_loader.qimage_from_pil(pil_image)

        # 2. Fallback to ffmpeg subprocess if PyAV fails
        try:
            frame_data = extract_video_frame(
                path,
                at=target,
                scale=target_size,
                format="jpeg",
            )
            if frame_data:
                if rotation_cw:
                    # Decode bytes to PIL Image so we can apply the rotation,
                    # then convert back to QImage.
                    from PIL import Image  # local import

                    with io.BytesIO(frame_data) as bio:
                        pil_frame = Image.open(bio)
                        pil_frame.load()
                    pil_frame = _apply_pil_rotation(pil_frame, rotation_cw)
                    return image_loader.qimage_from_pil(pil_frame)
                return image_loader.qimage_from_bytes(frame_data)
        except ExternalToolError:
            continue

    return None


def _seek_targets(
    still_image_time: Optional[float], duration: Optional[float]
) -> Iterable[Optional[float]]:
    targets: List[Optional[float]] = []
    seen: set[Optional[float]] = set()

    def add(candidate: Optional[float]) -> None:
        if candidate is None:
            key: Optional[float] = None
            value: Optional[float] = None
        else:
            value = _normalize_seek(candidate, duration)
            key = value
        if key in seen:
            return
        seen.add(key)
        targets.append(value)

    if still_image_time is not None:
        add(still_image_time)
    elif duration is not None and duration > 0:
        add(duration / 2.0)
    add(None)
    return targets


def _normalize_seek(value: float, duration: Optional[float]) -> float:
    normalized = max(value, 0.0)
    if duration and duration > 0:
        guard = min(
            max(THUMBNAIL_SEEK_GUARD_SEC, duration * 0.1),
            duration / 2.0,
        )
        max_seek = max(duration - guard, 0.0)
        if normalized > max_seek:
            normalized = max_seek
    return normalized
