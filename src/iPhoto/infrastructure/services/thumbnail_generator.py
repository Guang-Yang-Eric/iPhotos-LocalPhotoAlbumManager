from pathlib import Path
from typing import Optional, Tuple
import logging
import io

from PIL import Image, ImageOps

from iPhoto.application.interfaces import IThumbnailGenerator
from iPhoto.core.raw_processor import is_raw_extension, load_raw_to_pil
from iPhoto.utils.ffmpeg import (
    apply_video_rotation,
    extract_frame_with_pyav,
    extract_oriented_video_frame,
    extract_video_frame,
    probe_video_rotation,
)
from iPhoto.utils.image_loader import generate_micro_thumbnail

LOGGER = logging.getLogger(__name__)


# Backward-compatible alias for existing tests and integrations.
_apply_video_rotation = apply_video_rotation


def _extract_video_frame_with_fallbacks(
    source: Path,
    *,
    at: float,
    scale: tuple[int, int],
    format: str,
    allow_pyav: bool,
) -> bytes:
    """Compatibility shim kept for legacy thumbnail tests.

    Historically this helper exposed the bytes-producing fallback path from the
    thumbnail generator module. The current implementation delegates to the
    shared ffmpeg helper while preserving the old injection point.
    """

    return extract_video_frame(
        source,
        at=at,
        scale=scale,
        format=format,
    )


class PillowThumbnailGenerator(IThumbnailGenerator):
    """Generate thumbnails using Pillow for images and ffmpeg for videos."""

    def generate_micro_thumbnail(self, path: Path) -> Optional[str]:
        if not path.exists():
            return None
        return generate_micro_thumbnail(path)

    def generate(self, path: Path, size: Tuple[int, int]) -> Optional[Image.Image]:
        """Generate a thumbnail for *path* constrained to *size*."""

        try:
            if not path.exists():
                return None

            video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
            if path.suffix.lower() in video_exts:
                return self._generate_video_thumbnail(path, size)

            if is_raw_extension(path.suffix):
                return self._generate_raw_thumbnail(path, size)

            return self._generate_image_thumbnail(path, size)
        except Exception as exc:
            LOGGER.warning(f"Failed to generate thumbnail for {path}: {exc}")
            return None

    def _generate_image_thumbnail(
        self,
        path: Path,
        size: Tuple[int, int],
    ) -> Optional[Image.Image]:
        try:
            with Image.open(path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = ImageOps.exif_transpose(img)
                img.thumbnail(size, Image.Resampling.LANCZOS)
                return img.copy()
        except Exception as exc:
            LOGGER.warning(f"Pillow failed to open {path}: {exc}")
            return None

    def _generate_raw_thumbnail(
        self,
        path: Path,
        size: Tuple[int, int],
    ) -> Optional[Image.Image]:
        """Generate a thumbnail from a RAW camera file using rawpy."""

        try:
            pil_img = load_raw_to_pil(path, half_size=True, target_size=size)
            if pil_img is None:
                return None
            pil_img.thumbnail(size, Image.Resampling.LANCZOS)
            return pil_img
        except Exception as exc:
            LOGGER.warning(f"rawpy failed to generate thumbnail for {path}: {exc}")
            return None

    def _generate_video_thumbnail(
        self,
        path: Path,
        size: Tuple[int, int],
    ) -> Optional[Image.Image]:
        try:
            if not path.exists():
                return None
            rotation_cw, _, _ = probe_video_rotation(path)
            if rotation_cw == 0:
                image = extract_frame_with_pyav(path, at=0.0, scale=size)
                if image is not None:
                    return image.copy()
                data = _extract_video_frame_with_fallbacks(
                    path,
                    at=0.0,
                    scale=size,
                    format="jpeg",
                    allow_pyav=False,
                )
                if not data:
                    return None
                with io.BytesIO(data) as bio:
                    with Image.open(bio) as image:
                        image.load()
                        return image.copy()
            return extract_oriented_video_frame(path, at=0.0, scale=size)
        except Exception as exc:
            LOGGER.warning(f"Failed to extract frame from {path}: {exc}")
            return None
