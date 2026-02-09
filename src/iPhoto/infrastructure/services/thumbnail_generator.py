from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageOps
import logging
import io

from src.iPhoto.application.interfaces import IThumbnailGenerator
from src.iPhoto.utils.image_loader import generate_micro_thumbnail
from src.iPhoto.utils.ffmpeg import extract_video_frame
from src.iPhoto.media_classifier import RAW_EXTENSIONS

LOGGER = logging.getLogger(__name__)

class PillowThumbnailGenerator(IThumbnailGenerator):
    """
    Generates thumbnails using Pillow for images and FFmpeg for videos.
    """

    def generate_micro_thumbnail(self, path: Path) -> Optional[str]:
        # Reuse existing utility
        if not path.exists():
            return None
        return generate_micro_thumbnail(path)

    def generate(self, path: Path, size: Tuple[int, int]) -> Optional[Image.Image]:
        """
        Generate a thumbnail for the given path at the specified size (width, height).
        Returns a PIL Image object or None on failure.
        """
        try:
            if not path.exists():
                return None
            # Determine if video based on extension
            video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
            if path.suffix.lower() in video_exts:
                return self._generate_video_thumbnail(path, size)

            # Default to Image
            return self._generate_image_thumbnail(path, size)

        except Exception as e:
            LOGGER.warning(f"Failed to generate thumbnail for {path}: {e}")
            return None

    def _generate_image_thumbnail(self, path: Path, size: Tuple[int, int]) -> Optional[Image.Image]:
        # For RAW files, try rawpy first since Pillow cannot decode them
        if path.suffix.lower() in RAW_EXTENSIONS:
            raw_result = self._generate_raw_thumbnail(path, size)
            if raw_result is not None:
                return raw_result

        try:
            with Image.open(path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Apply EXIF orientation
                img = ImageOps.exif_transpose(img)

                # Create thumbnail using LANCZOS for quality
                img.thumbnail(size, Image.Resampling.LANCZOS)
                return img.copy()
        except Exception as e:
            LOGGER.warning(f"Pillow failed to open {path}: {e}")
            return None

    def _generate_raw_thumbnail(self, path: Path, size: Tuple[int, int]) -> Optional[Image.Image]:
        """Decode a RAW file via rawpy and return a PIL thumbnail."""
        from src.iPhoto.utils.deps import load_rawpy
        support = load_rawpy()
        if support is None:
            return None
        rawpy = support.rawpy
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail(size, Image.Resampling.LANCZOS)
            return pil_img
        except Exception as e:
            LOGGER.warning(f"rawpy failed to open {path}: {e}")
            return None

    def _generate_video_thumbnail(self, path: Path, size: Tuple[int, int]) -> Optional[Image.Image]:
        try:
            if not path.exists():
                return None
            data = extract_video_frame(path, at=0.0, scale=size, format="jpeg")
            if data:
                with io.BytesIO(data) as bio:
                    img = Image.open(bio)
                    img.load()
                    return img.copy()
        except Exception as e:
            LOGGER.warning(f"FFmpeg failed to extract frame from {path}: {e}")
            return None
        return None
