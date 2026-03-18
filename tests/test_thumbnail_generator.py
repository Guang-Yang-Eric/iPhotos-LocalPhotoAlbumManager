"""Tests for PillowThumbnailGenerator — focus on video thumbnail generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
from PIL import Image

from iPhoto.infrastructure.services import thumbnail_generator as tg_module
from iPhoto.infrastructure.services.thumbnail_generator import PillowThumbnailGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(width: int = 100, height: int = 100) -> Image.Image:
    return Image.new("RGB", (width, height))


# ---------------------------------------------------------------------------
# Video thumbnail: PyAV-first strategy
# ---------------------------------------------------------------------------


class TestGenerateVideoThumbnail:
    """_generate_video_thumbnail should prefer PyAV and fall back to ffmpeg."""

    def test_pyav_used_first_when_available(self, tmp_path: Path) -> None:
        """When PyAV succeeds, extract_video_frame (subprocess) must NOT be called."""
        video = tmp_path / "clip.mp4"
        video.touch()

        expected = _make_pil_image(64, 36)

        with (
            patch.object(tg_module, "extract_frame_with_pyav", return_value=expected) as mock_pyav,
            patch.object(tg_module, "extract_video_frame") as mock_ffmpeg,
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (64, 36))

        assert result is expected
        mock_pyav.assert_called_once_with(video, at=0.0, scale=(64, 36))
        mock_ffmpeg.assert_not_called()

    def test_falls_back_to_ffmpeg_when_pyav_returns_none(self, tmp_path: Path) -> None:
        """When PyAV returns None, ffmpeg subprocess fallback should be used."""
        video = tmp_path / "clip.mp4"
        video.touch()

        # Minimal valid JPEG bytes (1×1 white pixel)
        import io
        buf = io.BytesIO()
        Image.new("RGB", (1, 1)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        with (
            patch.object(tg_module, "extract_frame_with_pyav", return_value=None),
            patch.object(tg_module, "extract_video_frame", return_value=jpeg_bytes) as mock_ffmpeg,
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (64, 36))

        assert result is not None
        assert isinstance(result, Image.Image)
        mock_ffmpeg.assert_called_once_with(video, at=0.0, scale=(64, 36), format="jpeg")

    def test_falls_back_to_ffmpeg_when_pyav_raises(self, tmp_path: Path) -> None:
        """When PyAV raises an exception, ffmpeg subprocess fallback is still attempted."""
        video = tmp_path / "clip.mp4"
        video.touch()

        import io
        buf = io.BytesIO()
        Image.new("RGB", (1, 1)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        with (
            patch.object(tg_module, "extract_frame_with_pyav", side_effect=RuntimeError("av error")),
            patch.object(tg_module, "extract_video_frame", return_value=jpeg_bytes) as mock_ffmpeg,
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (64, 36))

        # PyAV exception is isolated; ffmpeg fallback returns a valid image
        assert result is not None
        assert isinstance(result, Image.Image)
        mock_ffmpeg.assert_called_once_with(video, at=0.0, scale=(64, 36), format="jpeg")

    def test_returns_none_when_both_pyav_and_ffmpeg_fail(self, tmp_path: Path) -> None:
        """Returns None (gracefully) when both strategies fail."""
        video = tmp_path / "corrupt.mp4"
        video.touch()

        from iPhoto.errors import ExternalToolError

        with (
            patch.object(tg_module, "extract_frame_with_pyav", return_value=None),
            patch.object(tg_module, "extract_video_frame", side_effect=ExternalToolError("fail")),
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (64, 36))

        assert result is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Non-existent file returns None without calling any extractor."""
        missing = tmp_path / "missing.mp4"

        with (
            patch.object(tg_module, "extract_frame_with_pyav") as mock_pyav,
            patch.object(tg_module, "extract_video_frame") as mock_ffmpeg,
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(missing, (64, 36))

        assert result is None
        mock_pyav.assert_not_called()
        mock_ffmpeg.assert_not_called()

    def test_generate_routes_video_to_pyav_first(self, tmp_path: Path) -> None:
        """Top-level generate() for a video file uses PyAV-first strategy."""
        video = tmp_path / "video.mov"
        video.touch()

        expected = _make_pil_image(128, 72)

        with patch.object(tg_module, "extract_frame_with_pyav", return_value=expected) as mock_pyav:
            gen = PillowThumbnailGenerator()
            result = gen.generate(video, (128, 72))

        assert result is expected
        mock_pyav.assert_called_once_with(video, at=0.0, scale=(128, 72))
