"""Tests for PillowThumbnailGenerator — focus on video thumbnail generation."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image

from iPhoto.infrastructure.services import thumbnail_generator as tg_module
from iPhoto.infrastructure.services.thumbnail_generator import (
    PillowThumbnailGenerator,
    _apply_video_rotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(width: int = 100, height: int = 100) -> Image.Image:
    return Image.new("RGB", (width, height))


# ---------------------------------------------------------------------------
# _apply_video_rotation unit tests
# ---------------------------------------------------------------------------


class TestApplyVideoRotation:
    """Unit tests for the _apply_video_rotation helper."""

    def test_no_rotation_returns_same_object(self) -> None:
        img = _make_pil_image(100, 60)
        result = _apply_video_rotation(img, 0)
        assert result is img

    def test_360_treated_as_no_rotation(self) -> None:
        img = _make_pil_image(100, 60)
        result = _apply_video_rotation(img, 360)
        assert result is img

    def test_90_cw_swaps_dimensions(self) -> None:
        img = _make_pil_image(100, 60)
        result = _apply_video_rotation(img, 90)
        assert result.size == (60, 100)

    def test_180_preserves_dimensions(self) -> None:
        img = _make_pil_image(100, 60)
        result = _apply_video_rotation(img, 180)
        assert result.size == (100, 60)

    def test_270_cw_swaps_dimensions(self) -> None:
        img = _make_pil_image(100, 60)
        result = _apply_video_rotation(img, 270)
        assert result.size == (60, 100)

    def test_non_multiple_of_90_returns_unchanged(self) -> None:
        img = _make_pil_image(100, 60)
        result = _apply_video_rotation(img, 45)
        assert result is img


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
            patch.object(tg_module, "probe_video_rotation", return_value=(0, 0, 0)),
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
        buf = io.BytesIO()
        Image.new("RGB", (1, 1)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(0, 0, 0)),
            patch.object(tg_module, "extract_frame_with_pyav", return_value=None),
            patch.object(tg_module, "extract_video_frame", return_value=jpeg_bytes) as mock_ffmpeg,
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (64, 36))

        assert result is not None
        assert isinstance(result, Image.Image)
        mock_ffmpeg.assert_called_once_with(video, at=0.0, scale=(64, 36), format="jpeg")

    def test_returns_none_when_pyav_raises_unexpectedly(self, tmp_path: Path) -> None:
        """When PyAV unexpectedly raises, the outer exception handler catches it and returns None.

        Note: In practice extract_frame_with_pyav catches all its own exceptions
        and returns None, so the outer handler is only a safety net. This test
        verifies that an unexpected raise does not propagate to the caller.
        """
        video = tmp_path / "clip.mp4"
        video.touch()

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(0, 0, 0)),
            patch.object(tg_module, "extract_frame_with_pyav", side_effect=RuntimeError("av error")),
            patch.object(tg_module, "extract_video_frame") as mock_ffmpeg,
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (64, 36))

        # The outer except Exception catches the RuntimeError; ffmpeg is not reached.
        assert result is None
        mock_ffmpeg.assert_not_called()

    def test_returns_none_when_both_pyav_and_ffmpeg_fail(self, tmp_path: Path) -> None:
        """Returns None (gracefully) when both strategies fail."""
        video = tmp_path / "corrupt.mp4"
        video.touch()

        from iPhoto.errors import ExternalToolError

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(0, 0, 0)),
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

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(0, 0, 0)),
            patch.object(tg_module, "extract_frame_with_pyav", return_value=expected) as mock_pyav,
        ):
            gen = PillowThumbnailGenerator()
            result = gen.generate(video, (128, 72))

        assert result is expected
        mock_pyav.assert_called_once_with(video, at=0.0, scale=(128, 72))


# ---------------------------------------------------------------------------
# Video thumbnail: rotation correction
# ---------------------------------------------------------------------------


class TestVideoThumbnailRotation:
    """_generate_video_thumbnail must apply Display Matrix rotation to thumbnails."""

    def test_90_cw_rotation_swaps_dimensions_pyav_path(self, tmp_path: Path) -> None:
        """90° CW rotation turns a 100×60 frame into a 60×100 thumbnail."""
        video = tmp_path / "portrait.mp4"
        video.touch()

        raw_frame = _make_pil_image(100, 60)

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(90, 100, 60)),
            patch.object(tg_module, "extract_frame_with_pyav", return_value=raw_frame),
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (100, 60))

        assert result is not None
        assert result.size == (60, 100)

    def test_180_rotation_preserves_dimensions_pyav_path(self, tmp_path: Path) -> None:
        """180° rotation keeps the same dimensions (just flipped content)."""
        video = tmp_path / "upside_down.mp4"
        video.touch()

        raw_frame = _make_pil_image(100, 60)

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(180, 100, 60)),
            patch.object(tg_module, "extract_frame_with_pyav", return_value=raw_frame),
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (100, 60))

        assert result is not None
        assert result.size == (100, 60)

    def test_270_cw_rotation_swaps_dimensions_pyav_path(self, tmp_path: Path) -> None:
        """270° CW rotation turns a 100×60 frame into a 60×100 thumbnail."""
        video = tmp_path / "portrait_ccw.mp4"
        video.touch()

        raw_frame = _make_pil_image(100, 60)

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(270, 100, 60)),
            patch.object(tg_module, "extract_frame_with_pyav", return_value=raw_frame),
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (100, 60))

        assert result is not None
        assert result.size == (60, 100)

    def test_no_rotation_returns_frame_unmodified(self, tmp_path: Path) -> None:
        """When rotation is 0 the original image object is returned unchanged."""
        video = tmp_path / "landscape.mp4"
        video.touch()

        raw_frame = _make_pil_image(100, 60)

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(0, 100, 60)),
            patch.object(tg_module, "extract_frame_with_pyav", return_value=raw_frame),
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (100, 60))

        assert result is raw_frame  # same object — no copy made for 0°

    def test_90_cw_rotation_applied_on_ffmpeg_fallback_path(self, tmp_path: Path) -> None:
        """Rotation is also applied when the result comes from the ffmpeg fallback."""
        video = tmp_path / "portrait_ffmpeg.mp4"
        video.touch()

        # The ffmpeg subprocess returns the raw coded frame: 60 wide × 100 tall
        # (a portrait video stored without rotation applied).  probe_video_rotation
        # reports (90, 60, 100): "rotate 90° CW; raw coded dimensions are 60×100".
        buf = io.BytesIO()
        Image.new("RGB", (60, 100)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(90, 60, 100)),
            patch.object(tg_module, "extract_frame_with_pyav", return_value=None),
            patch.object(tg_module, "extract_video_frame", return_value=jpeg_bytes),
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (100, 60))

        assert result is not None
        # 60×100 raw frame rotated 90° CW → 100×60 (landscape)
        assert result.size == (100, 60)

    def test_probe_rotation_failure_still_returns_image(self, tmp_path: Path) -> None:
        """If probe_video_rotation fails gracefully (returns 0,0,0), image is still returned."""
        video = tmp_path / "noprobe.mp4"
        video.touch()

        raw_frame = _make_pil_image(100, 60)

        with (
            patch.object(tg_module, "probe_video_rotation", return_value=(0, 0, 0)),
            patch.object(tg_module, "extract_frame_with_pyav", return_value=raw_frame),
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (100, 60))

        assert result is not None
        assert result.size == (100, 60)

