"""Tests for video_frame_grabber — focus on rotation correction via Display Matrix."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock
import pytest

pytest.importorskip("PySide6", reason="PySide6 required", exc_type=ImportError)

from PIL import Image

from iPhoto.gui.ui.tasks import video_frame_grabber as vfg_module
from iPhoto.gui.ui.tasks.video_frame_grabber import (
    _apply_pil_rotation,
    grab_video_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height))


def _make_mock_qimage() -> MagicMock:
    q = MagicMock()
    q.isNull.return_value = False
    return q


# ---------------------------------------------------------------------------
# _apply_pil_rotation — pure unit tests (no Qt required)
# ---------------------------------------------------------------------------


class TestApplyPilRotation:
    """Unit tests for the _apply_pil_rotation helper."""

    def test_no_rotation_returns_same_object(self) -> None:
        img = _make_pil(100, 60)
        assert _apply_pil_rotation(img, 0) is img

    def test_360_treated_as_no_rotation(self) -> None:
        img = _make_pil(100, 60)
        assert _apply_pil_rotation(img, 360) is img

    def test_non_multiple_of_90_unchanged(self) -> None:
        img = _make_pil(100, 60)
        assert _apply_pil_rotation(img, 45) is img

    def test_90_cw_swaps_dimensions(self) -> None:
        img = _make_pil(100, 60)
        result = _apply_pil_rotation(img, 90)
        assert result.size == (60, 100)

    def test_180_preserves_dimensions(self) -> None:
        img = _make_pil(100, 60)
        result = _apply_pil_rotation(img, 180)
        assert result.size == (100, 60)

    def test_270_cw_swaps_dimensions(self) -> None:
        img = _make_pil(100, 60)
        result = _apply_pil_rotation(img, 270)
        assert result.size == (60, 100)

    def test_180_flips_pixel_content(self) -> None:
        """180° rotation must actually invert the pixel layout."""
        img = Image.new("RGB", (4, 2))
        pixels = [(r, 0, 0) for r in range(8)]
        img.putdata(pixels)

        result = _apply_pil_rotation(img, 180)
        assert list(result.getdata()) == list(reversed(pixels))


# ---------------------------------------------------------------------------
# grab_video_frame — rotation correction integration
# ---------------------------------------------------------------------------


class TestGrabVideoFrameRotation:
    """grab_video_frame must apply Display Matrix rotation before returning."""

    def _run(
        self,
        tmp_path: Path,
        rotation_cw: int,
        raw_width: int,
        raw_height: int,
        *,
        pyav_image: Optional[Image.Image],
    ):
        """Exercise grab_video_frame with mocked I/O; return captured PIL image."""
        from PySide6.QtCore import QSize

        video = tmp_path / "clip.mov"
        video.touch()

        captured: list[Image.Image] = []

        def fake_qimage_from_pil(img: Image.Image):
            captured.append(img)
            return _make_mock_qimage()

        with (
            patch.object(
                vfg_module, "probe_video_rotation",
                return_value=(rotation_cw, raw_width, raw_height),
            ),
            patch.object(
                vfg_module, "extract_frame_with_pyav",
                return_value=pyav_image,
            ),
            patch.object(
                vfg_module.image_loader, "qimage_from_pil",
                side_effect=fake_qimage_from_pil,
            ),
        ):
            result = grab_video_frame(video, QSize(raw_width, raw_height))

        return result, captured

    def test_no_rotation_passes_frame_unchanged(self, tmp_path: Path) -> None:
        """When rotation is 0 the PIL image must not be transformed."""
        raw = _make_pil(1280, 720)
        _, captured = self._run(tmp_path, 0, 1280, 720, pyav_image=raw)
        assert len(captured) == 1
        assert captured[0] is raw

    def test_90_cw_swaps_dimensions(self, tmp_path: Path) -> None:
        """Raw landscape frame + 90° CW rotation → portrait thumbnail."""
        raw = _make_pil(1280, 720)
        _, captured = self._run(tmp_path, 90, 1280, 720, pyav_image=raw)
        assert len(captured) == 1
        assert captured[0].size == (720, 1280)

    def test_180_preserves_dimensions(self, tmp_path: Path) -> None:
        """180° rotation (the -180° display matrix case) keeps the same size."""
        raw = _make_pil(1280, 720)
        _, captured = self._run(tmp_path, 180, 1280, 720, pyav_image=raw)
        assert len(captured) == 1
        # Dimensions are unchanged for 180°
        assert captured[0].size == (1280, 720)

    def test_270_cw_swaps_dimensions(self, tmp_path: Path) -> None:
        """270° CW rotation → portrait thumbnail."""
        raw = _make_pil(1280, 720)
        _, captured = self._run(tmp_path, 270, 1280, 720, pyav_image=raw)
        assert len(captured) == 1
        assert captured[0].size == (720, 1280)

    def test_minus_180_display_matrix_case(self, tmp_path: Path) -> None:
        """Regression: -180° display matrix (cw=180) must produce a flipped thumbnail.

        The specific video reported in the issue has
        ``displaymatrix: rotation of -180.00 degrees`` which maps to
        ``cw=180`` via probe_video_rotation.
        """
        raw = _make_pil(1280, 720)
        _, captured = self._run(tmp_path, 180, 1280, 720, pyav_image=raw)
        assert len(captured) == 1
        assert captured[0].size == (1280, 720)

    def test_ffmpeg_fallback_applies_rotation(self, tmp_path: Path) -> None:
        """When PyAV returns None the ffmpeg bytes path must also rotate."""
        from PySide6.QtCore import QSize
        from iPhoto.errors import ExternalToolError

        video = tmp_path / "clip.mov"
        video.touch()

        # Build a minimal JPEG for the ffmpeg output
        buf = io.BytesIO()
        Image.new("RGB", (1280, 720)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        captured: list[Image.Image] = []

        def fake_qimage_from_pil(img: Image.Image):
            captured.append(img)
            return _make_mock_qimage()

        with (
            patch.object(
                vfg_module, "probe_video_rotation",
                return_value=(90, 1280, 720),
            ),
            patch.object(
                vfg_module, "extract_frame_with_pyav",
                return_value=None,
            ),
            patch.object(
                vfg_module, "extract_video_frame",
                return_value=jpeg_bytes,
            ),
            patch.object(
                vfg_module.image_loader, "qimage_from_pil",
                side_effect=fake_qimage_from_pil,
            ),
        ):
            result = grab_video_frame(video, QSize(1280, 720))

        assert result is not None
        assert len(captured) == 1
        # 90° CW rotation swaps width/height
        assert captured[0].size == (720, 1280)

    def test_no_rotation_ffmpeg_fallback_uses_bytes_directly(
        self, tmp_path: Path
    ) -> None:
        """When rotation is 0 and PyAV fails, raw bytes are decoded by qimage_from_bytes."""
        from PySide6.QtCore import QSize

        video = tmp_path / "clip.mov"
        video.touch()

        buf = io.BytesIO()
        Image.new("RGB", (640, 480)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        captured_bytes: list[bytes] = []

        def fake_qimage_from_bytes(data: bytes):
            captured_bytes.append(data)
            return _make_mock_qimage()

        with (
            patch.object(
                vfg_module, "probe_video_rotation",
                return_value=(0, 640, 480),
            ),
            patch.object(vfg_module, "extract_frame_with_pyav", return_value=None),
            patch.object(
                vfg_module, "extract_video_frame", return_value=jpeg_bytes
            ),
            patch.object(
                vfg_module.image_loader, "qimage_from_bytes",
                side_effect=fake_qimage_from_bytes,
            ),
        ):
            result = grab_video_frame(video, QSize(640, 480))

        assert result is not None
        assert captured_bytes == [jpeg_bytes]
