"""Tests for ``video_frame_grabber`` shared-orientation integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

pytest.importorskip("PySide6", reason="PySide6 required", exc_type=ImportError)

from iPhoto.gui.ui.tasks import video_frame_grabber as vfg_module
from iPhoto.gui.ui.tasks.video_frame_grabber import (
    _apply_pil_rotation,
    grab_video_frame,
)


def _make_pil(width: int = 100, height: int = 60) -> Image.Image:
    return Image.new("RGB", (width, height))


def _make_mock_qimage() -> MagicMock:
    image = MagicMock()
    image.isNull.return_value = False
    return image


class TestApplyPilRotation:
    def test_no_rotation_returns_same_object(self) -> None:
        img = _make_pil()
        assert _apply_pil_rotation(img, 0) is img

    def test_360_treated_as_no_rotation(self) -> None:
        img = _make_pil()
        assert _apply_pil_rotation(img, 360) is img

    def test_non_multiple_of_90_unchanged(self) -> None:
        img = _make_pil()
        assert _apply_pil_rotation(img, 45) is img

    def test_90_cw_swaps_dimensions(self) -> None:
        img = _make_pil()
        assert _apply_pil_rotation(img, 90).size == (60, 100)

    def test_180_preserves_dimensions(self) -> None:
        img = _make_pil()
        assert _apply_pil_rotation(img, 180).size == (100, 60)

    def test_270_cw_swaps_dimensions(self) -> None:
        img = _make_pil()
        assert _apply_pil_rotation(img, 270).size == (60, 100)

    def test_180_flips_pixel_content(self) -> None:
        img = Image.new("RGB", (4, 2))
        pixels = [(r, 0, 0) for r in range(8)]
        img.putdata(pixels)
        assert list(_apply_pil_rotation(img, 180).getdata()) == list(reversed(pixels))


class TestGrabVideoFrame:
    def test_uses_shared_oriented_extractor(self, tmp_path: Path) -> None:
        from PySide6.QtCore import QSize

        video = tmp_path / "clip.mov"
        video.touch()
        pil_frame = _make_pil(1280, 720)
        qimage = _make_mock_qimage()

        with (
            patch.object(
                vfg_module,
                "extract_oriented_video_frame",
                return_value=pil_frame,
            ) as mock_extract,
            patch.object(
                vfg_module.image_loader,
                "qimage_from_pil",
                return_value=qimage,
            ) as mock_convert,
        ):
            result = grab_video_frame(video, QSize(1280, 720))

        assert result is qimage
        mock_extract.assert_called_once_with(video, at=None, scale=(1280, 720))
        mock_convert.assert_called_once_with(pil_frame)

    def test_retries_seek_targets_until_frame_available(self, tmp_path: Path) -> None:
        from PySide6.QtCore import QSize

        video = tmp_path / "clip.mov"
        video.touch()
        pil_frame = _make_pil(640, 480)
        qimage = _make_mock_qimage()

        with (
            patch.object(
                vfg_module,
                "extract_oriented_video_frame",
                side_effect=[None, pil_frame],
            ) as mock_extract,
            patch.object(
                vfg_module.image_loader,
                "qimage_from_pil",
                return_value=qimage,
            ),
        ):
            result = grab_video_frame(
                video,
                QSize(640, 480),
                still_image_time=1.25,
                duration=4.0,
            )

        assert result is qimage
        assert mock_extract.call_count == 2
        assert mock_extract.call_args_list[0].kwargs == {
            "at": 1.25,
            "scale": (640, 480),
        }
        assert mock_extract.call_args_list[1].kwargs == {
            "at": None,
            "scale": (640, 480),
        }

    def test_returns_none_when_all_targets_fail(self, tmp_path: Path) -> None:
        from PySide6.QtCore import QSize

        video = tmp_path / "clip.mov"
        video.touch()

        with patch.object(
            vfg_module,
            "extract_oriented_video_frame",
            return_value=None,
        ) as mock_extract:
            result = grab_video_frame(
                video,
                QSize(640, 480),
                still_image_time=1.0,
                duration=3.0,
            )

        assert result is None
        assert mock_extract.call_count == 2
