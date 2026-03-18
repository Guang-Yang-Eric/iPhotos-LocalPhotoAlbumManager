"""Tests for ``PillowThumbnailGenerator`` video thumbnail integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from iPhoto.infrastructure.services import thumbnail_generator as tg_module
from iPhoto.infrastructure.services.thumbnail_generator import (
    PillowThumbnailGenerator,
    _apply_video_rotation,
)


def _make_pil_image(width: int = 100, height: int = 60) -> Image.Image:
    return Image.new("RGB", (width, height))


class TestApplyVideoRotation:
    def test_no_rotation_returns_same_object(self) -> None:
        img = _make_pil_image()
        assert _apply_video_rotation(img, 0) is img

    def test_360_treated_as_no_rotation(self) -> None:
        img = _make_pil_image()
        assert _apply_video_rotation(img, 360) is img

    def test_90_cw_swaps_dimensions(self) -> None:
        img = _make_pil_image()
        assert _apply_video_rotation(img, 90).size == (60, 100)

    def test_180_preserves_dimensions(self) -> None:
        img = _make_pil_image()
        assert _apply_video_rotation(img, 180).size == (100, 60)

    def test_270_cw_swaps_dimensions(self) -> None:
        img = _make_pil_image()
        assert _apply_video_rotation(img, 270).size == (60, 100)

    def test_non_multiple_of_90_returns_unchanged(self) -> None:
        img = _make_pil_image()
        assert _apply_video_rotation(img, 45) is img


class TestGenerateVideoThumbnail:
    def test_uses_shared_oriented_extractor(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        video.touch()
        expected = _make_pil_image(64, 36)

        with patch.object(
            tg_module,
            "extract_oriented_video_frame",
            return_value=expected,
        ) as mock_extract:
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (64, 36))

        assert result is expected
        mock_extract.assert_called_once_with(video, at=0.0, scale=(64, 36))

    def test_returns_none_when_shared_extractor_raises(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        video.touch()

        with patch.object(
            tg_module,
            "extract_oriented_video_frame",
            side_effect=RuntimeError("boom"),
        ):
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(video, (64, 36))

        assert result is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.mp4"

        with patch.object(
            tg_module,
            "extract_oriented_video_frame",
        ) as mock_extract:
            gen = PillowThumbnailGenerator()
            result = gen._generate_video_thumbnail(missing, (64, 36))

        assert result is None
        mock_extract.assert_not_called()

    def test_generate_routes_video_to_shared_extractor(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mov"
        video.touch()
        expected = _make_pil_image(128, 72)

        with patch.object(
            tg_module,
            "extract_oriented_video_frame",
            return_value=expected,
        ) as mock_extract:
            gen = PillowThumbnailGenerator()
            result = gen.generate(video, (128, 72))

        assert result is expected
        mock_extract.assert_called_once_with(video, at=0.0, scale=(128, 72))
