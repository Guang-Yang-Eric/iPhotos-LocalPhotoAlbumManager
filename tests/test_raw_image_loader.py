"""Tests for RAW image loading via rawpy in image_loader."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtGui import QImage

from src.iPhoto.utils.image_loader import (
    _is_raw_file,
    _load_raw_with_rawpy,
    load_qimage,
    generate_micro_thumbnail,
    _generate_raw_micro_thumbnail,
)
from src.iPhoto.media_classifier import RAW_EXTENSIONS


class TestIsRawFile:
    @pytest.mark.parametrize("ext", list(RAW_EXTENSIONS)[:5])
    def test_recognises_raw_extensions(self, ext: str) -> None:
        assert _is_raw_file(Path(f"/photos/img{ext}")) is True

    @pytest.mark.parametrize("ext", [".jpg", ".png", ".heic", ".mp4"])
    def test_rejects_non_raw(self, ext: str) -> None:
        assert _is_raw_file(Path(f"/photos/img{ext}")) is False

    def test_case_insensitive(self) -> None:
        assert _is_raw_file(Path("/photos/img.CR2")) is True
        assert _is_raw_file(Path("/photos/img.Nef")) is True


class TestLoadRawWithRawpy:
    def test_returns_none_when_rawpy_unavailable(self) -> None:
        with patch("src.iPhoto.utils.image_loader.load_rawpy", return_value=None):
            result = _load_raw_with_rawpy(Path("/fake/img.cr2"))
            assert result is None

    def test_returns_qimage_from_mock_rawpy(self, tmp_path: Path) -> None:
        fake_rgb = np.zeros((100, 150, 3), dtype=np.uint8)
        fake_rgb[..., 0] = 128  # red channel

        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = fake_rgb
        mock_raw.__enter__ = MagicMock(return_value=mock_raw)
        mock_raw.__exit__ = MagicMock(return_value=False)

        mock_rawpy_module = MagicMock()
        mock_rawpy_module.imread.return_value = mock_raw

        mock_support = MagicMock()
        mock_support.rawpy = mock_rawpy_module

        with patch("src.iPhoto.utils.image_loader.load_rawpy", return_value=mock_support):
            result = _load_raw_with_rawpy(Path("/fake/img.cr2"))
            assert result is not None
            assert isinstance(result, QImage)
            assert result.width() == 150
            assert result.height() == 100

    def test_respects_target_scaling(self, tmp_path: Path) -> None:
        fake_rgb = np.zeros((2000, 3000, 3), dtype=np.uint8)

        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = fake_rgb
        mock_raw.__enter__ = MagicMock(return_value=mock_raw)
        mock_raw.__exit__ = MagicMock(return_value=False)

        mock_rawpy_module = MagicMock()
        mock_rawpy_module.imread.return_value = mock_raw

        mock_support = MagicMock()
        mock_support.rawpy = mock_rawpy_module

        with patch("src.iPhoto.utils.image_loader.load_rawpy", return_value=mock_support):
            result = _load_raw_with_rawpy(Path("/fake/img.nef"), QSize(300, 200))
            assert result is not None
            # Should be downscaled preserving aspect ratio
            assert result.width() <= 300
            assert result.height() <= 200

    def test_handles_rawpy_exception(self) -> None:
        mock_rawpy_module = MagicMock()
        mock_rawpy_module.imread.side_effect = RuntimeError("bad file")

        mock_support = MagicMock()
        mock_support.rawpy = mock_rawpy_module

        with patch("src.iPhoto.utils.image_loader.load_rawpy", return_value=mock_support):
            result = _load_raw_with_rawpy(Path("/fake/img.cr2"))
            assert result is None


class TestLoadQimageRawFallback:
    def test_raw_file_attempts_rawpy_first(self, tmp_path: Path) -> None:
        raw_file = tmp_path / "photo.cr2"
        raw_file.write_bytes(b"\x00" * 16)

        fake_rgb = np.zeros((50, 80, 3), dtype=np.uint8)
        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = fake_rgb
        mock_raw.__enter__ = MagicMock(return_value=mock_raw)
        mock_raw.__exit__ = MagicMock(return_value=False)

        mock_rawpy_module = MagicMock()
        mock_rawpy_module.imread.return_value = mock_raw

        mock_support = MagicMock()
        mock_support.rawpy = mock_rawpy_module

        with patch("src.iPhoto.utils.image_loader.load_rawpy", return_value=mock_support):
            result = load_qimage(raw_file)
            assert result is not None
            assert result.width() == 80
            assert result.height() == 50

    def test_non_raw_skips_rawpy(self, tmp_path: Path) -> None:
        jpg_file = tmp_path / "photo.jpg"
        # Write a minimal valid JPEG
        from PIL import Image
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        img.save(str(jpg_file), format="JPEG")

        with patch("src.iPhoto.utils.image_loader._load_raw_with_rawpy") as mock_raw:
            result = load_qimage(jpg_file)
            mock_raw.assert_not_called()
            assert result is not None


class TestGenerateRawMicroThumbnail:
    def test_generates_thumbnail_from_mock_rawpy(self) -> None:
        fake_rgb = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = fake_rgb
        mock_raw.__enter__ = MagicMock(return_value=mock_raw)
        mock_raw.__exit__ = MagicMock(return_value=False)

        mock_rawpy_module = MagicMock()
        mock_rawpy_module.imread.return_value = mock_raw

        mock_support = MagicMock()
        mock_support.rawpy = mock_rawpy_module

        with patch("src.iPhoto.utils.image_loader.load_rawpy", return_value=mock_support):
            result = _generate_raw_micro_thumbnail(Path("/fake/img.arw"))
            assert result is not None
            assert isinstance(result, bytes)
            # Verify it's a valid JPEG
            from PIL import Image
            img = Image.open(BytesIO(result))
            assert img.format == "JPEG"
            assert max(img.size) <= 16

    def test_returns_none_when_rawpy_unavailable(self) -> None:
        with patch("src.iPhoto.utils.image_loader.load_rawpy", return_value=None):
            result = _generate_raw_micro_thumbnail(Path("/fake/img.cr2"))
            assert result is None

    def test_generate_micro_thumbnail_dispatches_raw(self, tmp_path: Path) -> None:
        raw_file = tmp_path / "photo.dng"
        raw_file.write_bytes(b"\x00" * 16)

        with patch("src.iPhoto.utils.image_loader._generate_raw_micro_thumbnail", return_value=b"jpeg") as mock_fn:
            result = generate_micro_thumbnail(raw_file)
            mock_fn.assert_called_once_with(raw_file)
            assert result == b"jpeg"
