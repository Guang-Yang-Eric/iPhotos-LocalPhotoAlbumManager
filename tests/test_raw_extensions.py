"""Tests for RAW extension support in media_classifier."""

from pathlib import Path

import pytest

from src.iPhoto.media_classifier import (
    IMAGE_EXTENSIONS,
    RAW_EXTENSIONS,
    classify_media,
    get_media_type,
    MediaType,
)


# All RAW extensions that should be recognised as images.
_RAW_EXTS = [
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".orf", ".rw2", ".raf", ".dng", ".pef", ".raw", ".rwl",
    ".3fr", ".iiq", ".x3f", ".srw", ".erf",
]


class TestRawExtensionsInImageExtensions:
    """Every RAW extension must be part of IMAGE_EXTENSIONS."""

    @pytest.mark.parametrize("ext", _RAW_EXTS)
    def test_raw_ext_in_image_extensions(self, ext: str) -> None:
        assert ext in IMAGE_EXTENSIONS

    @pytest.mark.parametrize("ext", _RAW_EXTS)
    def test_raw_ext_in_raw_extensions(self, ext: str) -> None:
        assert ext in RAW_EXTENSIONS


class TestClassifyMediaRaw:
    """classify_media should identify RAW files as images."""

    @pytest.mark.parametrize("ext", _RAW_EXTS)
    def test_classify_raw_by_extension(self, ext: str) -> None:
        row = {"rel": f"photo{ext}"}
        assert classify_media(row) == (True, False)

    @pytest.mark.parametrize("ext", [e.upper() for e in _RAW_EXTS])
    def test_classify_raw_uppercase(self, ext: str) -> None:
        row = {"rel": f"PHOTO{ext}"}
        assert classify_media(row) == (True, False)


class TestGetMediaTypeRaw:
    """get_media_type should return IMAGE for RAW files."""

    @pytest.mark.parametrize("ext", _RAW_EXTS)
    def test_get_media_type_raw(self, ext: str) -> None:
        assert get_media_type(Path(f"img{ext}")) == MediaType.IMAGE
