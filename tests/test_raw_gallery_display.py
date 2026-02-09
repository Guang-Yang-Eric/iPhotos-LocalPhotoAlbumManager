"""Tests for the _to_dto pano detection with None width/height values."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.iPhoto.gui.viewmodels.asset_data_source import AssetDataSource


def _make_mock_asset(
    *,
    path: str = "photo.cr2",
    media_type: int = 0,
    width=None,
    height=None,
    size_bytes=None,
    metadata=None,
    is_favorite: bool = False,
    live_photo_group_id=None,
    duration=None,
    created_at=None,
    id: str = "as_test123",
):
    asset = MagicMock()
    asset.path = Path(path)
    asset.media_type = media_type
    asset.width = width
    asset.height = height
    asset.size_bytes = size_bytes
    asset.metadata = metadata
    asset.is_favorite = is_favorite
    asset.live_photo_group_id = live_photo_group_id
    asset.duration = duration
    asset.created_at = created_at
    asset.id = id
    return asset


def _make_data_source(library_root: Path) -> AssetDataSource:
    repo = MagicMock()
    ds = AssetDataSource(repo, library_root)
    return ds


class TestToDtoNoneWidthHeight:
    """Regression tests for TypeError when width/height are None."""

    def test_none_width_height_no_crash(self, tmp_path: Path) -> None:
        """_to_dto should not crash when asset.width and asset.height are None."""
        ds = _make_data_source(tmp_path)
        asset = _make_mock_asset(width=None, height=None)
        # Should not raise TypeError: '>' not supported between 'NoneType' and 'int'
        dto = ds._to_dto(asset)
        assert dto is not None
        assert dto.width == 0
        assert dto.height == 0

    def test_none_width_with_valid_height(self, tmp_path: Path) -> None:
        """_to_dto should handle None width with valid height."""
        ds = _make_data_source(tmp_path)
        asset = _make_mock_asset(width=None, height=1000)
        dto = ds._to_dto(asset)
        assert dto is not None
        assert dto.width == 0
        assert dto.height == 1000

    def test_valid_width_height(self, tmp_path: Path) -> None:
        """_to_dto should correctly handle valid width/height."""
        ds = _make_data_source(tmp_path)
        asset = _make_mock_asset(width=4000, height=3000)
        dto = ds._to_dto(asset)
        assert dto is not None
        assert dto.width == 4000
        assert dto.height == 3000

    def test_raw_file_none_dimensions_from_metadata(self, tmp_path: Path) -> None:
        """RAW file with dimensions in metadata should produce valid DTO."""
        ds = _make_data_source(tmp_path)
        asset = _make_mock_asset(
            path="photo.cr2",
            width=None,
            height=None,
            metadata={"w": 6000, "h": 4000},
        )
        dto = ds._to_dto(asset)
        assert dto is not None
        assert dto.width == 6000
        assert dto.height == 4000

    def test_pano_detection_with_none_dimensions(self, tmp_path: Path) -> None:
        """Pano detection should not crash when dimensions are None."""
        ds = _make_data_source(tmp_path)
        asset = _make_mock_asset(
            width=None,
            height=None,
            size_bytes=5_000_000,
        )
        dto = ds._to_dto(asset)
        assert dto is not None
        assert dto.is_pano is False


class TestMetadataProviderRawExtensions:
    """Ensure RAW extensions are recognised by the metadata provider."""

    def test_raw_extensions_in_image_set(self) -> None:
        from src.iPhoto.infrastructure.services.metadata_provider import ExifToolMetadataProvider
        provider = ExifToolMetadataProvider()
        for ext in (".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf"):
            assert ext in provider._IMAGE_EXTENSIONS, f"{ext} not in _IMAGE_EXTENSIONS"

    def test_raw_file_gets_image_media_type(self, tmp_path: Path) -> None:
        from src.iPhoto.infrastructure.services.metadata_provider import ExifToolMetadataProvider
        provider = ExifToolMetadataProvider()
        # Create a dummy file
        raw_file = tmp_path / "test.cr2"
        raw_file.write_bytes(b"\x00" * 64)
        # normalize_metadata should assign media_type=0 (IMAGE) for RAW files
        row = provider.normalize_metadata(tmp_path, raw_file, {})
        assert row["media_type"] == 0


class TestPairingRawExtensions:
    """Ensure RAW extensions are included in pairing image set."""

    def test_raw_is_photo(self) -> None:
        from src.iPhoto.core.pairing import _is_photo
        for ext in (".cr2", ".nef", ".arw", ".dng"):
            row = {"rel": f"photo{ext}"}
            assert _is_photo(row), f"RAW file with {ext} not recognised as photo"
