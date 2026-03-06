"""Tests for live photo display fixes: SQL filter, DTO is_live detection, and viewport rendering."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from iPhoto.domain.models import Asset, MediaType
from iPhoto.domain.models.query import AssetQuery
from iPhoto.gui.viewmodels.asset_dto_converter import to_dto, scan_row_to_dto
from iPhoto.infrastructure.db.pool import ConnectionPool
from iPhoto.infrastructure.repositories.sqlite_asset_repository import SQLiteAssetRepository


# ---------------------------------------------------------------------------
# SQL filter: Live Photos collection must find assets with NULL live_role
# ---------------------------------------------------------------------------

class TestLivePhotoSQLFilter:
    """Verify the Live Photos collection query handles live_role edge cases."""

    @pytest.fixture()
    def repo(self, tmp_path: Path) -> SQLiteAssetRepository:
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path)
        repo = SQLiteAssetRepository(pool)
        return repo

    def _insert_asset(
        self,
        repo: SQLiteAssetRepository,
        rel: str,
        *,
        media_type: int = 0,
        live_role=0,
        live_partner_rel=None,
        live_photo_group_id=None,
    ) -> None:
        with repo._pool.connection() as conn:
            conn.execute(
                "INSERT INTO assets "
                "(rel, id, media_type, bytes, dt, w, h, parent_album_path, "
                " live_role, live_partner_rel, live_photo_group_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rel,
                    rel,
                    media_type,
                    1000,
                    "2024-01-01T12:00:00",
                    100,
                    100,
                    "album",
                    live_role,
                    live_partner_rel,
                    live_photo_group_id,
                ),
            )

    def test_live_collection_finds_asset_with_null_live_role(self, repo):
        """Assets with live_partner_rel but NULL live_role must appear in the Live Photos collection."""
        self._insert_asset(
            repo,
            "photo.heic",
            live_role=None,
            live_partner_rel="motion.mov",
        )
        query = AssetQuery(media_types=[MediaType.LIVE_PHOTO])
        assets = repo.find_by_query(query)
        assert len(assets) == 1

    def test_live_collection_finds_asset_with_zero_live_role(self, repo):
        """Assets with live_role=0 and live_partner_rel must appear normally."""
        self._insert_asset(
            repo,
            "photo.heic",
            live_role=0,
            live_partner_rel="motion.mov",
        )
        query = AssetQuery(media_types=[MediaType.LIVE_PHOTO])
        assets = repo.find_by_query(query)
        assert len(assets) == 1

    def test_live_collection_finds_asset_with_group_id(self, repo):
        """Assets with live_photo_group_id (not video) must appear."""
        self._insert_asset(
            repo,
            "photo.heic",
            media_type=0,
            live_photo_group_id="group-1",
        )
        query = AssetQuery(media_types=[MediaType.LIVE_PHOTO])
        assets = repo.find_by_query(query)
        assert len(assets) == 1

    def test_live_collection_excludes_motion_component(self, repo):
        """Motion components (live_role=1) must not appear."""
        self._insert_asset(
            repo,
            "motion.mov",
            media_type=1,
            live_role=1,
            live_partner_rel="photo.heic",
        )
        query = AssetQuery(media_types=[MediaType.LIVE_PHOTO])
        assets = repo.find_by_query(query)
        assert len(assets) == 0


# ---------------------------------------------------------------------------
# _map_row_to_asset: synthetic group from metadata JSON
# ---------------------------------------------------------------------------

class TestMapRowSyntheticGroup:
    """_map_row_to_asset should derive live_photo_group_id from metadata JSON
    when the DB column is NULL but the metadata JSON has live_partner_rel."""

    @pytest.fixture()
    def repo(self, tmp_path: Path) -> SQLiteAssetRepository:
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path)
        repo = SQLiteAssetRepository(pool)
        return repo

    def test_synthetic_group_from_metadata_json(self, repo):
        """When live_partner_rel is in metadata JSON but not in the column,
        the asset should still get a synthetic group ID."""
        meta_json = json.dumps({"live_partner_rel": "motion.mov", "live_role": 0})
        with repo._pool.connection() as conn:
            conn.execute(
                "INSERT INTO assets "
                "(rel, id, media_type, bytes, dt, w, h, parent_album_path, "
                " metadata, live_role, live_partner_rel, live_photo_group_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "photo.heic",
                    "id-1",
                    0,
                    1000,
                    "2024-01-01T12:00:00",
                    100,
                    100,
                    "album",
                    meta_json,
                    None,  # live_role column NULL
                    None,  # live_partner_rel column NULL
                    None,  # live_photo_group_id column NULL
                ),
            )
        query = AssetQuery()
        assets = repo.find_by_query(query)
        assert len(assets) == 1
        assert assets[0].live_photo_group_id == "motion.mov"

    def test_column_takes_precedence_over_json(self, repo):
        """When the column has data, it should take precedence over JSON."""
        meta_json = json.dumps({"live_partner_rel": "old_motion.mov"})
        with repo._pool.connection() as conn:
            conn.execute(
                "INSERT INTO assets "
                "(rel, id, media_type, bytes, dt, w, h, parent_album_path, "
                " metadata, live_role, live_partner_rel) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "photo.heic",
                    "id-1",
                    0,
                    1000,
                    "2024-01-01T12:00:00",
                    100,
                    100,
                    "album",
                    meta_json,
                    0,
                    "new_motion.mov",
                ),
            )
        query = AssetQuery()
        assets = repo.find_by_query(query)
        assert len(assets) == 1
        # Column value should win
        assert assets[0].live_photo_group_id == "new_motion.mov"


# ---------------------------------------------------------------------------
# to_dto: is_live flag detection
# ---------------------------------------------------------------------------

class TestDtoIsLiveDetection:
    """Verify is_live is set correctly across all detection paths."""

    def test_is_live_via_group_id(self):
        asset = Asset(
            id="1",
            album_id="a",
            path=Path("photo.heic"),
            media_type=MediaType.IMAGE,
            size_bytes=0,
            live_photo_group_id="group-1",
        )
        dto = to_dto(asset, library_root=None)
        assert dto.is_live is True

    def test_is_live_via_metadata_partner(self):
        asset = Asset(
            id="2",
            album_id="a",
            path=Path("photo.heic"),
            media_type=MediaType.IMAGE,
            size_bytes=0,
            metadata={"live_partner_rel": "motion.mov", "live_role": 0},
        )
        dto = to_dto(asset, library_root=None)
        assert dto.is_live is True

    def test_is_live_via_metadata_partner_null_role(self):
        """live_role=None (not in metadata) should still yield is_live=True."""
        asset = Asset(
            id="3",
            album_id="a",
            path=Path("photo.heic"),
            media_type=MediaType.IMAGE,
            size_bytes=0,
            metadata={"live_partner_rel": "motion.mov"},
        )
        dto = to_dto(asset, library_root=None)
        assert dto.is_live is True

    def test_video_with_group_id_not_live(self):
        asset = Asset(
            id="4",
            album_id="a",
            path=Path("video.mov"),
            media_type=MediaType.VIDEO,
            size_bytes=0,
            live_photo_group_id="group-1",
        )
        dto = to_dto(asset, library_root=None)
        assert dto.is_live is False

    def test_no_pairing_data_not_live(self):
        asset = Asset(
            id="5",
            album_id="a",
            path=Path("photo.jpg"),
            media_type=MediaType.IMAGE,
            size_bytes=0,
        )
        dto = to_dto(asset, library_root=None)
        assert dto.is_live is False


# ---------------------------------------------------------------------------
# scan_row_to_dto: is_live flag for scan rows
# ---------------------------------------------------------------------------

class TestScanRowDtoIsLive:
    """Verify scan_row_to_dto correctly detects live photos."""

    def test_scan_row_with_live_partner(self):
        row = {
            "id": "r1",
            "media_type": 0,
            "live_partner_rel": "motion.mov",
        }
        dto = scan_row_to_dto(Path("/tmp"), "photo.heic", row)
        assert dto is not None
        assert dto.is_live is True

    def test_scan_row_video_not_live(self):
        row = {
            "id": "r2",
            "media_type": 1,
            "live_partner_rel": "photo.heic",
        }
        dto = scan_row_to_dto(Path("/tmp"), "motion.mov", row)
        assert dto is not None
        assert dto.is_live is False

    def test_scan_row_no_pairing_not_live(self):
        row = {"id": "r3", "media_type": 0}
        dto = scan_row_to_dto(Path("/tmp"), "photo.jpg", row)
        assert dto is not None
        assert dto.is_live is False
