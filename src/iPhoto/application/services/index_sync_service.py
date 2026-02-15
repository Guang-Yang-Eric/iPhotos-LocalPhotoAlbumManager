"""Index synchronisation logic extracted from app.py."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...cache.index_store import get_global_repository
from ...cache.lock import FileLock
from ...config import WORK_DIR_NAME
from ...core.pairing import pair_live
from ...errors import IndexCorruptedError, ManifestInvalidError
from ...domain.models.core import LiveGroup
from ...utils.jsonio import read_json, write_json
from ...utils.logging import get_logger
from .path_normalizer import PathNormalizer

LOGGER = get_logger()


class IndexSyncService:
    """Encapsulates index synchronisation operations."""

    @staticmethod
    def ensure_links(
        root: Path, rows: List[dict], library_root: Optional[Path] = None
    ) -> None:
        """Ensure links.json and DB are synchronized with the given rows.

        Args:
            root: The album root directory.
            rows: List of asset rows (with album-relative paths).
            library_root: If provided, use this as the database root.
        """
        work_dir = root / WORK_DIR_NAME
        links_path = work_dir / "links.json"
        groups, payload = IndexSyncService.compute_links_payload(rows)

        if links_path.exists():
            try:
                existing: Dict[str, object] = read_json(links_path)
            except ManifestInvalidError:
                existing = {}
            if existing == payload:
                IndexSyncService.sync_live_roles_to_db(
                    root, groups, library_root=library_root
                )
                return

        LOGGER.info("Updating links.json for %s", root)
        IndexSyncService.write_links(root, payload)
        IndexSyncService.sync_live_roles_to_db(
            root, groups, library_root=library_root
        )

    @staticmethod
    def compute_links_payload(
        rows: List[dict],
    ) -> tuple[List[LiveGroup], Dict[str, object]]:
        groups = pair_live(rows)
        payload: Dict[str, object] = {
            "schema": "iPhoto/links@1",
            "live_groups": [asdict(group) for group in groups],
            "clips": [],
        }
        return groups, payload

    @staticmethod
    def write_links(root: Path, payload: Dict[str, object]) -> None:
        work_dir = root / WORK_DIR_NAME
        with FileLock(root, "links"):
            write_json(
                work_dir / "links.json", payload, backup_dir=work_dir / "manifest.bak"
            )

    @staticmethod
    def sync_live_roles_to_db(
        root: Path, groups: List[LiveGroup], library_root: Optional[Path] = None
    ) -> None:
        """Propagate live photo roles from computed groups to the repository.

        Args:
            root: The album root directory.
            groups: List of LiveGroup objects to sync.
            library_root: If provided, use this as the database root (global database).
        """
        updates: List[Tuple[str, int, Optional[str]]] = []

        album_prefix = ""
        if library_root:
            rel = PathNormalizer.compute_album_path(root, library_root)
            if rel:
                album_prefix = f"{rel}/"

        for group in groups:
            if not group.still or not group.motion:
                continue

            still_rel = (
                f"{album_prefix}{group.still}" if album_prefix else group.still
            )
            motion_rel = (
                f"{album_prefix}{group.motion}" if album_prefix else group.motion
            )
            updates.append((still_rel, 0, motion_rel))
            updates.append((motion_rel, 1, still_rel))

        db_root = library_root if library_root else root
        store = get_global_repository(db_root)
        if album_prefix:
            store.apply_live_role_updates_for_prefix(album_prefix, updates)
        else:
            store.apply_live_role_updates(updates)

    @staticmethod
    def update_index_snapshot(
        root: Path,
        materialised_rows: List[dict],
        library_root: Optional[Path] = None,
    ) -> None:
        """Apply *materialised_rows* to the global database using additive-only updates.

        This function implements **Constraint #4: Additive-Only "Fact Supplementation"**:
        - Scanning is for discovering facts, not removing them
        - Files not found during a partial scan are NOT deleted from the database
        - Deletion is a separate lifecycle event and never occurs during scan

        The function uses idempotent upsert operations to ensure duplicate scans
        don't create duplicate data (Constraint #3).

        Args:
            root: The album root directory.
            materialised_rows: List of rows to update/insert.
            library_root: If provided, use this as the database root (global database).
        """
        db_root = library_root if library_root else root
        store = get_global_repository(db_root)

        corrupted_during_read = False
        try:
            list(store.read_all())
        except IndexCorruptedError:
            corrupted_during_read = True

        fresh_rows: Dict[str, dict] = {}
        for row in materialised_rows:
            rel_key = PathNormalizer.normalise_rel_key(row.get("rel"))
            if rel_key is None:
                continue
            fresh_rows[rel_key] = row

        materialised_snapshot = list(fresh_rows.values())

        if corrupted_during_read:
            store.write_rows(materialised_snapshot)
            return

        if not fresh_rows:
            return

        try:
            store.append_rows(materialised_snapshot)
        except IndexCorruptedError:
            store.write_rows(materialised_snapshot)
