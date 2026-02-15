"""Tests verifying that parallel scan results stream incrementally via
worker threads and the ViewModel's chunkedDtosReady signal routes correctly.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Set
from unittest.mock import MagicMock, patch

import pytest

import iPhoto.io.scanner_adapter as _sa_mod

_HAS_QTWIDGETS = False
try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
    _HAS_QTWIDGETS = True
except (ImportError, OSError):
    pass


# ---------------------------------------------------------------------------
# scan_album streaming: results should arrive incrementally from workers
# ---------------------------------------------------------------------------


class TestScanAlbumStreaming:
    """Verify that scan_album yields results incrementally via parallel workers."""

    def test_results_stream_across_workers(self, tmp_path: Path):
        """All files should be yielded and processed by parallel workers."""
        for i in range(40):
            (tmp_path / f"img_{i:03d}.jpg").write_text("x" * 10)

        observed_threads: Set[str] = set()
        lock = threading.Lock()

        def _tracking_normalize(root, path, raw_meta):
            with lock:
                observed_threads.add(threading.current_thread().name)
            time.sleep(0.02)  # simulate I/O per file
            return {
                "rel": path.relative_to(root).as_posix(),
                "bytes": 10,
                "ts": 0,
                "id": f"as_{path.name}",
                "media_type": 0,
            }

        def _fake_et_batch(paths):
            return [{"SourceFile": p.as_posix()} for p in paths]

        with patch.object(_sa_mod._metadata_provider, "normalize_metadata", side_effect=_tracking_normalize), \
             patch("iPhoto.io.scanner_adapter.get_exiftool_pool") as mock_pool:

            fake_et = MagicMock()
            fake_et.get_metadata_batch.side_effect = _fake_et_batch
            mock_pool.return_value.get.return_value = fake_et
            mock_pool.return_value.put.return_value = None

            rows = list(_sa_mod.scan_album(tmp_path, ["*.jpg"], [], num_workers=4))

        assert len(rows) == 40

        # Multiple ScanWorker threads should have processed files
        scan_threads = {t for t in observed_threads if t.startswith("ScanWorker-")}
        assert len(scan_threads) > 1, (
            f"Expected multiple ScanWorker threads, got: {scan_threads}"
        )

    def test_cached_results_stream_immediately(self, tmp_path: Path):
        """Rows matching the cache should be yielded immediately without
        waiting for exiftool to process new files."""
        for i in range(10):
            p = tmp_path / f"img_{i:03d}.jpg"
            p.write_text("x" * 10)

        # Build a fake index that matches all files
        existing_index = {}
        for i in range(10):
            p = tmp_path / f"img_{i:03d}.jpg"
            st = p.stat()
            existing_index[f"img_{i:03d}.jpg"] = {
                "rel": f"img_{i:03d}.jpg",
                "bytes": st.st_size,
                "ts": int(st.st_mtime * 1_000_000),
                "id": f"as_img_{i:03d}",
                "media_type": 0,
            }

        et_call_count = 0

        def _counting_et_batch(paths):
            nonlocal et_call_count
            et_call_count += 1
            return [{"SourceFile": p.as_posix()} for p in paths]

        with patch("iPhoto.io.scanner_adapter.get_exiftool_pool") as mock_pool:
            fake_et = MagicMock()
            fake_et.get_metadata_batch.side_effect = _counting_et_batch
            mock_pool.return_value.get.return_value = fake_et
            mock_pool.return_value.put.return_value = None

            rows = list(_sa_mod.scan_album(
                tmp_path, ["*.jpg"], [],
                existing_index=existing_index,
                num_workers=2,
            ))

        assert len(rows) == 10
        assert et_call_count == 0, "Cached items should not trigger exiftool"

    def test_exiftool_batch_size_is_20(self, tmp_path: Path):
        """Verify that workers receive batches of ≤20 files (the _EXIFTOOL_BATCH)."""
        for i in range(50):
            (tmp_path / f"img_{i:03d}.jpg").write_text("x" * 10)

        batch_sizes: List[int] = []
        lock = threading.Lock()

        def _recording_et_batch(paths):
            with lock:
                batch_sizes.append(len(paths))
            return [{"SourceFile": p.as_posix()} for p in paths]

        def _fake_normalize(root, path, raw_meta):
            return {
                "rel": path.relative_to(root).as_posix(),
                "bytes": 10,
                "ts": 0,
                "id": f"as_{path.name}",
                "media_type": 0,
            }

        with patch.object(_sa_mod._metadata_provider, "normalize_metadata", side_effect=_fake_normalize), \
             patch("iPhoto.io.scanner_adapter.get_exiftool_pool") as mock_pool:

            fake_et = MagicMock()
            fake_et.get_metadata_batch.side_effect = _recording_et_batch
            mock_pool.return_value.get.return_value = fake_et
            mock_pool.return_value.put.return_value = None

            rows = list(_sa_mod.scan_album(tmp_path, ["*.jpg"], [], num_workers=2))

        assert len(rows) == 50
        for bs in batch_sizes:
            assert bs <= 20, f"Batch size {bs} exceeds expected maximum of 20"


# ---------------------------------------------------------------------------
# AssetDataSource: chunkedDtosReady signal
# ---------------------------------------------------------------------------


class TestAssetDataSourceChunkedSignal:
    """Verify that handle_scan_chunk emits chunkedDtosReady instead of
    the generic dataChanged signal."""

    @pytest.mark.skipif(
        not _HAS_QTWIDGETS,
        reason="QtWidgets unavailable in headless environment",
    )
    def test_handle_scan_chunk_emits_chunked_signal(self, tmp_path: Path):
        """The DataSource should emit chunkedDtosReady (not dataChanged) for
        scan chunks so the ViewModel can use beginInsertRows."""
        from iPhoto.gui.viewmodels.asset_data_source import AssetDataSource
        from iPhoto.domain.models.query import AssetQuery

        repo = MagicMock()
        ds = AssetDataSource(repo, library_root=tmp_path)

        # Set up state so handle_scan_chunk will process rows
        ds._current_query = AssetQuery(album_path=str(tmp_path))
        ds.set_active_root(tmp_path)

        # Track signals
        data_changed_calls = []
        chunked_calls = []
        ds.dataChanged.connect(lambda: data_changed_calls.append(1))
        ds.chunkedDtosReady.connect(lambda dtos: chunked_calls.append(dtos))

        # Create test files and build a scan chunk
        (tmp_path / "test.jpg").write_text("x")
        chunk = [{
            "rel": "test.jpg",
            "bytes": 1,
            "ts": 0,
            "id": "as_test",
            "media_type": 0,
        }]

        ds.handle_scan_chunk(tmp_path, chunk)

        # chunkedDtosReady should fire, NOT dataChanged
        assert len(chunked_calls) == 1, "chunkedDtosReady should have been emitted"
        assert len(data_changed_calls) == 0, "dataChanged should NOT be emitted for scan chunks"
        assert len(chunked_calls[0]) == 1
