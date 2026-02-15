"""Tests proving that ParallelScanner uses multiple threads and
scan_album correctly yields all discovered files using parallel workers."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Set
from unittest.mock import MagicMock, patch

import pytest

from iPhoto.application.services.parallel_scanner import ParallelScanner, ScanResult
from iPhoto.domain.models.core import Asset, MediaType
import iPhoto.io.scanner_adapter as _sa_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(path: Path) -> Asset:
    return Asset(
        id=path.name,
        album_id="test",
        path=path,
        media_type=MediaType.IMAGE,
        size_bytes=0,
    )


# ---------------------------------------------------------------------------
# ParallelScanner: prove multiple threads are used
# ---------------------------------------------------------------------------

class TestParallelScannerThreading:
    """Prove that ParallelScanner.scan() distributes work across threads."""

    def test_multiple_threads_used(self, tmp_path: Path):
        """Create enough files so that the thread pool must use >1 thread."""
        for i in range(20):
            (tmp_path / f"img_{i:03d}.jpg").write_text("x")

        observed_threads: Set[str] = set()
        lock = threading.Lock()

        def _slow_scan(path: Path) -> Asset:
            with lock:
                observed_threads.add(threading.current_thread().name)
            time.sleep(0.05)  # simulate I/O
            return _make_asset(path)

        scanner = ParallelScanner(max_workers=4, scan_file_fn=_slow_scan)
        result = scanner.scan(tmp_path)

        assert result.total_processed == 20
        # With 20 files at 50ms each and 4 workers, multiple threads must fire
        assert len(observed_threads) > 1, (
            f"Expected multiple threads, got: {observed_threads}"
        )

    def test_thread_names_in_log(self, tmp_path: Path, caplog):
        """Verify that per-thread distribution is logged."""
        for i in range(8):
            (tmp_path / f"p{i}.jpg").write_text("x")

        scanner = ParallelScanner(max_workers=2, scan_file_fn=_make_asset)
        with caplog.at_level(logging.INFO, logger="iPhoto.application.services.parallel_scanner"):
            scanner.scan(tmp_path)

        log_text = caplog.text
        assert "Per-thread work distribution" in log_text
        assert "PScan" in log_text  # thread name prefix
        assert "ParallelScanner.scan complete" in log_text


# ---------------------------------------------------------------------------
# scan_album: prove parallel batch processing with multiple workers
# ---------------------------------------------------------------------------

class TestScanAlbumParallelWorkers:
    """Prove that scan_album uses multiple _MetadataWorkerThread instances."""

    def test_all_files_yielded_with_workers(self, tmp_path: Path):
        """All discovered files should be yielded via parallel workers."""
        for i in range(40):
            (tmp_path / f"photo_{i:04d}.jpg").write_text("x" * 100)

        # The workers call _metadata_provider.normalize_metadata and
        # et.get_metadata_batch.  We patch them to return synthetic rows
        # without needing exiftool.
        def _fake_normalize(root, path, raw_meta):
            return {
                "rel": path.relative_to(root).as_posix(),
                "bytes": 100,
                "ts": 0,
                "id": f"as_{path.name}",
                "media_type": 0,
            }

        def _fake_et_batch(paths):
            return [{"SourceFile": p.as_posix()} for p in paths]

        with patch.object(_sa_mod._metadata_provider, "normalize_metadata", side_effect=_fake_normalize), \
             patch.object(_sa_mod._metadata_provider, "get_metadata_batch", side_effect=_fake_et_batch), \
             patch("iPhoto.io.scanner_adapter.get_exiftool_pool") as mock_pool:

            # Create a fake pool that returns a mock exiftool instance
            fake_et = MagicMock()
            fake_et.get_metadata_batch.side_effect = _fake_et_batch
            mock_pool.return_value.get.return_value = fake_et
            mock_pool.return_value.put.return_value = None

            rows = list(_sa_mod.scan_album(tmp_path, ["*.jpg"], [], num_workers=4))

        assert len(rows) == 40

    def test_multiple_worker_threads_used(self, tmp_path: Path):
        """Multiple ScanWorker-N threads should process files concurrently."""
        for i in range(60):
            (tmp_path / f"photo_{i:04d}.jpg").write_text("x" * 100)

        observed_threads: Set[str] = set()
        lock = threading.Lock()

        original_normalize = _sa_mod._metadata_provider.normalize_metadata

        def _tracking_normalize(root, path, raw_meta):
            with lock:
                observed_threads.add(threading.current_thread().name)
            time.sleep(0.02)  # simulate I/O
            return {
                "rel": path.relative_to(root).as_posix(),
                "bytes": 100,
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

        assert len(rows) == 60
        scan_threads = {t for t in observed_threads if t.startswith("ScanWorker-")}
        assert len(scan_threads) > 1, (
            f"Expected multiple ScanWorker threads, got: {scan_threads}"
        )

    def test_cached_items_bypass_exiftool(self, tmp_path: Path):
        """Cached items should be yielded without exiftool extraction."""
        for i in range(10):
            p = tmp_path / f"img_{i:03d}.jpg"
            p.write_text("x" * 10)

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
