"""Tests proving that ParallelScanner uses multiple threads and
scan_album correctly yields all discovered files."""

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
# scan_album: prove sequential batch processing with streaming results
# ---------------------------------------------------------------------------

class TestScanAlbumBatches:
    """Prove that scan_album processes batches and yields results incrementally."""

    def test_all_files_yielded(self, tmp_path: Path):
        """Create files and verify all are yielded from scan_album."""
        for i in range(25):
            (tmp_path / f"photo_{i:04d}.jpg").write_text("x" * 100)

        def _fake_process(root, image_paths, video_paths):
            for p in image_paths + video_paths:
                yield {
                    "rel": p.relative_to(root).as_posix(),
                    "bytes": 100,
                    "ts": 0,
                    "id": f"as_{p.name}",
                    "media_type": 0,
                }

        with patch.object(
            _sa_mod, "process_media_paths",
            side_effect=_fake_process,
        ):
            rows = list(_sa_mod.scan_album(tmp_path, ["*.jpg"], []))

        assert len(rows) == 25

    def test_cached_items_bypass_processing(self, tmp_path: Path):
        """Cached items should be yielded without calling process_media_paths."""
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

        call_count = 0
        original = _sa_mod.process_media_paths

        def _counting_process(root, image_paths, video_paths):
            nonlocal call_count
            call_count += 1
            yield from original(root, image_paths, video_paths)

        with patch.object(_sa_mod, "process_media_paths", side_effect=_counting_process):
            rows = list(_sa_mod.scan_album(
                tmp_path, ["*.jpg"], [],
                existing_index=existing_index,
            ))

        assert len(rows) == 10
        assert call_count == 0, "Cached items should not trigger metadata extraction"
