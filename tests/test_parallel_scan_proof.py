"""Tests proving that scan_album and ParallelScanner use multiple threads."""

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
# scan_album: prove parallel batch processing
# ---------------------------------------------------------------------------

class TestScanAlbumParallelBatches:
    """Prove that scan_album processes batches on multiple threads."""

    def test_multiple_threads_for_batches(self, tmp_path: Path):
        """Create >100 files so multiple batches are created, then verify
        that more than one ScanWorker thread processed them."""
        # Create 120 media files to trigger at least 2 batches of 50
        for i in range(120):
            (tmp_path / f"photo_{i:04d}.jpg").write_text("x" * 100)

        observed_threads: Set[str] = set()
        lock = threading.Lock()

        original_process = None  # will be set via patching

        def _tracking_process(root, image_paths, video_paths):
            """Wrapper that records the thread before delegating."""
            with lock:
                observed_threads.add(threading.current_thread().name)
            time.sleep(0.1)  # simulate I/O to create overlap
            # Yield simple rows instead of calling exiftool
            for p in image_paths + video_paths:
                yield {
                    "rel": p.relative_to(root).as_posix(),
                    "bytes": 100,
                    "ts": 0,
                    "id": f"as_{p.name}",
                    "media_type": 0,
                }

        with patch(
            "iPhoto.io.scanner_adapter.process_media_paths",
            side_effect=_tracking_process,
        ):
            from iPhoto.io.scanner_adapter import scan_album

            rows = list(scan_album(
                tmp_path,
                ["*.jpg"],
                [],
                num_workers=4,
            ))

        assert len(rows) == 120
        # Batches should have run on separate ScanWorker threads
        assert len(observed_threads) > 1, (
            f"Expected multiple threads, got: {observed_threads}"
        )

    def test_scan_album_log_output(self, tmp_path: Path, caplog):
        """Verify that scan_album logs parallel scan start/complete messages."""
        for i in range(5):
            (tmp_path / f"img{i}.jpg").write_text("data")

        def _fake_process(root, image_paths, video_paths):
            for p in image_paths + video_paths:
                yield {
                    "rel": p.relative_to(root).as_posix(),
                    "bytes": 4,
                    "ts": 0,
                    "id": f"as_{p.name}",
                    "media_type": 0,
                }

        with patch(
            "iPhoto.io.scanner_adapter.process_media_paths",
            side_effect=_fake_process,
        ):
            from iPhoto.io.scanner_adapter import scan_album

            with caplog.at_level(logging.INFO, logger="iPhoto.scanner"):
                rows = list(scan_album(tmp_path, ["*.jpg"], [], num_workers=2))

        log_text = caplog.text
        assert "Parallel scan started" in log_text
        assert "Parallel scan complete" in log_text
        assert "workers=2" in log_text
