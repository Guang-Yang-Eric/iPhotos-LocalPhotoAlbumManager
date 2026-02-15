"""Tests verifying that parallel scan results stream incrementally and the
ViewModel uses efficient beginInsertRows instead of full model resets.
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
# scan_album streaming: results should arrive as soon as each file is
# processed, not only when an entire batch future completes.
# ---------------------------------------------------------------------------


class TestScanAlbumStreaming:
    """Verify that scan_album yields results incrementally via result queue."""

    def test_results_stream_during_batch(self, tmp_path: Path):
        """Even within a single batch being processed, rows should stream
        to the caller as soon as each file is ready — not all at once."""
        for i in range(20):
            (tmp_path / f"img_{i:03d}.jpg").write_text("x" * 10)

        yield_times: List[float] = []
        t0 = time.monotonic()

        def _slow_process(root, image_paths, video_paths):
            for p in image_paths + video_paths:
                time.sleep(0.05)  # simulate I/O per file
                yield {
                    "rel": p.relative_to(root).as_posix(),
                    "bytes": 10,
                    "ts": 0,
                    "id": f"as_{p.name}",
                    "media_type": 0,
                }

        with patch.object(_sa_mod, "process_media_paths", side_effect=_slow_process):
            for row in _sa_mod.scan_album(tmp_path, ["*.jpg"], [], num_workers=4):
                yield_times.append(time.monotonic() - t0)

        assert len(yield_times) == 20

        # With 4 workers and 10 files per batch (2 batches), results should
        # start arriving much sooner than the total processing time.
        # If all 20 files at 50ms each were sequential, it'd take ~1s.
        # With 4 workers, ~250ms.  But key point: the FIRST result should
        # arrive well before the last.
        first_arrival = yield_times[0]
        last_arrival = yield_times[-1]
        assert last_arrival - first_arrival > 0.01, (
            "Results should arrive incrementally, not all at once"
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

        # process_media_paths should NOT be called for cached files
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
                num_workers=2,
            ))

        assert len(rows) == 10
        # All items were cached, so process_media_paths should not have been called
        assert call_count == 0, "Cached items should not trigger metadata extraction"

    def test_batch_size_is_10(self, tmp_path: Path):
        """Verify that scan_album uses batch size 10, matching the
        ScannerWorker.SCAN_CHUNK_SIZE for consistent streaming."""
        for i in range(25):
            (tmp_path / f"img_{i:03d}.jpg").write_text("x" * 10)

        batch_sizes: List[int] = []
        lock = threading.Lock()

        def _recording_process(root, image_paths, video_paths):
            with lock:
                batch_sizes.append(len(image_paths) + len(video_paths))
            for p in image_paths + video_paths:
                yield {
                    "rel": p.relative_to(root).as_posix(),
                    "bytes": 10,
                    "ts": 0,
                    "id": f"as_{p.name}",
                    "media_type": 0,
                }

        with patch.object(_sa_mod, "process_media_paths", side_effect=_recording_process):
            rows = list(_sa_mod.scan_album(tmp_path, ["*.jpg"], [], num_workers=1))

        assert len(rows) == 25
        # Batches should be ≤10 (the new batch size)
        for bs in batch_sizes:
            assert bs <= 10, f"Batch size {bs} exceeds expected maximum of 10"


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
