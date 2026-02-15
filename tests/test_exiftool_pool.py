"""Tests for the persistent ExifToolPool."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Set
from unittest.mock import patch, MagicMock

import pytest

from iPhoto.utils.exiftool_pool import (
    ExifToolPool,
    PersistentExifTool,
    get_exiftool_pool,
    shutdown_exiftool_pool,
)


class TestExifToolPool:
    """Verify pool lifecycle and concurrent access patterns."""

    def test_pool_creates_correct_number_of_instances(self):
        """Pool should contain exactly N exiftool instances."""
        pool = ExifToolPool(size=3)
        assert pool.size == 3
        assert len(pool._tools) == 3

    def test_pool_get_put_roundtrip(self):
        """Borrowing and returning an instance should work without error."""
        pool = ExifToolPool(size=2)
        pool._started = True
        pool._queue.put(pool._tools[0])
        pool._queue.put(pool._tools[1])

        borrowed1 = pool.get(timeout=1.0)
        borrowed2 = pool.get(timeout=1.0)
        assert borrowed1 is pool._tools[0]
        assert borrowed2 is pool._tools[1]

        # Queue should now be empty
        assert pool._queue.empty()

        # Return them
        pool.put(borrowed1)
        pool.put(borrowed2)
        assert not pool._queue.empty()

    def test_pool_concurrent_access(self):
        """Multiple threads should be able to borrow and return safely."""
        pool = ExifToolPool(size=4)
        pool._started = True
        for tool in pool._tools:
            pool._queue.put(tool)

        observed_threads: Set[str] = set()
        lock = threading.Lock()

        def _worker():
            et = pool.get(timeout=5.0)
            with lock:
                observed_threads.add(threading.current_thread().name)
            import time
            time.sleep(0.05)
            pool.put(et)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # All 8 threads should have run
        assert len(observed_threads) == 8

    def test_shutdown_idempotent(self):
        """Calling shutdown multiple times should not raise."""
        pool = ExifToolPool(size=1)
        pool._started = True
        pool.shutdown()
        pool.shutdown()  # Should not raise

    def test_global_pool_singleton(self):
        """get_exiftool_pool should return the same pool on repeated calls."""
        shutdown_exiftool_pool()  # Clean state

        with patch.object(PersistentExifTool, "start"):
            pool1 = get_exiftool_pool(size=2)
            pool2 = get_exiftool_pool(size=2)

        assert pool1 is pool2
        shutdown_exiftool_pool()  # Cleanup


class TestPersistentExifTool:
    """Verify PersistentExifTool lifecycle (without real exiftool)."""

    def test_init_does_not_require_exiftool(self):
        """PersistentExifTool() should succeed without exiftool installed."""
        et = PersistentExifTool()
        assert et._executable is None
        assert not et.alive

    def test_get_metadata_batch_empty(self):
        """Empty paths list should return empty results."""
        et = PersistentExifTool()
        assert et.get_metadata_batch([]) == []
