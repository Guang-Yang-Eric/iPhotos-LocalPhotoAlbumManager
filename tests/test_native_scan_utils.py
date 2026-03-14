"""Tests for the iPhoto._native scan C extension (P1–P3).

Tests are skipped gracefully when the C extension is unavailable
(e.g. gcc not installed, Windows, etc.).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module-level fixture: skip all tests if C extension not compiled
# ---------------------------------------------------------------------------

def _native_available() -> bool:
    try:
        from iPhoto._native import _C_AVAILABLE  # noqa: PLC0415
        return bool(_C_AVAILABLE)
    except Exception:
        return False


needs_native = pytest.mark.skipif(
    not _native_available(),
    reason="iPhoto._native C extension not available",
)


# ---------------------------------------------------------------------------
# P1 — parse_dt_fast (ISO 8601 → Unix microseconds)
# ---------------------------------------------------------------------------

class TestParseDtFast:
    """Tests for parse_dt_fast — ISO 8601 datetime parsing in C."""

    @needs_native
    def test_utc_basic(self):
        from iPhoto._native import parse_dt_fast
        from dateutil import parser as dp

        value = "2024-03-15T10:30:00Z"
        c_us = parse_dt_fast(value)
        python_us = int(dp.isoparse(value).timestamp() * 1_000_000)
        assert c_us == python_us

    @needs_native
    def test_positive_offset(self):
        from iPhoto._native import parse_dt_fast
        from dateutil import parser as dp

        value = "2024-03-15T18:30:00+08:00"
        c_us = parse_dt_fast(value)
        python_us = int(dp.isoparse(value).timestamp() * 1_000_000)
        assert c_us == python_us

    @needs_native
    def test_negative_offset(self):
        from iPhoto._native import parse_dt_fast
        from dateutil import parser as dp

        value = "2024-03-15T05:30:00-05:00"
        c_us = parse_dt_fast(value)
        python_us = int(dp.isoparse(value).timestamp() * 1_000_000)
        assert c_us == python_us

    @needs_native
    def test_subseconds(self):
        from iPhoto._native import parse_dt_fast
        from dateutil import parser as dp

        value = "2024-03-15T10:30:00.123456Z"
        c_us = parse_dt_fast(value)
        python_us = int(dp.isoparse(value).timestamp() * 1_000_000)
        assert c_us == python_us

    @needs_native
    def test_none_returns_none(self):
        from iPhoto._native import parse_dt_fast
        assert parse_dt_fast(None) is None

    @needs_native
    def test_empty_string_returns_none(self):
        from iPhoto._native import parse_dt_fast
        assert parse_dt_fast("") is None

    @needs_native
    def test_invalid_string_returns_none(self):
        from iPhoto._native import parse_dt_fast
        # Strings that are clearly not ISO 8601 (too short, wrong format)
        assert parse_dt_fast("not-a-date") is None
        assert parse_dt_fast("2024-03-15") is None  # date only, no time
        assert parse_dt_fast("hello world this is long enough") is None

    @needs_native
    def test_returns_int(self):
        from iPhoto._native import parse_dt_fast
        result = parse_dt_fast("2024-01-01T00:00:00Z")
        assert isinstance(result, int)

    def test_fallback_available(self):
        """parse_dt_fast must exist and callable even without C extension."""
        from iPhoto._native import parse_dt_fast
        assert callable(parse_dt_fast)
        # Should at minimum not raise on valid input
        result = parse_dt_fast("2024-03-15T10:30:00Z")
        assert result is not None and isinstance(result, int)


# ---------------------------------------------------------------------------
# P2 — compute_file_id_fast (file content hashing)
# ---------------------------------------------------------------------------

class TestComputeFileIdFast:
    """Tests for compute_file_id_fast — mmap-based XXH3 hashing in C."""

    @needs_native
    def test_small_file_matches_python(self, tmp_path):
        import xxhash
        from iPhoto._native import compute_file_id_fast

        data = b"hello world" * 1000  # ~11 KB (< 2 MB threshold)
        p = tmp_path / "small.bin"
        p.write_bytes(data)

        c_hex = compute_file_id_fast(p)
        python_hex = xxhash.xxh3_128(data).hexdigest()
        assert c_hex == python_hex

    @needs_native
    def test_empty_file(self, tmp_path):
        import xxhash
        from iPhoto._native import compute_file_id_fast

        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        c_hex = compute_file_id_fast(p)
        assert c_hex is not None
        assert len(c_hex) == 32
        assert c_hex == xxhash.xxh3_128(b"").hexdigest()

    @needs_native
    def test_nonexistent_file_returns_none(self):
        from iPhoto._native import compute_file_id_fast
        result = compute_file_id_fast(Path("/nonexistent/file_12345.bin"))
        assert result is None

    @needs_native
    def test_returns_32_char_hex(self, tmp_path):
        from iPhoto._native import compute_file_id_fast
        p = tmp_path / "test.bin"
        p.write_bytes(b"data")
        result = compute_file_id_fast(p)
        assert result is not None
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    @needs_native
    def test_consistent_with_hashutils(self, tmp_path):
        from iPhoto._native import compute_file_id_fast
        from iPhoto.utils.hashutils import compute_file_id

        data = b"test content for consistency check" * 500
        p = tmp_path / "check.bin"
        p.write_bytes(data)

        c_hex = compute_file_id_fast(p)
        py_hex = compute_file_id(p)
        assert c_hex == py_hex

    def test_fallback_available(self, tmp_path):
        """compute_file_id_fast must exist and callable even without C ext."""
        from iPhoto._native import compute_file_id_fast
        assert callable(compute_file_id_fast)
        p = tmp_path / "fallback.bin"
        p.write_bytes(b"fallback test")
        result = compute_file_id_fast(p)
        assert result is not None and len(result) == 32


# ---------------------------------------------------------------------------
# P3 — should_include_fast (glob path filtering)
# ---------------------------------------------------------------------------

class TestShouldIncludeFast:
    """Tests for should_include_fast — C fnmatch-based glob filtering."""

    @needs_native
    def test_simple_include(self):
        from iPhoto._native import should_include_fast
        assert should_include_fast("photo.jpg", ["*.jpg"], []) is True

    @needs_native
    def test_simple_exclude_wins(self):
        from iPhoto._native import should_include_fast
        assert should_include_fast("photo.jpg", ["*.jpg"], ["photo.jpg"]) is False

    @needs_native
    def test_recursive_pattern(self):
        from iPhoto._native import should_include_fast
        assert should_include_fast(
            "folder/sub/photo.jpg", ["**/*.jpg"], []
        ) is True

    @needs_native
    def test_no_match_returns_false(self):
        from iPhoto._native import should_include_fast
        assert should_include_fast("photo.txt", ["*.jpg"], []) is False

    @needs_native
    def test_multiple_include_patterns(self):
        from iPhoto._native import should_include_fast
        assert should_include_fast(
            "photo.jpg", ["*.png", "*.jpg"], []
        ) is True

    @needs_native
    def test_returns_bool(self):
        from iPhoto._native import should_include_fast
        result = should_include_fast("photo.jpg", ["*.jpg"], [])
        assert isinstance(result, bool)

    @needs_native
    def test_empty_include_returns_false(self):
        from iPhoto._native import should_include_fast
        assert should_include_fast("photo.jpg", [], []) is False

    def test_fallback_available(self):
        """should_include_fast must exist and callable even without C ext."""
        from iPhoto._native import should_include_fast
        assert callable(should_include_fast)
        result = should_include_fast("photo.jpg", ["*.jpg"], [])
        assert result is True


# ---------------------------------------------------------------------------
# Integration: pairing uses C-accelerated datetime parsing
# ---------------------------------------------------------------------------

class TestPairingWithCExtension:
    """Verify that pair_live still works correctly end-to-end with C parsing."""

    def _iso(self, y, mo, d, h, mi, s):
        from datetime import datetime, timezone
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    def test_time_match_with_c_parsing(self):
        from iPhoto.core.pairing import pair_live

        dt = self._iso(2024, 1, 1, 12, 0, 0)
        rows = [
            {"rel": "IMG_0001.HEIC", "mime": "image/heic", "dt": dt},
            {"rel": "IMG_0001.MOV",  "mime": "video/quicktime", "dt": dt, "dur": 1.5},
        ]
        groups = pair_live(rows)
        assert len(groups) == 1
        assert groups[0].still == "IMG_0001.HEIC"
        assert groups[0].motion == "IMG_0001.MOV"

    def test_time_match_respects_delta(self):
        from iPhoto.core.pairing import pair_live
        from iPhoto.config import PAIR_TIME_DELTA_SEC
        from datetime import datetime, timezone, timedelta

        dt_photo = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Video 1 second apart (within delta)
        dt_video_near = dt_photo + timedelta(seconds=1)
        # Video far outside delta
        dt_video_far = dt_photo + timedelta(seconds=PAIR_TIME_DELTA_SEC + 60)

        def iso(dt):
            return dt.isoformat().replace("+00:00", "Z")

        rows = [
            {"rel": "IMG_0001.HEIC", "mime": "image/heic", "dt": iso(dt_photo)},
            {"rel": "IMG_near.MOV",  "mime": "video/quicktime", "dt": iso(dt_video_near), "dur": 1.5},
            {"rel": "IMG_far.MOV",   "mime": "video/quicktime", "dt": iso(dt_video_far), "dur": 1.5},
        ]
        groups = pair_live(rows)
        assert len(groups) == 1
        assert groups[0].motion == "IMG_near.MOV"


# ---------------------------------------------------------------------------
# Integration: compute_file_id uses C-accelerated hashing
# ---------------------------------------------------------------------------

class TestHashUtilsWithCExtension:
    """Verify compute_file_id integrates C extension correctly."""

    def test_small_file_hash(self, tmp_path):
        import xxhash
        from iPhoto.utils.hashutils import compute_file_id

        data = b"test content" * 200
        p = tmp_path / "test.bin"
        p.write_bytes(data)

        result = compute_file_id(p)
        expected = xxhash.xxh3_128(data).hexdigest()
        assert result == expected

    def test_hash_is_deterministic(self, tmp_path):
        from iPhoto.utils.hashutils import compute_file_id

        p = tmp_path / "det.bin"
        p.write_bytes(b"deterministic content")

        assert compute_file_id(p) == compute_file_id(p)


# ---------------------------------------------------------------------------
# Integration: pathutils uses C-accelerated glob filtering
# ---------------------------------------------------------------------------

class TestPathUtilsWithCExtension:
    """Verify should_include / is_excluded integrate C extension correctly."""

    def test_should_include_jpg(self, tmp_path):
        from iPhoto.utils.pathutils import should_include
        p = tmp_path / "photo.jpg"
        p.touch()
        assert should_include(p, ["*.jpg"], [], root=tmp_path) is True

    def test_should_exclude_overrides_include(self, tmp_path):
        from iPhoto.utils.pathutils import should_include
        p = tmp_path / "photo.jpg"
        p.touch()
        assert should_include(p, ["*.jpg"], ["photo.jpg"], root=tmp_path) is False

    def test_is_excluded_recursive(self, tmp_path):
        from iPhoto.utils.pathutils import is_excluded
        sub = tmp_path / "sub"
        sub.mkdir()
        p = sub / "photo.jpg"
        p.touch()
        assert is_excluded(p, ["**/*.jpg"], root=tmp_path) is True
