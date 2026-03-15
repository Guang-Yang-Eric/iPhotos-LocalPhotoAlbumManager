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


# ---------------------------------------------------------------------------
# P4 — discover_files_fast (nftw-based recursive file discovery)
# ---------------------------------------------------------------------------

class TestDiscoverFilesFast:
    """Tests for discover_files_fast — C nftw-based file discovery."""

    @needs_native
    def test_finds_image_files(self, tmp_path):
        from iPhoto._native import discover_files_fast

        (tmp_path / "photo.jpg").touch()
        (tmp_path / "photo.heic").touch()
        (tmp_path / "document.pdf").touch()  # not supported

        result = discover_files_fast(tmp_path)
        assert result is not None
        names = {p.name for p in result}
        assert "photo.jpg" in names
        assert "photo.heic" in names
        assert "document.pdf" not in names

    @needs_native
    def test_finds_video_files(self, tmp_path):
        from iPhoto._native import discover_files_fast

        (tmp_path / "clip.mov").touch()
        (tmp_path / "movie.mp4").touch()

        result = discover_files_fast(tmp_path)
        assert result is not None
        names = {p.name for p in result}
        assert "clip.mov" in names
        assert "movie.mp4" in names

    @needs_native
    def test_recurses_into_subdirectories(self, tmp_path):
        from iPhoto._native import discover_files_fast

        sub = tmp_path / "2024" / "january"
        sub.mkdir(parents=True)
        (sub / "deep.png").touch()

        result = discover_files_fast(tmp_path)
        assert result is not None
        found_names = {p.name for p in result}
        assert "deep.png" in found_names

    @needs_native
    def test_skips_hidden_directories(self, tmp_path):
        from iPhoto._native import discover_files_fast

        hidden = tmp_path / ".iphoto"
        hidden.mkdir()
        (hidden / "hidden.jpg").touch()
        (tmp_path / "visible.jpg").touch()

        result = discover_files_fast(tmp_path)
        assert result is not None
        names = {p.name for p in result}
        assert "visible.jpg" in names
        assert "hidden.jpg" not in names

    @needs_native
    def test_returns_list_of_paths(self, tmp_path):
        from pathlib import Path
        from iPhoto._native import discover_files_fast

        (tmp_path / "test.jpg").touch()
        result = discover_files_fast(tmp_path)
        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)

    @needs_native
    def test_empty_directory_returns_empty_list(self, tmp_path):
        from iPhoto._native import discover_files_fast

        result = discover_files_fast(tmp_path)
        assert result is not None
        assert result == []

    def test_fallback_returns_none_without_c(self):
        """When C extension is unavailable, discover_files_fast returns None."""
        from iPhoto._native import _C_AVAILABLE, discover_files_fast
        from pathlib import Path

        # If C is available, the function returns a list; otherwise None.
        # Both cases are valid; this test ensures the function is callable.
        result = discover_files_fast(Path("."))
        if _C_AVAILABLE:
            assert result is not None  # list (may be empty)
        else:
            assert result is None


# ---------------------------------------------------------------------------
# P5 — parse_dt_full_fast (ISO 8601 → unix_us + year + month in one call)
# ---------------------------------------------------------------------------

class TestParseDtFullFast:
    """Tests for parse_dt_full_fast — C-accelerated full datetime parsing."""

    @needs_native
    def test_utc_basic(self):
        from iPhoto._native import parse_dt_full_fast
        from dateutil import parser as dp

        value = "2024-03-15T10:30:00Z"
        result = parse_dt_full_fast(value)
        assert result is not None
        unix_us, year, month = result
        assert year == 2024
        assert month == 3
        expected_us = int(dp.isoparse(value).timestamp() * 1_000_000)
        assert unix_us == expected_us

    @needs_native
    def test_year_month_extraction(self):
        from iPhoto._native import parse_dt_full_fast

        result = parse_dt_full_fast("2019-07-04T00:00:00Z")
        assert result is not None
        _, year, month = result
        assert year == 2019
        assert month == 7

    @needs_native
    def test_positive_offset(self):
        from iPhoto._native import parse_dt_full_fast
        from dateutil import parser as dp

        value = "2023-12-25T08:00:00+08:00"
        result = parse_dt_full_fast(value)
        assert result is not None
        unix_us, year, month = result
        assert year == 2023
        assert month == 12
        expected_us = int(dp.isoparse(value).timestamp() * 1_000_000)
        assert unix_us == expected_us

    @needs_native
    def test_none_input_returns_none(self):
        from iPhoto._native import parse_dt_full_fast
        assert parse_dt_full_fast(None) is None

    @needs_native
    def test_empty_string_returns_none(self):
        from iPhoto._native import parse_dt_full_fast
        assert parse_dt_full_fast("") is None

    @needs_native
    def test_invalid_string_returns_none(self):
        from iPhoto._native import parse_dt_full_fast
        assert parse_dt_full_fast("not-a-date") is None

    @needs_native
    def test_returns_tuple_of_three(self):
        from iPhoto._native import parse_dt_full_fast
        result = parse_dt_full_fast("2024-01-01T00:00:00Z")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_fallback_available(self):
        """parse_dt_full_fast must be callable even without C extension."""
        from iPhoto._native import parse_dt_full_fast
        assert callable(parse_dt_full_fast)
        result = parse_dt_full_fast("2024-03-15T10:30:00Z")
        assert result is not None
        unix_us, year, month = result
        assert year == 2024
        assert month == 3

    @needs_native
    def test_unix_us_matches_parse_dt_fast(self):
        """parse_dt_full_fast's unix_us must equal parse_dt_fast for the same input."""
        from iPhoto._native import parse_dt_fast, parse_dt_full_fast

        value = "2024-06-15T12:34:56.789012Z"
        fast_us = parse_dt_fast(value)
        full_result = parse_dt_full_fast(value)
        assert full_result is not None
        assert full_result[0] == fast_us


# ---------------------------------------------------------------------------
# P6 — normalise_content_id_fast (strip + casefold for Live Photo IDs)
# ---------------------------------------------------------------------------

class TestNormaliseContentIdFast:
    """Tests for normalise_content_id_fast — C-accelerated content-ID normalisation."""

    @needs_native
    def test_basic_uuid(self):
        from iPhoto._native import normalise_content_id_fast

        cid = "ABCDEF12-3456-7890-ABCD-EF1234567890"
        result = normalise_content_id_fast(cid)
        assert result == cid.lower()

    @needs_native
    def test_leading_trailing_whitespace_stripped(self):
        from iPhoto._native import normalise_content_id_fast

        result = normalise_content_id_fast("  ABCD1234  ")
        assert result == "abcd1234"

    @needs_native
    def test_already_lower(self):
        from iPhoto._native import normalise_content_id_fast

        result = normalise_content_id_fast("abcd1234")
        assert result == "abcd1234"

    @needs_native
    def test_empty_string_returns_none(self):
        from iPhoto._native import normalise_content_id_fast

        assert normalise_content_id_fast("") is None

    @needs_native
    def test_whitespace_only_returns_none(self):
        from iPhoto._native import normalise_content_id_fast

        assert normalise_content_id_fast("   ") is None

    @needs_native
    def test_non_string_returns_none(self):
        from iPhoto._native import normalise_content_id_fast

        assert normalise_content_id_fast(None) is None
        assert normalise_content_id_fast(123) is None

    @needs_native
    def test_matches_python_fallback(self):
        """C result must equal Python strip().casefold()."""
        from iPhoto._native import normalise_content_id_fast

        samples = [
            "  UUID-1234-ABCD  ",
            "ABC",
            "abc",
            "  ",
            "Mixed-Case-123",
        ]
        for s in samples:
            c_result = normalise_content_id_fast(s)
            py_result = s.strip().casefold() or None
            assert c_result == py_result, f"mismatch for {s!r}: {c_result!r} != {py_result!r}"

    def test_fallback_available(self):
        """normalise_content_id_fast must be callable even without C extension."""
        from iPhoto._native import normalise_content_id_fast
        assert callable(normalise_content_id_fast)
        result = normalise_content_id_fast("  TEST-UUID  ")
        assert result == "test-uuid"


# ---------------------------------------------------------------------------
# Integration: pairing uses C-accelerated content-ID normalisation (P6)
# ---------------------------------------------------------------------------

class TestPairingWithP6:
    """Verify that pair_live uses the C content-ID normalisation correctly."""

    def test_content_id_match_case_insensitive(self):
        """Pairing must work even when photo and video content IDs differ in case."""
        from iPhoto.core.pairing import pair_live

        rows = [
            {
                "rel": "IMG_0001.HEIC",
                "mime": "image/heic",
                "content_id": "ABCD-1234",
            },
            {
                "rel": "IMG_0001.MOV",
                "mime": "video/quicktime",
                "content_id": "abcd-1234",
                "dur": 2.0,
            },
        ]
        groups = pair_live(rows)
        assert len(groups) == 1
        assert groups[0].still == "IMG_0001.HEIC"
        assert groups[0].motion == "IMG_0001.MOV"

    def test_content_id_match_with_whitespace(self):
        """Pairing must strip whitespace from content IDs before matching."""
        from iPhoto.core.pairing import pair_live

        rows = [
            {
                "rel": "IMG_0002.HEIC",
                "mime": "image/heic",
                "content_id": "  UUID-5678  ",
            },
            {
                "rel": "IMG_0002.MOV",
                "mime": "video/quicktime",
                "content_id": "UUID-5678",
                "dur": 2.0,
            },
        ]
        groups = pair_live(rows)
        assert len(groups) == 1


# ---------------------------------------------------------------------------
# Integration: parallel_scanner uses C-accelerated file discovery (P4)
# ---------------------------------------------------------------------------

class TestParallelScannerWithP4:
    """Verify that ParallelScanner uses the C file discovery (P4) correctly."""

    def test_scan_finds_images(self, tmp_path):
        from iPhoto.application.services.parallel_scanner import ParallelScanner

        (tmp_path / "photo.jpg").touch()
        (tmp_path / "photo.heic").touch()
        (tmp_path / "doc.pdf").touch()

        scanner = ParallelScanner()
        # Patch _scan_file_fn to return a dummy Asset-like sentinel
        found_paths = []

        def fake_scan(path):
            found_paths.append(path)
            return None  # no actual Asset needed

        scanner._scan_file_fn = fake_scan
        scanner.scan(tmp_path)

        names = {p.name for p in found_paths}
        assert "photo.jpg" in names
        assert "photo.heic" in names
        assert "doc.pdf" not in names

    def test_scan_skips_hidden_directories(self, tmp_path):
        from iPhoto.application.services.parallel_scanner import ParallelScanner

        hidden = tmp_path / ".iphoto"
        hidden.mkdir()
        (hidden / "hidden.jpg").touch()
        (tmp_path / "visible.jpg").touch()

        found_paths = []

        def fake_scan(path):
            found_paths.append(path)
            return None

        scanner = ParallelScanner()
        scanner._scan_file_fn = fake_scan
        scanner.scan(tmp_path)

        names = {p.name for p in found_paths}
        assert "visible.jpg" in names
        assert "hidden.jpg" not in names
