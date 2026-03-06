"""Tests for ExifTool-based video duration fallback parsing."""

import pytest

from iPhoto.io.metadata import _parse_exiftool_duration


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Seconds with unit suffix
        ("2.50 s", 2.5),
        ("12.5 s", 12.5),
        ("0.5s", 0.5),
        ("3 s", 3.0),
        # Bare number string
        ("2.50", 2.5),
        ("120", 120.0),
        # Time-code H:MM:SS
        ("0:02:30", 150.0),
        ("1:30:00", 5400.0),
        # Time-code MM:SS
        ("02:30", 150.0),
        ("0:05", 5.0),
        # Time-code with fractional seconds
        ("0:02:30.5", 150.5),
        # Numeric float
        (2.5, 2.5),
        (120, 120.0),
        # None / invalid
        (None, None),
        ("", None),
        ("   ", None),
        (0, None),
        (-5.0, None),
        # Invalid time-code ranges (minutes/seconds >= 60)
        ("99:99", None),
        ("0:60:00", None),
        ("0:00:60", None),
        # Multiple decimal points (should fall through to float fallback)
        ("1.2.3", None),
    ],
)
def test_parse_exiftool_duration(raw, expected):
    result = _parse_exiftool_duration(raw)
    if expected is None:
        assert result is None, f"Expected None for input {raw!r}, got {result}"
    else:
        assert result == pytest.approx(expected), f"Expected {expected} for input {raw!r}, got {result}"


def test_read_video_meta_exiftool_fallback():
    """ExifTool duration is used when FFprobe is unavailable."""
    from unittest.mock import patch
    from pathlib import Path
    from iPhoto.io.metadata import read_video_meta
    from iPhoto.errors import ExternalToolError

    # Simulate ExifTool metadata with QuickTime:Duration
    exiftool_meta = {
        "QuickTime:Duration": "3.50 s",
        "QuickTime:ImageWidth": 1920,
        "QuickTime:ImageHeight": 1080,
    }

    # Patch probe_media to fail (simulating missing FFprobe)
    with patch("iPhoto.io.metadata.probe_media", side_effect=ExternalToolError("ffprobe not found")):
        info = read_video_meta(Path("/fake/video.mov"), exiftool_meta)

    assert info["dur"] == pytest.approx(3.5)
    assert info["w"] == 1920
    assert info["h"] == 1080


def test_read_video_meta_ffprobe_overrides_exiftool():
    """FFprobe duration takes precedence when available."""
    from unittest.mock import patch
    from pathlib import Path
    from iPhoto.io.metadata import read_video_meta

    exiftool_meta = {
        "QuickTime:Duration": "3.50 s",
    }

    ffprobe_result = {
        "format": {"duration": "4.20"},
        "streams": [],
    }

    with patch("iPhoto.io.metadata.probe_media", return_value=ffprobe_result):
        info = read_video_meta(Path("/fake/video.mov"), exiftool_meta)

    # FFprobe duration should override ExifTool
    assert info["dur"] == pytest.approx(4.2)
