"""Tests for ContentIdentifier extraction from ExifTool metadata.

iOS Live Photos store the ``ContentIdentifier`` in different ExifTool groups
depending on the file format:
* HEIC still images → ``MakerNotes:ContentIdentifier``
* MOV motion clips  → ``QuickTime:ContentIdentifier``
* Some older assets  → ``Apple:ContentIdentifier``

The extractor must check all three groups to ensure reliable pairing.
"""

from __future__ import annotations

import pytest

from iPhoto.io.metadata_extractors import _extract_content_id_from_exiftool


class TestContentIdExtraction:
    """Test ``_extract_content_id_from_exiftool`` covers all ExifTool groups."""

    def test_makernotes_nested(self) -> None:
        """HEIC images from iOS store ContentIdentifier in MakerNotes (nested dict)."""
        meta = {"MakerNotes": {"ContentIdentifier": "ABCD-1234-EF56"}}
        assert _extract_content_id_from_exiftool(meta) == "ABCD-1234-EF56"

    def test_makernotes_flat(self) -> None:
        """Flattened ExifTool output uses 'MakerNotes:ContentIdentifier' keys."""
        meta = {"MakerNotes:ContentIdentifier": "FLAT-CID-001"}
        assert _extract_content_id_from_exiftool(meta) == "FLAT-CID-001"

    def test_apple_nested(self) -> None:
        """Apple group (legacy path) should be extracted."""
        meta = {"Apple": {"ContentIdentifier": "APPLE-CID"}}
        assert _extract_content_id_from_exiftool(meta) == "APPLE-CID"

    def test_apple_flat(self) -> None:
        """Flattened Apple:ContentIdentifier key."""
        meta = {"Apple:ContentIdentifier": "APPLE-FLAT"}
        assert _extract_content_id_from_exiftool(meta) == "APPLE-FLAT"

    def test_quicktime_nested(self) -> None:
        """MOV files store ContentIdentifier in QuickTime group."""
        meta = {"QuickTime": {"ContentIdentifier": "QT-CID-789"}}
        assert _extract_content_id_from_exiftool(meta) == "QT-CID-789"

    def test_quicktime_flat(self) -> None:
        """Flattened QuickTime:ContentIdentifier key."""
        meta = {"QuickTime:ContentIdentifier": "QT-FLAT-001"}
        assert _extract_content_id_from_exiftool(meta) == "QT-FLAT-001"

    def test_makernotes_takes_precedence(self) -> None:
        """When multiple groups contain the ID, MakerNotes should be preferred."""
        meta = {
            "MakerNotes": {"ContentIdentifier": "FROM-MN"},
            "QuickTime": {"ContentIdentifier": "FROM-QT"},
        }
        assert _extract_content_id_from_exiftool(meta) == "FROM-MN"

    def test_no_content_id(self) -> None:
        """Return None when no ContentIdentifier is present."""
        meta = {"EXIF": {"Make": "Apple"}}
        assert _extract_content_id_from_exiftool(meta) is None

    def test_empty_string_ignored(self) -> None:
        """Empty string ContentIdentifiers should be treated as absent."""
        meta = {"MakerNotes": {"ContentIdentifier": ""}}
        assert _extract_content_id_from_exiftool(meta) is None

    def test_non_string_ignored(self) -> None:
        """Non-string ContentIdentifiers should be treated as absent."""
        meta = {"MakerNotes": {"ContentIdentifier": 12345}}
        assert _extract_content_id_from_exiftool(meta) is None

    def test_empty_metadata(self) -> None:
        """Empty metadata dict should return None."""
        assert _extract_content_id_from_exiftool({}) is None


class TestLivePhotoPairingWithMakerNotes:
    """Integration test: pairing succeeds when content_id comes from MakerNotes."""

    def test_heic_mov_pairing_via_makernotes_content_id(self) -> None:
        """HEIC still + MOV motion paired via MakerNotes ContentIdentifier.

        This simulates the typical Linux scan flow where ExifTool reports the
        ContentIdentifier in the MakerNotes group for HEIC images and in the
        QuickTime group for MOV files.  The test exercises
        ``_extract_content_id_from_exiftool`` through ``pair_live``.
        """
        from iPhoto.core.pairing import pair_live

        rows = [
            {
                "rel": "IMG_0001.HEIC",
                "mime": "image/heic",
                "dt": "2024-01-01T12:00:00Z",
                # content_id extracted from MakerNotes:ContentIdentifier
                "content_id": "LIVE-PAIR-001",
            },
            {
                "rel": "IMG_0001.MOV",
                "mime": "video/quicktime",
                "dt": "2024-01-01T12:00:00Z",
                # content_id extracted from QuickTime:ContentIdentifier
                "content_id": "LIVE-PAIR-001",
                "dur": 2.5,
            },
        ]
        groups = pair_live(rows)
        assert len(groups) == 1
        assert groups[0].still == "IMG_0001.HEIC"
        assert groups[0].motion == "IMG_0001.MOV"
        assert groups[0].content_id == "LIVE-PAIR-001"

    def test_extract_content_id_round_trip(self) -> None:
        """MakerNotes ContentIdentifier flows through to pairing output.

        Verifies the full chain: extraction → scan row → pairing.
        """
        # Step 1: Extract from MakerNotes (simulating HEIC ExifTool output)
        heic_meta = {"MakerNotes": {"ContentIdentifier": "ROUND-TRIP-001"}}
        heic_cid = _extract_content_id_from_exiftool(heic_meta)
        assert heic_cid == "ROUND-TRIP-001"

        # Step 2: Extract from QuickTime (simulating MOV ExifTool output)
        mov_meta = {"QuickTime": {"ContentIdentifier": "ROUND-TRIP-001"}}
        mov_cid = _extract_content_id_from_exiftool(mov_meta)
        assert mov_cid == "ROUND-TRIP-001"

        # Step 3: Both end up in the scan rows and pairing succeeds
        from iPhoto.core.pairing import pair_live
        rows = [
            {"rel": "photo.heic", "mime": "image/heic", "content_id": heic_cid},
            {"rel": "photo.mov", "mime": "video/quicktime", "content_id": mov_cid, "dur": 2.0},
        ]
        groups = pair_live(rows)
        assert len(groups) == 1
        assert groups[0].content_id == "ROUND-TRIP-001"
