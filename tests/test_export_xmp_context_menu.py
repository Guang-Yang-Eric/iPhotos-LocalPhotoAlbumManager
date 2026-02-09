"""Tests for the Export XMP context menu action and _texture_size fix."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import pytest


class TestExportXmpContextMenu:
    """Verify the Export XMP action in the context menu for RAW files."""

    def test_export_xmp_writes_file(self, tmp_path: Path):
        """export_xmp should create a .xmp sidecar next to the RAW file."""
        from src.iPhoto.io.xmp_sidecar import export_xmp

        raw_file = tmp_path / "photo.cr2"
        raw_file.write_bytes(b"\x00" * 64)

        adjustments = {"Light_Exposure": 0.5, "Light_Contrast": 0.2}
        result = export_xmp(raw_file, adjustments)

        assert result.exists()
        assert result.suffix == ".xmp"
        assert result.stem == "photo"
        content = result.read_text(encoding="utf-8")
        assert "xmpmeta" in content

    def test_export_xmp_roundtrip(self, tmp_path: Path):
        """IPO adjustments exported to XMP should be re-importable."""
        from src.iPhoto.io.xmp_sidecar import export_xmp, load_xmp_adjustments

        raw_file = tmp_path / "test.dng"
        raw_file.write_bytes(b"\x00" * 64)

        # IPO format uses keys without "Light_" prefix for adjustment values
        original = {"Exposure": 1.5, "Contrast": -0.3}
        export_xmp(raw_file, original)

        loaded = load_xmp_adjustments(raw_file)
        assert abs(loaded.get("Exposure", 0.0) - 1.5) < 0.01
        assert abs(loaded.get("Contrast", 0.0) - (-0.3)) < 0.01


class TestTextureSizeFix:
    """Verify the _texture_size -> _texture_dimensions rename in widget.py."""

    def test_no_texture_size_reference_in_paint_gl(self):
        """The paintGL method should not reference _texture_size."""
        widget_path = Path(
            "/home/runner/work/iPhotos-LocalPhotoAlbumManager/"
            "iPhotos-LocalPhotoAlbumManager/src/iPhoto/gui/ui/widgets/"
            "gl_image_viewer/widget.py"
        )
        content = widget_path.read_text(encoding="utf-8")
        # _texture_size() should NOT appear in the file (it was a typo)
        assert "_texture_size()" not in content, (
            "_texture_size() still referenced in widget.py; "
            "should be _texture_dimensions()"
        )

    def test_texture_dimensions_method_exists(self):
        """GLImageViewer should have a _texture_dimensions method."""
        widget_path = Path(
            "/home/runner/work/iPhotos-LocalPhotoAlbumManager/"
            "iPhotos-LocalPhotoAlbumManager/src/iPhoto/gui/ui/widgets/"
            "gl_image_viewer/widget.py"
        )
        content = widget_path.read_text(encoding="utf-8")
        assert "def _texture_dimensions(" in content
