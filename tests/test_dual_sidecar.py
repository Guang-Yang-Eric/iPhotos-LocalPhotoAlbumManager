"""Tests for dual sidecar priority logic (IPO preferred over XMP)."""

from __future__ import annotations

import pytest
from pathlib import Path

from src.iPhoto.io.sidecar import load_adjustments, save_adjustments
from src.iPhoto.io.xmp_sidecar import export_xmp


class TestDualSidecarPriority:
    """When both .ipo and .xmp sidecars exist, .ipo takes precedence."""

    def test_ipo_preferred_over_xmp(self, tmp_path: Path) -> None:
        """When both sidecars exist, load_adjustments should read from .ipo."""
        asset = tmp_path / "photo.cr2"
        asset.touch()

        # Write .ipo with specific Exposure value
        ipo_adj = {"Exposure": 0.42, "Light_Enabled": True, "Light_Master": 0.0}
        save_adjustments(asset, ipo_adj)

        # Write .xmp with different Exposure value
        xmp_adj = {"Exposure": 0.88}
        export_xmp(asset, xmp_adj)

        # Both files should exist
        assert (tmp_path / "photo.ipo").exists()
        assert (tmp_path / "photo.xmp").exists()

        # load_adjustments should read from .ipo
        loaded = load_adjustments(asset)
        assert pytest.approx(0.42, abs=0.01) == loaded.get("Exposure", 0.0)

    def test_fallback_to_xmp_when_ipo_missing(self, tmp_path: Path) -> None:
        """When only .xmp exists, load_adjustments should read from it."""
        asset = tmp_path / "photo.nef"
        asset.touch()

        # Only write .xmp
        xmp_adj = {"Exposure": 0.65, "Contrast": -0.2}
        export_xmp(asset, xmp_adj)

        assert not (tmp_path / "photo.ipo").exists()
        assert (tmp_path / "photo.xmp").exists()

        loaded = load_adjustments(asset)
        assert pytest.approx(0.65, abs=0.02) == loaded.get("Exposure")
        assert pytest.approx(-0.2, abs=0.02) == loaded.get("Contrast")

    def test_empty_when_no_sidecar(self, tmp_path: Path) -> None:
        """When neither sidecar exists, return empty dict."""
        asset = tmp_path / "photo.arw"
        asset.touch()

        loaded = load_adjustments(asset)
        assert loaded == {}

    def test_save_always_writes_ipo(self, tmp_path: Path) -> None:
        """save_adjustments should always write a .ipo file."""
        asset = tmp_path / "photo.dng"
        asset.touch()

        adj = {"Exposure": 0.3, "Light_Enabled": True, "Light_Master": 0.0}
        result_path = save_adjustments(asset, adj)

        assert result_path.suffix == ".ipo"
        assert result_path.exists()
        # Ensure no .xmp was created
        assert not (tmp_path / "photo.xmp").exists()


class TestDualSidecarRawFiles:
    """Sidecar operations work correctly with various RAW extensions."""

    @pytest.mark.parametrize("ext", [".cr2", ".nef", ".arw", ".dng", ".raf"])
    def test_ipo_roundtrip_for_raw(self, tmp_path: Path, ext: str) -> None:
        asset = tmp_path / f"img{ext}"
        asset.touch()

        adj = {
            "Light_Enabled": True,
            "Light_Master": 0.1,
            "Exposure": 0.25,
            "Contrast": -0.15,
        }
        save_adjustments(asset, adj)
        loaded = load_adjustments(asset)

        assert pytest.approx(0.25, abs=0.01) == loaded.get("Exposure")
        assert pytest.approx(-0.15, abs=0.01) == loaded.get("Contrast")
