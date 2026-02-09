"""Tests for XMP sidecar read/write and IPO↔XMP conversion."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from src.iPhoto.io.xmp_sidecar import (
    export_xmp,
    ipo_to_xmp,
    load_xmp_adjustments,
    xmp_sidecar_path_for_asset,
    xmp_to_ipo,
    _encode_lut_base64,
    _decode_lut_base64,
)

_NS_CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
_NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


class TestXmpSidecarPath:
    def test_xmp_path_for_jpg(self) -> None:
        assert xmp_sidecar_path_for_asset(Path("/a/b.jpg")) == Path("/a/b.xmp")

    def test_xmp_path_for_raw(self) -> None:
        assert xmp_sidecar_path_for_asset(Path("/a/b.cr2")) == Path("/a/b.xmp")


class TestLutEncoding:
    def test_roundtrip(self) -> None:
        lut = np.random.rand(256, 3).astype(np.float32)
        encoded = _encode_lut_base64(lut)
        decoded = _decode_lut_base64(encoded)
        assert decoded is not None
        np.testing.assert_array_almost_equal(lut, decoded, decimal=6)

    def test_decode_bad_data(self) -> None:
        assert _decode_lut_base64("not-valid-base64!!!") is None

    def test_decode_wrong_size(self) -> None:
        import base64
        data = base64.b64encode(b"\x00" * 100).decode()
        assert _decode_lut_base64(data) is None


class TestIpoToXmpConversion:
    def test_light_params_mapped(self) -> None:
        adj = {"Exposure": 0.5, "Contrast": -0.3}
        xmp = ipo_to_xmp(adj)
        parsed = xmp_to_ipo(xmp)
        assert pytest.approx(0.5, abs=0.02) == parsed["Exposure"]
        assert pytest.approx(-0.3, abs=0.02) == parsed["Contrast"]

    def test_wb_custom_roundtrip(self) -> None:
        adj = {
            "WB_Enabled": True,
            "WB_Temperature": 0.5,
            "WB_Tint": -0.3,
        }
        xmp = ipo_to_xmp(adj)
        parsed = xmp_to_ipo(xmp)
        assert parsed.get("WB_Enabled") is True
        assert pytest.approx(0.5, abs=0.02) == parsed["WB_Temperature"]
        assert pytest.approx(-0.3, abs=0.02) == parsed["WB_Tint"]

    def test_crop_roundtrip(self) -> None:
        adj = {
            "Crop_CX": 0.6,
            "Crop_CY": 0.4,
            "Crop_W": 0.5,
            "Crop_H": 0.3,
            "Crop_Straighten": 2.5,
        }
        xmp = ipo_to_xmp(adj)
        parsed = xmp_to_ipo(xmp)
        assert pytest.approx(0.6, abs=0.01) == parsed["Crop_CX"]
        assert pytest.approx(0.4, abs=0.01) == parsed["Crop_CY"]
        assert pytest.approx(0.5, abs=0.01) == parsed["Crop_W"]
        assert pytest.approx(0.3, abs=0.01) == parsed["Crop_H"]
        assert pytest.approx(2.5, abs=0.01) == parsed["Crop_Straighten"]

    def test_bw_flag(self) -> None:
        adj = {"BW_Enabled": True}
        xmp = ipo_to_xmp(adj)
        parsed = xmp_to_ipo(xmp)
        assert parsed.get("BW_Enabled") is True

    def test_selective_color_roundtrip(self) -> None:
        ranges = [
            [0.0, 0.083, 0.1, 0.2, 0.3],
            [0.166, 0.083, 0.0, 0.0, 0.0],
            [0.333, 0.083, 0.0, 0.0, 0.0],
            [0.5, 0.083, 0.0, 0.0, 0.0],
            [0.666, 0.083, 0.0, 0.0, 0.0],
            [0.833, 0.083, 0.0, 0.0, 0.0],
        ]
        adj = {
            "SelectiveColor_Enabled": True,
            "SelectiveColor_Ranges": ranges,
        }
        xmp = ipo_to_xmp(adj)
        parsed = xmp_to_ipo(xmp)
        assert parsed.get("SelectiveColor_Enabled") is True
        assert len(parsed["SelectiveColor_Ranges"]) == 6
        assert pytest.approx(0.1, abs=0.001) == parsed["SelectiveColor_Ranges"][0][2]

    def test_color_master_resolves_into_crs(self) -> None:
        adj = {"Color_Master": 1.0}
        xmp = ipo_to_xmp(adj)
        root = ET.fromstring(xmp)
        desc = root.find(f".//{{{_NS_RDF}}}Description")
        assert desc is not None
        sat = float(desc.get(f"{{{_NS_CRS}}}Saturation", "0"))
        vib = float(desc.get(f"{{{_NS_CRS}}}Vibrance", "0"))
        assert abs(sat) > 0.01 or abs(vib) > 0.01


class TestXmpToIpoConversion:
    def test_empty_xmp(self) -> None:
        assert xmp_to_ipo("") == {}

    def test_invalid_xml(self) -> None:
        assert xmp_to_ipo("<not><valid>") == {}

    def test_baked_lut_priority(self) -> None:
        """When a baked LUT is present, it should be loaded."""
        lut = np.linspace(0, 1, 256 * 3, dtype=np.float32).reshape(256, 3)
        encoded = _encode_lut_base64(lut)
        xmp = f"""<?xml version='1.0'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/"
           xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
           xmlns:ipo="http://ns.iphoto.app/sidecar/1.0/">
  <rdf:RDF>
    <rdf:Description ipo:BakedLUT="{encoded}" ipo:BakedLUTSize="256"/>
  </rdf:RDF>
</x:xmpmeta>"""
        parsed = xmp_to_ipo(xmp)
        assert parsed.get("BakedLUT_Enabled") is True
        assert parsed["BakedLUT"] is not None
        np.testing.assert_array_almost_equal(lut, parsed["BakedLUT"], decimal=5)


class TestXmpFileOperations:
    def test_export_and_load_xmp(self, tmp_path: Path) -> None:
        asset = tmp_path / "photo.cr2"
        asset.touch()
        adj = {
            "Exposure": 0.3,
            "Contrast": -0.1,
            "Crop_CX": 0.5,
            "Crop_CY": 0.5,
            "Crop_W": 0.8,
            "Crop_H": 0.6,
        }
        xmp_path = export_xmp(asset, adj)
        assert xmp_path.exists()
        assert xmp_path.suffix == ".xmp"

        loaded = load_xmp_adjustments(asset)
        assert pytest.approx(0.3, abs=0.02) == loaded["Exposure"]
        assert pytest.approx(-0.1, abs=0.02) == loaded["Contrast"]

    def test_load_missing_xmp(self, tmp_path: Path) -> None:
        asset = tmp_path / "missing.cr2"
        asset.touch()
        assert load_xmp_adjustments(asset) == {}
