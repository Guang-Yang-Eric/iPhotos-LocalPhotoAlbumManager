"""Tests for Adobe-compatible curve export in XMP sidecar files."""

import xml.etree.ElementTree as ET

from src.iPhoto.io.xmp_sidecar import ipo_to_xmp, xmp_to_ipo

_NS_CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
_NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


class TestAdobeCurveExport:
    """Verify that curve adjustments produce Adobe Camera Raw compatible XMP."""

    def _parse_tone_curve(self, xmp: str, tag: str) -> list[tuple[int, int]]:
        """Extract (x, y) integer pairs from a crs:ToneCurve* element."""
        root = ET.fromstring(xmp)
        curve_el = root.find(f".//{{{_NS_CRS}}}{tag}")
        assert curve_el is not None, f"crs:{tag} element not found"
        seq = curve_el.find(f"{{{_NS_RDF}}}Seq")
        assert seq is not None, f"rdf:Seq not found inside crs:{tag}"
        pairs = []
        for li in seq.findall(f"{{{_NS_RDF}}}li"):
            parts = li.text.split(",")
            pairs.append((int(parts[0].strip()), int(parts[1].strip())))
        return pairs

    def test_baked_lut_produces_dense_per_channel_curves(self):
        """When curves are enabled, per-channel ToneCurvePV2012 should have 256 points."""
        adj = {
            "Curve_Enabled": True,
            "Curve_RGB": [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
        }
        xmp = ipo_to_xmp(adj)
        # Master curve should be identity (2 points)
        master = self._parse_tone_curve(xmp, "ToneCurvePV2012")
        assert master == [(0, 0), (255, 255)]
        # Per-channel curves should be densely sampled (256 points)
        for tag in ("ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"):
            pairs = self._parse_tone_curve(xmp, tag)
            assert len(pairs) == 256, f"{tag} should have 256 points, got {len(pairs)}"
            # Inputs should be 0..255
            assert pairs[0][0] == 0
            assert pairs[255][0] == 255

    def test_dense_curve_captures_non_identity(self):
        """A non-identity RGB curve should produce non-identity per-channel output."""
        adj = {
            "Curve_Enabled": True,
            "Curve_RGB": [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
        }
        xmp = ipo_to_xmp(adj)
        red = self._parse_tone_curve(xmp, "ToneCurvePV2012Red")
        # Mid-point should be elevated due to the curve lifting 0.5 → ~0.7
        mid_out = red[128][1]
        assert mid_out > 140, f"Mid-point output should be >140, got {mid_out}"
        assert mid_out < 200, f"Mid-point output should be <200, got {mid_out}"

    def test_identity_curves_produce_256_point_identity(self):
        """When Curve_Enabled=True but no curve data, per-channel should be identity LUT."""
        adj = {"Curve_Enabled": True}
        xmp = ipo_to_xmp(adj)
        for tag in ("ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"):
            pairs = self._parse_tone_curve(xmp, tag)
            assert len(pairs) == 256
            # All points should be identity: (i, i)
            for i, (x, y) in enumerate(pairs):
                assert x == i, f"{tag}[{i}] x mismatch"
                assert y == i, f"{tag}[{i}] y mismatch: expected {i}, got {y}"

    def test_no_tone_curve_when_disabled(self):
        """When Curve_Enabled is False, no ToneCurvePV2012 elements should appear."""
        adj = {
            "Curve_Enabled": False,
            "Curve_RGB": [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
        }
        xmp = ipo_to_xmp(adj)
        root = ET.fromstring(xmp)
        assert root.find(f".//{{{_NS_CRS}}}ToneCurvePV2012") is None

    def test_curve_roundtrip_via_xmp(self):
        """Control points survive IPO → XMP → IPO roundtrip via ipo: namespace."""
        original = {
            "Curve_Enabled": True,
            "Curve_RGB": [(0.0, 0.0), (0.25, 0.35), (0.75, 0.65), (1.0, 1.0)],
            "Curve_Red": [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)],
            "Curve_Green": [(0.0, 0.0), (1.0, 1.0)],
            "Curve_Blue": [(0.0, 0.0), (0.5, 0.4), (1.0, 1.0)],
        }
        xmp = ipo_to_xmp(original)
        loaded = xmp_to_ipo(xmp)

        assert loaded["Curve_Enabled"] is True
        for key in ("Curve_RGB", "Curve_Red", "Curve_Green", "Curve_Blue"):
            orig_pts = original[key]
            load_pts = loaded[key]
            assert len(load_pts) == len(orig_pts), f"{key} point count mismatch"
            for (ox, oy), (lx, ly) in zip(orig_pts, load_pts):
                assert abs(ox - lx) < 1e-6, f"{key} x mismatch: {ox} vs {lx}"
                assert abs(oy - ly) < 1e-6, f"{key} y mismatch: {oy} vs {ly}"

    def test_adobe_xmp_parsed_without_ipo_lut(self):
        """An XMP file written by Adobe (no ipo:BakedLUT) should parse curves."""
        # Simulate what Adobe Camera Raw would write
        xmp = """<?xml version='1.0' encoding='utf-8'?>
<x:xmpmeta xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
           xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
           xmlns:x="adobe:ns:meta/">
  <rdf:RDF>
    <rdf:Description crs:Exposure2012="+1.00">
      <crs:ToneCurvePV2012>
        <rdf:Seq>
          <rdf:li>0, 0</rdf:li>
          <rdf:li>64, 90</rdf:li>
          <rdf:li>192, 166</rdf:li>
          <rdf:li>255, 255</rdf:li>
        </rdf:Seq>
      </crs:ToneCurvePV2012>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""
        loaded = xmp_to_ipo(xmp)
        assert loaded["Curve_Enabled"] is True
        pts = loaded["Curve_RGB"]
        assert len(pts) == 4
        assert abs(pts[0][0] - 0.0) < 0.01
        assert abs(pts[1][0] - 64 / 255) < 0.01
        assert abs(pts[1][1] - 90 / 255) < 0.01
        assert abs(pts[2][0] - 192 / 255) < 0.01

    def test_light_params_baked_into_curve(self):
        """Light adjustments (exposure, contrast, etc.) should be baked into LUT."""
        adj = {
            "Exposure": 0.5,
            "Contrast": 0.2,
            "Curve_Enabled": True,
            "Curve_RGB": [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
        }
        xmp = ipo_to_xmp(adj)
        # CRS light params should be zeroed
        root = ET.fromstring(xmp)
        desc = root.find(f".//{{{_NS_RDF}}}Description")
        assert desc.get(f"{{{_NS_CRS}}}Exposure2012") == "+0.00"
        assert desc.get(f"{{{_NS_CRS}}}Contrast2012") == "+0.00"
        # But per-channel curves should exist and be non-identity
        red = self._parse_tone_curve(xmp, "ToneCurvePV2012Red")
        assert len(red) == 256
        # With exposure=0.5, input 0 should map to something > 0
        assert red[0][1] > 0, "Exposure should lift blacks"

    def test_all_params_roundtrip_losslessly(self):
        """All IPO parameters survive round-trip via ipo: namespace."""
        original = {
            "Exposure": 0.5,
            "Brightness": 0.3,
            "Brilliance": 0.2,
            "WB_Enabled": True,
            "WB_Warmth": 0.15,
            "WB_Temperature": 0.3,
            "WB_Tint": -0.1,
            "BW_Enabled": True,
            "BW_Intensity": 0.7,
            "BW_Neutrals": 0.4,
            "BW_Tone": 0.3,
            "BW_Grain": 0.1,
        }
        xmp = ipo_to_xmp(original)
        loaded = xmp_to_ipo(xmp)
        for key, value in original.items():
            if isinstance(value, float):
                assert abs(loaded[key] - value) < 1e-6, f"{key} mismatch"
            elif isinstance(value, bool):
                assert loaded[key] == value, f"{key} mismatch"
