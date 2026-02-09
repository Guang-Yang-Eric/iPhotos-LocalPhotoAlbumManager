"""Tests for Adobe-compatible curve export in XMP sidecar files."""

import xml.etree.ElementTree as ET

import pytest

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

    def test_rgb_curve_exported_as_tone_curve_pv2012(self):
        """RGB curve control points should appear in crs:ToneCurvePV2012."""
        adj = {
            "Curve_Enabled": True,
            "Curve_RGB": [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
        }
        xmp = ipo_to_xmp(adj)
        pairs = self._parse_tone_curve(xmp, "ToneCurvePV2012")
        assert pairs == [(0, 0), (128, 178), (255, 255)]

    def test_per_channel_curves_exported(self):
        """R/G/B channel curves should appear in ToneCurvePV2012Red/Green/Blue."""
        adj = {
            "Curve_Enabled": True,
            "Curve_Red": [(0.0, 0.0), (0.25, 0.35), (1.0, 1.0)],
            "Curve_Green": [(0.0, 0.1), (1.0, 0.9)],
            "Curve_Blue": [(0.0, 0.0), (1.0, 1.0)],
        }
        xmp = ipo_to_xmp(adj)

        red = self._parse_tone_curve(xmp, "ToneCurvePV2012Red")
        assert red == [(0, 0), (64, 89), (255, 255)]

        green = self._parse_tone_curve(xmp, "ToneCurvePV2012Green")
        assert green == [(0, 26), (255, 230)]

        blue = self._parse_tone_curve(xmp, "ToneCurvePV2012Blue")
        assert blue == [(0, 0), (255, 255)]

    def test_identity_curves_when_no_channel_data(self):
        """Missing channel data should produce identity curves [(0,0), (255,255)]."""
        adj = {"Curve_Enabled": True}
        xmp = ipo_to_xmp(adj)
        for tag in [
            "ToneCurvePV2012",
            "ToneCurvePV2012Red",
            "ToneCurvePV2012Green",
            "ToneCurvePV2012Blue",
        ]:
            pairs = self._parse_tone_curve(xmp, tag)
            assert pairs == [(0, 0), (255, 255)], f"{tag} is not identity"

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
        """Control points survive IPO → XMP → IPO roundtrip."""
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
        # Allow small rounding from 0-1 → 0-255 → 0-1
        for key in ("Curve_RGB", "Curve_Red", "Curve_Green", "Curve_Blue"):
            orig_pts = original[key]
            load_pts = loaded[key]
            assert len(load_pts) == len(orig_pts), f"{key} point count mismatch"
            for (ox, oy), (lx, ly) in zip(orig_pts, load_pts):
                assert abs(ox - lx) < 0.005, f"{key} x mismatch: {ox} vs {lx}"
                assert abs(oy - ly) < 0.005, f"{key} y mismatch: {oy} vs {ly}"

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
