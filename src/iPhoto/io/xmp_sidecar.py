"""Read/write helpers for Adobe ``.xmp`` sidecar files and IPO↔XMP conversion.

This module provides:
- XMP sidecar path resolution
- Loading adjustments from XMP files
- Exporting adjustments to XMP format
- Bidirectional conversion between IPO internal format and XMP

The conversion strategy prioritises **result consistency** over process consistency.
For colour transforms (curves, levels) the module stores baked 256×3 LUTs as
Base64-encoded data in a custom ``ipo:`` XML namespace rather than attempting to
map control points, which would produce different visual results across software.
"""

from __future__ import annotations

import base64
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..core.curve_resolver import (
    CurveParams,
    CurveChannel,
    DEFAULT_CURVE_POINTS,
    generate_curve_lut,
)
from ..core.levels_resolver import DEFAULT_LEVELS_HANDLES, build_levels_lut
from ..core.light_resolver import LIGHT_KEYS
from ..core.color_resolver import COLOR_KEYS
from ..core.wb_resolver import WB_KEYS, WB_DEFAULTS
from ..core.selective_color_resolver import DEFAULT_SELECTIVE_COLOR_RANGES, NUM_RANGES

# ---------------------------------------------------------------------------
# XML namespaces
# ---------------------------------------------------------------------------

_NS_X = "adobe:ns:meta/"
_NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_NS_CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
_NS_IPO = "http://ns.iphoto.app/sidecar/1.0/"

_NSMAP = {
    "x": _NS_X,
    "rdf": _NS_RDF,
    "crs": _NS_CRS,
    "ipo": _NS_IPO,
}

# Register namespaces so ``ET.tostring`` uses readable prefixes.
for _prefix, _uri in _NSMAP.items():
    ET.register_namespace(_prefix, _uri)


# ---------------------------------------------------------------------------
# Parameter mapping: IPO ↔ XMP/CRS
# ---------------------------------------------------------------------------

# (ipo_key, crs_attr, ipo_to_xmp_factor, xmp_to_ipo_factor)
_LIGHT_PARAM_MAP: List[Tuple[str, str, float, float]] = [
    ("Exposure", "Exposure2012", 5.0, 0.2),
    ("Contrast", "Contrast2012", 100.0, 0.01),
    ("Highlights", "Highlights2012", 100.0, 0.01),
    ("Shadows", "Shadows2012", 100.0, 0.01),
    ("Brightness", "Brightness", 150.0, 1.0 / 150.0),
    ("BlackPoint", "Blacks2012", 100.0, 0.01),
]

_COLOR_PARAM_MAP: List[Tuple[str, str, float, float]] = [
    ("Saturation", "Saturation", 100.0, 0.01),
    ("Vibrance", "Vibrance", 100.0, 0.01),
]

# White-balance mapping: iPhoto [-1,1] → XMP colour-temperature domain
_WB_TEMP_CENTER = 5500.0
_WB_TEMP_RANGE = 4500.0
_WB_TINT_FACTOR = 150.0

# Curve channel mapping: (ipo_key, crs_tag_local_name)
_CURVE_CHANNEL_MAP: List[Tuple[str, str]] = [
    ("Curve_RGB", "ToneCurvePV2012"),
    ("Curve_Red", "ToneCurvePV2012Red"),
    ("Curve_Green", "ToneCurvePV2012Green"),
    ("Curve_Blue", "ToneCurvePV2012Blue"),
]


# ---------------------------------------------------------------------------
# LUT encoding helpers
# ---------------------------------------------------------------------------

def _encode_lut_base64(lut: np.ndarray) -> str:
    """Encode a ``(256, 3)`` float32 LUT as a Base64 string."""
    arr = np.ascontiguousarray(lut, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _decode_lut_base64(data: str) -> Optional[np.ndarray]:
    """Decode a Base64 string back to a ``(256, 3)`` float32 LUT."""
    try:
        raw = base64.b64decode(data)
        expected = 256 * 3 * 4  # float32 = 4 bytes
        if len(raw) != expected:
            return None
        arr = np.frombuffer(raw, dtype=np.float32).reshape((256, 3)).copy()
        return arr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# XMP sidecar path
# ---------------------------------------------------------------------------

def xmp_sidecar_path_for_asset(asset_path: Path) -> Path:
    """Return the expected ``.xmp`` sidecar path for *asset_path*."""
    return asset_path.with_suffix(".xmp")


# ---------------------------------------------------------------------------
# IPO → XMP conversion
# ---------------------------------------------------------------------------

def _ipo_crop_to_xmp(values: Mapping[str, Any]) -> Dict[str, str]:
    """Convert IPO centre-based crop to XMP boundary-based crop."""
    cx = float(values.get("Crop_CX", 0.5))
    cy = float(values.get("Crop_CY", 0.5))
    w = float(values.get("Crop_W", 1.0))
    h = float(values.get("Crop_H", 1.0))
    left = max(0.0, cx - w / 2)
    top = max(0.0, cy - h / 2)
    right = min(1.0, cx + w / 2)
    bottom = min(1.0, cy + h / 2)
    angle = float(values.get("Crop_Straighten", 0.0))
    result: Dict[str, str] = {
        "CropLeft": f"{left:.6f}",
        "CropTop": f"{top:.6f}",
        "CropRight": f"{right:.6f}",
        "CropBottom": f"{bottom:.6f}",
    }
    if abs(angle) > 1e-6:
        result["CropAngle"] = f"{angle:.6f}"
    return result


def _bake_combined_lut(adjustments: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Bake curves and levels into a single 256×3 LUT if non-identity."""
    curve_enabled = bool(adjustments.get("Curve_Enabled", False))
    levels_enabled = bool(adjustments.get("Levels_Enabled", False))
    if not curve_enabled and not levels_enabled:
        return None

    # Start with identity
    lut = np.stack([np.linspace(0, 1, 256, dtype=np.float32)] * 3, axis=1)

    # Apply levels first
    if levels_enabled:
        handles = adjustments.get("Levels_Handles")
        if isinstance(handles, list) and len(handles) == 5:
            levels_lut = build_levels_lut(handles)
            for c in range(3):
                indices = np.clip((lut[:, c] * 255).astype(int), 0, 255)
                lut[:, c] = levels_lut[indices, c]

    # Apply curves
    if curve_enabled:
        params = CurveParams()
        params.enabled = True
        for key, attr in [
            ("Curve_RGB", "rgb"),
            ("Curve_Red", "red"),
            ("Curve_Green", "green"),
            ("Curve_Blue", "blue"),
        ]:
            raw = adjustments.get(key)
            if raw and isinstance(raw, list):
                setattr(params, attr, CurveChannel.from_list(raw))
        curve_lut = generate_curve_lut(params)
        for c in range(3):
            indices = np.clip((lut[:, c] * 255).astype(int), 0, 255)
            lut[:, c] = curve_lut[indices, c]

    return lut


def _write_tone_curve_seq(
    parent: ET.Element,
    tag_local: str,
    points: List[Tuple[float, float]],
) -> None:
    """Write a ``crs:ToneCurvePV2012*`` element as an ``rdf:Seq``.

    Adobe Camera Raw stores tone curves as nested ``rdf:Seq`` lists where
    each ``rdf:li`` value is ``"x, y"`` with coordinates in the 0-255
    integer range.  The *points* are expected in the IPO 0.0-1.0 range
    and are scaled automatically.
    """
    curve_el = ET.SubElement(parent, f"{{{_NS_CRS}}}{tag_local}")
    seq = ET.SubElement(curve_el, f"{{{_NS_RDF}}}Seq")
    for x, y in points:
        li = ET.SubElement(seq, f"{{{_NS_RDF}}}li")
        li.text = f"{round(x * 255)}, {round(y * 255)}"


def _read_tone_curve_seq(
    desc: ET.Element,
    tag_local: str,
) -> Optional[List[Tuple[float, float]]]:
    """Read an Adobe ``crs:ToneCurvePV2012*`` sequence back to 0-1 points."""
    curve_el = desc.find(f"{{{_NS_CRS}}}{tag_local}")
    if curve_el is None:
        return None
    seq = curve_el.find(f"{{{_NS_RDF}}}Seq")
    if seq is None:
        return None
    points: List[Tuple[float, float]] = []
    for li in seq.findall(f"{{{_NS_RDF}}}li"):
        if li.text is None:
            continue
        parts = li.text.split(",")
        if len(parts) != 2:
            continue
        try:
            x = float(parts[0].strip()) / 255.0
            y = float(parts[1].strip()) / 255.0
            points.append((x, y))
        except (ValueError, TypeError):
            continue
    return points if points else None


def ipo_to_xmp(adjustments: Dict[str, Any]) -> str:
    """Convert IPO internal adjustments to an XMP XML string.

    The returned string is a valid XMP document with Adobe Camera Raw
    compatible parameters in the ``crs:`` namespace and baked LUT data
    in the ``ipo:`` custom namespace.
    """
    root = ET.Element(f"{{{_NS_X}}}xmpmeta")
    rdf = ET.SubElement(root, f"{{{_NS_RDF}}}RDF")
    desc = ET.SubElement(rdf, f"{{{_NS_RDF}}}Description")

    # Light parameters
    for ipo_key, crs_attr, factor, _ in _LIGHT_PARAM_MAP:
        value = float(adjustments.get(ipo_key, 0.0))
        xmp_val = value * factor
        desc.set(f"{{{_NS_CRS}}}{crs_attr}", f"{xmp_val:+.2f}")

    # Brilliance → Clarity (approximate)
    brilliance = float(adjustments.get("Brilliance", 0.0))
    desc.set(f"{{{_NS_CRS}}}Clarity2012", f"{brilliance * 100:+.2f}")

    # Color parameters
    for ipo_key, crs_attr, factor, _ in _COLOR_PARAM_MAP:
        value = float(adjustments.get(ipo_key, 0.0))
        xmp_val = value * factor
        desc.set(f"{{{_NS_CRS}}}{crs_attr}", f"{xmp_val:+.2f}")

    # White balance
    if bool(adjustments.get("WB_Enabled", False)):
        temp = float(adjustments.get("WB_Temperature", 0.0))
        tint = float(adjustments.get("WB_Tint", 0.0))
        xmp_temp = _WB_TEMP_CENTER + temp * _WB_TEMP_RANGE
        xmp_tint = tint * _WB_TINT_FACTOR
        desc.set(f"{{{_NS_CRS}}}Temperature", f"{xmp_temp:.0f}")
        desc.set(f"{{{_NS_CRS}}}Tint", f"{xmp_tint:+.0f}")
        desc.set(f"{{{_NS_CRS}}}WhiteBalance", "Custom")
    else:
        desc.set(f"{{{_NS_CRS}}}WhiteBalance", "As Shot")

    # Crop
    crop_attrs = _ipo_crop_to_xmp(adjustments)
    for attr, val in crop_attrs.items():
        desc.set(f"{{{_NS_CRS}}}{attr}", val)

    # Baked LUT (curves + levels combined)
    lut = _bake_combined_lut(adjustments)
    if lut is not None:
        desc.set(f"{{{_NS_IPO}}}BakedLUT", _encode_lut_base64(lut))
        desc.set(f"{{{_NS_IPO}}}BakedLUTSize", "256")

    # Adobe-compatible tone curves (crs:ToneCurvePV2012*)
    # Written alongside the baked LUT so Adobe Photoshop / Camera Raw can
    # recognise the curves natively.
    curve_enabled = bool(adjustments.get("Curve_Enabled", False))
    if curve_enabled:
        for ipo_key, crs_tag in _CURVE_CHANNEL_MAP:
            raw = adjustments.get(ipo_key)
            if raw and isinstance(raw, list):
                _write_tone_curve_seq(desc, crs_tag, raw)
            else:
                # Identity curve
                _write_tone_curve_seq(desc, crs_tag, [(0.0, 0.0), (1.0, 1.0)])

    # B&W
    if bool(adjustments.get("BW_Enabled", False)):
        desc.set(f"{{{_NS_CRS}}}ConvertToGrayscale", "True")

    # Selective Color — store in ipo namespace for lossless round-trip
    sc_enabled = bool(adjustments.get("SelectiveColor_Enabled", False))
    if sc_enabled:
        ranges = adjustments.get("SelectiveColor_Ranges")
        if isinstance(ranges, list) and len(ranges) == NUM_RANGES:
            parts: List[str] = []
            for r in ranges:
                if isinstance(r, (list, tuple)) and len(r) >= 5:
                    parts.append(",".join(f"{float(v):.6f}" for v in r[:5]))
            desc.set(f"{{{_NS_IPO}}}SelectiveColorRanges", ";".join(parts))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# XMP → IPO conversion
# ---------------------------------------------------------------------------

def _xmp_crop_to_ipo(desc: ET.Element) -> Dict[str, Any]:
    """Convert XMP boundary-based crop to IPO centre-based crop."""
    result: Dict[str, Any] = {}

    def _get_crs(attr: str) -> Optional[float]:
        val = desc.get(f"{{{_NS_CRS}}}{attr}")
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    left = _get_crs("CropLeft")
    top = _get_crs("CropTop")
    right = _get_crs("CropRight")
    bottom = _get_crs("CropBottom")

    if left is not None and top is not None and right is not None and bottom is not None:
        w = max(0.0, min(1.0, right - left))
        h = max(0.0, min(1.0, bottom - top))
        cx = left + w / 2
        cy = top + h / 2
        result["Crop_CX"] = cx
        result["Crop_CY"] = cy
        result["Crop_W"] = w
        result["Crop_H"] = h

    angle = _get_crs("CropAngle")
    if angle is not None:
        result["Crop_Straighten"] = angle

    return result


def xmp_to_ipo(xmp_content: str) -> Dict[str, Any]:
    """Parse an XMP XML string and return IPO internal adjustments.

    Priority:
    1. If a baked LUT is present in the ``ipo:`` namespace, curve/levels
       data is reconstructed from the LUT.
    2. Otherwise scalar CRS parameters are mapped to IPO equivalents.
    """
    result: Dict[str, Any] = {}

    try:
        root = ET.fromstring(xmp_content)
    except ET.ParseError:
        return result

    # Find the rdf:Description element
    desc = root.find(f".//{{{_NS_RDF}}}Description")
    if desc is None:
        return result

    # Light parameters
    light_found = False
    for ipo_key, crs_attr, _, inv_factor in _LIGHT_PARAM_MAP:
        val = desc.get(f"{{{_NS_CRS}}}{crs_attr}")
        if val is not None:
            try:
                result[ipo_key] = float(val) * inv_factor
                light_found = True
            except (ValueError, TypeError):
                pass

    # Brilliance ← Clarity
    clarity = desc.get(f"{{{_NS_CRS}}}Clarity2012")
    if clarity is not None:
        try:
            result["Brilliance"] = float(clarity) * 0.01
            light_found = True
        except (ValueError, TypeError):
            pass

    if light_found:
        result["Light_Enabled"] = True
        result["Light_Master"] = 0.0

    # Color parameters
    color_found = False
    for ipo_key, crs_attr, _, inv_factor in _COLOR_PARAM_MAP:
        val = desc.get(f"{{{_NS_CRS}}}{crs_attr}")
        if val is not None:
            try:
                result[ipo_key] = float(val) * inv_factor
                color_found = True
            except (ValueError, TypeError):
                pass

    if color_found:
        result["Color_Enabled"] = True
        result["Color_Master"] = 0.0

    # White balance
    wb_mode = desc.get(f"{{{_NS_CRS}}}WhiteBalance")
    if wb_mode and wb_mode.lower() == "custom":
        result["WB_Enabled"] = True
        temp_str = desc.get(f"{{{_NS_CRS}}}Temperature")
        if temp_str is not None:
            try:
                xmp_temp = float(temp_str)
                result["WB_Temperature"] = (xmp_temp - _WB_TEMP_CENTER) / _WB_TEMP_RANGE
            except (ValueError, TypeError):
                pass
        tint_str = desc.get(f"{{{_NS_CRS}}}Tint")
        if tint_str is not None:
            try:
                result["WB_Tint"] = float(tint_str) / _WB_TINT_FACTOR
            except (ValueError, TypeError):
                pass

    # Crop
    result.update(_xmp_crop_to_ipo(desc))

    # Baked LUT
    lut_b64 = desc.get(f"{{{_NS_IPO}}}BakedLUT")
    if lut_b64:
        lut = _decode_lut_base64(lut_b64)
        if lut is not None:
            result["BakedLUT"] = lut
            result["BakedLUT_Enabled"] = True

    # Adobe-compatible tone curves (fallback when no baked LUT)
    curve_found = False
    for ipo_key, crs_tag in _CURVE_CHANNEL_MAP:
        pts = _read_tone_curve_seq(desc, crs_tag)
        if pts is not None:
            result[ipo_key] = pts
            curve_found = True
    if curve_found:
        result["Curve_Enabled"] = True

    # B&W
    grayscale = desc.get(f"{{{_NS_CRS}}}ConvertToGrayscale")
    if grayscale and grayscale.lower() in {"true", "1", "yes"}:
        result["BW_Enabled"] = True

    # Selective Color from ipo namespace
    sc_data = desc.get(f"{{{_NS_IPO}}}SelectiveColorRanges")
    if sc_data:
        ranges: List[List[float]] = []
        for part in sc_data.split(";"):
            vals = part.split(",")
            if len(vals) >= 5:
                try:
                    ranges.append([float(v) for v in vals[:5]])
                except (ValueError, TypeError):
                    continue
        if len(ranges) == NUM_RANGES:
            result["SelectiveColor_Enabled"] = True
            result["SelectiveColor_Ranges"] = ranges

    return result


# ---------------------------------------------------------------------------
# File-level operations
# ---------------------------------------------------------------------------

def load_xmp_adjustments(asset_path: Path) -> Dict[str, Any]:
    """Load adjustments from a ``.xmp`` sidecar file next to *asset_path*.

    Returns an empty dict when the file is missing or unparseable.
    """
    xmp_path = xmp_sidecar_path_for_asset(asset_path)
    if not xmp_path.exists():
        return {}
    try:
        content = xmp_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    return xmp_to_ipo(content)


def export_xmp(asset_path: Path, adjustments: Mapping[str, Any]) -> Path:
    """Export *adjustments* as an Adobe XMP sidecar next to *asset_path*.

    Returns the path to the written ``.xmp`` file.
    """
    xmp_path = xmp_sidecar_path_for_asset(asset_path)
    xmp_path.parent.mkdir(parents=True, exist_ok=True)
    xmp_content = ipo_to_xmp(dict(adjustments))
    tmp_path = xmp_path.with_suffix(xmp_path.suffix + ".tmp")
    tmp_path.write_text(xmp_content, encoding="utf-8")
    try:
        tmp_path.replace(xmp_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
    return xmp_path
