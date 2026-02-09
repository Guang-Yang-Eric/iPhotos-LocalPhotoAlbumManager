"""Read/write helpers for Adobe ``.xmp`` sidecar files and IPO↔XMP conversion.

This module provides:
- XMP sidecar path resolution
- Loading adjustments from XMP files
- Exporting adjustments to XMP format
- Bidirectional conversion between IPO internal format and XMP

The conversion strategy prioritises **result consistency** over process consistency.
For per-channel colour transforms (light adjustments, curves, levels) the module
bakes a complete 256×3 LUT and writes it both as:
1. ``ipo:BakedLUT`` (Base64 blob) for exact IPO round-trip
2. ``crs:ToneCurvePV2012Red/Green/Blue`` with 256 densely-sampled points so
   Adobe Camera Raw reproduces the same colour result without relying on a
   spline interpolation that may differ between applications.

All raw IPO parameter values are additionally stored in the ``ipo:`` namespace
so the app can reconstruct exact slider positions on round-trip, even when
the CRS approximations are lossy.
"""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..core.curve_resolver import (
    CurveParams,
    CurveChannel,
    generate_curve_lut,
)
from ..core.levels_resolver import build_levels_lut
from ..core.selective_color_resolver import NUM_RANGES

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

# Facade pre-processing factors (must match facade.py for consistent baking)
_EXPOSURE_FACTOR = 1.5
_BRIGHTNESS_FACTOR = 0.75
_BRILLIANCE_FACTOR = 0.6

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


def _apply_channel_adjustments_lut(
    channel: np.ndarray,
    exposure: float,
    brightness: float,
    brilliance: float,
    highlights: float,
    shadows: float,
    contrast_factor: float,
    black_point: float,
) -> np.ndarray:
    """Replicate the tone curve from ``numpy_executor`` for LUT baking.

    This must match ``_np_apply_channel_adjustments`` exactly so the baked
    LUT produces identical results to the live rendering path.
    """
    adjusted = channel + exposure + brightness

    mid_distance = channel - 0.5
    adjusted = adjusted + brilliance * (1.0 - (mid_distance * 2.0) ** 2)

    cond_high = adjusted > 0.65
    cond_low = adjusted < 0.35

    ratio_high = (adjusted - 0.65) / 0.35
    delta_high = highlights * ratio_high

    ratio_low = (0.35 - adjusted) / 0.35
    delta_low = shadows * ratio_low

    val_high = adjusted + delta_high
    val_low = adjusted + delta_low

    adjusted = np.select([cond_high, cond_low], [val_high, val_low], default=adjusted)

    adjusted = (adjusted - 0.5) * contrast_factor + 0.5

    if black_point > 0:
        adjusted = adjusted - black_point * (1.0 - adjusted)
    elif black_point < 0:
        adjusted = adjusted - black_point * adjusted

    return np.clip(adjusted, 0.0, 1.0)


def _bake_full_pipeline_lut(adjustments: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Bake light adjustments, curves and levels into a single 256×3 LUT.

    The baked LUT captures the complete per-channel processing pipeline
    that the app applies, in the same order as ``facade.py``:
    1. Light (exposure, brightness, brilliance, highlights, shadows, contrast, black_point)
    2. Curves
    3. Levels

    Returns ``None`` when all stages are identity (nothing to bake).
    """
    # Extract light parameters using the same pre-processing as facade.py
    exposure = float(adjustments.get("Exposure", 0.0))
    brightness = float(adjustments.get("Brightness", 0.0))
    brilliance = float(adjustments.get("Brilliance", 0.0))
    highlights = float(adjustments.get("Highlights", 0.0))
    shadows = float(adjustments.get("Shadows", 0.0))
    contrast = float(adjustments.get("Contrast", 0.0))
    black_point = float(adjustments.get("BlackPoint", 0.0))

    # Facade pre-processing (factors must match facade.py)
    exposure_term = exposure * _EXPOSURE_FACTOR
    brightness_term = brightness * _BRIGHTNESS_FACTOR
    brilliance_strength = brilliance * _BRILLIANCE_FACTOR
    contrast_factor = 1.0 + contrast

    light_active = any(
        abs(v) > 1e-6
        for v in (exposure, brightness, brilliance, highlights, shadows, contrast, black_point)
    )

    curve_enabled = bool(adjustments.get("Curve_Enabled", False))
    levels_enabled = bool(adjustments.get("Levels_Enabled", False))

    if not light_active and not curve_enabled and not levels_enabled:
        return None

    # Start with identity
    identity = np.linspace(0, 1, 256, dtype=np.float32)
    lut = np.stack([identity.copy(), identity.copy(), identity.copy()], axis=1)

    # 1. Light adjustments (per-channel, same algorithm as numpy_executor)
    if light_active:
        for c in range(3):
            lut[:, c] = _apply_channel_adjustments_lut(
                lut[:, c],
                exposure_term,
                brightness_term,
                brilliance_strength,
                highlights,
                shadows,
                contrast_factor,
                black_point,
            )

    # 2. Curves (applied after light in facade.py, before levels)
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

    # 3. Levels (applied after curves in facade.py)
    if levels_enabled:
        handles = adjustments.get("Levels_Handles")
        if isinstance(handles, list) and len(handles) == 5:
            levels_lut = build_levels_lut(handles)
            for c in range(3):
                indices = np.clip((lut[:, c] * 255).astype(int), 0, 255)
                lut[:, c] = levels_lut[indices, c]

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


def _write_tone_curve_from_lut(
    parent: ET.Element,
    tag_local: str,
    lut_channel: np.ndarray,
) -> None:
    """Write a dense ToneCurvePV2012 by sampling every level of *lut_channel*.

    *lut_channel* must be a 256-element float32 array mapping normalised
    input levels to normalised output levels.  Each integer input (0-255)
    is written with its exact output, giving Adobe Camera Raw a direct
    lookup table with no interpolation needed.
    """
    curve_el = ET.SubElement(parent, f"{{{_NS_CRS}}}{tag_local}")
    seq = ET.SubElement(curve_el, f"{{{_NS_RDF}}}Seq")
    for i in range(256):
        out = int(round(float(lut_channel[i]) * 255.0))
        out = max(0, min(255, out))
        li = ET.SubElement(seq, f"{{{_NS_RDF}}}li")
        li.text = f"{i}, {out}"


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


# ---------------------------------------------------------------------------
# Raw IPO value serialisation (ipo: namespace, lossless round-trip)
# ---------------------------------------------------------------------------

# All IPO scalar keys that should be preserved in the ipo: namespace.
_IPO_SCALAR_KEYS: List[str] = [
    # Light
    "Brilliance", "Exposure", "Highlights", "Shadows",
    "Brightness", "Contrast", "BlackPoint",
    # Color
    "Saturation", "Vibrance", "Cast",
    # WB
    "WB_Warmth", "WB_Temperature", "WB_Tint",
    # B&W
    "BW_Intensity", "BW_Neutrals", "BW_Tone", "BW_Grain",
    # Master sliders
    "Light_Master", "Color_Master",
    # Gains
    "Color_Gain_R", "Color_Gain_G", "Color_Gain_B",
]

# Boolean IPO keys to preserve.
_IPO_BOOL_KEYS: List[str] = [
    "Light_Enabled", "Color_Enabled", "WB_Enabled",
    "BW_Enabled", "Curve_Enabled", "Levels_Enabled",
    "SelectiveColor_Enabled",
]


def _store_raw_ipo_values(desc: ET.Element, adjustments: Mapping[str, Any]) -> None:
    """Store all raw IPO parameter values in the ``ipo:`` namespace.

    This enables lossless round-trip: when the XMP is read back by the app,
    the exact slider positions and parameter values are recovered regardless
    of any lossy CRS approximation.
    """
    # Scalar float values
    for key in _IPO_SCALAR_KEYS:
        if key in adjustments:
            desc.set(f"{{{_NS_IPO}}}{key}", f"{float(adjustments[key]):.8f}")

    # Boolean flags
    for key in _IPO_BOOL_KEYS:
        if key in adjustments:
            desc.set(f"{{{_NS_IPO}}}{key}", "1" if bool(adjustments[key]) else "0")

    # Curve control points (semicolon-separated "x,y" pairs)
    for ipo_key in ("Curve_RGB", "Curve_Red", "Curve_Green", "Curve_Blue"):
        pts = adjustments.get(ipo_key)
        if isinstance(pts, list) and pts:
            serialised = ";".join(
                f"{float(p[0]):.8f},{float(p[1]):.8f}" for p in pts
            )
            desc.set(f"{{{_NS_IPO}}}{ipo_key}", serialised)

    # Levels handles (comma-separated floats)
    handles = adjustments.get("Levels_Handles")
    if isinstance(handles, list) and len(handles) == 5:
        desc.set(
            f"{{{_NS_IPO}}}Levels_Handles",
            ",".join(f"{float(h):.8f}" for h in handles),
        )


def _load_raw_ipo_values(desc: ET.Element) -> Dict[str, Any]:
    """Recover raw IPO parameter values from the ``ipo:`` namespace.

    Returns an empty dict when no ``ipo:`` raw values are present.
    """
    result: Dict[str, Any] = {}

    # Scalar floats
    for key in _IPO_SCALAR_KEYS:
        val = desc.get(f"{{{_NS_IPO}}}{key}")
        if val is not None:
            try:
                result[key] = float(val)
            except (ValueError, TypeError):
                pass  # Skip non-numeric scalar values from XMP
    for key in _IPO_BOOL_KEYS:
        val = desc.get(f"{{{_NS_IPO}}}{key}")
        if val is not None:
            result[key] = val in ("1", "true", "True")

    # Curve control points
    for ipo_key in ("Curve_RGB", "Curve_Red", "Curve_Green", "Curve_Blue"):
        val = desc.get(f"{{{_NS_IPO}}}{ipo_key}")
        if val:
            pts: List[Tuple[float, float]] = []
            for pair in val.split(";"):
                parts = pair.split(",")
                if len(parts) == 2:
                    try:
                        pts.append((float(parts[0]), float(parts[1])))
                    except (ValueError, TypeError):
                        continue
            if pts:
                result[ipo_key] = pts

    # Levels handles
    handles_str = desc.get(f"{{{_NS_IPO}}}Levels_Handles")
    if handles_str:
        try:
            handles = [float(v) for v in handles_str.split(",")]
            if len(handles) == 5:
                result["Levels_Handles"] = handles
        except (ValueError, TypeError):
            pass  # Ignore malformed levels handles; rely on defaults

    return result


def ipo_to_xmp(adjustments: Dict[str, Any]) -> str:
    """Convert IPO internal adjustments to an XMP XML string.

    The returned string is a valid XMP document with:
    - Adobe Camera Raw compatible scalar parameters in ``crs:``
    - Densely-sampled per-channel tone curves in ``crs:ToneCurvePV2012*``
      that capture light+curve+levels effects for result equivalence
    - Complete raw IPO values in ``ipo:`` for lossless round-trip
    - Baked 256×3 LUT in ``ipo:BakedLUT`` for exact per-channel reproduction
    """
    root = ET.Element(f"{{{_NS_X}}}xmpmeta")
    rdf = ET.SubElement(root, f"{{{_NS_RDF}}}RDF")
    desc = ET.SubElement(rdf, f"{{{_NS_RDF}}}Description")

    # --- CRS parameters (best-effort for Adobe UI) ---

    # Light parameters (CRS approximation)
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

    # --- Baked per-channel pipeline LUT ---
    # Combines light + curves + levels into a single 256×3 LUT
    lut = _bake_full_pipeline_lut(adjustments)
    if lut is not None:
        desc.set(f"{{{_NS_IPO}}}BakedLUT", _encode_lut_base64(lut))
        desc.set(f"{{{_NS_IPO}}}BakedLUTSize", "256")

        # Adobe-compatible tone curves: densely sample the baked LUT at all
        # 256 levels for each channel.  This gives Adobe Camera Raw an exact
        # per-pixel mapping with no spline interpolation ambiguity.
        # Set master curve to identity since per-channel curves already include
        # the RGB master curve effect.
        _write_tone_curve_seq(desc, "ToneCurvePV2012", [(0.0, 0.0), (1.0, 1.0)])
        _write_tone_curve_from_lut(desc, "ToneCurvePV2012Red", lut[:, 0])
        _write_tone_curve_from_lut(desc, "ToneCurvePV2012Green", lut[:, 1])
        _write_tone_curve_from_lut(desc, "ToneCurvePV2012Blue", lut[:, 2])

        # Zero out the CRS light scalar parameters since their effect is
        # already baked into the per-channel tone curves.  This prevents
        # Adobe from double-applying light adjustments.
        for _, crs_attr, _, _ in _LIGHT_PARAM_MAP:
            desc.set(f"{{{_NS_CRS}}}{crs_attr}", "+0.00")
        desc.set(f"{{{_NS_CRS}}}Clarity2012", "+0.00")

    else:
        # No per-channel effects — only write sparse curve points if curves
        # were explicitly enabled (backward compatibility)
        curve_enabled = bool(adjustments.get("Curve_Enabled", False))
        if curve_enabled:
            for ipo_key, crs_tag in _CURVE_CHANNEL_MAP:
                raw = adjustments.get(ipo_key)
                if raw and isinstance(raw, list):
                    _write_tone_curve_seq(desc, crs_tag, raw)
                else:
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

    # --- Raw IPO values for lossless round-trip ---
    _store_raw_ipo_values(desc, adjustments)

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
    1. If raw IPO values exist in the ``ipo:`` namespace, use those for
       lossless round-trip (exact slider positions recovered).
    2. If a baked LUT is present, include it for exact per-channel rendering.
    3. Fall back to CRS scalar parameter conversions for Adobe-written XMPs.
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

    # --- Try ipo: raw values first (lossless round-trip) ---
    raw_ipo = _load_raw_ipo_values(desc)
    if raw_ipo:
        result.update(raw_ipo)

    # --- CRS fallback for parameters NOT present in ipo: raw values ---

    # Light parameters
    light_found = False
    for ipo_key, crs_attr, _, inv_factor in _LIGHT_PARAM_MAP:
        if ipo_key not in result:
            val = desc.get(f"{{{_NS_CRS}}}{crs_attr}")
            if val is not None:
                try:
                    result[ipo_key] = float(val) * inv_factor
                    light_found = True
                except (ValueError, TypeError):
                    pass  # Skip non-numeric CRS light parameter values
    if "Brilliance" not in result:
        clarity = desc.get(f"{{{_NS_CRS}}}Clarity2012")
        if clarity is not None:
            try:
                result["Brilliance"] = float(clarity) * 0.01
                light_found = True
            except (ValueError, TypeError):
                pass  # Ignore malformed Clarity values from XMP

    if light_found and "Light_Enabled" not in result:
        result["Light_Enabled"] = True
        result.setdefault("Light_Master", 0.0)

    # Color parameters
    color_found = False
    for ipo_key, crs_attr, _, inv_factor in _COLOR_PARAM_MAP:
        if ipo_key not in result:
            val = desc.get(f"{{{_NS_CRS}}}{crs_attr}")
            if val is not None:
                try:
                    result[ipo_key] = float(val) * inv_factor
                    color_found = True
                except (ValueError, TypeError):
                    pass  # Skip non-numeric CRS color parameter values

    if color_found and "Color_Enabled" not in result:
        result["Color_Enabled"] = True
        result.setdefault("Color_Master", 0.0)

    # White balance
    if "WB_Enabled" not in result:
        wb_mode = desc.get(f"{{{_NS_CRS}}}WhiteBalance")
        if wb_mode and wb_mode.lower() == "custom":
            result["WB_Enabled"] = True
            if "WB_Temperature" not in result:
                temp_str = desc.get(f"{{{_NS_CRS}}}Temperature")
                if temp_str is not None:
                    try:
                        xmp_temp = float(temp_str)
                        result["WB_Temperature"] = (xmp_temp - _WB_TEMP_CENTER) / _WB_TEMP_RANGE
                    except (ValueError, TypeError):
                        pass  # Ignore malformed temperature value
            if "WB_Tint" not in result:
                tint_str = desc.get(f"{{{_NS_CRS}}}Tint")
                if tint_str is not None:
                    try:
                        result["WB_Tint"] = float(tint_str) / _WB_TINT_FACTOR
                    except (ValueError, TypeError):
                        pass  # Ignore malformed tint value

    # Crop
    result.update(_xmp_crop_to_ipo(desc))

    # Baked LUT
    lut_b64 = desc.get(f"{{{_NS_IPO}}}BakedLUT")
    if lut_b64:
        lut = _decode_lut_base64(lut_b64)
        if lut is not None:
            result["BakedLUT"] = lut
            result["BakedLUT_Enabled"] = True

    # Adobe-compatible tone curves (fallback when no ipo: curve data)
    if "Curve_RGB" not in result:
        curve_found = False
        for ipo_key, crs_tag in _CURVE_CHANNEL_MAP:
            if ipo_key not in result:
                pts = _read_tone_curve_seq(desc, crs_tag)
                if pts is not None:
                    result[ipo_key] = pts
                    curve_found = True
        if curve_found and "Curve_Enabled" not in result:
            result["Curve_Enabled"] = True

    # B&W
    if "BW_Enabled" not in result:
        grayscale = desc.get(f"{{{_NS_CRS}}}ConvertToGrayscale")
        if grayscale and grayscale.lower() in {"true", "1", "yes"}:
            result["BW_Enabled"] = True

    # Selective Color from ipo namespace
    if "SelectiveColor_Ranges" not in result:
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
