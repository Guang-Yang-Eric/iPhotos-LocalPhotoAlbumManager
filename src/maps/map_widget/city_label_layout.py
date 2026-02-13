"""City annotation models and projection helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QRectF

MERCATOR_LAT_BOUND = 85.05112878


@dataclass(frozen=True)
class CityAnnotation:
    """Descriptor describing a lightweight label drawn directly on the map."""

    longitude: float
    latitude: float
    display_name: str
    full_name: str


@dataclass
class RenderedCityLabel:
    """Runtime data cached for hit-testing rendered city annotations."""

    bounds: QRectF
    full_name: str


def lonlat_to_world(lon: float, lat: float, world_size: float) -> Optional[tuple[float, float]]:
    """Convert geographic coordinates to Mercator world coordinates."""

    try:
        lon_value = float(lon)
        lat_value = float(lat)
    except (TypeError, ValueError):
        return None

    lat_value = max(min(lat_value, MERCATOR_LAT_BOUND), -MERCATOR_LAT_BOUND)
    x = (lon_value + 180.0) / 360.0 * world_size
    sin_lat = math.sin(math.radians(lat_value))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * world_size
    return x, y


__all__ = ["CityAnnotation", "RenderedCityLabel", "lonlat_to_world"]
