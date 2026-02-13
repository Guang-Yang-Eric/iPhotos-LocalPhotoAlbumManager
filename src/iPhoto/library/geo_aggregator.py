"""Geotagged asset models used by library and map workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class GeotaggedAsset:
    """Lightweight descriptor describing an asset with GPS metadata."""

    library_relative: str
    album_relative: str
    absolute_path: Path
    album_path: Path
    asset_id: str
    latitude: float
    longitude: float
    is_image: bool
    is_video: bool
    still_image_time: float | None
    duration: float | None
    location_name: str | None

