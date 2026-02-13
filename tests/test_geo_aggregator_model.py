import importlib.util
from pathlib import Path
import sys


def _load_geo_aggregator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src/iPhoto/library/geo_aggregator.py"
    spec = importlib.util.spec_from_file_location("test_geo_aggregator", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_geo_aggregator"] = module
    spec.loader.exec_module(module)
    return module


GeotaggedAsset = _load_geo_aggregator().GeotaggedAsset


def test_geotagged_asset_fields_are_preserved():
    item = GeotaggedAsset(
        library_relative="Album/a.jpg",
        album_relative="a.jpg",
        absolute_path=Path("/tmp/lib/Album/a.jpg"),
        album_path=Path("/tmp/lib/Album"),
        asset_id="asset-1",
        latitude=1.25,
        longitude=103.5,
        is_image=True,
        is_video=False,
        still_image_time=None,
        duration=None,
        location_name="Singapore",
    )

    assert item.asset_id == "asset-1"
    assert item.location_name == "Singapore"
    assert item.absolute_path.name == "a.jpg"
