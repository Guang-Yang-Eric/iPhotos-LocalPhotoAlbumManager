from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


viewport = _load_module("test_map_viewport", "src/maps/map_widget/viewport.py")
tile_collector = _load_module("test_map_tile_collector", "src/maps/map_widget/tile_collector.py")


class _StubTileManager:
    def __init__(self, data: dict[tuple[int, int, int], dict], missing: set[tuple[int, int, int]]):
        self._data = data
        self._missing = missing

    def get_tile(self, tile_key: tuple[int, int, int]):
        return self._data.get(tile_key)

    def is_tile_missing(self, tile_key: tuple[int, int, int]) -> bool:
        return tile_key in self._missing


def test_compute_view_state_clamps_fetch_zoom():
    state = viewport.compute_view_state(
        tile_size=256,
        center_x=0.5,
        center_y=0.5,
        zoom=9.3,
        width=1000,
        height=600,
    )

    assert state.fetch_zoom == viewport.MAX_TILE_ZOOM_LEVEL
    assert state.scaled_tile_size > 256
    assert state.width == 1000
    assert state.height == 600


def test_collect_tiles_wraps_x_and_schedules_non_missing_tiles():
    state = viewport.compute_view_state(
        tile_size=256,
        center_x=0.5,
        center_y=0.5,
        zoom=1.0,
        width=256,
        height=256,
    )
    available_key = (state.fetch_zoom, 1, 1)
    manager = _StubTileManager(data={available_key: {"land": {}}}, missing={(state.fetch_zoom, 0, 1)})

    draw, request = tile_collector.collect_tiles(view_state=state, tile_manager=manager)

    assert any(item[0] == available_key for item in draw)
    assert all(item[1] != (state.fetch_zoom, 0, 1) for item in request)
