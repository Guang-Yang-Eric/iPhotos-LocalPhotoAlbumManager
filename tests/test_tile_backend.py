from __future__ import annotations

from dataclasses import dataclass

from maps.map_sources import MapBackendMetadata
from maps.tile_backend import FallbackTileBackend, TileBackendUnavailableError, TileRenderError


@dataclass
class _FakeBackend:
    metadata: MapBackendMetadata
    probe_error: Exception | None = None
    load_error: Exception | None = None
    tile: object | None = None

    def __post_init__(self) -> None:
        self.probe_calls = 0
        self.load_calls = 0
        self.cleared = False
        self.shutdown_called = False
        self.device_scales: list[float] = []

    def probe(self) -> MapBackendMetadata:
        self.probe_calls += 1
        if self.probe_error is not None:
            raise self.probe_error
        return self.metadata

    def load_tile(self, z: int, x: int, y: int) -> object | None:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        return self.tile

    def clear_cache(self) -> None:
        self.cleared = True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def set_device_scale(self, scale: float) -> None:
        self.device_scales.append(scale)


def test_fallback_backend_uses_legacy_metadata_when_primary_probe_fails() -> None:
    fallback_metadata = MapBackendMetadata(0.0, 6.0, False, "vector")
    primary = _FakeBackend(
        metadata=MapBackendMetadata(2.0, 19.0, True, "raster"),
        probe_error=TileBackendUnavailableError("helper missing"),
    )
    fallback = _FakeBackend(metadata=fallback_metadata, tile={"legacy": True})

    backend = FallbackTileBackend(primary, fallback)

    assert backend.probe() == fallback_metadata
    assert backend.load_tile(1, 2, 3) == {"legacy": True}
    assert primary.load_calls == 0
    assert fallback.load_calls == 1


def test_fallback_backend_disables_primary_after_runtime_unavailable() -> None:
    primary = _FakeBackend(
        metadata=MapBackendMetadata(2.0, 18.0, True, "raster"),
        load_error=TileBackendUnavailableError("helper crashed"),
    )
    fallback = _FakeBackend(
        metadata=MapBackendMetadata(0.0, 6.0, False, "vector"),
        tile={"legacy": True},
    )

    backend = FallbackTileBackend(primary, fallback)

    first_tile = backend.load_tile(4, 5, 6)
    second_tile = backend.load_tile(4, 5, 7)

    assert first_tile == {"legacy": True}
    assert second_tile == {"legacy": True}
    assert primary.load_calls == 1
    assert fallback.load_calls == 2
    assert backend.probe() == fallback.metadata


def test_fallback_backend_keeps_primary_enabled_after_render_error() -> None:
    """A per-tile TileRenderError must not permanently disable the primary backend.

    OsmAndRasterBackend converts render-phase TileBackendUnavailableErrors into
    TileRenderErrors so that a slow or stuck helper process does not kill the OBF
    backend for the entire session.  FallbackTileBackend must keep _primary_enabled
    True and retry the primary on subsequent tiles.
    """
    primary_metadata = MapBackendMetadata(2.0, 18.0, True, "raster")
    primary = _FakeBackend(
        metadata=primary_metadata,
        load_error=TileRenderError("render timed out"),
    )
    fallback = _FakeBackend(
        metadata=MapBackendMetadata(0.0, 6.0, False, "vector"),
        tile={"legacy": True},
    )

    backend = FallbackTileBackend(primary, fallback)

    first_tile = backend.load_tile(1, 0, 0)
    second_tile = backend.load_tile(1, 0, 1)

    assert first_tile == {"legacy": True}
    assert second_tile == {"legacy": True}
    # Primary must be tried for BOTH tiles (not permanently disabled after the first error).
    assert primary.load_calls == 2
    assert fallback.load_calls == 2
    # Backend metadata stays as raster because the primary was never permanently disabled.
    assert backend.probe() == primary_metadata
