"""DI container bootstrap — registers all application & infrastructure services."""

from __future__ import annotations

import logging

from .container import Container
from iPhoto.events.bus import EventBus
from iPhoto.infrastructure.services.cache_stats import CacheStatsCollector
from iPhoto.infrastructure.services.thumbnail_cache import MemoryThumbnailCache
from iPhoto.infrastructure.services.weak_asset_cache import WeakAssetCache
from iPhoto.infrastructure.services.memory_monitor import MemoryMonitor, GiB
from iPhoto.infrastructure.services.gpu_pipeline import (
    FBOPool,
    StreamingTextureUploader,
)

_LOGGER = logging.getLogger(__name__)


def bootstrap(container: Container) -> None:
    """Register all application services in the DI container."""
    # Core
    container.register_singleton(EventBus, EventBus)

    # Performance: cache infrastructure
    container.register_singleton(CacheStatsCollector, CacheStatsCollector)
    container.register_singleton(MemoryThumbnailCache, MemoryThumbnailCache, max_size=500)
    container.register_singleton(WeakAssetCache, WeakAssetCache, max_size=5000)

    # Performance: memory monitor with sensible defaults
    container.register_singleton(
        MemoryMonitor,
        MemoryMonitor,
        warning_bytes=1 * GiB,
        critical_bytes=2 * GiB,
    )

    # Performance: GPU pipeline components
    container.register_singleton(
        StreamingTextureUploader,
        StreamingTextureUploader,
        chunk_height=256,
    )
    container.register_singleton(FBOPool, FBOPool, max_size=4)

    _LOGGER.info(
        "DI bootstrap complete — registered: EventBus, CacheStatsCollector, "
        "MemoryThumbnailCache(500), WeakAssetCache(5000), MemoryMonitor(1/2 GiB), "
        "StreamingTextureUploader(chunk=256), FBOPool(max=4)"
    )
