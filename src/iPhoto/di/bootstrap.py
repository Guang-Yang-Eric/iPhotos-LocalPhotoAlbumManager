from .container import Container
from iPhoto.events.bus import EventBus
from iPhoto.infrastructure.services.cache_stats import CacheStatsCollector
from iPhoto.infrastructure.services.memory_monitor import MemoryMonitor, GiB
from iPhoto.infrastructure.services.thumbnail_cache import MemoryThumbnailCache
from iPhoto.infrastructure.services.disk_thumbnail_cache import DiskThumbnailCache
from iPhoto.infrastructure.services.weak_asset_cache import WeakAssetCache
from iPhoto.application.services.parallel_scanner import ParallelScanner

def bootstrap(container: Container) -> None:
    """Register all application services in the DI container."""
    # Core infrastructure
    container.register_singleton(EventBus, EventBus)

    # Performance: cache statistics
    container.register_singleton(CacheStatsCollector, CacheStatsCollector)

    # Performance: memory monitor (warning at 1 GiB, critical at 2 GiB)
    container.register_factory(
        MemoryMonitor,
        lambda: MemoryMonitor(warning_bytes=1 * GiB, critical_bytes=2 * GiB),
        singleton=True,
    )

    # Performance: thumbnail caches (L1 memory)
    container.register_factory(
        MemoryThumbnailCache,
        lambda: MemoryThumbnailCache(max_size=500),
        singleton=True,
    )

    # Performance: weak asset cache
    container.register_factory(
        WeakAssetCache,
        lambda: WeakAssetCache(max_size=1000),
        singleton=True,
    )

    # Performance: parallel scanner
    container.register_factory(
        ParallelScanner,
        lambda: ParallelScanner(
            max_workers=4,
            batch_size=100,
            event_bus=container.resolve(EventBus),
        ),
        singleton=True,
    )
