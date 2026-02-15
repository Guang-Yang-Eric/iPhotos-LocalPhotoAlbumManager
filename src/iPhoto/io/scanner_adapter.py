"""Adapter to bridge legacy scanner calls to the new infrastructure.

Scanning is fully parallel: a :class:`FileDiscoveryThread` feeds file paths
into a shared queue while *N* :class:`_MetadataWorkerThread` threads drain it.
Each worker owns a dedicated persistent ``exiftool -stay_open`` process from
:class:`~iPhoto.utils.exiftool_pool.ExifToolPool`, so metadata extraction
happens concurrently without subprocess-spawn overhead.

Completed rows are pushed to a thread-safe *result_queue* and yielded by the
:func:`scan_album` generator as soon as they arrive.  The downstream
:class:`ScannerWorker` then chunks them into ``chunkReady`` signals that reach
the UI through the existing throttled pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Callable, Iterable
import queue
import unicodedata
from datetime import datetime
import mimetypes

from ..application.interfaces import IMetadataProvider, IThumbnailGenerator
from ..infrastructure.services.metadata_provider import ExifToolMetadataProvider
from ..infrastructure.services.thumbnail_generator import PillowThumbnailGenerator
from ..utils.pathutils import should_include
from ..config import DEFAULT_INCLUDE, DEFAULT_EXCLUDE
from ..application.use_cases.scan_album import FileDiscoveryThread
from ..utils.exiftool_pool import get_exiftool_pool, PersistentExifTool

_scan_logger = logging.getLogger("iPhoto.scanner")

# Number of parallel metadata worker threads.
_NUM_WORKERS = 4

# Batch size for exiftool calls within each worker.
_EXIFTOOL_BATCH = 20

# Instantiate services directly for the adapter (stateless)
_metadata_provider = ExifToolMetadataProvider()
_thumbnail_generator = PillowThumbnailGenerator()


def process_media_paths(
    root: Path, image_paths: List[Path], video_paths: List[Path]
) -> Iterator[Dict[str, Any]]:
    """Yield populated index rows for the provided media paths."""

    all_paths = image_paths + video_paths
    if not all_paths:
        return

    # Process in batches
    BATCH_SIZE = 50
    for i in range(0, len(all_paths), BATCH_SIZE):
        batch = all_paths[i : i + BATCH_SIZE]

        # Get metadata
        meta_batch = _metadata_provider.get_metadata_batch(batch)

        # Build lookup
        meta_lookup = {}
        for m in meta_batch:
            src = m.get("SourceFile")
            if src:
                meta_lookup[src] = m
                meta_lookup[unicodedata.normalize('NFC', src)] = m
                meta_lookup[unicodedata.normalize('NFD', src)] = m

        for path in batch:
            try:
                raw_meta = meta_lookup.get(path.as_posix())
                if not raw_meta:
                    raw_meta = meta_lookup.get(unicodedata.normalize('NFC', path.as_posix()))

                # Normalize
                row = _metadata_provider.normalize_metadata(root, path, raw_meta or {})

                # Micro-thumbnail for images
                if row.get("media_type") == 0:
                    mt = _thumbnail_generator.generate_micro_thumbnail(path)
                    if mt:
                        row["micro_thumbnail"] = mt

                yield row
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Parallel metadata worker
# ---------------------------------------------------------------------------

class _MetadataWorkerThread(threading.Thread):
    """Worker thread that owns a dedicated persistent exiftool process.

    Pulls batches of file paths from *path_queue*, extracts metadata using its
    dedicated :class:`PersistentExifTool`, normalises the rows, generates
    micro-thumbnails, and pushes completed rows to *result_queue*.
    """

    def __init__(
        self,
        worker_id: int,
        root: Path,
        path_queue: "queue.Queue[Optional[List[Path]]]",
        result_queue: "queue.Queue[Dict[str, Any]]",
        existing_index: Optional[Dict[str, Dict[str, Any]]],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"ScanWorker-{worker_id}", daemon=True)
        self._worker_id = worker_id
        self._root = root
        self._path_queue = path_queue
        self._result_queue = result_queue
        self._existing_index = existing_index
        self._stop_event = stop_event
        self._processed = 0

    @property
    def processed(self) -> int:
        return self._processed

    def run(self) -> None:
        pool = get_exiftool_pool(_NUM_WORKERS)
        et = pool.get()
        try:
            self._work_loop(et)
        finally:
            pool.put(et)

    def _work_loop(self, et: PersistentExifTool) -> None:
        while not self._stop_event.is_set():
            try:
                batch = self._path_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if batch is None:
                break
            self._process_batch(et, batch)

    def _process_batch(self, et: PersistentExifTool, paths: List[Path]) -> None:
        """Process a batch: check cache, extract metadata, generate thumbnails."""
        to_extract: List[Path] = []

        # 1. Cache check — yield cached rows immediately
        for p in paths:
            if self._stop_event.is_set():
                return
            rel = p.relative_to(self._root).as_posix()
            cached = None
            if self._existing_index:
                cached = self._existing_index.get(rel)
                if not cached:
                    cached = self._existing_index.get(
                        unicodedata.normalize("NFC", rel)
                    )
            if cached:
                try:
                    st = p.stat()
                    cached_ts = cached.get("ts")
                    current_ts = int(st.st_mtime * 1_000_000)
                    if (
                        cached.get("bytes") == st.st_size
                        and abs((cached_ts or 0) - current_ts) <= 1_000_000
                    ):
                        self._result_queue.put(cached)
                        self._processed += 1
                        continue
                except OSError:
                    pass
            to_extract.append(p)

        if not to_extract or self._stop_event.is_set():
            return

        # 2. Metadata extraction using the persistent exiftool
        meta_batch = et.get_metadata_batch(to_extract)

        meta_lookup: Dict[str, Dict[str, Any]] = {}
        for m in meta_batch:
            src = m.get("SourceFile")
            if src:
                meta_lookup[src] = m
                meta_lookup[unicodedata.normalize("NFC", src)] = m
                meta_lookup[unicodedata.normalize("NFD", src)] = m

        # 3. Normalise + micro-thumbnail per file
        for p in to_extract:
            if self._stop_event.is_set():
                return
            try:
                raw_meta = meta_lookup.get(p.as_posix())
                if not raw_meta:
                    raw_meta = meta_lookup.get(
                        unicodedata.normalize("NFC", p.as_posix())
                    )
                row = _metadata_provider.normalize_metadata(
                    self._root, p, raw_meta or {}
                )
                if row.get("media_type") == 0:
                    mt = _thumbnail_generator.generate_micro_thumbnail(p)
                    if mt:
                        row["micro_thumbnail"] = mt
                self._result_queue.put(row)
                self._processed += 1
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_album(
    root: Path,
    include_globs: Iterable[str],
    exclude_globs: Iterable[str],
    existing_index: Optional[Dict[str, Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    num_workers: int = _NUM_WORKERS,
) -> Iterator[Dict[str, Any]]:
    """Yield index rows for all matching assets in *root*, scanning in parallel.

    File discovery runs in a background thread.  Discovered files are split
    into batches and distributed to *num_workers* metadata worker threads,
    each backed by a dedicated persistent ``exiftool`` process from
    :class:`~iPhoto.utils.exiftool_pool.ExifToolPool`.

    Completed rows stream through a thread-safe *result_queue* so the caller
    receives them as soon as each file is processed — not when an entire
    batch finishes.  This keeps the downstream :class:`ScannerWorker` emitting
    ``chunkReady`` signals at a steady cadence and the UI responsive.
    """

    scan_start = time.monotonic()
    _scan_logger.info(
        "▶ Parallel scan started: %s (workers=%d, exiftool_batch=%d)",
        root.name,
        num_workers,
        _EXIFTOOL_BATCH,
    )

    # ── file discovery ────────────────────────────────────────────────
    discovery_queue: queue.Queue[Optional[Path]] = queue.Queue(maxsize=1000)
    discoverer = FileDiscoveryThread(
        root,
        discovery_queue,
        include=list(include_globs),
        exclude=list(exclude_globs),
    )
    discoverer.start()

    # ── batching → worker input queue ─────────────────────────────────
    #    Each item is a List[Path] of length ≤ _EXIFTOOL_BATCH, or None as
    #    a stop sentinel.
    work_queue: queue.Queue[Optional[List[Path]]] = queue.Queue(maxsize=100)
    result_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    stop_event = threading.Event()

    # ── spawn metadata workers ────────────────────────────────────────
    workers: List[_MetadataWorkerThread] = []
    for i in range(num_workers):
        w = _MetadataWorkerThread(
            worker_id=i,
            root=root,
            path_queue=work_queue,
            result_queue=result_queue,
            existing_index=existing_index,
            stop_event=stop_event,
        )
        w.start()
        workers.append(w)

    total_yielded = 0
    total_discovered = 0

    try:
        if progress_callback:
            progress_callback(0, 0)

        # ── main loop: feed discovery → batch → workers ───────────────
        batch: List[Path] = []
        done_discovery = False

        while True:
            # Phase 1: pull from discovery queue (non-blocking-ish)
            while not done_discovery:
                try:
                    p = discovery_queue.get(timeout=0.05)
                except queue.Empty:
                    if not discoverer.is_alive():
                        done_discovery = True
                    break
                if p is None:
                    done_discovery = True
                    break
                batch.append(p)
                total_discovered += 1
                if len(batch) >= _EXIFTOOL_BATCH:
                    break

            # Phase 2: submit ready batch to workers
            if len(batch) >= _EXIFTOOL_BATCH or (done_discovery and batch):
                work_queue.put(list(batch))
                batch = []

            # Phase 3: drain result queue → yield rows
            drained = 0
            while True:
                try:
                    row = result_queue.get_nowait()
                    yield row
                    total_yielded += 1
                    drained += 1
                except queue.Empty:
                    break

            if progress_callback and drained > 0:
                progress_callback(total_yielded, discoverer.total_found)

            # Phase 4: check if we're done
            if done_discovery and not batch:
                if work_queue.empty():
                    # All work has been submitted and picked up.
                    # Send one sentinel per worker so each exits cleanly.
                    for _ in range(num_workers):
                        work_queue.put(None)
                    for w in workers:
                        w.join(timeout=5.0)
                    # Final drain
                    while True:
                        try:
                            row = result_queue.get_nowait()
                            yield row
                            total_yielded += 1
                        except queue.Empty:
                            break
                    break

    finally:
        # Signal workers to stop
        stop_event.set()
        for _ in range(num_workers):
            work_queue.put(None)

        # Wait for workers to finish
        for w in workers:
            w.join(timeout=2.0)

        # Cleanup discovery thread
        discoverer.stop()
        while True:
            try:
                discovery_queue.get(timeout=0.1)
            except queue.Empty:
                if not discoverer.is_alive():
                    break
        discoverer.join(timeout=1.0)

        elapsed = time.monotonic() - scan_start
        worker_stats = {w.name: w.processed for w in workers}
        if worker_stats:
            _scan_logger.info("── Per-worker distribution ──")
            for name, count in sorted(worker_stats.items()):
                _scan_logger.info("  %-20s : %4d files", name, count)
        _scan_logger.info(
            "◀ Parallel scan complete: %d discovered → %d yielded, %.2fs total",
            total_discovered,
            total_yielded,
            elapsed,
        )
