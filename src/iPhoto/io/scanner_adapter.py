"""Adapter to bridge legacy scanner calls to the new infrastructure.

The :func:`scan_album` generator discovers media files in a background thread
and processes metadata + thumbnails in parallel via a :class:`ThreadPoolExecutor`.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

_scan_logger = logging.getLogger("iPhoto.scanner")

# Default number of parallel worker threads for batch processing.
_NUM_WORKERS = 4

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

def scan_album(
    root: Path,
    include_globs: Iterable[str],
    exclude_globs: Iterable[str],
    existing_index: Optional[Dict[str, Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    num_workers: int = _NUM_WORKERS,
) -> Iterator[Dict[str, Any]]:
    """Yield index rows for all matching assets in *root*, scanning in parallel.

    File discovery runs in a background thread.  Discovered files are grouped
    into batches of *BATCH_SIZE* and submitted to a :class:`ThreadPoolExecutor`
    with *num_workers* threads so that metadata extraction and thumbnail
    generation happen concurrently.  Results are yielded to the caller as
    each batch completes.
    """

    BATCH_SIZE = 50
    scan_start = time.monotonic()

    _scan_logger.info(
        "▶ Parallel scan started: %s (workers=%d, batch_size=%d)",
        root.name, num_workers, BATCH_SIZE,
    )

    path_queue: queue.Queue = queue.Queue(maxsize=1000)
    discoverer = FileDiscoveryThread(
        root, path_queue,
        include=list(include_globs),
        exclude=list(exclude_globs),
    )
    discoverer.start()

    total_yielded = 0
    batches_submitted = 0
    thread_stats: Dict[str, int] = {}
    _stats_lock = threading.Lock()

    def _process_batch(paths: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of files on a worker thread. Returns list of rows."""
        t_name = threading.current_thread().name
        t_start = time.monotonic()
        results: List[Dict[str, Any]] = []
        cached_count = 0

        to_process: List[Path] = []
        for p in paths:
            rel = p.relative_to(root).as_posix()
            hit = None
            if existing_index:
                hit = existing_index.get(rel)
                if not hit:
                    hit = existing_index.get(unicodedata.normalize('NFC', rel))
            if hit:
                try:
                    st = p.stat()
                    cached_ts = hit.get("ts")
                    current_ts = int(st.st_mtime * 1_000_000)
                    if (hit.get("bytes") == st.st_size
                            and abs((cached_ts or 0) - current_ts) <= 1_000_000):
                        results.append(hit)
                        cached_count += 1
                        continue
                except OSError:
                    pass
            to_process.append(p)

        if to_process:
            for row in process_media_paths(root, to_process, []):
                results.append(row)

        dt = time.monotonic() - t_start
        with _stats_lock:
            thread_stats[t_name] = thread_stats.get(t_name, 0) + len(paths)
        _scan_logger.info(
            "  [%s] batch: %d files (%d cached + %d new) → %.3fs",
            t_name, len(paths), cached_count, len(to_process), dt,
        )
        return results

    executor = ThreadPoolExecutor(
        max_workers=num_workers, thread_name_prefix="ScanWorker",
    )
    pending: Dict[Any, int] = {}   # future → batch file count

    try:
        if progress_callback:
            progress_callback(0, 0)

        batch: List[Path] = []
        done_discovery = False
        total_processed = 0

        while True:
            # ── Phase 1: collect files from discovery queue ──
            while not done_discovery:
                try:
                    p = path_queue.get(timeout=0.1)
                except queue.Empty:
                    if not discoverer.is_alive():
                        done_discovery = True
                    break
                if p is None:
                    done_discovery = True
                    break
                batch.append(p)
                if len(batch) >= BATCH_SIZE:
                    break

            # ── Phase 2: submit ready batch to thread pool ──
            if len(batch) >= BATCH_SIZE or (done_discovery and batch):
                fut = executor.submit(_process_batch, list(batch))
                pending[fut] = len(batch)
                batches_submitted += 1
                batch = []

            # ── Phase 3: yield completed results ──
            completed = [f for f in pending if f.done()]
            for f in completed:
                n = pending.pop(f)
                try:
                    for row in f.result():
                        yield row
                        total_yielded += 1
                except Exception as exc:
                    _scan_logger.error("Batch processing failed: %s", exc)
                total_processed += n
                if progress_callback:
                    progress_callback(total_processed, discoverer.total_found)

            # ── exit when all work is done ──
            if done_discovery and not pending and not batch:
                break

    finally:
        executor.shutdown(wait=False, cancel_futures=True)

        discoverer.stop()
        # Drain queue so discovery thread can unblock from put()
        while True:
            try:
                path_queue.get(timeout=0.1)
            except queue.Empty:
                if not discoverer.is_alive():
                    break
        discoverer.join(timeout=1.0)

        elapsed = time.monotonic() - scan_start
        if thread_stats:
            _scan_logger.info("── Per-thread work distribution ──")
            for tname, count in sorted(thread_stats.items()):
                _scan_logger.info("  %-20s : %4d files", tname, count)
        _scan_logger.info(
            "◀ Parallel scan complete: %d discovered → %d yielded, "
            "%d batches, %.2fs total",
            discoverer.total_found, total_yielded, batches_submitted, elapsed,
        )
