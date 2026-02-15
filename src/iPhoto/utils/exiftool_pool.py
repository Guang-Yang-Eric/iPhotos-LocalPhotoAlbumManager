"""Pool of persistent ``exiftool -stay_open`` processes.

Each :class:`PersistentExifTool` keeps a single ``exiftool`` child process
running in ``-stay_open True`` mode.  Batches of file paths are sent through
stdin and the resulting JSON is read from stdout — **no** per-batch subprocess
spawn overhead.

:class:`ExifToolPool` manages *N* such processes and exposes a simple
:meth:`get` / :meth:`put` API for borrowing an instance from the pool.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from ..errors import ExternalToolError

_logger = logging.getLogger(__name__)

# Sentinel written by exiftool after each batch when using ``-stay_open``
# (``-execute`` causes it to print ``{ready<N>}``).  We read lines until this
# marker appears.
_READY_SENTINEL = b"{ready}"


class PersistentExifTool:
    """A single long-running ``exiftool -stay_open True`` child process.

    Thread-safety: **not** thread-safe.  Each instance should only be used by
    one thread at a time.  :class:`ExifToolPool` enforces this by lending one
    instance per worker thread.
    """

    def __init__(self) -> None:
        self._executable: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the underlying ``exiftool`` process (idempotent)."""
        if self._process is not None and self._process.poll() is None:
            return  # already running

        if self._executable is None:
            self._executable = shutil.which("exiftool")
            if self._executable is None:
                raise ExternalToolError(
                    "exiftool executable not found.  Install it from "
                    "https://exiftool.org/ and ensure it is available on PATH."
                )

        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self._process = subprocess.Popen(
            [
                self._executable,
                "-stay_open",
                "True",
                "-@",          # read arguments from stdin
                "-",           # stdin
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )

    def stop(self) -> None:
        """Gracefully terminate the ``exiftool`` process."""
        proc = self._process
        if proc is None:
            return
        self._process = None
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write(b"-stay_open\nFalse\n")
                proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ── query ─────────────────────────────────────────────────────────

    def get_metadata_batch(self, paths: List[Path]) -> List[Dict[str, Any]]:
        """Send a batch of file paths and return the parsed JSON metadata.

        The instance **must** already be :meth:`start`-ed.  If the underlying
        process has died it is restarted transparently.
        """
        if not paths:
            return []

        # Auto-restart if the process died
        if not self.alive:
            self.start()

        proc = self._process
        assert proc is not None and proc.stdin is not None and proc.stdout is not None

        # Build the argument block for this batch:
        #   -n -g1 -json -charset filename=utf8
        #   <file1>
        #   <file2>
        #   ...
        #   -execute
        lines: List[str] = [
            "-n",
            "-g1",
            "-json",
            "-charset",
            "filename=utf8",
        ]
        for p in paths:
            safe = p.absolute().as_posix()
            if "\n" in safe or "\r" in safe:
                _logger.warning("Skipping path with newlines: %r", safe)
                continue
            lines.append(safe)
        lines.append("-execute\n")

        cmd_bytes = ("\n".join(lines)).encode("utf-8")

        try:
            proc.stdin.write(cmd_bytes)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            # Process died mid-write — restart and retry once
            self.start()
            proc = self._process
            assert proc is not None and proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(cmd_bytes)
            proc.stdin.flush()

        # Read stdout until we see the ``{ready}`` sentinel
        output_chunks: List[bytes] = []
        while True:
            line = proc.stdout.readline()
            if not line:
                # EOF — process died
                break
            if line.strip().startswith(_READY_SENTINEL):
                break
            output_chunks.append(line)

        raw_output = b"".join(output_chunks).decode("utf-8", errors="replace").strip()
        if not raw_output:
            return []

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError as exc:
            _logger.error("Failed to parse exiftool JSON: %s — raw: %.200s", exc, raw_output)
            return []


class ExifToolPool:
    """Fixed-size pool of :class:`PersistentExifTool` instances.

    Usage::

        pool = ExifToolPool(size=4)
        pool.start()

        et = pool.get()        # borrow
        try:
            result = et.get_metadata_batch(paths)
        finally:
            pool.put(et)       # return

        pool.shutdown()
    """

    def __init__(self, size: int = 4) -> None:
        self._size = size
        self._tools: List[PersistentExifTool] = [PersistentExifTool() for _ in range(size)]
        self._queue: Queue[PersistentExifTool] = Queue()
        self._started = False
        self._atexit_registered = False

    @property
    def size(self) -> int:
        return self._size

    def start(self) -> None:
        """Start all ``exiftool`` processes and populate the pool queue."""
        if self._started:
            return
        self._started = True
        for tool in self._tools:
            try:
                tool.start()
            except ExternalToolError:
                _logger.warning("Failed to start an exiftool instance")
            self._queue.put(tool)
        if not self._atexit_registered:
            self._atexit_registered = True
            atexit.register(self.shutdown)

    def get(self, timeout: float = 30.0) -> PersistentExifTool:
        """Borrow an exiftool instance (blocks until one is available)."""
        return self._queue.get(timeout=timeout)

    def put(self, tool: PersistentExifTool) -> None:
        """Return an exiftool instance to the pool."""
        self._queue.put(tool)

    def shutdown(self) -> None:
        """Stop all exiftool processes."""
        if not self._started:
            return
        self._started = False
        # Drain the queue and stop each tool
        for tool in self._tools:
            tool.stop()


# ── module-level singleton ────────────────────────────────────────────

_global_pool: Optional[ExifToolPool] = None
_pool_lock = threading.Lock()


def get_exiftool_pool(size: int = 4) -> ExifToolPool:
    """Return (and lazily create) the process-wide :class:`ExifToolPool`."""
    global _global_pool
    if _global_pool is not None and _global_pool._started:
        return _global_pool
    with _pool_lock:
        if _global_pool is None or not _global_pool._started:
            _global_pool = ExifToolPool(size=size)
            _global_pool.start()
    return _global_pool


def shutdown_exiftool_pool() -> None:
    """Shut down the global pool (called during app shutdown)."""
    global _global_pool
    with _pool_lock:
        if _global_pool is not None:
            _global_pool.shutdown()
            _global_pool = None
