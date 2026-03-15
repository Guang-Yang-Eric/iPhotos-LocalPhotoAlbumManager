"""iPhoto scan phase — C native acceleration module.

The shared library (scan_utils.c) is JIT-compiled on first import
(requires gcc/cc and libxxhash).  When compilation fails, all public
functions fall back transparently to pure-Python implementations.

Exported functions::

    parse_dt_fast              ISO 8601 string -> Unix µs timestamp (int | None)
    compute_file_id_fast       file path -> 128-bit XXH3 hex digest (str | None)
    should_include_fast        (rel_path, include_globs, exclude_globs) -> bool
    discover_files_fast        root_dir -> list[Path] | None   (P4)
    parse_dt_full_fast         ISO 8601 str -> (unix_us, year, month) | None (P5)
    normalise_content_id_fast  content-ID str -> normalised str | None (P6)

A startup log line (at DEBUG level) indicates whether the C extension was
loaded successfully or whether the module is operating in Python-fallback
mode.  Call-sites log at DEBUG level when they first switch to or from the
C path so individual hotspots can be traced.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import threading

_logger = logging.getLogger(__name__)

_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_C_AVAILABLE = False

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "scan_utils.c")

if sys.platform == "win32":
    _OUT = os.path.join(_DIR, "_scan_utils.dll")
elif sys.platform == "darwin":
    _OUT = os.path.join(_DIR, "_scan_utils.dylib")
else:
    _OUT = os.path.join(_DIR, "_scan_utils.so")

_INT64_MIN = -(2**63)


def _compile() -> None:
    global _lib, _C_AVAILABLE

    if sys.platform == "win32":
        # mmap / timegm / nftw are not available on Windows; skip compilation.
        return

    import shutil

    gcc = shutil.which("gcc") or shutil.which("cc")
    if not gcc:
        return

    if not os.path.isfile(_SRC):
        return

    common_flags = ["-shared", "-fPIC", "-o", _OUT, _SRC, "-lxxhash"]
    compiled = False
    for opt in (["-O3", "-march=native"], ["-O2"]):
        try:
            subprocess.run(
                [gcc] + opt + common_flags,
                check=True,
                capture_output=True,
                timeout=60,
            )
            compiled = True
            break
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired):
            continue

    if not compiled:
        return

    try:
        _load_lib(_OUT)
    except OSError:
        pass


def _load_lib(path: str) -> None:
    """Load the shared library and register all function signatures."""
    global _lib, _C_AVAILABLE

    lib = ctypes.CDLL(path)

    # P1: parse_iso8601_to_unix_us
    lib.parse_iso8601_to_unix_us.restype = ctypes.c_int64
    lib.parse_iso8601_to_unix_us.argtypes = [ctypes.c_char_p]

    # P2: compute_file_id_c
    lib.compute_file_id_c.restype = ctypes.c_int
    lib.compute_file_id_c.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

    # P3: should_include_c
    lib.should_include_c.restype = ctypes.c_int
    lib.should_include_c.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]

    # P4: discover_files_c
    lib.discover_files_c.restype = None
    lib.discover_files_c.argtypes = [
        ctypes.c_char_p,               # root_dir
        ctypes.c_void_p,               # callback (FileFoundCallback)
        ctypes.c_void_p,               # userdata
    ]

    # P5: parse_iso8601_full_c
    lib.parse_iso8601_full_c.restype = ctypes.c_int
    lib.parse_iso8601_full_c.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]

    # P6: normalise_content_id_c
    lib.normalise_content_id_c.restype = ctypes.c_int
    lib.normalise_content_id_c.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
    ]

    _lib = lib
    _C_AVAILABLE = True


# ---------------------------------------------------------------------------
# JIT compile once at import time (if .so not already present)
# ---------------------------------------------------------------------------

with _lock:
    if not _C_AVAILABLE and os.path.isfile(_SRC):
        if not os.path.isfile(_OUT):
            _compile()
        else:
            # .so already exists — just load it; recompile on failure.
            try:
                _load_lib(_OUT)
            except OSError:
                _compile()

if _C_AVAILABLE:
    _logger.debug(
        "iPhoto._native: C extension loaded (%s) — P1–P6 scan hotspots accelerated",
        _OUT,
    )
else:
    _logger.debug(
        "iPhoto._native: C extension unavailable "
        "(gcc/libxxhash missing or compile failed) — "
        "all scan hotspots using Python fallbacks",
    )


# =====================================================================
# ctypes helper
# =====================================================================

# P4 callback type — must be kept alive while discover_files_c is running.
_FileFoundCbType = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


def _build_c_str_array(patterns: list[str]) -> ctypes.Array:
    """Build a NULL-terminated ctypes c_char_p array from *patterns*."""
    encoded = [p.encode() for p in patterns] + [None]
    return (ctypes.c_char_p * len(encoded))(*encoded)


# =====================================================================
# Pure-Python fallback implementations (no dependency on C extension)
# =====================================================================

def _py_compute_file_id(path_str: str) -> str | None:
    """Pure-Python XXH3-128 file hash — identical algorithm to the C version."""
    try:
        import xxhash
        import os as _os

        threshold = 2 * 1024 * 1024

        with open(path_str, "rb") as f:
            file_size = _os.fstat(f.fileno()).st_size

            if file_size <= threshold:
                hasher = xxhash.xxh3_128()
                chunk_size = 1024 * 1024
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
                return hasher.hexdigest()

            # Large file: sample head + middle + tail
            hasher = xxhash.xxh3_128()
            hasher.update(file_size.to_bytes(8, "little"))

            chunk_size = 256 * 1024
            hasher.update(f.read(chunk_size))

            if file_size > chunk_size * 2:
                f.seek(file_size // 2 - chunk_size // 2)
                hasher.update(f.read(chunk_size))

            if file_size > chunk_size:
                f.seek(max(0, file_size - chunk_size))
                hasher.update(f.read(chunk_size))

        return hasher.hexdigest()
    except OSError:
        return None


def _py_parse_dt_full(value: str) -> tuple[int, int, int] | None:
    """Pure-Python fallback for parse_dt_full_fast."""
    from dateutil import parser as _dp
    import math
    try:
        dt = _dp.isoparse(value)
        us = math.floor(dt.timestamp() * 1_000_000)
        return (us, dt.year, dt.month)
    except (ValueError, TypeError):
        return None


# =====================================================================
# Public API
# =====================================================================

def parse_dt_fast(value: str | None) -> int | None:
    """Parse an ISO 8601 string to a Unix microsecond timestamp.

    Returns an ``int`` (Unix µs since epoch) on success, ``None`` on
    failure or empty input.  The C implementation is ~10–50× faster than
    :func:`dateutil.parser.isoparse`.
    """
    if not value:
        return None

    if _C_AVAILABLE and _lib is not None:
        result = _lib.parse_iso8601_to_unix_us(value.encode())
        return None if result == _INT64_MIN else int(result)

    from dateutil import parser as _dp
    try:
        dt = _dp.isoparse(value)
        import math
        return math.floor(dt.timestamp() * 1_000_000)
    except (ValueError, TypeError):
        return None


def compute_file_id_fast(path: object) -> str | None:
    """Return a 128-bit XXH3 hex digest for the file at *path*.

    Uses ``mmap`` + ``pread`` in C for best performance.  Falls back to
    a pure-Python implementation when the C extension is unavailable.

    Returns ``None`` on I/O error.
    """
    path_str = str(path)

    if _C_AVAILABLE and _lib is not None:
        out = ctypes.create_string_buffer(33)
        rc = _lib.compute_file_id_c(path_str.encode(), out)
        if rc == 0:
            return out.value.decode()
        return None

    return _py_compute_file_id(path_str)


def should_include_fast(
    rel_path: str,
    include_globs: list[str],
    exclude_globs: list[str],
) -> bool:
    """Return ``True`` if *rel_path* should be included in the scan.

    *include_globs* and *exclude_globs* must already be brace-expanded
    (each entry is a plain glob pattern without ``{a,b}`` syntax).

    Uses C ``fnmatch(3)`` for best performance.  Falls back to Python
    :func:`fnmatch.fnmatch` when the C extension is unavailable.
    """
    if _C_AVAILABLE and _lib is not None:
        inc_arr = _build_c_str_array(include_globs)
        exc_arr = _build_c_str_array(exclude_globs)
        return bool(_lib.should_include_c(
            rel_path.encode(), inc_arr, exc_arr
        ))

    import fnmatch as _fnm

    for pat in exclude_globs:
        if _fnm.fnmatch(rel_path, pat):
            return False
        if pat.startswith("**/") and _fnm.fnmatch(rel_path, pat[3:]):
            return False

    for pat in include_globs:
        if _fnm.fnmatch(rel_path, pat):
            return True
        if pat.startswith("**/") and _fnm.fnmatch(rel_path, pat[3:]):
            return True

    return False


def discover_files_fast(root: object) -> list | None:
    """Walk *root* and return a list of :class:`~pathlib.Path` objects for all
    supported media files (P4).

    Returns ``None`` to signal that the caller should use its own Python
    fallback (e.g. when the C extension is unavailable or on Windows).

    Uses POSIX ``nftw`` for efficient recursive traversal with hidden-
    directory pruning, bypassing the Python generator/recursion overhead.
    """
    if not _C_AVAILABLE or _lib is None:
        return None

    from pathlib import Path as _Path

    found: list = []

    @_FileFoundCbType
    def _cb(path_bytes: bytes, _userdata: object) -> None:  # type: ignore[misc]
        if path_bytes:
            found.append(_Path(path_bytes.decode(errors="replace")))

    _lib.discover_files_c(str(root).encode(), _cb, None)
    return found


def parse_dt_full_fast(value: str | None) -> tuple[int, int, int] | None:
    """Parse an ISO 8601 string and return ``(unix_us, year, month)`` (P5).

    This avoids the double-parse in :func:`~iPhoto.infrastructure.services.\
metadata_provider.ExifToolMetadataProvider.normalize_metadata` where the
    same timestamp string is parsed once for ``ts`` and again for
    ``year``/``month``.

    Returns ``None`` on failure or empty input.
    """
    if not value:
        return None

    if _C_AVAILABLE and _lib is not None:
        out_us    = ctypes.c_int64()
        out_year  = ctypes.c_int()
        out_month = ctypes.c_int()
        rc = _lib.parse_iso8601_full_c(
            value.encode(),
            ctypes.byref(out_us),
            ctypes.byref(out_year),
            ctypes.byref(out_month),
        )
        if rc == 0:
            return (int(out_us.value), int(out_year.value), int(out_month.value))
        return None

    return _py_parse_dt_full(value)


def normalise_content_id_fast(value: object) -> str | None:
    """Return a normalised Live Photo content identifier (P6).

    Strips leading/trailing whitespace and folds ASCII to lower-case.
    Returns ``None`` when *value* is not a non-empty string.

    The C implementation avoids the two heap allocations that
    ``str.strip()`` + ``str.casefold()`` would produce in Python.
    """
    if not isinstance(value, str):
        return None

    if _C_AVAILABLE and _lib is not None:
        # Allocate an output buffer the same size as the input — result can
        # only be equal in length or shorter after stripping.
        out_size = len(value.encode()) + 1
        out = ctypes.create_string_buffer(out_size)
        rc = _lib.normalise_content_id_c(value.encode(), out, out_size)
        if rc > 0:
            return out.value.decode()
        if rc == 0:
            return None  # empty after stripping
        return None  # error

    trimmed = value.strip()
    return trimmed.casefold() if trimmed else None
