"""iPhoto scan phase — C native acceleration module.

The shared library (scan_utils.c) is JIT-compiled on first import
(requires gcc/cc and libxxhash).  When compilation fails, all three
public functions fall back transparently to pure-Python implementations.

Exported functions::

    parse_dt_fast         ISO 8601 string -> Unix microsecond timestamp (int | None)
    compute_file_id_fast  file path -> 128-bit XXH3 hex digest (str | None)
    should_include_fast   (rel_path, include_globs, exclude_globs) -> bool
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading

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
        # mmap / timegm are not available on Windows; skip compilation.
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
        lib = ctypes.CDLL(_OUT)

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

        _lib = lib
        _C_AVAILABLE = True
    except OSError:
        pass


# JIT compile once at import time (if .so not already present)
with _lock:
    if not _C_AVAILABLE and os.path.isfile(_SRC):
        if not os.path.isfile(_OUT):
            _compile()
        else:
            # .so already exists — just load it
            try:
                lib = ctypes.CDLL(_OUT)
                lib.parse_iso8601_to_unix_us.restype = ctypes.c_int64
                lib.parse_iso8601_to_unix_us.argtypes = [ctypes.c_char_p]
                lib.compute_file_id_c.restype = ctypes.c_int
                lib.compute_file_id_c.argtypes = [ctypes.c_char_p,
                                                   ctypes.c_char_p]
                lib.should_include_c.restype = ctypes.c_int
                lib.should_include_c.argtypes = [
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_char_p),
                    ctypes.POINTER(ctypes.c_char_p),
                ]
                _lib = lib
                _C_AVAILABLE = True
            except OSError:
                _compile()


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

    # Python fallback
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
    the pure-Python :func:`~iPhoto.utils.hashutils.compute_file_id`
    implementation when the C extension is unavailable.

    Returns ``None`` on I/O error.
    """
    path_str = str(path)

    if _C_AVAILABLE and _lib is not None:
        out = ctypes.create_string_buffer(33)
        rc = _lib.compute_file_id_c(path_str.encode(), out)
        if rc == 0:
            return out.value.decode()
        return None

    # Python fallback
    from iPhoto.utils.hashutils import compute_file_id
    from pathlib import Path
    try:
        return compute_file_id(Path(path_str))
    except OSError:
        return None


def _build_c_str_array(patterns: list[str]) -> ctypes.Array:
    """Build a NULL-terminated ctypes c_char_p array from *patterns*."""
    encoded = [p.encode() for p in patterns] + [None]
    return (ctypes.c_char_p * len(encoded))(*encoded)


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

    # Python fallback
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
