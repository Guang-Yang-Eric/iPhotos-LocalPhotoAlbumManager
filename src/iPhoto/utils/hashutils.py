"""Hashing utilities."""

from __future__ import annotations

from pathlib import Path
import os

import xxhash

try:
    from iPhoto._native import compute_file_id_fast as _compute_file_id_fast
except Exception:
    _compute_file_id_fast = None  # type: ignore[assignment]


def file_xxh3(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the XXH3 128-bit hash of *path*."""

    hasher = xxhash.xxh3_128()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_file_id(path: Path) -> str:
    """
    Return a hash of the file content, optimized for speed.
    For small files (< 2MB), hashes the entire content using XXH3.
    For large files, hashes a sample of the content (Head/Mid/Tail) + Size.

    Uses a C extension (mmap + xxhash) when available for best performance.
    Falls back to pure Python when the C extension is unavailable.
    """
    if _compute_file_id_fast is not None:
        result = _compute_file_id_fast(path)
        if result is not None:
            return result
        # Fall through to Python implementation on C-side I/O error

    threshold = 2 * 1024 * 1024  # 2 MB

    with path.open("rb") as f:
        # Use fstat to get size of the opened file handle to avoid TOCTOU race conditions
        try:
            file_size = os.fstat(f.fileno()).st_size
        except OSError:
            # Fallback if fstat fails (unlikely)
            file_size = path.stat().st_size

        if file_size <= threshold:
            # For small files, hash the whole content
            hasher = xxhash.xxh3_128()
            chunk_size = 1024 * 1024
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
            return hasher.hexdigest()

        # For large files, use partial hashing
        hasher = xxhash.xxh3_128()
        # Mix in the size
        hasher.update(file_size.to_bytes(8, "little"))

        chunk_size = 256 * 1024  # 256KB

        # Head
        hasher.update(f.read(chunk_size))

        # Middle
        if file_size > chunk_size * 2:
            f.seek(file_size // 2 - chunk_size // 2)
            hasher.update(f.read(chunk_size))

        # Tail
        if file_size > chunk_size:
            f.seek(max(0, file_size - chunk_size))
            hasher.update(f.read(chunk_size))

    return hasher.hexdigest()
