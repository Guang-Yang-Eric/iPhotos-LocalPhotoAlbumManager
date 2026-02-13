"""Cache path and file lifecycle helpers for thumbnails."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from PySide6.QtCore import QSize

from ....config import WORK_DIR_NAME


def safe_unlink(path: Path) -> None:
    """Safely delete a file, handling permission errors gracefully."""

    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        try:
            path.rename(path.with_suffix(path.suffix + ".stale"))
        except OSError:
            pass
    except OSError:
        pass


def stat_mtime_ns(stat_result: os.stat_result) -> int:
    stamp = getattr(stat_result, "st_mtime_ns", None)
    if stamp is None:
        stamp = int(stat_result.st_mtime * 1_000_000_000)
    return int(stamp)


def generate_cache_path(library_root: Path, abs_path: Path, size: QSize, stamp: int) -> Path:
    """Generate the file path for a cached thumbnail image."""

    path_str = str(abs_path.resolve())
    digest = hashlib.blake2b(path_str.encode("utf-8"), digest_size=20).hexdigest()
    filename = f"{digest}_{stamp}_{size.width()}x{size.height()}.png"
    return library_root / WORK_DIR_NAME / "thumbs" / filename

