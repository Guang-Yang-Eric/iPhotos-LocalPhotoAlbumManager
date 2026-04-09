import mimetypes
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from iPhoto.application.interfaces import IMetadataProvider
from iPhoto.utils.exiftool import get_metadata_batch
from iPhoto.io.metadata import read_image_meta_with_exiftool, read_video_meta
from iPhoto.utils.hashutils import compute_file_id
from iPhoto.domain.models import MediaType

logger = logging.getLogger(__name__)

# P5: C-accelerated ISO 8601 parsing that also returns year/month in one call,
# eliminating the double-parse in normalize_metadata.
try:
    from iPhoto._native import parse_dt_full_fast as _parse_dt_full_fast  # type: ignore[attr-defined]
    _PARSE_FULL_C = True
    logger.debug(
        "metadata_provider: C extension available for timestamp parsing (P5)"
    )
except Exception:
    _parse_dt_full_fast = None  # type: ignore[assignment]
    _PARSE_FULL_C = False
    logger.debug(
        "metadata_provider: C extension unavailable for timestamp parsing (P5) "
        "— using Python fallback"
    )

class ExifToolMetadataProvider(IMetadataProvider):
    _IMAGE_EXTENSIONS = {".heic", ".heif", ".heifs", ".heicf", ".jpg", ".jpeg", ".png", ".webp"}
    _VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".qt", ".avi", ".mkv"}

    def get_metadata_batch(self, paths: List[Path]) -> List[Dict[str, Any]]:
        try:
            return get_metadata_batch(paths)
        except Exception as e:
            logger.error(f"Failed to get metadata batch: {e}")
            return []

    def normalize_metadata(self, root: Path, file_path: Path, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize raw metadata using logic similar to the legacy scanner.
        """
        stat = file_path.stat()
        rel = file_path.relative_to(root).as_posix()

        # Base row structure
        row = {
            "rel": rel,
            "bytes": stat.st_size,
            "dt": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "ts": int(stat.st_mtime * 1_000_000),
            "id": f"as_{compute_file_id(file_path)}",
            "mime": mimetypes.guess_type(file_path.name)[0],
        }

        # Apply ExifTool metadata if available
        # Note: raw_metadata is the raw JSON from ExifTool.
        # The legacy code had helper functions `read_image_meta_with_exiftool` and `read_video_meta`
        # that did further processing (extracting GPS, rotation, etc.)
        # We should reuse or reimplement those helpers to ensure data compatibility.

        suffix = file_path.suffix.lower()
        processed_meta = {}

        if suffix in self._IMAGE_EXTENSIONS:
            # We pass raw_metadata as override to avoid calling exiftool again
            processed_meta = read_image_meta_with_exiftool(file_path, raw_metadata)
        elif suffix in self._VIDEO_EXTENSIONS:
            processed_meta = read_video_meta(file_path, raw_metadata)

        # Merge processed metadata into row
        for key, value in processed_meta.items():
            if value is not None:
                row[key] = value

        # Fixup 'dt'/'ts'/'year'/'month': use parse_dt_full_fast (P5) to parse
        # the best available dt string once, deriving ts, year, and month in a
        # single C call instead of calling fromisoformat() twice.
        dt_str_for_full: str | None = None
        if "dt" in processed_meta and isinstance(processed_meta["dt"], str):
            dt_str_for_full = processed_meta["dt"]
        elif "dt" in row and isinstance(row["dt"], str):
            dt_str_for_full = row["dt"]

        if dt_str_for_full is not None:
            # Normalise the "Z" suffix that Python's fromisoformat requires.
            dt_str_norm = dt_str_for_full.replace("Z", "+00:00")

            # P5: single-call parse returning (unix_us, year, month)
            full_result = (
                _parse_dt_full_fast(dt_str_norm)
                if _PARSE_FULL_C and _parse_dt_full_fast is not None
                else None
            )

            if full_result is not None:
                unix_us, year, month = full_result
                if "dt" in processed_meta and isinstance(processed_meta["dt"], str):
                    row["ts"] = unix_us
                row["year"] = year
                row["month"] = month
            else:
                # Python fallback — retain original two-parse logic
                if "dt" in processed_meta and isinstance(processed_meta["dt"], str):
                    try:
                        dt_obj = datetime.fromisoformat(dt_str_norm)
                        row["ts"] = int(dt_obj.timestamp() * 1_000_000)
                    except (ValueError, TypeError):
                        pass

                if "dt" in row and isinstance(row["dt"], str):
                    try:
                        fallback_dt_str = row["dt"].replace("Z", "+00:00")
                        fallback_dt_obj = datetime.fromisoformat(fallback_dt_str)
                        row["year"] = fallback_dt_obj.year
                        row["month"] = fallback_dt_obj.month
                    except (ValueError, TypeError):
                        row["year"] = None
                        row["month"] = None
        else:
            row["year"] = None
            row["month"] = None

        # Aspect Ratio
        w = row.get("w")
        h = row.get("h")
        if isinstance(w, (int, float)) and isinstance(h, (int, float)) and h > 0:
            row["aspect_ratio"] = float(w) / float(h)
        else:
            row["aspect_ratio"] = None

        # Media Type
        if suffix in self._VIDEO_EXTENSIONS:
            row["media_type"] = 1 # MediaType.VIDEO
        elif suffix in self._IMAGE_EXTENSIONS:
            row["media_type"] = 0 # MediaType.IMAGE
        else:
            mime = row.get("mime", "")
            if mime and mime.startswith("video/"):
                row["media_type"] = 1
            elif mime and mime.startswith("image/"):
                row["media_type"] = 0
            else:
                row["media_type"] = None

        return row
