"""Media type classification helpers shared by UI models."""

from __future__ import annotations

import mimetypes
from enum import IntEnum
from pathlib import Path
from typing import Mapping, Tuple

RAW_EXTENSIONS: frozenset[str] = frozenset({
    ".cr2",
    ".cr3",
    ".nef",
    ".nrw",
    ".arw",
    ".srf",
    ".sr2",
    ".orf",
    ".rw2",
    ".raf",
    ".dng",
    ".pef",
    ".raw",
    ".rwl",
    ".3fr",
    ".iiq",
    ".x3f",
    ".srw",
    ".erf",
})

# Register MIME types for RAW extensions that Python's mimetypes module
# does not know about.  This ensures ``mimetypes.guess_type()`` returns a
# useful ``image/`` type for these files rather than ``None``.
# Module-level registration is idempotent and runs once per process.
_RAW_MIME_MAP = {
    ".cr2": "image/x-canon-cr2",
    ".cr3": "image/x-canon-cr3",
    ".nef": "image/x-nikon-nef",
    ".nrw": "image/x-nikon-nrw",
    ".arw": "image/x-sony-arw",
    ".srf": "image/x-sony-srf",
    ".sr2": "image/x-sony-sr2",
    ".orf": "image/x-olympus-orf",
    ".rw2": "image/x-panasonic-rw2",
    ".raf": "image/x-fuji-raf",
    ".dng": "image/x-adobe-dng",
    ".pef": "image/x-pentax-pef",
    ".raw": "image/x-raw",
    ".rwl": "image/x-leica-rwl",
    ".3fr": "image/x-hasselblad-3fr",
    ".iiq": "image/x-phaseone-iiq",
    ".x3f": "image/x-sigma-x3f",
    ".srw": "image/x-samsung-srw",
    ".erf": "image/x-epson-erf",
}
for _ext, _mime in _RAW_MIME_MAP.items():
    mimetypes.add_type(_mime, _ext)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".heif",
    ".heifs",
    ".heicf",
}) | RAW_EXTENSIONS

VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mov",
    ".mp4",
    ".m4v",
    ".qt",
    ".avi",
    ".wmv",
    ".mkv",
})


def _normalise_mime(value: object) -> str:
    """Return a lower-case MIME type string or an empty string."""

    if isinstance(value, str):
        return value.strip().lower()
    return ""


def _suffix_from_row(row: Mapping[str, object]) -> str:
    """Extract a normalised file suffix from *row* if available."""

    rel = row.get("rel")
    if isinstance(rel, Path):
        return rel.suffix.lower()
    if isinstance(rel, str):
        return Path(rel).suffix.lower()
    return ""


def classify_media(row: Mapping[str, object]) -> Tuple[bool, bool]:
    """Return booleans indicating whether *row* describes an image or video.

    The function inspects MIME types, legacy ``type`` fields, and file
    extensions in order of preference. Additional video formats beyond the
    default MP4/MOV set are supported to handle albums with mixed footage.
    """

    mime = _normalise_mime(row.get("mime"))

    # If the MIME type implies an image but the extension is unambiguously video
    # (e.g. .mov), trust the extension. This protects against system registries
    # that misreport QuickTime container files as images.
    suffix = _suffix_from_row(row)
    if mime.startswith("image/") and suffix in VIDEO_EXTENSIONS:
        return False, True

    if mime.startswith("image/"):
        return True, False
    if mime.startswith("video/"):
        return False, True

    legacy_kind = row.get("type")
    if isinstance(legacy_kind, str):
        kind = legacy_kind.strip().lower()
        if kind == "image":
            return True, False
        if kind == "video":
            return False, True

    if suffix in IMAGE_EXTENSIONS:
        return True, False
    if suffix in VIDEO_EXTENSIONS:
        return False, True
    return False, False


class MediaType(IntEnum):
    IMAGE = 1
    VIDEO = 2
    UNKNOWN = 0


def get_media_type(path: Path) -> MediaType:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    if suffix in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    return MediaType.UNKNOWN


__all__ = ["classify_media", "get_media_type", "MediaType", "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "RAW_EXTENSIONS"]
