"""Tests for legacy ``ThumbnailCacheService`` cache key versioning."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="PySide6 required", exc_type=ImportError)

from PySide6.QtCore import QSize

from iPhoto.config import VIDEO_THUMBNAIL_CACHE_VERSION
from iPhoto.infrastructure.services.thumbnail_cache_service import ThumbnailCacheService


def test_cache_key_keeps_image_entries_unchanged() -> None:
    service = ThumbnailCacheService.__new__(ThumbnailCacheService)
    path = Path("library/photos/image.jpg")
    size = QSize(512, 512)

    key = ThumbnailCacheService._cache_key(service, path, size)
    expected = hashlib.md5(f"{path.as_posix()}_512x512".encode("utf-8")).hexdigest()

    assert key == expected


def test_cache_key_adds_video_version_salt() -> None:
    service = ThumbnailCacheService.__new__(ThumbnailCacheService)
    path = Path("library/videos/clip.mp4")
    size = QSize(512, 512)

    key = ThumbnailCacheService._cache_key(service, path, size)
    expected = hashlib.md5(
        f"{VIDEO_THUMBNAIL_CACHE_VERSION}|{path.as_posix()}_512x512".encode("utf-8")
    ).hexdigest()

    assert key == expected
