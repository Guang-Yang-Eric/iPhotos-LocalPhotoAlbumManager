"""Lightweight models for queued move/delete operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from iPhoto.application.dtos import AssetDTO


@dataclass(frozen=True)
class PendingMove:
    dto: AssetDTO
    source_abs: Path
    destination_root: Path
    destination_album_path: str
    destination_abs: Path
    destination_rel: Path
    is_delete: bool

