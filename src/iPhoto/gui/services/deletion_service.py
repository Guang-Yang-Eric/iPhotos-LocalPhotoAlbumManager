"""Service encapsulating asset-deletion logic extracted from AppFacade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Set, TYPE_CHECKING

from ...errors import AlbumOperationError

if TYPE_CHECKING:
    from ...library.manager import LibraryManager
    from .asset_move_service import AssetMoveService


def _normalize(path: Path) -> Path:
    """Resolve *path*, falling back to the original on OS errors."""
    try:
        return path.resolve()
    except OSError:
        return path


class DeletionService:
    """Move selected assets into the dedicated deleted-items folder."""

    def __init__(
        self,
        *,
        library_manager_getter: Callable[[], Optional["LibraryManager"]],
        model_provider: Callable[[], Optional[Callable[[], Any]]],
        move_service: "AssetMoveService",
        error_callback: Callable[[str], None],
    ) -> None:
        self._library_manager_getter = library_manager_getter
        self._model_provider = model_provider
        self._move_service = move_service
        self._error_callback = error_callback

    def delete_assets(self, sources: Iterable[Path]) -> None:
        """Move *sources* into the dedicated deleted-items folder."""

        library = self._library_manager_getter()
        if library is None:
            self._error_callback("Basic Library has not been configured.")
            return

        try:
            deleted_root = library.ensure_deleted_directory()
        except AlbumOperationError as exc:
            self._error_callback(str(exc))
            return

        normalized: List[Path] = []
        seen: Set[str] = set()
        for raw_path in sources:
            candidate = _normalize(Path(raw_path))
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)

        if not normalized:
            return

        model_provider = self._model_provider()
        model = model_provider() if model_provider else None

        for still_path in list(normalized):
            metadata = None
            if model and hasattr(model, "metadata_for_path"):
                metadata = model.metadata_for_path(still_path)

            if not metadata or not metadata.get("is_live"):
                continue
            motion_raw = metadata.get("live_motion_abs")
            if not motion_raw:
                continue
            motion_path = _normalize(Path(str(motion_raw)))
            motion_key = str(motion_path)
            if motion_key not in seen:
                seen.add(motion_key)
                normalized.append(motion_path)

        self._move_service.move_assets(normalized, deleted_root, operation="delete")
