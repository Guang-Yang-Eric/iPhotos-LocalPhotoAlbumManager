import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import UseCase, UseCaseRequest, UseCaseResponse


@dataclass(frozen=True)
class ManageTrashRequest(UseCaseRequest):
    action: str = "cleanup"  # "ensure", "cleanup", "get_path"
    library_root: Path = Path(".")


@dataclass(frozen=True)
class ManageTrashResponse(UseCaseResponse):
    trash_path: Optional[str] = None
    cleaned_count: int = 0


class ManageTrashUseCase(UseCase):
    """Orchestrates trash/deleted-items management."""

    def __init__(self, trash_manager=None):
        self._trash_manager = trash_manager
        self._logger = logging.getLogger(__name__)

    def execute(self, request: ManageTrashRequest) -> ManageTrashResponse:
        if self._trash_manager is None:
            return ManageTrashResponse(success=False, error="Trash manager not configured")

        try:
            if request.action == "ensure":
                path = self._trash_manager.ensure_deleted_directory()
                return ManageTrashResponse(trash_path=str(path) if path else None)
            elif request.action == "cleanup":
                count = self._trash_manager.cleanup_deleted_index()
                return ManageTrashResponse(cleaned_count=count)
            elif request.action == "get_path":
                path = self._trash_manager.deleted_directory()
                return ManageTrashResponse(trash_path=str(path) if path else None)
            else:
                return ManageTrashResponse(success=False, error=f"Unknown action: {request.action}")
        except Exception as e:
            self._logger.error("ManageTrash failed: %s", e)
            return ManageTrashResponse(success=False, error=str(e))
