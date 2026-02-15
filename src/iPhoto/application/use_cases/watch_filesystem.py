import logging
from dataclasses import dataclass
from typing import Optional

from .base import UseCase, UseCaseRequest, UseCaseResponse


@dataclass(frozen=True)
class WatchFilesystemRequest(UseCaseRequest):
    action: str = "status"  # "pause", "resume", "rebuild", "status"


@dataclass(frozen=True)
class WatchFilesystemResponse(UseCaseResponse):
    is_paused: bool = False


class WatchFilesystemUseCase(UseCase):
    """Orchestrates filesystem watching lifecycle."""

    def __init__(self, watcher=None):
        self._watcher = watcher
        self._logger = logging.getLogger(__name__)

    def execute(self, request: WatchFilesystemRequest) -> WatchFilesystemResponse:
        if self._watcher is None:
            return WatchFilesystemResponse(success=False, error="Filesystem watcher not configured")

        try:
            if request.action == "pause":
                self._watcher.pause_watcher()
                return WatchFilesystemResponse(is_paused=True)
            elif request.action == "resume":
                self._watcher.resume_watcher()
                return WatchFilesystemResponse(is_paused=False)
            elif request.action == "rebuild":
                self._watcher._rebuild_watches()
                return WatchFilesystemResponse()
            elif request.action == "status":
                return WatchFilesystemResponse()
            else:
                return WatchFilesystemResponse(success=False, error=f"Unknown action: {request.action}")
        except Exception as e:
            self._logger.error("WatchFilesystem failed: %s", e)
            return WatchFilesystemResponse(success=False, error=str(e))
