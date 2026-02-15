import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .base import UseCase, UseCaseRequest, UseCaseResponse

DEFAULT_JPEG_QUALITY = 100


@dataclass(frozen=True)
class ApplyEditRequest(UseCaseRequest):
    asset_path: Path = Path(".")
    adjustments: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[Path] = None


@dataclass(frozen=True)
class ApplyEditResponse(UseCaseResponse):
    output_path: Optional[str] = None


class ApplyEditUseCase(UseCase):
    """Orchestrates applying edits (filters, adjustments) to an asset."""

    def __init__(self, render_fn=None, sidecar_loader=None):
        self._render_fn = render_fn
        self._sidecar_loader = sidecar_loader
        self._logger = logging.getLogger(__name__)

    def execute(self, request: ApplyEditRequest) -> ApplyEditResponse:
        try:
            if self._render_fn is None:
                from iPhoto.core.export import render_image
                self._render_fn = render_image

            result = self._render_fn(request.asset_path)
            if result is None:
                return ApplyEditResponse(success=False, error="Render returned no result")

            output = request.output_path or request.asset_path.with_suffix(".edited.jpg")
            result.save(str(output), "JPG", DEFAULT_JPEG_QUALITY)
            return ApplyEditResponse(output_path=str(output))
        except Exception as e:
            self._logger.error("ApplyEdit failed for %s: %s", request.asset_path, e)
            return ApplyEditResponse(success=False, error=str(e))
