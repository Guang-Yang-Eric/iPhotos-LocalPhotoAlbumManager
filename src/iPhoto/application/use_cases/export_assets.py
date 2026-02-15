import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .base import UseCase, UseCaseRequest, UseCaseResponse


@dataclass(frozen=True)
class ExportAssetsRequest(UseCaseRequest):
    source_paths: List[Path] = field(default_factory=list)
    export_root: Path = Path(".")
    library_root: Path = Path(".")


@dataclass(frozen=True)
class ExportAssetsResponse(UseCaseResponse):
    exported_count: int = 0
    failed_count: int = 0
    failed_paths: List[str] = field(default_factory=list)


class ExportAssetsUseCase(UseCase):
    """Orchestrates asset export with adjustments applied."""

    def __init__(self, export_fn=None):
        self._export_fn = export_fn
        self._logger = logging.getLogger(__name__)

    def execute(self, request: ExportAssetsRequest) -> ExportAssetsResponse:
        if self._export_fn is None:
            try:
                from iPhoto.core.export import export_asset
                self._export_fn = export_asset
            except ImportError:
                return ExportAssetsResponse(success=False, error="Export module not available")

        exported = 0
        failed = 0
        failed_paths: list[str] = []

        for path in request.source_paths:
            try:
                ok = self._export_fn(path, request.export_root, request.library_root)
                if ok:
                    exported += 1
                else:
                    failed += 1
                    failed_paths.append(str(path))
            except Exception as e:
                failed += 1
                failed_paths.append(str(path))
                self._logger.error("Export failed for %s: %s", path, e)

        return ExportAssetsResponse(
            exported_count=exported,
            failed_count=failed,
            failed_paths=failed_paths,
        )
