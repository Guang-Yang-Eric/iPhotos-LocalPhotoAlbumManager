import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from .base import UseCase, UseCaseRequest, UseCaseResponse


@dataclass(frozen=True)
class AggregateGeoDataRequest(UseCaseRequest):
    library_root: Path = Path(".")


@dataclass(frozen=True)
class AggregateGeoDataResponse(UseCaseResponse):
    assets: List[Any] = field(default_factory=list)
    total_count: int = 0


class AggregateGeoDataUseCase(UseCase):
    """Orchestrates geotagged asset aggregation."""

    def __init__(self, geo_aggregator=None):
        self._geo_aggregator = geo_aggregator
        self._logger = logging.getLogger(__name__)

    def execute(self, request: AggregateGeoDataRequest) -> AggregateGeoDataResponse:
        if self._geo_aggregator is None:
            return AggregateGeoDataResponse(success=False, error="Geo aggregator not configured")

        try:
            assets = self._geo_aggregator.get_geotagged_assets()
            return AggregateGeoDataResponse(
                assets=assets,
                total_count=len(assets),
            )
        except Exception as e:
            self._logger.error("AggregateGeoData failed: %s", e)
            return AggregateGeoDataResponse(success=False, error=str(e))
