"""Data models for market data service boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class MarketDataRequest:
    lookback_days: int = 90
    include_forecast_day: bool = True


@dataclass
class MarketDataMetadata:
    training_period: str
    training_rows: int
    forecast_day_included: bool
    forecast_rows: int
    total_rows: int
    data_completeness: Dict[str, float] = field(default_factory=dict)
    entsoe_status: str = "not_configured"
    fingrid_status: str = "not_configured"
    generated_for_date: date | None = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        if self.generated_for_date is not None:
            payload["generated_for_date"] = self.generated_for_date.isoformat()
        return payload


@dataclass
class MarketDataBundle:
    frame: pd.DataFrame
    metadata: MarketDataMetadata

    def as_tuple(self) -> tuple[pd.DataFrame, Dict[str, object]]:
        return self.frame, self.metadata.as_dict()
