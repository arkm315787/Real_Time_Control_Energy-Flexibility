"""Market Service: TSO-side market data ingestion and training splits."""

from __future__ import annotations

import logging
import hashlib
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

from config.settings import AppSettings
from flexihome.core.market_data import (
    configured_fingrid_datasets,
    fetch_entsoe_day_ahead_prices,
    fetch_fingrid_dataset,
    normalize_activation,
)
from models.market_data import MarketDataBundle, MarketDataMetadata
from services.cache_manager import CacheManager
from utils.data_sync import data_completeness, split_training_and_operational

logger = logging.getLogger(__name__)


class MarketDataService:
    """Handles TSO-side market data from ENTSO-E and Fingrid APIs."""

    def __init__(
        self,
        entsoe_key: Optional[str] = None,
        fingrid_key: Optional[str] = None,
        settings: AppSettings | None = None,
        cache: CacheManager[MarketDataBundle] | None = None,
    ) -> None:
        self.settings = settings or AppSettings.from_env()
        self.entsoe_key = entsoe_key or self.settings.entsoe_api_key
        self.fingrid_key = fingrid_key or self.settings.fingrid_api_key
        self.cache = cache or CacheManager(
            ttl_seconds=self.settings.market_data_cache_ttl_seconds,
            cache_dir=self.settings.market_data_cache_dir,
        )

    def fetch_market_data(self, lookback_days: int = 90, include_forecast_day: bool = True) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Fetch market data for training plus optional operational day."""

        request_day = datetime.now(timezone.utc).date()
        cache_key = self.cache.make_key(
            "market_data",
            lookback_days,
            include_forecast_day,
            request_day.isoformat(),
            self._secret_fingerprint(self.entsoe_key),
            self._secret_fingerprint(self.fingrid_key),
        )

        def load() -> MarketDataBundle:
            end_date = request_day
            start_date = end_date - timedelta(days=int(lookback_days))
            logger.info("Fetching market data: %s to %s", start_date, end_date)
            training_df, training_errors = self._fetch_from_apis(start_date, end_date)

            forecast_df = pd.DataFrame()
            forecast_errors: list[str] = []
            if include_forecast_day:
                forecast_start = end_date + timedelta(days=1)
                logger.info("Fetching operational forecast day: %s", forecast_start)
                forecast_df, forecast_errors = self._fetch_from_apis(forecast_start, forecast_start)

            combined_df = pd.concat([training_df, forecast_df], axis=0).sort_index()
            metadata = MarketDataMetadata(
                training_period=f"{start_date} to {end_date}",
                training_rows=len(training_df),
                forecast_day_included=bool(include_forecast_day),
                forecast_rows=len(forecast_df),
                total_rows=len(combined_df),
                data_completeness=data_completeness(combined_df),
                entsoe_status="available" if self.entsoe_key else "not_configured",
                fingrid_status="available" if self.fingrid_key else "not_configured",
                generated_for_date=request_day,
                errors=training_errors + forecast_errors,
            )
            return MarketDataBundle(combined_df, metadata)

        bundle = self.cache.get_or_set(cache_key, load)
        return bundle.as_tuple()

    def split_training_and_operational(
        self,
        combined_df: pd.DataFrame,
        training_ratio: float = 90 / 91,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return split_training_and_operational(combined_df, training_ratio=training_ratio)

    def _fetch_from_apis(self, start_date: date, end_date: date) -> tuple[pd.DataFrame, list[str]]:
        frames: list[pd.DataFrame] = []
        errors: list[str] = []

        if self.entsoe_key:
            try:
                entsoe_df = self._call_entsoe(start_date, end_date)
                if not entsoe_df.empty:
                    frames.append(entsoe_df)
                    logger.info("ENTSO-E fetched %s rows", len(entsoe_df))
            except Exception as exc:  # pragma: no cover - depends on live API
                message = f"ENTSO-E API failed: {exc}"
                logger.warning(message)
                errors.append(message)

        if self.fingrid_key:
            try:
                fingrid_df = self._call_fingrid(start_date, end_date)
                if not fingrid_df.empty:
                    frames.append(fingrid_df)
                    logger.info("Fingrid fetched %s rows", len(fingrid_df))
            except Exception as exc:  # pragma: no cover - depends on live API
                message = f"Fingrid API failed: {exc}"
                logger.warning(message)
                errors.append(message)

        if not frames:
            return pd.DataFrame(), errors

        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.join(frame, how="outer")
        return combined.sort_index(), errors

    def _call_entsoe(self, start_date: date, end_date: date) -> pd.DataFrame:
        start, end = self._date_window(start_date, end_date)
        spot = fetch_entsoe_day_ahead_prices(start, end, str(self.entsoe_key))
        return spot.rename("spot_price_eur_per_mwh").to_frame()

    def _call_fingrid(self, start_date: date, end_date: date) -> pd.DataFrame:
        start, end = self._date_window(start_date, end_date)
        columns: list[pd.Series] = []
        for target_col, (_, dataset_id) in configured_fingrid_datasets().items():
            series = fetch_fingrid_dataset(dataset_id, start, end, str(self.fingrid_key))
            if series.empty:
                continue
            if target_col.endswith("_act_frac"):
                series = normalize_activation(series)
            columns.append(series.rename(target_col))
        if not columns:
            return pd.DataFrame()
        return pd.concat(columns, axis=1).sort_index()

    @staticmethod
    def _date_window(start_date: date, end_date: date) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(start_date).tz_localize("UTC")
        end = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
        return start, end

    @staticmethod
    def _secret_fingerprint(value: str | None) -> str:
        if not value:
            return "not_configured"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
