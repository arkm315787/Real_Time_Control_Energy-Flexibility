"""Real market data overlays for the Coverly simulator.

The simulator remains runnable without external services. When API keys are
available, this module overlays historical ENTSO-E and Fingrid market values on
top of the synthetic baseline used by the teaching model.
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests


ENTSOE_API_URL = "https://web-api.tp.entsoe.eu/api"
FINLAND_BIDDING_ZONE_EIC = "10YFI-1--------U"
FINGRID_CURRENT_API_URL = "https://data.fingrid.fi/api"
FINGRID_DEFAULT_MIN_INTERVAL_SECONDS = 2.1
FINGRID_DEFAULT_MAX_RETRIES = 4
_LAST_FINGRID_CALL_AT = 0.0

FINGRID_DATASETS = {
    "fcrn_capacity_eur_per_mw_h": ("FLEXIHOME_FINGRID_FCRN_PRICE_DATASET_ID", 317),
    "afrr_up_capacity_eur_per_mw_h": ("FLEXIHOME_FINGRID_AFRR_UP_CAPACITY_PRICE_DATASET_ID", 52),
    "afrr_down_capacity_eur_per_mw_h": ("FLEXIHOME_FINGRID_AFRR_DOWN_CAPACITY_PRICE_DATASET_ID", 51),
    "afrr_up_energy_eur_per_mwh": ("FLEXIHOME_FINGRID_AFRR_UP_ENERGY_PRICE_DATASET_ID", 347),
    "afrr_down_energy_eur_per_mwh": ("FLEXIHOME_FINGRID_AFRR_DOWN_ENERGY_PRICE_DATASET_ID", 348),
    "afrr_up_act_frac": ("FLEXIHOME_FINGRID_AFRR_UP_ACTIVATION_DATASET_ID", 349),
    "afrr_down_act_frac": ("FLEXIHOME_FINGRID_AFRR_DOWN_ACTIVATION_DATASET_ID", 350),
}

PRICE_COLUMNS = {
    "fcrn_capacity_eur_per_mw_h",
    "afrr_up_capacity_eur_per_mw_h",
    "afrr_down_capacity_eur_per_mw_h",
    "afrr_up_energy_eur_per_mwh",
    "afrr_down_energy_eur_per_mwh",
}

ACTIVATION_COLUMNS = {"afrr_up_act_frac", "afrr_down_act_frac"}


@dataclass
class MarketDataStatus:
    mode_requested: str
    mode_used: str = "synthetic"
    entsoe: str = "not_configured"
    fingrid: str = "not_configured"
    columns_loaded: list[str] = field(default_factory=list)
    observations_loaded: Dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    query_start_utc: str | None = None
    query_end_utc: str | None = None
    training_observations: int = 0
    training_days: float = 0.0
    timescaledb: str = "not_configured"
    rows_persisted: int = 0
    generated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, object]:
        return {
            "mode_requested": self.mode_requested,
            "mode_used": self.mode_used,
            "entsoe": self.entsoe,
            "fingrid": self.fingrid,
            "columns_loaded": self.columns_loaded,
            "observations_loaded": self.observations_loaded,
            "errors": self.errors,
            "query_start_utc": self.query_start_utc,
            "query_end_utc": self.query_end_utc,
            "training_observations": self.training_observations,
            "training_days": self.training_days,
            "timescaledb": self.timescaledb,
            "rows_persisted": self.rows_persisted,
            "generated_at_utc": self.generated_at_utc,
        }


def overlay_real_market_data(
    index: pd.DatetimeIndex,
    prices: pd.DataFrame,
    activation: pd.DataFrame,
    preview_4s: pd.DataFrame,
    mode: str | None = None,
    entsoe_api_key: str | None = None,
    fingrid_api_key: str | None = None,
    lookback_days: int | None = None,
    persist_to_timescale: bool | None = None,
    timeout_seconds: float = 12.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Overlay configured real market data onto synthetic market series."""

    load_local_env()
    selected_mode = (mode or os.getenv("FLEXIHOME_MARKET_DATA_MODE", "auto")).strip().lower()
    if selected_mode not in {"auto", "synthetic", "real"}:
        selected_mode = "auto"
    status = MarketDataStatus(mode_requested=selected_mode)
    if selected_mode == "synthetic":
        return prices, activation, preview_4s, status.as_dict()

    start, end = _query_window(index, lookback_days)
    status.query_start_utc = _iso_utc(start)
    status.query_end_utc = _iso_utc(end)
    output_prices = prices.copy()
    output_activation = activation.copy()
    raw_series: Dict[str, pd.Series] = {}

    entsoe_key = entsoe_api_key or os.getenv("ENTSOE_API_KEY") or os.getenv("ENTSOE_SECURITY_TOKEN")
    if entsoe_key:
        try:
            spot = fetch_entsoe_day_ahead_prices(start, end, entsoe_key, timeout_seconds)
            spot_aligned = align_series_to_index(spot, index)
            if spot_aligned.notna().any():
                output_prices["spot_price_eur_per_mwh"] = spot_aligned
                # Keep a market-price proxy available for energy settlement when Fingrid
                # aFRR energy price data is missing for the requested period.
                output_prices["afrr_up_energy_eur_per_mwh"] = output_prices["afrr_up_energy_eur_per_mwh"].where(
                    output_prices["afrr_up_energy_eur_per_mwh"].notna(),
                    spot_aligned,
                )
                status.entsoe = "connected"
                _mark_loaded(status, "spot_price_eur_per_mwh", spot)
                raw_series["entsoe:spot_price_eur_per_mwh"] = spot
            else:
                status.entsoe = "empty"
        except Exception as exc:  # pragma: no cover - exercised by smoke script with live keys
            status.entsoe = "unavailable"
            status.errors.append(f"ENTSO-E: {exc}")
    elif selected_mode == "real":
        status.errors.append("ENTSOE_API_KEY is not configured.")

    fingrid_key = fingrid_api_key or os.getenv("FINGRID_API_KEY") or os.getenv("FINGRID_OPENDATA_API_KEY")
    if fingrid_key:
        loaded_any = False
        for target_col, (_, dataset_id) in configured_fingrid_datasets().items():
            try:
                series = fetch_fingrid_dataset(dataset_id, start, end, fingrid_key, timeout_seconds)
                if series.empty:
                    continue
                aligned = align_series_to_index(series, index)
                if target_col in ACTIVATION_COLUMNS:
                    aligned = normalize_activation(aligned)
                    output_activation[target_col] = aligned.fillna(output_activation[target_col])
                    output_activation["afrr_signal_norm"] = (
                        output_activation["afrr_up_act_frac"] - output_activation["afrr_down_act_frac"]
                    ).clip(-1.0, 1.0)
                elif target_col in PRICE_COLUMNS:
                    output_prices[target_col] = aligned.fillna(output_prices[target_col])
                _mark_loaded(status, target_col, series)
                raw_series[f"fingrid:{dataset_id}:{target_col}"] = series
                loaded_any = True
            except Exception as exc:  # pragma: no cover - exercised by smoke script with live keys
                status.errors.append(f"Fingrid dataset {dataset_id} -> {target_col}: {exc}")
        status.fingrid = "connected" if loaded_any else "empty"
    elif selected_mode == "real":
        status.errors.append("FINGRID_API_KEY is not configured.")

    _mark_training_window(status, raw_series)
    if persist_to_timescale is None:
        persist_to_timescale = os.getenv("FLEXIHOME_MARKET_DATA_PERSIST", "1").lower() in {"1", "true", "yes"}
    if persist_to_timescale and raw_series:
        persist_market_data(raw_series, status)

    status.mode_used = _resolve_mode(status)
    if selected_mode == "real" and status.mode_used == "synthetic":
        raise RuntimeError("Real market data mode was requested, but no external market data could be loaded.")
    return output_prices, output_activation, preview_4s, status.as_dict()


def configured_fingrid_datasets() -> Dict[str, Tuple[str, int]]:
    configured = {}
    for column, (env_name, default_id) in FINGRID_DATASETS.items():
        value = os.getenv(env_name)
        configured[column] = (env_name, int(value) if value else default_id)
    return configured


def load_local_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file without extra deps."""

    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def fetch_entsoe_day_ahead_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    security_token: str,
    timeout_seconds: float = 12.0,
) -> pd.Series:
    params = {
        "securityToken": security_token,
        "documentType": "A44",
        "processType": "A01",
        "in_Domain": FINLAND_BIDDING_ZONE_EIC,
        "out_Domain": FINLAND_BIDDING_ZONE_EIC,
        "periodStart": _entsoe_time(start),
        "periodEnd": _entsoe_time(end),
    }
    response = requests.get(ENTSOE_API_URL, params=params, timeout=timeout_seconds)
    if response.status_code == 401:
        raise RuntimeError("unauthorized security token")
    response.raise_for_status()
    return parse_entsoe_price_xml(response.text)


def fetch_fingrid_dataset(
    dataset_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: str,
    timeout_seconds: float = 12.0,
) -> pd.Series:
    start_text = _iso_utc(start)
    end_text = _iso_utc(end)
    headers = {"x-api-key": api_key, "User-Agent": "Coverly-market-data/0.1"}

    current_url = f"{FINGRID_CURRENT_API_URL}/datasets/{dataset_id}/data"
    rows = []
    page = 1
    while page <= 20:
        params = {
            "startTime": start_text,
            "endTime": end_text,
            "format": "json",
            "locale": "en",
            "pageSize": 20000,
            "page": page,
            "sortBy": "startTime",
            "sortOrder": "asc",
        }
        response = None
        for attempt in range(_fingrid_max_retries() + 1):
            try:
                _sleep_before_fingrid_call()
                response = requests.get(current_url, params=params, headers=headers, timeout=timeout_seconds)
            except requests.RequestException as exc:
                raise RuntimeError(f"current Fingrid API request failed: {exc}") from exc
            if response.status_code != 429 or attempt >= _fingrid_max_retries():
                break
            time.sleep(_retry_delay_seconds(response.text))
        if response is None:
            raise RuntimeError("current Fingrid API request did not return a response")
        if not response.ok:
            raise RuntimeError(f"current Fingrid API returned HTTP {response.status_code}: {_short_response(response.text)}")
        page_rows = _extract_rows(response.json())
        if not page_rows:
            break
        rows.extend(page_rows)
        if len(page_rows) < int(params["pageSize"]):
            break
        page += 1
    return parse_fingrid_json({"data": rows})


def persist_market_data(raw_series: Dict[str, pd.Series], status: MarketDataStatus) -> None:
    dsn = os.getenv("FLEXIHOME_TIMESCALE_DSN", "").strip()
    if not dsn:
        status.timescaledb = "not_configured"
        return
    schema = _validate_identifier(os.getenv("FLEXIHOME_TIMESCALE_SCHEMA", "coverly").strip() or "coverly")
    table = _validate_identifier(os.getenv("FLEXIHOME_MARKET_DATA_TABLE", "market_data_observations").strip() or "market_data_observations")
    try:
        import psycopg
        from psycopg import sql
        from psycopg.rows import dict_row
    except ImportError:
        status.timescaledb = "driver_missing"
        return

    rows = []
    fetched_at = datetime.now(timezone.utc)
    for key, series in raw_series.items():
        parts = key.split(":")
        source = parts[0]
        dataset_id = parts[1] if len(parts) == 3 else None
        metric = parts[-1]
        for ts, value in series.dropna().items():
            rows.append((source, dataset_id, metric, _to_utc_aware(pd.Timestamp(ts)), float(value), fetched_at))
    if not rows:
        status.timescaledb = "empty"
        return

    with psycopg.connect(dsn, connect_timeout=_connect_timeout_seconds(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.{} (
                        source TEXT NOT NULL,
                        dataset_id TEXT,
                        metric TEXT NOT NULL,
                        observed_at TIMESTAMPTZ NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        fetched_at TIMESTAMPTZ NOT NULL
                    )
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                "SELECT create_hypertable(%s, 'observed_at', if_not_exists => TRUE, migrate_data => TRUE)",
                (f"{schema}.{table}",),
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (metric, observed_at DESC)").format(
                    sql.Identifier(f"idx_{table}_metric_time"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {}.{} (source, dataset_id, metric, observed_at, value, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table)),
                rows,
            )
    status.timescaledb = "connected"
    status.rows_persisted = len(rows)


def parse_entsoe_price_xml(xml_text: str) -> pd.Series:
    root = ET.fromstring(xml_text)
    points = []
    for period in _iter_local(root, "Period"):
        start_text = _child_text(period, "start")
        resolution_text = _child_text(period, "resolution") or "PT60M"
        if not start_text:
            continue
        period_start = _to_utc_naive(pd.Timestamp(start_text))
        delta = pd.Timedelta(minutes=_resolution_minutes(resolution_text))
        for point in _iter_local(period, "Point"):
            position_text = _child_text(point, "position")
            price_text = _child_text(point, "price.amount")
            if not position_text or price_text is None:
                continue
            ts = period_start + (int(position_text) - 1) * delta
            points.append((ts, float(price_text)))
    return _series_from_points(points, name="spot_price_eur_per_mwh")


def parse_fingrid_json(payload: object) -> pd.Series:
    rows = _extract_rows(payload)
    points = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = _first(row, "start_time", "startTime", "start_time_utc", "startTimeUtc")
        value = _first(row, "value", "Value")
        if start is None or value is None:
            continue
        points.append((_to_utc_naive(pd.Timestamp(start)), float(value)))
    return _series_from_points(points, name="value")


def align_series_to_index(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    if series.empty:
        return pd.Series(np.nan, index=index)
    cleaned = series.dropna().sort_index()
    if cleaned.empty:
        return pd.Series(np.nan, index=index)
    target_start = _to_utc_naive(pd.Timestamp(index.min()))
    target_end = _to_utc_naive(pd.Timestamp(index.max()))
    if cleaned.index.max() >= target_start and cleaned.index.min() <= target_end:
        return cleaned.reindex(index, method="ffill")
    values = np.resize(cleaned.to_numpy(dtype=float), len(index))
    return pd.Series(values, index=index, name=series.name)


def normalize_activation(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return series
    max_abs = float(series.abs().quantile(0.98))
    if max_abs <= 0:
        max_abs = float(series.abs().max() or 1.0)
    return (series.abs() / max(max_abs, 1e-6)).clip(0.0, 1.0)


def _query_window(index: pd.DatetimeIndex, lookback_days: int | None = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if lookback_days and lookback_days > 0:
        end = pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None)
        start = end - pd.Timedelta(days=int(lookback_days))
        return start, end
    start = _to_utc_naive(pd.Timestamp(index.min()).floor("h"))
    end = _to_utc_naive(pd.Timestamp(index.max()).ceil("h") + pd.Timedelta(hours=1))
    return start, end


def _resolve_mode(status: MarketDataStatus) -> str:
    has_entsoe = status.entsoe == "connected"
    has_fingrid = status.fingrid == "connected"
    if has_entsoe and has_fingrid:
        return "real"
    if has_entsoe or has_fingrid:
        return "mixed"
    return "synthetic"


def _mark_loaded(status: MarketDataStatus, column: str, series: pd.Series) -> None:
    if column not in status.columns_loaded:
        status.columns_loaded.append(column)
    status.observations_loaded[column] = int(series.dropna().shape[0])


def _mark_training_window(status: MarketDataStatus, raw_series: Dict[str, pd.Series]) -> None:
    if not raw_series:
        return
    counts = [int(series.dropna().shape[0]) for series in raw_series.values()]
    starts = [series.dropna().index.min() for series in raw_series.values() if not series.dropna().empty]
    ends = [series.dropna().index.max() for series in raw_series.values() if not series.dropna().empty]
    status.training_observations = int(sum(counts))
    if starts and ends:
        span = pd.Timestamp(max(ends)) - pd.Timestamp(min(starts))
        status.training_days = round(max(span.total_seconds(), 0.0) / 86400.0, 2)


def _entsoe_time(value: pd.Timestamp) -> str:
    return _to_utc_naive(value).strftime("%Y%m%d%H%M")


def _iso_utc(value: pd.Timestamp) -> str:
    return _to_utc_naive(value).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _to_utc_naive(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _to_utc_aware(value: pd.Timestamp) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _resolution_minutes(value: str) -> int:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", value)
    if not match:
        return 60
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    return hours * 60 + minutes


def _iter_local(root: ET.Element, local_name: str) -> Iterable[ET.Element]:
    suffix = f"}}{local_name}"
    for element in root.iter():
        if element.tag == local_name or element.tag.endswith(suffix):
            yield element


def _child_text(root: ET.Element, local_name: str) -> Optional[str]:
    suffix = f"}}{local_name}"
    for child in root.iter():
        if child is root:
            continue
        if child.tag == local_name or child.tag.endswith(suffix):
            return child.text
    return None


def _extract_rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "events", "value"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if {"startTime", "value"}.issubset(payload) or {"start_time", "value"}.issubset(payload):
            return [payload]
    return []


def _first(row: dict, *keys: str) -> Optional[object]:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _validate_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid database identifier: {value!r}")
    return value


def _short_response(text: str) -> str:
    normalized = " ".join(str(text).split())
    return normalized[:240] if normalized else "empty response"


def _sleep_before_fingrid_call() -> None:
    global _LAST_FINGRID_CALL_AT
    min_interval = _fingrid_min_interval_seconds()
    elapsed = time.monotonic() - _LAST_FINGRID_CALL_AT
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_FINGRID_CALL_AT = time.monotonic()


def _retry_delay_seconds(text: str) -> float:
    match = re.search(r"try again in\s+(\d+)\s+seconds?", str(text), flags=re.IGNORECASE)
    if match:
        return max(float(match.group(1)) + 0.5, _fingrid_min_interval_seconds())
    return _fingrid_min_interval_seconds()


def _fingrid_min_interval_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("FLEXIHOME_FINGRID_MIN_INTERVAL_SECONDS", FINGRID_DEFAULT_MIN_INTERVAL_SECONDS)))
    except ValueError:
        return FINGRID_DEFAULT_MIN_INTERVAL_SECONDS


def _fingrid_max_retries() -> int:
    try:
        return max(0, int(os.getenv("FLEXIHOME_FINGRID_MAX_RETRIES", str(FINGRID_DEFAULT_MAX_RETRIES))))
    except ValueError:
        return FINGRID_DEFAULT_MAX_RETRIES


def _connect_timeout_seconds() -> int:
    try:
        return max(1, int(os.getenv("FLEXIHOME_TIMESCALE_CONNECT_TIMEOUT", "5")))
    except ValueError:
        return 5


def _series_from_points(points: Iterable[tuple[pd.Timestamp, float]], name: str) -> pd.Series:
    rows = list(points)
    if not rows:
        return pd.Series(dtype=float, name=name)
    frame = pd.DataFrame(rows, columns=["timestamp", name])
    return frame.groupby("timestamp", sort=True)[name].mean()
