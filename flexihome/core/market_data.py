"""Real market data overlays for the FlexiHome simulator.

The simulator remains runnable without external services. When API keys are
available, this module overlays historical ENTSO-E and Fingrid market values on
top of the synthetic baseline used by the teaching model.
"""

from __future__ import annotations

import os
import re
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
FINGRID_LEGACY_API_URL = "https://api.fingrid.fi/v1"

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
            "generated_at_utc": self.generated_at_utc,
        }


def overlay_real_market_data(
    index: pd.DatetimeIndex,
    prices: pd.DataFrame,
    activation: pd.DataFrame,
    preview_4s: pd.DataFrame,
    timeout_seconds: float = 12.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Overlay configured real market data onto synthetic market series."""

    load_local_env()
    mode = os.getenv("FLEXIHOME_MARKET_DATA_MODE", "auto").strip().lower()
    if mode not in {"auto", "synthetic", "real"}:
        mode = "auto"
    status = MarketDataStatus(mode_requested=mode)
    if mode == "synthetic":
        return prices, activation, preview_4s, status.as_dict()

    start, end = _query_window(index)
    output_prices = prices.copy()
    output_activation = activation.copy()

    entsoe_key = os.getenv("ENTSOE_API_KEY") or os.getenv("ENTSOE_SECURITY_TOKEN")
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
            else:
                status.entsoe = "empty"
        except Exception as exc:  # pragma: no cover - exercised by smoke script with live keys
            status.entsoe = "unavailable"
            status.errors.append(f"ENTSO-E: {exc}")
    elif mode == "real":
        status.errors.append("ENTSOE_API_KEY is not configured.")

    fingrid_key = os.getenv("FINGRID_API_KEY") or os.getenv("FINGRID_OPENDATA_API_KEY")
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
                loaded_any = True
            except Exception as exc:  # pragma: no cover - exercised by smoke script with live keys
                status.errors.append(f"Fingrid dataset {dataset_id} -> {target_col}: {exc}")
        status.fingrid = "connected" if loaded_any else "empty"
    elif mode == "real":
        status.errors.append("FINGRID_API_KEY is not configured.")

    status.mode_used = _resolve_mode(status)
    if mode == "real" and status.mode_used == "synthetic":
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
    headers = {"x-api-key": api_key, "version": "FlexiHome-market-data/0.1"}

    current_url = f"{FINGRID_CURRENT_API_URL}/datasets/{dataset_id}/data"
    current_params = {"startTime": start_text, "endTime": end_text, "pageSize": 20000}
    try:
        response = requests.get(current_url, params=current_params, headers=headers, timeout=timeout_seconds)
        if response.ok:
            return parse_fingrid_json(response.json())
    except requests.RequestException:
        pass

    legacy_url = f"{FINGRID_LEGACY_API_URL}/variable/{dataset_id}/events/json"
    legacy_params = {"start_time": start_text, "end_time": end_text}
    response = requests.get(legacy_url, params=legacy_params, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    return parse_fingrid_json(response.json())


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
    return cleaned.reindex(index, method="ffill")


def normalize_activation(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return series
    max_abs = float(series.abs().quantile(0.98))
    if max_abs <= 0:
        max_abs = float(series.abs().max() or 1.0)
    return (series.abs() / max(max_abs, 1e-6)).clip(0.0, 1.0)


def _query_window(index: pd.DatetimeIndex) -> Tuple[pd.Timestamp, pd.Timestamp]:
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


def _entsoe_time(value: pd.Timestamp) -> str:
    return _to_utc_naive(value).strftime("%Y%m%d%H%M")


def _iso_utc(value: pd.Timestamp) -> str:
    return _to_utc_naive(value).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _to_utc_naive(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


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


def _series_from_points(points: Iterable[tuple[pd.Timestamp, float]], name: str) -> pd.Series:
    rows = list(points)
    if not rows:
        return pd.Series(dtype=float, name=name)
    frame = pd.DataFrame(rows, columns=["timestamp", name])
    return frame.groupby("timestamp", sort=True)[name].mean()
