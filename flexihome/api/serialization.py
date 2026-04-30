"""JSON serialization helpers for pandas/numpy optimizer outputs."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def dataframe_to_records(frame: pd.DataFrame, limit: int | None = None) -> list[Dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    working = frame if limit is None else frame.head(limit)
    records = working.reset_index().to_dict(orient="records")
    normalized = []
    for record in records:
        if "index" in record and "timestamp" not in record:
            record["timestamp"] = record.pop("index")
        normalized.append(json_safe(record))
    return normalized


def serialize_optimization_result(result: Dict[str, Any]) -> Dict[str, Any]:
    history = result.get("history", pd.DataFrame())
    tracking = result.get("tracking_4s", pd.DataFrame())
    first_schedule = result.get("first_schedule", pd.DataFrame())
    return {
        "summary": json_safe(result.get("summary", {})),
        "market_data_status": json_safe(result.get("market_data_status", {})),
        "compliance": json_safe(result.get("compliance", {})),
        "history": dataframe_to_records(history),
        "tracking_4s": dataframe_to_records(tracking),
        "first_schedule": dataframe_to_records(first_schedule),
        "row_counts": {
            "history": int(len(history)) if history is not None else 0,
            "tracking_4s": int(len(tracking)) if tracking is not None else 0,
            "first_schedule": int(len(first_schedule)) if first_schedule is not None else 0,
        },
    }
