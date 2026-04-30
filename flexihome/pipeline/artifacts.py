"""Artifact helpers shared by local module runners."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from flexihome.api.serialization import json_safe


def new_run_dir(root: str | Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def write_frame(path: str | Path, frame: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)

