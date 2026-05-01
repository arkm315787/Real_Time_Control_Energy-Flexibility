"""Artifact helpers shared by local module runners."""

from __future__ import annotations

import json
import hashlib
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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return hashlib.sha256(b"empty-dataframe").hexdigest()
    csv_payload = frame.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(csv_payload).hexdigest()


def dataframe_schema(frame: pd.DataFrame) -> Dict[str, str]:
    if frame is None:
        return {}
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def artifact_record(name: str, path: str | Path, kind: str, frame: pd.DataFrame | None = None, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    target = Path(path)
    record = {
        "name": name,
        "kind": kind,
        "path": target.as_posix(),
        "sha256": file_sha256(target) if target.exists() else None,
        "metadata": metadata or {},
    }
    if frame is not None:
        record.update(
            {
                "rows": int(len(frame)),
                "columns": list(map(str, frame.columns)),
                "schema": dataframe_schema(frame),
                "dataframe_sha256": dataframe_fingerprint(frame),
            }
        )
    return record
