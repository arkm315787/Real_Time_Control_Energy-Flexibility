"""Validate latest run artifacts for Airflow DAG guard tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the latest Coverly artifact directory.")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--prefix", required=True, help="Run directory prefix, such as pipeline, market_data, forecast, or optimization.")
    parser.add_argument("--require-file", action="append", default=[])
    parser.add_argument("--require-artifact", action="append", default=[])
    return parser.parse_args()


def latest_run_dir(root: Path, prefix: str) -> Path:
    candidates = sorted(root.glob(f"{prefix}_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No run directories found for prefix '{prefix}' under {root}.")
    return candidates[0]


def main() -> None:
    args = parse_args()
    run_dir = latest_run_dir(Path(args.root), args.prefix)
    print(f"validating: {run_dir}")

    for file_name in args.require_file:
        path = run_dir / file_name
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")
        print(f"file ok: {path.name}")

    manifest_path = run_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"manifest ok: {manifest_path.name}")

    if args.require_artifact:
        if manifest is None:
            raise SystemExit(f"Artifact validation requires manifest.json in {run_dir}.")
        available = {item.get("name") for item in manifest.get("artifacts", [])}
        missing = [name for name in args.require_artifact if name not in available]
        if missing:
            raise SystemExit(f"Missing required manifest artifacts: {missing}; available={sorted(available)}")
        for name in args.require_artifact:
            print(f"artifact ok: {name}")

    if not args.require_file and not args.require_artifact and manifest is None:
        raise SystemExit(f"No manifest.json found in {run_dir} and no explicit files were requested.")


if __name__ == "__main__":
    main()
