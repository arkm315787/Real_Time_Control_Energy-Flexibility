"""Smoke-test the explicit Residential VPP pipeline DAG and artifact manifest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.pipeline import PipelineRunConfig, build_default_pipeline


def main() -> None:
    config = PipelineRunConfig(
        days=1,
        n_homes=80,
        freq_minutes=15,
        horizon_hours=1,
        dispatch_hours=1,
        output_root="runs",
    )
    pipeline = build_default_pipeline(config)
    context = pipeline.run()
    manifest = context.artifacts.get("manifest", {})
    run_dir = context.artifacts.get("run_dir")
    result = context.artifacts.get("optimization_result", {})
    history = result.get("history")

    print("pipeline dag:", pipeline.describe_dag())
    print("run dir:", run_dir)
    print("manifest artifacts:", len(manifest.get("artifacts", [])))
    print("data versions:", manifest.get("data_versions", {}))

    if not run_dir or not Path(run_dir).exists():
        raise SystemExit("Pipeline did not create a run directory.")
    if not (Path(run_dir) / "manifest.json").exists():
        raise SystemExit("Pipeline did not write manifest.json.")
    if history is None or history.empty:
        raise SystemExit("Pipeline optimization result has no history rows.")
    if not manifest.get("data_versions", {}).get("portfolio_timeseries"):
        raise SystemExit("Pipeline manifest is missing portfolio data version.")


if __name__ == "__main__":
    main()
