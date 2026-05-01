"""Run the incoming market-data ingestion slice and write versioned artifacts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.core.base import PipelineContext
from flexihome.pipeline.artifacts import artifact_record, dataframe_fingerprint, new_run_dir, write_frame, write_json
from flexihome.pipeline.contracts import PipelineManifest, PipelineRunConfig
from flexihome.pipeline.stages import FeatureEngineeringStage, MarketDataIngestionStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FlexiHome incoming market-data pipeline slice.")
    parser.add_argument("--start-date", default="2026-01-15")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--n-homes", type=int, default=80)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--market-data-mode", choices=["synthetic", "auto", "real"], default="auto")
    parser.add_argument("--market-lookback-days", type=int, default=30)
    parser.add_argument("--output-root", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineRunConfig(
        start_date=args.start_date,
        days=args.days,
        n_homes=args.n_homes,
        freq_minutes=args.freq_minutes,
        market_data_mode=args.market_data_mode,
        market_lookback_days=args.market_lookback_days if args.market_data_mode != "synthetic" else None,
        output_root=args.output_root,
    )
    stages = [MarketDataIngestionStage(config), FeatureEngineeringStage(config)]
    context = PipelineContext()
    context.with_metadata(
        "dag",
        [
            {
                "nodes": ["market_data_ingestion", "feature_engineering"],
                "edges": [{"from": "market_data_ingestion", "to": "feature_engineering"}],
            }
        ],
    )
    context.with_metadata("contracts", [stage.get_metadata().get("contract", {}) for stage in stages])
    for stage in stages:
        stage.validate_inputs(context)
        context = stage.execute(context)

    run_dir = new_run_dir(config.output_root, "market_data")
    artifact_records = []

    frame_path = run_dir / "market_data_frame.csv"
    write_frame(frame_path, context.data)
    artifact_records.append(artifact_record("market_data_frame", frame_path, "dataframe", context.data))

    summary_path = run_dir / "market_data_summary.json"
    write_json(
        summary_path,
        {
            "config": config.to_dict(),
            "portfolio_summary": context.artifacts.get("portfolio_summary", {}),
            "market_data_status": context.artifacts.get("market_data_status", {}),
            "feature_columns": context.artifacts.get("feature_columns", []),
            "rows": len(context.data),
        },
    )
    artifact_records.append(artifact_record("market_data_summary", summary_path, "json"))

    manifest = PipelineManifest(
        run_id=run_dir.name,
        pipeline_name="flexihome_market_data_ingestion",
        config=config.to_dict(),
        dag=context.metadata.get("dag", []),
        contracts=context.metadata.get("contracts", []),
        artifacts=artifact_records,
        data_versions={"market_data_frame": dataframe_fingerprint(context.data)},
        started_at=context.started_at.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest.to_dict())
    artifact_records.append(artifact_record("manifest", manifest_path, "json"))
    print(f"market data artifacts: {run_dir}")


if __name__ == "__main__":
    main()
