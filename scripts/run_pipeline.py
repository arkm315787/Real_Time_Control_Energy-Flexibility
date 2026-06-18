"""Run the full Residential VPP market-data to optimization pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.pipeline import PipelineRunConfig, build_default_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Residential VPP pipeline and write a manifest.")
    parser.add_argument("--start-date", default="2026-01-15")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--n-homes", type=int, default=80)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--market-data-mode", choices=["synthetic", "auto", "real"], default="synthetic")
    parser.add_argument("--market-lookback-days", type=int, default=30)
    parser.add_argument("--forecaster-plugin", default="xgboost_default")
    parser.add_argument("--optimizer-plugin", default="pulp_default")
    parser.add_argument("--market-mode", choices=["Combined", "FCR-N", "aFRR"], default="Combined")
    parser.add_argument("--resource-mode", choices=["Hybrid portfolio", "Fast only"], default="Hybrid portfolio")
    parser.add_argument("--horizon-hours", type=int, default=1)
    parser.add_argument("--dispatch-hours", type=int, default=1)
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
        forecaster_plugin=args.forecaster_plugin,
        optimizer_plugin=args.optimizer_plugin,
        market_mode=args.market_mode,
        resource_mode=args.resource_mode,
        horizon_hours=args.horizon_hours,
        dispatch_hours=args.dispatch_hours,
        output_root=args.output_root,
    )
    context = build_default_pipeline(config).run()
    print(f"pipeline artifacts: {context.artifacts['run_dir']}")


if __name__ == "__main__":
    main()
