"""Run forecasting as a standalone local module and write artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.core.data import generate_synthetic_portfolio
from flexihome.core.base.registry import get_global_registry
from flexihome.core.engine import serialize_forecast_spec
from flexihome.pipeline.artifacts import new_run_dir, write_frame, write_json
from flexihome.plugins import DEFAULT_FORECASTER_PLUGIN, register_default_plugins


DEFAULT_PREDICTORS = [
    "temp_out_c",
    "irradiance_wm2",
    "base_load_kw",
    "pv_available_kw",
    "net_load_baseline_kw",
    "fcr_signed_act",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Coverly forecast model outside the dashboard.")
    parser.add_argument("--target", default="net_load_baseline_kw")
    parser.add_argument("--predictors", default=",".join(DEFAULT_PREDICTORS))
    parser.add_argument("--lags", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="2026-01-15")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--n-homes", type=int, default=1500)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--market-data-mode", choices=["synthetic", "auto", "real"], default="synthetic")
    parser.add_argument("--market-lookback-days", type=int, default=30)
    parser.add_argument("--forecaster-plugin", default=DEFAULT_FORECASTER_PLUGIN)
    parser.add_argument("--output-root", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = generate_synthetic_portfolio(
        start_date=args.start_date,
        days=args.days,
        n_homes=args.n_homes,
        ev_pen=0.30,
        bess_pen=0.22,
        pv_pen=0.45,
        hvac_pen=0.82,
        cloudiness=0.55,
        climate_shift_c=0.0,
        solar_scale=1.0,
        seed=args.seed,
        setpoint_c=21.0,
        comfort_band_c=1.0,
        scarcity=1.0,
        freq_minutes=args.freq_minutes,
        hvac_mode="Inverter / variable-speed",
        market_data_mode=args.market_data_mode,
        market_lookback_days=args.market_lookback_days if args.market_data_mode != "synthetic" else None,
    )
    df = bundle["data"]
    predictors = [item.strip() for item in args.predictors.split(",") if item.strip()]
    registry = register_default_plugins(get_global_registry())
    forecaster = registry.get_forecaster(args.forecaster_plugin, seed=args.seed)
    comparison = forecaster.fit_from_frame(df, args.target, predictors, args.lags, args.horizon_steps)
    spec = forecaster.to_forecast_spec()

    run_dir = new_run_dir(args.output_root, "forecast")
    write_frame(run_dir / "training_data.csv", df)
    write_frame(run_dir / "forecast_comparison.csv", comparison)
    (run_dir / "model.pkl").write_bytes(serialize_forecast_spec(spec))
    write_json(
        run_dir / "forecast_summary.json",
        {
            "module": "forecasting",
            "forecaster_plugin": args.forecaster_plugin,
            "target": spec.target,
            "predictors": spec.predictors,
            "lags": spec.lags,
            "horizon_steps": spec.horizon_steps,
            "metrics": spec.metrics,
            "rows": len(df),
            "market_data_status": bundle["summary"].get("market_data_status", {}),
            "artifacts": {
                "training_data": "training_data.csv",
                "comparison": "forecast_comparison.csv",
                "model": "model.pkl",
            },
        },
    )
    print(f"forecast artifacts: {run_dir}")


if __name__ == "__main__":
    main()
