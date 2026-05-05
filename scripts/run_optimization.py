"""Run MPC/MILP optimization as a standalone local module and write artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.api.serialization import serialize_optimization_result
from flexihome.core.base.registry import get_global_registry
from flexihome.core.data import generate_synthetic_portfolio
from flexihome.core.engine import SUPPORTED_MPC_TARGETS, default_hvac_response_seconds, run_mpc_controller
from flexihome.pipeline.artifacts import new_run_dir, write_frame, write_json
from flexihome.plugins import DEFAULT_FORECASTER_PLUGIN, DEFAULT_OPTIMIZER_PLUGIN, register_default_plugins


DEFAULT_PREDICTORS = [
    "temp_out_c",
    "irradiance_wm2",
    "base_load_kw",
    "pv_available_kw",
    "net_load_baseline_kw",
    "fcrn_capacity_eur_per_mw_h",
    "afrr_up_capacity_eur_per_mw_h",
    "afrr_down_capacity_eur_per_mw_h",
    "afrr_up_act_frac",
    "afrr_down_act_frac",
    "fcr_signed_act",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FlexiHome MPC/MILP optimization outside the dashboard.")
    parser.add_argument("--market-mode", choices=["Combined", "FCR-N", "aFRR"], default="Combined")
    parser.add_argument("--resource-mode", choices=["Hybrid portfolio", "Fast only"], default="Hybrid portfolio")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--dispatch-hours", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="2026-01-15")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--n-homes", type=int, default=1500)
    parser.add_argument("--freq-minutes", type=int, default=15)
    parser.add_argument("--market-data-mode", choices=["synthetic", "auto", "real"], default="synthetic")
    parser.add_argument("--market-lookback-days", type=int, default=30)
    parser.add_argument("--forecaster-plugin", default=DEFAULT_FORECASTER_PLUGIN)
    parser.add_argument("--optimizer-plugin", default=DEFAULT_OPTIMIZER_PLUGIN)
    parser.add_argument("--inner-controller-mode", choices=["mpc"], default="mpc")
    parser.add_argument("--inner-dt-seconds", type=int, default=4)
    parser.add_argument("--inner-mpc-horizon-seconds", type=int, default=20)
    parser.add_argument("--rotation-strategy", choices=["usage_aware"], default="usage_aware")
    parser.add_argument("--gateway-mode", choices=["simulated_centralized"], default="simulated_centralized")
    parser.add_argument("--output-root", default="runs")
    return parser.parse_args()


def train_mpc_forecasters(df, seed: int, forecaster_plugin: str):
    registry = register_default_plugins(get_global_registry())
    models = {}
    for target in SUPPORTED_MPC_TARGETS:
        forecaster = registry.get_forecaster(
            forecaster_plugin,
            name=f"{forecaster_plugin}_{target}",
            seed=seed,
        )
        forecaster.fit_from_frame(df, target, DEFAULT_PREDICTORS, lags=4, horizon_steps=1)
        models[target] = forecaster
    return models


def main() -> None:
    args = parse_args()
    hvac_mode = "Inverter / variable-speed"
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
        hvac_mode=hvac_mode,
        market_data_mode=args.market_data_mode,
        market_lookback_days=args.market_lookback_days if args.market_data_mode != "synthetic" else None,
    )
    df = bundle["data"]
    registry = register_default_plugins(get_global_registry())
    models = train_mpc_forecasters(df, args.seed, args.forecaster_plugin)
    optimizer = registry.get_optimizer(args.optimizer_plugin)
    result = run_mpc_controller(
        df=df,
        models=models,
        fleet_meta={
            "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
            "setpoint_c": 21.0,
            "comfort_band_c": 1.0,
            "n_hvac": int(bundle["summary"]["n_hvac"]),
            "hvac_mode": hvac_mode,
            "hvac_response_s": default_hvac_response_seconds(hvac_mode),
        },
        market_mode=args.market_mode,
        resource_mode=args.resource_mode,
        horizon_hours=args.horizon_hours,
        dispatch_hours=args.dispatch_hours,
        penalty_weights={"degradation": 18.0, "comfort": 120.0, "departure": 160.0},
        preview_4s=bundle["preview_4s"],
        optimizer=optimizer,
        device_roster=bundle.get("device_roster"),
        inner_controller_mode=args.inner_controller_mode,
        inner_dt_seconds=args.inner_dt_seconds,
        inner_mpc_horizon_seconds=args.inner_mpc_horizon_seconds,
        rotation_strategy=args.rotation_strategy,
        gateway_mode=args.gateway_mode,
    )

    run_dir = new_run_dir(args.output_root, "optimization")
    write_frame(run_dir / "training_data.csv", df)
    write_frame(run_dir / "device_roster.csv", bundle.get("device_roster"))
    write_frame(run_dir / "history.csv", result.get("history"))
    write_frame(run_dir / "tracking_4s.csv", result.get("tracking_4s"))
    write_frame(run_dir / "first_schedule.csv", result.get("first_schedule"))
    for name in (
        "household_contributions",
        "appliance_contributions",
        "upper_device_schedule",
        "inner_mpc_trace",
        "gateway_commands",
        "usage_fatigue_summary",
    ):
        write_frame(run_dir / f"{name}.csv", result.get(name))
    serialized = serialize_optimization_result(result)
    write_json(
        run_dir / "optimization_summary.json",
        {
            "module": "optimization",
            "forecaster_plugin": args.forecaster_plugin,
            "optimizer_plugin": args.optimizer_plugin,
            "inner_controller_mode": args.inner_controller_mode,
            "inner_dt_seconds": args.inner_dt_seconds,
            "inner_mpc_horizon_seconds": args.inner_mpc_horizon_seconds,
            "rotation_strategy": args.rotation_strategy,
            "gateway_mode": args.gateway_mode,
            "market_mode": args.market_mode,
            "resource_mode": args.resource_mode,
            "horizon_hours": args.horizon_hours,
            "dispatch_hours": args.dispatch_hours,
            "summary": serialized.get("summary", {}),
            "compliance": serialized.get("compliance", {}),
            "row_counts": serialized.get("row_counts", {}),
            "market_data_status": bundle["summary"].get("market_data_status", {}),
            "artifacts": {
                "training_data": "training_data.csv",
                "device_roster": "device_roster.csv",
                "history": "history.csv",
                "tracking_4s": "tracking_4s.csv",
                "first_schedule": "first_schedule.csv",
                "household_contributions": "household_contributions.csv",
                "appliance_contributions": "appliance_contributions.csv",
                "upper_device_schedule": "upper_device_schedule.csv",
                "inner_mpc_trace": "inner_mpc_trace.csv",
                "gateway_commands": "gateway_commands.csv",
                "usage_fatigue_summary": "usage_fatigue_summary.csv",
            },
        },
    )
    print(f"optimization artifacts: {run_dir}")


if __name__ == "__main__":
    main()
