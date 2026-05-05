"""Smoke-test the centralized household roster and 4-second MPC layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.core.centralized_controller import CentralizedVPPController, InnerControllerConfig
from flexihome.core.data import generate_synthetic_portfolio


def constant_signal(start, periods: int = 3) -> pd.DataFrame:
    index = pd.date_range(pd.Timestamp(start), periods=periods, freq="4s")
    return pd.DataFrame(
        {
            "frequency_hz": 49.90,
            "fcr_signal_norm": 1.0,
            "afrr_signal_norm": 0.0,
            "fcr_up_act_frac": 1.0,
            "fcr_down_act_frac": 0.0,
            "afrr_up_act_frac": 0.0,
            "afrr_down_act_frac": 0.0,
        },
        index=index,
    )


def make_bundle(n_homes: int = 40):
    return generate_synthetic_portfolio(
        start_date="2026-01-15",
        days=1,
        n_homes=n_homes,
        ev_pen=0.50,
        bess_pen=0.50,
        pv_pen=0.30,
        hvac_pen=0.50,
        cloudiness=0.55,
        climate_shift_c=0.0,
        solar_scale=1.0,
        seed=42,
        setpoint_c=21.0,
        comfort_band_c=1.0,
        scarcity=1.0,
        freq_minutes=15,
        hvac_mode="Inverter / variable-speed",
    )


def main() -> None:
    bundle = make_bundle()
    roster = bundle["device_roster"]
    required = {"household_id", "device_id", "device_type", "gateway_id", "response_time_s"}
    missing = required.difference(roster.columns)
    if missing:
        raise SystemExit(f"Device roster missing columns: {sorted(missing)}")
    if roster.empty or roster["household_id"].nunique() < 2:
        raise SystemExit("Expected a traceable multi-household device roster.")

    df = bundle["data"]
    fleet_meta = {
        "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
        "setpoint_c": 21.0,
        "comfort_band_c": 1.0,
        "n_hvac": int(bundle["summary"]["n_hvac"]),
        "hvac_mode": "Inverter / variable-speed",
        "hvac_response_s": 20.0,
    }
    controller = CentralizedVPPController(
        roster,
        fleet_meta,
        InnerControllerConfig(max_devices_per_type=8, horizon_seconds=20),
    )
    plan = {
        "bess_fcr_kw": 20.0,
        "ev_fcr_kw": 20.0,
        "hvac_fcr_kw": 0.0,
        "bess_afrr_up_kw": 0.0,
        "bess_afrr_down_kw": 0.0,
        "ev_afrr_up_kw": 0.0,
        "ev_afrr_down_kw": 0.0,
        "hvac_afrr_up_kw": 0.0,
        "hvac_afrr_down_kw": 0.0,
        "pv_afrr_down_kw": 0.0,
    }
    summary, tracking, upper, commands = controller.execute_interval(
        row=df.iloc[0],
        fine_signals=constant_signal(df.index[0]),
        plan=plan,
        interval_index=0,
        market_mode="FCR-N",
        resource_mode="Hybrid portfolio",
    )
    if tracking.empty or commands.empty or upper.empty:
        raise SystemExit("Expected centralized 4-second MPC tracking, commands, and selected devices.")
    if "inner_solver_status" not in tracking or "tracking_error_kw" not in tracking:
        raise SystemExit("Centralized MPC trace missing solver diagnostics.")
    if summary["tracking_samples"] != len(tracking):
        raise SystemExit("Interval summary did not preserve tracking sample count.")

    household, appliance, fatigue = controller.contribution_frames(commands)
    if household.empty or appliance.empty or fatigue.empty:
        raise SystemExit("Expected household/appliance contributions and fatigue summary.")

    overload_plan = dict(plan)
    overload_plan["bess_fcr_kw"] = 100000.0
    overload_summary, overload_tracking, _, _ = controller.execute_interval(
        row=df.iloc[1],
        fine_signals=constant_signal(df.index[1]),
        plan=overload_plan,
        interval_index=1,
        market_mode="FCR-N",
        resource_mode="Fast only",
    )
    if overload_summary["shortfall_kw"] <= 0.0 and overload_tracking["shortfall_kw"].max() <= 0.0:
        raise SystemExit("Expected shortfall diagnostics for an infeasible high reserve request.")

    rotation_bundle = make_bundle(n_homes=12)
    rotation_roster = rotation_bundle["device_roster"].copy()
    rotation_roster = rotation_roster[rotation_roster["device_type"] == "BESS"].head(4).copy()
    rotation_roster["max_consecutive_seconds"] = 4
    rotation_roster["cooldown_seconds_required"] = 900
    rotation_df = rotation_bundle["data"]
    rotation_controller = CentralizedVPPController(
        rotation_roster,
        fleet_meta,
        InnerControllerConfig(max_devices_per_type=1, horizon_seconds=20),
    )
    small_plan = {key: 0.0 for key in plan}
    small_plan["bess_fcr_kw"] = 4.0
    _, _, first_upper, _ = rotation_controller.execute_interval(
        row=rotation_df.iloc[0],
        fine_signals=constant_signal(rotation_df.index[0]),
        plan=small_plan,
        interval_index=0,
        market_mode="FCR-N",
        resource_mode="Fast only",
    )
    _, _, second_upper, _ = rotation_controller.execute_interval(
        row=rotation_df.iloc[1],
        fine_signals=constant_signal(rotation_df.index[1]),
        plan=small_plan,
        interval_index=1,
        market_mode="FCR-N",
        resource_mode="Fast only",
    )
    first_device = str(first_upper["device_id"].iloc[0])
    second_device = str(second_upper["device_id"].iloc[0])
    if first_device == second_device:
        raise SystemExit("Usage-aware rotation did not move selection away from a cooled-down device.")

    print("centralized MPC smoke checks passed")
    print("tracking rows:", len(tracking))
    print("households:", len(household))
    print("rotated devices:", first_device, "->", second_device)


if __name__ == "__main__":
    main()
