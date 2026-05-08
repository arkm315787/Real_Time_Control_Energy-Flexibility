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


def combined_up_signal(start, periods: int = 3) -> pd.DataFrame:
    signals = constant_signal(start, periods)
    signals["afrr_signal_norm"] = 1.0
    signals["afrr_up_act_frac"] = 1.0
    return signals


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
        InnerControllerConfig(max_devices_per_type=8, horizon_seconds=4),
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
    required_tracking = {
        "inner_solver_status",
        "lower_tracking_mode",
        "fleet_power_before_kw",
        "fleet_power_after_kw",
        "ideal_delivered_kw",
        "telemetry_error_kw",
        "raw_request_kw",
        "socket_up_kw",
        "socket_down_kw",
        "socket_violation_up_kw",
        "socket_violation_down_kw",
        "target_command_kw",
        "tracking_delta_kw",
        "tracking_error_kw",
        "tracking_tolerance_kw",
        "error_rising",
        "recovery_mode",
        "fast_bridge_used_kw",
        "buffer_used_kw",
        "active_selected_devices",
        "buffer_selected_devices",
        "control_latency_ms",
        "control_deadline_ms",
        "control_deadline_met",
        "optimization_strategy",
    }
    missing_tracking = required_tracking.difference(tracking.columns)
    if missing_tracking:
        raise SystemExit(f"Centralized MPC trace missing fast-tracker diagnostics: {sorted(missing_tracking)}")
    if tracking["lower_tracking_mode"].nunique() != 1 or tracking["lower_tracking_mode"].iloc[0] != "proportional_buffer_tracker":
        raise SystemExit("Lower controller should run as a one-step proportional buffer tracker.")
    if int(tracking["inner_mpc_horizon_seconds"].max()) != 4:
        raise SystemExit("Lower tracking horizon should default to one 4-second step.")
    if not tracking["control_deadline_met"].all():
        raise SystemExit("Lower controller should meet the 4-second control deadline in the smoke case.")
    if "inner_solver_status" not in tracking or "tracking_error_kw" not in tracking:
        raise SystemExit("Centralized MPC trace missing solver diagnostics.")
    if summary["tracking_samples"] != len(tracking):
        raise SystemExit("Interval summary did not preserve tracking sample count.")
    if tracking["telemetry_error_kw"].abs().max() <= 0.0:
        raise SystemExit("Lower controller should report measured telemetry error instead of perfect predicted delivery.")
    if tracking["tracking_error_kw"].abs().max() <= 0.0:
        raise SystemExit("Lower controller should expose nonzero residual tracking error from the measured plant.")

    socket_controller = CentralizedVPPController(
        roster,
        fleet_meta,
        InnerControllerConfig(max_devices_per_type=8, horizon_seconds=4),
    )
    socket_plan = dict(plan)
    socket_plan["socket_up_kw"] = 10.0
    socket_plan["socket_down_kw"] = 10.0
    _, socket_tracking, _, _ = socket_controller.execute_interval(
        row=df.iloc[1],
        fine_signals=constant_signal(df.index[1], periods=2),
        plan=socket_plan,
        interval_index=1,
        market_mode="FCR-N",
        resource_mode="Hybrid portfolio",
    )
    if socket_tracking["requested_up_kw"].max() > 10.001:
        raise SystemExit("Lower controller must clamp TSO request to the upper socket.")
    if socket_tracking["socket_violation_up_kw"].max() <= 0.0:
        raise SystemExit("Lower controller should log raw request pressure above the upper socket.")

    buffer_plan = dict(plan)
    buffer_plan["hvac_fcr_kw"] = 80.0
    buffer_plan["reserve_buffer_kw"] = 40.0
    _, buffer_tracking, buffer_upper, buffer_commands = controller.execute_interval(
        row=df.iloc[2],
        fine_signals=constant_signal(df.index[2], periods=4),
        plan=buffer_plan,
        interval_index=2,
        market_mode="FCR-N",
        resource_mode="Hybrid portfolio",
    )
    if "buffer" not in set(buffer_upper.get("selection_role", [])):
        raise SystemExit("Upper device selection should keep a separate standby buffer roster.")
    if buffer_tracking["fast_bridge_used_kw"].max() <= 0.0:
        raise SystemExit("Expected lower controller to bridge slow HVAC response with fast active capacity.")
    fast_bridge_up = buffer_tracking["bess_up_kw"] + buffer_tracking["ev_up_kw"]
    if fast_bridge_up.max() <= 0.0:
        raise SystemExit("Expected BESS/EV to contribute during fast bridging instead of pure HVAC tracking.")
    if buffer_tracking["buffer_used_kw"].max() > 0.0 and not (buffer_commands.get("selection_role") == "buffer").any():
        raise SystemExit("Buffer dispatch should create at least one gateway command to a buffer resource.")

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
    if overload_plan["bess_fcr_kw"] >= 100000.0:
        raise SystemExit("Infeasible upper commitment should be derated to the selected active roster.")
    if overload_tracking["requested_up_kw"].max() > overload_tracking["committed_up_kw"].max() + 1e-6:
        raise SystemExit("Requested reserve should never exceed committed upward reserve.")

    derated_market_plan = {
        "bess_fcr_kw": 350.0,
        "ev_fcr_kw": 250.0,
        "hvac_fcr_kw": 100.0,
        "bess_afrr_up_kw": 3000.0,
        "ev_afrr_up_kw": 1000.0,
        "hvac_afrr_up_kw": 508.92,
        "bess_afrr_down_kw": 1000.0,
        "ev_afrr_down_kw": 500.0,
        "hvac_afrr_down_kw": 174.933,
        "pv_afrr_down_kw": 0.0,
    }
    controller._snap_derated_plan_to_market_segments(derated_market_plan)
    derated_fcr = derated_market_plan["bess_fcr_kw"] + derated_market_plan["ev_fcr_kw"] + derated_market_plan["hvac_fcr_kw"]
    derated_afrr_up = derated_market_plan["bess_afrr_up_kw"] + derated_market_plan["ev_afrr_up_kw"] + derated_market_plan["hvac_afrr_up_kw"]
    derated_afrr_down = (
        derated_market_plan["bess_afrr_down_kw"]
        + derated_market_plan["ev_afrr_down_kw"]
        + derated_market_plan["hvac_afrr_down_kw"]
        + derated_market_plan["pv_afrr_down_kw"]
    )
    if abs(derated_fcr - 700.0) > 1e-6 or abs(derated_afrr_up - 4000.0) > 1e-6 or abs(derated_afrr_down - 1000.0) > 1e-6:
        raise SystemExit("Selected-roster derating must snap market bids down to valid FCR-N/aFRR segments.")
    if derated_market_plan["afrr_up_segments"] != 4 or derated_market_plan["afrr_down_segments"] != 1:
        raise SystemExit("Derated aFRR segment metadata should match the executable market bid.")

    combined_plan = dict(plan)
    combined_plan.update(
        {
            "bess_fcr_kw": 100.0,
            "ev_fcr_kw": 0.0,
            "bess_afrr_up_kw": 80.0,
            "ev_afrr_up_kw": 0.0,
        }
    )
    combined_summary, combined_tracking, _, _ = controller.execute_interval(
        row=df.iloc[1],
        fine_signals=combined_up_signal(df.index[1]),
        plan=combined_plan,
        interval_index=3,
        market_mode="Combined",
        resource_mode="Fast only",
    )
    expected_combined_socket_kw = combined_plan["bess_fcr_kw"] + combined_plan["bess_afrr_up_kw"]
    if abs(float(combined_tracking["committed_up_kw"].max()) - expected_combined_socket_kw) > 1e-6:
        raise SystemExit("Combined product socket must equal FCR plus aFRR in the same direction.")
    if combined_tracking["requested_up_kw"].max() < expected_combined_socket_kw - 1e-6:
        raise SystemExit("Combined product request should not be silently shrunk to max(FCR, aFRR).")
    if combined_tracking["requested_up_kw"].max() > combined_tracking["committed_up_kw"].max() + 1e-6:
        raise SystemExit("Combined product request should be clipped to the summed committed reserve.")

    rotation_bundle = make_bundle(n_homes=12)
    rotation_roster = rotation_bundle["device_roster"].copy()
    rotation_roster = rotation_roster[rotation_roster["device_type"] == "BESS"].head(4).copy()
    rotation_roster["max_consecutive_seconds"] = 4
    rotation_roster["cooldown_seconds_required"] = 900
    rotation_df = rotation_bundle["data"]
    rotation_controller = CentralizedVPPController(
        rotation_roster,
        fleet_meta,
        InnerControllerConfig(max_devices_per_type=1, horizon_seconds=4),
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
