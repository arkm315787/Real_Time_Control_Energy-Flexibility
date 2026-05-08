"""Smoke-test risk-aware upper MPC and optional lower-MPC execution."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexihome.core.engine import (
    SUPPORTED_MPC_TARGETS,
    _available_up_down,
    _market_gate_decision,
    fcr_n_dynamic_deliverable_fraction,
    generate_synthetic_portfolio,
    run_lower_mpc_from_upper_result,
    run_mpc_controller,
)


class PersistenceForecaster:
    def __init__(self, target: str, predictors: list[str]) -> None:
        self.target = target
        self.predictors = predictors
        self.lags = 1

    def predict(self, row: pd.DataFrame, horizon_steps: int = 1):
        return [float(row[self.target].iloc[0])]


def make_models(df: pd.DataFrame):
    predictors = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return {target: PersistenceForecaster(target, predictors) for target in SUPPORTED_MPC_TARGETS}


def run_case(risk_quantile: float, execute_lower_mpc: bool):
    bundle = generate_synthetic_portfolio(
        start_date="2026-01-15",
        days=1,
        n_homes=1200,
        ev_pen=0.35,
        bess_pen=0.30,
        pv_pen=0.45,
        hvac_pen=0.80,
        cloudiness=0.55,
        climate_shift_c=0.0,
        solar_scale=1.0,
        seed=7,
        setpoint_c=21.0,
        comfort_band_c=1.0,
        scarcity=1.8,
        freq_minutes=15,
        hvac_mode="Inverter / variable-speed",
    )
    df = bundle["data"]
    df[["fcrn_capacity_eur_per_mw_h", "afrr_up_capacity_eur_per_mw_h", "afrr_down_capacity_eur_per_mw_h"]] = 1000.0
    result = run_mpc_controller(
        df=df,
        models=make_models(df),
        fleet_meta={
            "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
            "setpoint_c": 21.0,
            "comfort_band_c": 1.0,
            "n_hvac": int(bundle["summary"]["n_hvac"]),
            "hvac_mode": "Inverter / variable-speed",
            "hvac_response_s": 20.0,
        },
        market_mode="Combined",
        resource_mode="Hybrid portfolio",
        horizon_hours=0.5,
        dispatch_hours=0.5,
        penalty_weights={
            "degradation": 35.0,
            "comfort": 480.0,
            "departure": 1000.0,
            "risk_quantile": risk_quantile,
            "reserve_buffer_pct": 0.08,
            "non_delivery": 250.0,
            "activation_uncertainty": 100.0,
            "asset_fatigue": 75.0,
        },
        preview_4s=bundle["preview_4s"],
        device_roster=bundle.get("device_roster"),
        execute_lower_mpc=execute_lower_mpc,
    )
    return df, result, bundle


def main() -> None:
    df, upper_only, bundle = run_case(0.80, execute_lower_mpc=False)
    history = upper_only["history"]
    if len(history) != 2:
        raise SystemExit(f"Expected a 30-minute dispatch window to produce 2 intervals, got {len(history)}")
    if not upper_only["tracking_4s"].empty:
        raise SystemExit("Upper-only market decision must not run the lower 4-second MPC.")
    required_cols = {
        "risk_adjusted_profit_eur",
        "non_delivery_risk_cost_eur",
        "market_gate_status",
        "reserve_buffer_kw",
        "bid_block_start",
        "bid_block_end",
        "fcr_bid_mw",
        "afrr_up_bid_mw",
        "afrr_down_bid_mw",
        "fcr_bid_segments",
        "afrr_up_segments",
        "afrr_down_segments",
        "stacking_ok",
        "combined_stacking_model",
        "combined_shared_capacity_ok",
        "combined_split_capacity_ok",
        "combined_socket_rule",
        "bess_combined_up_stack_kw",
        "bess_combined_down_stack_kw",
        "ev_combined_up_stack_kw",
        "ev_combined_down_stack_kw",
        "hvac_combined_up_stack_kw",
        "hvac_combined_down_stack_kw",
        "energy_endurance_ok",
        "granularity_ok",
        "simulation_only_notice",
    }
    if not required_cols.issubset(history.columns):
        raise SystemExit(f"Missing risk-aware upper MPC columns: {sorted(required_cols - set(history.columns))}")
    dynamic_cols = {"hvac_fcr_dynamic_cap_kw", "fcr_response_60_fraction", "fcr_response_180_fraction", "fcr_energy_60_seconds", "fcr_dynamic_response_ok"}
    if not dynamic_cols.issubset(history.columns):
        raise SystemExit(f"Missing FCR-N dynamic deliverability columns: {sorted(dynamic_cols - set(history.columns))}")
    if (history["hvac_fcr_kw"] > history["hvac_fcr_dynamic_cap_kw"] + 1e-6).any():
        raise SystemExit("HVAC FCR-N schedule must stay inside the dynamically deliverable cap.")
    if not history["fcr_dynamic_response_ok"].all():
        raise SystemExit("FCR-N aggregate droop schedule must pass dynamic response checks.")
    if not history["combined_shared_capacity_ok"].all() or not history["combined_split_capacity_ok"].all():
        raise SystemExit("Combined socket summing must be backed by explicit split-capacity audit flags.")
    if (history["combined_stacking_model"] != "co_optimized_split_capacity").any():
        raise SystemExit("Combined optimizer output must label its FCR+aFRR socket as co-optimized split capacity.")
    if (history["socket_up_kw"] + 1e-6 < history["fcr_bid_kw"] + history["afrr_up_bid_kw"]).any():
        raise SystemExit("Audited Combined up socket must cover FCR plus aFRR up.")
    if (history["socket_down_kw"] + 1e-6 < history["fcr_bid_kw"] + history["afrr_down_bid_kw"]).any():
        raise SystemExit("Audited Combined down socket must cover FCR plus aFRR down.")

    lower_from_socket = run_lower_mpc_from_upper_result(
        df=df,
        upper_result=upper_only,
        fleet_meta={
            "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
            "setpoint_c": 21.0,
            "comfort_band_c": 1.0,
            "n_hvac": int(bundle["summary"]["n_hvac"]),
            "hvac_mode": "Inverter / variable-speed",
            "hvac_response_s": 20.0,
        },
        market_mode="Combined",
        resource_mode="Hybrid portfolio",
        preview_4s=bundle["preview_4s"],
        device_roster=bundle.get("device_roster"),
    )
    if lower_from_socket["tracking_4s"].empty:
        raise SystemExit("Live lower MPC attach should produce tracking rows from an upper socket.")
    if not lower_from_socket["summary"].get("execute_lower_mpc"):
        raise SystemExit("Live lower MPC attach should mark lower execution active.")
    partial_lower = run_lower_mpc_from_upper_result(
        df=df,
        upper_result=upper_only,
        fleet_meta={
            "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
            "setpoint_c": 21.0,
            "comfort_band_c": 1.0,
            "n_hvac": int(bundle["summary"]["n_hvac"]),
            "hvac_mode": "Inverter / variable-speed",
            "hvac_response_s": 20.0,
        },
        market_mode="Combined",
        resource_mode="Hybrid portfolio",
        preview_4s=bundle["preview_4s"],
        device_roster=bundle.get("device_roster"),
        elapsed_seconds=8.0,
    )
    if len(partial_lower["tracking_4s"]) > 3:
        raise SystemExit("Live lower MPC attach should not precompute future ticks beyond elapsed market time.")
    required_live_cols = {"requested_signed_kw", "delivered_signed_kw", "optimized_net_load_kw", "baseline_net_load_kw"}
    if not required_live_cols.issubset(partial_lower["tracking_4s"].columns):
        raise SystemExit(f"Live lower tracking is missing visualization columns: {sorted(required_live_cols - set(partial_lower['tracking_4s'].columns))}")
    tick_h = 4.0 / 3600.0
    expected_up_mwh = float(partial_lower["tracking_4s"]["delivered_up_kw"].sum() * tick_h / 1000.0)
    if abs(float(partial_lower["summary"].get("delivered_up_mwh", 0.0)) - expected_up_mwh) > 1e-9:
        raise SystemExit("Partial live lower summary must scale delivered energy by solved 4-second ticks.")

    _, lower, _ = run_case(0.80, execute_lower_mpc=True)
    if lower["tracking_4s"].empty:
        raise SystemExit("Explicit lower-MPC execution should produce 4-second tracking rows.")

    _, p50, _ = run_case(0.50, execute_lower_mpc=False)
    _, p90, _ = run_case(0.90, execute_lower_mpc=False)
    p50_bid = float((p50["history"]["fcr_bid_kw"] + p50["history"]["afrr_up_bid_kw"] + p50["history"]["afrr_down_bid_kw"]).mean())
    p90_bid = float((p90["history"]["fcr_bid_kw"] + p90["history"]["afrr_up_bid_kw"] + p90["history"]["afrr_down_bid_kw"]).mean())
    if p90_bid > p50_bid + 1e-6:
        raise SystemExit(f"P90 reliable bid should not exceed P50 mean-style bid: P50={p50_bid}, P90={p90_bid}")
    fast_hvac_fraction = fcr_n_dynamic_deliverable_fraction(20.0, 4.0)
    slow_hvac_fraction = fcr_n_dynamic_deliverable_fraction(180.0, 4.0)
    if not (0.0 < slow_hvac_fraction < fast_hvac_fraction <= 1.0):
        raise SystemExit("FCR-N dynamic deliverability should derate slow HVAC more than fast inverter HVAC.")

    state = {"bess_soc_mwh": float(df["bess_energy_cap_mwh"].iloc[0]) * 0.95, "ev_soc_delta_mwh": 0.0, "temp_delta_c": 0.0}
    avail = _available_up_down(
        df.iloc[0],
        state,
        {
            "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
            "setpoint_c": 21.0,
            "comfort_band_c": 1.0,
            "n_hvac": 1,
            "hvac_mode": "Inverter / variable-speed",
            "hvac_response_s": 20.0,
        },
    )
    if avail["bess_down_av"] > 1e-6:
        raise SystemExit("A full BESS must not provide downward charging headroom.")

    afrr_only_gate = _market_gate_decision({"bess_afrr_up_kw": 1000.0, "risk_adjusted_profit_eur": 5.0}, "Combined")
    if afrr_only_gate["market_gate_status"] != "Participate":
        raise SystemExit("Combined market mode must allow an aFRR-only bid that clears the aFRR minimum.")
    fcr_min_gate = _market_gate_decision({"bess_fcr_kw": 100.0, "risk_adjusted_profit_eur": 1.0}, "Combined")
    if fcr_min_gate["market_gate_status"] != "Participate":
        raise SystemExit("Combined market mode must allow a 0.1 MW FCR-N bid.")
    fcr_only_gate = _market_gate_decision({"bess_fcr_kw": 200.0, "risk_adjusted_profit_eur": 1.0}, "Combined")
    if fcr_only_gate["market_gate_status"] != "Participate":
        raise SystemExit("Combined market mode must allow an FCR-only bid that clears the FCR-N minimum.")
    misaligned_fcr_gate = _market_gate_decision({"bess_fcr_kw": 120.0, "risk_adjusted_profit_eur": 1.0}, "Combined")
    if misaligned_fcr_gate["market_gate_status"] != "Wait":
        raise SystemExit("Combined market mode must reject FCR-N bids that miss the 0.1 MW granularity.")
    misaligned_afrr_gate = _market_gate_decision({"bess_afrr_up_kw": 1200.0, "risk_adjusted_profit_eur": 5.0}, "Combined")
    if misaligned_afrr_gate["market_gate_status"] != "Wait":
        raise SystemExit("Combined market mode must reject aFRR bids that miss the 1 MW granularity.")
    mixed_invalid_gate = _market_gate_decision({"bess_fcr_kw": 100.0, "bess_afrr_up_kw": 1200.0, "risk_adjusted_profit_eur": 5.0}, "Combined")
    if mixed_invalid_gate["market_gate_status"] != "Wait":
        raise SystemExit("Combined market mode must not accept a valid FCR-N bid while a nonzero aFRR bid is misaligned.")
    unsafe_combined_gate = _market_gate_decision(
        {
            "bess_fcr_kw": 100.0,
            "bess_afrr_up_kw": 1000.0,
            "combined_shared_capacity_ok": False,
            "risk_adjusted_profit_eur": 5.0,
        },
        "Combined",
    )
    if unsafe_combined_gate["market_gate_status"] != "Wait":
        raise SystemExit("Combined market gate must reject FCR+aFRR bids that fail the split-capacity stacking audit.")
    afrr_below_min_gate = _market_gate_decision({"bess_afrr_up_kw": 900.0, "risk_adjusted_profit_eur": 5.0}, "Combined")
    if afrr_below_min_gate["market_gate_status"] != "Wait":
        raise SystemExit("Combined market mode must reject aFRR bids below the 1 MW minimum.")
    hvac_share_gate = _market_gate_decision(
        {
            "bess_fcr_kw": 100.0,
            "hvac_fcr_kw": 100.0,
            "hvac_fcr_dynamic_cap_kw": 200.0,
            "fcr_dynamic_response_ok": True,
            "risk_adjusted_profit_eur": 1.0,
        },
        "FCR-N",
    )
    if hvac_share_gate["market_gate_status"] != "Wait":
        raise SystemExit("FCR-N gate must reject HVAC bids that exceed the configured comfort share cap.")
    tiny_gate = _market_gate_decision({"bess_fcr_kw": 20.0, "bess_afrr_up_kw": 200.0, "risk_adjusted_profit_eur": 5.0}, "Combined")
    if tiny_gate["market_gate_status"] != "Wait":
        raise SystemExit("Combined market mode must still reject bids below both product minimum sizes.")

    print("risk-aware MPC smoke checks passed")
    print(f"P50 mean bid kW: {p50_bid:.2f}")
    print(f"P90 reliable bid kW: {p90_bid:.2f}")


if __name__ == "__main__":
    main()
