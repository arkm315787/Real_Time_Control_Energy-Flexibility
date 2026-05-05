"""Smoke-test risk-aware upper MPC and optional lower-MPC execution."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexihome.core.engine import SUPPORTED_MPC_TARGETS, _available_up_down, generate_synthetic_portfolio, run_mpc_controller


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
            "degradation": 18.0,
            "comfort": 120.0,
            "departure": 160.0,
            "risk_quantile": risk_quantile,
            "reserve_buffer_pct": 0.08,
            "non_delivery": 250.0,
            "activation_uncertainty": 90.0,
            "asset_fatigue": 30.0,
        },
        preview_4s=bundle["preview_4s"],
        device_roster=bundle.get("device_roster"),
        execute_lower_mpc=execute_lower_mpc,
    )
    return df, result


def main() -> None:
    df, upper_only = run_case(0.80, execute_lower_mpc=False)
    history = upper_only["history"]
    if len(history) != 2:
        raise SystemExit(f"Expected a 30-minute dispatch window to produce 2 intervals, got {len(history)}")
    if not upper_only["tracking_4s"].empty:
        raise SystemExit("Upper-only market decision must not run the lower 4-second MPC.")
    required_cols = {"risk_adjusted_profit_eur", "non_delivery_risk_cost_eur", "market_gate_status", "reserve_buffer_kw"}
    if not required_cols.issubset(history.columns):
        raise SystemExit(f"Missing risk-aware upper MPC columns: {sorted(required_cols - set(history.columns))}")

    _, lower = run_case(0.80, execute_lower_mpc=True)
    if lower["tracking_4s"].empty:
        raise SystemExit("Explicit lower-MPC execution should produce 4-second tracking rows.")

    _, p50 = run_case(0.50, execute_lower_mpc=False)
    _, p90 = run_case(0.90, execute_lower_mpc=False)
    p50_bid = float((p50["history"]["fcr_bid_kw"] + p50["history"]["afrr_up_bid_kw"] + p50["history"]["afrr_down_bid_kw"]).mean())
    p90_bid = float((p90["history"]["fcr_bid_kw"] + p90["history"]["afrr_up_bid_kw"] + p90["history"]["afrr_down_bid_kw"]).mean())
    if p90_bid > p50_bid + 1e-6:
        raise SystemExit(f"P90 reliable bid should not exceed P50 mean-style bid: P50={p50_bid}, P90={p90_bid}")

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

    print("risk-aware MPC smoke checks passed")
    print(f"P50 mean bid kW: {p50_bid:.2f}")
    print(f"P90 reliable bid kW: {p90_bid:.2f}")


if __name__ == "__main__":
    main()
