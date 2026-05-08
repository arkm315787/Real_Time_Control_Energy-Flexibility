"""Multi-scenario smoke test for the VPP MVP market/control story.

This catches the failures that a single happy-path smoke misses: market
granularity, Combined socket stacking, HVAC FCR caps, resource revenue
attribution, and BESS/EV bridging when slow HVAC is committed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.core.data import generate_synthetic_portfolio
from flexihome.core.engine import SUPPORTED_MPC_TARGETS, run_mpc_controller


class PersistenceForecaster:
    def __init__(self, target: str, predictors: list[str]) -> None:
        self.target = target
        self.predictors = predictors
        self.lags = 1

    def predict(self, row: pd.DataFrame, horizon_steps: int = 1):
        return [float(row[self.target].iloc[0])]


def make_models(df: pd.DataFrame) -> dict[str, PersistenceForecaster]:
    predictors = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return {target: PersistenceForecaster(target, predictors) for target in SUPPORTED_MPC_TARGETS}


def base_weights() -> dict[str, float]:
    return {
        "degradation": 35.0,
        "comfort": 480.0,
        "departure": 1000.0,
        "risk_quantile": 0.80,
        "reserve_buffer_pct": 0.08,
        "non_delivery": 250.0,
        "activation_uncertainty": 100.0,
        "asset_fatigue": 75.0,
        "risk_policy": "investor_balanced",
    }


def build_case(
    *,
    seed: int,
    n_homes: int,
    ev_pen: float,
    bess_pen: float,
    hvac_pen: float,
    pv_pen: float,
) -> dict[str, object]:
    bundle = generate_synthetic_portfolio(
        start_date="2026-01-15",
        days=1,
        n_homes=n_homes,
        ev_pen=ev_pen,
        bess_pen=bess_pen,
        pv_pen=pv_pen,
        hvac_pen=hvac_pen,
        cloudiness=0.55,
        climate_shift_c=0.0,
        solar_scale=1.0,
        seed=seed,
        setpoint_c=21.0,
        comfort_band_c=1.0,
        scarcity=1.8,
        freq_minutes=15,
        hvac_mode="Inverter / variable-speed",
    )
    df = bundle["data"]
    df[["fcrn_capacity_eur_per_mw_h", "afrr_up_capacity_eur_per_mw_h", "afrr_down_capacity_eur_per_mw_h"]] = 1200.0
    df["afrr_up_energy_eur_per_mwh"] = 250.0
    df["afrr_down_energy_eur_per_mwh"] = 180.0
    return bundle


def run_case(name: str, market_mode: str, resource_mode: str, execute_lower_mpc: bool, **portfolio_kwargs):
    bundle = build_case(**portfolio_kwargs)
    df = bundle["data"]
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
        market_mode=market_mode,
        resource_mode=resource_mode,
        horizon_hours=0.5,
        dispatch_hours=0.25,
        penalty_weights=base_weights(),
        preview_4s=bundle["preview_4s"],
        device_roster=bundle.get("device_roster"),
        inner_controller_mode="mpc",
        inner_dt_seconds=4,
        inner_mpc_horizon_seconds=4,
        execute_lower_mpc=execute_lower_mpc,
    )
    history = result["history"]
    if history.empty:
        raise SystemExit(f"{name}: expected at least one upper-MPC interval.")
    if history["solver_status"].ne("Optimal").any():
        raise SystemExit(f"{name}: upper optimizer did not report Optimal for every interval.")
    return bundle, result


def assert_participates(name: str, history: pd.DataFrame) -> pd.DataFrame:
    accepted = history[history["market_gate_status"] == "Participate"]
    if accepted.empty:
        reasons = history["market_gate_reason"].dropna().unique().tolist()
        raise SystemExit(f"{name}: no market interval participated. Reasons: {reasons}")
    return accepted


def is_multiple_kw(value: float, granularity_kw: float) -> bool:
    if abs(value) <= 1e-6:
        return True
    return abs((value / granularity_kw) - round(value / granularity_kw)) <= 1e-6


def assert_market_granularity(name: str, accepted: pd.DataFrame) -> None:
    for _, row in accepted.iterrows():
        fcr = float(row["fcr_bid_kw"])
        afrr_up = float(row["afrr_up_bid_kw"])
        afrr_down = float(row["afrr_down_bid_kw"])
        if fcr > 0.0 and (fcr < 100.0 or not is_multiple_kw(fcr, 100.0)):
            raise SystemExit(f"{name}: invalid FCR-N bid granularity {fcr:.3f} kW.")
        if afrr_up > 0.0 and (afrr_up < 1000.0 or not is_multiple_kw(afrr_up, 1000.0)):
            raise SystemExit(f"{name}: invalid aFRR up bid granularity {afrr_up:.3f} kW.")
        if afrr_down > 0.0 and (afrr_down < 1000.0 or not is_multiple_kw(afrr_down, 1000.0)):
            raise SystemExit(f"{name}: invalid aFRR down bid granularity {afrr_down:.3f} kW.")
        if int(row["fcr_bid_segments"]) != int(round(fcr / 100.0)):
            raise SystemExit(f"{name}: FCR segment metadata does not match final bid.")
        if int(row["afrr_up_segments"]) != int(round(afrr_up / 1000.0)):
            raise SystemExit(f"{name}: aFRR up segment metadata does not match final bid.")
        if int(row["afrr_down_segments"]) != int(round(afrr_down / 1000.0)):
            raise SystemExit(f"{name}: aFRR down segment metadata does not match final bid.")


def assert_combined_socket(name: str, accepted: pd.DataFrame) -> None:
    for _, row in accepted.iterrows():
        if str(row.get("combined_stacking_model", "")) != "co_optimized_split_capacity":
            raise SystemExit(f"{name}: accepted Combined interval is missing co-optimized split-capacity proof.")
        if not bool(row.get("combined_shared_capacity_ok", False)):
            raise SystemExit(f"{name}: accepted Combined interval failed the shared-capacity stacking audit.")
        up_required = float(row["fcr_bid_kw"] + row["afrr_up_bid_kw"])
        down_required = float(row["fcr_bid_kw"] + row["afrr_down_bid_kw"])
        if float(row["socket_up_kw"]) + 1e-6 < up_required:
            raise SystemExit(f"{name}: Combined up socket is smaller than FCR+aFRR.")
        if float(row["socket_down_kw"]) + 1e-6 < down_required:
            raise SystemExit(f"{name}: Combined down socket is smaller than FCR+aFRR.")
        if not bool(row["stacking_ok"]):
            raise SystemExit(f"{name}: Combined stacking flag is false for an accepted interval.")


def assert_fcr_hvac_policy(name: str, history: pd.DataFrame) -> None:
    if (history["hvac_fcr_kw"] > history["hvac_fcr_dynamic_cap_kw"] + 1e-6).any():
        raise SystemExit(f"{name}: HVAC FCR-N exceeds dynamic deliverable cap.")
    if not history["fcr_hvac_share_ok"].all():
        raise SystemExit(f"{name}: HVAC FCR-N share cap is violated.")


def assert_revenue_attribution(name: str, result: dict[str, object]) -> None:
    summary = result["summary"]
    resource_revenue = float(sum(summary["resource_revenue"].values()))
    market_revenue = float(summary["capacity_revenue_eur"] + summary["activation_revenue_eur"])
    tolerance = max(0.02, abs(market_revenue) * 0.02)
    if abs(resource_revenue - market_revenue) > tolerance:
        raise SystemExit(
            f"{name}: resource revenue attribution {resource_revenue:.3f} EUR "
            f"does not reconcile to market revenue {market_revenue:.3f} EUR."
        )


def assert_fast_bridge(name: str, result: dict[str, object]) -> None:
    tracking = result["tracking_4s"]
    if tracking.empty:
        raise SystemExit(f"{name}: lower 4-second tracking did not run.")
    slow_abs = tracking["hvac_up_kw"] + tracking["hvac_down_kw"] + tracking["pv_down_kw"]
    fast_abs = tracking["bess_up_kw"] + tracking["bess_down_kw"] + tracking["ev_up_kw"] + tracking["ev_down_kw"]
    positive_request = (tracking["requested_up_kw"] + tracking["requested_down_kw"]) > 25.0
    if slow_abs.max() > 25.0 and tracking["fast_bridge_used_kw"].max() <= 0.0:
        raise SystemExit(f"{name}: slow HVAC/PV commitment was not bridged by fast active capacity.")
    first_active = tracking[positive_request].head(12)
    if not first_active.empty:
        fast_share = float(fast_abs.loc[first_active.index].sum() / max((fast_abs.loc[first_active.index] + slow_abs.loc[first_active.index]).sum(), 1.0))
        if fast_share < 0.20:
            raise SystemExit(f"{name}: first 4-second ticks are still too slow-resource dominated ({fast_share:.1%} fast share).")


def assert_fast_only_has_no_slow_dispatch(name: str, result: dict[str, object]) -> None:
    tracking = result["tracking_4s"]
    if tracking.empty:
        raise SystemExit(f"{name}: lower 4-second tracking did not run.")
    slow_total = float((tracking["hvac_up_kw"] + tracking["hvac_down_kw"] + tracking["pv_down_kw"]).abs().sum())
    if slow_total > 1e-6:
        raise SystemExit(f"{name}: Fast-only scenario dispatched HVAC/PV slow resources.")


def main() -> None:
    scenarios = []

    scenarios.append(
        (
            "Combined hybrid lower",
            *run_case(
                "Combined hybrid lower",
                "Combined",
                "Hybrid portfolio",
                True,
                seed=7,
                n_homes=1500,
                ev_pen=0.35,
                bess_pen=0.30,
                hvac_pen=0.80,
                pv_pen=0.45,
            ),
        )
    )
    scenarios.append(
        (
            "FCR-N hybrid upper",
            *run_case(
                "FCR-N hybrid upper",
                "FCR-N",
                "Hybrid portfolio",
                False,
                seed=11,
                n_homes=1200,
                ev_pen=0.35,
                bess_pen=0.35,
                hvac_pen=0.70,
                pv_pen=0.30,
            ),
        )
    )
    scenarios.append(
        (
            "aFRR hybrid upper",
            *run_case(
                "aFRR hybrid upper",
                "aFRR",
                "Hybrid portfolio",
                False,
                seed=13,
                n_homes=1600,
                ev_pen=0.40,
                bess_pen=0.35,
                hvac_pen=0.75,
                pv_pen=0.45,
            ),
        )
    )
    scenarios.append(
        (
            "Combined fast-only lower",
            *run_case(
                "Combined fast-only lower",
                "Combined",
                "Fast only",
                True,
                seed=17,
                n_homes=1400,
                ev_pen=0.60,
                bess_pen=0.55,
                hvac_pen=0.25,
                pv_pen=0.20,
            ),
        )
    )

    for name, _, result in scenarios:
        history = result["history"]
        accepted = assert_participates(name, history)
        assert_market_granularity(name, accepted)
        assert_revenue_attribution(name, result)
        if "Combined" in name:
            assert_combined_socket(name, accepted)
        if "FCR-N" in name or "Combined" in name:
            assert_fcr_hvac_policy(name, history)
        if "hybrid lower" in name:
            assert_fast_bridge(name, result)
        if "fast-only lower" in name:
            assert_fast_only_has_no_slow_dispatch(name, result)

    print("multi-scenario VPP MVP smoke checks passed")
    for name, _, result in scenarios:
        history = result["history"]
        accepted = int((history["market_gate_status"] == "Participate").sum())
        max_bid_kw = float((history["fcr_bid_kw"] + history["afrr_up_bid_kw"] + history["afrr_down_bid_kw"]).max())
        print(f"{name}: accepted={accepted}, max stacked bid={max_bid_kw:.1f} kW")


if __name__ == "__main__":
    main()
