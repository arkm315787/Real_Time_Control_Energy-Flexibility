"""Five-context technical compliance smoke for the VPP simulation MVP.

The goal is not to claim live Fingrid certification. The smoke verifies that
synthetic 4-second simulations expose aFRR timing, aFRR accuracy, aFRR energy
reconciliation, FCR-N step response, FCR-N stability screening, and dashboard
plot columns instead of hiding them behind interval averages.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

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


def base_weights() -> dict[str, float | str]:
    return {
        "degradation": 30.0,
        "comfort": 420.0,
        "departure": 1000.0,
        "risk_quantile": 0.75,
        "reserve_buffer_pct": 0.06,
        "non_delivery": 250.0,
        "activation_uncertainty": 100.0,
        "asset_fatigue": 70.0,
        "risk_policy": "investor_balanced",
    }


def force_step_signals(bundle: dict[str, object], *, fcr_signal: float, afrr_signal: float, minutes: int = 30) -> None:
    df = bundle["data"]
    preview = bundle["preview_4s"].copy()
    start = pd.Timestamp(df.index[0])
    end = start + pd.Timedelta(minutes=minutes)

    preview[["fcr_signal_norm", "afrr_signal_norm"]] = 0.0
    preview["frequency_hz"] = 50.0
    active_preview = (preview.index >= start) & (preview.index < end)
    preview.loc[active_preview, "fcr_signal_norm"] = float(fcr_signal)
    preview.loc[active_preview, "afrr_signal_norm"] = float(afrr_signal)
    preview.loc[active_preview, "frequency_hz"] = 50.0 - 0.1 * float(fcr_signal)
    bundle["preview_4s"] = preview

    active_outer = (df.index >= start) & (df.index < end)
    df.loc[active_outer, "fcr_signed_act"] = float(fcr_signal)
    df.loc[active_outer, "fcr_up_act_frac"] = max(float(fcr_signal), 0.0)
    df.loc[active_outer, "fcr_down_act_frac"] = max(-float(fcr_signal), 0.0)
    df.loc[active_outer, "afrr_signal_norm"] = float(afrr_signal)
    df.loc[active_outer, "afrr_up_act_frac"] = max(float(afrr_signal), 0.0)
    df.loc[active_outer, "afrr_down_act_frac"] = max(-float(afrr_signal), 0.0)
    df.loc[active_outer, "frequency_hz"] = 50.0 - 0.1 * float(fcr_signal)


def build_bundle(*, seed: int, n_homes: int, ev_pen: float, bess_pen: float, hvac_pen: float, pv_pen: float, hvac_response_s: float) -> dict[str, object]:
    bundle = generate_synthetic_portfolio(
        start_date="2026-01-15",
        days=1,
        n_homes=n_homes,
        ev_pen=ev_pen,
        bess_pen=bess_pen,
        pv_pen=pv_pen,
        hvac_pen=hvac_pen,
        cloudiness=0.50,
        climate_shift_c=0.0,
        solar_scale=1.0,
        seed=seed,
        setpoint_c=21.0,
        comfort_band_c=1.0,
        scarcity=2.0,
        freq_minutes=15,
        hvac_mode="Inverter / variable-speed",
        hvac_response_s=hvac_response_s,
    )
    df = bundle["data"]
    df[["fcrn_capacity_eur_per_mw_h", "afrr_up_capacity_eur_per_mw_h", "afrr_down_capacity_eur_per_mw_h"]] = 1400.0
    df["afrr_up_energy_eur_per_mwh"] = 260.0
    df["afrr_down_energy_eur_per_mwh"] = 190.0
    return bundle


def run_case(
    name: str,
    *,
    market_mode: str,
    resource_mode: str,
    fcr_signal: float,
    afrr_signal: float,
    seed: int,
    n_homes: int,
    ev_pen: float,
    bess_pen: float,
    hvac_pen: float,
    pv_pen: float,
    hvac_response_s: float = 20.0,
    fcr_hvac_share_cap: float = 0.20,
) -> dict[str, object]:
    bundle = build_bundle(
        seed=seed,
        n_homes=n_homes,
        ev_pen=ev_pen,
        bess_pen=bess_pen,
        hvac_pen=hvac_pen,
        pv_pen=pv_pen,
        hvac_response_s=hvac_response_s,
    )
    force_step_signals(bundle, fcr_signal=fcr_signal, afrr_signal=afrr_signal)
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
            "hvac_response_s": hvac_response_s,
            "fcr_hvac_share_cap": fcr_hvac_share_cap,
        },
        market_mode=market_mode,
        resource_mode=resource_mode,
        horizon_hours=0.25,
        dispatch_hours=0.25,
        penalty_weights=base_weights(),
        preview_4s=bundle["preview_4s"],
        device_roster=bundle.get("device_roster"),
        inner_controller_mode="mpc",
        inner_dt_seconds=4,
        inner_mpc_horizon_seconds=4,
        execute_lower_mpc=True,
    )
    result["_case_name"] = name
    return result


def audit_plot_figures(result: dict[str, object]) -> dict[str, go.Figure]:
    tracking = result["tracking_4s"]
    figures: dict[str, go.Figure] = {}
    signed_request = tracking["requested_up_kw"] - tracking["requested_down_kw"]
    signed_delivered = tracking["delivered_up_kw"] - tracking["delivered_down_kw"]
    tracking_fig = go.Figure()
    tracking_fig.add_trace(go.Scatter(x=tracking.index, y=signed_request, name="requested_kw"))
    tracking_fig.add_trace(go.Scatter(x=tracking.index, y=signed_delivered, name="delivered_kw"))
    figures["request_delivery"] = tracking_fig

    if {"afrr_accuracy_ratio", "afrr_accuracy_low", "afrr_accuracy_high"}.issubset(tracking.columns):
        accuracy_fig = go.Figure()
        accuracy_fig.add_trace(go.Scatter(x=tracking.index, y=tracking["afrr_accuracy_ratio"], name="afrr_ratio"))
        accuracy_fig.add_trace(go.Scatter(x=tracking.index, y=tracking["afrr_accuracy_low"], name="low"))
        accuracy_fig.add_trace(go.Scatter(x=tracking.index, y=tracking["afrr_accuracy_high"], name="high"))
        figures["afrr_accuracy"] = accuracy_fig

    energy_audit = result.get("afrr_energy_audit", pd.DataFrame())
    if isinstance(energy_audit, pd.DataFrame) and not energy_audit.empty:
        energy_fig = go.Figure()
        energy_fig.add_trace(go.Bar(x=energy_audit["isp_start"], y=energy_audit["afrr_reported_energy_mwh"], name="reported"))
        energy_fig.add_trace(go.Bar(x=energy_audit["isp_start"], y=energy_audit["afrr_fingrid_calculated_energy_mwh"], name="calculated"))
        figures["afrr_energy"] = energy_fig
    return figures


def assert_plot_ready(name: str, result: dict[str, object], *, expect_afrr: bool) -> None:
    tracking = result["tracking_4s"]
    required = {
        "requested_signed_kw",
        "delivered_signed_kw",
        "afrr_accuracy_ratio",
        "afrr_reporting_error_kw",
        "fcr_response_ratio",
    }
    missing = required - set(tracking.columns)
    if missing:
        raise SystemExit(f"{name}: tracking frame is missing plot/audit columns {sorted(missing)}")
    figures = audit_plot_figures(result)
    if len(figures.get("request_delivery", go.Figure()).data) < 2:
        raise SystemExit(f"{name}: request/delivery plot does not contain both traces.")
    if expect_afrr and len(figures.get("afrr_accuracy", go.Figure()).data) < 3:
        raise SystemExit(f"{name}: aFRR accuracy plot is not render-ready.")
    if expect_afrr and len(figures.get("afrr_energy", go.Figure()).data) < 2:
        raise SystemExit(f"{name}: aFRR energy reconciliation plot is not render-ready.")


def assert_case(name: str, result: dict[str, object], *, expect_afrr: bool, expect_fcr: bool) -> None:
    history = result["history"]
    tracking = result["tracking_4s"]
    compliance = result["compliance"]
    if history.empty or tracking.empty:
        raise SystemExit(f"{name}: expected populated history and 4-second tracking.")
    if int((history["market_gate_status"] == "Participate").sum()) < 1:
        raise SystemExit(f"{name}: expected at least one cleared synthetic market interval.")
    assert_plot_ready(name, result, expect_afrr=expect_afrr)

    if expect_afrr:
        for key in [
            "afrr_start_within_30s_ok",
            "afrr_full_activation_5min_ok",
            "accuracy_ok",
            "afrr_energy_reporting_ok",
            "afrr_settlement_reconciliation_ok",
        ]:
            if not bool(compliance.get(key, False)):
                raise SystemExit(f"{name}: expected {key} to pass, got {compliance.get(key)!r}.")
        if int(compliance.get("afrr_energy_reporting_periods", 0)) < 1:
            raise SystemExit(f"{name}: expected at least one aFRR settlement-period energy audit.")

    if expect_fcr:
        for key in ["fcr_response_ok", "fcr_actual_dynamic_response_ok", "fcr_stability_margin_ok", "fcr_prequalification_evidence_ok"]:
            if not bool(compliance.get(key, False)):
                raise SystemExit(f"{name}: expected {key} to pass, got {compliance.get(key)!r}.")
        if not bool(compliance.get("fcr_tracking_response_evaluated", False)):
            raise SystemExit(f"{name}: expected FCR 60s/180s tracking response to be evaluated.")


def assert_slow_hvac_stress(result: dict[str, object]) -> None:
    name = str(result.get("_case_name", "Slow HVAC stress"))
    history = result["history"]
    compliance = result["compliance"]
    if history.empty:
        raise SystemExit(f"{name}: expected history for stress case.")
    if (history["hvac_fcr_kw"] > history["hvac_fcr_dynamic_cap_kw"] + 1e-6).any():
        raise SystemExit(f"{name}: slow HVAC exceeded the dynamic FCR cap.")
    if not bool(compliance.get("fcr_stability_screen_evaluated", False)):
        raise SystemExit(f"{name}: FCR stability screen was not evaluated.")
    if not bool(compliance.get("fcr_stability_margin_ok", True)) and bool(compliance.get("fcr_prequalification_evidence_ok", True)):
        raise SystemExit(f"{name}: failed stability screen must not be presented as prequalification-ready.")
    assert_plot_ready(name, result, expect_afrr=True)


def main() -> None:
    cases = [
        (
            "Combined up sustained aFRR socket",
            run_case(
                "Combined up sustained aFRR socket",
                market_mode="Combined",
                resource_mode="Hybrid portfolio",
                fcr_signal=1.0,
                afrr_signal=1.0,
                seed=21,
                n_homes=1400,
                ev_pen=0.45,
                bess_pen=0.45,
                hvac_pen=0.65,
                pv_pen=0.35,
            ),
            True,
            False,
        ),
        (
            "Combined down sustained aFRR socket",
            run_case(
                "Combined down sustained aFRR socket",
                market_mode="Combined",
                resource_mode="Hybrid portfolio",
                fcr_signal=-1.0,
                afrr_signal=-1.0,
                seed=22,
                n_homes=1500,
                ev_pen=0.40,
                bess_pen=0.45,
                hvac_pen=0.70,
                pv_pen=0.50,
            ),
            True,
            False,
        ),
        (
            "FCR-N positive frequency step",
            run_case(
                "FCR-N positive frequency step",
                market_mode="FCR-N",
                resource_mode="Hybrid portfolio",
                fcr_signal=1.0,
                afrr_signal=0.0,
                seed=23,
                n_homes=1100,
                ev_pen=0.45,
                bess_pen=0.50,
                hvac_pen=0.55,
                pv_pen=0.20,
            ),
            False,
            True,
        ),
        (
            "aFRR fast-storage only",
            run_case(
                "aFRR fast-storage only",
                market_mode="aFRR",
                resource_mode="Fast only",
                fcr_signal=0.0,
                afrr_signal=1.0,
                seed=24,
                n_homes=1300,
                ev_pen=0.60,
                bess_pen=0.55,
                hvac_pen=0.20,
                pv_pen=0.15,
            ),
            True,
            False,
        ),
    ]

    for name, result, expect_afrr, expect_fcr in cases:
        assert_case(name, result, expect_afrr=expect_afrr, expect_fcr=expect_fcr)

    stress = run_case(
        "Slow-HVAC stress screen",
        market_mode="Combined",
        resource_mode="Hybrid portfolio",
        fcr_signal=1.0,
        afrr_signal=1.0,
        seed=25,
        n_homes=1200,
        ev_pen=0.18,
        bess_pen=0.18,
        hvac_pen=0.90,
        pv_pen=0.40,
        hvac_response_s=260.0,
        fcr_hvac_share_cap=0.60,
    )
    assert_slow_hvac_stress(stress)

    print("technical compliance smoke checks passed")
    for name, result, _, _ in cases + [("Slow-HVAC stress screen", stress, True, True)]:
        compliance = result["compliance"]
        print(
            f"{name}: score={result['summary'].get('requirement_score_pct', 0.0):.1f}%, "
            f"afrr_energy_diff={float(compliance.get('afrr_energy_reporting_max_diff_pct', 0.0) or 0.0):.3f}, "
            f"fcr_stability={float(compliance.get('fcr_stability_margin_pct', 0.0) or 0.0):.1f}%"
        )


if __name__ == "__main__":
    main()
