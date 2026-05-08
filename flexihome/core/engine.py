"""Core FlexiHome modeling, forecasting, and optimization logic.

This module is intentionally free of Streamlit imports so it can be shared by
the Streamlit dashboard, FastAPI service, tests, and future worker processes.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from pulp import LpInteger, LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum
from scipy import signal
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .centralized_controller import (
    CentralizedVPPController,
    InnerControllerConfig,
    generate_household_device_roster,
    representative_roster_from_frame,
)
from .market_data import overlay_real_market_data

FINGRID_RULES = {
    "FCR-N": {
        "min_bid_kw": 100.0,
        "bid_granularity_kw": 100.0,
        "max_bid_kw_per_timeseries": 5000.0,
        "band_hz": 0.1,
        "activation_60_fraction": 0.63,
        "activation_180_fraction": 0.95,
        "energy_60_seconds": 24.0,
        "response_63_seconds": 60.0,
        "response_95_seconds": 180.0,
        "storage_endurance_hours": 1.0,
        "steady_state_under_delivery_tolerance": 0.05,
        "steady_state_over_delivery_tolerance": 0.20,
        "market_period_minutes": 60,
        "control_source": "local_frequency_measurement",
        "symmetric": True,
        "description": "Symmetric local frequency response around 50 Hz, with no TSO activation signal.",
    },
    "aFRR": {
        "min_bid_kw": 1000.0,
        "bid_granularity_kw": 1000.0,
        "signal_interval_seconds": 4.0,
        "start_seconds": 30.0,
        "full_activation_seconds": 300.0,
        "market_period_minutes": 15,
        "control_source": "tso_activation_signal",
        "accuracy_low": 0.90,
        "accuracy_high": 1.10,
        "storage_endurance_hours": 1.0,
        "description": "Centralized 4-second signal with full activation inside 5 minutes.",
    },
}

RESOURCE_RESPONSE_SECONDS = {
    "BESS": 2.0,
    "EV": 4.0,
    "HVAC": 20.0,
    "PV": 5.0,
}

FCR_N_LOCAL_CONTROL_TYPES = {"BESS", "EV", "HVAC"}

RESOURCE_FORECAST_UNCERTAINTY = {
    "BESS": 0.08,
    "EV": 0.18,
    "HVAC": 0.24,
    "PV": 0.20,
}

COMBINED_STACKING_SPLIT_MODEL = "co_optimized_split_capacity"
COMBINED_STACKING_SHARED_MODEL = "shared_capacity_max_socket"

SUPPORTED_MPC_TARGETS = [
    "net_load_baseline_kw",
    "pv_available_kw",
    "fcrn_capacity_eur_per_mw_h",
    "afrr_up_capacity_eur_per_mw_h",
    "afrr_down_capacity_eur_per_mw_h",
    "afrr_up_act_frac",
    "afrr_down_act_frac",
    "fcr_signed_act",
]

HVAC_MODES = ["Conventional / thermostat-led", "Inverter / variable-speed"]

HVAC_FAST_FRACTION = {
    "Conventional / thermostat-led": 0.10,
    "Inverter / variable-speed": 0.80,
}

HVAC_DEFAULT_RESPONSE_S = {
    "Conventional / thermostat-led": 180.0,
    "Inverter / variable-speed": 20.0,
}

SIMULATION_ONLY_NOTICE = "Simulation and prequalification-prep output only; not a live Fingrid BSP bid, dispatch instruction, or settlement record."
PREQUALIFICATION_PREP_NOTICE = (
    "Simulation screening only; formal Fingrid prequalification still requires approved test data, "
    "measurement evidence, telemetry, baseline validation, settlement setup, and TSO review."
)
FORECAST_QUANTILES = (0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95)
PRICE_FORECAST_TARGETS = {
    "fcrn_capacity_eur_per_mw_h",
    "afrr_up_capacity_eur_per_mw_h",
    "afrr_down_capacity_eur_per_mw_h",
}

RISK_POLICY_PRESETS = {
    "investor_balanced": {
        "label": "Investor Balanced",
        "risk_quantile": 0.70,
        "reserve_buffer_pct": 0.06,
        "non_delivery": 500.0,
        "activation_uncertainty": 100.0,
        "asset_fatigue": 75.0,
    },
    "technical_conservative": {
        "label": "Technical Conservative",
        "risk_quantile": 0.85,
        "reserve_buffer_pct": 0.12,
        "non_delivery": 1200.0,
        "activation_uncertainty": 150.0,
        "asset_fatigue": 125.0,
    },
    "stress_test": {
        "label": "Stress Test",
        "risk_quantile": 0.95,
        "reserve_buffer_pct": 0.25,
        "non_delivery": 2500.0,
        "activation_uncertainty": 300.0,
        "asset_fatigue": 250.0,
    },
}

FCR_N_HVAC_SHARE_CAP = 0.20
FCR_N_DEFAULT_DELAY_S = {
    "BESS": 0.0,
    "EV": 0.0,
    "HVAC": 4.0,
    "PV": 0.0,
}

@dataclass
class ForecastSpec:
    target: str
    predictors: List[str]
    lags: int
    horizon_steps: int
    model: XGBRegressor
    metrics: Dict[str, float]
    residual_quantiles: Dict[str, float] | None = None

def hvac_fast_fraction(hvac_mode: str) -> float:
    return float(HVAC_FAST_FRACTION.get(hvac_mode, HVAC_FAST_FRACTION[HVAC_MODES[0]]))

def default_hvac_response_seconds(hvac_mode: str) -> float:
    return float(HVAC_DEFAULT_RESPONSE_S.get(hvac_mode, HVAC_DEFAULT_RESPONSE_S[HVAC_MODES[0]]))

def _first_order_step_fraction(response_seconds: float, elapsed_seconds: float, delay_seconds: float = 0.0) -> float:
    effective_seconds = max(float(elapsed_seconds) - max(float(delay_seconds), 0.0), 0.0)
    if effective_seconds <= 0.0:
        return 0.0
    tau = max(float(response_seconds), 1e-6)
    return float(np.clip(1.0 - math.exp(-effective_seconds / tau), 0.0, 1.0))

def _first_order_step_energy_seconds(response_seconds: float, elapsed_seconds: float, delay_seconds: float = 0.0) -> float:
    effective_seconds = max(float(elapsed_seconds) - max(float(delay_seconds), 0.0), 0.0)
    if effective_seconds <= 0.0:
        return 0.0
    tau = max(float(response_seconds), 1e-6)
    return float(max(effective_seconds - tau * (1.0 - math.exp(-effective_seconds / tau)), 0.0))

def fcr_n_dynamic_deliverable_fraction(response_seconds: float, delay_seconds: float = 0.0) -> float:
    """Map steady-state FCR-N capacity to dynamically deliverable capacity."""

    rules = FINGRID_RULES["FCR-N"]
    frac_60 = _first_order_step_fraction(response_seconds, rules["response_63_seconds"], delay_seconds)
    frac_180 = _first_order_step_fraction(response_seconds, rules["response_95_seconds"], delay_seconds)
    energy_60_s = _first_order_step_energy_seconds(response_seconds, rules["response_63_seconds"], delay_seconds)
    caps = [
        frac_60 / max(float(rules["activation_60_fraction"]), 1e-9),
        frac_180 / max(float(rules["activation_180_fraction"]), 1e-9),
        energy_60_s / max(float(rules["energy_60_seconds"]), 1e-9),
        1.0,
    ]
    return float(np.clip(min(caps), 0.0, 1.0))

def fcr_n_dynamic_deliverable_kw(
    up_kw: float,
    down_kw: float,
    response_seconds: float,
    delay_seconds: float = 0.0,
) -> float:
    symmetric_kw = min(max(float(up_kw), 0.0), max(float(down_kw), 0.0))
    return float(symmetric_kw * fcr_n_dynamic_deliverable_fraction(response_seconds, delay_seconds))

def _fcr_n_resource_response_seconds(resource: str, fleet_meta: Dict[str, Any]) -> float:
    resource_key = str(resource).upper()
    if resource_key == "HVAC":
        return float(fleet_meta.get("hvac_response_s", RESOURCE_RESPONSE_SECONDS["HVAC"]))
    return float(RESOURCE_RESPONSE_SECONDS.get(resource_key, RESOURCE_RESPONSE_SECONDS["EV"]))

def _fcr_n_resource_delay_seconds(resource: str, fleet_meta: Dict[str, Any]) -> float:
    resource_key = str(resource).upper()
    lower = resource_key.lower()
    return float(
        fleet_meta.get(
            f"{lower}_fcr_delay_s",
            fleet_meta.get(
                f"{lower}_communication_delay_s",
                fleet_meta.get("fcr_communication_delay_s", FCR_N_DEFAULT_DELAY_S.get(resource_key, 0.0)),
            ),
        )
    )

def _fcr_n_hvac_share_cap(fleet_meta: Dict[str, Any]) -> float:
    return float(np.clip(fleet_meta.get("fcr_hvac_share_cap", FCR_N_HVAC_SHARE_CAP), 0.0, 1.0))

def _fcr_n_hvac_share_limit(fast_fcr_kw: float, fleet_meta: Dict[str, Any]) -> float:
    share_cap = _fcr_n_hvac_share_cap(fleet_meta)
    if share_cap >= 0.999:
        return float("inf")
    if share_cap <= 0.0:
        return 0.0
    return float(max(fast_fcr_kw, 0.0) * share_cap / max(1.0 - share_cap, 1e-9))

def _fcr_n_bid_dynamic_metrics(values: Dict[str, float], fleet_meta: Dict[str, Any]) -> Dict[str, float | bool]:
    bids = {
        "BESS": max(float(values.get("bess_fcr_kw", 0.0)), 0.0),
        "EV": max(float(values.get("ev_fcr_kw", 0.0)), 0.0),
        "HVAC": max(float(values.get("hvac_fcr_kw", 0.0)), 0.0),
    }
    total_kw = sum(bids.values())
    if total_kw <= 1e-9:
        return {
            "fcr_response_60_fraction": 1.0,
            "fcr_response_180_fraction": 1.0,
            "fcr_energy_60_seconds": 0.0,
            "fcr_dynamic_response_ok": True,
        }

    rules = FINGRID_RULES["FCR-N"]
    response_60_kw = 0.0
    response_180_kw = 0.0
    energy_60_kws = 0.0
    for resource, bid_kw in bids.items():
        response_s = _fcr_n_resource_response_seconds(resource, fleet_meta)
        delay_s = _fcr_n_resource_delay_seconds(resource, fleet_meta)
        response_60_kw += bid_kw * _first_order_step_fraction(response_s, rules["response_63_seconds"], delay_s)
        response_180_kw += bid_kw * _first_order_step_fraction(response_s, rules["response_95_seconds"], delay_s)
        energy_60_kws += bid_kw * _first_order_step_energy_seconds(response_s, rules["response_63_seconds"], delay_s)

    response_60_fraction = response_60_kw / total_kw
    response_180_fraction = response_180_kw / total_kw
    energy_60_seconds = energy_60_kws / total_kw
    dynamic_ok = (
        response_60_fraction + 1e-9 >= float(rules["activation_60_fraction"])
        and response_180_fraction + 1e-9 >= float(rules["activation_180_fraction"])
        and energy_60_seconds + 1e-9 >= float(rules["energy_60_seconds"])
    )
    return {
        "fcr_response_60_fraction": float(response_60_fraction),
        "fcr_response_180_fraction": float(response_180_fraction),
        "fcr_energy_60_seconds": float(energy_60_seconds),
        "fcr_dynamic_response_ok": bool(dynamic_ok),
    }

def fcr_n_local_control_capable(resource: str) -> bool:
    """Return whether the simulated resource has a local frequency-control path."""

    return str(resource).upper() in FCR_N_LOCAL_CONTROL_TYPES

def _fcr_bid_total(values: Dict[str, float]) -> float:
    return float(values.get("bess_fcr_kw", 0.0) + values.get("ev_fcr_kw", 0.0) + values.get("hvac_fcr_kw", 0.0))

def risk_policy_defaults(policy: str | None) -> Dict[str, float]:
    policy_key = str(policy or "investor_balanced").strip().lower().replace(" ", "_").replace("-", "_")
    return dict(RISK_POLICY_PRESETS.get(policy_key, RISK_POLICY_PRESETS["investor_balanced"]))

def apply_risk_policy_defaults(penalty_weights: Dict[str, float]) -> Dict[str, float]:
    policy_defaults = risk_policy_defaults(str(penalty_weights.get("risk_policy", "investor_balanced")))
    merged = dict(policy_defaults)
    merged.update(penalty_weights)
    merged["risk_policy"] = str(penalty_weights.get("risk_policy", "investor_balanced"))
    return merged

def _bid_segments(total_kw: float, granularity_kw: float) -> int:
    if abs(float(total_kw)) <= 1e-6:
        return 0
    return int(round(float(total_kw) / max(float(granularity_kw), 1e-9)))

def _granularity_ok(total_kw: float, granularity_kw: float) -> bool:
    if abs(float(total_kw)) <= 1e-6:
        return True
    scaled = float(total_kw) / max(float(granularity_kw), 1e-9)
    return bool(abs(scaled - round(scaled)) <= 1e-5)

def _series_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if column in frame:
        return frame[column].astype(float)
    return pd.Series([float(default)] * len(frame), index=frame.index, dtype=float)

def _tracking_step_seconds(frame: pd.DataFrame, fallback_seconds: float = 4.0) -> float:
    if frame is None or len(frame.index) < 2:
        return float(fallback_seconds)
    index = pd.to_datetime(frame.index)
    diffs = index.to_series().diff().dt.total_seconds().dropna()
    if diffs.empty:
        return float(fallback_seconds)
    return float(max(diffs.median(), 1.0))

def _active_episodes(mask: pd.Series) -> List[tuple[int, int]]:
    if mask.empty:
        return []
    positions = np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool))
    if len(positions) == 0:
        return []
    episodes: List[tuple[int, int]] = []
    start = int(positions[0])
    previous = int(positions[0])
    for raw_pos in positions[1:]:
        pos = int(raw_pos)
        if pos != previous + 1:
            episodes.append((start, previous))
            start = pos
        previous = pos
    episodes.append((start, previous))
    return episodes

def _enrich_tracking_for_market_audit(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Add product-specific audit columns used by compliance views and plots."""

    if tracking_df is None or tracking_df.empty:
        return pd.DataFrame()
    frame = tracking_df.copy()
    signed_request = (
        _series_or_default(frame, "requested_signed_kw")
        if "requested_signed_kw" in frame
        else _series_or_default(frame, "requested_up_kw") - _series_or_default(frame, "requested_down_kw")
    )
    signed_delivery = (
        _series_or_default(frame, "delivered_signed_kw")
        if "delivered_signed_kw" in frame
        else _series_or_default(frame, "delivered_up_kw") - _series_or_default(frame, "delivered_down_kw")
    )
    fcr_request = _series_or_default(frame, "fcr_local_request_kw")
    afrr_request = _series_or_default(frame, "afrr_signal_request_kw")
    fcr_delivered = _series_or_default(frame, "fcr_delivered_kw")
    afrr_delivered = _series_or_default(frame, "afrr_delivered_kw")
    if afrr_delivered.abs().sum() <= 1e-9 and (signed_delivery - fcr_delivered).abs().sum() > 1e-9:
        afrr_delivered = signed_delivery - fcr_delivered

    frame["audit_signed_request_kw"] = signed_request
    frame["audit_signed_delivered_kw"] = signed_delivery
    frame["afrr_audit_request_kw"] = afrr_request
    frame["afrr_audit_delivered_kw"] = afrr_delivered
    frame["fcr_audit_request_kw"] = fcr_request
    frame["fcr_audit_delivered_kw"] = fcr_delivered

    afrr_active = afrr_request.abs() > 1.0
    fcr_active = fcr_request.abs() > 1.0
    frame["afrr_accuracy_ratio"] = np.where(afrr_active, afrr_delivered.abs() / afrr_request.abs().replace(0.0, np.nan), np.nan)
    frame["afrr_accuracy_low"] = float(FINGRID_RULES["aFRR"]["accuracy_low"])
    frame["afrr_accuracy_high"] = float(FINGRID_RULES["aFRR"]["accuracy_high"])
    frame["afrr_accuracy_ok_tick"] = (
        frame["afrr_accuracy_ratio"].between(frame["afrr_accuracy_low"], frame["afrr_accuracy_high"])
        & (np.sign(afrr_delivered.fillna(0.0)) == np.sign(afrr_request.fillna(0.0)))
    )
    frame.loc[~afrr_active, "afrr_accuracy_ok_tick"] = True
    frame["fcr_response_ratio"] = np.where(fcr_active, fcr_delivered.abs() / fcr_request.abs().replace(0.0, np.nan), np.nan)
    frame["afrr_reported_power_kw"] = afrr_delivered
    frame["afrr_fingrid_calculated_power_kw"] = signed_delivery - fcr_delivered
    frame["afrr_reporting_error_kw"] = frame["afrr_reported_power_kw"] - frame["afrr_fingrid_calculated_power_kw"]
    return frame

def _audit_afrr_tracking(tracking_df: pd.DataFrame, dt_seconds: float, market_mode: str) -> Dict[str, object]:
    rules = FINGRID_RULES["aFRR"]
    if market_mode not in {"aFRR", "Combined"}:
        return {
            "afrr_tracking_audit_required": False,
            "afrr_tracking_audit_evaluated": True,
            "afrr_start_within_30s_ok": True,
            "afrr_full_activation_5min_ok": True,
            "afrr_power_accuracy_ok": True,
            "afrr_response_ok": True,
            "accuracy_ok": True,
            "afrr_activation_episodes": 0,
            "afrr_full_activation_evaluated_episodes": 0,
            "afrr_accuracy_evaluated_ticks": 0,
            "afrr_min_accuracy_ratio": np.nan,
            "afrr_max_accuracy_ratio": np.nan,
            "afrr_max_start_delay_s": 0.0,
        }
    if tracking_df is None or tracking_df.empty or "afrr_audit_request_kw" not in tracking_df:
        return {
            "afrr_tracking_audit_required": True,
            "afrr_tracking_audit_evaluated": False,
            "afrr_start_within_30s_ok": False,
            "afrr_full_activation_5min_ok": False,
            "afrr_power_accuracy_ok": False,
            "afrr_response_ok": False,
            "accuracy_ok": False,
            "afrr_activation_episodes": 0,
            "afrr_full_activation_evaluated_episodes": 0,
            "afrr_accuracy_evaluated_ticks": 0,
            "afrr_min_accuracy_ratio": np.nan,
            "afrr_max_accuracy_ratio": np.nan,
            "afrr_max_start_delay_s": np.nan,
        }

    request = tracking_df["afrr_audit_request_kw"].astype(float)
    delivered = tracking_df["afrr_audit_delivered_kw"].astype(float)
    threshold_kw = max(5.0, 0.01 * float(request.abs().max()))
    active = request.abs() > threshold_kw
    episodes = _active_episodes(active)
    if not episodes:
        return {
            "afrr_tracking_audit_required": True,
            "afrr_tracking_audit_evaluated": True,
            "afrr_start_within_30s_ok": True,
            "afrr_full_activation_5min_ok": True,
            "afrr_power_accuracy_ok": True,
            "afrr_response_ok": True,
            "accuracy_ok": True,
            "afrr_activation_episodes": 0,
            "afrr_full_activation_evaluated_episodes": 0,
            "afrr_accuracy_evaluated_ticks": 0,
            "afrr_min_accuracy_ratio": np.nan,
            "afrr_max_accuracy_ratio": np.nan,
            "afrr_max_start_delay_s": 0.0,
        }

    start_limit_s = float(rules["start_seconds"])
    full_limit_s = float(rules["full_activation_seconds"])
    low = float(rules["accuracy_low"])
    high = float(rules["accuracy_high"])
    ratios: List[float] = []
    start_ok = True
    full_ok = True
    ramp_accuracy_ok = True
    max_start_delay = 0.0
    full_evaluated = 0
    accuracy_ticks = 0
    req_values = request.to_numpy(dtype=float)
    del_values = delivered.to_numpy(dtype=float)

    for start, end in episodes:
        start_delay = np.nan
        episode_full_evaluated = False
        for pos in range(start, end + 1):
            if np.sign(del_values[pos]) == np.sign(req_values[pos]) and abs(del_values[pos]) > max(1.0, 0.01 * abs(req_values[pos])):
                start_delay = (pos - start) * dt_seconds
                break
        if np.isnan(start_delay):
            start_ok = False
        else:
            max_start_delay = max(max_start_delay, float(start_delay))
            if start_delay - 1e-9 > start_limit_s:
                start_ok = False

        for pos in range(start, end + 1):
            req_abs = abs(req_values[pos])
            if req_abs <= threshold_kw:
                continue
            elapsed = (pos - start) * dt_seconds
            ratio = abs(del_values[pos]) / max(req_abs, 1e-9)
            ratios.append(float(ratio))
            sign_ok = np.sign(del_values[pos]) == np.sign(req_values[pos])
            if elapsed < start_limit_s:
                min_allowed = 0.0
            elif elapsed < full_limit_s:
                min_allowed = low * (elapsed - start_limit_s) / max(full_limit_s - start_limit_s, 1e-9)
            else:
                min_allowed = low
                episode_full_evaluated = True
            if elapsed >= start_limit_s:
                accuracy_ticks += 1
                if ratio + 1e-9 < min_allowed or ratio - 1e-9 > high or not sign_ok:
                    ramp_accuracy_ok = False
            if elapsed >= full_limit_s and (ratio + 1e-9 < low or ratio - 1e-9 > high or not sign_ok):
                full_ok = False
        if episode_full_evaluated:
            full_evaluated += 1

    if full_evaluated == 0:
        full_ok = False
    min_ratio = float(min(ratios)) if ratios else np.nan
    max_ratio = float(max(ratios)) if ratios else np.nan
    response_ok = bool(start_ok and full_ok)
    return {
        "afrr_tracking_audit_required": True,
        "afrr_tracking_audit_evaluated": True,
        "afrr_start_within_30s_ok": bool(start_ok),
        "afrr_full_activation_5min_ok": bool(full_ok),
        "afrr_power_accuracy_ok": bool(ramp_accuracy_ok and full_ok),
        "afrr_response_ok": response_ok,
        "accuracy_ok": bool(ramp_accuracy_ok and full_ok),
        "afrr_activation_episodes": int(len(episodes)),
        "afrr_full_activation_evaluated_episodes": int(full_evaluated),
        "afrr_accuracy_evaluated_ticks": int(accuracy_ticks),
        "afrr_min_accuracy_ratio": min_ratio,
        "afrr_max_accuracy_ratio": max_ratio,
        "afrr_max_start_delay_s": float(max_start_delay),
    }

def _afrr_energy_reporting_audit(tracking_df: pd.DataFrame, dt_seconds: float, market_mode: str) -> tuple[pd.DataFrame, Dict[str, object]]:
    if market_mode not in {"aFRR", "Combined"}:
        return pd.DataFrame(), {
            "afrr_energy_reporting_required": False,
            "afrr_energy_reporting_evaluated": True,
            "afrr_energy_reporting_ok": True,
            "afrr_energy_reporting_max_diff_pct": 0.0,
            "afrr_energy_reporting_periods": 0,
        }
    if tracking_df is None or tracking_df.empty or "afrr_reported_power_kw" not in tracking_df:
        return pd.DataFrame(), {
            "afrr_energy_reporting_required": True,
            "afrr_energy_reporting_evaluated": False,
            "afrr_energy_reporting_ok": False,
            "afrr_energy_reporting_max_diff_pct": np.nan,
            "afrr_energy_reporting_periods": 0,
        }

    dt_h = float(dt_seconds) / 3600.0
    rows = []
    frame = tracking_df.copy()
    frame["_isp_start"] = pd.to_datetime(frame.index).floor(f"{int(FINGRID_RULES['aFRR']['market_period_minutes'])}min")
    for isp_start, group in frame.groupby("_isp_start"):
        reported_mwh = float(group["afrr_reported_power_kw"].sum() * dt_h / 1000.0)
        calculated_mwh = float(group["afrr_fingrid_calculated_power_kw"].sum() * dt_h / 1000.0)
        denominator = max(abs(reported_mwh), 1e-6)
        diff_pct = abs(reported_mwh - calculated_mwh) / denominator
        active = max(abs(reported_mwh), abs(calculated_mwh)) > 1e-6
        rows.append(
            {
                "isp_start": pd.Timestamp(isp_start),
                "isp_end": pd.Timestamp(isp_start) + pd.Timedelta(minutes=int(FINGRID_RULES["aFRR"]["market_period_minutes"])),
                "afrr_reported_energy_mwh": reported_mwh,
                "afrr_fingrid_calculated_energy_mwh": calculated_mwh,
                "afrr_energy_reporting_diff_pct": float(diff_pct) if active else 0.0,
                "afrr_energy_reporting_ok": bool((diff_pct <= 0.10) if active else True),
                "active": bool(active),
            }
        )
    audit_df = pd.DataFrame(rows)
    active_df = audit_df[audit_df["active"]] if not audit_df.empty else audit_df
    if active_df.empty:
        return audit_df, {
            "afrr_energy_reporting_required": True,
            "afrr_energy_reporting_evaluated": True,
            "afrr_energy_reporting_ok": True,
            "afrr_energy_reporting_max_diff_pct": 0.0,
            "afrr_energy_reporting_periods": 0,
        }
    return audit_df, {
        "afrr_energy_reporting_required": True,
        "afrr_energy_reporting_evaluated": True,
        "afrr_energy_reporting_ok": bool(active_df["afrr_energy_reporting_ok"].all()),
        "afrr_energy_reporting_max_diff_pct": float(active_df["afrr_energy_reporting_diff_pct"].max()),
        "afrr_energy_reporting_periods": int(len(active_df)),
    }

def _audit_fcr_tracking(tracking_df: pd.DataFrame, dt_seconds: float, market_mode: str) -> Dict[str, object]:
    rules = FINGRID_RULES["FCR-N"]
    if market_mode not in {"FCR-N", "Combined"}:
        return {
            "fcr_tracking_response_required": False,
            "fcr_tracking_response_evaluated": True,
            "fcr_tracking_response_ok": True,
            "fcr_tracking_60_fraction": np.nan,
            "fcr_tracking_180_fraction": np.nan,
            "fcr_tracking_evaluated_episodes": 0,
        }
    if tracking_df is None or tracking_df.empty or "fcr_audit_request_kw" not in tracking_df:
        return {
            "fcr_tracking_response_required": True,
            "fcr_tracking_response_evaluated": False,
            "fcr_tracking_response_ok": False,
            "fcr_tracking_60_fraction": np.nan,
            "fcr_tracking_180_fraction": np.nan,
            "fcr_tracking_evaluated_episodes": 0,
        }
    request = tracking_df["fcr_audit_request_kw"].astype(float)
    delivered = tracking_df["fcr_audit_delivered_kw"].astype(float)
    threshold_kw = max(5.0, 0.01 * float(request.abs().max()))
    episodes = _active_episodes(request.abs() > threshold_kw)
    fractions_60: List[float] = []
    fractions_180: List[float] = []
    req_values = request.to_numpy(dtype=float)
    del_values = delivered.to_numpy(dtype=float)
    for start, end in episodes:
        pos60 = start + int(math.ceil(float(rules["response_63_seconds"]) / max(dt_seconds, 1e-9)))
        pos180 = start + int(math.ceil(float(rules["response_95_seconds"]) / max(dt_seconds, 1e-9)))
        if pos60 <= end and abs(req_values[pos60]) > threshold_kw:
            sign_ok = np.sign(del_values[pos60]) == np.sign(req_values[pos60])
            fractions_60.append(float(abs(del_values[pos60]) / max(abs(req_values[pos60]), 1e-9)) if sign_ok else 0.0)
        if pos180 <= end and abs(req_values[pos180]) > threshold_kw:
            sign_ok = np.sign(del_values[pos180]) == np.sign(req_values[pos180])
            fractions_180.append(float(abs(del_values[pos180]) / max(abs(req_values[pos180]), 1e-9)) if sign_ok else 0.0)
    evaluated = bool(fractions_60 and fractions_180)
    fraction_60 = float(min(fractions_60)) if fractions_60 else np.nan
    fraction_180 = float(min(fractions_180)) if fractions_180 else np.nan
    ok = bool(
        evaluated
        and fraction_60 + 1e-9 >= float(rules["activation_60_fraction"])
        and fraction_180 + 1e-9 >= float(rules["activation_180_fraction"])
    )
    return {
        "fcr_tracking_response_required": True,
        "fcr_tracking_response_evaluated": evaluated,
        "fcr_tracking_response_ok": ok if evaluated else False,
        "fcr_tracking_60_fraction": fraction_60,
        "fcr_tracking_180_fraction": fraction_180,
        "fcr_tracking_evaluated_episodes": int(min(len(fractions_60), len(fractions_180))),
    }

def _fcr_stability_screen(result_df: pd.DataFrame, fleet_meta: Dict[str, Any], market_mode: str) -> Dict[str, object]:
    if market_mode not in {"FCR-N", "Combined"} or result_df is None or result_df.empty:
        return {
            "fcr_stability_margin_ok": True,
            "fcr_stability_screen_evaluated": True,
            "fcr_stability_margin_pct": np.nan,
            "fcr_stability_exemption_candidate": False,
            "fcr_prequalification_evidence_ok": True,
            "fcr_prequalification_evidence_level": "not_required_for_selected_market",
            "prequalification_prep_notice": PREQUALIFICATION_PREP_NOTICE,
        }
    margins = []
    for _, row in result_df.iterrows():
        bids = {
            "BESS": float(row.get("bess_fcr_kw", 0.0)),
            "EV": float(row.get("ev_fcr_kw", 0.0)),
            "HVAC": float(row.get("hvac_fcr_kw", 0.0)),
        }
        total = sum(max(value, 0.0) for value in bids.values())
        if total <= 1e-9:
            continue
        response_60 = float(row.get("fcr_response_60_fraction", 1.0))
        response_180 = float(row.get("fcr_response_180_fraction", 1.0))
        hvac_share = max(bids["HVAC"], 0.0) / max(total, 1e-9)
        weighted_delay = sum(
            max(bid, 0.0) * _fcr_n_resource_delay_seconds(resource, fleet_meta)
            for resource, bid in bids.items()
        ) / max(total, 1e-9)
        dynamic_score = 0.5 * min(response_60 / max(float(FINGRID_RULES["FCR-N"]["activation_60_fraction"]), 1e-9), 1.0)
        dynamic_score += 0.5 * min(response_180 / max(float(FINGRID_RULES["FCR-N"]["activation_180_fraction"]), 1e-9), 1.0)
        margin = 0.78 + 0.28 * dynamic_score - 0.10 * hvac_share - 0.004 * weighted_delay
        margins.append(float(np.clip(margin, 0.0, 1.20)))
    if not margins:
        return {
            "fcr_stability_margin_ok": True,
            "fcr_stability_screen_evaluated": True,
            "fcr_stability_margin_pct": np.nan,
            "fcr_stability_exemption_candidate": False,
            "fcr_prequalification_evidence_ok": True,
            "fcr_prequalification_evidence_level": "no_fcr_bid",
            "prequalification_prep_notice": PREQUALIFICATION_PREP_NOTICE,
        }
    margin = min(margins)
    margin_ok = margin + 1e-9 >= 0.95
    exemption_candidate = 0.75 <= margin < 0.95
    return {
        "fcr_stability_margin_ok": bool(margin_ok),
        "fcr_stability_screen_evaluated": True,
        "fcr_stability_margin_pct": float(100.0 * margin),
        "fcr_stability_exemption_candidate": bool(exemption_candidate),
        "fcr_prequalification_evidence_ok": bool(margin_ok),
        "fcr_prequalification_evidence_level": "simulation_screening_only",
        "prequalification_prep_notice": PREQUALIFICATION_PREP_NOTICE,
    }

def _market_technical_audit(
    result_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    fleet_meta: Dict[str, Any],
    market_mode: str,
    inner_dt_seconds: int,
) -> tuple[Dict[str, object], Dict[str, object], pd.DataFrame]:
    dt_seconds = _tracking_step_seconds(tracking_df, float(inner_dt_seconds))
    afrr_tracking = _audit_afrr_tracking(tracking_df, dt_seconds, market_mode)
    afrr_energy_df, afrr_energy = _afrr_energy_reporting_audit(tracking_df, dt_seconds, market_mode)
    fcr_tracking = _audit_fcr_tracking(tracking_df, dt_seconds, market_mode)
    fcr_stability = _fcr_stability_screen(result_df, fleet_meta, market_mode)
    compliance_updates: Dict[str, object] = {}
    compliance_updates.update(afrr_tracking)
    compliance_updates.update(afrr_energy)
    compliance_updates.update(fcr_tracking)
    compliance_updates.update(fcr_stability)
    if market_mode in {"FCR-N", "Combined"}:
        tracking_ok = bool(fcr_tracking["fcr_tracking_response_ok"]) if fcr_tracking["fcr_tracking_response_evaluated"] else True
        compliance_updates["fcr_actual_dynamic_response_ok"] = bool(tracking_ok)
    else:
        compliance_updates["fcr_actual_dynamic_response_ok"] = True
    if market_mode in {"aFRR", "Combined"}:
        compliance_updates["afrr_settlement_reconciliation_ok"] = bool(afrr_energy["afrr_energy_reporting_ok"])
    else:
        compliance_updates["afrr_settlement_reconciliation_ok"] = True
    summary_updates = {
        key: value
        for key, value in compliance_updates.items()
        if key
        in {
            "afrr_activation_episodes",
            "afrr_full_activation_evaluated_episodes",
            "afrr_accuracy_evaluated_ticks",
            "afrr_min_accuracy_ratio",
            "afrr_max_accuracy_ratio",
            "afrr_max_start_delay_s",
            "afrr_energy_reporting_max_diff_pct",
            "afrr_energy_reporting_periods",
            "fcr_tracking_60_fraction",
            "fcr_tracking_180_fraction",
            "fcr_tracking_evaluated_episodes",
            "fcr_stability_margin_pct",
            "fcr_prequalification_evidence_level",
            "prequalification_prep_notice",
        }
    }
    return compliance_updates, summary_updates, afrr_energy_df

def _bid_block_bounds(timestamp: pd.Timestamp, market_mode: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    ts = pd.Timestamp(timestamp)
    minutes = (
        FINGRID_RULES["FCR-N"]["market_period_minutes"]
        if market_mode in {"FCR-N", "Combined"}
        else FINGRID_RULES["aFRR"]["market_period_minutes"]
    )
    block_start = ts.floor(f"{int(minutes)}min")
    return block_start, block_start + pd.Timedelta(minutes=int(minutes))

def _market_bid_metadata(values: Dict[str, float], market_mode: str | None = None, timestamp: pd.Timestamp | None = None) -> Dict[str, object]:
    payload = dict(values)
    fcr_kw = _fcr_bid_total(payload)
    afrr_up_kw = float(payload.get("bess_afrr_up_kw", 0.0) + payload.get("ev_afrr_up_kw", 0.0) + payload.get("hvac_afrr_up_kw", 0.0))
    afrr_down_kw = float(
        payload.get("bess_afrr_down_kw", 0.0)
        + payload.get("ev_afrr_down_kw", 0.0)
        + payload.get("hvac_afrr_down_kw", 0.0)
        + payload.get("pv_afrr_down_kw", 0.0)
    )
    fcr_granularity = float(FINGRID_RULES["FCR-N"]["bid_granularity_kw"])
    afrr_granularity = float(FINGRID_RULES["aFRR"]["bid_granularity_kw"])
    fcr_ok = _granularity_ok(fcr_kw, fcr_granularity)
    afrr_up_ok = _granularity_ok(afrr_up_kw, afrr_granularity)
    afrr_down_ok = _granularity_ok(afrr_down_kw, afrr_granularity)
    payload.update(
        {
            "fcr_bid_kw": float(fcr_kw),
            "afrr_up_bid_kw": float(afrr_up_kw),
            "afrr_down_bid_kw": float(afrr_down_kw),
            "fcr_bid_mw": float(fcr_kw / 1000.0),
            "afrr_up_bid_mw": float(afrr_up_kw / 1000.0),
            "afrr_down_bid_mw": float(afrr_down_kw / 1000.0),
            "fcr_bid_segments": _bid_segments(fcr_kw, fcr_granularity),
            "afrr_up_segments": _bid_segments(afrr_up_kw, afrr_granularity),
            "afrr_down_segments": _bid_segments(afrr_down_kw, afrr_granularity),
            "fcr_bid_granularity_ok": bool(fcr_ok),
            "afrr_up_bid_granularity_ok": bool(afrr_up_ok),
            "afrr_down_bid_granularity_ok": bool(afrr_down_ok),
            "granularity_ok": bool(fcr_ok and afrr_up_ok and afrr_down_ok),
            "fcr_control_source": FINGRID_RULES["FCR-N"]["control_source"],
            "fcr_symmetric": True,
            "simulation_only_notice": SIMULATION_ONLY_NOTICE,
        }
    )
    if timestamp is not None and market_mode is not None:
        start, end = _bid_block_bounds(pd.Timestamp(timestamp), str(market_mode))
        payload["bid_block_start"] = start
        payload["bid_block_end"] = end
    return payload

def _apply_fcr_n_bid_rules(values: Dict[str, float]) -> Dict[str, float]:
    """Round FCR-N capacity to market granularity while preserving symmetry."""

    payload = dict(values)
    fcr_total = _fcr_bid_total(payload)
    min_kw = float(FINGRID_RULES["FCR-N"]["min_bid_kw"])
    granularity_kw = float(FINGRID_RULES["FCR-N"]["bid_granularity_kw"])
    max_timeseries_kw = float(FINGRID_RULES["FCR-N"]["max_bid_kw_per_timeseries"])
    if fcr_total <= 0.0:
        quantized_kw = 0.0
    else:
        quantized_kw = np.floor(fcr_total / max(granularity_kw, 1e-9)) * granularity_kw
        if quantized_kw < min_kw:
            quantized_kw = 0.0
    ratio = quantized_kw / fcr_total if fcr_total > 1e-9 else 0.0
    for key in ("bess_fcr_kw", "ev_fcr_kw", "hvac_fcr_kw"):
        payload[key] = float(payload.get(key, 0.0) * ratio)
    payload["fcr_bid_kw"] = float(quantized_kw)
    payload["fcr_bid_granularity_kw"] = granularity_kw
    payload["fcr_bid_timeseries_count"] = int(np.ceil(quantized_kw / max(max_timeseries_kw, 1e-9))) if quantized_kw > 0.0 else 0
    payload["fcr_control_source"] = FINGRID_RULES["FCR-N"]["control_source"]
    payload["fcr_symmetric"] = True
    return _market_bid_metadata(payload)

def serialize_forecast_spec(spec: ForecastSpec) -> bytes:
    payload = {
        "target": spec.target,
        "predictors": spec.predictors,
        "lags": spec.lags,
        "horizon_steps": spec.horizon_steps,
        "metrics": spec.metrics,
        "model": spec.model,
    }
    return pickle.dumps(payload)

def fmt_kw(value: float) -> str:
    return f"{value / 1000:.2f} MW" if abs(value) >= 1000 else f"{value:.0f} kW"

def fmt_money(value: float) -> str:
    return f"€{value:,.0f}"

def ar1_series(length: int, phi: float, sigma: float, rng: np.random.Generator, mean: float = 0.0) -> np.ndarray:
    out = np.zeros(length)
    out[0] = mean + rng.normal(0, sigma)
    for idx in range(1, length):
        out[idx] = mean + phi * (out[idx - 1] - mean) + rng.normal(0, sigma)
    return out

def hour_fraction(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour + index.minute / 60.0

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def time_feature_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    h = hour_fraction(index)
    dow = index.dayofweek.values
    doy = index.dayofyear.values
    data = {
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "doy_sin": np.sin(2 * np.pi * doy / 365),
        "doy_cos": np.cos(2 * np.pi * doy / 365),
        "is_weekend": (dow >= 5).astype(int),
    }
    return pd.DataFrame(data, index=index)

def generate_weather(
    index: pd.DatetimeIndex,
    seed: int,
    cloudiness: float,
    climate_shift_c: float,
    solar_scale: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = hour_fraction(index)
    doy = index.dayofyear.values
    season = np.sin(2 * np.pi * (doy - 80) / 365.0)
    temp_noise = ar1_series(len(index), phi=0.90, sigma=0.55, rng=rng)
    temp_out = 2.5 + 10.5 * season + 2.8 * np.sin(2 * np.pi * (hours - 14.0) / 24.0) + temp_noise + climate_shift_c

    seasonal_daylight = 8.5 + 7.5 * np.clip(np.sin(2 * np.pi * (doy - 80) / 365.0), -0.8, 0.95)
    sunrise = 12.0 - seasonal_daylight / 2.0
    sunset = 12.0 + seasonal_daylight / 2.0
    daylight_progress = (hours - sunrise) / np.maximum(sunset - sunrise, 0.1)
    solar_shape = np.sin(np.pi * np.clip(daylight_progress, 0, 1))
    cloud_factor = np.clip(
        1.0 - cloudiness + 0.45 * np.tanh(ar1_series(len(index), phi=0.96, sigma=0.45, rng=rng)),
        0.10,
        1.20,
    )
    seasonal_irr_scale = np.clip(0.25 + 0.90 * np.maximum(season, -0.15), 0.08, 1.0)
    irradiance = np.clip(920 * solar_shape**1.6 * cloud_factor * seasonal_irr_scale * solar_scale, 0, None)
    return pd.DataFrame({"temp_out_c": temp_out, "irradiance_wm2": irradiance}, index=index)

def generate_market_prices(index: pd.DatetimeIndex, seed: int, scarcity: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    hourly = pd.date_range(index.min().floor("h"), index.max().ceil("h"), freq="h")
    h = hourly.hour.values
    weekday = (hourly.dayofweek < 5).astype(float)
    scarcity_spikes = np.maximum(ar1_series(len(hourly), 0.78, 1.6, rng, 0), 0)

    fcr_price = np.clip(
        10.5
        + 2.8 * np.sin(2 * np.pi * (h - 6) / 24.0)
        + 1.8 * weekday
        + scarcity * scarcity_spikes
        + rng.normal(0, 1.1, len(hourly)),
        4.0,
        35.0,
    )
    afrr_up_cap = np.clip(
        6.0 + 2.2 * np.sin(2 * np.pi * (h - 7) / 24.0) + 0.9 * scarcity_spikes + rng.normal(0, 0.8, len(hourly)),
        2.0,
        22.0,
    )
    afrr_down_cap = np.clip(
        4.2 + 1.8 * np.sin(2 * np.pi * (h - 12) / 24.0) + 0.6 * scarcity_spikes + rng.normal(0, 0.7, len(hourly)),
        1.0,
        18.0,
    )
    afrr_up_energy = np.clip(
        95 + 34 * np.sin(2 * np.pi * (h - 8) / 24.0) + 18 * scarcity_spikes + rng.normal(0, 8.0, len(hourly)),
        35,
        280,
    )
    afrr_down_energy = np.clip(
        55 + 22 * np.sin(2 * np.pi * (h - 12) / 24.0) + 10 * scarcity_spikes + rng.normal(0, 6.0, len(hourly)),
        15,
        180,
    )

    price_df = pd.DataFrame(
        {
            "fcrn_capacity_eur_per_mw_h": fcr_price,
            "afrr_up_capacity_eur_per_mw_h": afrr_up_cap,
            "afrr_down_capacity_eur_per_mw_h": afrr_down_cap,
            "afrr_up_energy_eur_per_mwh": afrr_up_energy,
            "afrr_down_energy_eur_per_mwh": afrr_down_energy,
        },
        index=hourly,
    )
    return price_df.reindex(index, method="ffill")

def generate_activation_signals(index: pd.DatetimeIndex, seed: int, preview_days: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 2024)
    act = pd.DataFrame(index=index)
    signed = np.clip(ar1_series(len(index), 0.82, 0.11, rng), -1.0, 1.0)
    act["fcr_signed_act"] = signed
    act["fcr_up_act_frac"] = np.clip(signed, 0, 1)
    act["fcr_down_act_frac"] = np.clip(-signed, 0, 1)

    afrr_signal = np.clip(ar1_series(len(index), 0.88, 0.18, rng), -1.0, 1.0)
    act["afrr_signal_norm"] = afrr_signal
    act["afrr_up_act_frac"] = np.clip(afrr_signal, 0, 1)
    act["afrr_down_act_frac"] = np.clip(-afrr_signal, 0, 1)
    act["frequency_hz"] = 50.0 - 0.1 * signed

    preview_days = int(max(1, min(preview_days, 7)))
    preview_start = index.min().floor("d")
    preview_index = pd.date_range(preview_start, periods=preview_days * 24 * 3600 // 4, freq="4s")
    dev = np.zeros(len(preview_index))
    afrr_4s = np.zeros(len(preview_index))
    for i in range(1, len(preview_index)):
        event = rng.normal(0, 0.028) if rng.random() < 0.0015 else 0.0
        dev[i] = np.clip(0.965 * dev[i - 1] + rng.normal(0, 0.0024) + event, -0.105, 0.105)
        afrr_4s[i] = np.clip(0.95 * afrr_4s[i - 1] + rng.normal(0, 0.035) - 0.65 * dev[i], -1.0, 1.0)

    preview_df = pd.DataFrame(
        {
            "frequency_hz": 50.0 + dev,
            "fcr_signal_norm": np.clip(-dev / 0.1, -1, 1),
            "afrr_signal_norm": afrr_4s,
        },
        index=preview_index,
    )
    return act, preview_df

def simulate_hvac_baseline(
    temp_out: np.ndarray,
    n_hvac: int,
    setpoint_c: float,
    comfort_band_c: float,
    freq_minutes: int,
) -> Dict[str, np.ndarray]:
    steps = len(temp_out)
    dt_h = freq_minutes / 60.0
    hours = np.arange(steps) * dt_h
    internal_gain_kw = 0.18 + 0.12 * np.sin(2 * np.pi * (hours - 17) / 24.0) ** 2
    resistance = 2.8
    thermal_cap = 4.2
    cop = 2.8
    p_max = 4.6
    temp_in = np.zeros(steps)
    baseline_per_home = np.zeros(steps)
    temp_in[0] = setpoint_c
    for t in range(steps - 1):
        proportional = 0.38 * (setpoint_c - temp_in[t])
        target_power = ((setpoint_c - temp_out[t]) / resistance - internal_gain_kw[t]) / cop + proportional
        baseline_per_home[t] = np.clip(target_power, 0.18, p_max)
        temp_rate = ((temp_out[t] - temp_in[t]) / resistance + cop * baseline_per_home[t] + internal_gain_kw[t]) / thermal_cap
        temp_in[t + 1] = temp_in[t] + dt_h * temp_rate
    baseline_per_home[-1] = baseline_per_home[-2]
    baseline_kw = baseline_per_home * n_hvac

    temp_margin_low = np.clip(temp_in - (setpoint_c - comfort_band_c), 0, comfort_band_c + 0.1)
    temp_margin_high = np.clip((setpoint_c + comfort_band_c) - temp_in, 0, comfort_band_c + 0.1)
    hvac_up_kw = baseline_kw * np.clip(temp_margin_low / max(comfort_band_c, 0.2), 0, 1)
    hvac_down_kw = (p_max * n_hvac - baseline_kw) * np.clip(temp_margin_high / max(comfort_band_c, 0.2), 0, 1)
    return {
        "indoor_temp_c_ref": temp_in,
        "hvac_baseline_kw": baseline_kw,
        "hvac_up_kw": hvac_up_kw,
        "hvac_down_kw": hvac_down_kw,
    }

def generate_synthetic_portfolio(
    start_date: str,
    days: int,
    n_homes: int,
    ev_pen: float,
    bess_pen: float,
    pv_pen: float,
    hvac_pen: float,
    cloudiness: float,
    climate_shift_c: float,
    solar_scale: float,
    seed: int,
    setpoint_c: float,
    comfort_band_c: float,
    scarcity: float,
    freq_minutes: int = 15,
    hvac_mode: str = HVAC_MODES[0],
    hvac_response_s: float | None = None,
    market_data_mode: str | None = None,
    entsoe_api_key: str | None = None,
    fingrid_api_key: str | None = None,
    market_lookback_days: int | None = None,
    persist_market_data: bool | None = None,
) -> Dict[str, object]:
    dt_h = freq_minutes / 60.0
    steps = int(days * 24 * 60 / freq_minutes)
    index = pd.date_range(pd.to_datetime(start_date), periods=steps, freq=f"{freq_minutes}min")
    rng = np.random.default_rng(seed)

    weather = generate_weather(index, seed, cloudiness, climate_shift_c, solar_scale)
    prices = generate_market_prices(index, seed, scarcity)
    activation, preview = generate_activation_signals(index, seed, preview_days=min(days, 3))
    prices, activation, preview, market_data_status = overlay_real_market_data(
        index,
        prices,
        activation,
        preview,
        mode=market_data_mode,
        entsoe_api_key=entsoe_api_key,
        fingrid_api_key=fingrid_api_key,
        lookback_days=market_lookback_days,
        persist_to_timescale=persist_market_data,
    )
    tf = time_feature_frame(index)

    hours = hour_fraction(index)
    evening_peak = np.exp(-0.5 * ((hours - 19.0) / 2.6) ** 2)
    morning_peak = np.exp(-0.5 * ((hours - 7.6) / 1.8) ** 2)
    weekend = (index.dayofweek >= 5).astype(float)
    temp_out = weather["temp_out_c"].values
    irr = weather["irradiance_wm2"].values

    base_load_per_home = np.clip(
        0.24
        + 0.48 * morning_peak
        + 0.70 * evening_peak
        + 0.04 * weekend
        + 0.032 * np.clip(18 - temp_out, 0, None)
        + rng.normal(0, 0.03, steps),
        0.10,
        None,
    )
    base_load_kw = n_homes * base_load_per_home

    n_pv = int(round(n_homes * pv_pen))
    pv_capacity_kw = max(n_pv * 5.6, 0.0)
    inverter_eff = 0.92
    pv_available_kw = np.clip(pv_capacity_kw * inverter_eff * irr / 1000.0, 0, pv_capacity_kw)

    n_ev = int(round(n_homes * ev_pen))
    connected_frac = np.clip(
        0.10
        + 0.72 * sigmoid((18.0 - np.abs(hours - 21.0)) / 2.1)
        + 0.18 * np.exp(-0.5 * ((hours - 2.0) / 3.2) ** 2)
        - 0.22 * np.exp(-0.5 * ((hours - 10.0) / 2.8) ** 2)
        + 0.05 * weekend,
        0.05,
        0.96,
    )
    ev_connected_count = n_ev * connected_frac
    ev_charger_cap_kw = ev_connected_count * 7.0
    ev_energy_capacity_mwh = n_ev * 60.0 / 1000.0
    ev_soc_frac = np.clip(
        0.58
        + 0.14 * np.cos(2 * np.pi * (hours - 4.0) / 24.0)
        - 0.08 * np.exp(-0.5 * ((hours - 17.5) / 2.2) ** 2)
        + rng.normal(0, 0.015, steps),
        0.20,
        0.92,
    )
    ev_soc_ref_mwh = ev_energy_capacity_mwh * connected_frac * ev_soc_frac
    night_signal = np.clip(np.exp(-0.5 * ((hours - 1.2) / 3.0) ** 2) + 0.55 * np.exp(-0.5 * ((hours - 23.0) / 2.8) ** 2), 0, None)
    ev_baseline_kw = np.minimum(ev_charger_cap_kw * 0.75, ev_charger_cap_kw * night_signal * np.clip((0.82 - ev_soc_frac) / 0.45, 0, 1))
    ev_up_kw = ev_charger_cap_kw * np.clip((ev_soc_frac - 0.25) / 0.55, 0, 1)
    ev_down_kw = ev_charger_cap_kw * np.clip((0.92 - ev_soc_frac) / 0.55, 0, 1)
    ev_soc_min_mwh = ev_energy_capacity_mwh * connected_frac * (0.22 + 0.16 * np.exp(-0.5 * ((hours - 6.5) / 2.0) ** 2))
    ev_soc_max_mwh = ev_energy_capacity_mwh * connected_frac * 0.95

    n_bess = int(round(n_homes * bess_pen))
    bess_power_cap_kw = n_bess * 5.0
    bess_energy_cap_mwh = n_bess * 12.0 / 1000.0
    pv_norm = pv_available_kw / max(pv_capacity_kw, 1.0)
    bess_soc_frac = np.clip(
        0.52
        + 0.15 * np.sin(2 * np.pi * (hours - 13.0) / 24.0)
        + 0.16 * pv_norm
        + rng.normal(0, 0.03, steps),
        0.10,
        0.96,
    )
    bess_soc_ref_mwh = bess_energy_cap_mwh * bess_soc_frac
    bess_baseline_kw = bess_power_cap_kw * (
        0.35 * np.exp(-0.5 * ((hours - 19.5) / 2.6) ** 2) - 0.40 * np.exp(-0.5 * ((hours - 12.5) / 2.6) ** 2)
    )
    bess_up_kw = bess_power_cap_kw * np.clip((bess_soc_frac - 0.12) / 0.70, 0, 1)
    bess_down_kw = bess_power_cap_kw * np.clip((0.96 - bess_soc_frac) / 0.75, 0, 1)

    n_hvac = int(round(n_homes * hvac_pen))
    hvac = simulate_hvac_baseline(temp_out, n_hvac, setpoint_c, comfort_band_c, freq_minutes)
    hvac_response_s = float(hvac_response_s) if hvac_response_s is not None else default_hvac_response_seconds(hvac_mode)
    device_roster = generate_household_device_roster(
        n_homes=n_homes,
        ev_pen=ev_pen,
        bess_pen=bess_pen,
        pv_pen=pv_pen,
        hvac_pen=hvac_pen,
        seed=seed,
        hvac_response_s=hvac_response_s,
    )

    df = pd.DataFrame(index=index)
    df = pd.concat([df, weather, prices, activation, tf], axis=1)
    df["base_load_kw"] = base_load_kw
    df["pv_available_kw"] = pv_available_kw
    df["pv_down_kw"] = pv_available_kw
    df["ev_connected_frac"] = connected_frac
    df["ev_connected_count"] = ev_connected_count
    df["ev_baseline_kw"] = ev_baseline_kw
    df["ev_up_kw"] = ev_up_kw
    df["ev_down_kw"] = ev_down_kw
    df["ev_soc_ref_mwh"] = ev_soc_ref_mwh
    df["ev_soc_min_mwh"] = ev_soc_min_mwh
    df["ev_soc_max_mwh"] = ev_soc_max_mwh
    df["bess_baseline_kw"] = bess_baseline_kw
    df["bess_up_kw"] = bess_up_kw
    df["bess_down_kw"] = bess_down_kw
    df["bess_soc_ref_mwh"] = bess_soc_ref_mwh
    df["bess_energy_cap_mwh"] = bess_energy_cap_mwh
    df["hvac_baseline_kw"] = hvac["hvac_baseline_kw"]
    df["hvac_up_kw"] = hvac["hvac_up_kw"]
    df["hvac_down_kw"] = hvac["hvac_down_kw"]
    df["indoor_temp_c_ref"] = hvac["indoor_temp_c_ref"]
    hvac_fcr_dynamic_fraction = fcr_n_dynamic_deliverable_fraction(
        hvac_response_s,
        FCR_N_DEFAULT_DELAY_S["HVAC"],
    )
    df["hvac_fcr_dynamic_fraction"] = hvac_fcr_dynamic_fraction
    df["hvac_fcr_dynamic_cap_kw"] = hvac_fcr_dynamic_fraction * np.minimum(df["hvac_up_kw"], df["hvac_down_kw"])
    df["hvac_fast_kw"] = df["hvac_fcr_dynamic_cap_kw"]
    df["fast_sym_kw"] = np.minimum(df["bess_up_kw"], df["bess_down_kw"]) + np.minimum(df["ev_up_kw"], df["ev_down_kw"]) + df["hvac_fast_kw"]
    df["hybrid_fcr_kw"] = df["fast_sym_kw"]
    df["flex_up_kw"] = df["bess_up_kw"] + df["ev_up_kw"] + df["hvac_up_kw"]
    df["flex_down_kw"] = df["bess_down_kw"] + df["ev_down_kw"] + df["hvac_down_kw"] + df["pv_down_kw"]
    df["net_load_baseline_kw"] = (
        df["base_load_kw"] + df["ev_baseline_kw"] + df["hvac_baseline_kw"] - df["pv_available_kw"] - df["bess_baseline_kw"]
    )
    df["baseline_flex_margin_kw"] = df["flex_up_kw"] + df["flex_down_kw"]
    df["dt_h"] = dt_h

    summary = {
        "n_homes": n_homes,
        "n_ev": n_ev,
        "n_bess": n_bess,
        "n_pv": n_pv,
        "n_hvac": n_hvac,
        "peak_fast_mw": df["fast_sym_kw"].max() / 1000.0,
        "peak_total_up_mw": df["flex_up_kw"].max() / 1000.0,
        "peak_total_down_mw": df["flex_down_kw"].max() / 1000.0,
        "mean_net_load_mw": df["net_load_baseline_kw"].mean() / 1000.0,
        "homes_for_1mw_fast": int(np.ceil(1000 / max(df["fast_sym_kw"].mean() / max(n_homes, 1), 0.1))),
        "homes_for_1mw_total": int(np.ceil(1000 / max(df["flex_up_kw"].mean() / max(n_homes, 1), 0.1))),
        "pv_capacity_mw": pv_capacity_kw / 1000.0,
        "bess_energy_mwh": bess_energy_cap_mwh,
        "ev_energy_mwh": ev_energy_capacity_mwh,
        "hvac_mode": hvac_mode,
        "hvac_fcr_dynamic_fraction": hvac_fcr_dynamic_fraction,
        "fcr_hvac_share_cap": FCR_N_HVAC_SHARE_CAP,
        "device_count": int(len(device_roster)),
        "gateway_count": int(device_roster["gateway_id"].nunique()) if not device_roster.empty else 0,
        "freq_minutes": freq_minutes,
        "market_data_source": market_data_status["mode_used"],
        "market_data_status": market_data_status,
    }
    return {"data": df, "preview_4s": preview, "summary": summary, "device_roster": device_roster}

def build_feature_dataset(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    lags: int,
    horizon_steps: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    frame = time_feature_frame(df.index)
    for col in predictors:
        frame[col] = df[col]
        for lag in range(1, lags + 1):
            frame[f"{col}_lag{lag}"] = df[col].shift(lag)
    y = df[target].shift(-horizon_steps)
    assembled = frame.join(y.rename("target")).dropna()
    return assembled.drop(columns="target"), assembled["target"]

def train_forecaster(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    lags: int,
    horizon_steps: int,
    seed: int,
) -> Tuple[ForecastSpec, pd.DataFrame]:
    X, y = build_feature_dataset(df, target, predictors, lags, horizon_steps)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = XGBRegressor(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    pred = pd.Series(model.predict(X_test), index=y_test.index, name="prediction")
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    comparison = pd.DataFrame({"actual": y_test, "prediction": pred})
    residuals = (pd.Series(y_test, index=pred.index).astype(float) - pred.astype(float)).dropna()
    residual_quantiles = {
        f"p{int(round(q * 100)):02d}": float(np.nanquantile(residuals.to_numpy(dtype=float), q)) if not residuals.empty else 0.0
        for q in FORECAST_QUANTILES
    }
    for label, residual in residual_quantiles.items():
        comparison[f"prediction_{label}"] = pred + residual
    spec = ForecastSpec(
        target=target,
        predictors=predictors,
        lags=lags,
        horizon_steps=horizon_steps,
        model=model,
        metrics=metrics,
        residual_quantiles=residual_quantiles,
    )
    return spec, comparison

def build_single_feature_row(frame: pd.DataFrame, row_idx: int, predictors: List[str], lags: int) -> pd.DataFrame:
    idx = frame.index[row_idx]
    values = time_feature_frame(pd.DatetimeIndex([idx])).iloc[0].to_dict()
    current = frame.iloc[row_idx]
    for col in predictors:
        values[col] = current[col]
        for lag in range(1, lags + 1):
            lag_pos = max(0, row_idx - lag)
            values[f"{col}_lag{lag}"] = frame.iloc[lag_pos][col]
    return pd.DataFrame([values], index=pd.DatetimeIndex([idx]))

def train_default_mpc_models(df: pd.DataFrame, seed: int) -> Dict[str, ForecastSpec]:
    predictors = [
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
    registry: Dict[str, ForecastSpec] = {}
    for target in SUPPORTED_MPC_TARGETS:
        spec, _ = train_forecaster(df, target=target, predictors=predictors, lags=4, horizon_steps=1, seed=seed)
        registry[target] = spec
    return registry


def _forecast_quantile_label(quantile: float) -> str:
    return f"p{int(round(float(quantile) * 100)):02d}"


def _forecast_quantile_column(target: str, quantile: float) -> str:
    return f"{target}_forecast_{_forecast_quantile_label(quantile)}"


def _coerce_uncertainty_frame(uncertainty: Any, index: pd.Index) -> pd.DataFrame:
    if uncertainty is None:
        return pd.DataFrame(index=index)
    if isinstance(uncertainty, pd.Series):
        return uncertainty.to_frame()
    if isinstance(uncertainty, pd.DataFrame):
        return uncertainty.reindex(index)
    array = np.asarray(uncertainty)
    if array.ndim == 1:
        return pd.DataFrame({"p50": array}, index=index)
    return pd.DataFrame(array, index=index)


def _predict_one_step_with_uncertainty(spec: Any, row: pd.DataFrame) -> tuple[float, Dict[str, float]]:
    predict_fn = getattr(spec, "predict", None)
    uncertainty = None
    if callable(predict_fn):
        payload = predict_fn(row, horizon_steps=1)
        if isinstance(payload, tuple):
            predictions, uncertainty = payload
        else:
            predictions = payload
        if isinstance(predictions, pd.Series):
            point = float(predictions.iloc[0])
        else:
            values = np.asarray(predictions).reshape(-1)
            point = float(values[0])
    else:
        point = float(spec.model.predict(row)[0])

    quantiles: Dict[str, float] = {}
    uncertainty_frame = _coerce_uncertainty_frame(uncertainty, row.index)
    if not uncertainty_frame.empty:
        first = uncertainty_frame.iloc[0]
        for column, value in first.items():
            label = str(column)
            if label.startswith("prediction_"):
                label = label.replace("prediction_", "", 1)
            if label.startswith("p") and pd.notna(value):
                quantiles[label] = float(value)

    residual_quantiles = getattr(spec, "residual_quantiles", None) or {}
    for label, residual in dict(residual_quantiles).items():
        label = str(label)
        if label.startswith("p") and label not in quantiles:
            quantiles[label] = float(point + float(residual))

    if "p50" not in quantiles:
        quantiles["p50"] = point
    return point, quantiles


def iterative_forecast(
    df: pd.DataFrame,
    current_pos: int,
    horizon_steps: int,
    models: Dict[str, Any],
    targets: List[str],
    step_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    work = df.copy()
    future_slice = work.iloc[: current_pos + horizon_steps + 1].copy()
    result_index = work.index[current_pos + 1 : current_pos + 1 + horizon_steps]
    output = pd.DataFrame(index=result_index)

    for step_ahead in range(1, horizon_steps + 1):
        base_pos = current_pos + step_ahead - 1
        pred_pos = base_pos + 1
        for target in targets:
            spec = models[target]
            row = build_single_feature_row(future_slice, base_pos, spec.predictors, spec.lags)
            prediction, quantiles = _predict_one_step_with_uncertainty(spec, row)
            future_slice.loc[future_slice.index[pred_pos], target] = prediction
            output.loc[future_slice.index[pred_pos], target] = prediction
            for label, value in quantiles.items():
                output.loc[future_slice.index[pred_pos], f"{target}_forecast_{label}"] = value
        if step_callback:
            step_callback(step_ahead, horizon_steps)
    return output

def _predict_one_step(spec: Any, row: pd.DataFrame) -> float:
    return _predict_one_step_with_uncertainty(spec, row)[0]

def response_model_profiles(
    tau_bess_s: float,
    tau_ev_s: float,
    tau_hvac1_s: float,
    tau_hvac2_s: float,
    hvac_mode: str,
    tau_hvac_elec_s: float,
    duration_s: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    t = np.linspace(0, duration_s, 300)
    profiles: Dict[str, Dict[str, np.ndarray]] = {}

    bess_system = signal.TransferFunction([1], [tau_bess_s, 1])
    ev_system = signal.TransferFunction([1], [tau_ev_s, 1])
    if hvac_mode == HVAC_MODES[1]:
        hvac_system = signal.TransferFunction([1], [tau_hvac_elec_s, 1])
    else:
        hvac_system = signal.TransferFunction([1], [tau_hvac1_s * tau_hvac2_s, tau_hvac1_s + tau_hvac2_s, 1])
    t_bess, y_bess = signal.step(bess_system, T=t)
    t_ev, y_ev = signal.step(ev_system, T=t)
    t_hvac, y_hvac = signal.step(hvac_system, T=t)

    pv_time = t
    pv_response = np.ones_like(t)
    pv_response[t < 2.0] = t[t < 2.0] / 2.0

    profiles["BESS"] = {"t": t_bess, "y": y_bess}
    profiles["EV"] = {"t": t_ev, "y": y_ev}
    profiles["HVAC"] = {"t": t_hvac, "y": y_hvac}
    profiles["PV Curtailment"] = {"t": pv_time, "y": pv_response}
    return profiles

def bode_points(
    resource: str,
    tau_bess_s: float,
    tau_ev_s: float,
    tau_hvac1_s: float,
    tau_hvac2_s: float,
    hvac_mode: str,
    tau_hvac_elec_s: float,
) -> pd.DataFrame:
    omega = np.logspace(-4, 1, 240)
    try:
        import control as ctl
    except Exception:
        ctl = None

    if ctl is not None:
        if resource == "BESS":
            system = ctl.TransferFunction([1], [tau_bess_s, 1])
        elif resource == "EV":
            system = ctl.TransferFunction([1], [tau_ev_s, 1])
        elif hvac_mode == HVAC_MODES[1]:
            system = ctl.TransferFunction([1], [tau_hvac_elec_s, 1])
        else:
            system = ctl.TransferFunction([1], [tau_hvac1_s * tau_hvac2_s, tau_hvac1_s + tau_hvac2_s, 1])
        mag, phase, freq = ctl.bode(system, omega, plot=False)
        mag_db = 20 * np.log10(np.maximum(mag, 1e-6))
        phase_deg = np.degrees(phase)
        return pd.DataFrame({"omega": freq, "mag_db": mag_db, "phase_deg": phase_deg})

    if resource == "BESS":
        num, den = [1], [tau_bess_s, 1]
    elif resource == "EV":
        num, den = [1], [tau_ev_s, 1]
    elif hvac_mode == HVAC_MODES[1]:
        num, den = [1], [tau_hvac_elec_s, 1]
    else:
        num, den = [1], [tau_hvac1_s * tau_hvac2_s, tau_hvac1_s + tau_hvac2_s, 1]
    w, mag, phase = signal.bode((num, den), w=omega)
    return pd.DataFrame({"omega": w, "mag_db": mag, "phase_deg": phase})

def heuristic_mpc_step(
    forecast_df: pd.DataFrame,
    state: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    fleet_meta: Dict[str, float],
) -> Dict[str, object]:
    """Fallback scheduler used when the LP solver is unavailable.

    It preserves the same output schema as the LP optimizer and allocates
    reserve headroom greedily from fastest resources first. This is deliberately
    conservative: it avoids claiming HVAC/PV flexibility in "Fast only" mode and
    subtracts FCR commitments before assigning aFRR capacity.
    """

    rows = []
    only_fast = resource_mode == "Fast only"
    enable_hvac = 0.0 if only_fast else 1.0
    enable_pv = 0.0 if only_fast else 1.0
    bess_soc = float(state["bess_soc_mwh"])
    ev_delta = float(state["ev_soc_delta_mwh"])
    temp_delta = float(state["temp_delta_c"])
    bess_cap = max(float(fleet_meta["bess_energy_cap_mwh"]), 1e-6)
    eta_c = 0.95
    eta_d = 0.95
    fcr_endurance_h = float(FINGRID_RULES["FCR-N"]["storage_endurance_hours"])
    hvac_response_s = float(fleet_meta.get("hvac_response_s", RESOURCE_RESPONSE_SECONDS["HVAC"]))
    hvac_delay_s = _fcr_n_resource_delay_seconds("HVAC", fleet_meta)

    for _, row in forecast_df.iterrows():
        fcr_on = market_mode in {"FCR-N", "Combined"}
        afrr_on = market_mode in {"aFRR", "Combined"}
        dt_h = float(row["dt_h"])

        bess_fcr = ev_fcr = hvac_fcr = 0.0
        if fcr_on:
            bess_up_energy_kw = max((bess_soc - 0.10 * bess_cap) * 1000.0 * eta_d / max(fcr_endurance_h, 1e-6), 0.0)
            bess_down_energy_kw = max((0.95 * bess_cap - bess_soc) * 1000.0 / (max(fcr_endurance_h, 1e-6) * eta_c), 0.0)
            bess_fcr = float(min(row["bess_up_kw"], row["bess_down_kw"], bess_up_energy_kw, bess_down_energy_kw))
            ev_actual_soc = float(row["ev_soc_ref_mwh"]) + ev_delta
            ev_up_energy_kw = max((ev_actual_soc - float(row["ev_soc_min_mwh"])) * 1000.0 * eta_d / max(fcr_endurance_h, 1e-6), 0.0)
            ev_down_energy_kw = max((float(row["ev_soc_max_mwh"]) - ev_actual_soc) * 1000.0 / (max(fcr_endurance_h, 1e-6) * eta_c), 0.0)
            ev_fcr = float(min(row["ev_up_kw"], row["ev_down_kw"], ev_up_energy_kw, ev_down_energy_kw))
            hvac_fcr_cap = enable_hvac * fcr_n_dynamic_deliverable_kw(row["hvac_up_kw"], row["hvac_down_kw"], hvac_response_s, hvac_delay_s)
            hvac_fcr = float(min(hvac_fcr_cap, _fcr_n_hvac_share_limit(bess_fcr + ev_fcr, fleet_meta)))

        bess_up_av = max(float(row["bess_up_kw"]) - bess_fcr, 0.0)
        bess_down_av = max(float(row["bess_down_kw"]) - bess_fcr, 0.0)
        ev_up_av = max(float(row["ev_up_kw"]) - ev_fcr, 0.0)
        ev_down_av = max(float(row["ev_down_kw"]) - ev_fcr, 0.0)
        hvac_up_av = enable_hvac * max(float(row["hvac_up_kw"]) - hvac_fcr, 0.0)
        hvac_down_av = enable_hvac * max(float(row["hvac_down_kw"]) - hvac_fcr, 0.0)
        pv_down = enable_pv * float(row["pv_down_kw"]) if afrr_on else 0.0

        bess_up = bess_up_av if afrr_on else 0.0
        bess_down = bess_down_av if afrr_on else 0.0
        ev_up = ev_up_av if afrr_on else 0.0
        ev_down = ev_down_av if afrr_on else 0.0
        hvac_up = hvac_up_av if afrr_on else 0.0
        hvac_down = hvac_down_av if afrr_on else 0.0

        fcr_up_frac = float(row["fcr_up_act_frac"])
        fcr_down_frac = float(row["fcr_down_act_frac"])
        afrr_up_frac = float(row["afrr_up_act_frac"])
        afrr_down_frac = float(row["afrr_down_act_frac"])

        bess_soc += dt_h / 1000.0 * (
            eta_c * (afrr_down_frac * bess_down + fcr_down_frac * bess_fcr)
            - (afrr_up_frac * bess_up + fcr_up_frac * bess_fcr) / eta_d
        )
        bess_soc = float(np.clip(bess_soc, 0.10 * bess_cap, 0.95 * bess_cap))
        ev_delta += dt_h / 1000.0 * (
            eta_c * (afrr_down_frac * ev_down + fcr_down_frac * ev_fcr)
            - (afrr_up_frac * ev_up + fcr_up_frac * ev_fcr) / eta_d
        )
        temp_delta = 0.90 * temp_delta + 0.16 * dt_h * (
            ((afrr_down_frac * hvac_down + fcr_down_frac * hvac_fcr) - (afrr_up_frac * hvac_up + fcr_up_frac * hvac_fcr))
            / max(float(fleet_meta["n_hvac"]), 1.0)
        )

        values = _apply_fcr_n_bid_rules(
            {
                "bess_fcr_kw": bess_fcr,
                "ev_fcr_kw": ev_fcr,
                "hvac_fcr_kw": hvac_fcr,
                "bess_afrr_up_kw": bess_up,
                "bess_afrr_down_kw": bess_down,
                "ev_afrr_up_kw": ev_up,
                "ev_afrr_down_kw": ev_down,
                "hvac_afrr_up_kw": hvac_up,
                "hvac_afrr_down_kw": hvac_down,
                "pv_afrr_down_kw": pv_down,
                "bess_soc_mwh_plan": bess_soc,
                "ev_delta_mwh_plan": ev_delta,
                "temp_delta_c_plan": temp_delta,
                "hvac_fcr_dynamic_cap_kw": hvac_fcr_cap if fcr_on else 0.0,
                "hvac_fcr_dynamic_fraction": fcr_n_dynamic_deliverable_fraction(hvac_response_s, hvac_delay_s),
                "fcr_hvac_share_cap": _fcr_n_hvac_share_cap(fleet_meta),
            }
        )
        values.update(_fcr_n_bid_dynamic_metrics(values, fleet_meta))
        rows.append(values)

    schedule = pd.DataFrame(rows, index=forecast_df.index)
    schedule["fcr_bid_kw"] = schedule["bess_fcr_kw"] + schedule["ev_fcr_kw"] + schedule["hvac_fcr_kw"]
    schedule["afrr_up_bid_kw"] = schedule["bess_afrr_up_kw"] + schedule["ev_afrr_up_kw"] + schedule["hvac_afrr_up_kw"]
    schedule["afrr_down_bid_kw"] = (
        schedule["bess_afrr_down_kw"] + schedule["ev_afrr_down_kw"] + schedule["hvac_afrr_down_kw"] + schedule["pv_afrr_down_kw"]
    )
    return {"status": "HeuristicFallback", "controls": schedule.iloc[0].to_dict(), "schedule": schedule}


def _risk_factor(resource: str, risk_quantile: float, reserve_buffer_pct: float) -> float:
    quantile = float(np.clip(risk_quantile, 0.50, 0.95))
    buffer = float(np.clip(reserve_buffer_pct, 0.0, 0.40))
    stress = (quantile - 0.50) / 0.45
    uncertainty = RESOURCE_FORECAST_UNCERTAINTY.get(resource, 0.18)
    return float(np.clip((1.0 - uncertainty * stress) * (1.0 - buffer), 0.45, 1.0))


def _risk_factors(risk_quantile: float, reserve_buffer_pct: float) -> Dict[str, float]:
    return {resource: _risk_factor(resource, risk_quantile, reserve_buffer_pct) for resource in ["BESS", "EV", "HVAC", "PV"]}


def _price_bid_quantile(risk_quantile: float) -> float:
    return float(np.clip(1.0 - float(np.clip(risk_quantile, 0.50, 0.95)), 0.05, 0.50))


def _price_bid_column(price_column: str) -> str:
    return price_column.replace("_eur_per_mw_h", "_bid_eur_per_mw_h")


def _price_quantile_column(price_column: str) -> str:
    return price_column.replace("_eur_per_mw_h", "_bid_quantile")


def _price_forecast_mode_column(price_column: str) -> str:
    return price_column.replace("_eur_per_mw_h", "_forecast_mode")


def _market_price_bid_value(row: pd.Series, price_column: str, risk_quantile: float) -> tuple[float, float, str]:
    target_quantile = _price_bid_quantile(risk_quantile)
    candidates: list[tuple[float, str]] = []
    for quantile in FORECAST_QUANTILES:
        column = _forecast_quantile_column(price_column, quantile)
        if column in row and pd.notna(row.get(column)):
            candidates.append((float(quantile), column))
    if candidates:
        lower_or_equal = [(quantile, column) for quantile, column in candidates if quantile <= target_quantile + 1e-9]
        used_quantile, used_column = max(lower_or_equal, key=lambda item: item[0]) if lower_or_equal else min(candidates, key=lambda item: item[0])
        return float(max(float(row[used_column]), 0.0)), float(used_quantile), "quantile"
    return float(max(float(row.get(price_column, 0.0)), 0.0)), 0.50, "point"


def _attach_market_price_bid_forecasts(forecast_df: pd.DataFrame, risk_quantile: float) -> pd.DataFrame:
    frame = forecast_df.copy()
    for price_column in PRICE_FORECAST_TARGETS:
        bid_values = []
        used_quantiles = []
        modes = []
        for _, row in frame.iterrows():
            bid_value, used_quantile, mode = _market_price_bid_value(row, price_column, risk_quantile)
            bid_values.append(bid_value)
            used_quantiles.append(used_quantile)
            modes.append(mode)
        frame[_price_bid_column(price_column)] = bid_values
        frame[_price_quantile_column(price_column)] = used_quantiles
        frame[_price_forecast_mode_column(price_column)] = modes
    frame["price_bid_quantile"] = float(_price_bid_quantile(risk_quantile))
    frame["price_forecast_mode"] = "quantile" if any(
        frame[_price_forecast_mode_column(column)].eq("quantile").any() for column in PRICE_FORECAST_TARGETS
    ) else "point"
    return frame


def _market_price_bid_audit(row: pd.Series) -> Dict[str, object]:
    values: Dict[str, object] = {
        "price_forecast_mode": str(row.get("price_forecast_mode", "point")),
        "price_bid_quantile": float(row.get("price_bid_quantile", 0.50)),
    }
    for price_column in PRICE_FORECAST_TARGETS:
        bid_column = _price_bid_column(price_column)
        values[bid_column] = float(row.get(bid_column, row.get(price_column, 0.0)))
        values[_price_quantile_column(price_column)] = float(row.get(_price_quantile_column(price_column), 0.50))
        values[_price_forecast_mode_column(price_column)] = str(row.get(_price_forecast_mode_column(price_column), "point"))
    return values


def _risk_cost_terms(
    values: Dict[str, float],
    row: pd.Series,
    dt_h: float,
    risk_factors: Dict[str, float],
    penalty_weights: Dict[str, float],
) -> Dict[str, float]:
    non_delivery_penalty = float(penalty_weights.get("non_delivery", 900.0))
    activation_uncertainty_weight = float(penalty_weights.get("activation_uncertainty", 100.0))
    asset_fatigue_weight = float(penalty_weights.get("asset_fatigue", 75.0))
    q = float(np.clip(penalty_weights.get("risk_quantile", 0.80), 0.50, 0.95))

    fcr_signal_risk = float(np.clip(row.get("fcr_up_act_frac", 0.0) + row.get("fcr_down_act_frac", 0.0), 0.0, 1.0))
    afrr_up_risk = float(np.clip(row.get("afrr_up_act_frac", 0.0), 0.0, 1.0))
    afrr_down_risk = float(np.clip(row.get("afrr_down_act_frac", 0.0), 0.0, 1.0))
    activation_variance = fcr_signal_risk * (1.0 - fcr_signal_risk) + afrr_up_risk * (1.0 - afrr_up_risk) + afrr_down_risk * (1.0 - afrr_down_risk)
    delivery_exposure = float(np.clip(max(fcr_signal_risk, afrr_up_risk, afrr_down_risk, 0.05), 0.05, 1.0))

    resource_bid = {
        "BESS": values.get("bess_fcr_kw", 0.0) + values.get("bess_afrr_up_kw", 0.0) + values.get("bess_afrr_down_kw", 0.0),
        "EV": values.get("ev_fcr_kw", 0.0) + values.get("ev_afrr_up_kw", 0.0) + values.get("ev_afrr_down_kw", 0.0),
        "HVAC": values.get("hvac_fcr_kw", 0.0) + values.get("hvac_afrr_up_kw", 0.0) + values.get("hvac_afrr_down_kw", 0.0),
        "PV": values.get("pv_afrr_down_kw", 0.0),
    }
    total_bid_kw = sum(resource_bid.values())
    non_delivery_risk = dt_h / 1000.0 * non_delivery_penalty * delivery_exposure * sum(
        max(1.0 - risk_factors.get(resource, 1.0), 0.0) * bid for resource, bid in resource_bid.items()
    )
    activation_uncertainty = dt_h / 1000.0 * activation_uncertainty_weight * activation_variance * total_bid_kw
    asset_fatigue = dt_h / 1000.0 * asset_fatigue_weight * delivery_exposure * (
        1.0 * resource_bid["BESS"] + 0.65 * resource_bid["EV"] + 0.45 * resource_bid["HVAC"] + 0.20 * resource_bid["PV"]
    )
    reserve_buffer_kw = max((q - 0.50) / 0.45, 0.0) * sum(
        RESOURCE_FORECAST_UNCERTAINTY.get(resource, 0.18) * bid for resource, bid in resource_bid.items()
    )
    return {
        "non_delivery_risk_cost_eur": non_delivery_risk,
        "activation_uncertainty_cost_eur": activation_uncertainty,
        "asset_fatigue_cost_eur": asset_fatigue,
        "reserve_buffer_kw": reserve_buffer_kw,
    }


def solve_mpc_step(
    forecast_df: pd.DataFrame,
    state: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    penalty_weights: Dict[str, float],
    fleet_meta: Dict[str, float],
) -> Dict[str, object]:
    penalty_weights = apply_risk_policy_defaults(penalty_weights)
    risk_quantile = float(np.clip(penalty_weights.get("risk_quantile", 0.80), 0.50, 0.95))
    forecast_df = _attach_market_price_bid_forecasts(forecast_df, risk_quantile)
    horizon = len(forecast_df)
    dt_h = float(forecast_df["dt_h"].iloc[0])
    eta_c = 0.95
    eta_d = 0.95
    temp_alpha = 0.90
    temp_beta = 0.16 * dt_h
    reserve_buffer_pct = float(np.clip(penalty_weights.get("reserve_buffer_pct", 0.08), 0.0, 0.40))
    risk_factors = _risk_factors(risk_quantile, reserve_buffer_pct)

    only_fast = resource_mode == "Fast only"
    enable_hvac = 0.0 if only_fast else 1.0
    enable_pv = 0.0 if only_fast else 1.0
    hvac_response_s = float(fleet_meta.get("hvac_response_s", RESOURCE_RESPONSE_SECONDS["HVAC"]))
    hvac_delay_s = _fcr_n_resource_delay_seconds("HVAC", fleet_meta)
    hvac_fcr_dynamic_fraction = fcr_n_dynamic_deliverable_fraction(hvac_response_s, hvac_delay_s)
    fcr_endurance_h = float(FINGRID_RULES["FCR-N"]["storage_endurance_hours"])
    afrr_endurance_h = float(FINGRID_RULES["aFRR"]["storage_endurance_hours"])
    reserve_endurance_h = max(fcr_endurance_h, afrr_endurance_h)
    fcr_granularity_kw = float(FINGRID_RULES["FCR-N"]["bid_granularity_kw"])
    afrr_granularity_kw = float(FINGRID_RULES["aFRR"]["bid_granularity_kw"])
    fcr_eligible = {
        "BESS": 1.0 if fcr_n_local_control_capable("BESS") else 0.0,
        "EV": 1.0 if fcr_n_local_control_capable("EV") else 0.0,
        "HVAC": 1.0 if fcr_n_local_control_capable("HVAC") and hvac_fcr_dynamic_fraction > 1e-6 else 0.0,
    }
    afrr_eligible = {
        "BESS": 1.0 if RESOURCE_RESPONSE_SECONDS["BESS"] <= FINGRID_RULES["aFRR"]["full_activation_seconds"] else 0.0,
        "EV": 1.0 if RESOURCE_RESPONSE_SECONDS["EV"] <= FINGRID_RULES["aFRR"]["full_activation_seconds"] else 0.0,
        "HVAC": 1.0 if hvac_response_s <= FINGRID_RULES["aFRR"]["full_activation_seconds"] else 0.0,
        "PV": 1.0 if RESOURCE_RESPONSE_SECONDS["PV"] <= FINGRID_RULES["aFRR"]["full_activation_seconds"] else 0.0,
    }

    problem = LpProblem("FlexiHome_MPC", LpMaximize)
    idxs = list(range(horizon))

    soc_b = LpVariable.dicts("soc_b", range(horizon + 1), lowBound=0)
    ev_delta = LpVariable.dicts("ev_delta", range(horizon + 1))
    temp_delta = LpVariable.dicts("temp_delta", range(horizon + 1))

    bess_fcr = LpVariable.dicts("bess_fcr", idxs, lowBound=0)
    ev_fcr = LpVariable.dicts("ev_fcr", idxs, lowBound=0)
    hvac_fcr = LpVariable.dicts("hvac_fcr", idxs, lowBound=0)
    bess_up = LpVariable.dicts("bess_up", idxs, lowBound=0)
    bess_down = LpVariable.dicts("bess_down", idxs, lowBound=0)
    ev_up = LpVariable.dicts("ev_up", idxs, lowBound=0)
    ev_down = LpVariable.dicts("ev_down", idxs, lowBound=0)
    hvac_up = LpVariable.dicts("hvac_up", idxs, lowBound=0)
    hvac_down = LpVariable.dicts("hvac_down", idxs, lowBound=0)
    pv_down = LpVariable.dicts("pv_down", idxs, lowBound=0)
    fcr_segments = LpVariable.dicts("fcr_segments", idxs, lowBound=0, cat=LpInteger)
    afrr_up_segments = LpVariable.dicts("afrr_up_segments", idxs, lowBound=0, cat=LpInteger)
    afrr_down_segments = LpVariable.dicts("afrr_down_segments", idxs, lowBound=0, cat=LpInteger)

    ev_slack = LpVariable.dicts("ev_slack", idxs, lowBound=0)
    temp_low_slack = LpVariable.dicts("temp_low_slack", idxs, lowBound=0)
    temp_high_slack = LpVariable.dicts("temp_high_slack", idxs, lowBound=0)

    problem += soc_b[0] == state["bess_soc_mwh"]
    problem += ev_delta[0] == state["ev_soc_delta_mwh"]
    problem += temp_delta[0] == state["temp_delta_c"]

    objective_terms = []
    bess_cap = max(fleet_meta["bess_energy_cap_mwh"], 1e-6)

    for k in idxs:
        row = forecast_df.iloc[k]
        fcr_on = 1.0 if market_mode in {"FCR-N", "Combined"} else 0.0
        afrr_on = 1.0 if market_mode in {"aFRR", "Combined"} else 0.0
        bess_up_cap = float(row["bess_up_kw"]) * risk_factors["BESS"]
        bess_down_cap = float(row["bess_down_kw"]) * risk_factors["BESS"]
        ev_up_cap = float(row["ev_up_kw"]) * risk_factors["EV"]
        ev_down_cap = float(row["ev_down_kw"]) * risk_factors["EV"]
        hvac_up_cap = enable_hvac * float(row["hvac_up_kw"]) * risk_factors["HVAC"]
        hvac_down_cap = enable_hvac * float(row["hvac_down_kw"]) * risk_factors["HVAC"]
        hvac_fcr_cap = enable_hvac * min(float(row["hvac_up_kw"]), float(row["hvac_down_kw"])) * risk_factors["HVAC"] * hvac_fcr_dynamic_fraction * fcr_eligible["HVAC"]
        pv_down_cap = enable_pv * afrr_eligible["PV"] * float(row["pv_down_kw"]) * risk_factors["PV"]

        problem += bess_fcr[k] <= min(bess_up_cap, bess_down_cap) * fcr_eligible["BESS"]
        problem += ev_fcr[k] <= min(ev_up_cap, ev_down_cap) * fcr_eligible["EV"]
        problem += hvac_fcr[k] <= hvac_fcr_cap
        problem += bess_fcr[k] + bess_up[k] <= bess_up_cap
        problem += bess_fcr[k] + bess_down[k] <= bess_down_cap
        problem += ev_fcr[k] + ev_up[k] <= ev_up_cap
        problem += ev_fcr[k] + ev_down[k] <= ev_down_cap
        problem += hvac_fcr[k] + hvac_up[k] <= hvac_up_cap
        problem += hvac_fcr[k] + hvac_down[k] <= hvac_down_cap
        problem += hvac_up[k] <= hvac_up_cap * afrr_eligible["HVAC"]
        problem += hvac_down[k] <= hvac_down_cap * afrr_eligible["HVAC"]
        if _fcr_n_hvac_share_cap(fleet_meta) < 0.999:
            problem += (1.0 - _fcr_n_hvac_share_cap(fleet_meta)) * hvac_fcr[k] <= _fcr_n_hvac_share_cap(fleet_meta) * (bess_fcr[k] + ev_fcr[k])
        problem += pv_down[k] <= pv_down_cap
        fcr_bid = bess_fcr[k] + ev_fcr[k] + hvac_fcr[k]
        afrr_up_bid = bess_up[k] + ev_up[k] + hvac_up[k]
        afrr_down_bid = bess_down[k] + ev_down[k] + hvac_down[k] + pv_down[k]
        problem += fcr_bid == fcr_segments[k] * fcr_granularity_kw
        problem += afrr_up_bid == afrr_up_segments[k] * afrr_granularity_kw
        problem += afrr_down_bid == afrr_down_segments[k] * afrr_granularity_kw
        problem += bess_fcr[k] + bess_up[k] <= (soc_b[k] - 0.10 * bess_cap) * 1000.0 * eta_d / max(reserve_endurance_h, 1e-6)
        problem += bess_fcr[k] + bess_down[k] <= (0.95 * bess_cap - soc_b[k]) * 1000.0 / (max(reserve_endurance_h, 1e-6) * eta_c)

        if fcr_on < 0.5:
            problem += bess_fcr[k] == 0
            problem += ev_fcr[k] == 0
            problem += hvac_fcr[k] == 0
        if afrr_on < 0.5:
            problem += bess_up[k] == 0
            problem += bess_down[k] == 0
            problem += ev_up[k] == 0
            problem += ev_down[k] == 0
            problem += hvac_up[k] == 0
            problem += hvac_down[k] == 0
            problem += pv_down[k] == 0

        fcr_up_frac = row["fcr_up_act_frac"]
        fcr_down_frac = row["fcr_down_act_frac"]
        afrr_up_frac = row["afrr_up_act_frac"]
        afrr_down_frac = row["afrr_down_act_frac"]

        problem += soc_b[k + 1] == soc_b[k] + dt_h / 1000.0 * (
            eta_c * (afrr_down_frac * bess_down[k] + fcr_down_frac * bess_fcr[k])
            - (afrr_up_frac * bess_up[k] + fcr_up_frac * bess_fcr[k]) / eta_d
        )
        problem += soc_b[k + 1] >= 0.10 * bess_cap
        problem += soc_b[k + 1] <= 0.95 * bess_cap

        ev_ref = row["ev_soc_ref_mwh"]
        ev_min = row["ev_soc_min_mwh"]
        ev_max = row["ev_soc_max_mwh"]
        ev_actual = ev_ref + ev_delta[k]
        problem += ev_fcr[k] + ev_up[k] <= (ev_actual - ev_min) * 1000.0 * eta_d / max(reserve_endurance_h, 1e-6)
        problem += ev_fcr[k] + ev_down[k] <= (ev_max - ev_actual) * 1000.0 / (max(reserve_endurance_h, 1e-6) * eta_c)
        problem += ev_delta[k + 1] == ev_delta[k] + dt_h / 1000.0 * (
            eta_c * (afrr_down_frac * ev_down[k] + fcr_down_frac * ev_fcr[k])
            - (afrr_up_frac * ev_up[k] + fcr_up_frac * ev_fcr[k]) / eta_d
        )
        problem += ev_ref + ev_delta[k + 1] >= ev_min - ev_slack[k]
        problem += ev_ref + ev_delta[k + 1] <= ev_max

        temp_ref = row["indoor_temp_c_ref"]
        temp_lo = fleet_meta["setpoint_c"] - fleet_meta["comfort_band_c"]
        temp_hi = fleet_meta["setpoint_c"] + fleet_meta["comfort_band_c"]
        comfort_band = max(float(fleet_meta["comfort_band_c"]), 0.2)
        problem += hvac_fcr[k] + hvac_up[k] <= enable_hvac * max(float(row["hvac_up_kw"]), 0.0) * (temp_ref + temp_delta[k] - temp_lo + temp_low_slack[k]) / comfort_band
        problem += hvac_fcr[k] + hvac_down[k] <= enable_hvac * max(float(row["hvac_down_kw"]), 0.0) * (temp_hi - temp_ref - temp_delta[k] + temp_high_slack[k]) / comfort_band
        problem += temp_delta[k + 1] == temp_alpha * temp_delta[k] + temp_beta * (
            (
                (afrr_down_frac * hvac_down[k] + fcr_down_frac * hvac_fcr[k])
                - (afrr_up_frac * hvac_up[k] + fcr_up_frac * hvac_fcr[k])
            )
            / max(fleet_meta["n_hvac"], 1)
        )
        problem += temp_ref + temp_delta[k + 1] >= temp_lo - temp_low_slack[k]
        problem += temp_ref + temp_delta[k + 1] <= temp_hi + temp_high_slack[k]

        cap_revenue = dt_h / 1000.0 * (
            row[_price_bid_column("fcrn_capacity_eur_per_mw_h")] * fcr_bid
            + row[_price_bid_column("afrr_up_capacity_eur_per_mw_h")] * afrr_up_bid
            + row[_price_bid_column("afrr_down_capacity_eur_per_mw_h")] * afrr_down_bid
        )
        energy_revenue = dt_h / 1000.0 * (
            row["afrr_up_energy_eur_per_mwh"] * afrr_up_frac * afrr_up_bid
            + row["afrr_down_energy_eur_per_mwh"] * afrr_down_frac * afrr_down_bid
        )
        degradation = penalty_weights["degradation"] * dt_h / 1000.0 * (
            (afrr_up_frac * bess_up[k] + afrr_down_frac * bess_down[k] + (fcr_up_frac + fcr_down_frac) * bess_fcr[k])
            + 0.4 * (afrr_up_frac * ev_up[k] + afrr_down_frac * ev_down[k] + (fcr_up_frac + fcr_down_frac) * ev_fcr[k])
        )
        discomfort = (
            penalty_weights["comfort"] * dt_h * (temp_low_slack[k] + temp_high_slack[k])
            + penalty_weights["departure"] * ev_slack[k]
        )
        risk_terms = _risk_cost_terms(
            {
                "bess_fcr_kw": bess_fcr[k],
                "ev_fcr_kw": ev_fcr[k],
                "hvac_fcr_kw": hvac_fcr[k],
                "bess_afrr_up_kw": bess_up[k],
                "bess_afrr_down_kw": bess_down[k],
                "ev_afrr_up_kw": ev_up[k],
                "ev_afrr_down_kw": ev_down[k],
                "hvac_afrr_up_kw": hvac_up[k],
                "hvac_afrr_down_kw": hvac_down[k],
                "pv_afrr_down_kw": pv_down[k],
            },
            row,
            dt_h,
            risk_factors,
            penalty_weights,
        )
        objective_terms.append(
            cap_revenue
            + energy_revenue
            - degradation
            - discomfort
            - risk_terms["asset_fatigue_cost_eur"]
        )

    problem += lpSum(objective_terms)
    try:
        problem.solve(PULP_CBC_CMD(msg=False))
    except Exception:
        return heuristic_mpc_step(forecast_df, state, market_mode, resource_mode, fleet_meta)

    if LpStatus[problem.status] not in {"Optimal", "Not Solved"}:
        return heuristic_mpc_step(forecast_df, state, market_mode, resource_mode, fleet_meta)

    rows = []
    for k in idxs:
        row = forecast_df.iloc[k]
        values = {
            "bess_fcr_kw": float(bess_fcr[k].value() or 0.0),
            "ev_fcr_kw": float(ev_fcr[k].value() or 0.0),
            "hvac_fcr_kw": float(hvac_fcr[k].value() or 0.0),
            "bess_afrr_up_kw": float(bess_up[k].value() or 0.0),
            "bess_afrr_down_kw": float(bess_down[k].value() or 0.0),
            "ev_afrr_up_kw": float(ev_up[k].value() or 0.0),
            "ev_afrr_down_kw": float(ev_down[k].value() or 0.0),
            "hvac_afrr_up_kw": float(hvac_up[k].value() or 0.0),
            "hvac_afrr_down_kw": float(hvac_down[k].value() or 0.0),
            "pv_afrr_down_kw": float(pv_down[k].value() or 0.0),
        }
        values = _market_bid_metadata(values, market_mode=market_mode, timestamp=row.name)
        hvac_fcr_cap_val = enable_hvac * min(float(row["hvac_up_kw"]), float(row["hvac_down_kw"])) * risk_factors["HVAC"] * hvac_fcr_dynamic_fraction * fcr_eligible["HVAC"]
        soc_b_val = float(soc_b[k].value() if soc_b[k].value() is not None else state["bess_soc_mwh"])
        ev_actual_val = float(row["ev_soc_ref_mwh"]) + float(ev_delta[k].value() or 0.0)
        bess_energy_up_kw = max((soc_b_val - 0.10 * bess_cap) * 1000.0 * eta_d / max(reserve_endurance_h, 1e-6), 0.0)
        bess_energy_down_kw = max((0.95 * bess_cap - soc_b_val) * 1000.0 / (max(reserve_endurance_h, 1e-6) * eta_c), 0.0)
        ev_energy_up_kw = max((ev_actual_val - float(row["ev_soc_min_mwh"])) * 1000.0 * eta_d / max(reserve_endurance_h, 1e-6), 0.0)
        ev_energy_down_kw = max((float(row["ev_soc_max_mwh"]) - ev_actual_val) * 1000.0 / (max(reserve_endurance_h, 1e-6) * eta_c), 0.0)
        raw_fcr_symmetric_kw = (
            min(float(row["bess_up_kw"]), float(row["bess_down_kw"]))
            + min(float(row["ev_up_kw"]), float(row["ev_down_kw"]))
            + min(float(row["hvac_up_kw"]), float(row["hvac_down_kw"]))
        )
        fcr_dynamic_cap_total_kw = (
            min(float(row["bess_up_kw"]), float(row["bess_down_kw"]))
            + min(float(row["ev_up_kw"]), float(row["ev_down_kw"]))
            + enable_hvac * min(float(row["hvac_up_kw"]), float(row["hvac_down_kw"])) * hvac_fcr_dynamic_fraction
        )
        fcr_energy_cap_total_kw = (
            min(bess_up_cap, bess_down_cap, bess_energy_up_kw, bess_energy_down_kw)
            + min(ev_up_cap, ev_down_cap, ev_energy_up_kw, ev_energy_down_kw)
            + hvac_fcr_cap_val
        )
        values.update(
            {
                "hvac_fcr_dynamic_cap_kw": float(hvac_fcr_cap_val),
                "hvac_fcr_dynamic_fraction": float(hvac_fcr_dynamic_fraction),
                "fcr_hvac_share_cap": _fcr_n_hvac_share_cap(fleet_meta),
                "raw_fcr_symmetric_kw": float(raw_fcr_symmetric_kw),
                "fcr_dynamic_cap_total_kw": float(fcr_dynamic_cap_total_kw),
                "fcr_energy_cap_total_kw": float(fcr_energy_cap_total_kw),
            }
        )
        values.update(_fcr_n_bid_dynamic_metrics(values, fleet_meta))
        values.update(_market_price_bid_audit(row))
        fcr_bid_val = values["fcr_bid_kw"]
        afrr_up_bid_val = values["afrr_up_bid_kw"]
        afrr_down_bid_val = values["afrr_down_bid_kw"]
        cap_revenue = dt_h / 1000.0 * (
            row[_price_bid_column("fcrn_capacity_eur_per_mw_h")] * fcr_bid_val
            + row[_price_bid_column("afrr_up_capacity_eur_per_mw_h")] * afrr_up_bid_val
            + row[_price_bid_column("afrr_down_capacity_eur_per_mw_h")] * afrr_down_bid_val
        )
        energy_revenue = dt_h / 1000.0 * (
            row["afrr_up_energy_eur_per_mwh"] * row["afrr_up_act_frac"] * afrr_up_bid_val
            + row["afrr_down_energy_eur_per_mwh"] * row["afrr_down_act_frac"] * afrr_down_bid_val
        )
        degradation = penalty_weights["degradation"] * dt_h / 1000.0 * (
            (row["afrr_up_act_frac"] * values["bess_afrr_up_kw"] + row["afrr_down_act_frac"] * values["bess_afrr_down_kw"] + (row["fcr_up_act_frac"] + row["fcr_down_act_frac"]) * values["bess_fcr_kw"])
            + 0.4 * (row["afrr_up_act_frac"] * values["ev_afrr_up_kw"] + row["afrr_down_act_frac"] * values["ev_afrr_down_kw"] + (row["fcr_up_act_frac"] + row["fcr_down_act_frac"]) * values["ev_fcr_kw"])
        )
        discomfort = (
            penalty_weights["comfort"] * dt_h * float((temp_low_slack[k].value() or 0.0) + (temp_high_slack[k].value() or 0.0))
            + penalty_weights["departure"] * float(ev_slack[k].value() or 0.0)
        )
        risk_costs = _risk_cost_terms(values, row, dt_h, risk_factors, penalty_weights)
        risk_adjusted_profit = (
            cap_revenue
            + energy_revenue
            - degradation
            - discomfort
            - float(risk_costs["asset_fatigue_cost_eur"])
        )
        values.update(
            {
                "bess_soc_mwh_plan": float(soc_b[k + 1].value() or 0.0),
                "ev_delta_mwh_plan": float(ev_delta[k + 1].value() or 0.0),
                "temp_delta_c_plan": float(temp_delta[k + 1].value() or 0.0),
                "gross_revenue_eur": float(cap_revenue + energy_revenue),
                "expected_degradation_eur": float(degradation),
                "expected_comfort_cost_eur": float(discomfort),
                "non_delivery_risk_cost_eur": float(risk_costs["non_delivery_risk_cost_eur"]),
                "activation_uncertainty_cost_eur": float(risk_costs["activation_uncertainty_cost_eur"]),
                "asset_fatigue_cost_eur": float(risk_costs["asset_fatigue_cost_eur"]),
                "risk_adjusted_profit_eur": float(risk_adjusted_profit),
                "risk_policy": str(penalty_weights.get("risk_policy", "investor_balanced")),
                "risk_quantile": float(risk_quantile),
                "reserve_buffer_pct": float(reserve_buffer_pct),
                "reserve_buffer_kw": float(risk_costs["reserve_buffer_kw"]),
                "stacking_ok": True,
                "energy_endurance_ok": True,
            }
        )
        rows.append(values)
    schedule = pd.DataFrame(rows, index=forecast_df.index)
    schedule["fcr_bid_kw"] = schedule["bess_fcr_kw"] + schedule["ev_fcr_kw"] + schedule["hvac_fcr_kw"]
    schedule["afrr_up_bid_kw"] = schedule["bess_afrr_up_kw"] + schedule["ev_afrr_up_kw"] + schedule["hvac_afrr_up_kw"]
    schedule["afrr_down_bid_kw"] = (
        schedule["bess_afrr_down_kw"] + schedule["ev_afrr_down_kw"] + schedule["hvac_afrr_down_kw"] + schedule["pv_afrr_down_kw"]
    )
    return {"status": LpStatus[problem.status], "controls": schedule.iloc[0].to_dict(), "schedule": schedule}

def _available_up_down(
    row: pd.Series,
    state: Dict[str, float],
    fleet_meta: Dict[str, float],
    dt_h_override: float | None = None,
) -> Dict[str, float]:
    dt_h = float(dt_h_override if dt_h_override is not None else row["dt_h"])
    eta_c = 0.95
    eta_d = 0.95

    bess_cap = max(fleet_meta["bess_energy_cap_mwh"], 1e-6)
    bess_up_energy_kw = max((state["bess_soc_mwh"] - 0.10 * bess_cap) * 1000.0 * eta_d / max(dt_h, 1e-6), 0.0)
    bess_down_energy_kw = max((0.95 * bess_cap - state["bess_soc_mwh"]) * 1000.0 / (max(dt_h, 1e-6) * eta_c), 0.0)
    bess_up_av = min(row["bess_up_kw"], bess_up_energy_kw)
    bess_down_av = min(row["bess_down_kw"], bess_down_energy_kw)

    ev_actual_soc = row["ev_soc_ref_mwh"] + state["ev_soc_delta_mwh"]
    ev_up_energy_kw = max((ev_actual_soc - row["ev_soc_min_mwh"]) * 1000.0 * eta_d / max(dt_h, 1e-6), 0.0)
    ev_down_energy_kw = max((row["ev_soc_max_mwh"] - ev_actual_soc) * 1000.0 / (max(dt_h, 1e-6) * eta_c), 0.0)
    ev_up_av = min(row["ev_up_kw"], ev_up_energy_kw)
    ev_down_av = min(row["ev_down_kw"], ev_down_energy_kw)

    temp_actual = row["indoor_temp_c_ref"] + state["temp_delta_c"]
    temp_lo = fleet_meta["setpoint_c"] - fleet_meta["comfort_band_c"]
    temp_hi = fleet_meta["setpoint_c"] + fleet_meta["comfort_band_c"]
    comfort_factor_up = float(np.clip((temp_actual - temp_lo) / max(fleet_meta["comfort_band_c"], 0.2), 0, 1))
    comfort_factor_down = float(np.clip((temp_hi - temp_actual) / max(fleet_meta["comfort_band_c"], 0.2), 0, 1))
    hvac_up_av = row["hvac_up_kw"] * comfort_factor_up
    hvac_down_av = row["hvac_down_kw"] * comfort_factor_down
    hvac_response_s = float(fleet_meta.get("hvac_response_s", default_hvac_response_seconds(str(fleet_meta.get("hvac_mode", HVAC_MODES[0])))))
    hvac_fast_av = fcr_n_dynamic_deliverable_kw(
        hvac_up_av,
        hvac_down_av,
        hvac_response_s,
        _fcr_n_resource_delay_seconds("HVAC", fleet_meta),
    )

    return {
        "bess_up_av": bess_up_av,
        "bess_down_av": bess_down_av,
        "ev_up_av": ev_up_av,
        "ev_down_av": ev_down_av,
        "hvac_up_av": hvac_up_av,
        "hvac_down_av": hvac_down_av,
        "hvac_fast_av": hvac_fast_av,
        "pv_down_av": row["pv_down_kw"],
    }


def _market_gate_decision(plan: Dict[str, float], market_mode: str) -> Dict[str, object]:
    bess_fcr = float(plan.get("bess_fcr_kw", 0.0))
    ev_fcr = float(plan.get("ev_fcr_kw", 0.0))
    hvac_fcr = float(plan.get("hvac_fcr_kw", 0.0))
    fcr_bid = float(bess_fcr + ev_fcr + hvac_fcr)
    afrr_up = float(plan.get("bess_afrr_up_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0))
    afrr_down = float(
        plan.get("bess_afrr_down_kw", 0.0)
        + plan.get("ev_afrr_down_kw", 0.0)
        + plan.get("hvac_afrr_down_kw", 0.0)
        + plan.get("pv_afrr_down_kw", 0.0)
    )
    afrr_bid = max(afrr_up, afrr_down)
    risk_profit = float(plan.get("risk_adjusted_profit_eur", 0.0))
    fcr_enabled = market_mode in {"FCR-N", "Combined"}
    afrr_enabled = market_mode in {"aFRR", "Combined"}
    fcr_present = bool(fcr_enabled and fcr_bid > 1e-6)
    afrr_present = bool(afrr_enabled and afrr_bid > 1e-6)
    fcr_ok = fcr_bid >= FINGRID_RULES["FCR-N"]["min_bid_kw"] if fcr_enabled else False
    afrr_ok = afrr_bid >= FINGRID_RULES["aFRR"]["min_bid_kw"] if afrr_enabled else False
    fcr_local_ok = str(plan.get("fcr_control_source", FINGRID_RULES["FCR-N"]["control_source"])) == FINGRID_RULES["FCR-N"]["control_source"]
    hvac_dynamic_cap = plan.get("hvac_fcr_dynamic_cap_kw")
    fcr_dynamic_cap_ok = True if hvac_dynamic_cap is None else hvac_fcr <= float(hvac_dynamic_cap) + 1e-6
    fcr_dynamic_response_ok = bool(plan.get("fcr_dynamic_response_ok", True))
    hvac_share_cap = float(np.clip(plan.get("fcr_hvac_share_cap", FCR_N_HVAC_SHARE_CAP), 0.0, 1.0))
    fcr_hvac_share_ok = (
        True
        if hvac_share_cap >= 0.999
        else (1.0 - hvac_share_cap) * hvac_fcr <= hvac_share_cap * (bess_fcr + ev_fcr) + 1e-6
    )
    fcr_granularity_kw = float(FINGRID_RULES["FCR-N"]["bid_granularity_kw"])
    fcr_granularity_ok = (
        True
        if not fcr_enabled or fcr_bid <= 1e-6
        else abs((fcr_bid / max(fcr_granularity_kw, 1e-9)) - round(fcr_bid / max(fcr_granularity_kw, 1e-9))) <= 1e-6
    )
    afrr_granularity_kw = float(FINGRID_RULES["aFRR"]["bid_granularity_kw"])
    afrr_up_granularity_ok = (
        True
        if not afrr_enabled or afrr_up <= 1e-6
        else abs((afrr_up / max(afrr_granularity_kw, 1e-9)) - round(afrr_up / max(afrr_granularity_kw, 1e-9))) <= 1e-6
    )
    afrr_down_granularity_ok = (
        True
        if not afrr_enabled or afrr_down <= 1e-6
        else abs((afrr_down / max(afrr_granularity_kw, 1e-9)) - round(afrr_down / max(afrr_granularity_kw, 1e-9))) <= 1e-6
    )
    afrr_granularity_ok = afrr_up_granularity_ok and afrr_down_granularity_ok
    combined_stack_ok = True if market_mode != "Combined" else _truthy(plan.get("combined_shared_capacity_ok", True), default=True)
    fcr_product_ok = fcr_ok and fcr_local_ok and fcr_granularity_ok and fcr_dynamic_cap_ok and fcr_dynamic_response_ok and fcr_hvac_share_ok
    afrr_product_ok = afrr_ok and afrr_granularity_ok
    profitable = risk_profit >= 0.0
    if market_mode == "FCR-N":
        product_ok = fcr_product_ok
        size_reason = "Reliable FCR-N bid is below the selected market minimum size."
    elif market_mode == "aFRR":
        product_ok = afrr_product_ok
        size_reason = "Reliable aFRR bid is below the selected market minimum size."
    else:
        product_ok = (
            combined_stack_ok
            and (fcr_product_ok or afrr_product_ok)
            and (not fcr_present or fcr_product_ok)
            and (not afrr_present or afrr_product_ok)
        )
        size_reason = "Reliable FCR-N and aFRR bids are both below their market minimum sizes."

    if product_ok and profitable and (fcr_bid + afrr_bid > 1.0):
        status = "Participate"
        reason = "Risk-adjusted bid clears product size and expected profit gates."
    elif not profitable:
        status = "Wait"
        reason = "Expected revenue is lower than risk, fatigue, and delivery-cost allowance."
    elif fcr_present and not fcr_ok:
        status = "Wait"
        reason = "FCR-N bid is below the 0.1 MW market minimum."
    elif afrr_present and not afrr_ok:
        status = "Wait"
        reason = "aFRR bid is below the 1 MW market minimum."
    elif fcr_enabled and fcr_ok and not fcr_local_ok:
        status = "Wait"
        reason = "FCR-N requires locally measured symmetric frequency response; this slot includes centrally dispatched FCR capacity."
    elif fcr_enabled and fcr_ok and not fcr_granularity_ok:
        status = "Wait"
        reason = "FCR-N bid is not aligned to the 0.1 MW market granularity."
    elif afrr_enabled and afrr_ok and not afrr_granularity_ok:
        status = "Wait"
        reason = "aFRR bid is not aligned to the 1 MW market granularity."
    elif fcr_enabled and fcr_ok and not fcr_dynamic_cap_ok:
        status = "Wait"
        reason = "FCR-N HVAC contribution exceeds the dynamically deliverable capacity for the selected response speed."
    elif fcr_enabled and fcr_ok and not fcr_dynamic_response_ok:
        status = "Wait"
        reason = "FCR-N aggregate droop response does not satisfy the 60 s, 180 s, and 60 s energy checks."
    elif fcr_enabled and fcr_ok and not fcr_hvac_share_ok:
        status = "Wait"
        reason = "FCR-N HVAC contribution exceeds the configured comfort-disturbance share cap."
    elif market_mode == "Combined" and not combined_stack_ok:
        status = "Wait"
        reason = "Combined FCR-N+aFRR bid is not backed by a split-capacity stacking audit."
    elif not product_ok:
        status = "Wait"
        reason = size_reason
    else:
        status = "Wait"
        reason = "No reliable flexibility bid is available for this slot."
    return {
        "market_gate_status": status,
        "market_gate_reason": reason,
        "risk_adjusted_profit_eur": risk_profit,
        "reserve_buffer_kw": float(plan.get("reserve_buffer_kw", 0.0)),
        "fcr_local_control_ok": bool(fcr_local_ok),
        "fcr_bid_granularity_ok": bool(fcr_granularity_ok),
        "afrr_bid_granularity_ok": bool(afrr_granularity_ok),
        "fcr_dynamic_cap_ok": bool(fcr_dynamic_cap_ok),
        "fcr_dynamic_response_ok": bool(fcr_dynamic_response_ok),
        "fcr_hvac_share_ok": bool(fcr_hvac_share_ok),
        "combined_shared_capacity_ok": bool(combined_stack_ok),
        "fcr_bid_segments": int(plan.get("fcr_bid_segments", 0) or 0),
        "afrr_up_segments": int(plan.get("afrr_up_segments", 0) or 0),
        "afrr_down_segments": int(plan.get("afrr_down_segments", 0) or 0),
    }


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok"}
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return bool(value)


def _combined_split_capacity_enabled(plan: Dict[str, Any]) -> bool:
    return (
        str(plan.get("combined_stacking_model", "")).strip() == COMBINED_STACKING_SPLIT_MODEL
        and _truthy(plan.get("combined_shared_capacity_ok", plan.get("combined_split_capacity_ok", True)), default=True)
    )


def _combined_stack_audit(
    plan: Dict[str, Any],
    row: pd.Series | None,
    market_mode: str,
) -> Dict[str, Any]:
    """Explain whether Combined same-direction socket summing is backed by split capacity."""

    fcr_active = market_mode in {"FCR-N", "Combined"}
    afrr_active = market_mode in {"aFRR", "Combined"}
    resources = {
        "bess": ("bess_fcr_kw", "bess_afrr_up_kw", "bess_afrr_down_kw", "bess_up_kw", "bess_down_kw"),
        "ev": ("ev_fcr_kw", "ev_afrr_up_kw", "ev_afrr_down_kw", "ev_up_kw", "ev_down_kw"),
        "hvac": ("hvac_fcr_kw", "hvac_afrr_up_kw", "hvac_afrr_down_kw", "hvac_up_kw", "hvac_down_kw"),
        "pv": ("", "", "pv_afrr_down_kw", "", "pv_down_kw"),
    }
    audit: Dict[str, Any] = {}
    ok = True
    violations: List[str] = []

    for resource, (fcr_col, afrr_up_col, afrr_down_col, up_cap_col, down_cap_col) in resources.items():
        fcr_kw = float(plan.get(fcr_col, 0.0) or 0.0) if fcr_col and fcr_active else 0.0
        up_kw = fcr_kw + (float(plan.get(afrr_up_col, 0.0) or 0.0) if afrr_up_col and afrr_active else 0.0)
        down_kw = fcr_kw + (float(plan.get(afrr_down_col, 0.0) or 0.0) if afrr_down_col and afrr_active else 0.0)
        audit[f"{resource}_combined_up_stack_kw"] = float(max(up_kw, 0.0))
        audit[f"{resource}_combined_down_stack_kw"] = float(max(down_kw, 0.0))

        if row is None or market_mode != "Combined":
            continue
        for direction, value, cap_col in (("up", up_kw, up_cap_col), ("down", down_kw, down_cap_col)):
            if not cap_col or cap_col not in row:
                continue
            cap = float(row.get(cap_col, np.nan))
            if np.isnan(cap):
                continue
            tolerance = max(1e-6, 0.001 * max(abs(cap), 1.0))
            if value > cap + tolerance:
                ok = False
                violations.append(f"{resource}_{direction}: {value:.1f} kW > {cap:.1f} kW")

    if market_mode == "Combined":
        audit["combined_stacking_model"] = COMBINED_STACKING_SPLIT_MODEL if ok else COMBINED_STACKING_SHARED_MODEL
    else:
        audit["combined_stacking_model"] = "single_product_socket"
    audit["combined_shared_capacity_ok"] = bool(ok)
    audit["combined_split_capacity_ok"] = bool(ok)
    audit["combined_stack_violation_reason"] = "; ".join(violations)
    audit["combined_socket_rule"] = "sum_fcr_plus_afrr_when_split_capacity_proven" if market_mode == "Combined" and ok else "single_or_max_shared_capacity"
    return audit


def _socket_limits_from_plan(plan: Dict[str, Any], market_mode: str) -> tuple[float, float]:
    fcr_bid = float(plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0) + plan.get("hvac_fcr_kw", 0.0))
    afrr_up = float(plan.get("bess_afrr_up_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0))
    afrr_down = float(
        plan.get("bess_afrr_down_kw", 0.0)
        + plan.get("ev_afrr_down_kw", 0.0)
        + plan.get("hvac_afrr_down_kw", 0.0)
        + plan.get("pv_afrr_down_kw", 0.0)
    )
    if market_mode == "FCR-N":
        up_limit = fcr_bid
        down_limit = fcr_bid
    elif market_mode == "aFRR":
        up_limit = afrr_up
        down_limit = afrr_down
    else:
        if _combined_split_capacity_enabled(plan):
            up_limit = fcr_bid + afrr_up
            down_limit = fcr_bid + afrr_down
        else:
            up_limit = max(fcr_bid, afrr_up)
            down_limit = max(fcr_bid, afrr_down)
    return float(max(up_limit, 0.0)), float(max(down_limit, 0.0))


def _upper_only_interval_summary(
    row: pd.Series,
    plan: Dict[str, float],
    state: Dict[str, float],
    fleet_meta: Dict[str, float],
    market_mode: str,
) -> Dict[str, float]:
    dt_h = float(row["dt_h"])
    eta_c = 0.95
    eta_d = 0.95
    fcr_up_frac = float(row["fcr_up_act_frac"]) if market_mode in {"FCR-N", "Combined"} else 0.0
    fcr_down_frac = float(row["fcr_down_act_frac"]) if market_mode in {"FCR-N", "Combined"} else 0.0
    afrr_up_frac = float(row["afrr_up_act_frac"]) if market_mode in {"aFRR", "Combined"} else 0.0
    afrr_down_frac = float(row["afrr_down_act_frac"]) if market_mode in {"aFRR", "Combined"} else 0.0

    bess_up = fcr_up_frac * float(plan.get("bess_fcr_kw", 0.0)) + afrr_up_frac * float(plan.get("bess_afrr_up_kw", 0.0))
    bess_down = fcr_down_frac * float(plan.get("bess_fcr_kw", 0.0)) + afrr_down_frac * float(plan.get("bess_afrr_down_kw", 0.0))
    ev_up = fcr_up_frac * float(plan.get("ev_fcr_kw", 0.0)) + afrr_up_frac * float(plan.get("ev_afrr_up_kw", 0.0))
    ev_down = fcr_down_frac * float(plan.get("ev_fcr_kw", 0.0)) + afrr_down_frac * float(plan.get("ev_afrr_down_kw", 0.0))
    hvac_up = fcr_up_frac * float(plan.get("hvac_fcr_kw", 0.0)) + afrr_up_frac * float(plan.get("hvac_afrr_up_kw", 0.0))
    hvac_down = fcr_down_frac * float(plan.get("hvac_fcr_kw", 0.0)) + afrr_down_frac * float(plan.get("hvac_afrr_down_kw", 0.0))
    pv_down = afrr_down_frac * float(plan.get("pv_afrr_down_kw", 0.0))
    requested_up = bess_up + ev_up + hvac_up
    requested_down = bess_down + ev_down + hvac_down + pv_down

    bess_cap = max(float(fleet_meta["bess_energy_cap_mwh"]), 1e-6)
    state["bess_soc_mwh"] = float(np.clip(state["bess_soc_mwh"] + dt_h / 1000.0 * (eta_c * bess_down - bess_up / eta_d), 0.10 * bess_cap, 0.95 * bess_cap))
    state["ev_soc_delta_mwh"] += dt_h / 1000.0 * (eta_c * ev_down - ev_up / eta_d)
    state["temp_delta_c"] = 0.90 * state["temp_delta_c"] + 0.16 * dt_h * ((hvac_down - hvac_up) / max(fleet_meta["n_hvac"], 1))

    return {
        "frequency_hz": float(row.get("frequency_hz", 50.0)),
        "requested_up_kw": float(requested_up),
        "requested_down_kw": float(requested_down),
        "delivered_up_kw": float(requested_up),
        "delivered_down_kw": float(requested_down),
        "bess_up_kw": float(bess_up),
        "bess_down_kw": float(bess_down),
        "ev_up_kw": float(ev_up),
        "ev_down_kw": float(ev_down),
        "hvac_fcr_up_kw": float(fcr_up_frac * float(plan.get("hvac_fcr_kw", 0.0))),
        "hvac_fcr_down_kw": float(fcr_down_frac * float(plan.get("hvac_fcr_kw", 0.0))),
        "hvac_up_kw": float(hvac_up),
        "hvac_down_kw": float(hvac_down),
        "pv_down_kw": float(pv_down),
        "bess_soc_mwh": float(state["bess_soc_mwh"]),
        "ev_soc_mwh": float(row["ev_soc_ref_mwh"] + state["ev_soc_delta_mwh"]),
        "indoor_temp_c": float(row["indoor_temp_c_ref"] + state["temp_delta_c"]),
        "optimized_net_load_kw": float(row["net_load_baseline_kw"] - requested_up + requested_down),
        "baseline_net_load_kw": float(row["net_load_baseline_kw"]),
        "tracking_samples": 0,
        "tracking_error_kw": 0.0,
        "shortfall_kw": 0.0,
        "inner_solver_status": "NotStarted",
    }

def build_inner_tracking_window(
    df: pd.DataFrame,
    preview_4s: pd.DataFrame | None,
    row_idx: int,
    dt_seconds: int = 4,
) -> pd.DataFrame:
    row = df.iloc[row_idx]
    next_idx = min(row_idx + 1, len(df) - 1)
    next_row = df.iloc[next_idx]
    interval_seconds = max(int(round(float(row["dt_h"]) * 3600.0)), dt_seconds)
    steps = max(interval_seconds // dt_seconds, 1)
    start_ts = pd.Timestamp(row.name)
    inner_index = pd.date_range(start_ts, periods=steps, freq=f"{dt_seconds}s")

    fine = pd.DataFrame(index=inner_index)
    preview_window = pd.DataFrame()
    if preview_4s is not None and not preview_4s.empty:
        preview_window = preview_4s.reindex(inner_index)

    current_fcr = float(row["fcr_up_act_frac"] - row["fcr_down_act_frac"])
    next_fcr = float(next_row["fcr_up_act_frac"] - next_row["fcr_down_act_frac"])
    current_afrr = float(row["afrr_up_act_frac"] - row["afrr_down_act_frac"])
    next_afrr = float(next_row["afrr_up_act_frac"] - next_row["afrr_down_act_frac"])

    if not preview_window.empty and preview_window[["fcr_signal_norm", "afrr_signal_norm", "frequency_hz"]].notna().all().all():
        fine["fcr_signal_norm"] = preview_window["fcr_signal_norm"].astype(float).clip(-1.0, 1.0)
        fine["afrr_signal_norm"] = preview_window["afrr_signal_norm"].astype(float).clip(-1.0, 1.0)
        fine["frequency_hz"] = preview_window["frequency_hz"].astype(float)
    else:
        fine["fcr_signal_norm"] = np.linspace(current_fcr, next_fcr, steps)
        fine["afrr_signal_norm"] = np.linspace(current_afrr, next_afrr, steps)
        fine["frequency_hz"] = 50.0 - 0.1 * fine["fcr_signal_norm"]

    fine["fcr_up_act_frac"] = np.clip(fine["fcr_signal_norm"], 0.0, 1.0)
    fine["fcr_down_act_frac"] = np.clip(-fine["fcr_signal_norm"], 0.0, 1.0)
    fine["afrr_up_act_frac"] = np.clip(fine["afrr_signal_norm"], 0.0, 1.0)
    fine["afrr_down_act_frac"] = np.clip(-fine["afrr_signal_norm"], 0.0, 1.0)
    return fine

def simulate_tracking_interval(
    row: pd.Series,
    fine_signals: pd.DataFrame,
    plan: Dict[str, float],
    state: Dict[str, float],
    fleet_meta: Dict[str, float],
    market_mode: str,
    dt_seconds: int = 4,
) -> tuple[Dict[str, float], pd.DataFrame]:
    dt_h = dt_seconds / 3600.0
    eta_c = 0.95
    eta_d = 0.95
    outer_dt_h = float(row["dt_h"])
    temp_alpha_sub = 0.90 ** (dt_h / max(outer_dt_h, 1e-6))
    temp_beta_sub = 0.16 * dt_h
    bess_tau_s = 2.0
    ev_tau_s = 4.0
    hvac_tau_s = float(fleet_meta.get("hvac_response_s", default_hvac_response_seconds(str(fleet_meta.get("hvac_mode", HVAC_MODES[0])))))
    alpha_bess = min(1.0, dt_seconds / max(bess_tau_s, 1e-6))
    alpha_ev = min(1.0, dt_seconds / max(ev_tau_s, 1e-6))
    alpha_hvac = min(1.0, dt_seconds / max(hvac_tau_s, 1e-6))

    dispatch_state = {
        "bess_dispatch_kw": float(state.get("bess_dispatch_kw", 0.0)),
        "ev_dispatch_kw": float(state.get("ev_dispatch_kw", 0.0)),
        "hvac_dispatch_kw": float(state.get("hvac_dispatch_kw", 0.0)),
    }
    inner_records: List[Dict[str, float]] = []

    for ts, signal in fine_signals.iterrows():
        avail = _available_up_down(row, state, fleet_meta, dt_h_override=dt_h)
        fcr_signal = float(signal["fcr_signal_norm"]) if market_mode in {"FCR-N", "Combined"} else 0.0
        afrr_signal = float(signal["afrr_signal_norm"]) if market_mode in {"aFRR", "Combined"} else 0.0

        req_components = {
            "bess_fcr": fcr_signal * float(plan.get("bess_fcr_kw", 0.0)),
            "ev_fcr": fcr_signal * float(plan.get("ev_fcr_kw", 0.0)),
            "hvac_fcr": fcr_signal * float(plan.get("hvac_fcr_kw", 0.0)),
            "bess_afrr": max(afrr_signal, 0.0) * float(plan.get("bess_afrr_up_kw", 0.0)) - max(-afrr_signal, 0.0) * float(plan.get("bess_afrr_down_kw", 0.0)),
            "ev_afrr": max(afrr_signal, 0.0) * float(plan.get("ev_afrr_up_kw", 0.0)) - max(-afrr_signal, 0.0) * float(plan.get("ev_afrr_down_kw", 0.0)),
            "hvac_afrr": max(afrr_signal, 0.0) * float(plan.get("hvac_afrr_up_kw", 0.0)) - max(-afrr_signal, 0.0) * float(plan.get("hvac_afrr_down_kw", 0.0)),
        }

        pv_dispatch_kw = -min(max(-afrr_signal, 0.0) * float(plan.get("pv_afrr_down_kw", 0.0)), avail["pv_down_av"])
        targets = {
            "bess_dispatch_kw": float(np.clip(req_components["bess_fcr"] + req_components["bess_afrr"], -avail["bess_down_av"], avail["bess_up_av"])),
            "ev_dispatch_kw": float(np.clip(req_components["ev_fcr"] + req_components["ev_afrr"], -avail["ev_down_av"], avail["ev_up_av"])),
            "hvac_dispatch_kw": float(np.clip(req_components["hvac_fcr"] + req_components["hvac_afrr"], -avail["hvac_down_av"], avail["hvac_up_av"])),
        }

        dispatch_state["bess_dispatch_kw"] = float(
            np.clip(
                dispatch_state["bess_dispatch_kw"] + alpha_bess * (targets["bess_dispatch_kw"] - dispatch_state["bess_dispatch_kw"]),
                -avail["bess_down_av"],
                avail["bess_up_av"],
            )
        )
        dispatch_state["ev_dispatch_kw"] = float(
            np.clip(
                dispatch_state["ev_dispatch_kw"] + alpha_ev * (targets["ev_dispatch_kw"] - dispatch_state["ev_dispatch_kw"]),
                -avail["ev_down_av"],
                avail["ev_up_av"],
            )
        )
        dispatch_state["hvac_dispatch_kw"] = float(
            np.clip(
                dispatch_state["hvac_dispatch_kw"] + alpha_hvac * (targets["hvac_dispatch_kw"] - dispatch_state["hvac_dispatch_kw"]),
                -avail["hvac_down_av"],
                avail["hvac_up_av"],
            )
        )

        actual_signed = {
            "bess": dispatch_state["bess_dispatch_kw"],
            "ev": dispatch_state["ev_dispatch_kw"],
            "hvac": dispatch_state["hvac_dispatch_kw"],
        }

        def _split_actual(actual_kw: float, fcr_req_kw: float, afrr_req_kw: float) -> tuple[float, float]:
            total_abs = abs(fcr_req_kw) + abs(afrr_req_kw)
            if total_abs <= 1e-9:
                return 0.0, 0.0
            return actual_kw * abs(fcr_req_kw) / total_abs, actual_kw * abs(afrr_req_kw) / total_abs

        bess_fcr_actual_signed, bess_afrr_actual_signed = _split_actual(actual_signed["bess"], req_components["bess_fcr"], req_components["bess_afrr"])
        ev_fcr_actual_signed, ev_afrr_actual_signed = _split_actual(actual_signed["ev"], req_components["ev_fcr"], req_components["ev_afrr"])
        hvac_fcr_actual_signed, hvac_afrr_actual_signed = _split_actual(actual_signed["hvac"], req_components["hvac_fcr"], req_components["hvac_afrr"])

        bess_up = max(actual_signed["bess"], 0.0)
        bess_down = max(-actual_signed["bess"], 0.0)
        ev_up = max(actual_signed["ev"], 0.0)
        ev_down = max(-actual_signed["ev"], 0.0)
        hvac_up = max(actual_signed["hvac"], 0.0)
        hvac_down = max(-actual_signed["hvac"], 0.0)
        pv_down = max(-pv_dispatch_kw, 0.0)

        requested_signed_total = sum(req_components.values()) + pv_dispatch_kw
        delivered_signed_total = actual_signed["bess"] + actual_signed["ev"] + actual_signed["hvac"] + pv_dispatch_kw

        state["bess_soc_mwh"] += dt_h / 1000.0 * (eta_c * bess_down - bess_up / eta_d)
        state["ev_soc_delta_mwh"] += dt_h / 1000.0 * (eta_c * ev_down - ev_up / eta_d)
        state["temp_delta_c"] = temp_alpha_sub * state["temp_delta_c"] + temp_beta_sub * ((hvac_down - hvac_up) / max(fleet_meta["n_hvac"], 1))
        state["bess_dispatch_kw"] = dispatch_state["bess_dispatch_kw"]
        state["ev_dispatch_kw"] = dispatch_state["ev_dispatch_kw"]
        state["hvac_dispatch_kw"] = dispatch_state["hvac_dispatch_kw"]

        inner_records.append(
            {
                "timestamp": ts,
                "frequency_hz": float(signal["frequency_hz"]),
                "fcr_signal_norm": fcr_signal,
                "afrr_signal_norm": afrr_signal,
                "requested_up_kw": max(requested_signed_total, 0.0),
                "requested_down_kw": max(-requested_signed_total, 0.0),
                "delivered_up_kw": max(delivered_signed_total, 0.0),
                "delivered_down_kw": max(-delivered_signed_total, 0.0),
                "bess_up_kw": bess_up,
                "bess_down_kw": bess_down,
                "ev_up_kw": ev_up,
                "ev_down_kw": ev_down,
                "hvac_up_kw": hvac_up,
                "hvac_down_kw": hvac_down,
                "pv_down_kw": pv_down,
                "bess_fcr_up_kw": max(bess_fcr_actual_signed, 0.0),
                "bess_fcr_down_kw": max(-bess_fcr_actual_signed, 0.0),
                "ev_fcr_up_kw": max(ev_fcr_actual_signed, 0.0),
                "ev_fcr_down_kw": max(-ev_fcr_actual_signed, 0.0),
                "hvac_fcr_up_kw": max(hvac_fcr_actual_signed, 0.0),
                "hvac_fcr_down_kw": max(-hvac_fcr_actual_signed, 0.0),
                "bess_afrr_up_kw": max(bess_afrr_actual_signed, 0.0),
                "bess_afrr_down_kw": max(-bess_afrr_actual_signed, 0.0),
                "ev_afrr_up_kw": max(ev_afrr_actual_signed, 0.0),
                "ev_afrr_down_kw": max(-ev_afrr_actual_signed, 0.0),
                "hvac_afrr_up_kw": max(hvac_afrr_actual_signed, 0.0),
                "hvac_afrr_down_kw": max(-hvac_afrr_actual_signed, 0.0),
                "pv_afrr_down_kw": pv_down,
                "bess_soc_mwh": state["bess_soc_mwh"],
                "ev_soc_mwh": row["ev_soc_ref_mwh"] + state["ev_soc_delta_mwh"],
                "indoor_temp_c": row["indoor_temp_c_ref"] + state["temp_delta_c"],
                "optimized_net_load_kw": row["net_load_baseline_kw"] - max(delivered_signed_total, 0.0) + max(-delivered_signed_total, 0.0),
                "baseline_net_load_kw": row["net_load_baseline_kw"],
            }
        )

    tracking_df = pd.DataFrame(inner_records).set_index("timestamp")
    aggregated = {
        "frequency_hz": float(tracking_df["frequency_hz"].mean()),
        "requested_up_kw": float(tracking_df["requested_up_kw"].mean()),
        "requested_down_kw": float(tracking_df["requested_down_kw"].mean()),
        "delivered_up_kw": float(tracking_df["delivered_up_kw"].mean()),
        "delivered_down_kw": float(tracking_df["delivered_down_kw"].mean()),
        "bess_up_kw": float(tracking_df["bess_up_kw"].mean()),
        "bess_down_kw": float(tracking_df["bess_down_kw"].mean()),
        "ev_up_kw": float(tracking_df["ev_up_kw"].mean()),
        "ev_down_kw": float(tracking_df["ev_down_kw"].mean()),
        "hvac_up_kw": float(tracking_df["hvac_up_kw"].mean()),
        "hvac_down_kw": float(tracking_df["hvac_down_kw"].mean()),
        "pv_down_kw": float(tracking_df["pv_down_kw"].mean()),
        "hvac_fcr_up_kw": float(tracking_df["hvac_fcr_up_kw"].mean()),
        "hvac_fcr_down_kw": float(tracking_df["hvac_fcr_down_kw"].mean()),
        "bess_fcr_up_kw": float(tracking_df["bess_fcr_up_kw"].mean()),
        "bess_fcr_down_kw": float(tracking_df["bess_fcr_down_kw"].mean()),
        "ev_fcr_up_kw": float(tracking_df["ev_fcr_up_kw"].mean()),
        "ev_fcr_down_kw": float(tracking_df["ev_fcr_down_kw"].mean()),
        "bess_soc_mwh": float(state["bess_soc_mwh"]),
        "ev_soc_mwh": float(tracking_df["ev_soc_mwh"].iloc[-1]),
        "indoor_temp_c": float(tracking_df["indoor_temp_c"].iloc[-1]),
        "optimized_net_load_kw": float(tracking_df["optimized_net_load_kw"].mean()),
        "baseline_net_load_kw": float(tracking_df["baseline_net_load_kw"].mean()),
        "tracking_samples": len(tracking_df),
    }
    return aggregated, tracking_df

def _solve_mpc_with_plugin(
    forecast: pd.DataFrame,
    state: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    penalty_weights: Dict[str, float],
    fleet_meta: Dict[str, float],
    optimizer: Any | None,
) -> Dict[str, object]:
    if optimizer is None:
        return solve_mpc_step(forecast, state, market_mode, resource_mode, penalty_weights, fleet_meta)

    result = optimizer.solve(
        forecast_horizon=forecast,
        current_state=state,
        fleet_metadata=fleet_meta,
        market_mode=market_mode,
        resource_mode=resource_mode,
        penalty_weights=penalty_weights,
    )
    if hasattr(result, "to_legacy_dict"):
        return result.to_legacy_dict()
    return dict(result)

def run_mpc_controller(
    df: pd.DataFrame,
    models: Dict[str, Any],
    fleet_meta: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    horizon_hours: float,
    dispatch_hours: float,
    penalty_weights: Dict[str, float],
    preview_4s: pd.DataFrame | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    optimizer: Any | None = None,
    device_roster: pd.DataFrame | None = None,
    inner_controller_mode: str = "mpc",
    inner_dt_seconds: int = 4,
    inner_mpc_horizon_seconds: int = 4,
    rotation_strategy: str = "usage_aware",
    gateway_mode: str = "simulated_centralized",
    execute_lower_mpc: bool = True,
) -> Dict[str, object]:
    penalty_weights = apply_risk_policy_defaults(penalty_weights)
    fleet_meta = dict(fleet_meta)
    fleet_meta.setdefault("fcr_hvac_share_cap", FCR_N_HVAC_SHARE_CAP)
    sim_steps = min(max(int(round(float(dispatch_hours) / float(df["dt_h"].iloc[0]))), 1), len(df) - 2)
    horizon_steps = min(max(int(round(float(horizon_hours) / float(df["dt_h"].iloc[0]))), 1), len(df) - 2)
    state = {
        "bess_soc_mwh": float(df["bess_soc_ref_mwh"].iloc[0]),
        "ev_soc_delta_mwh": 0.0,
        "temp_delta_c": 0.0,
        "bess_dispatch_kw": 0.0,
        "ev_dispatch_kw": 0.0,
        "hvac_dispatch_kw": 0.0,
    }
    history: List[Dict[str, float]] = []
    schedule_snapshots: List[pd.DataFrame] = []
    tracking_history: List[pd.DataFrame] = []
    upper_device_schedules: List[pd.DataFrame] = []
    gateway_command_summaries: List[pd.DataFrame] = []
    work_per_interval = horizon_steps + (2 if execute_lower_mpc else 1)
    total_work = max(sim_steps * work_per_interval, 1)
    controller_config = InnerControllerConfig(
        mode=inner_controller_mode,
        dt_seconds=int(inner_dt_seconds),
        horizon_seconds=int(inner_mpc_horizon_seconds),
        rotation_strategy=rotation_strategy,
        gateway_mode=gateway_mode,
    )
    centralized_controller = CentralizedVPPController(
        device_roster=device_roster if device_roster is not None else representative_roster_from_frame(df, fleet_meta),
        fleet_meta=fleet_meta,
        config=controller_config,
    )

    for t in range(sim_steps):
        base_slice = df.iloc[t + 1 : t + 1 + horizon_steps].copy()
        base_progress = t * work_per_interval

        def _forecast_step_callback(step_idx: int, step_total: int) -> None:
            if progress_callback:
                progress_callback(
                    base_progress + step_idx,
                    total_work,
                    f"Forecasting horizon {step_idx}/{step_total} for MPC interval {t + 1}/{sim_steps}",
                )

        preds = iterative_forecast(
            df,
            t,
            horizon_steps,
            models,
            list(dict.fromkeys(SUPPORTED_MPC_TARGETS)),
            step_callback=_forecast_step_callback,
        )
        if progress_callback:
            progress_callback(
                base_progress + horizon_steps + 1,
                total_work,
                f"Solving MPC interval {t + 1}/{sim_steps}",
            )
        for col in preds.columns:
            base_slice[col] = preds[col].values
        forecast = base_slice[
            [
                "dt_h",
                "bess_up_kw",
                "bess_down_kw",
                "ev_up_kw",
                "ev_down_kw",
                "hvac_up_kw",
                "hvac_down_kw",
                "hvac_fast_kw",
                "pv_down_kw",
                "ev_soc_ref_mwh",
                "ev_soc_min_mwh",
                "ev_soc_max_mwh",
                "indoor_temp_c_ref",
                "fcr_up_act_frac",
                "fcr_down_act_frac",
                "afrr_up_act_frac",
                "afrr_down_act_frac",
                "fcrn_capacity_eur_per_mw_h",
                "afrr_up_capacity_eur_per_mw_h",
                "afrr_down_capacity_eur_per_mw_h",
                "afrr_up_energy_eur_per_mwh",
                "afrr_down_energy_eur_per_mwh",
            ]
        ].copy()
        price_quantile_cols = [
            col
            for col in base_slice.columns
            if any(col.startswith(f"{target}_forecast_p") for target in PRICE_FORECAST_TARGETS)
        ]
        if price_quantile_cols:
            forecast = forecast.join(base_slice[price_quantile_cols])
        solution = _solve_mpc_with_plugin(
            forecast=forecast,
            state=state,
            market_mode=market_mode,
            resource_mode=resource_mode,
            penalty_weights=penalty_weights,
            fleet_meta=fleet_meta,
            optimizer=optimizer,
        )
        if solution["schedule"].empty:
            break
        schedule_snapshots.append(solution["schedule"])
        plan = solution["controls"]
        execution_row_idx = min(t + 1, len(df) - 1)
        row = df.iloc[execution_row_idx]
        dt_h = row["dt_h"]
        plan.update(_combined_stack_audit(plan, row, market_mode))
        if execute_lower_mpc:
            fine_signals = build_inner_tracking_window(df, preview_4s, execution_row_idx, dt_seconds=int(inner_dt_seconds))
            if progress_callback:
                progress_callback(
                    base_progress + horizon_steps + 2,
                    total_work,
                    f"Solving centralized 4-second MPC for interval {t + 1}/{sim_steps}",
                )
            interval_summary, interval_tracking, upper_schedule, command_summary = centralized_controller.execute_interval(
                row=row,
                fine_signals=fine_signals,
                plan=plan,
                interval_index=t,
                market_mode=market_mode,
                resource_mode=resource_mode,
            )
            tracking_history.append(interval_tracking)
            if not command_summary.empty:
                gateway_command_summaries.append(command_summary)
        else:
            if progress_callback:
                progress_callback(
                    base_progress + horizon_steps + 1,
                    total_work,
                    f"Evaluating market gate for interval {t + 1}/{sim_steps}",
                )
            upper_schedule = centralized_controller.select_upper_schedule(row, plan, t, resource_mode)
            interval_summary = _upper_only_interval_summary(
                row=row,
                plan=plan,
                state=state,
                fleet_meta=fleet_meta,
                market_mode=market_mode,
            )
            interval_tracking = pd.DataFrame()
            command_summary = pd.DataFrame()
        if not upper_schedule.empty:
            upper_device_schedules.append(upper_schedule)
        state["bess_soc_mwh"] = interval_summary["bess_soc_mwh"]
        state["ev_soc_delta_mwh"] = interval_summary["ev_soc_mwh"] - float(row["ev_soc_ref_mwh"])
        state["temp_delta_c"] = interval_summary["indoor_temp_c"] - float(row["indoor_temp_c_ref"])

        actual_bess_up = interval_summary["bess_up_kw"]
        actual_bess_down = interval_summary["bess_down_kw"]
        actual_ev_up = interval_summary["ev_up_kw"]
        actual_ev_down = interval_summary["ev_down_kw"]
        actual_hvac_fcr_up = interval_summary["hvac_fcr_up_kw"]
        actual_hvac_fcr_down = interval_summary["hvac_fcr_down_kw"]
        actual_hvac_up = interval_summary["hvac_up_kw"]
        actual_hvac_down = interval_summary["hvac_down_kw"]
        actual_pv_down = interval_summary["pv_down_kw"]
        requested_up = interval_summary["requested_up_kw"]
        requested_down = interval_summary["requested_down_kw"]
        delivered_up = interval_summary["delivered_up_kw"]
        delivered_down = interval_summary["delivered_down_kw"]
        fcr_up_frac_actual = float(row["fcr_up_act_frac"]) if market_mode in {"FCR-N", "Combined"} else 0.0
        fcr_down_frac_actual = float(row["fcr_down_act_frac"]) if market_mode in {"FCR-N", "Combined"} else 0.0
        actual_bess_afrr_up = max(actual_bess_up - fcr_up_frac_actual * float(plan.get("bess_fcr_kw", 0.0)), 0.0)
        actual_bess_afrr_down = max(actual_bess_down - fcr_down_frac_actual * float(plan.get("bess_fcr_kw", 0.0)), 0.0)
        actual_ev_afrr_up = max(actual_ev_up - fcr_up_frac_actual * float(plan.get("ev_fcr_kw", 0.0)), 0.0)
        actual_ev_afrr_down = max(actual_ev_down - fcr_down_frac_actual * float(plan.get("ev_fcr_kw", 0.0)), 0.0)
        actual_hvac_afrr_up = max(actual_hvac_up - actual_hvac_fcr_up, 0.0)
        actual_hvac_afrr_down = max(actual_hvac_down - actual_hvac_fcr_down, 0.0)

        fcr_bid_kw = plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0) + plan.get("hvac_fcr_kw", 0.0)
        afrr_up_bid_kw = plan.get("bess_afrr_up_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0)
        afrr_down_bid_kw = (
            plan.get("bess_afrr_down_kw", 0.0)
            + plan.get("ev_afrr_down_kw", 0.0)
            + plan.get("hvac_afrr_down_kw", 0.0)
            + plan.get("pv_afrr_down_kw", 0.0)
        )
        stack_audit = _combined_stack_audit(plan, row, market_mode)
        plan.update(stack_audit)
        socket_up_kw, socket_down_kw = _socket_limits_from_plan(plan, market_mode)
        plan["socket_up_kw"] = socket_up_kw
        plan["socket_down_kw"] = socket_down_kw
        bid_block_start, bid_block_end = _bid_block_bounds(pd.Timestamp(row.name), market_mode)
        stacking_ok = (
            True
            if market_mode != "Combined"
            else bool(stack_audit["combined_shared_capacity_ok"])
            and socket_up_kw + 1e-6 >= fcr_bid_kw + afrr_up_bid_kw
            and socket_down_kw + 1e-6 >= fcr_bid_kw + afrr_down_bid_kw
        )
        capacity_revenue = dt_h / 1000.0 * (
            row["fcrn_capacity_eur_per_mw_h"] * fcr_bid_kw
            + row["afrr_up_capacity_eur_per_mw_h"] * afrr_up_bid_kw
            + row["afrr_down_capacity_eur_per_mw_h"] * afrr_down_bid_kw
        )
        activation_revenue = dt_h / 1000.0 * (
            row["afrr_up_energy_eur_per_mwh"] * (actual_bess_afrr_up + actual_ev_afrr_up + actual_hvac_afrr_up)
            + row["afrr_down_energy_eur_per_mwh"] * (actual_bess_afrr_down + actual_ev_afrr_down + actual_hvac_afrr_down + actual_pv_down)
        )
        degradation_cost = penalty_weights["degradation"] * dt_h / 1000.0 * (
            actual_bess_up + actual_bess_down + 0.4 * (actual_ev_up + actual_ev_down)
        )
        comfort_penalty = penalty_weights["comfort"] * dt_h * max(
            0.0,
            abs(row["indoor_temp_c_ref"] + state["temp_delta_c"] - fleet_meta["setpoint_c"]) - fleet_meta["comfort_band_c"],
        )
        gate = _market_gate_decision(plan, market_mode)
        granularity_ok = bool(gate.get("fcr_bid_granularity_ok", True) and gate.get("afrr_bid_granularity_ok", True))
        net_revenue = (
            capacity_revenue
            + activation_revenue
            - degradation_cost
            - comfort_penalty
            - float(plan.get("asset_fatigue_cost_eur", 0.0))
        )

        history.append(
            {
                "timestamp": row.name,
                "bid_block_start": bid_block_start,
                "bid_block_end": bid_block_end,
                "simulation_only_notice": SIMULATION_ONLY_NOTICE,
                "baseline_net_load_kw": interval_summary["baseline_net_load_kw"],
                "optimized_net_load_kw": interval_summary["optimized_net_load_kw"],
                "frequency_hz": interval_summary["frequency_hz"],
                "solver_status": str(solution.get("status", "unknown")),
                "inner_solver_status": str(interval_summary.get("inner_solver_status", "unknown")),
                "inner_controller_mode": inner_controller_mode,
                "fcr_bid_kw": fcr_bid_kw,
                "afrr_up_bid_kw": afrr_up_bid_kw,
                "afrr_down_bid_kw": afrr_down_bid_kw,
                "fcr_bid_mw": fcr_bid_kw / 1000.0,
                "afrr_up_bid_mw": afrr_up_bid_kw / 1000.0,
                "afrr_down_bid_mw": afrr_down_bid_kw / 1000.0,
                "fcr_bid_segments": int(plan.get("fcr_bid_segments", 0) or 0),
                "afrr_up_segments": int(plan.get("afrr_up_segments", 0) or 0),
                "afrr_down_segments": int(plan.get("afrr_down_segments", 0) or 0),
                "stacking_ok": bool(stacking_ok),
                **stack_audit,
                "energy_endurance_ok": bool(plan.get("energy_endurance_ok", True)),
                "granularity_ok": bool(granularity_ok),
                "socket_up_kw": socket_up_kw,
                "socket_down_kw": socket_down_kw,
                "bess_fcr_kw": float(plan.get("bess_fcr_kw", 0.0)),
                "ev_fcr_kw": float(plan.get("ev_fcr_kw", 0.0)),
                "hvac_fcr_kw": float(plan.get("hvac_fcr_kw", 0.0)),
                "bess_afrr_up_kw": float(plan.get("bess_afrr_up_kw", 0.0)),
                "bess_afrr_down_kw": float(plan.get("bess_afrr_down_kw", 0.0)),
                "ev_afrr_up_kw": float(plan.get("ev_afrr_up_kw", 0.0)),
                "ev_afrr_down_kw": float(plan.get("ev_afrr_down_kw", 0.0)),
                "hvac_afrr_up_kw": float(plan.get("hvac_afrr_up_kw", 0.0)),
                "hvac_afrr_down_kw": float(plan.get("hvac_afrr_down_kw", 0.0)),
                "pv_afrr_down_kw": float(plan.get("pv_afrr_down_kw", 0.0)),
                "fcrn_capacity_eur_per_mw_h": float(row["fcrn_capacity_eur_per_mw_h"]),
                "afrr_up_capacity_eur_per_mw_h": float(row["afrr_up_capacity_eur_per_mw_h"]),
                "afrr_down_capacity_eur_per_mw_h": float(row["afrr_down_capacity_eur_per_mw_h"]),
                "price_forecast_mode": str(plan.get("price_forecast_mode", "point")),
                "price_bid_quantile": float(plan.get("price_bid_quantile", 0.50)),
                "fcrn_capacity_bid_eur_per_mw_h": float(plan.get("fcrn_capacity_bid_eur_per_mw_h", row["fcrn_capacity_eur_per_mw_h"])),
                "afrr_up_capacity_bid_eur_per_mw_h": float(plan.get("afrr_up_capacity_bid_eur_per_mw_h", row["afrr_up_capacity_eur_per_mw_h"])),
                "afrr_down_capacity_bid_eur_per_mw_h": float(plan.get("afrr_down_capacity_bid_eur_per_mw_h", row["afrr_down_capacity_eur_per_mw_h"])),
                "afrr_up_energy_eur_per_mwh": float(row["afrr_up_energy_eur_per_mwh"]),
                "afrr_down_energy_eur_per_mwh": float(row["afrr_down_energy_eur_per_mwh"]),
                "requested_up_kw": requested_up,
                "requested_down_kw": requested_down,
                "delivered_up_kw": delivered_up,
                "delivered_down_kw": delivered_down,
                "bess_up_kw": actual_bess_up,
                "bess_down_kw": actual_bess_down,
                "ev_up_kw": actual_ev_up,
                "ev_down_kw": actual_ev_down,
                "hvac_fcr_up_kw": actual_hvac_fcr_up,
                "hvac_fcr_down_kw": actual_hvac_fcr_down,
                "hvac_up_kw": actual_hvac_up,
                "hvac_down_kw": actual_hvac_down,
                "bess_afrr_up_delivered_kw": actual_bess_afrr_up,
                "bess_afrr_down_delivered_kw": actual_bess_afrr_down,
                "ev_afrr_up_delivered_kw": actual_ev_afrr_up,
                "ev_afrr_down_delivered_kw": actual_ev_afrr_down,
                "hvac_afrr_up_delivered_kw": actual_hvac_afrr_up,
                "hvac_afrr_down_delivered_kw": actual_hvac_afrr_down,
                "pv_down_kw": actual_pv_down,
                "capacity_revenue_eur": capacity_revenue,
                "activation_revenue_eur": activation_revenue,
                "degradation_cost_eur": degradation_cost,
                "comfort_penalty_eur": comfort_penalty,
                "non_delivery_risk_cost_eur": float(plan.get("non_delivery_risk_cost_eur", 0.0)),
                "activation_uncertainty_cost_eur": float(plan.get("activation_uncertainty_cost_eur", 0.0)),
                "asset_fatigue_cost_eur": float(plan.get("asset_fatigue_cost_eur", 0.0)),
                "reserve_buffer_kw": float(plan.get("reserve_buffer_kw", 0.0)),
                "risk_policy": str(plan.get("risk_policy", penalty_weights.get("risk_policy", "investor_balanced"))),
                "risk_quantile": float(plan.get("risk_quantile", penalty_weights.get("risk_quantile", 0.80))),
                "reserve_buffer_pct": float(plan.get("reserve_buffer_pct", penalty_weights.get("reserve_buffer_pct", 0.08))),
                "raw_fcr_symmetric_kw": float(plan.get("raw_fcr_symmetric_kw", 0.0)),
                "fcr_dynamic_cap_total_kw": float(plan.get("fcr_dynamic_cap_total_kw", plan.get("hvac_fcr_dynamic_cap_kw", 0.0))),
                "fcr_energy_cap_total_kw": float(plan.get("fcr_energy_cap_total_kw", 0.0)),
                "fcr_granular_bid_kw": float(plan.get("fcr_bid_kw", fcr_bid_kw)),
                "hvac_fcr_dynamic_cap_kw": float(plan.get("hvac_fcr_dynamic_cap_kw", 0.0)),
                "hvac_fcr_dynamic_fraction": float(plan.get("hvac_fcr_dynamic_fraction", 0.0)),
                "fcr_control_source": str(plan.get("fcr_control_source", FINGRID_RULES["FCR-N"]["control_source"])),
                "fcr_symmetric": bool(plan.get("fcr_symmetric", True)),
                "fcr_response_60_fraction": float(plan.get("fcr_response_60_fraction", 1.0)),
                "fcr_response_180_fraction": float(plan.get("fcr_response_180_fraction", 1.0)),
                "fcr_energy_60_seconds": float(plan.get("fcr_energy_60_seconds", 0.0)),
                "fcr_dynamic_response_ok": bool(plan.get("fcr_dynamic_response_ok", True)),
                "fcr_dynamic_cap_ok": gate["fcr_dynamic_cap_ok"],
                "fcr_hvac_share_ok": gate["fcr_hvac_share_ok"],
                "risk_adjusted_profit_eur": gate["risk_adjusted_profit_eur"],
                "market_gate_status": gate["market_gate_status"],
                "market_gate_reason": gate["market_gate_reason"],
                "net_revenue_eur": net_revenue,
                "bess_soc_mwh": interval_summary["bess_soc_mwh"],
                "ev_soc_mwh": interval_summary["ev_soc_mwh"],
                "indoor_temp_c": interval_summary["indoor_temp_c"],
                "fcr_accuracy_ratio": delivered_up / requested_up if requested_up > 1 else np.nan,
                "down_accuracy_ratio": delivered_down / requested_down if requested_down > 1 else np.nan,
                "inner_tracking_samples": interval_summary["tracking_samples"],
                "tracking_error_kw": interval_summary.get("tracking_error_kw", 0.0),
                "mean_signed_error_kw": interval_summary.get("mean_signed_error_kw", interval_summary.get("tracking_error_kw", 0.0)),
                "max_abs_error_per_tick_kw": interval_summary.get("max_abs_error_per_tick_kw", abs(interval_summary.get("tracking_error_kw", 0.0))),
                "socket_violation_up_kw": interval_summary.get("socket_violation_up_kw", 0.0),
                "socket_violation_down_kw": interval_summary.get("socket_violation_down_kw", 0.0),
                "shortfall_kw": interval_summary.get("shortfall_kw", 0.0),
            }
        )

    result_df = pd.DataFrame(history).set_index("timestamp")
    if result_df.empty:
        return {
            "history": result_df,
            "summary": {},
            "compliance": {},
            "first_schedule": pd.DataFrame(),
            "tracking_4s": pd.DataFrame(),
            "inner_mpc_trace": pd.DataFrame(),
            "upper_device_schedule": pd.DataFrame(),
            "gateway_commands": pd.DataFrame(),
            "household_contributions": pd.DataFrame(),
            "appliance_contributions": pd.DataFrame(),
            "usage_fatigue_summary": pd.DataFrame(),
            "afrr_energy_audit": pd.DataFrame(),
        }
    if progress_callback:
        progress_callback(total_work, total_work, "Simulation finished")

    tracking_df = pd.concat(tracking_history) if tracking_history else pd.DataFrame()
    tracking_df = _enrich_tracking_for_market_audit(tracking_df)
    technical_compliance, technical_summary, afrr_energy_audit = _market_technical_audit(
        result_df=result_df,
        tracking_df=tracking_df,
        fleet_meta=fleet_meta,
        market_mode=market_mode,
        inner_dt_seconds=inner_dt_seconds,
    )

    dt_h_result = float(df["dt_h"].iloc[0])
    resource_capacity_revenue = {
        "BESS": float(
            (
                result_df["fcrn_capacity_eur_per_mw_h"] * result_df["bess_fcr_kw"]
                + result_df["afrr_up_capacity_eur_per_mw_h"] * result_df["bess_afrr_up_kw"]
                + result_df["afrr_down_capacity_eur_per_mw_h"] * result_df["bess_afrr_down_kw"]
            ).sum()
            * dt_h_result
            / 1000.0
        ),
        "EV": float(
            (
                result_df["fcrn_capacity_eur_per_mw_h"] * result_df["ev_fcr_kw"]
                + result_df["afrr_up_capacity_eur_per_mw_h"] * result_df["ev_afrr_up_kw"]
                + result_df["afrr_down_capacity_eur_per_mw_h"] * result_df["ev_afrr_down_kw"]
            ).sum()
            * dt_h_result
            / 1000.0
        ),
        "HVAC": float(
            (
                result_df["fcrn_capacity_eur_per_mw_h"] * result_df["hvac_fcr_kw"]
                + result_df["afrr_up_capacity_eur_per_mw_h"] * result_df["hvac_afrr_up_kw"]
                + result_df["afrr_down_capacity_eur_per_mw_h"] * result_df["hvac_afrr_down_kw"]
            ).sum()
            * dt_h_result
            / 1000.0
        ),
        "PV": float((result_df["afrr_down_capacity_eur_per_mw_h"] * result_df["pv_afrr_down_kw"]).sum() * dt_h_result / 1000.0),
    }
    resource_activation_revenue = {
        "BESS": float(
            (
                result_df["afrr_up_energy_eur_per_mwh"] * result_df.get("bess_afrr_up_delivered_kw", result_df["bess_up_kw"])
                + result_df["afrr_down_energy_eur_per_mwh"] * result_df.get("bess_afrr_down_delivered_kw", result_df["bess_down_kw"])
            ).sum()
            * dt_h_result
            / 1000.0
        ),
        "EV": float(
            (
                result_df["afrr_up_energy_eur_per_mwh"] * result_df.get("ev_afrr_up_delivered_kw", result_df["ev_up_kw"])
                + result_df["afrr_down_energy_eur_per_mwh"] * result_df.get("ev_afrr_down_delivered_kw", result_df["ev_down_kw"])
            ).sum()
            * dt_h_result
            / 1000.0
        ),
        "HVAC": float(
            (
                result_df["afrr_up_energy_eur_per_mwh"] * result_df.get("hvac_afrr_up_delivered_kw", (result_df["hvac_up_kw"] - result_df["hvac_fcr_up_kw"]).clip(lower=0.0))
                + result_df["afrr_down_energy_eur_per_mwh"] * result_df.get("hvac_afrr_down_delivered_kw", (result_df["hvac_down_kw"] - result_df["hvac_fcr_down_kw"]).clip(lower=0.0))
            ).sum()
            * dt_h_result
            / 1000.0
        ),
        "PV": float((result_df["afrr_down_energy_eur_per_mwh"] * result_df["pv_down_kw"]).sum() * dt_h_result / 1000.0),
    }
    resource_revenue = {
        resource: float(resource_capacity_revenue.get(resource, 0.0) + resource_activation_revenue.get(resource, 0.0))
        for resource in ["BESS", "EV", "HVAC", "PV"]
    }
    resource_energy_kwh = {
        "BESS": float((result_df["bess_up_kw"] + result_df["bess_down_kw"]).sum() * dt_h_result),
        "EV": float((result_df["ev_up_kw"] + result_df["ev_down_kw"]).sum() * dt_h_result),
        "HVAC": float((result_df["hvac_up_kw"] + result_df["hvac_down_kw"]).sum() * dt_h_result),
        "PV": float(result_df["pv_down_kw"].sum() * dt_h_result),
    }
    accuracy_ratios = pd.concat([result_df["fcr_accuracy_ratio"].dropna(), result_df["down_accuracy_ratio"].dropna()])
    mean_accuracy = float(accuracy_ratios.mean()) if not accuracy_ratios.empty else np.nan

    avg_fcr_bid = float(result_df["fcr_bid_kw"].mean())
    avg_afrr_bid = float(np.maximum(result_df["afrr_up_bid_kw"], result_df["afrr_down_bid_kw"]).mean())
    hvac_response_s = float(fleet_meta.get("hvac_response_s", default_hvac_response_seconds(str(fleet_meta.get("hvac_mode", HVAC_MODES[0])))))
    rules_fcr = FINGRID_RULES["FCR-N"]
    fcr_response_60_fraction = float(result_df.get("fcr_response_60_fraction", pd.Series([1.0] * len(result_df), index=result_df.index)).min())
    fcr_response_180_fraction = float(result_df.get("fcr_response_180_fraction", pd.Series([1.0] * len(result_df), index=result_df.index)).min())
    fcr_energy_60_seconds = float(result_df.get("fcr_energy_60_seconds", pd.Series([0.0] * len(result_df), index=result_df.index)).replace([np.inf, -np.inf], np.nan).min())
    if avg_fcr_bid <= 1.0:
        fcr_response_60_fraction = 1.0
        fcr_response_180_fraction = 1.0
        fcr_energy_60_seconds = 0.0
    fcr_response_ok = (
        fcr_response_60_fraction + 1e-9 >= float(rules_fcr["activation_60_fraction"])
        and fcr_response_180_fraction + 1e-9 >= float(rules_fcr["activation_180_fraction"])
        and (fcr_energy_60_seconds + 1e-9 >= float(rules_fcr["energy_60_seconds"]) if avg_fcr_bid > 1.0 else True)
    )
    fcr_response_63_s = float(rules_fcr["response_63_seconds"]) if fcr_response_ok and avg_fcr_bid > 1.0 else 0.0
    fcr_response_95_s = float(rules_fcr["response_95_seconds"]) if fcr_response_ok and avg_fcr_bid > 1.0 else 0.0
    fcr_local_control_ok = bool(
        (
            result_df.get(
                "fcr_control_source",
                pd.Series([rules_fcr["control_source"]] * len(result_df), index=result_df.index),
            )
            == rules_fcr["control_source"]
        ).all()
    )
    fcr_dynamic_cap_ok = bool(result_df.get("fcr_dynamic_cap_ok", pd.Series([True] * len(result_df), index=result_df.index)).all())
    fcr_dynamic_response_ok = bool(result_df.get("fcr_dynamic_response_ok", pd.Series([True] * len(result_df), index=result_df.index)).all())
    fcr_hvac_share_ok = bool(result_df.get("fcr_hvac_share_ok", pd.Series([True] * len(result_df), index=result_df.index)).all())
    fcr_granularity_kw = float(FINGRID_RULES["FCR-N"]["bid_granularity_kw"])
    fcr_bid_granularity_ok = bool(
        all(
            abs((bid / max(fcr_granularity_kw, 1e-9)) - round(bid / max(fcr_granularity_kw, 1e-9))) <= 1e-6
            for bid in result_df["fcr_bid_kw"].dropna()
            if abs(float(bid)) > 1e-6
        )
    )
    afrr_response_s = (
        2.0 * (result_df["bess_up_kw"].sum() + result_df["bess_down_kw"].sum())
        + 4.0 * (result_df["ev_up_kw"].sum() + result_df["ev_down_kw"].sum())
        + hvac_response_s * ((result_df["hvac_up_kw"] - result_df["hvac_fcr_up_kw"]).clip(lower=0).sum() + (result_df["hvac_down_kw"] - result_df["hvac_fcr_down_kw"]).clip(lower=0).sum())
        + 5.0 * result_df["pv_down_kw"].sum()
    ) / max(
        (
            result_df["bess_up_kw"].sum()
            + result_df["bess_down_kw"].sum()
            + result_df["ev_up_kw"].sum()
            + result_df["ev_down_kw"].sum()
            + (result_df["hvac_up_kw"] - result_df["hvac_fcr_up_kw"]).clip(lower=0).sum()
            + (result_df["hvac_down_kw"] - result_df["hvac_fcr_down_kw"]).clip(lower=0).sum()
            + result_df["pv_down_kw"].sum()
        ),
        1.0,
    )
    storage_up_energy_mwh = float(max(result_df["bess_soc_mwh"].mean() - 0.10 * max(fleet_meta["bess_energy_cap_mwh"], 0.0), 0))
    storage_down_energy_mwh = float(max(0.95 * max(fleet_meta["bess_energy_cap_mwh"], 0.0) - result_df["bess_soc_mwh"].mean(), 0))
    storage_up_bid_mw = float((result_df["bess_up_kw"] + result_df["ev_up_kw"]).mean() / 1000.0)
    storage_down_bid_mw = float((result_df["bess_down_kw"] + result_df["ev_down_kw"]).mean() / 1000.0)
    fcr_storage_bid_mw = float((result_df["bess_fcr_kw"] + result_df["ev_fcr_kw"]).mean() / 1000.0)
    fcr_storage_endurance_ok = (
        storage_up_energy_mwh >= fcr_storage_bid_mw * FINGRID_RULES["FCR-N"]["storage_endurance_hours"]
        and storage_down_energy_mwh >= fcr_storage_bid_mw * FINGRID_RULES["FCR-N"]["storage_endurance_hours"]
    )
    afrr_storage_endurance_ok = (
        storage_up_energy_mwh >= storage_up_bid_mw * FINGRID_RULES["aFRR"]["storage_endurance_hours"]
        and storage_down_energy_mwh >= storage_down_bid_mw * FINGRID_RULES["aFRR"]["storage_endurance_hours"]
    )

    fcr_min_bid_ok = (
        avg_fcr_bid >= FINGRID_RULES["FCR-N"]["min_bid_kw"]
        if market_mode == "FCR-N"
        else (avg_fcr_bid <= 1.0 or avg_fcr_bid >= FINGRID_RULES["FCR-N"]["min_bid_kw"]) if market_mode == "Combined" else True
    )
    afrr_min_bid_ok = (
        avg_afrr_bid >= FINGRID_RULES["aFRR"]["min_bid_kw"]
        if market_mode == "aFRR"
        else (avg_afrr_bid <= 1.0 or avg_afrr_bid >= FINGRID_RULES["aFRR"]["min_bid_kw"]) if market_mode == "Combined" else True
    )
    compliance = {
        "fcr_min_bid_ok": fcr_min_bid_ok,
        "fcr_symmetric_bid_ok": True if market_mode not in {"FCR-N", "Combined"} else bool(result_df["fcr_bid_kw"].ge(0.0).all()),
        "fcr_local_control_ok": True if market_mode not in {"FCR-N", "Combined"} else fcr_local_control_ok,
        "fcr_bid_granularity_ok": True if market_mode not in {"FCR-N", "Combined"} else fcr_bid_granularity_ok,
        "fcr_dynamic_cap_ok": True if market_mode not in {"FCR-N", "Combined"} else fcr_dynamic_cap_ok,
        "fcr_dynamic_response_ok": True if market_mode not in {"FCR-N", "Combined"} else fcr_dynamic_response_ok,
        "fcr_hvac_share_ok": True if market_mode not in {"FCR-N", "Combined"} else fcr_hvac_share_ok,
        "afrr_min_bid_ok": afrr_min_bid_ok,
        "fcr_response_ok": fcr_response_ok if market_mode in {"FCR-N", "Combined"} else True,
        "afrr_response_ok": afrr_response_s <= FINGRID_RULES["aFRR"]["full_activation_seconds"] if market_mode in {"aFRR", "Combined"} else True,
        "accuracy_ok": (
            FINGRID_RULES["aFRR"]["accuracy_low"] <= mean_accuracy <= FINGRID_RULES["aFRR"]["accuracy_high"]
            if not np.isnan(mean_accuracy) and market_mode in {"aFRR", "Combined"}
            else True
        ),
        "storage_endurance_ok": (
            afrr_storage_endurance_ok and (fcr_storage_endurance_ok if market_mode in {"FCR-N", "Combined"} else True)
            if market_mode in {"aFRR", "Combined", "FCR-N"}
            else True
        ),
        "baseline_method_ok": {"net_load_baseline_kw", "indoor_temp_c_ref", "ev_soc_ref_mwh"}.issubset(result_df.columns.union(df.columns)),
        "mean_accuracy": mean_accuracy,
        "fcr_response_s": fcr_response_95_s,
        "fcr_response_63_s": fcr_response_63_s,
        "fcr_response_95_s": fcr_response_95_s,
        "fcr_response_60_fraction": fcr_response_60_fraction,
        "fcr_response_180_fraction": fcr_response_180_fraction,
        "fcr_energy_60_seconds": fcr_energy_60_seconds,
        "afrr_response_s": afrr_response_s,
    }
    compliance.update(technical_compliance)
    if market_mode in {"FCR-N", "Combined"}:
        fcr_tracking_required_ok = (
            bool(compliance.get("fcr_actual_dynamic_response_ok", True))
            if bool(compliance.get("fcr_tracking_response_evaluated", False))
            else True
        )
        compliance["fcr_response_ok"] = bool(
            compliance["fcr_response_ok"]
            and fcr_tracking_required_ok
            and compliance.get("fcr_stability_margin_ok", True)
        )
        compliance["fcr_prequalification_evidence_ok"] = bool(
            compliance.get("fcr_prequalification_evidence_ok", True)
            and compliance["fcr_response_ok"]
            and compliance.get("fcr_local_control_ok", True)
            and compliance.get("fcr_bid_granularity_ok", True)
        )
    if market_mode in {"aFRR", "Combined"}:
        compliance["afrr_response_ok"] = bool(compliance.get("afrr_response_ok", False))
        compliance["accuracy_ok"] = bool(compliance.get("accuracy_ok", False))
    passed = sum(bool(v) for k, v in compliance.items() if k.endswith("_ok"))
    total_checks = len([k for k in compliance if k.endswith("_ok")])
    eligible_rows = result_df[result_df.get("market_gate_status", pd.Series(index=result_df.index, dtype=object)) == "Participate"]
    next_participation_start = str(eligible_rows.index[0]) if not eligible_rows.empty else ""

    summary = {
        "simulation_only_notice": SIMULATION_ONLY_NOTICE,
        "total_revenue_eur": float(result_df["net_revenue_eur"].sum()),
        "capacity_revenue_eur": float(result_df["capacity_revenue_eur"].sum()),
        "activation_revenue_eur": float(result_df["activation_revenue_eur"].sum()),
        "degradation_cost_eur": float(result_df["degradation_cost_eur"].sum()),
        "comfort_penalty_eur": float(result_df["comfort_penalty_eur"].sum()),
        "non_delivery_risk_cost_eur": float(result_df.get("non_delivery_risk_cost_eur", pd.Series([0.0])).sum()),
        "activation_uncertainty_cost_eur": float(result_df.get("activation_uncertainty_cost_eur", pd.Series([0.0])).sum()),
        "asset_fatigue_cost_eur": float(result_df.get("asset_fatigue_cost_eur", pd.Series([0.0])).sum()),
        "risk_adjusted_profit_eur": float(result_df.get("risk_adjusted_profit_eur", pd.Series([0.0])).sum()),
        "delivered_up_mwh": float(result_df["delivered_up_kw"].sum() * dt_h_result / 1000.0),
        "delivered_down_mwh": float(result_df["delivered_down_kw"].sum() * dt_h_result / 1000.0),
        "co2_avoided_kg": float(result_df["delivered_up_kw"].sum() * dt_h_result / 1000.0 * 140.0),
        "comfort_violations_h": float((result_df["comfort_penalty_eur"] > 0).sum() * dt_h_result),
        "requirement_score_pct": 100.0 * passed / max(total_checks, 1),
        "resource_revenue": resource_revenue,
        "resource_capacity_revenue": resource_capacity_revenue,
        "resource_activation_revenue": resource_activation_revenue,
        "resource_energy_kwh": resource_energy_kwh,
        "risk_policy": str(penalty_weights.get("risk_policy", "investor_balanced")),
        "price_forecast_mode": str(result_df.get("price_forecast_mode", pd.Series(["point"])).mode().iloc[0])
        if "price_forecast_mode" in result_df and not result_df["price_forecast_mode"].empty
        else "point",
        "price_bid_quantile": float(result_df.get("price_bid_quantile", pd.Series([0.50])).mean()),
        "fast_vs_slow": {
            "Fast (BESS + EV)": float(resource_revenue["BESS"] + resource_revenue["EV"]),
            "Slow (HVAC + PV)": float(resource_revenue["HVAC"] + resource_revenue["PV"]),
        },
        "fast_vs_slow_energy_kwh": {
            "Fast (BESS + EV)": float(resource_energy_kwh["BESS"] + resource_energy_kwh["EV"]),
            "Slow (HVAC + PV)": float(resource_energy_kwh["HVAC"] + resource_energy_kwh["PV"]),
        },
        "inner_controller_mode": inner_controller_mode,
        "inner_dt_seconds": int(inner_dt_seconds),
        "inner_mpc_horizon_seconds": int(inner_mpc_horizon_seconds),
        "rotation_strategy": rotation_strategy,
        "gateway_mode": gateway_mode,
        "execute_lower_mpc": bool(execute_lower_mpc),
        "market_gate_participation_intervals": int(len(eligible_rows)),
        "next_participation_start": next_participation_start,
        "mean_tracking_error_kw": float(result_df["tracking_error_kw"].mean()) if "tracking_error_kw" in result_df else 0.0,
        "mean_signed_error_kw": float(result_df.get("mean_signed_error_kw", pd.Series([0.0])).mean()),
        "max_abs_error_per_tick_kw": float(result_df.get("max_abs_error_per_tick_kw", pd.Series([0.0])).max()),
        "socket_violation_up_kw": float(result_df.get("socket_violation_up_kw", pd.Series([0.0])).sum()),
        "socket_violation_down_kw": float(result_df.get("socket_violation_down_kw", pd.Series([0.0])).sum()),
        "mean_shortfall_kw": float(result_df["shortfall_kw"].mean()) if "shortfall_kw" in result_df else 0.0,
    }
    summary.update(technical_summary)
    gateway_commands = pd.concat(gateway_command_summaries, ignore_index=True) if gateway_command_summaries else pd.DataFrame()
    household_contrib, appliance_contrib, usage_fatigue = centralized_controller.contribution_frames(gateway_commands)
    return {
        "history": result_df,
        "tracking_4s": tracking_df,
        "inner_mpc_trace": tracking_df,
        "summary": summary,
        "compliance": compliance,
        "first_schedule": schedule_snapshots[0] if schedule_snapshots else pd.DataFrame(),
        "upper_device_schedule": pd.concat(upper_device_schedules, ignore_index=True) if upper_device_schedules else pd.DataFrame(),
        "gateway_commands": gateway_commands,
        "household_contributions": household_contrib,
        "appliance_contributions": appliance_contrib,
        "usage_fatigue_summary": usage_fatigue,
        "afrr_energy_audit": afrr_energy_audit,
    }

def run_lower_mpc_from_upper_result(
    df: pd.DataFrame,
    upper_result: Dict[str, object],
    fleet_meta: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    preview_4s: pd.DataFrame | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    device_roster: pd.DataFrame | None = None,
    inner_controller_mode: str = "mpc",
    inner_dt_seconds: int = 4,
    inner_mpc_horizon_seconds: int = 4,
    rotation_strategy: str = "usage_aware",
    gateway_mode: str = "live_market_coupled",
    elapsed_seconds: float | None = None,
) -> Dict[str, object]:
    """Attach the lower 4-second controller to an already solved upper MPC socket.

    This path intentionally does not forecast or re-solve the upper MPC. It
    uses the reserve commitments stored in ``upper_result["history"]`` and
    consumes the 4-second market feed only when the live market session starts.
    """

    upper_history = upper_result.get("history", pd.DataFrame()) if isinstance(upper_result, dict) else pd.DataFrame()
    if not isinstance(upper_history, pd.DataFrame) or upper_history.empty:
        empty = pd.DataFrame()
        return {
            "history": empty,
            "summary": {},
            "compliance": {},
            "first_schedule": pd.DataFrame(),
            "tracking_4s": empty,
            "inner_mpc_trace": empty,
            "upper_device_schedule": empty,
            "gateway_commands": empty,
            "household_contributions": empty,
            "appliance_contributions": empty,
            "usage_fatigue_summary": empty,
            "afrr_energy_audit": empty,
        }

    numeric_plan_columns = [
        "bess_fcr_kw",
        "ev_fcr_kw",
        "hvac_fcr_kw",
        "bess_afrr_up_kw",
        "bess_afrr_down_kw",
        "ev_afrr_up_kw",
        "ev_afrr_down_kw",
        "hvac_afrr_up_kw",
        "hvac_afrr_down_kw",
        "pv_afrr_down_kw",
        "non_delivery_risk_cost_eur",
        "activation_uncertainty_cost_eur",
        "asset_fatigue_cost_eur",
        "reserve_buffer_kw",
        "risk_quantile",
        "reserve_buffer_pct",
        "socket_up_kw",
        "socket_down_kw",
        "price_bid_quantile",
        "fcrn_capacity_bid_eur_per_mw_h",
        "afrr_up_capacity_bid_eur_per_mw_h",
        "afrr_down_capacity_bid_eur_per_mw_h",
        "fcrn_capacity_bid_quantile",
        "afrr_up_capacity_bid_quantile",
        "afrr_down_capacity_bid_quantile",
    ]
    passthrough_plan_columns = [
        "combined_stacking_model",
        "combined_shared_capacity_ok",
        "combined_split_capacity_ok",
        "combined_stack_violation_reason",
        "combined_socket_rule",
        "price_forecast_mode",
        "fcrn_capacity_forecast_mode",
        "afrr_up_capacity_forecast_mode",
        "afrr_down_capacity_forecast_mode",
    ]
    controller_config = InnerControllerConfig(
        mode=inner_controller_mode,
        dt_seconds=int(inner_dt_seconds),
        horizon_seconds=int(inner_mpc_horizon_seconds),
        rotation_strategy=rotation_strategy,
        gateway_mode=gateway_mode,
    )
    centralized_controller = CentralizedVPPController(
        device_roster=device_roster if device_roster is not None else representative_roster_from_frame(df, fleet_meta),
        fleet_meta=fleet_meta,
        config=controller_config,
    )

    history: List[Dict[str, float]] = []
    tracking_history: List[pd.DataFrame] = []
    upper_device_schedules: List[pd.DataFrame] = []
    gateway_command_summaries: List[pd.DataFrame] = []
    remaining_live_ticks = None
    if elapsed_seconds is not None:
        remaining_live_ticks = max(int(float(elapsed_seconds) // max(int(inner_dt_seconds), 1)) + 1, 0)
    total_work = max(len(upper_history), 1)

    for t, (timestamp, upper_row) in enumerate(upper_history.iterrows()):
        if remaining_live_ticks is not None and remaining_live_ticks <= 0:
            break
        try:
            row_idx = int(df.index.get_loc(pd.Timestamp(timestamp)))
        except KeyError:
            row_idx = min(t + 1, len(df) - 1)
        row = df.iloc[row_idx]
        price_defaults = {
            "fcrn_capacity_bid_eur_per_mw_h": float(row["fcrn_capacity_eur_per_mw_h"]),
            "afrr_up_capacity_bid_eur_per_mw_h": float(row["afrr_up_capacity_eur_per_mw_h"]),
            "afrr_down_capacity_bid_eur_per_mw_h": float(row["afrr_down_capacity_eur_per_mw_h"]),
            "fcrn_capacity_bid_quantile": 0.50,
            "afrr_up_capacity_bid_quantile": 0.50,
            "afrr_down_capacity_bid_quantile": 0.50,
            "price_bid_quantile": 0.50,
        }
        plan = {}
        for key in numeric_plan_columns:
            value = upper_row.get(key, price_defaults.get(key, 0.0))
            if pd.isna(value):
                value = price_defaults.get(key, 0.0)
            plan[key] = float(value)
        for key in passthrough_plan_columns:
            if key in upper_row:
                plan[key] = upper_row.get(key)
        plan.update(_combined_stack_audit(plan, row, market_mode))
        if plan["socket_up_kw"] <= 0.0 and plan["socket_down_kw"] <= 0.0:
            plan["socket_up_kw"], plan["socket_down_kw"] = _socket_limits_from_plan(plan, market_mode)
        fine_signals = build_inner_tracking_window(df, preview_4s, row_idx, dt_seconds=int(inner_dt_seconds))
        if remaining_live_ticks is not None:
            fine_signals = fine_signals.iloc[:remaining_live_ticks].copy()
            remaining_live_ticks -= len(fine_signals)
        if fine_signals.empty:
            continue
        if progress_callback:
            progress_callback(t + 1, total_work, f"Lower MPC live interval {t + 1}/{total_work}")
        interval_summary, interval_tracking, upper_schedule, command_summary = centralized_controller.execute_interval(
            row=row,
            fine_signals=fine_signals,
            plan=plan,
            interval_index=t,
            market_mode=market_mode,
            resource_mode=resource_mode,
        )
        outer_dt_h = max(float(row["dt_h"]), 1e-9)
        effective_dt_h = max(len(interval_tracking), 1) * max(int(inner_dt_seconds), 1) / 3600.0
        interval_fraction = float(np.clip(effective_dt_h / outer_dt_h, 0.0, 1.0))
        tracking_history.append(interval_tracking)
        if not upper_schedule.empty:
            upper_device_schedules.append(upper_schedule)
        if not command_summary.empty:
            gateway_command_summaries.append(command_summary)

        actual_bess_up = interval_summary["bess_up_kw"]
        actual_bess_down = interval_summary["bess_down_kw"]
        actual_ev_up = interval_summary["ev_up_kw"]
        actual_ev_down = interval_summary["ev_down_kw"]
        actual_hvac_up = interval_summary["hvac_up_kw"]
        actual_hvac_down = interval_summary["hvac_down_kw"]
        actual_pv_down = interval_summary["pv_down_kw"]
        afrr_up_share = float(
            np.clip(
                abs(interval_tracking.get("afrr_signal_request_kw", pd.Series([0.0])).clip(lower=0.0).mean())
                / max(interval_summary["requested_up_kw"], 1e-9),
                0.0,
                1.0,
            )
        )
        afrr_down_share = float(
            np.clip(
                abs((-interval_tracking.get("afrr_signal_request_kw", pd.Series([0.0])).clip(upper=0.0)).mean())
                / max(interval_summary["requested_down_kw"], 1e-9),
                0.0,
                1.0,
            )
        )
        actual_bess_afrr_up = actual_bess_up * afrr_up_share
        actual_ev_afrr_up = actual_ev_up * afrr_up_share
        actual_hvac_afrr_up = actual_hvac_up * afrr_up_share
        actual_bess_afrr_down = actual_bess_down * afrr_down_share
        actual_ev_afrr_down = actual_ev_down * afrr_down_share
        actual_hvac_afrr_down = actual_hvac_down * afrr_down_share
        fcr_bid_kw = plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0) + plan.get("hvac_fcr_kw", 0.0)
        afrr_up_bid_kw = plan.get("bess_afrr_up_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0)
        afrr_down_bid_kw = (
            plan.get("bess_afrr_down_kw", 0.0)
            + plan.get("ev_afrr_down_kw", 0.0)
            + plan.get("hvac_afrr_down_kw", 0.0)
            + plan.get("pv_afrr_down_kw", 0.0)
        )
        stack_audit = _combined_stack_audit(plan, row, market_mode)
        plan.update(stack_audit)
        plan["socket_up_kw"], plan["socket_down_kw"] = _socket_limits_from_plan(plan, market_mode)
        stacking_ok = (
            True
            if market_mode != "Combined"
            else bool(stack_audit["combined_shared_capacity_ok"])
            and float(plan["socket_up_kw"]) + 1e-6 >= fcr_bid_kw + afrr_up_bid_kw
            and float(plan["socket_down_kw"]) + 1e-6 >= fcr_bid_kw + afrr_down_bid_kw
        )
        capacity_revenue = effective_dt_h / 1000.0 * (
            row["fcrn_capacity_eur_per_mw_h"] * fcr_bid_kw
            + row["afrr_up_capacity_eur_per_mw_h"] * afrr_up_bid_kw
            + row["afrr_down_capacity_eur_per_mw_h"] * afrr_down_bid_kw
        )
        activation_revenue = effective_dt_h / 1000.0 * (
            row["afrr_up_energy_eur_per_mwh"] * (actual_bess_afrr_up + actual_ev_afrr_up + actual_hvac_afrr_up)
            + row["afrr_down_energy_eur_per_mwh"] * (actual_bess_afrr_down + actual_ev_afrr_down + actual_hvac_afrr_down + actual_pv_down)
        )
        degradation_cost = float(upper_row.get("degradation_cost_eur", 0.0)) * interval_fraction
        comfort_penalty = float(upper_row.get("comfort_penalty_eur", 0.0)) * interval_fraction
        fatigue_cost = float(plan.get("asset_fatigue_cost_eur", 0.0)) * interval_fraction
        record = upper_row.to_dict()
        record.update(
            {
                "timestamp": row.name,
                "inner_solver_status": str(interval_summary.get("inner_solver_status", "unknown")),
                "inner_controller_mode": inner_controller_mode,
                "fcr_bid_kw": fcr_bid_kw,
                "afrr_up_bid_kw": afrr_up_bid_kw,
                "afrr_down_bid_kw": afrr_down_bid_kw,
                "socket_up_kw": float(plan.get("socket_up_kw", 0.0)),
                "socket_down_kw": float(plan.get("socket_down_kw", 0.0)),
                "price_forecast_mode": str(plan.get("price_forecast_mode", upper_row.get("price_forecast_mode", "point"))),
                "price_bid_quantile": float(plan.get("price_bid_quantile", upper_row.get("price_bid_quantile", 0.50))),
                "fcrn_capacity_bid_eur_per_mw_h": float(plan.get("fcrn_capacity_bid_eur_per_mw_h", row["fcrn_capacity_eur_per_mw_h"])),
                "afrr_up_capacity_bid_eur_per_mw_h": float(plan.get("afrr_up_capacity_bid_eur_per_mw_h", row["afrr_up_capacity_eur_per_mw_h"])),
                "afrr_down_capacity_bid_eur_per_mw_h": float(plan.get("afrr_down_capacity_bid_eur_per_mw_h", row["afrr_down_capacity_eur_per_mw_h"])),
                "stacking_ok": bool(stacking_ok),
                **stack_audit,
                "requested_up_kw": interval_summary["requested_up_kw"],
                "requested_down_kw": interval_summary["requested_down_kw"],
                "delivered_up_kw": interval_summary["delivered_up_kw"],
                "delivered_down_kw": interval_summary["delivered_down_kw"],
                "bess_up_kw": actual_bess_up,
                "bess_down_kw": actual_bess_down,
                "ev_up_kw": actual_ev_up,
                "ev_down_kw": actual_ev_down,
                "hvac_up_kw": actual_hvac_up,
                "hvac_down_kw": actual_hvac_down,
                "bess_afrr_up_delivered_kw": actual_bess_afrr_up,
                "bess_afrr_down_delivered_kw": actual_bess_afrr_down,
                "ev_afrr_up_delivered_kw": actual_ev_afrr_up,
                "ev_afrr_down_delivered_kw": actual_ev_afrr_down,
                "hvac_afrr_up_delivered_kw": actual_hvac_afrr_up,
                "hvac_afrr_down_delivered_kw": actual_hvac_afrr_down,
                "pv_down_kw": actual_pv_down,
                "capacity_revenue_eur": capacity_revenue,
                "activation_revenue_eur": activation_revenue,
                "degradation_cost_eur": degradation_cost,
                "comfort_penalty_eur": comfort_penalty,
                "non_delivery_risk_cost_eur": float(plan.get("non_delivery_risk_cost_eur", 0.0)) * interval_fraction,
                "activation_uncertainty_cost_eur": float(plan.get("activation_uncertainty_cost_eur", 0.0)) * interval_fraction,
                "asset_fatigue_cost_eur": float(plan.get("asset_fatigue_cost_eur", 0.0)) * interval_fraction,
                "net_revenue_eur": capacity_revenue + activation_revenue - degradation_cost - comfort_penalty - fatigue_cost,
                "bess_soc_mwh": interval_summary["bess_soc_mwh"],
                "ev_soc_mwh": interval_summary["ev_soc_mwh"],
                "indoor_temp_c": interval_summary["indoor_temp_c"],
                "fcr_accuracy_ratio": interval_summary["delivered_up_kw"] / interval_summary["requested_up_kw"] if interval_summary["requested_up_kw"] > 1 else np.nan,
                "down_accuracy_ratio": interval_summary["delivered_down_kw"] / interval_summary["requested_down_kw"] if interval_summary["requested_down_kw"] > 1 else np.nan,
                "inner_tracking_samples": interval_summary["tracking_samples"],
                "tracking_error_kw": interval_summary.get("tracking_error_kw", 0.0),
                "mean_signed_error_kw": interval_summary.get("mean_signed_error_kw", interval_summary.get("tracking_error_kw", 0.0)),
                "max_abs_error_per_tick_kw": interval_summary.get("max_abs_error_per_tick_kw", abs(interval_summary.get("tracking_error_kw", 0.0))),
                "socket_violation_up_kw": interval_summary.get("socket_violation_up_kw", 0.0),
                "socket_violation_down_kw": interval_summary.get("socket_violation_down_kw", 0.0),
                "shortfall_kw": interval_summary.get("shortfall_kw", 0.0),
            }
        )
        history.append(record)

    if not history:
        empty = pd.DataFrame()
        return {
            "history": empty,
            "summary": {**(dict(upper_result.get("summary", {})) if isinstance(upper_result, dict) else {}), "execute_lower_mpc": True},
            "compliance": dict(upper_result.get("compliance", {})) if isinstance(upper_result, dict) else {},
            "first_schedule": upper_result.get("first_schedule", pd.DataFrame()) if isinstance(upper_result, dict) else pd.DataFrame(),
            "tracking_4s": empty,
            "inner_mpc_trace": empty,
            "upper_device_schedule": empty,
            "gateway_commands": empty,
            "household_contributions": empty,
            "appliance_contributions": empty,
            "usage_fatigue_summary": empty,
            "afrr_energy_audit": empty,
        }

    result_df = pd.DataFrame(history).set_index("timestamp")
    tracking_df = pd.concat(tracking_history) if tracking_history else pd.DataFrame()
    tracking_df = _enrich_tracking_for_market_audit(tracking_df)
    technical_compliance, technical_summary, afrr_energy_audit = _market_technical_audit(
        result_df=result_df,
        tracking_df=tracking_df,
        fleet_meta=fleet_meta,
        market_mode=market_mode,
        inner_dt_seconds=inner_dt_seconds,
    )
    gateway_commands = pd.concat(gateway_command_summaries, ignore_index=True) if gateway_command_summaries else pd.DataFrame()
    household_contrib, appliance_contrib, usage_fatigue = centralized_controller.contribution_frames(gateway_commands)
    summary = dict(upper_result.get("summary", {})) if isinstance(upper_result, dict) else {}
    tick_h = max(int(inner_dt_seconds), 1) / 3600.0
    if not tracking_df.empty:
        delivered_up_mwh = float(tracking_df["delivered_up_kw"].sum() * tick_h / 1000.0)
        delivered_down_mwh = float(tracking_df["delivered_down_kw"].sum() * tick_h / 1000.0)
        resource_energy_kwh = {
            "BESS": float((tracking_df["bess_up_kw"] + tracking_df["bess_down_kw"]).sum() * tick_h),
            "EV": float((tracking_df["ev_up_kw"] + tracking_df["ev_down_kw"]).sum() * tick_h),
            "HVAC": float((tracking_df["hvac_up_kw"] + tracking_df["hvac_down_kw"]).sum() * tick_h),
            "PV": float(tracking_df["pv_down_kw"].sum() * tick_h),
        }
    else:
        delivered_up_mwh = float(result_df["delivered_up_kw"].sum() * float(df["dt_h"].iloc[0]) / 1000.0)
        delivered_down_mwh = float(result_df["delivered_down_kw"].sum() * float(df["dt_h"].iloc[0]) / 1000.0)
        resource_energy_kwh = {
            "BESS": float((result_df["bess_up_kw"] + result_df["bess_down_kw"]).sum() * float(df["dt_h"].iloc[0])),
            "EV": float((result_df["ev_up_kw"] + result_df["ev_down_kw"]).sum() * float(df["dt_h"].iloc[0])),
            "HVAC": float((result_df["hvac_up_kw"] + result_df["hvac_down_kw"]).sum() * float(df["dt_h"].iloc[0])),
            "PV": float(result_df["pv_down_kw"].sum() * float(df["dt_h"].iloc[0])),
        }
    total_resource_kwh = max(sum(resource_energy_kwh.values()), 1e-9)
    total_revenue = float(result_df.get("net_revenue_eur", pd.Series([0.0])).sum())
    summary.update(
        {
            "simulation_only_notice": SIMULATION_ONLY_NOTICE,
            "total_revenue_eur": total_revenue,
            "capacity_revenue_eur": float(result_df.get("capacity_revenue_eur", pd.Series([0.0])).sum()),
            "activation_revenue_eur": float(result_df.get("activation_revenue_eur", pd.Series([0.0])).sum()),
            "delivered_up_mwh": delivered_up_mwh,
            "delivered_down_mwh": delivered_down_mwh,
            "resource_revenue": {resource: total_revenue * energy_kwh / total_resource_kwh for resource, energy_kwh in resource_energy_kwh.items()},
            "fast_vs_slow": {
                "Fast (BESS + EV)": resource_energy_kwh["BESS"] + resource_energy_kwh["EV"],
                "Slow (HVAC + PV)": resource_energy_kwh["HVAC"] + resource_energy_kwh["PV"],
            },
            "co2_avoided_kg": delivered_up_mwh * 140.0,
            "inner_controller_mode": inner_controller_mode,
            "inner_dt_seconds": int(inner_dt_seconds),
            "inner_mpc_horizon_seconds": int(inner_mpc_horizon_seconds),
            "rotation_strategy": rotation_strategy,
            "gateway_mode": gateway_mode,
            "execute_lower_mpc": True,
            "mean_tracking_error_kw": float(tracking_df["tracking_error_kw"].mean()) if "tracking_error_kw" in tracking_df else 0.0,
            "mean_abs_tracking_error_kw": float(tracking_df["tracking_error_kw"].abs().mean()) if "tracking_error_kw" in tracking_df else 0.0,
            "mean_signed_error_kw": float(result_df.get("mean_signed_error_kw", pd.Series([0.0])).mean()),
            "max_abs_error_per_tick_kw": float(result_df.get("max_abs_error_per_tick_kw", pd.Series([0.0])).max()),
            "socket_violation_up_kw": float(tracking_df.get("socket_violation_up_kw", pd.Series([0.0])).sum()),
            "socket_violation_down_kw": float(tracking_df.get("socket_violation_down_kw", pd.Series([0.0])).sum()),
            "mean_shortfall_kw": float(tracking_df["shortfall_kw"].mean()) if "shortfall_kw" in tracking_df else 0.0,
        }
    )
    summary.update(technical_summary)
    compliance = dict(upper_result.get("compliance", {})) if isinstance(upper_result, dict) else {}
    compliance.update(technical_compliance)
    if market_mode in {"FCR-N", "Combined"}:
        fcr_tracking_required_ok = (
            bool(compliance.get("fcr_actual_dynamic_response_ok", True))
            if bool(compliance.get("fcr_tracking_response_evaluated", False))
            else True
        )
        compliance["fcr_response_ok"] = bool(
            compliance.get("fcr_response_ok", True)
            and fcr_tracking_required_ok
            and compliance.get("fcr_stability_margin_ok", True)
        )
        compliance["fcr_prequalification_evidence_ok"] = bool(
            compliance.get("fcr_prequalification_evidence_ok", True)
            and compliance.get("fcr_response_ok", True)
            and compliance.get("fcr_local_control_ok", True)
            and compliance.get("fcr_bid_granularity_ok", True)
        )
    return {
        "history": result_df,
        "tracking_4s": tracking_df,
        "inner_mpc_trace": tracking_df,
        "summary": summary,
        "compliance": compliance,
        "first_schedule": upper_result.get("first_schedule", pd.DataFrame()) if isinstance(upper_result, dict) else pd.DataFrame(),
        "upper_device_schedule": pd.concat(upper_device_schedules, ignore_index=True) if upper_device_schedules else pd.DataFrame(),
        "gateway_commands": gateway_commands,
        "household_contributions": household_contrib,
        "appliance_contributions": appliance_contrib,
        "usage_fatigue_summary": usage_fatigue,
        "afrr_energy_audit": afrr_energy_audit,
    }

def make_pdf_summary(summary: Dict[str, float], compliance: Dict[str, float], config: Dict[str, object]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40

    lines = [
        "FlexiHome MPC Formulation Summary",
        "",
        f"Market mode: {config['market_mode']}",
        f"Resource mode: {config['resource_mode']}",
        f"Horizon: {config['horizon_hours']} h",
        f"Dispatch window: {config['dispatch_hours']} h",
        "",
        "Objective:",
        "maximize capacity revenue + activation revenue - degradation - comfort penalties - EV departure shortfalls",
        "",
        "State equations:",
        "BESS SOC(k+1) = SOC(k) + eta_c * down_activation - up_activation / eta_d",
        "EV delta(k+1) = EV delta(k) + eta_c * down_activation - up_activation / eta_d",
        "Temp delta(k+1) = alpha * Temp delta(k) + beta * (HVAC down - HVAC up)",
        "",
        "Compliance summary:",
    ]
    for key, value in compliance.items():
        lines.append(f"{key}: {value}")
    lines += ["", "Revenue summary:"]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"{key}: {json.dumps(value, indent=2)}")
        else:
            lines.append(f"{key}: {value}")

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, lines[0])
    y -= 24
    pdf.setFont("Helvetica", 10)
    for line in lines[1:]:
        wrapped = textwrap.wrap(str(line), 95) or [""]
        for segment in wrapped:
            if y < 40:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - 40
            pdf.drawString(40, y, segment)
            y -= 14
    pdf.save()
    buffer.seek(0)
    return buffer.read()

def sensitivity_scan(
    base_params: Dict[str, object],
    price_multipliers: List[float],
    bess_pen_options: List[float],
    market_mode: str,
    resource_mode: str,
    horizon_hours: int,
    dispatch_hours: int,
    penalty_weights: Dict[str, float],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    records = []
    total_cases = max(len(price_multipliers) * len(bess_pen_options), 1)
    case_idx = 0
    for price_mult in price_multipliers:
        for bess_pen in bess_pen_options:
            case_idx += 1
            if progress_callback:
                progress_callback(case_idx - 1, total_cases, f"Preparing scenario {case_idx}/{total_cases}")
            bundle = generate_synthetic_portfolio(
                start_date=base_params["start_date"],
                days=1,
                n_homes=base_params["n_homes"],
                ev_pen=base_params["ev_pen"],
                bess_pen=bess_pen,
                pv_pen=base_params["pv_pen"],
                hvac_pen=base_params["hvac_pen"],
                cloudiness=base_params["cloudiness"],
                climate_shift_c=base_params["climate_shift_c"],
                solar_scale=base_params["solar_scale"],
                seed=base_params["seed"],
                setpoint_c=base_params["setpoint_c"],
                comfort_band_c=base_params["comfort_band_c"],
                scarcity=base_params["scarcity"] * price_mult,
                freq_minutes=int(base_params.get("freq_minutes", 15)),
                hvac_mode=str(base_params.get("hvac_mode", HVAC_MODES[0])),
            )
            df = bundle["data"]
            df["fcrn_capacity_eur_per_mw_h"] *= price_mult
            df["afrr_up_capacity_eur_per_mw_h"] *= price_mult
            df["afrr_down_capacity_eur_per_mw_h"] *= price_mult
            df["afrr_up_energy_eur_per_mwh"] *= price_mult
            df["afrr_down_energy_eur_per_mwh"] *= price_mult
            models = train_default_mpc_models(df, base_params["seed"])
            fleet_meta = {
                "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
                "setpoint_c": base_params["setpoint_c"],
                "comfort_band_c": base_params["comfort_band_c"],
                "n_hvac": bundle["summary"]["n_hvac"],
                "hvac_mode": base_params.get("hvac_mode", HVAC_MODES[0]),
                "hvac_response_s": base_params.get("hvac_response_s", default_hvac_response_seconds(str(base_params.get("hvac_mode", HVAC_MODES[0])))),
            }
            result = run_mpc_controller(
                df,
                models,
                fleet_meta,
                market_mode,
                resource_mode,
                horizon_hours,
                dispatch_hours,
                penalty_weights,
                preview_4s=bundle["preview_4s"],
            )
            records.append({"price_multiplier": price_mult, "bess_pen": bess_pen, "revenue_eur": result["summary"].get("total_revenue_eur", 0.0)})
            if progress_callback:
                progress_callback(case_idx, total_cases, f"Finished scenario {case_idx}/{total_cases}")
    return pd.DataFrame(records)
