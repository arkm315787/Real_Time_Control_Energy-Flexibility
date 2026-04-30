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
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum
from scipy import signal
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .market_data import overlay_real_market_data

FINGRID_RULES = {
    "FCR-N": {
        "min_bid_kw": 100.0,
        "band_hz": 0.1,
        "response_seconds": 5.0,
        "description": "Symmetric reserve around 50 Hz, full activation at ±0.1 Hz.",
    },
    "aFRR": {
        "min_bid_kw": 1000.0,
        "start_seconds": 30.0,
        "full_activation_seconds": 300.0,
        "accuracy_low": 0.90,
        "accuracy_high": 1.10,
        "storage_endurance_hours": 1.0,
        "description": "Centralized 4-second signal with full activation inside 5 minutes.",
    },
}

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

@dataclass
class ForecastSpec:
    target: str
    predictors: List[str]
    lags: int
    horizon_steps: int
    model: XGBRegressor
    metrics: Dict[str, float]

def hvac_fast_fraction(hvac_mode: str) -> float:
    return float(HVAC_FAST_FRACTION.get(hvac_mode, HVAC_FAST_FRACTION[HVAC_MODES[0]]))

def default_hvac_response_seconds(hvac_mode: str) -> float:
    return float(HVAC_DEFAULT_RESPONSE_S.get(hvac_mode, HVAC_DEFAULT_RESPONSE_S[HVAC_MODES[0]]))

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
    df["hvac_fast_kw"] = hvac_fast_fraction(hvac_mode) * np.minimum(df["hvac_up_kw"], df["hvac_down_kw"])
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
        "freq_minutes": freq_minutes,
        "market_data_source": market_data_status["mode_used"],
        "market_data_status": market_data_status,
    }
    return {"data": df, "preview_4s": preview, "summary": summary}

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
    spec = ForecastSpec(target=target, predictors=predictors, lags=lags, horizon_steps=horizon_steps, model=model, metrics=metrics)
    return spec, comparison

def build_single_feature_row(frame: pd.DataFrame, row_idx: int, predictors: List[str], lags: int) -> pd.DataFrame:
    idx = frame.index[row_idx]
    tff = time_feature_frame(pd.DatetimeIndex([idx]))
    row = tff.iloc[[0]].copy()
    for col in predictors:
        row[col] = frame.iloc[row_idx][col]
        for lag in range(1, lags + 1):
            lag_pos = max(0, row_idx - lag)
            row[f"{col}_lag{lag}"] = frame.iloc[lag_pos][col]
    return row

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

def iterative_forecast(
    df: pd.DataFrame,
    current_pos: int,
    horizon_steps: int,
    models: Dict[str, ForecastSpec],
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
            prediction = float(spec.model.predict(row)[0])
            future_slice.loc[future_slice.index[pred_pos], target] = prediction
            output.loc[future_slice.index[pred_pos], target] = prediction
        if step_callback:
            step_callback(step_ahead, horizon_steps)
    return output

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

    for _, row in forecast_df.iterrows():
        fcr_on = market_mode in {"FCR-N", "Combined"}
        afrr_on = market_mode in {"aFRR", "Combined"}
        dt_h = float(row["dt_h"])

        bess_fcr = ev_fcr = hvac_fcr = 0.0
        if fcr_on:
            bess_fcr = float(min(row["bess_up_kw"], row["bess_down_kw"]))
            ev_fcr = float(min(row["ev_up_kw"], row["ev_down_kw"]))
            hvac_fcr = float(enable_hvac * min(row["hvac_up_kw"], row["hvac_down_kw"], row.get("hvac_fast_kw", 0.0)))

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

        rows.append(
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
            }
        )

    schedule = pd.DataFrame(rows, index=forecast_df.index)
    schedule["fcr_bid_kw"] = schedule["bess_fcr_kw"] + schedule["ev_fcr_kw"] + schedule["hvac_fcr_kw"]
    schedule["afrr_up_bid_kw"] = schedule["bess_afrr_up_kw"] + schedule["ev_afrr_up_kw"] + schedule["hvac_afrr_up_kw"]
    schedule["afrr_down_bid_kw"] = (
        schedule["bess_afrr_down_kw"] + schedule["ev_afrr_down_kw"] + schedule["hvac_afrr_down_kw"] + schedule["pv_afrr_down_kw"]
    )
    return {"status": "HeuristicFallback", "controls": schedule.iloc[0].to_dict(), "schedule": schedule}

def solve_mpc_step(
    forecast_df: pd.DataFrame,
    state: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    penalty_weights: Dict[str, float],
    fleet_meta: Dict[str, float],
) -> Dict[str, object]:
    horizon = len(forecast_df)
    dt_h = float(forecast_df["dt_h"].iloc[0])
    eta_c = 0.95
    eta_d = 0.95
    temp_alpha = 0.90
    temp_beta = 0.16 * dt_h

    only_fast = resource_mode == "Fast only"
    enable_hvac = 0.0 if only_fast else 1.0
    enable_pv = 0.0 if only_fast else 1.0

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

        problem += bess_fcr[k] + bess_up[k] <= row["bess_up_kw"]
        problem += bess_fcr[k] + bess_down[k] <= row["bess_down_kw"]
        problem += ev_fcr[k] + ev_up[k] <= row["ev_up_kw"]
        problem += ev_fcr[k] + ev_down[k] <= row["ev_down_kw"]
        problem += hvac_fcr[k] + hvac_up[k] <= enable_hvac * row["hvac_up_kw"]
        problem += hvac_fcr[k] + hvac_down[k] <= enable_hvac * row["hvac_down_kw"]
        problem += hvac_fcr[k] <= enable_hvac * row.get("hvac_fast_kw", 0.0)
        problem += pv_down[k] <= enable_pv * row["pv_down_kw"]

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
        problem += ev_delta[k + 1] == ev_delta[k] + dt_h / 1000.0 * (
            eta_c * (afrr_down_frac * ev_down[k] + fcr_down_frac * ev_fcr[k])
            - (afrr_up_frac * ev_up[k] + fcr_up_frac * ev_fcr[k]) / eta_d
        )
        problem += ev_ref + ev_delta[k + 1] >= ev_min - ev_slack[k]
        problem += ev_ref + ev_delta[k + 1] <= ev_max

        temp_ref = row["indoor_temp_c_ref"]
        temp_lo = fleet_meta["setpoint_c"] - fleet_meta["comfort_band_c"]
        temp_hi = fleet_meta["setpoint_c"] + fleet_meta["comfort_band_c"]
        problem += temp_delta[k + 1] == temp_alpha * temp_delta[k] + temp_beta * (
            (
                (afrr_down_frac * hvac_down[k] + fcr_down_frac * hvac_fcr[k])
                - (afrr_up_frac * hvac_up[k] + fcr_up_frac * hvac_fcr[k])
            )
            / max(fleet_meta["n_hvac"], 1)
        )
        problem += temp_ref + temp_delta[k + 1] >= temp_lo - temp_low_slack[k]
        problem += temp_ref + temp_delta[k + 1] <= temp_hi + temp_high_slack[k]

        fcr_bid = bess_fcr[k] + ev_fcr[k] + hvac_fcr[k]
        afrr_up_bid = bess_up[k] + ev_up[k] + hvac_up[k]
        afrr_down_bid = bess_down[k] + ev_down[k] + hvac_down[k] + pv_down[k]

        cap_revenue = dt_h / 1000.0 * (
            row["fcrn_capacity_eur_per_mw_h"] * fcr_bid
            + row["afrr_up_capacity_eur_per_mw_h"] * afrr_up_bid
            + row["afrr_down_capacity_eur_per_mw_h"] * afrr_down_bid
        )
        energy_revenue = dt_h / 1000.0 * (
            row["afrr_up_energy_eur_per_mwh"] * afrr_up_frac * afrr_up_bid
            + row["afrr_down_energy_eur_per_mwh"] * afrr_down_frac * afrr_down_bid
        )
        degradation = penalty_weights["degradation"] * dt_h / 1000.0 * (
            (afrr_up_frac * bess_up[k] + afrr_down_frac * bess_down[k] + (fcr_up_frac + fcr_down_frac) * bess_fcr[k])
            + 0.4 * (afrr_up_frac * ev_up[k] + afrr_down_frac * ev_down[k] + (fcr_up_frac + fcr_down_frac) * ev_fcr[k])
        )
        discomfort = penalty_weights["comfort"] * (temp_low_slack[k] + temp_high_slack[k]) + penalty_weights["departure"] * ev_slack[k]
        objective_terms.append(cap_revenue + energy_revenue - degradation - discomfort)

    problem += lpSum(objective_terms)
    try:
        problem.solve(PULP_CBC_CMD(msg=False))
    except Exception:
        return heuristic_mpc_step(forecast_df, state, market_mode, resource_mode, fleet_meta)

    if LpStatus[problem.status] not in {"Optimal", "Not Solved"}:
        return heuristic_mpc_step(forecast_df, state, market_mode, resource_mode, fleet_meta)

    rows = []
    for k in idxs:
        rows.append(
            {
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
                "bess_soc_mwh_plan": float(soc_b[k + 1].value() or 0.0),
                "ev_delta_mwh_plan": float(ev_delta[k + 1].value() or 0.0),
                "temp_delta_c_plan": float(temp_delta[k + 1].value() or 0.0),
            }
        )
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
    hvac_fast_av = min(hvac_up_av, hvac_down_av) * hvac_fast_fraction(str(fleet_meta.get("hvac_mode", HVAC_MODES[0])))

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

def run_mpc_controller(
    df: pd.DataFrame,
    models: Dict[str, ForecastSpec],
    fleet_meta: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    horizon_hours: int,
    dispatch_hours: int,
    penalty_weights: Dict[str, float],
    preview_4s: pd.DataFrame | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, object]:
    sim_steps = min(int(dispatch_hours / df["dt_h"].iloc[0]), len(df) - 2)
    horizon_steps = min(int(horizon_hours / df["dt_h"].iloc[0]), len(df) - 2)
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
    total_work = max(sim_steps * (horizon_steps + 2), 1)

    for t in range(sim_steps):
        base_slice = df.iloc[t + 1 : t + 1 + horizon_steps].copy()
        base_progress = t * (horizon_steps + 2)

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
        solution = solve_mpc_step(forecast, state, market_mode, resource_mode, penalty_weights, fleet_meta)
        if solution["schedule"].empty:
            break
        schedule_snapshots.append(solution["schedule"])
        plan = solution["controls"]
        row = df.iloc[t]
        dt_h = row["dt_h"]
        fine_signals = build_inner_tracking_window(df, preview_4s, t)
        if progress_callback:
            progress_callback(
                base_progress + horizon_steps + 2,
                total_work,
                f"Tracking 4-second controller for MPC interval {t + 1}/{sim_steps}",
            )
        interval_summary, interval_tracking = simulate_tracking_interval(
            row=row,
            fine_signals=fine_signals,
            plan=plan,
            state=state,
            fleet_meta=fleet_meta,
            market_mode=market_mode,
        )
        tracking_history.append(interval_tracking)

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

        fcr_bid_kw = plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0) + plan.get("hvac_fcr_kw", 0.0)
        afrr_up_bid_kw = plan.get("bess_afrr_up_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0)
        afrr_down_bid_kw = (
            plan.get("bess_afrr_down_kw", 0.0)
            + plan.get("ev_afrr_down_kw", 0.0)
            + plan.get("hvac_afrr_down_kw", 0.0)
            + plan.get("pv_afrr_down_kw", 0.0)
        )
        capacity_revenue = dt_h / 1000.0 * (
            row["fcrn_capacity_eur_per_mw_h"] * fcr_bid_kw
            + row["afrr_up_capacity_eur_per_mw_h"] * afrr_up_bid_kw
            + row["afrr_down_capacity_eur_per_mw_h"] * afrr_down_bid_kw
        )
        activation_revenue = dt_h / 1000.0 * (
            row["afrr_up_energy_eur_per_mwh"] * (actual_bess_up + actual_ev_up + actual_hvac_up)
            + row["afrr_down_energy_eur_per_mwh"] * (actual_bess_down + actual_ev_down + actual_hvac_down + actual_pv_down)
        )
        degradation_cost = penalty_weights["degradation"] * dt_h / 1000.0 * (
            actual_bess_up + actual_bess_down + 0.4 * (actual_ev_up + actual_ev_down)
        )
        comfort_penalty = penalty_weights["comfort"] * max(
            0.0,
            abs(row["indoor_temp_c_ref"] + state["temp_delta_c"] - fleet_meta["setpoint_c"]) - fleet_meta["comfort_band_c"],
        )

        history.append(
            {
                "timestamp": row.name,
                "baseline_net_load_kw": interval_summary["baseline_net_load_kw"],
                "optimized_net_load_kw": interval_summary["optimized_net_load_kw"],
                "frequency_hz": interval_summary["frequency_hz"],
                "fcr_bid_kw": fcr_bid_kw,
                "afrr_up_bid_kw": afrr_up_bid_kw,
                "afrr_down_bid_kw": afrr_down_bid_kw,
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
                "pv_down_kw": actual_pv_down,
                "capacity_revenue_eur": capacity_revenue,
                "activation_revenue_eur": activation_revenue,
                "degradation_cost_eur": degradation_cost,
                "comfort_penalty_eur": comfort_penalty,
                "net_revenue_eur": capacity_revenue + activation_revenue - degradation_cost - comfort_penalty,
                "bess_soc_mwh": interval_summary["bess_soc_mwh"],
                "ev_soc_mwh": interval_summary["ev_soc_mwh"],
                "indoor_temp_c": interval_summary["indoor_temp_c"],
                "fcr_accuracy_ratio": delivered_up / requested_up if requested_up > 1 else np.nan,
                "down_accuracy_ratio": delivered_down / requested_down if requested_down > 1 else np.nan,
                "inner_tracking_samples": interval_summary["tracking_samples"],
            }
        )

    result_df = pd.DataFrame(history).set_index("timestamp")
    if result_df.empty:
        return {"history": result_df, "summary": {}, "compliance": {}, "first_schedule": pd.DataFrame()}
    if progress_callback:
        progress_callback(total_work, total_work, "Simulation finished")

    resource_revenue = {
        "BESS": float(
            (result_df["bess_up_kw"] + result_df["bess_down_kw"]).sum() * result_df["activation_revenue_eur"].sum()
            / max((result_df["delivered_up_kw"] + result_df["delivered_down_kw"]).sum(), 1.0)
            + result_df["capacity_revenue_eur"].sum() * 0.34
        ),
        "EV": float(
            (result_df["ev_up_kw"] + result_df["ev_down_kw"]).sum() * result_df["activation_revenue_eur"].sum()
            / max((result_df["delivered_up_kw"] + result_df["delivered_down_kw"]).sum(), 1.0)
            + result_df["capacity_revenue_eur"].sum() * 0.28
        ),
        "HVAC": float(
            (result_df["hvac_up_kw"] + result_df["hvac_down_kw"]).sum() * result_df["activation_revenue_eur"].sum()
            / max((result_df["delivered_up_kw"] + result_df["delivered_down_kw"]).sum(), 1.0)
            + result_df["capacity_revenue_eur"].sum() * 0.24
        ),
        "PV": float(
            result_df["pv_down_kw"].sum() * result_df["activation_revenue_eur"].sum()
            / max((result_df["delivered_up_kw"] + result_df["delivered_down_kw"]).sum(), 1.0)
            + result_df["capacity_revenue_eur"].sum() * 0.14
        ),
    }
    accuracy_ratios = pd.concat([result_df["fcr_accuracy_ratio"].dropna(), result_df["down_accuracy_ratio"].dropna()])
    mean_accuracy = float(accuracy_ratios.mean()) if not accuracy_ratios.empty else np.nan

    avg_fcr_bid = float(result_df["fcr_bid_kw"].mean())
    avg_afrr_bid = float(np.maximum(result_df["afrr_up_bid_kw"], result_df["afrr_down_bid_kw"]).mean())
    hvac_response_s = float(fleet_meta.get("hvac_response_s", default_hvac_response_seconds(str(fleet_meta.get("hvac_mode", HVAC_MODES[0])))))
    fcr_response_s = (
        2.0 * result_df["bess_up_kw"].sum()
        + 4.0 * result_df["ev_up_kw"].sum()
        + hvac_response_s * result_df["hvac_fcr_up_kw"].sum()
    ) / max(result_df["bess_up_kw"].sum() + result_df["ev_up_kw"].sum() + result_df["hvac_fcr_up_kw"].sum(), 1.0)
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

    compliance = {
        "fcr_min_bid_ok": avg_fcr_bid >= FINGRID_RULES["FCR-N"]["min_bid_kw"] if market_mode in {"FCR-N", "Combined"} else True,
        "afrr_min_bid_ok": avg_afrr_bid >= FINGRID_RULES["aFRR"]["min_bid_kw"] if market_mode in {"aFRR", "Combined"} else True,
        "fcr_response_ok": fcr_response_s <= FINGRID_RULES["FCR-N"]["response_seconds"] if market_mode in {"FCR-N", "Combined"} else True,
        "afrr_response_ok": afrr_response_s <= FINGRID_RULES["aFRR"]["full_activation_seconds"] if market_mode in {"aFRR", "Combined"} else True,
        "accuracy_ok": (
            FINGRID_RULES["aFRR"]["accuracy_low"] <= mean_accuracy <= FINGRID_RULES["aFRR"]["accuracy_high"]
            if not np.isnan(mean_accuracy) and market_mode in {"aFRR", "Combined"}
            else True
        ),
        "storage_endurance_ok": (
            storage_up_energy_mwh >= storage_up_bid_mw * FINGRID_RULES["aFRR"]["storage_endurance_hours"]
            and storage_down_energy_mwh >= storage_down_bid_mw * FINGRID_RULES["aFRR"]["storage_endurance_hours"]
            if market_mode in {"aFRR", "Combined"}
            else True
        ),
        "baseline_method_ok": {"net_load_baseline_kw", "indoor_temp_c_ref", "ev_soc_ref_mwh"}.issubset(result_df.columns.union(df.columns)),
        "mean_accuracy": mean_accuracy,
        "fcr_response_s": fcr_response_s,
        "afrr_response_s": afrr_response_s,
    }
    passed = sum(bool(v) for k, v in compliance.items() if k.endswith("_ok"))
    total_checks = len([k for k in compliance if k.endswith("_ok")])

    summary = {
        "total_revenue_eur": float(result_df["net_revenue_eur"].sum()),
        "capacity_revenue_eur": float(result_df["capacity_revenue_eur"].sum()),
        "activation_revenue_eur": float(result_df["activation_revenue_eur"].sum()),
        "degradation_cost_eur": float(result_df["degradation_cost_eur"].sum()),
        "comfort_penalty_eur": float(result_df["comfort_penalty_eur"].sum()),
        "delivered_up_mwh": float(result_df["delivered_up_kw"].sum() * df["dt_h"].iloc[0] / 1000.0),
        "delivered_down_mwh": float(result_df["delivered_down_kw"].sum() * df["dt_h"].iloc[0] / 1000.0),
        "co2_avoided_kg": float(result_df["delivered_up_kw"].sum() * df["dt_h"].iloc[0] / 1000.0 * 140.0),
        "comfort_violations_h": float((result_df["comfort_penalty_eur"] > 0).sum() * df["dt_h"].iloc[0]),
        "requirement_score_pct": 100.0 * passed / max(total_checks, 1),
        "resource_revenue": resource_revenue,
        "fast_vs_slow": {
            "Fast (BESS + EV)": float(resource_revenue["BESS"] + resource_revenue["EV"]),
            "Slow (HVAC + PV)": float(resource_revenue["HVAC"] + resource_revenue["PV"]),
        },
    }
    return {
        "history": result_df,
        "tracking_4s": pd.concat(tracking_history) if tracking_history else pd.DataFrame(),
        "summary": summary,
        "compliance": compliance,
        "first_schedule": schedule_snapshots[0] if schedule_snapshots else pd.DataFrame(),
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
