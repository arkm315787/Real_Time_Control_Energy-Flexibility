"""
FlexiHome: Aggregated Residential Flexibility for Fingrid Balancing Markets
============================================================================

README
------
1. Create a virtual environment (recommended).
2. Install dependencies:
       pip install -r requirements.txt
3. Run the dashboard:
       streamlit run app.py

This educational Streamlit application demonstrates how aggregated residential
flexibility from EVs, BESS, PV, and thermal loads can participate in
Fingrid's FCR-N and aFRR balancing products. It uses fully synthetic but
realistic Finland-inspired data, XGBoost forecasting, dynamic response
models, and a receding-horizon MPC-style optimizer built with PuLP.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import time
import textwrap
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum
from scipy import signal
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    import control as ctl

    HAS_CONTROL = True
except Exception:
    HAS_CONTROL = False


st.set_page_config(
    page_title="FlexiHome",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


RESOURCE_COLORS = {
    "BESS": "#7B61FF",
    "EV": "#6E44FF",
    "PV": "#2F9E44",
    "HVAC": "#FF922B",
    "Base": "#355070",
    "Net": "#0B7285",
    "Frequency": "#D9480F",
    "aFRR": "#C2255C",
}

PLOTLY_TEMPLATE = {"Dark": "plotly_dark", "Light": "plotly_white"}

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


def inject_css(theme_mode: str) -> None:
    bg = (
        "linear-gradient(140deg, rgba(12,42,74,0.16), rgba(31,100,69,0.08) 45%, rgba(255,146,43,0.10))"
        if theme_mode == "Dark"
        else "linear-gradient(140deg, rgba(207,231,255,0.55), rgba(226,255,244,0.68) 52%, rgba(255,236,214,0.72))"
    )
    card_bg = "rgba(9,17,28,0.72)" if theme_mode == "Dark" else "rgba(255,255,255,0.86)"
    card_border = "rgba(123, 97, 255, 0.30)" if theme_mode == "Dark" else "rgba(11, 114, 133, 0.20)"
    text_color = "#F8F9FA" if theme_mode == "Dark" else "#183153"
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: {bg};
                background-attachment: fixed;
                color: {text_color};
                font-family: "Aptos", "Segoe UI", "Trebuchet MS", sans-serif;
            }}
            .block-container {{
                padding-top: 1.4rem !important;
                padding-bottom: 2rem !important;
                max-width: none !important;
                padding-left: 1.35rem !important;
                padding-right: 1.65rem !important;
            }}
            h1, .stApp h1 {{
                font-size: 3.2rem !important;
                font-weight: 800 !important;
                line-height: 1.08 !important;
                letter-spacing: -0.03em !important;
            }}
            h2, .stApp h2 {{
                font-size: 2.25rem !important;
                font-weight: 760 !important;
                line-height: 1.16 !important;
            }}
            h3, .stApp h3 {{
                font-size: 1.65rem !important;
                font-weight: 720 !important;
            }}
            h4, .stApp h4 {{
                font-size: 1.28rem !important;
                font-weight: 720 !important;
            }}
            p, li, label, .stMarkdown, .stCaption, .small-note {{
                font-size: 1.04rem !important;
                line-height: 1.56 !important;
            }}
            [data-testid="stMetricLabel"] {{
                font-size: 1.08rem !important;
                font-weight: 700 !important;
            }}
            [data-testid="stMetricValue"] {{
                font-size: 2.2rem !important;
                font-weight: 800 !important;
                line-height: 1.05 !important;
            }}
            [data-baseweb="tab-list"] {{
                align-items: center !important;
                overflow: visible !important;
                gap: 0.15rem !important;
                min-height: 3.2rem !important;
                padding-top: 0.3rem !important;
                padding-bottom: 0.15rem !important;
            }}
            div[data-testid="stTabs"] {{
                margin-top: 0.65rem !important;
            }}
            [data-baseweb="tab"] {{
                overflow: visible !important;
            }}
            [data-baseweb="tab-list"] button {{
                font-size: 1.02rem !important;
                font-weight: 700 !important;
                min-height: 3rem !important;
                height: auto !important;
                line-height: 1.35 !important;
                padding: 0.55rem 0.85rem 0.55rem 0.85rem !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                overflow: visible !important;
                white-space: nowrap !important;
            }}
            [data-baseweb="tab-list"] button p,
            [data-baseweb="tab"] p {{
                margin: 0 !important;
                padding: 0 !important;
                line-height: 1.4 !important;
                font-size: 1.02rem !important;
                font-weight: 700 !important;
                display: inline-block !important;
                overflow: visible !important;
                position: relative !important;
                top: 2px !important;
            }}
            div[data-testid="stTabs"] > div:first-child {{
                overflow: visible !important;
                min-height: 3.25rem !important;
                padding-top: 0.2rem !important;
            }}
            [data-testid="stSidebar"] {{
                min-width: 365px !important;
                max-width: 365px !important;
            }}
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] div[role="radiogroup"] *,
            [data-testid="stSidebar"] .stSlider p {{
                font-size: 1rem !important;
            }}
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {{
                font-size: 1.24rem !important;
                font-weight: 760 !important;
            }}
            [data-testid="stExpander"] summary p {{
                font-size: 1.02rem !important;
                font-weight: 650 !important;
            }}
            [data-testid="stDataFrame"] div,
            [data-testid="stTable"] div {{
                font-size: 0.98rem !important;
            }}
            .flexi-card {{
                padding: 1.2rem 1.35rem;
                border-radius: 18px;
                background: {card_bg};
                border: 1px solid {card_border};
                box-shadow: 0 18px 36px rgba(0,0,0,0.08);
                margin-bottom: 0.7rem;
            }}
            .flexi-card h4 {{
                margin: 0 0 0.4rem 0;
                letter-spacing: 0.02em;
                font-size: 1.38rem !important;
                font-weight: 760 !important;
            }}
            .flexi-badge {{
                display: inline-block;
                margin-right: 0.35rem;
                margin-top: 0.2rem;
                padding: 0.34rem 0.72rem;
                border-radius: 999px;
                background: rgba(123,97,255,0.12);
                border: 1px solid rgba(123,97,255,0.25);
                font-size: 0.96rem;
                font-weight: 650;
            }}
            .small-note {{
                opacity: 0.78;
                font-size: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hvac_fast_fraction(hvac_mode: str) -> float:
    return float(HVAC_FAST_FRACTION.get(hvac_mode, HVAC_FAST_FRACTION[HVAC_MODES[0]]))


def default_hvac_response_seconds(hvac_mode: str) -> float:
    return float(HVAC_DEFAULT_RESPONSE_S.get(hvac_mode, HVAC_DEFAULT_RESPONSE_S[HVAC_MODES[0]]))


def apply_chart_style(fig: go.Figure, template: str, height: int | None = None, title: str | None = None) -> go.Figure:
    existing_title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    if isinstance(existing_title, str) and existing_title.strip().lower() == "undefined":
        existing_title = None
    resolved_title = title if title not in {None, "undefined"} else existing_title
    layout_updates = dict(
        template=template,
        height=height if height is not None else fig.layout.height,
        font=dict(size=16),
        legend=dict(font=dict(size=16), title_font=dict(size=17)),
        hoverlabel=dict(font_size=16),
        margin=dict(l=20, r=20, t=50, b=12),
    )
    if resolved_title:
        layout_updates["title"] = dict(text=resolved_title, font=dict(size=22))
    else:
        fig.update_layout(title=None)
    fig.update_layout(**layout_updates)
    fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=15))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=15))
    return fig


def make_progress_tracker(
    progress_bar,
    text_placeholder,
    label: str,
) -> Callable[[int, int, str], None]:
    started = time.perf_counter()

    def update(current: int, total: int, message: str) -> None:
        total = max(int(total), 1)
        current = max(0, min(int(current), total))
        fraction = current / total
        elapsed = time.perf_counter() - started
        progress_bar.progress(fraction)
        text_placeholder.caption(f"{label}: {message} • {fraction:.0%} complete • {elapsed:.1f}s elapsed")

    return update


def styled_dataframe(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    return frame.style.set_properties(**{"font-size": "16px"}).set_table_styles(
        [
            {"selector": "th", "props": [("font-size", "16px"), ("font-weight", "700")]},
            {"selector": "td", "props": [("font-size", "15px")]},
        ]
    )


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


@st.cache_data(show_spinner=False)
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
) -> Dict[str, object]:
    dt_h = freq_minutes / 60.0
    steps = int(days * 24 * 60 / freq_minutes)
    index = pd.date_range(pd.to_datetime(start_date), periods=steps, freq=f"{freq_minutes}min")
    rng = np.random.default_rng(seed)

    weather = generate_weather(index, seed, cloudiness, climate_shift_c, solar_scale)
    prices = generate_market_prices(index, seed, scarcity)
    activation, preview = generate_activation_signals(index, seed, preview_days=min(days, 3))
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


def ensure_default_models(df: pd.DataFrame, seed: int) -> Dict[str, ForecastSpec]:
    signature = (
        len(df),
        str(df.index[0]),
        str(df.index[-1]),
        seed,
        int(round(float(df["dt_h"].iloc[0]) * 60.0)),
        float(df["base_load_kw"].mean()),
        float(df["pv_available_kw"].mean()),
        float(df["fast_sym_kw"].mean()),
    )
    if "default_mpc_models" not in st.session_state or st.session_state.get("default_mpc_signature") != signature:
        st.session_state["default_mpc_models"] = train_default_mpc_models(df, seed)
        st.session_state["default_mpc_signature"] = signature
    custom = st.session_state.get("custom_mpc_models", {})
    merged = dict(st.session_state["default_mpc_models"])
    merged.update(custom)
    return merged


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
    if HAS_CONTROL:
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


def indicator_figure(value: float, title: str, threshold: float, mode: str = "gauge+number") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode=mode,
            value=value,
            title={"text": title, "font": {"size": 20}},
            number={"font": {"size": 34}},
            gauge={
                "axis": {"range": [0, max(value * 1.15, threshold * 1.2, 1)]},
                "bar": {"color": RESOURCE_COLORS["Net"]},
                "threshold": {"line": {"color": "#E03131", "width": 4}, "value": threshold},
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10))
    return fig


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
    problem.solve(PULP_CBC_CMD(msg=False))

    if LpStatus[problem.status] not in {"Optimal", "Not Solved"}:
        return {"status": LpStatus[problem.status], "controls": {}, "schedule": pd.DataFrame(index=forecast_df.index)}

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


def _available_up_down(row: pd.Series, state: Dict[str, float], fleet_meta: Dict[str, float]) -> Dict[str, float]:
    dt_h = row["dt_h"]
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


def run_mpc_controller(
    df: pd.DataFrame,
    models: Dict[str, ForecastSpec],
    fleet_meta: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    horizon_hours: int,
    dispatch_hours: int,
    penalty_weights: Dict[str, float],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Dict[str, object]:
    sim_steps = min(int(dispatch_hours / df["dt_h"].iloc[0]), len(df) - 2)
    horizon_steps = min(int(horizon_hours / df["dt_h"].iloc[0]), len(df) - 2)
    state = {
        "bess_soc_mwh": float(df["bess_soc_ref_mwh"].iloc[0]),
        "ev_soc_delta_mwh": 0.0,
        "temp_delta_c": 0.0,
    }
    history: List[Dict[str, float]] = []
    schedule_snapshots: List[pd.DataFrame] = []
    total_work = max(sim_steps * (horizon_steps + 1), 1)

    for t in range(sim_steps):
        base_slice = df.iloc[t + 1 : t + 1 + horizon_steps].copy()
        base_progress = t * (horizon_steps + 1)

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
        avail = _available_up_down(row, state, fleet_meta)
        dt_h = row["dt_h"]
        eta_c = 0.95
        eta_d = 0.95
        temp_alpha = 0.90
        temp_beta = 0.16 * dt_h

        fcr_up_frac = row["fcr_up_act_frac"] if market_mode in {"FCR-N", "Combined"} else 0.0
        fcr_down_frac = row["fcr_down_act_frac"] if market_mode in {"FCR-N", "Combined"} else 0.0
        afrr_up_frac = row["afrr_up_act_frac"] if market_mode in {"aFRR", "Combined"} else 0.0
        afrr_down_frac = row["afrr_down_act_frac"] if market_mode in {"aFRR", "Combined"} else 0.0

        bess_fcr_av = min(plan.get("bess_fcr_kw", 0.0), avail["bess_up_av"], avail["bess_down_av"])
        ev_fcr_av = min(plan.get("ev_fcr_kw", 0.0), avail["ev_up_av"], avail["ev_down_av"])
        hvac_fcr_av = min(plan.get("hvac_fcr_kw", 0.0), avail["hvac_fast_av"])
        bess_afrr_up = min(plan.get("bess_afrr_up_kw", 0.0), max(avail["bess_up_av"] - bess_fcr_av, 0.0))
        bess_afrr_down = min(plan.get("bess_afrr_down_kw", 0.0), max(avail["bess_down_av"] - bess_fcr_av, 0.0))
        ev_afrr_up = min(plan.get("ev_afrr_up_kw", 0.0), max(avail["ev_up_av"] - ev_fcr_av, 0.0))
        ev_afrr_down = min(plan.get("ev_afrr_down_kw", 0.0), max(avail["ev_down_av"] - ev_fcr_av, 0.0))
        hvac_afrr_up = min(plan.get("hvac_afrr_up_kw", 0.0), max(avail["hvac_up_av"] - hvac_fcr_av, 0.0))
        hvac_afrr_down = min(plan.get("hvac_afrr_down_kw", 0.0), max(avail["hvac_down_av"] - hvac_fcr_av, 0.0))
        pv_afrr_down = min(plan.get("pv_afrr_down_kw", 0.0), avail["pv_down_av"])

        actual_bess_up = fcr_up_frac * bess_fcr_av + afrr_up_frac * bess_afrr_up
        actual_bess_down = fcr_down_frac * bess_fcr_av + afrr_down_frac * bess_afrr_down
        actual_ev_up = fcr_up_frac * ev_fcr_av + afrr_up_frac * ev_afrr_up
        actual_ev_down = fcr_down_frac * ev_fcr_av + afrr_down_frac * ev_afrr_down
        actual_hvac_fcr_up = fcr_up_frac * hvac_fcr_av
        actual_hvac_fcr_down = fcr_down_frac * hvac_fcr_av
        actual_hvac_up = actual_hvac_fcr_up + afrr_up_frac * hvac_afrr_up
        actual_hvac_down = actual_hvac_fcr_down + afrr_down_frac * hvac_afrr_down
        actual_pv_down = afrr_down_frac * pv_afrr_down

        requested_up = fcr_up_frac * (plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0) + plan.get("hvac_fcr_kw", 0.0)) + afrr_up_frac * (
            plan.get("bess_afrr_up_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0)
        )
        requested_down = fcr_down_frac * (plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0) + plan.get("hvac_fcr_kw", 0.0)) + afrr_down_frac * (
            plan.get("bess_afrr_down_kw", 0.0)
            + plan.get("ev_afrr_down_kw", 0.0)
            + plan.get("hvac_afrr_down_kw", 0.0)
            + plan.get("pv_afrr_down_kw", 0.0)
        )
        delivered_up = actual_bess_up + actual_ev_up + actual_hvac_up
        delivered_down = actual_bess_down + actual_ev_down + actual_hvac_down + actual_pv_down

        state["bess_soc_mwh"] += dt_h / 1000.0 * (eta_c * actual_bess_down - actual_bess_up / eta_d)
        state["ev_soc_delta_mwh"] += dt_h / 1000.0 * (eta_c * actual_ev_down - actual_ev_up / eta_d)
        state["temp_delta_c"] = temp_alpha * state["temp_delta_c"] + temp_beta * (
            (actual_hvac_down - actual_hvac_up) / max(fleet_meta["n_hvac"], 1)
        )

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
                "baseline_net_load_kw": row["net_load_baseline_kw"],
                "optimized_net_load_kw": row["net_load_baseline_kw"] - delivered_up + delivered_down,
                "frequency_hz": row["frequency_hz"],
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
                "bess_soc_mwh": state["bess_soc_mwh"],
                "ev_soc_mwh": row["ev_soc_ref_mwh"] + state["ev_soc_delta_mwh"],
                "indoor_temp_c": row["indoor_temp_c_ref"] + state["temp_delta_c"],
                "fcr_accuracy_ratio": delivered_up / requested_up if requested_up > 1 else np.nan,
                "down_accuracy_ratio": delivered_down / requested_down if requested_down > 1 else np.nan,
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


def plot_sankey(summary: Dict[str, float]) -> go.Figure:
    labels = ["Homes", "EV", "BESS", "PV", "HVAC", "Aggregator", "FCR-N", "aFRR Up", "aFRR Down"]
    values = [summary["n_ev"], summary["n_bess"], summary["n_pv"], summary["n_hvac"]]
    source = [0, 0, 0, 0, 1, 2, 4, 3, 5, 5, 5]
    target = [1, 2, 3, 4, 5, 5, 5, 5, 6, 7, 8]
    flow = [
        values[0],
        values[1],
        values[2],
        values[3],
        values[0] * 0.75,
        values[1] * 0.95,
        values[3] * 0.8,
        values[2] * 0.85,
        max(summary["peak_fast_mw"], 0.01),
        max(summary["peak_total_up_mw"], 0.01),
        max(summary["peak_total_down_mw"], 0.01),
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=28,
                    thickness=18,
                    label=labels,
                    color=["#2E86DE", "#74C0FC", "#FF6B6B", "#FFB3B3", "#4ECDC4", "#8CE99A", "#95A5A6", "#F783AC", "#B197FC"],
                    x=[0.02, 0.33, 0.33, 0.33, 0.33, 0.67, 0.94, 0.94, 0.94],
                    y=[0.50, 0.70, 0.88, 0.52, 0.20, 0.50, 0.54, 0.40, 0.68],
                    line=dict(color="rgba(24,49,83,0.30)", width=1),
                ),
                textfont=dict(size=22, color="#183153"),
                link=dict(
                    source=source,
                    target=target,
                    value=flow,
                    color=[
                        "rgba(170,170,170,0.55)",
                        "rgba(170,170,170,0.55)",
                        "rgba(170,170,170,0.55)",
                        "rgba(170,170,170,0.55)",
                        "rgba(123,97,255,0.18)",
                        "rgba(123,97,255,0.18)",
                        "rgba(255,146,43,0.22)",
                        "rgba(47,158,68,0.20)",
                        "rgba(149,39,204,0.16)",
                        "rgba(220,53,69,0.24)",
                        "rgba(102,126,234,0.24)",
                    ],
                ),
            )
        ]
    )
    fig.update_layout(height=470, margin=dict(l=15, r=40, t=20, b=10), font=dict(size=18))
    return fig


def availability_heatmap(df: pd.DataFrame, column: str, title: str, color_scale: str) -> go.Figure:
    heat = df.copy()
    heat["day"] = heat.index.strftime("%Y-%m-%d")
    heat["hour"] = heat.index.hour
    pivot = heat.pivot_table(index="day", columns="hour", values=column, aggfunc="mean")
    fig = go.Figure(
        go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=color_scale, colorbar={"title": column})
    )
    fig.update_layout(height=320, title=title, margin=dict(l=20, r=20, t=45, b=15))
    fig.update_traces(colorbar={"title": {"text": column, "font": {"size": 16}}, "tickfont": {"size": 14}})
    return fig


def bar_comparison(summary: Dict[str, float], title: str) -> go.Figure:
    frame = pd.DataFrame({"Category": list(summary.keys()), "Value": list(summary.values())})
    fig = px.bar(frame, x="Category", y="Value", color="Category", text_auto=".1f", title=title)
    fig.update_layout(showlegend=False, height=330, margin=dict(l=20, r=20, t=55, b=20))
    fig.update_traces(textfont_size=16)
    return fig


def line_dual_axis(result_df: pd.DataFrame, template: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df["optimized_net_load_kw"] / 1000.0,
            name="Optimized net load (MW)",
            line={"color": RESOURCE_COLORS["Net"], "width": 3},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df["baseline_net_load_kw"] / 1000.0,
            name="Baseline net load (MW)",
            line={"color": RESOURCE_COLORS["Base"], "dash": "dot"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df["frequency_hz"],
            name="Frequency (Hz)",
            line={"color": RESOURCE_COLORS["Frequency"]},
        ),
        secondary_y=True,
    )
    fig.update_layout(template=template, height=420, margin=dict(l=20, r=20, t=30, b=10))
    fig.update_yaxes(title_text="Power (MW)", secondary_y=False)
    fig.update_yaxes(title_text="Frequency (Hz)", secondary_y=True)
    return fig


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
            result = run_mpc_controller(df, models, fleet_meta, market_mode, resource_mode, horizon_hours, dispatch_hours, penalty_weights)
            records.append({"price_multiplier": price_mult, "bess_pen": bess_pen, "revenue_eur": result["summary"].get("total_revenue_eur", 0.0)})
            if progress_callback:
                progress_callback(case_idx, total_cases, f"Finished scenario {case_idx}/{total_cases}")
    return pd.DataFrame(records)


def main() -> None:
    theme_mode = st.sidebar.radio("Dashboard theme", ["Dark", "Light"], horizontal=True, help="Applies to charts and dashboard styling.")
    inject_css(theme_mode)
    template = PLOTLY_TEMPLATE[theme_mode]

    st.sidebar.header("Portfolio Setup")
    start_date = st.sidebar.date_input("Simulation start", value=pd.Timestamp("2026-01-15"))
    days = st.sidebar.slider("Simulation length (days)", 1, 365, 7)
    freq_minutes = st.sidebar.slider("Simulation timestep (minutes)", 1, 15, 15, 1)
    n_homes = st.sidebar.slider("Aggregated homes", 500, 5000, 1500, step=100)
    ev_pen = st.sidebar.slider("EV penetration", 0.0, 1.0, 0.30, 0.05)
    bess_pen = st.sidebar.slider("BESS penetration", 0.0, 1.0, 0.22, 0.05)
    pv_pen = st.sidebar.slider("PV penetration", 0.0, 1.0, 0.45, 0.05)
    hvac_pen = st.sidebar.slider("Smart HVAC penetration", 0.0, 1.0, 0.82, 0.05)
    hvac_mode = st.sidebar.selectbox("HVAC technology", HVAC_MODES, index=1)
    hvac_response_default = default_hvac_response_seconds(hvac_mode)
    hvac_response_s = st.sidebar.slider(
        "Inverter HVAC electrical response (s)" if hvac_mode == HVAC_MODES[1] else "HVAC electrical response (s)",
        2.0,
        180.0,
        float(hvac_response_default),
        1.0,
    )

    st.sidebar.header("Synthetic Finland Context")
    cloudiness = st.sidebar.slider("Cloudiness / variability", 0.1, 1.0, 0.55, 0.05)
    climate_shift_c = st.sidebar.slider("Temperature anomaly (°C)", -8.0, 8.0, 0.0, 0.5)
    solar_scale = st.sidebar.slider("Solar resource multiplier", 0.5, 1.5, 1.0, 0.05)
    scarcity = st.sidebar.slider("Market scarcity / price stress", 0.6, 1.8, 1.0, 0.05)
    setpoint_c = st.sidebar.slider("HVAC setpoint (°C)", 19.0, 23.0, 21.0, 0.5)
    comfort_band_c = st.sidebar.slider("Comfort band (±°C)", 0.5, 2.5, 1.0, 0.1)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    with st.spinner("Generating synthetic households, weather, and balancing market conditions..."):
        bundle = generate_synthetic_portfolio(
            start_date=str(start_date),
            days=days,
            n_homes=n_homes,
            ev_pen=ev_pen,
            bess_pen=bess_pen,
            pv_pen=pv_pen,
            hvac_pen=hvac_pen,
            cloudiness=cloudiness,
            climate_shift_c=climate_shift_c,
            solar_scale=solar_scale,
            seed=int(seed),
            setpoint_c=setpoint_c,
            comfort_band_c=comfort_band_c,
            scarcity=scarcity,
            freq_minutes=int(freq_minutes),
            hvac_mode=hvac_mode,
        )
    df = bundle["data"]
    preview_4s = bundle["preview_4s"]
    summary = bundle["summary"]

    tabs = st.tabs(
        [
            "Home",
            "Synthetic Data Explorer",
            "Response Models",
            "Forecasting Lab",
            "MPC Optimizer & Market Participation",
            "Simulation & Impact Visualizations",
            "Advanced / Export",
        ]
    )

    with tabs[0]:
        st.title("FlexiHome: Aggregated Residential Flexibility for Fingrid Balancing Markets")
        left, right = st.columns([1.2, 1.0], gap="large")
        with left:
            st.markdown(
                """
                <div class="flexi-card">
                    <h4>What this dashboard teaches</h4>
                    <p>
                        Residential flexibility becomes market-relevant only when it is aggregated, forecasted well,
                        and dispatched with product-aware control. Fast assets like <b>BESS</b> and <b>EVs</b> dominate FCR-N,
                        while <b>thermal loads</b> and <b>PV curtailment</b> add depth for aFRR and hybrid reserve strategies.
                    </p>
                    <span class="flexi-badge">FCR-N: symmetric, local, seconds</span>
                    <span class="flexi-badge">aFRR: signal-based, 4-second control, 5-minute full activation</span>
                    <span class="flexi-badge">Aggregation unlocks 0.1 MW and 1 MW thresholds</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(plot_sankey(summary), use_container_width=True)
        with right:
            c1, c2 = st.columns(2)
            c1.metric("Fast reserve headroom", f"{summary['peak_fast_mw']:.2f} MW")
            c2.metric("Total upward flexibility", f"{summary['peak_total_up_mw']:.2f} MW")
            c1.metric("Homes needed for 1 MW aFRR", f"{summary['homes_for_1mw_total']:,}")
            c2.metric("Homes needed for fast 1 MW", f"{summary['homes_for_1mw_fast']:,}")
            c1.metric("PV fleet size", f"{summary['pv_capacity_mw']:.2f} MW")
            c2.metric("Accessible storage energy", f"{summary['bess_energy_mwh'] + summary['ev_energy_mwh']:.2f} MWh")
            st.markdown(
                """
                <div class="flexi-card">
                    <h4>Fingrid checkpoints embedded in the app</h4>
                    <p class="small-note">
                        The optimizer and compliance view track minimum bid sizes, response speed, expected accuracy,
                        storage endurance, and the presence of an explicit baseline. Capacity and activation prices
                        are synthetic but calibrated to typical-looking Finnish balancing market ranges.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Educational notes and equations", expanded=False):
                st.markdown(
                    r"""
                    - **FCR-N activation**: full response at $\pm 0.1$ Hz from nominal frequency.
                    - **aFRR**: a centralized signal updated every 4 seconds, with 30-second start and full activation within 5 minutes.
                    - **Thermal response transfer function**:
                      $$
                      G(s) = \frac{K}{\tau_1 \tau_2 s^2 + (\tau_1 + \tau_2)s + 1}
                      $$
                    - **MPC concept**: at each time step, forecast the next horizon, optimize reserve bids and state trajectories, apply the first move, then re-optimize.
                    """
                )

    with tabs[1]:
        st.subheader("Synthetic Data Explorer")
        st.caption(f"All data is generated locally at a {freq_minutes}-minute resolution. Adjust sidebar parameters to regenerate the portfolio.")
        export_col, explain_col = st.columns([1, 1.3])
        export_col.download_button(
            "Download synthetic data as CSV",
            data=df.to_csv().encode("utf-8"),
            file_name="flexihome_synthetic_data.csv",
            mime="text/csv",
        )
        explain_col.info(
            "The synthetic engine mixes daily demand shape, seasonal Helsinki-style weather, PV irradiance, EV parking uncertainty, BESS headroom, and thermal comfort dynamics."
        )

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df.index, y=df["base_load_kw"] / 1000.0, name="Base load", line={"color": RESOURCE_COLORS["Base"]}))
        fig_ts.add_trace(go.Scatter(x=df.index, y=df["pv_available_kw"] / 1000.0, name="PV available", line={"color": RESOURCE_COLORS["PV"]}))
        fig_ts.add_trace(go.Scatter(x=df.index, y=df["ev_baseline_kw"] / 1000.0, name="EV baseline charging", line={"color": RESOURCE_COLORS["EV"]}))
        fig_ts.add_trace(go.Scatter(x=df.index, y=df["hvac_baseline_kw"] / 1000.0, name="HVAC baseline", line={"color": RESOURCE_COLORS["HVAC"]}))
        fig_ts.add_trace(go.Scatter(x=df.index, y=df["net_load_baseline_kw"] / 1000.0, name="Net baseline load", line={"color": RESOURCE_COLORS["Net"], "width": 3}))
        fig_ts.update_layout(yaxis_title="MW")
        st.plotly_chart(
            apply_chart_style(fig_ts, template, height=430, title="Synthetic portfolio baseline and resource traces"),
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            flex_fig = go.Figure()
            flex_fig.add_trace(go.Scatter(x=df.index, y=df["flex_up_kw"] / 1000.0, name="Upward flexibility", stackgroup="one", line={"color": RESOURCE_COLORS["Net"]}))
            flex_fig.add_trace(go.Scatter(x=df.index, y=df["flex_down_kw"] / 1000.0, name="Downward flexibility", stackgroup="two", line={"color": RESOURCE_COLORS["aFRR"]}))
            flex_fig.update_layout(yaxis_title="MW")
            st.plotly_chart(apply_chart_style(flex_fig, template, height=360, title="Aggregate flexible headroom"), use_container_width=True)
        with col_b:
            hist_df = pd.DataFrame(
                {
                    "Fast symmetric flexibility (kW)": df["fast_sym_kw"],
                    "Upward flexibility (kW)": df["flex_up_kw"],
                    "Downward flexibility (kW)": df["flex_down_kw"],
                }
            ).melt(var_name="Series", value_name="Value")
            hist_fig = px.histogram(hist_df, x="Value", color="Series", barmode="overlay", nbins=40, template=template, title="Distribution of flexible capacity")
            st.plotly_chart(apply_chart_style(hist_fig, template, height=360), use_container_width=True)

        heat1, heat2 = st.columns(2)
        heat1.plotly_chart(apply_chart_style(availability_heatmap(df, "fast_sym_kw", "Fast-response availability heatmap", "Viridis"), template, height=320), use_container_width=True)
        heat2.plotly_chart(apply_chart_style(availability_heatmap(df, "hvac_up_kw", "Thermal upward flexibility heatmap", "YlOrBr"), template, height=320), use_container_width=True)

    with tabs[2]:
        st.subheader("Response Models")
        c1, c2, c3, c4 = st.columns(4)
        tau_bess_s = c1.slider("BESS time constant (s)", 1.0, 5.0, 2.0, 0.5)
        tau_ev_s = c2.slider("EV time constant (s)", 1.0, 8.0, 4.0, 0.5)
        tau_hvac1_min = c3.slider("HVAC fast thermal mode τ1 (min)", 5.0, 15.0, 8.0, 1.0)
        tau_hvac2_min = c4.slider("HVAC slow thermal mode τ2 (min)", 30.0, 60.0, 45.0, 1.0)
        st.caption(
            f"HVAC is currently modeled as {hvac_mode}. "
            + ("Inverter mode uses the electrical response for the fast HVAC trace." if hvac_mode == HVAC_MODES[1] else "Conventional mode uses the slower thermal second-order response.")
        )
        profiles = response_model_profiles(
            tau_bess_s,
            tau_ev_s,
            tau_hvac1_min * 60.0,
            tau_hvac2_min * 60.0,
            hvac_mode,
            hvac_response_s,
            duration_s=3600,
        )

        step_fig = go.Figure()
        for resource, data in profiles.items():
            color = RESOURCE_COLORS["HVAC"] if "HVAC" in resource else RESOURCE_COLORS.get(resource.replace(" Curtailment", ""), RESOURCE_COLORS["Net"])
            step_fig.add_trace(go.Scatter(x=data["t"], y=data["y"], name=resource, line={"color": color, "width": 3 if resource in {"BESS", "HVAC"} else 2}))
        step_fig.update_layout(xaxis_title="Seconds", yaxis_title="Delivered fraction")
        st.plotly_chart(apply_chart_style(step_fig, template, height=420, title="Normalized step response to a reserve activation request"), use_container_width=True)

        anim_records = []
        for resource, data in profiles.items():
            for i, (tt, yy) in enumerate(zip(data["t"][::12], data["y"][::12])):
                anim_records.append({"Resource": resource, "time": float(tt), "response": float(yy), "frame": i})
        anim_df = pd.DataFrame(anim_records)
        anim_fig = px.scatter(
            anim_df,
            x="time",
            y="response",
            color="Resource",
            animation_frame="frame",
            range_y=[0, 1.05],
            range_x=[0, float(max(profiles["HVAC"]["t"]))],
            template=template,
            title="Animated reserve activation playback",
        )
        st.plotly_chart(
            apply_chart_style(anim_fig, template, height=420, title="Animated reserve activation playback"),
            use_container_width=True,
        )

        bode_cols = st.columns(2)
        with bode_cols[0]:
            bode_fig = go.Figure()
            for resource in ["BESS", "EV", "HVAC"]:
                bode = bode_points(resource, tau_bess_s, tau_ev_s, tau_hvac1_min * 60.0, tau_hvac2_min * 60.0, hvac_mode, hvac_response_s)
                bode_fig.add_trace(go.Scatter(x=bode["omega"], y=bode["mag_db"], name=resource))
            bode_fig.update_layout(xaxis_type="log", xaxis_title="rad/s", yaxis_title="dB")
            st.plotly_chart(apply_chart_style(bode_fig, template, height=360, title="Bode magnitude"), use_container_width=True)
        with bode_cols[1]:
            phase_fig = go.Figure()
            for resource in ["BESS", "EV", "HVAC"]:
                bode = bode_points(resource, tau_bess_s, tau_ev_s, tau_hvac1_min * 60.0, tau_hvac2_min * 60.0, hvac_mode, hvac_response_s)
                phase_fig.add_trace(go.Scatter(x=bode["omega"], y=bode["phase_deg"], name=resource))
            phase_fig.update_layout(xaxis_type="log", xaxis_title="rad/s", yaxis_title="Degrees")
            st.plotly_chart(apply_chart_style(phase_fig, template, height=360, title="Bode phase"), use_container_width=True)

        st.info(
            "Interpretation: BESS and EV fleets can closely track the required reserve trajectory in seconds. HVAC thermal flexibility adds volume, but its slower second-order behavior makes it more suitable for aFRR or hybrid strategies than for pure FCR-N."
        )

    with tabs[3]:
        st.subheader("Forecasting Lab")
        target_options = [
            "net_load_baseline_kw",
            "pv_available_kw",
            "fcrn_capacity_eur_per_mw_h",
            "afrr_up_capacity_eur_per_mw_h",
            "afrr_down_capacity_eur_per_mw_h",
            "fcr_signed_act",
            "afrr_up_act_frac",
            "afrr_down_act_frac",
        ]
        predictor_pool = [
            "temp_out_c",
            "irradiance_wm2",
            "base_load_kw",
            "pv_available_kw",
            "net_load_baseline_kw",
            "ev_connected_frac",
            "ev_baseline_kw",
            "bess_soc_ref_mwh",
            "hvac_baseline_kw",
            "fcrn_capacity_eur_per_mw_h",
            "afrr_up_capacity_eur_per_mw_h",
            "afrr_down_capacity_eur_per_mw_h",
            "afrr_up_act_frac",
            "afrr_down_act_frac",
            "fcr_signed_act",
        ]
        col1, col2, col3, col4 = st.columns([1.2, 1.5, 0.8, 0.8])
        target = col1.selectbox("Target", target_options, index=0)
        predictors = col2.multiselect(
            "Predictors",
            predictor_pool,
            default=["temp_out_c", "irradiance_wm2", "base_load_kw", "pv_available_kw", "net_load_baseline_kw", "fcr_signed_act"],
        )
        lags = col3.slider("Lag depth", 1, 8, 4)
        max_horizon_steps = max(4, min(96, int((24 * 60) / max(freq_minutes, 1))))
        horizon_steps = col4.slider(f"Forecast horizon ({freq_minutes} min steps)", 1, max_horizon_steps, min(4, max_horizon_steps))

        if st.button("Train XGBoost model", type="primary"):
            with st.spinner("Training the forecaster..."):
                spec, comparison = train_forecaster(df, target, predictors, lags, horizon_steps, int(seed))
            st.session_state["last_forecast_spec"] = spec
            st.session_state["last_forecast_comparison"] = comparison

        if "last_forecast_spec" in st.session_state:
            spec: ForecastSpec = st.session_state["last_forecast_spec"]
            comparison = st.session_state["last_forecast_comparison"]
            m1, m2 = st.columns(2)
            m1.metric("MAE", f"{spec.metrics['mae']:.2f}")
            m2.metric("RMSE", f"{spec.metrics['rmse']:.2f}")

            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=comparison.index, y=comparison["actual"], name="Actual", line={"color": RESOURCE_COLORS["Base"]}))
            pred_fig.add_trace(go.Scatter(x=comparison.index, y=comparison["prediction"], name="Prediction", line={"color": RESOURCE_COLORS["Net"], "dash": "dot"}))
            st.plotly_chart(apply_chart_style(pred_fig, template, height=380, title=f"Forecast performance: {spec.target}"), use_container_width=True)

            importance = pd.DataFrame({"feature": spec.model.feature_names_in_, "importance": spec.model.feature_importances_}).sort_values(
                "importance", ascending=False
            )
            imp_fig = px.bar(importance.head(12), x="importance", y="feature", orientation="h", template=template, title="Top feature importances")
            st.plotly_chart(apply_chart_style(imp_fig, template, height=380, title="Top feature importances"), use_container_width=True)

            model_bytes = serialize_forecast_spec(spec)
            st.download_button("Download trained model", data=model_bytes, file_name=f"{spec.target}_xgboost.pkl", mime="application/octet-stream")
            if spec.target in SUPPORTED_MPC_TARGETS and spec.horizon_steps == 1:
                if st.button("Use this model inside the MPC loop"):
                    custom = st.session_state.get("custom_mpc_models", {})
                    custom[spec.target] = spec
                    st.session_state["custom_mpc_models"] = custom
                    st.success(f"{spec.target} is now promoted into the MPC forecasting registry.")
            else:
                st.caption("To promote a model into the MPC loop, train a 1-step model for one of the MPC targets.")

    with tabs[4]:
        st.subheader("MPC Optimizer & Market Participation")
        st.caption(f"The synthetic portfolio, forecasting loop, and MPC controller are all running at a {freq_minutes}-minute interval in this scenario.")
        opt1, opt2, opt3, opt4 = st.columns(4)
        market_mode = opt1.selectbox("Market product", ["Combined", "FCR-N", "aFRR"], help="Combined allows the MPC to split the portfolio across both products.")
        resource_mode = opt2.selectbox("Resource strategy", ["Hybrid portfolio", "Fast only"], help="Fast only uses BESS + EV. Hybrid also activates HVAC and PV for aFRR.")
        horizon_hours = opt3.slider("MPC horizon (hours)", 4, 48, 24)
        dispatch_hours = opt4.slider("Dispatch simulation horizon (hours)", 6, min(72, days * 24), min(24, days * 24))

        w1, w2, w3 = st.columns(3)
        degradation_w = w1.slider("Degradation weight", 1.0, 60.0, 18.0, 1.0)
        comfort_w = w2.slider("Comfort violation weight", 10.0, 300.0, 120.0, 5.0)
        departure_w = w3.slider("EV departure shortfall weight", 10.0, 300.0, 160.0, 5.0)

        st.caption(
            f"Current setup: about {int(horizon_hours / (freq_minutes / 60.0))} forecast/control steps per horizon and "
            f"{int(dispatch_hours / (freq_minutes / 60.0))} MPC intervals over the full dispatch simulation."
        )
        run_mpc = st.button("Run MPC dispatch", type="primary")
        mpc_progress_bar = st.progress(0.0)
        mpc_progress_text = st.empty()
        if run_mpc:
            with st.spinner("Training default MPC forecasters and running receding-horizon optimization..."):
                mpc_tracker = make_progress_tracker(mpc_progress_bar, mpc_progress_text, "MPC dispatch")
                mpc_tracker(1, 100, "Preparing default forecasting models")
                models = ensure_default_models(df, int(seed))
                mpc_tracker(8, 100, "Forecasting models ready")
                fleet_meta = {
                    "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
                    "setpoint_c": setpoint_c,
                    "comfort_band_c": comfort_band_c,
                    "n_hvac": summary["n_hvac"],
                    "hvac_mode": hvac_mode,
                    "hvac_response_s": hvac_response_s,
                }
                st.session_state["mpc_config"] = {
                    "market_mode": market_mode,
                    "resource_mode": resource_mode,
                    "horizon_hours": horizon_hours,
                    "dispatch_hours": dispatch_hours,
                }
                st.session_state["mpc_result"] = run_mpc_controller(
                    df=df,
                    models=models,
                    fleet_meta=fleet_meta,
                    market_mode=market_mode,
                    resource_mode=resource_mode,
                    horizon_hours=horizon_hours,
                    dispatch_hours=dispatch_hours,
                    penalty_weights={"degradation": degradation_w, "comfort": comfort_w, "departure": departure_w},
                    progress_callback=lambda current, total, message: mpc_tracker(
                        8 + int(round((current / max(total, 1)) * 92)),
                        100,
                        message,
                    ),
                )
                mpc_tracker(100, 100, "MPC dispatch finished")
        result = st.session_state.get("mpc_result", {"history": pd.DataFrame(), "summary": {}, "compliance": {}, "first_schedule": pd.DataFrame()})
        result_df = result["history"]

        if result_df.empty:
            st.info("Configure the scenario and click Run MPC dispatch. The optimizer will not run automatically.")
        else:
            s = result["summary"]
            c = result["compliance"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total revenue", fmt_money(s["total_revenue_eur"]))
            m2.metric("Delivered up energy", f"{s['delivered_up_mwh']:.2f} MWh")
            m3.metric("Requirement score", f"{s['requirement_score_pct']:.0f}%")
            m4.metric("Comfort violations", f"{s['comfort_violations_h']:.1f} h")

            left, right = st.columns([1.45, 1.0])
            with left:
                dispatch_fig = go.Figure()
                dispatch_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["fcr_bid_kw"] / 1000.0, name="FCR-N bid", stackgroup="one"))
                dispatch_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["afrr_up_bid_kw"] / 1000.0, name="aFRR up bid", stackgroup="one"))
                dispatch_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["afrr_down_bid_kw"] / 1000.0, name="aFRR down bid", stackgroup="two"))
                dispatch_fig.update_layout(yaxis_title="MW")
                st.plotly_chart(apply_chart_style(dispatch_fig, template, height=400, title="MPC reserve schedule"), use_container_width=True)
            with right:
                compliance_cols = st.columns(2)
                compliance_cols[0].plotly_chart(apply_chart_style(indicator_figure(float(result_df["fcr_bid_kw"].mean()), "Average FCR bid (kW)", 100.0), template, height=240), use_container_width=True)
                compliance_cols[1].plotly_chart(apply_chart_style(indicator_figure(float(np.maximum(result_df["afrr_up_bid_kw"], result_df["afrr_down_bid_kw"]).mean()), "Average aFRR bid (kW)", 1000.0), template, height=240), use_container_width=True)
                st.markdown(
                    f"""
                    <div class="flexi-card">
                        <h4>Compliance snapshot</h4>
                        <p class="small-note">
                            FCR response estimate: <b>{c['fcr_response_s']:.1f} s</b><br/>
                            aFRR response estimate: <b>{c['afrr_response_s']:.0f} s</b><br/>
                            Mean delivered/requested ratio: <b>{c['mean_accuracy']:.2f}</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            first_schedule = result["first_schedule"].copy()
            if not first_schedule.empty:
                first_schedule.index.name = "timestamp"
                st.caption("First MPC horizon snapshot. This is the forward-looking schedule before only the first control move is applied.")
                st.dataframe(styled_dataframe(first_schedule.head(12).round(2)), use_container_width=True, height=260)

            rule_items = [
                ("FCR-N minimum 0.1 MW", c["fcr_min_bid_ok"]),
                ("aFRR minimum 1 MW", c["afrr_min_bid_ok"]),
                ("FCR-N fast response", c["fcr_response_ok"]),
                ("aFRR 5-minute full activation", c["afrr_response_ok"]),
                ("aFRR 90-110% accuracy", c["accuracy_ok"]),
                ("Storage 1 h endurance per direction", c["storage_endurance_ok"]),
                ("Baseline methodology available", c["baseline_method_ok"]),
            ]
            rule_df = pd.DataFrame(rule_items, columns=["Rule", "Pass"]).assign(Status=lambda x: np.where(x["Pass"], "Pass", "Needs attention"))
            st.dataframe(styled_dataframe(rule_df), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("Simulation & Impact Visualizations")
        result = st.session_state.get("mpc_result")
        if not result or result["history"].empty:
            st.info("Run the MPC tab to populate the simulation results.")
        else:
            result_df = result["history"]
            s = result["summary"]
            ts_left, ts_right = st.columns([1.45, 1.0])
            ts_left.plotly_chart(
                apply_chart_style(
                    line_dual_axis(result_df, template),
                    template,
                    height=420,
                    title="Optimized load, baseline, and frequency",
                ),
                use_container_width=True,
            )

            contribution_fig = go.Figure()
            contribution_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["bess_up_kw"] / 1000.0, name="BESS", stackgroup="one", line={"color": RESOURCE_COLORS["BESS"]}))
            contribution_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["ev_up_kw"] / 1000.0, name="EV", stackgroup="one", line={"color": RESOURCE_COLORS["EV"]}))
            contribution_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["hvac_up_kw"] / 1000.0, name="HVAC", stackgroup="one", line={"color": RESOURCE_COLORS["HVAC"]}))
            contribution_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["pv_down_kw"] / 1000.0, name="PV down", stackgroup="two", line={"color": RESOURCE_COLORS["PV"]}))
            contribution_fig.update_layout(yaxis_title="MW")
            ts_right.plotly_chart(apply_chart_style(contribution_fig, template, height=420, title="Resource contribution stack"), use_container_width=True)

            row1 = st.columns(3)
            row1[0].metric("Capacity revenue", fmt_money(s["capacity_revenue_eur"]))
            row1[1].metric("Activation revenue", fmt_money(s["activation_revenue_eur"]))
            row1[2].metric("CO2 avoided (proxy)", f"{s['co2_avoided_kg']:.0f} kg")

            charts = st.columns(3)
            revenue_fig = bar_comparison(s["resource_revenue"], "Revenue by resource")
            fast_slow_fig = bar_comparison(s["fast_vs_slow"], "Fast vs slow contribution")
            energy_fig = bar_comparison(
                {
                    "Up energy (MWh)": s["delivered_up_mwh"],
                    "Down energy (MWh)": s["delivered_down_mwh"],
                    "Comfort penalty (€)": s["comfort_penalty_eur"],
                },
                "Impact summary",
            )
            charts[0].plotly_chart(apply_chart_style(revenue_fig, template, height=330), use_container_width=True)
            charts[1].plotly_chart(apply_chart_style(fast_slow_fig, template, height=330), use_container_width=True)
            charts[2].plotly_chart(apply_chart_style(energy_fig, template, height=330), use_container_width=True)

            heat_cols = st.columns(2)
            bid_success = result_df.assign(success=(result_df["delivered_up_kw"] >= 0.9 * result_df["requested_up_kw"]).astype(int))
            heat_cols[0].plotly_chart(apply_chart_style(availability_heatmap(result_df, "fcr_bid_kw", "FCR/aFRR bid heatmap", "PuBu"), template, height=320), use_container_width=True)
            heat_cols[1].plotly_chart(apply_chart_style(availability_heatmap(bid_success, "success", "Bid success heatmap", "Greens"), template, height=320), use_container_width=True)

            st.markdown("### What-if scenario playground")
            wf1, wf2, wf3 = st.columns(3)
            price_multiplier = wf1.slider("Price multiplier", 0.7, 1.5, 1.0, 0.05)
            cold_snap = wf2.slider("Cold snap impact (°C)", 0.0, 10.0, 2.0, 0.5)
            thermal_speed = wf3.slider("Thermal response multiplier", 0.4, 1.4, 1.0, 0.1)
            whatif_progress_bar = st.progress(0.0)
            whatif_progress_text = st.empty()

            if st.button("Re-run what-if comparison"):
                with st.spinner("Running the what-if scenario..."):
                    whatif_tracker = make_progress_tracker(whatif_progress_bar, whatif_progress_text, "What-if scenario")
                    whatif_tracker(2, 100, "Preparing alternative scenario inputs")
                    alt_df = df.copy()
                    alt_df["temp_out_c"] -= cold_snap
                    alt_df["hvac_up_kw"] *= thermal_speed
                    alt_df["hvac_down_kw"] *= thermal_speed
                    if "hvac_fast_kw" in alt_df.columns:
                        alt_df["hvac_fast_kw"] *= thermal_speed
                    alt_df["fcrn_capacity_eur_per_mw_h"] *= price_multiplier
                    alt_df["afrr_up_capacity_eur_per_mw_h"] *= price_multiplier
                    alt_df["afrr_down_capacity_eur_per_mw_h"] *= price_multiplier
                    alt_df["afrr_up_energy_eur_per_mwh"] *= price_multiplier
                    alt_df["afrr_down_energy_eur_per_mwh"] *= price_multiplier
                    whatif_tracker(12, 100, "Refreshing forecasting models for the what-if case")
                    models = ensure_default_models(alt_df, int(seed))
                    whatif_tracker(18, 100, "Forecasting models ready")
                    alt_result = run_mpc_controller(
                        alt_df,
                        models,
                        {
                            "bess_energy_cap_mwh": float(alt_df["bess_energy_cap_mwh"].iloc[0]),
                            "setpoint_c": setpoint_c,
                            "comfort_band_c": comfort_band_c,
                            "n_hvac": summary["n_hvac"],
                            "hvac_mode": hvac_mode,
                            "hvac_response_s": hvac_response_s,
                        },
                        st.session_state["mpc_config"]["market_mode"],
                        st.session_state["mpc_config"]["resource_mode"],
                        st.session_state["mpc_config"]["horizon_hours"],
                        st.session_state["mpc_config"]["dispatch_hours"],
                        {"degradation": degradation_w, "comfort": comfort_w, "departure": departure_w},
                        progress_callback=lambda current, total, message: whatif_tracker(
                            18 + int(round((current / max(total, 1)) * 76)),
                            100,
                            message,
                        ),
                    )
                    compare_df = pd.DataFrame(
                        {
                            "Scenario": ["Base", "What-if"],
                            "Revenue (€)": [s["total_revenue_eur"], alt_result["summary"].get("total_revenue_eur", 0.0)],
                            "Delivered up (MWh)": [s["delivered_up_mwh"], alt_result["summary"].get("delivered_up_mwh", 0.0)],
                            "Requirement score (%)": [s["requirement_score_pct"], alt_result["summary"].get("requirement_score_pct", 0.0)],
                        }
                    )
                    compare_fig = px.bar(compare_df.melt(id_vars="Scenario", var_name="Metric", value_name="Value"), x="Metric", y="Value", color="Scenario", barmode="group", template=template)
                    whatif_tracker(97, 100, "Rendering what-if comparison")
                    st.plotly_chart(apply_chart_style(compare_fig, template, height=360, title="Base vs what-if comparison"), use_container_width=True)
                    whatif_tracker(100, 100, "What-if scenario finished")

    with tabs[6]:
        st.subheader("Advanced / Export")
        config = st.session_state.get("mpc_config", {"market_mode": "Combined", "resource_mode": "Hybrid portfolio", "horizon_hours": 24, "dispatch_hours": 24})
        result = st.session_state.get("mpc_result", {"history": pd.DataFrame(), "summary": {}, "compliance": {}})

        export_left, export_right = st.columns([1, 1])
        export_left.download_button("Download simulation bundle (JSON)", data=json.dumps(summary, indent=2).encode("utf-8"), file_name="flexihome_summary.json", mime="application/json")
        export_right.download_button("Download high-resolution preview (CSV)", data=preview_4s.to_csv().encode("utf-8"), file_name="flexihome_4s_preview.csv", mime="text/csv")

        if result.get("summary"):
            pdf_bytes = make_pdf_summary(result["summary"], result["compliance"], config)
            st.download_button("Export MPC formulation summary as PDF", data=pdf_bytes, file_name="flexihome_mpc_summary.pdf", mime="application/pdf")

        st.markdown("### Sensitivity analysis")
        st.caption("This runs a compact 1-day scan over BESS penetration and market scarcity/price multipliers.")
        sensitivity_progress_bar = st.progress(0.0)
        sensitivity_progress_text = st.empty()
        if st.button("Run sensitivity scan"):
            with st.spinner("Scanning scenario space..."):
                sensitivity_tracker = make_progress_tracker(sensitivity_progress_bar, sensitivity_progress_text, "Sensitivity scan")
                sensitivity_tracker(1, 100, "Preparing scan grid")
                sens_df = sensitivity_scan(
                    base_params={
                        "start_date": str(start_date),
                        "n_homes": n_homes,
                        "ev_pen": ev_pen,
                        "bess_pen": bess_pen,
                        "pv_pen": pv_pen,
                        "hvac_pen": hvac_pen,
                        "cloudiness": cloudiness,
                        "climate_shift_c": climate_shift_c,
                        "solar_scale": solar_scale,
                        "seed": int(seed),
                        "setpoint_c": setpoint_c,
                        "comfort_band_c": comfort_band_c,
                        "scarcity": scarcity,
                        "freq_minutes": int(freq_minutes),
                        "hvac_mode": hvac_mode,
                        "hvac_response_s": hvac_response_s,
                    },
                    price_multipliers=[0.8, 1.0, 1.2],
                    bess_pen_options=[max(0.05, bess_pen - 0.10), bess_pen, min(0.9, bess_pen + 0.10)],
                    market_mode=config["market_mode"],
                    resource_mode=config["resource_mode"],
                    horizon_hours=min(config["horizon_hours"], 24),
                    dispatch_hours=min(config["dispatch_hours"], 24),
                    penalty_weights={"degradation": 18.0, "comfort": 120.0, "departure": 160.0},
                    progress_callback=lambda current, total, message: sensitivity_tracker(
                        5 + int(round((current / max(total, 1)) * 90)),
                        100,
                        message,
                    ),
                )
                heat = sens_df.pivot(index="bess_pen", columns="price_multiplier", values="revenue_eur")
                heat_fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Turbo", colorbar={"title": "Revenue €"}))
                sensitivity_tracker(97, 100, "Rendering sensitivity outputs")
                st.plotly_chart(apply_chart_style(heat_fig, template, height=380, title="Sensitivity: revenue vs BESS penetration and price multiplier"), use_container_width=True)
                st.dataframe(styled_dataframe(sens_df), use_container_width=True, hide_index=True)
                sensitivity_tracker(100, 100, "Sensitivity scan finished")

        with st.expander("Market rule references embedded in this dashboard"):
            st.markdown(
                f"""
                - **FCR-N**: minimum bid {FINGRID_RULES['FCR-N']['min_bid_kw'] / 1000:.1f} MW, symmetric response, activation band 49.9-50.1 Hz.
                - **aFRR**: minimum bid {FINGRID_RULES['aFRR']['min_bid_kw'] / 1000:.1f} MW, start within {FINGRID_RULES['aFRR']['start_seconds']:.0f} s, full activation within {FINGRID_RULES['aFRR']['full_activation_seconds'] / 60:.0f} min.
                - **Storage endurance heuristic**: at least {FINGRID_RULES['aFRR']['storage_endurance_hours']:.0f} hour of energy per direction for the storage share of an aFRR bid.
                - **Accuracy check**: expected delivery in the {100 * FINGRID_RULES['aFRR']['accuracy_low']:.0f}-{100 * FINGRID_RULES['aFRR']['accuracy_high']:.0f}% band.
                """
            )


if __name__ == "__main__":
    main()
