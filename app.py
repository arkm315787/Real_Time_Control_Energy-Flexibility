"""
Coverly: Aggregated Residential Flexibility for Fingrid Balancing Markets
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
realistic Finland-inspired data, probabilistic forecasting, dynamic response
models, and a receding-horizon MPC-style optimizer built with PuLP.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from html import escape
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.elements.plotly_chart import PlotlyMixin
from plotly.subplots import make_subplots
from flexihome.core.base.registry import get_global_registry
from flexihome.core.engine import (
    FINGRID_RULES,
    HVAC_MODES,
    RISK_POLICY_PRESETS,
    SIMULATION_ONLY_NOTICE,
    SUPPORTED_MPC_TARGETS,
    bode_points,
    default_hvac_response_seconds,
    fmt_kw,
    fmt_money,
    generate_synthetic_portfolio as generate_synthetic_portfolio_core,
    make_pdf_summary,
    response_model_profiles,
    run_lower_mpc_from_upper_result,
    run_mpc_controller,
    sensitivity_scan,
    serialize_forecast_spec,
)
from flexihome.core.market_simulator import MarketSimulator, MarketSimulatorConfig
from flexihome.plugins import DEFAULT_FORECASTER_PLUGIN, DEFAULT_OPTIMIZER_PLUGIN, register_default_plugins
from services.cache_manager import CacheManager
from services.market_service import MarketDataService
from services.vpp_optimizer_service import VPPOptimizerService
from utils.data_sync import merge_market_data

PLUGIN_REGISTRY = register_default_plugins(get_global_registry())

st.set_page_config(
    page_title="Coverly",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def generate_synthetic_portfolio(*args, **kwargs) -> Dict[str, object]:
    return generate_synthetic_portfolio_core(*args, **kwargs)


def generate_portfolio_runtime(*args, use_cache: bool = True, **kwargs) -> Dict[str, object]:
    if use_cache:
        return generate_synthetic_portfolio(*args, **kwargs)
    return generate_synthetic_portfolio_core(*args, **kwargs)


def default_scenario_inputs() -> Dict[str, object]:
    return {
        "start_date": pd.Timestamp("2026-01-15").date(),
        "days": 7,
        "freq_minutes": 15,
        "n_homes": 1500,
        "ev_pen": 0.30,
        "bess_pen": 0.22,
        "pv_pen": 0.45,
        "hvac_pen": 0.82,
        "hvac_mode": HVAC_MODES[1],
        "hvac_response_s": default_hvac_response_seconds(HVAC_MODES[1]),
        "cloudiness": 0.55,
        "climate_shift_c": 0.0,
        "solar_scale": 1.0,
        "scarcity": 1.0,
        "setpoint_c": 21.0,
        "comfort_band_c": 1.0,
        "seed": 42,
        "market_source": "Synthetic",
        "market_lookback_days": 30,
        "entsoe_key": "",
        "fingrid_key": "",
        "persist_market_data": True,
    }


def scenario_portfolio_signature(scenario: Dict[str, object]) -> tuple:
    """Return the applied-scenario fields that require portfolio regeneration."""

    return (
        str(scenario["start_date"]),
        int(scenario["days"]),
        int(scenario["freq_minutes"]),
        int(scenario["n_homes"]),
        round(float(scenario["ev_pen"]), 6),
        round(float(scenario["bess_pen"]), 6),
        round(float(scenario["pv_pen"]), 6),
        round(float(scenario["hvac_pen"]), 6),
        str(scenario["hvac_mode"]),
        round(float(scenario["hvac_response_s"]), 6),
        round(float(scenario["cloudiness"]), 6),
        round(float(scenario["climate_shift_c"]), 6),
        round(float(scenario["solar_scale"]), 6),
        round(float(scenario["scarcity"]), 6),
        round(float(scenario["setpoint_c"]), 6),
        round(float(scenario["comfort_band_c"]), 6),
        int(scenario["seed"]),
        str(scenario["market_source"]),
        int(scenario["market_lookback_days"]),
        str(scenario.get("entsoe_key", "")),
        str(scenario.get("fingrid_key", "")),
        bool(scenario["persist_market_data"]),
    )


def market_data_signature(scenario: Dict[str, object]) -> tuple:
    """Return the applied market-service fields that require a TSO data refresh."""

    return (
        str(scenario.get("market_source", "Synthetic")),
        int(scenario.get("market_lookback_days", 90)),
        bool(str(scenario.get("entsoe_key", "")).strip()),
        bool(str(scenario.get("fingrid_key", "")).strip()),
        pd.Timestamp.utcnow().date().isoformat(),
    )


def portfolio_bundle_is_current(session_state: Dict[str, object], signature: tuple) -> bool:
    return bool(session_state.get("portfolio_bundle")) and session_state.get("portfolio_signature") == signature


RESOURCE_COLORS = {
    "BESS": "#2563EB",
    "EV": "#16A34A",
    "PV": "#059669",
    "HVAC": "#D97706",
    "Base": "#64748B",
    "Net": "#0E7490",
    "Frequency": "#DC2626",
    "aFRR": "#7C3AED",
}

PLOTLY_TEMPLATE = {"Dark": "plotly_dark", "Light": "plotly_white"}
SUPPORT_LINE_COLOR = "#94A3B8"

APP_BUILD_ID = "ui-lightgbm-visible-2026-05-16"


FORECASTER_LABELS = {
    "lightgbm_quantile": "LightGBM quantile (P05-P95)",
    "lightgbm_probabilistic": "LightGBM probabilistic (P05-P95)",
    "xgboost_default": "XGBoost residual bands",
    "xgboost_v1": "XGBoost v1 residual bands",
}


def forecaster_display_name(key: str, forecaster_plugins: Dict[str, str]) -> str:
    return FORECASTER_LABELS.get(str(key), f"{key} ({forecaster_plugins[key]})")


def preferred_forecaster_index(plugin_options: List[str]) -> int:
    for key in ("lightgbm_quantile", "lightgbm_probabilistic", DEFAULT_FORECASTER_PLUGIN):
        if key in plugin_options:
            return plugin_options.index(key)
    return 0


def _git_metadata(args: List[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except Exception:
        return "unavailable"
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unavailable"


def dashboard_runtime_context() -> Dict[str, str]:
    app_path = Path(__file__).resolve()
    commit = _git_metadata(["rev-parse", "--short", "HEAD"])
    branch = _git_metadata(["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "build": APP_BUILD_ID,
        "branch": branch,
        "commit": commit,
        "app_path": str(app_path),
    }


def lightgbm_runtime_status(plugin_options: List[str]) -> Dict[str, str]:
    if "lightgbm_quantile" not in plugin_options:
        return {
            "label": "Not registered",
            "tone": "bad",
            "detail": "The LightGBM quantile plugin is missing from the active registry.",
        }
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        return {
            "label": "Fallback active",
            "tone": "warn",
            "detail": f"Install requirements to use native LightGBM. Current fallback reason: {exc}",
        }
    return {
        "label": "Native LightGBM ready",
        "tone": "good",
        "detail": "The active Python environment can import lightgbm.",
    }


def render_status_grid(items: List[Dict[str, str]], class_name: str = "status-grid") -> None:
    cards = []
    for item in items:
        tone = escape(str(item.get("tone", "")))
        label = escape(str(item.get("label", "")))
        value = escape(str(item.get("value", "")))
        note = escape(str(item.get("note", "")))
        cards.append(
            f'<div class="status-card {tone}">'
            f'<div class="status-label">{label}</div>'
            f'<div class="status-value">{value}</div>'
            f'<div class="status-note">{note}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="{escape(class_name)}">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_sidebar_runtime(context: Dict[str, str]) -> None:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-eyebrow">Coverly VPP</div>
            <div class="sidebar-title">MPC Market Console</div>
        </div>
        <div class="runtime-card">
            <div class="runtime-label">Running build</div>
            <div class="runtime-value">{escape(context["branch"])} @ {escape(context["commit"])}</div>
            <div class="runtime-note">{escape(context["build"])}</div>
            <div class="runtime-path">{escape(context["app_path"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )







def inject_css(theme_mode: str) -> None:
    bg = "#0F172A" if theme_mode == "Dark" else "#F4F7FA"
    surface = "#111827" if theme_mode == "Dark" else "#FFFFFF"
    sidebar_bg = "#0B1220" if theme_mode == "Dark" else "#FFFFFF"
    card_bg = "#111827" if theme_mode == "Dark" else "#FFFFFF"
    muted_bg = "#1F2937" if theme_mode == "Dark" else "#F2F5F9"
    card_border = "#334155" if theme_mode == "Dark" else "#D4DCE7"
    accent = "#5CC8BE" if theme_mode == "Dark" else "#0F766E"
    accent_2 = "#FCA5A5" if theme_mode == "Dark" else "#B42318"
    text_color = "#E5E7EB" if theme_mode == "Dark" else "#111827"
    muted_text = "#9CA3AF" if theme_mode == "Dark" else "#526071"
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {bg};
                color: {text_color};
                font-family: "Aptos", "Segoe UI", "Trebuchet MS", sans-serif;
            }}
            [data-stale="true"],
            [data-stale="true"] *,
            div[data-testid="stElementContainer"].stale,
            div[data-testid="stElementContainer"][data-stale="true"],
            div[data-testid="stVerticalBlock"].stale,
            div[data-testid="stVerticalBlock"][data-stale="true"],
            div[data-testid="stHorizontalBlock"].stale,
            div[data-testid="stHorizontalBlock"][data-stale="true"],
            .element-container.stale {{
                opacity: 1 !important;
                filter: none !important;
                transition: none !important;
            }}
            .block-container {{
                padding-top: 0.85rem !important;
                padding-bottom: 2rem !important;
                max-width: none !important;
                width: 100% !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
                padding-left: 0.9rem !important;
                padding-right: 1rem !important;
            }}
            .main .block-container {{
                margin-left: 0 !important;
            }}
            div[data-testid="stVerticalBlock"],
            div[data-testid="column"],
            div[data-testid="stHorizontalBlock"],
            .element-container {{
                min-width: 0 !important;
                box-sizing: border-box !important;
            }}
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            footer {{
                visibility: hidden !important;
                height: 0 !important;
            }}
            header {{
                background: transparent !important;
                height: 0 !important;
                pointer-events: none !important;
            }}
            h1, .stApp h1 {{
                font-size: 1.9rem !important;
                font-weight: 760 !important;
                line-height: 1.18 !important;
                letter-spacing: 0 !important;
                color: {text_color} !important;
            }}
            h2, .stApp h2 {{
                font-size: 1.55rem !important;
                font-weight: 720 !important;
                line-height: 1.16 !important;
            }}
            h3, .stApp h3 {{
                font-size: 1.25rem !important;
                font-weight: 700 !important;
                color: {text_color} !important;
            }}
            h4, .stApp h4 {{
                font-size: 1.05rem !important;
                font-weight: 700 !important;
            }}
            .stApp,
            .stApp p,
            .stApp li,
            .stApp label,
            .stApp span,
            [data-testid="stMarkdownContainer"],
            .stMarkdown,
            .stCaption,
            .small-note {{
                color: {text_color};
            }}
            p, li, label, .stMarkdown, .stCaption, .small-note {{
                font-size: 0.94rem !important;
                line-height: 1.5 !important;
                overflow-wrap: normal !important;
                word-break: normal !important;
            }}
            [data-testid="stMetricLabel"] {{
                font-size: 0.80rem !important;
                font-weight: 650 !important;
                color: {muted_text} !important;
                line-height: 1.25 !important;
                white-space: normal !important;
            }}
            [data-testid="stMetricValue"] {{
                font-size: 1.46rem !important;
                font-weight: 760 !important;
                line-height: 1.12 !important;
                color: {text_color} !important;
                overflow-wrap: anywhere !important;
            }}
            [data-testid="stMetric"] {{
                background: {surface};
                border: 1px solid {card_border};
                border-radius: 8px;
                padding: 0.78rem 0.85rem;
                min-height: 5.15rem;
                overflow: hidden;
            }}
            [data-baseweb="tab-list"] {{
                align-items: center !important;
                gap: 0.15rem !important;
                min-height: 2.7rem !important;
                padding: 0.25rem !important;
                background: {surface};
                border: 1px solid {card_border};
                border-radius: 8px;
                overflow-x: auto !important;
                overflow-y: hidden !important;
            }}
            div[data-testid="stTabs"] {{
                margin-top: 0.45rem !important;
            }}
            [data-baseweb="tab-list"] button {{
                font-size: 0.89rem !important;
                font-weight: 650 !important;
                min-height: 2.25rem !important;
                height: auto !important;
                line-height: 1.2 !important;
                padding: 0.38rem 0.65rem !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                white-space: nowrap !important;
                border-radius: 6px !important;
            }}
            button[data-baseweb="tab"][aria-selected="true"] {{
                background: {muted_bg} !important;
                border-bottom: 2px solid {accent} !important;
            }}
            [data-baseweb="tab-list"] button p,
            [data-baseweb="tab"] p {{
                margin: 0 !important;
                padding: 0 !important;
                line-height: 1.25 !important;
                font-size: 0.89rem !important;
                font-weight: 650 !important;
                display: inline-block !important;
                color: {text_color} !important;
            }}
            [aria-selected="true"] p {{
                color: {accent} !important;
            }}
            [data-testid="stSidebar"] {{
                min-width: 292px !important;
                max-width: 292px !important;
                background: {sidebar_bg};
                border-right: 1px solid {card_border};
            }}
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] div[role="radiogroup"] *,
            [data-testid="stSidebar"] .stSlider p {{
                font-size: 0.88rem !important;
            }}
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {{
                font-size: 1.05rem !important;
                font-weight: 760 !important;
            }}
            [data-testid="stSidebar"] [data-testid="stForm"] {{
                border: 1px solid {card_border};
                border-radius: 8px;
                background: {surface};
                padding: 0.85rem;
            }}
            [data-testid="stExpander"] summary p {{
                font-size: 0.92rem !important;
                font-weight: 650 !important;
            }}
            [data-testid="stDataFrame"] div,
            [data-testid="stTable"] div {{
                font-size: 0.88rem !important;
                line-height: 1.35 !important;
            }}
            [data-testid="stDataFrame"] {{
                border: 1px solid {card_border} !important;
                border-radius: 8px !important;
                overflow: hidden !important;
                background: transparent !important;
            }}
            [data-testid="stTable"],
            [data-testid="stTable"] table,
            [data-testid="stTable"] thead,
            [data-testid="stTable"] tbody,
            [data-testid="stTable"] tr,
            [data-testid="stTable"] th,
            [data-testid="stTable"] td {{
                background: {surface} !important;
                color: {text_color} !important;
                border-color: {card_border} !important;
            }}
            .modebar {{
                display: none !important;
            }}
            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            textarea,
            input {{
                border-radius: 6px !important;
                background: {surface} !important;
                color: {text_color} !important;
                border-color: {card_border} !important;
            }}
            div[data-baseweb="select"] input,
            div[data-baseweb="input"] input,
            textarea {{
                color: {text_color} !important;
                -webkit-text-fill-color: {text_color} !important;
            }}
            div[data-baseweb="select"] span,
            div[data-baseweb="select"] svg,
            div[data-baseweb="input"] svg {{
                color: {text_color} !important;
                fill: {text_color} !important;
            }}
            div[role="listbox"] {{
                min-width: 340px !important;
                background: {surface} !important;
                color: {text_color} !important;
                border: 1px solid {card_border} !important;
            }}
            div[data-baseweb="select"] span {{
                font-size: 0.92rem !important;
            }}
            div[data-baseweb="slider"] [role="slider"] {{
                background-color: {accent} !important;
                border-color: {accent} !important;
                box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.12) !important;
            }}
            div[data-baseweb="tag"],
            span[data-baseweb="tag"] {{
                background: {accent} !important;
                border-radius: 6px !important;
            }}
            div[data-baseweb="tag"] span,
            span[data-baseweb="tag"] span {{
                color: #FFFFFF !important;
                font-weight: 650 !important;
            }}
            .stButton > button,
            .stDownloadButton > button {{
                border-radius: 6px !important;
                background: {accent} !important;
                border: 1px solid {accent} !important;
                color: #FFFFFF !important;
                font-weight: 650 !important;
                min-height: 2.45rem !important;
            }}
            .stButton > button:hover,
            .stDownloadButton > button:hover {{
                background: {accent_2} !important;
                border-color: {accent_2} !important;
            }}
            .flexi-card {{
                padding: 0.95rem 1.05rem;
                border-radius: 8px;
                background: {card_bg};
                border: 1px solid {card_border};
                box-shadow: none;
                margin-bottom: 0.7rem;
            }}
            .flexi-card h4 {{
                margin: 0 0 0.4rem 0;
                letter-spacing: 0;
                font-size: 1.02rem !important;
                font-weight: 720 !important;
            }}
            .flexi-badge {{
                display: inline-block;
                margin-right: 0.35rem;
                margin-top: 0.2rem;
                padding: 0.24rem 0.5rem;
                border-radius: 6px;
                background: {muted_bg};
                border: 1px solid {card_border};
                font-size: 0.82rem;
                font-weight: 650;
            }}
            .small-note {{
                opacity: 0.78;
                font-size: 0.88rem;
            }}
            .sidebar-brand {{
                padding: 0.7rem 0.15rem 0.55rem 0.15rem;
            }}
            .sidebar-eyebrow {{
                color: {accent};
                font-size: 0.76rem;
                font-weight: 760;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .sidebar-title {{
                margin-top: 0.15rem;
                color: {text_color};
                font-size: 1.25rem;
                font-weight: 780;
                line-height: 1.12;
            }}
            .runtime-card {{
                margin: 0.35rem 0 0.85rem 0;
                padding: 0.75rem 0.8rem;
                border: 1px solid {card_border};
                border-left: 4px solid {accent};
                border-radius: 8px;
                background: {muted_bg};
            }}
            .runtime-label, .status-label {{
                color: {muted_text};
                font-size: 0.74rem;
                font-weight: 760;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }}
            .runtime-value {{
                color: {text_color};
                font-size: 0.98rem;
                font-weight: 760;
                margin-top: 0.2rem;
            }}
            .runtime-note {{
                color: {accent};
                font-size: 0.82rem;
                font-weight: 700;
                margin-top: 0.18rem;
            }}
            .runtime-path {{
                color: {muted_text};
                font-size: 0.72rem;
                line-height: 1.25;
                margin-top: 0.35rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .ops-header {{
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 1rem;
                padding: 1.0rem 1.1rem;
                border: 1px solid {card_border};
                border-left: 4px solid {accent};
                border-radius: 8px;
                background: {surface};
                margin: 0.55rem 0 0.9rem 0;
            }}
            .ops-kicker {{
                color: {accent};
                font-size: 0.76rem;
                font-weight: 760;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.18rem;
            }}
            .ops-title {{
                font-size: 1.68rem;
                line-height: 1.18;
                font-weight: 760;
                color: {text_color};
                margin: 0;
            }}
            .ops-subtitle {{
                font-size: 0.92rem;
                color: {muted_text};
                margin: 0.25rem 0 0 0;
            }}
            .ops-header-meta {{
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 0.35rem;
                min-width: 260px;
            }}
            .ops-pill, .model-pill {{
                display: inline-flex;
                align-items: center;
                min-height: 1.65rem;
                padding: 0.25rem 0.5rem;
                border-radius: 6px;
                border: 1px solid {card_border};
                background: {muted_bg};
                color: {text_color};
                font-size: 0.78rem;
                font-weight: 700;
            }}
            .section-label {{
                color: {muted_text};
                font-size: 0.78rem;
                font-weight: 760;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.45rem;
            }}
            .status-grid, .readiness-grid, .model-status-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 0.65rem;
                margin: 0.35rem 0 0.9rem 0;
            }}
            .readiness-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
            .model-status-grid {{
                grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            }}
            .status-card {{
                min-height: 5.2rem;
                padding: 0.72rem 0.82rem;
                border-radius: 8px;
                background: {card_bg};
                border: 1px solid {card_border};
            }}
            .status-card.good {{
                border-left: 4px solid {accent};
            }}
            .status-card.warn {{
                border-left: 4px solid #B87514;
            }}
            .status-card.bad {{
                border-left: 4px solid {accent_2};
            }}
            .status-value {{
                color: {text_color};
                font-size: 1.18rem;
                line-height: 1.18;
                font-weight: 780;
                margin-top: 0.32rem;
                overflow-wrap: anywhere;
            }}
            .status-note {{
                color: {muted_text};
                font-size: 0.78rem;
                line-height: 1.28;
                margin-top: 0.28rem;
            }}
            .model-banner {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.7rem;
                padding: 0.75rem 0.85rem;
                border: 1px solid {card_border};
                border-left: 4px solid {accent};
                border-radius: 8px;
                background: {surface};
                margin-bottom: 0.85rem;
            }}
            .model-banner-title {{
                color: {text_color};
                font-size: 1.0rem;
                font-weight: 760;
            }}
            .model-banner-note {{
                color: {muted_text};
                font-size: 0.84rem;
                line-height: 1.35;
                margin-top: 0.18rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )






def chart_theme_tokens(template: str) -> Dict[str, str]:
    dark = "dark" in str(template).lower()
    if dark:
        return {
            "paper": "#0F172A",
            "plot": "#0B1220",
            "text": "#E5E7EB",
            "muted": "#CBD5E1",
            "grid": "rgba(148,163,184,0.18)",
            "axis": "rgba(203,213,225,0.45)",
            "zero": "rgba(203,213,225,0.38)",
            "legend_bg": "rgba(15,23,42,0.96)",
            "legend_border": "rgba(148,163,184,0.26)",
        }
    return {
        "paper": "#FFFFFF",
        "plot": "#F8FAFC",
        "text": "#111827",
        "muted": "#475569",
        "grid": "rgba(148,163,184,0.28)",
        "axis": "rgba(100,116,139,0.42)",
        "zero": "rgba(100,116,139,0.34)",
        "legend_bg": "rgba(255,255,255,0.96)",
        "legend_border": "rgba(203,213,225,0.90)",
    }


def visible_legend_items(fig: go.Figure) -> int:
    if fig.layout.showlegend is False:
        return 0
    return sum(
        1
        for trace in fig.data
        if getattr(trace, "showlegend", None) is not False and bool(getattr(trace, "name", None))
    )


def apply_chart_style(fig: go.Figure, template: str, height: int | None = None, title: str | None = None) -> go.Figure:
    existing_title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    if isinstance(existing_title, str) and existing_title.strip().lower() == "undefined":
        existing_title = None
    resolved_title = title if title not in {None, "undefined"} else existing_title
    tokens = chart_theme_tokens(template)
    legend_items = visible_legend_items(fig)
    legend_rows = max(1, min(4, (legend_items + 4) // 5)) if legend_items else 0
    resolved_height = height if height is not None else fig.layout.height
    small_chart = bool(resolved_height and int(resolved_height) <= 320)
    top_margin = 74 if resolved_title else 34
    legend_y = -0.30 if small_chart else -0.18
    bottom_margin = (76 + legend_rows * 24) if small_chart and legend_items else (54 + legend_rows * 24 if legend_items else 30)
    layout_updates = dict(
        template=template,
        height=resolved_height,
        paper_bgcolor=tokens["paper"],
        plot_bgcolor=tokens["plot"],
        font=dict(size=13, color=tokens["text"]),
        legend=dict(
            font=dict(size=11, color=tokens["text"]),
            title_font=dict(size=12, color=tokens["muted"]),
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            bordercolor=tokens["legend_border"],
            borderwidth=0,
            itemsizing="constant",
            itemwidth=34,
            tracegroupgap=16,
        ),
        hoverlabel=dict(font_size=13, bgcolor=tokens["paper"], font_color=tokens["text"], bordercolor=tokens["legend_border"]),
        margin=dict(l=54, r=36, t=top_margin, b=bottom_margin),
    )
    if resolved_title:
        layout_updates["title"] = dict(
            text=resolved_title,
            font=dict(size=16, color=tokens["text"]),
            x=0,
            xanchor="left",
            y=0.985,
            yanchor="top",
        )
    else:
        fig.update_layout(title=None)
    fig.update_layout(**layout_updates)
    fig.update_annotations(font=dict(size=13, color=tokens["muted"]))
    fig.update_xaxes(
        title_font=dict(size=13, color=tokens["muted"]),
        tickfont=dict(size=11, color=tokens["muted"]),
        showgrid=True,
        gridcolor=tokens["grid"],
        zeroline=True,
        zerolinecolor=tokens["zero"],
        linecolor=tokens["axis"],
        color=tokens["muted"],
        automargin=True,
    )
    fig.update_yaxes(
        title_font=dict(size=13, color=tokens["muted"]),
        tickfont=dict(size=11, color=tokens["muted"]),
        showgrid=True,
        gridcolor=tokens["grid"],
        zeroline=True,
        zerolinecolor=tokens["zero"],
        linecolor=tokens["axis"],
        color=tokens["muted"],
        automargin=True,
    )
    return fig


def plotly_export_name(fig: go.Figure, key: object | None = None) -> str:
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    raw_name = key or title or "coverly-chart"
    raw_name = re.sub(r"<[^>]+>", "", str(raw_name))
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw_name).strip("-").lower()
    return (slug or "coverly-chart")[:72]


def plotly_download_config(fig: go.Figure | None, config: Dict[str, object] | None = None, key: object | None = None) -> Dict[str, object]:
    merged = dict(config or {})
    image_options = dict(merged.get("toImageButtonOptions") or {})
    image_options.setdefault("format", "png")
    image_options.setdefault("filename", plotly_export_name(fig, key) if fig is not None else "coverly-chart")
    image_options.setdefault("scale", 2)
    merged["toImageButtonOptions"] = image_options
    merged.setdefault("displayModeBar", True)
    merged.setdefault("displaylogo", False)
    merged.setdefault("responsive", True)
    return merged


def as_plotly_figure(figure_or_data: object) -> go.Figure | None:
    if isinstance(figure_or_data, go.Figure):
        return figure_or_data
    try:
        return go.Figure(figure_or_data)
    except (TypeError, ValueError):
        return None


def render_plotly_download_controls(container, fig: go.Figure, key: object | None = None) -> None:
    base_name = plotly_export_name(fig, key)
    widget_key = f"{base_name}-{id(fig):x}"
    html_bytes = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config=plotly_download_config(fig, key=key),
    ).encode("utf-8")
    json_bytes = fig.to_json(pretty=True).encode("utf-8")
    host = container.popover("Export / view figure") if hasattr(container, "popover") else container.expander("Export / view figure", expanded=False)
    with host:
        left, right = st.columns(2)
        left.download_button(
            "Viewable HTML",
            data=html_bytes,
            file_name=f"{base_name}.html",
            mime="text/html",
            key=f"{widget_key}-html",
        )
        right.download_button(
            "Plotly figure",
            data=json_bytes,
            file_name=f"{base_name}.json",
            mime="application/json",
            key=f"{widget_key}-json",
        )


def install_plotly_download_exporter() -> None:
    if getattr(PlotlyMixin, "_coverly_download_exporter_installed", False):
        st.plotly_chart = st._main.plotly_chart
        return
    native_plotly_chart = PlotlyMixin.plotly_chart

    def downloadable_plotly_chart(self, figure_or_data=None, *args, **kwargs):
        fig = as_plotly_figure(figure_or_data)
        key = kwargs.get("key")
        kwargs["config"] = plotly_download_config(fig, kwargs.get("config"), key)
        result = native_plotly_chart(self, figure_or_data, *args, **kwargs)
        if fig is not None:
            render_plotly_download_controls(self, fig, key=key)
        return result

    PlotlyMixin.plotly_chart = downloadable_plotly_chart
    st.plotly_chart = st._main.plotly_chart
    PlotlyMixin._coverly_download_exporter_installed = True


install_plotly_download_exporter()


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
    dark = st.session_state.get("dashboard_theme_mode", "Light") == "Dark"
    bg = "#0F172A" if dark else "#FFFFFF"
    header_bg = "#111827" if dark else "#F8FAFC"
    text = "#E5E7EB" if dark else "#111827"
    muted = "#CBD5E1" if dark else "#475569"
    border = "#334155" if dark else "#D4DCE7"
    return frame.style.set_properties(
        **{
            "background-color": bg,
            "border-color": border,
            "color": text,
            "font-size": "16px",
        }
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", header_bg),
                    ("border-color", border),
                    ("color", muted),
                    ("font-size", "16px"),
                    ("font-weight", "700"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", bg),
                    ("border-color", border),
                    ("color", text),
                    ("font-size", "15px"),
                ],
            },
        ]
    )


def numeric_signal_options(df: pd.DataFrame, preferred: List[str] | None = None) -> List[str]:
    preferred = preferred or []
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    ordered = [col for col in preferred if col in numeric_cols]
    ordered.extend([col for col in numeric_cols if col not in ordered])
    return ordered


def signal_line_figure(df: pd.DataFrame, columns: List[str], title: str, y_title: str | None = None) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines"))
    if y_title:
        fig.update_layout(yaxis_title=y_title)
    fig.update_layout(title=title)
    return fig


def numeric_trace(df: pd.DataFrame, column: str, scale: float = 1.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0) / float(scale)


def has_visible_signal(series: pd.Series, threshold: float = 1e-6) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return not values.empty and float(values.abs().max()) > threshold


def upper_mpc_decision_figure(upper_plot_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        row_heights=[0.58, 0.42],
        specs=[[{}], [{"secondary_y": True}]],
        subplot_titles=("Reserve socket and accepted bid", "Revenue response and market prices"),
    )
    fig.add_trace(
        go.Scatter(x=upper_plot_df.index, y=numeric_trace(upper_plot_df, "fcr_bid_kw", 1000.0), name="FCR-N symmetric bid", line={"color": RESOURCE_COLORS["Frequency"], "width": 3}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=upper_plot_df.index, y=numeric_trace(upper_plot_df, "afrr_up_bid_kw", 1000.0), name="aFRR up bid", line={"color": RESOURCE_COLORS["Net"], "dash": "dash"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=upper_plot_df.index, y=-numeric_trace(upper_plot_df, "afrr_down_bid_kw", 1000.0), name="aFRR down bid", line={"color": RESOURCE_COLORS["aFRR"], "dash": "dot"}),
        row=1,
        col=1,
    )
    if {"socket_up_kw", "socket_down_kw"}.issubset(upper_plot_df.columns):
        fig.add_trace(
            go.Scatter(x=upper_plot_df.index, y=numeric_trace(upper_plot_df, "socket_up_kw", 1000.0), name="Executable up socket", line={"color": "#6b7280", "dash": "longdash"}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=upper_plot_df.index, y=-numeric_trace(upper_plot_df, "socket_down_kw", 1000.0), name="Executable down socket", line={"color": "#6b7280", "dash": "longdashdot"}),
            row=1,
            col=1,
        )
    if "net_revenue_eur" in upper_plot_df:
        fig.add_trace(
            go.Scatter(x=upper_plot_df.index, y=numeric_trace(upper_plot_df, "net_revenue_eur"), name="Net revenue", line={"color": RESOURCE_COLORS["Net"], "width": 3}),
            row=2,
            col=1,
            secondary_y=False,
        )
    for column, label, dash in [
        ("fcrn_capacity_eur_per_mw_h", "FCR-N price", "dot"),
        ("afrr_up_capacity_eur_per_mw_h", "aFRR up price", "dash"),
        ("afrr_down_capacity_eur_per_mw_h", "aFRR down price", "longdash"),
    ]:
        if column in upper_plot_df:
            fig.add_trace(
                go.Scatter(x=upper_plot_df.index, y=numeric_trace(upper_plot_df, column), name=label, line={"dash": dash}),
                row=2,
                col=1,
                secondary_y=True,
            )
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="EUR", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="EUR/MW/h", row=2, col=1, secondary_y=True)
    fig.update_layout(hovermode="x unified")
    return fig


def lower_mpc_execution_figure(tracking_window: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.105,
        row_heights=[0.42, 0.28, 0.30],
        specs=[[{}], [{}], [{"secondary_y": True}]],
        subplot_titles=(
            "Reserve tracking",
            "Resource dispatch",
            "Control error and latency",
        ),
    )

    def add_line(
        row: int,
        col: int,
        y: pd.Series,
        name: str,
        color: str,
        *,
        dash: str = "solid",
        width: float = 2.0,
        opacity: float = 1.0,
        secondary_y: bool | None = None,
        visible: bool | str | None = None,
        legendgroup: str | None = None,
        hover_suffix: str = "",
        show_if_zero: bool = False,
        showlegend: bool = True,
    ) -> None:
        if not show_if_zero and not has_visible_signal(y):
            return
        trace = go.Scatter(
            x=tracking_window.index,
            y=y,
            name=name,
            mode="lines",
            line={"color": color, "dash": dash, "width": width},
            opacity=opacity,
            connectgaps=True,
            legendgroup=legendgroup,
            showlegend=showlegend,
            hovertemplate=f"%{{x|%H:%M:%S}}<br>{name}: %{{y:.3f}}{hover_suffix}<extra></extra>",
        )
        if visible is not None:
            trace.visible = visible
        if secondary_y is None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace, row=row, col=col, secondary_y=secondary_y)

    signed_request = numeric_trace(tracking_window, "requested_up_kw") - numeric_trace(tracking_window, "requested_down_kw")
    signed_delivery = numeric_trace(tracking_window, "delivered_up_kw") - numeric_trace(tracking_window, "delivered_down_kw")
    signed_request_mw = signed_request / 1000.0
    signed_delivery_mw = signed_delivery / 1000.0
    tolerance_mw = numeric_trace(tracking_window, "tracking_tolerance_kw", 1000.0)
    if has_visible_signal(tolerance_mw):
        fig.add_trace(
            go.Scatter(
                x=tracking_window.index,
                y=signed_request_mw - tolerance_mw,
                mode="lines",
                line={"width": 0},
                name="Tolerance lower",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=tracking_window.index,
                y=signed_request_mw + tolerance_mw,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(14,116,144,0.12)",
                name="Tolerance band",
                legendgroup="reserve",
                hovertemplate="%{x|%H:%M:%S}<br>Tolerance band<extra></extra>",
            ),
            row=1,
            col=1,
        )
    add_line(1, 1, signed_request_mw, "TSO request", RESOURCE_COLORS["Frequency"], dash="dash", width=2.4, legendgroup="reserve", hover_suffix=" MW", show_if_zero=True)
    add_line(1, 1, signed_delivery_mw, "Delivered reserve", RESOURCE_COLORS["Net"], width=3.4, legendgroup="reserve", hover_suffix=" MW", show_if_zero=True)
    if {"committed_up_kw", "committed_down_kw"}.issubset(tracking_window.columns):
        add_line(1, 1, numeric_trace(tracking_window, "committed_up_kw", 1000.0), "Upper socket", SUPPORT_LINE_COLOR, dash="dash", opacity=0.7, hover_suffix=" MW", showlegend=False)
        add_line(1, 1, -numeric_trace(tracking_window, "committed_down_kw", 1000.0), "Lower socket", SUPPORT_LINE_COLOR, dash="dash", opacity=0.7, hover_suffix=" MW", showlegend=False)

    for resource, up_col, down_col, color in [
        ("BESS", "bess_up_kw", "bess_down_kw", RESOURCE_COLORS["BESS"]),
        ("HVAC", "hvac_up_kw", "hvac_down_kw", RESOURCE_COLORS["HVAC"]),
        ("EV", "ev_up_kw", "ev_down_kw", RESOURCE_COLORS["EV"]),
        ("PV", "", "pv_down_kw", RESOURCE_COLORS["PV"]),
    ]:
        up = numeric_trace(tracking_window, up_col, 1000.0) if up_col else pd.Series(0.0, index=tracking_window.index)
        down = numeric_trace(tracking_window, down_col, 1000.0) if down_col in tracking_window else pd.Series(0.0, index=tracking_window.index)
        signed_resource = up - down
        if has_visible_signal(signed_resource):
            fig.add_trace(
                go.Bar(
                    x=tracking_window.index,
                    y=signed_resource,
                    name=resource,
                    marker={"color": color, "line": {"width": 0}},
                    opacity=0.86,
                    legendgroup="resources",
                    hovertemplate=f"%{{x|%H:%M:%S}}<br>{resource}: %{{y:.3f}} MW<extra></extra>",
                ),
                row=2,
                col=1,
            )

    tracking_error = numeric_trace(tracking_window, "tracking_error_kw")
    if has_visible_signal(tolerance_mw):
        tolerance_kw = tolerance_mw * 1000.0
        fig.add_trace(
            go.Scatter(
                x=tracking_window.index,
                y=-tolerance_kw,
                mode="lines",
                line={"width": 0},
                name="Error tolerance lower",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=tracking_window.index,
                y=tolerance_kw,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(148,163,184,0.16)",
                name="Error tolerance",
                legendgroup="control",
                hovertemplate="%{x|%H:%M:%S}<br>Error tolerance<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
    if has_visible_signal(tracking_error):
        error_colors = np.where(tracking_error.abs() <= (tolerance_mw * 1000.0), "#16A34A", "#DC2626")
        fig.add_trace(
            go.Bar(
                x=tracking_window.index,
                y=tracking_error,
                name="Tracking error",
                marker={"color": error_colors, "line": {"width": 0}},
                opacity=0.82,
                legendgroup="control",
                hovertemplate="%{x|%H:%M:%S}<br>Tracking error: %{y:.1f} kW<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
    if "control_latency_ms" in tracking_window:
        add_line(3, 1, numeric_trace(tracking_window, "control_latency_ms"), "Latency", RESOURCE_COLORS["aFRR"], width=2.6, secondary_y=True, legendgroup="control", hover_suffix=" ms")
    if "control_deadline_ms" in tracking_window:
        add_line(3, 1, numeric_trace(tracking_window, "control_deadline_ms"), "Deadline", SUPPORT_LINE_COLOR, dash="dash", width=1.4, opacity=0.62, secondary_y=True, legendgroup="control", hover_suffix=" ms", showlegend=False)

    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)
    fig.update_yaxes(title_text="ms", row=3, col=1, secondary_y=True)
    for row in range(1, 3):
        fig.add_hline(y=0, row=row, col=1, line_color="rgba(148,163,184,0.42)", line_width=1)
    fig.add_hline(y=0, row=3, col=1, line_color="rgba(148,163,184,0.42)", line_width=1)
    fig.update_layout(barmode="relative", bargap=0.08, hovermode="x unified", legend_traceorder="grouped")
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="rgba(148,163,184,0.45)", spikethickness=1)
    return fig


def forecast_performance_figure(comparison: pd.DataFrame, target: str) -> go.Figure:
    fig = go.Figure()
    for lower_col, upper_col, label, color in [
        ("prediction_p05", "prediction_p95", "P05-P95", "rgba(11, 114, 133, 0.12)"),
        ("prediction_p20", "prediction_p80", "P20-P80", "rgba(11, 114, 133, 0.20)"),
    ]:
        if {lower_col, upper_col}.issubset(comparison.columns):
            fig.add_trace(
                go.Scatter(
                    x=comparison.index,
                    y=comparison[lower_col],
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{label} lower",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=comparison.index,
                    y=comparison[upper_col],
                    fill="tonexty",
                    fillcolor=color,
                    line={"width": 0},
                    name=label,
                )
            )
    fig.add_trace(go.Scatter(x=comparison.index, y=comparison["actual"], name=f"Actual {target}", line={"color": RESOURCE_COLORS["Base"]}))
    fig.add_trace(
        go.Scatter(
            x=comparison.index,
            y=comparison["prediction"],
            name=f"Predicted {target}",
            line={"color": RESOURCE_COLORS["Net"], "dash": "dot"},
        )
    )
    fig.update_layout(yaxis_title=target)
    return fig


def forecast_revision_figure(forecast_trace: pd.DataFrame, target: str, max_iterations: int) -> go.Figure:
    fig = go.Figure()
    required = {"mpc_iteration", "issued_at", "horizon_timestamp", "horizon_step", target}
    if forecast_trace.empty or not required.issubset(forecast_trace.columns):
        fig.update_layout(yaxis_title=target, xaxis_title="Horizon timestamp")
        return fig

    working = forecast_trace.copy()
    working["mpc_iteration"] = pd.to_numeric(working["mpc_iteration"], errors="coerce").astype("Int64")
    working["horizon_step"] = pd.to_numeric(working["horizon_step"], errors="coerce").astype("Int64")
    working["issued_at"] = pd.to_datetime(working["issued_at"], errors="coerce")
    working["horizon_timestamp"] = pd.to_datetime(working["horizon_timestamp"], errors="coerce")
    working = working.dropna(subset=["mpc_iteration", "issued_at", "horizon_timestamp", target])
    iterations = list(working["mpc_iteration"].dropna().drop_duplicates().sort_values().tail(max_iterations))
    if not iterations:
        fig.update_layout(yaxis_title=target, xaxis_title="Horizon timestamp")
        return fig

    latest_iteration = iterations[-1]
    latest = working[working["mpc_iteration"] == latest_iteration].sort_values("horizon_step")
    lower_col = f"{target}_forecast_p05"
    upper_col = f"{target}_forecast_p95"
    if {lower_col, upper_col}.issubset(latest.columns):
        fig.add_trace(
            go.Scatter(
                x=latest["horizon_timestamp"],
                y=latest[lower_col],
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                name="Latest P05",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=latest["horizon_timestamp"],
                y=latest[upper_col],
                fill="tonexty",
                fillcolor="rgba(194, 37, 92, 0.15)",
                line={"width": 0},
                name="Latest P05-P95",
            )
        )

    palette = px.colors.qualitative.Safe + px.colors.qualitative.Set2
    for offset, iteration in enumerate(iterations):
        slice_df = working[working["mpc_iteration"] == iteration].sort_values("horizon_step")
        if slice_df.empty:
            continue
        issued_at = pd.Timestamp(slice_df["issued_at"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=slice_df["horizon_timestamp"],
                y=slice_df[target],
                mode="lines+markers",
                name=f"MPC {int(iteration)} - {issued_at:%m-%d %H:%M}",
                line={"color": palette[offset % len(palette)], "width": 3 if iteration == latest_iteration else 1.5},
            )
        )
    fig.update_layout(yaxis_title=target, xaxis_title="Horizon timestamp")
    return fig


def live_market_status(session: Dict[str, object] | None, now_s: float | None = None) -> Dict[str, float | str]:
    if not session:
        return {"status": "idle", "countdown_seconds": 0.0, "elapsed_seconds": 0.0, "remaining_seconds": 0.0}
    now = float(time.time() if now_s is None else now_s)
    start_at = float(session.get("start_at_wall_s", now))
    if session.get("stopped_at_wall_s") is not None:
        stopped_at = float(session.get("stopped_at_wall_s", now))
        return {
            "status": "closed",
            "countdown_seconds": 0.0,
            "elapsed_seconds": max(stopped_at - start_at, 0.0),
            "remaining_seconds": 0.0,
        }
    duration = max(float(session.get("duration_seconds", 0.0)), 0.0)
    end_at = start_at + duration
    if now < start_at:
        return {
            "status": "scheduled",
            "countdown_seconds": start_at - now,
            "elapsed_seconds": 0.0,
            "remaining_seconds": duration,
        }
    if duration <= 0.0 or now <= end_at:
        return {
            "status": "live",
            "countdown_seconds": 0.0,
            "elapsed_seconds": max(now - start_at, 0.0),
            "remaining_seconds": max(end_at - now, 0.0),
        }
    return {
        "status": "closed",
        "countdown_seconds": 0.0,
        "elapsed_seconds": duration,
        "remaining_seconds": 0.0,
    }


def live_visible_frame(frame: pd.DataFrame, session: Dict[str, object] | None, now_s: float | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    status = live_market_status(session, now_s)
    if status["status"] == "idle":
        return frame
    dt_seconds = max(int(float(session.get("dt_seconds", 4))) if session else 4, 1)
    if status["status"] == "scheduled":
        return frame.iloc[:0]
    visible_rows = min(int(float(status["elapsed_seconds"]) // dt_seconds) + 1, len(frame))
    return frame.iloc[:visible_rows]


@st.cache_data(show_spinner=False, ttl=4)
def cached_lower_mpc_from_upper(
    df: pd.DataFrame,
    upper_result: Dict[str, object],
    fleet_meta: Dict[str, float],
    market_mode: str,
    resource_mode: str,
    preview_4s: pd.DataFrame,
    device_roster: pd.DataFrame,
    elapsed_tick: int,
    inner_dt_seconds: int = 4,
) -> Dict[str, object]:
    elapsed_seconds = float(max(int(elapsed_tick), 0) * max(int(inner_dt_seconds), 1))
    return run_lower_mpc_from_upper_result(
        df=df,
        upper_result=upper_result,
        fleet_meta=fleet_meta,
        market_mode=market_mode,
        resource_mode=resource_mode,
        preview_4s=preview_4s,
        device_roster=device_roster,
        inner_controller_mode="mpc",
        inner_dt_seconds=int(inner_dt_seconds),
        inner_mpc_horizon_seconds=4,
        rotation_strategy="usage_aware",
        gateway_mode="live_market_coupled",
        elapsed_seconds=elapsed_seconds,
    )


def result_history(result: Dict[str, object] | None) -> pd.DataFrame:
    if not isinstance(result, dict):
        return pd.DataFrame()
    history = result.get("history", pd.DataFrame())
    return history if isinstance(history, pd.DataFrame) else pd.DataFrame()


def result_frame(result: Dict[str, object] | None, key: str) -> pd.DataFrame:
    if not isinstance(result, dict):
        return pd.DataFrame()
    frame = result.get(key, pd.DataFrame())
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def result_summary(result: Dict[str, object] | None) -> Dict[str, object]:
    if not isinstance(result, dict):
        return {}
    summary = result.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def market_replay_source_label(status: Dict[str, object] | None) -> str:
    if not isinstance(status, dict):
        return "Synthetic"
    source = str(status.get("real_time_preview_source", "synthetic"))
    if source == "real_activation":
        return "Real activation"
    return "Synthetic"


def upper_socket_metrics(result: Dict[str, object] | None) -> Dict[str, object]:
    history = result_history(result)
    summary = result_summary(result)
    if history.empty:
        return {
            "ready": False,
            "reason": "Run the upper MPC layer first.",
            "accepted_slots": 0,
            "max_committed_kw": 0.0,
            "mean_fcr_bid_kw": 0.0,
            "mean_afrr_bid_kw": 0.0,
            "mean_buffer_kw": 0.0,
        }

    def numeric_column(name: str) -> pd.Series:
        if name in history:
            return pd.to_numeric(history[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=history.index)

    fcr_bid = numeric_column("fcr_bid_kw")
    afrr_up_bid = numeric_column("afrr_up_bid_kw")
    afrr_down_bid = numeric_column("afrr_down_bid_kw")
    buffer_kw = numeric_column("reserve_buffer_kw")
    committed_kw = pd.concat([fcr_bid, afrr_up_bid, afrr_down_bid], axis=1).max(axis=1)
    max_committed_kw = float(committed_kw.max()) if not committed_kw.empty else 0.0
    mean_fcr_bid_kw = float(fcr_bid.mean()) if not fcr_bid.empty else 0.0
    mean_afrr_bid_kw = float(np.maximum(afrr_up_bid, afrr_down_bid).mean()) if not afrr_up_bid.empty else 0.0
    mean_buffer_kw = float(buffer_kw.mean()) if not buffer_kw.empty else 0.0

    if "market_gate_status" in history:
        accepted_slots = int((history["market_gate_status"].astype(str) == "Participate").sum())
    else:
        accepted_slots = int(summary.get("market_gate_participation_intervals", 0) or 0)

    has_commitment = max_committed_kw > 1e-3
    has_accepted_slot = accepted_slots > 0
    if has_commitment and has_accepted_slot:
        reason = "Upper MPC cleared an executable reserve socket."
    elif has_commitment:
        reason = "Upper MPC produced capacity, but no market interval cleared the participation gate."
    else:
        reason = "Upper MPC produced no committed reserve capacity."

    return {
        "ready": bool(has_commitment and has_accepted_slot),
        "reason": reason,
        "accepted_slots": accepted_slots,
        "max_committed_kw": max_committed_kw,
        "mean_fcr_bid_kw": mean_fcr_bid_kw,
        "mean_afrr_bid_kw": mean_afrr_bid_kw,
        "mean_buffer_kw": mean_buffer_kw,
    }


def render_lower_live_trace(
    lower_result: Dict[str, object] | None,
    session: Dict[str, object] | None,
    template: str,
    key_prefix: str,
    fallback_result: Dict[str, object] | None = None,
    title: str = "Lower MPC 4-second Optimization Trace",
    include_debug_table: bool = True,
) -> bool:
    lower_tracking_full = result_frame(lower_result, "tracking_4s")
    if lower_tracking_full.empty:
        return False

    st.markdown(f"### {title}")
    live_lower_tracking = live_visible_frame(lower_tracking_full, session)
    lower_visible = live_lower_tracking if not live_lower_tracking.empty else lower_tracking_full
    if lower_visible.empty:
        return False

    lm1, lm2, lm3, lm4 = st.columns(4)
    lm1.metric("Visible 4-sec ticks", f"{len(lower_visible):,}")
    lm2.metric("Visible mean abs error", f"{float(lower_visible['tracking_error_kw'].abs().mean()):.2f} kW")
    lm3.metric("Visible max latency", f"{float(lower_visible.get('control_latency_ms', pd.Series([0.0])).max()):.1f} ms")
    lm4.metric("Visible recovery ticks", f"{int(lower_visible.get('recovery_mode', pd.Series(dtype=bool)).astype(bool).sum()):,}")

    st.plotly_chart(
        apply_chart_style(lower_mpc_execution_figure(lower_visible), template, height=560, title="Lower MPC execution console"),
        use_container_width=True,
        key=f"{key_prefix}_execution_console",
    )

    afrr_energy_audit = result_frame(lower_result, "afrr_energy_audit")
    if afrr_energy_audit.empty:
        afrr_energy_audit = result_frame(fallback_result, "afrr_energy_audit")
    if not afrr_energy_audit.empty:
        st.markdown("### aFRR settlement-period energy audit")
        energy_audit_fig = go.Figure()
        energy_audit_fig.add_trace(go.Bar(x=afrr_energy_audit["isp_start"], y=afrr_energy_audit["afrr_reported_energy_mwh"], name="BSP reported aFRR energy"))
        energy_audit_fig.add_trace(go.Bar(x=afrr_energy_audit["isp_start"], y=afrr_energy_audit["afrr_fingrid_calculated_energy_mwh"], name="Fingrid-calculated proxy"))
        energy_audit_fig.add_trace(
            go.Scatter(
                x=afrr_energy_audit["isp_start"],
                y=afrr_energy_audit["afrr_energy_reporting_diff_pct"],
                name="absolute diff %",
                yaxis="y2",
                line={"color": RESOURCE_COLORS["Frequency"], "width": 3},
            )
        )
        energy_audit_fig.update_layout(
            barmode="group",
            yaxis_title="MWh",
            yaxis2={
                "title": "diff ratio",
                "overlaying": "y",
                "side": "right",
                "range": [0, max(0.12, float(afrr_energy_audit["afrr_energy_reporting_diff_pct"].max()) * 1.2)],
            },
        )
        st.plotly_chart(
            apply_chart_style(energy_audit_fig, template, height=320, title="aFRR delivered energy reconciliation"),
            use_container_width=True,
            key=f"{key_prefix}_afrr_energy_audit",
        )

    if include_debug_table:
        lower_cols = [
            "inner_solver_status",
            "optimization_strategy",
            "requested_signed_kw",
            "requested_up_kw",
            "requested_down_kw",
            "delivered_signed_kw",
            "delivered_up_kw",
            "delivered_down_kw",
            "ideal_delivered_kw",
            "telemetry_error_kw",
            "raw_request_kw",
            "socket_up_kw",
            "socket_down_kw",
            "socket_violation_up_kw",
            "socket_violation_down_kw",
            "tracking_error_kw",
            "tracking_tolerance_kw",
            "active_capacity_kw",
            "buffer_capacity_kw",
            "buffer_used_kw",
            "fast_bridge_used_kw",
            "afrr_accuracy_ratio",
            "afrr_reporting_error_kw",
            "fcr_response_ratio",
            "recovery_mode",
            "error_rising",
            "control_latency_ms",
            "control_deadline_met",
        ]
        st.dataframe(
            styled_dataframe(lower_tracking_full[[col for col in lower_cols if col in lower_tracking_full.columns]].head(80).round(3)),
            use_container_width=True,
            height=320,
        )
    return True


def build_live_market_feed(
    preview_4s: pd.DataFrame,
    upper_result: Dict[str, object],
    duration_seconds: int,
    dt_seconds: int = 4,
) -> pd.DataFrame:
    upper_history = upper_result.get("history", pd.DataFrame()) if isinstance(upper_result, dict) else pd.DataFrame()
    if isinstance(upper_history, pd.DataFrame) and not upper_history.empty:
        start_ts = pd.Timestamp(upper_history.index[0])
    elif preview_4s is not None and not preview_4s.empty:
        start_ts = pd.Timestamp(preview_4s.index[0])
    else:
        start_ts = pd.Timestamp.utcnow().tz_localize(None)
    periods = max(int(duration_seconds // max(dt_seconds, 1)) + 1, 1)
    index = pd.date_range(start_ts, periods=periods, freq=f"{int(dt_seconds)}s")
    columns = ["frequency_hz", "fcr_signal_norm", "afrr_signal_norm"]
    if preview_4s is not None and not preview_4s.empty:
        feed = preview_4s.reindex(index)
        feed = feed[[col for col in columns if col in feed.columns]].interpolate(method="time").ffill().bfill()
    else:
        feed = pd.DataFrame(index=index)
    for col, default in {"frequency_hz": 50.0, "fcr_signal_norm": 0.0, "afrr_signal_norm": 0.0}.items():
        if col not in feed:
            feed[col] = default
        feed[col] = feed[col].fillna(default)
    return feed[columns]


def sync_market_simulator_feed(
    preview_4s: pd.DataFrame,
    upper_result: Dict[str, object],
    session: Dict[str, object] | None,
    duration_seconds: int,
    dt_seconds: int = 4,
    now_s: float | None = None,
) -> pd.DataFrame:
    feed = build_live_market_feed(preview_4s, upper_result, duration_seconds, dt_seconds=dt_seconds)
    if not session:
        return feed

    start_at = float(session.get("start_at_wall_s", 0.0))
    simulator = st.session_state.get("market_simulator")
    simulator_feed_len = len(getattr(simulator, "preview_signals", pd.DataFrame())) if isinstance(simulator, MarketSimulator) else -1
    if (
        not isinstance(simulator, MarketSimulator)
        or float(getattr(simulator, "start_wall_time_s", 0.0) or 0.0) != start_at
        or simulator_feed_len != len(feed)
    ):
        simulator = MarketSimulator(
            preview_signals=feed,
            config=MarketSimulatorConfig(dt_seconds=int(dt_seconds)),
            start_wall_time_s=start_at,
        )
        st.session_state["market_simulator"] = simulator
        st.session_state["live_market_ticks"] = []

    signal_time = float(session.get("stopped_at_wall_s", time.time() if now_s is None else now_s))
    signal = simulator.get_current_signal(signal_time)
    ticks: List[Dict[str, object]] = list(st.session_state.get("live_market_ticks", []))
    if signal is not None:
        tick_index = int(signal.get("tick_index", 0))
        payload = {
            "tick_index": tick_index,
            "frequency_hz": float(signal.get("frequency_hz", 50.0)),
            "fcr_signal_norm": float(signal.get("fcr_signal_norm", 0.0)),
            "afrr_signal_norm": float(signal.get("afrr_signal_norm", 0.0)),
            "wall_elapsed_seconds": float(signal.get("wall_elapsed_seconds", 0.0)),
        }
        ticks = [tick for tick in ticks if int(tick.get("tick_index", -1)) != tick_index]
        ticks.append(payload)
        ticks = sorted(ticks, key=lambda tick: int(tick.get("tick_index", 0)))
        st.session_state["live_market_ticks"] = ticks

    for tick in ticks:
        tick_index = int(tick.get("tick_index", -1))
        if tick_index < 0 or tick_index >= len(feed):
            continue
        for col in ["frequency_hz", "fcr_signal_norm", "afrr_signal_norm"]:
            feed.iat[tick_index, feed.columns.get_loc(col)] = float(tick.get(col, feed.iloc[tick_index][col]))
    return feed


def data_quality_figure(df: pd.DataFrame, columns: List[str], template: str) -> go.Figure:
    rows = []
    for col in columns:
        if col in df.columns:
            rows.append({"Signal": col, "Completeness %": float(df[col].notna().mean() * 100.0)})
    frame = pd.DataFrame(rows) if rows else pd.DataFrame({"Signal": [], "Completeness %": []})
    fig = px.bar(frame, x="Signal", y="Completeness %", color="Completeness %", range_y=[0, 100], template=template)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Complete rows (%)")
    return fig


































def train_plugin_mpc_models(df: pd.DataFrame, seed: int, forecaster_plugin: str) -> Dict[str, object]:
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
    models: Dict[str, object] = {}
    for target in SUPPORTED_MPC_TARGETS:
        forecaster = PLUGIN_REGISTRY.get_forecaster(
            forecaster_plugin,
            name=f"{forecaster_plugin}_{target}",
            seed=seed,
        )
        forecaster.fit_from_frame(df, target, predictors, lags=4, horizon_steps=1)
        models[target] = forecaster
    return models


def ensure_default_models(df: pd.DataFrame, seed: int, forecaster_plugin: str = DEFAULT_FORECASTER_PLUGIN) -> Dict[str, object]:
    signature = (
        len(df),
        str(df.index[0]),
        str(df.index[-1]),
        seed,
        forecaster_plugin,
        int(round(float(df["dt_h"].iloc[0]) * 60.0)),
        float(df["base_load_kw"].mean()),
        float(df["pv_available_kw"].mean()),
        float(df["fast_sym_kw"].mean()),
    )
    if "default_mpc_models" not in st.session_state or st.session_state.get("default_mpc_signature") != signature:
        st.session_state["default_mpc_models"] = train_plugin_mpc_models(df, seed, forecaster_plugin)
        st.session_state["default_mpc_signature"] = signature
    custom = st.session_state.get("custom_mpc_models", {})
    merged = dict(st.session_state["default_mpc_models"])
    merged.update(custom)
    return merged








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














def plot_sankey(summary: Dict[str, float], template: str = "plotly_white") -> go.Figure:
    tokens = chart_theme_tokens(template)
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
                    line=dict(color=tokens["axis"], width=1),
                ),
                textfont=dict(size=22, color=tokens["text"]),
                link=dict(
                    source=source,
                    target=target,
                    value=flow,
                    color=[
                        "rgba(148,163,184,0.48)",
                        "rgba(148,163,184,0.48)",
                        "rgba(148,163,184,0.48)",
                        "rgba(148,163,184,0.48)",
                        "rgba(37,99,235,0.24)",
                        "rgba(37,99,235,0.24)",
                        "rgba(217,119,6,0.28)",
                        "rgba(5,150,105,0.26)",
                        "rgba(124,58,237,0.24)",
                        "rgba(220,38,38,0.26)",
                        "rgba(14,116,144,0.26)",
                    ],
                ),
            )
        ]
    )
    fig.update_layout(
        template=template,
        paper_bgcolor=tokens["paper"],
        plot_bgcolor=tokens["plot"],
        height=470,
        margin=dict(l=15, r=40, t=20, b=10),
        font=dict(size=18, color=tokens["text"]),
    )
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




def main() -> None:
    theme_mode = st.sidebar.radio("Dashboard theme", ["Light", "Dark"], horizontal=True, help="Applies to charts and dashboard styling.")
    st.session_state["dashboard_theme_mode"] = theme_mode
    dev_mode = st.sidebar.toggle(
        "Developer / research panels",
        value=False,
        help="Shows educational response-model plots, Sankey diagrams, equations, and animated diagnostics that are not operator controls.",
    )
    inject_css(theme_mode)
    template = PLOTLY_TEMPLATE[theme_mode]
    runtime_context = dashboard_runtime_context()
    render_sidebar_runtime(runtime_context)

    if "scenario_inputs" not in st.session_state:
        st.session_state["scenario_inputs"] = default_scenario_inputs()
    scenario_defaults = dict(st.session_state["scenario_inputs"])

    with st.sidebar.form("scenario_form"):
        st.header("Scenario Setup")
        start_date_input = st.date_input("Simulation start", value=pd.Timestamp(scenario_defaults["start_date"]))
        days_input = st.slider("Simulation length (days)", 1, 365, int(scenario_defaults["days"]))
        freq_minutes_input = st.slider("Simulation timestep (minutes)", 1, 15, int(scenario_defaults["freq_minutes"]), 1)
        n_homes_input = st.slider("Aggregated homes", 500, 5000, int(scenario_defaults["n_homes"]), step=100)
        ev_pen_input = st.slider("EV penetration", 0.0, 1.0, float(scenario_defaults["ev_pen"]), 0.05)
        bess_pen_input = st.slider("BESS penetration", 0.0, 1.0, float(scenario_defaults["bess_pen"]), 0.05)
        pv_pen_input = st.slider("PV penetration", 0.0, 1.0, float(scenario_defaults["pv_pen"]), 0.05)
        hvac_pen_input = st.slider("Smart HVAC penetration", 0.0, 1.0, float(scenario_defaults["hvac_pen"]), 0.05)
        hvac_mode_input = st.selectbox(
            "HVAC technology",
            HVAC_MODES,
            index=HVAC_MODES.index(str(scenario_defaults["hvac_mode"])) if str(scenario_defaults["hvac_mode"]) in HVAC_MODES else 1,
        )
        hvac_response_default = default_hvac_response_seconds(hvac_mode_input)
        hvac_response_s_input = st.slider(
            "Delivered HVAC response time (s)",
            20.0,
            300.0,
            float(np.clip(float(scenario_defaults.get("hvac_response_s", hvac_response_default)), 20.0, 300.0)),
            1.0,
            help="Delivered aggregate response used for FCR-N dynamic-cap screening, not just command latency.",
        )

        st.header("Synthetic Finland Context")
        cloudiness_input = st.slider("Cloudiness / variability", 0.1, 1.0, float(scenario_defaults["cloudiness"]), 0.05)
        climate_shift_c_input = st.slider("Temperature anomaly (deg C)", -8.0, 8.0, float(scenario_defaults["climate_shift_c"]), 0.5)
        solar_scale_input = st.slider("Solar resource multiplier", 0.5, 1.5, float(scenario_defaults["solar_scale"]), 0.05)
        scarcity_input = st.slider("Market scarcity / price stress", 0.6, 1.8, float(scenario_defaults["scarcity"]), 0.05)
        setpoint_c_input = st.slider("HVAC setpoint (deg C)", 19.0, 23.0, float(scenario_defaults["setpoint_c"]), 0.5)
        comfort_band_c_input = st.slider("Comfort band (+/- deg C)", 0.5, 2.5, float(scenario_defaults["comfort_band_c"]), 0.1)
        seed_input = st.number_input("Random seed", value=int(scenario_defaults["seed"]), step=1)

        st.header("Market Data Source")
        market_source_input = st.radio(
        "Training and market signals",
        ["Synthetic", "Real market APIs"],
        index=0 if scenario_defaults["market_source"] == "Synthetic" else 1,
        help="Synthetic keeps the dashboard offline. Real market APIs uses ENTSO-E and Fingrid data where available.",
    )
        use_real_market_input = market_source_input == "Real market APIs"
        market_lookback_days_input = st.slider("Real market lookback days", 1, 120, int(scenario_defaults["market_lookback_days"]), 1, disabled=not use_real_market_input)
        entsoe_key_input_form = st.text_input("ENTSO-E security token", value=str(scenario_defaults.get("entsoe_key", "")), type="password", disabled=not use_real_market_input)
        fingrid_key_input_form = st.text_input("Fingrid Open Data API key", value=str(scenario_defaults.get("fingrid_key", "")), type="password", disabled=not use_real_market_input)
        persist_market_data_input = st.toggle("Cache fetched market data locally", value=bool(scenario_defaults["persist_market_data"]), disabled=not use_real_market_input)
        apply_scenario = st.form_submit_button("Apply Scenario", type="primary")

    if apply_scenario:
        next_scenario = {
            "start_date": start_date_input,
            "days": int(days_input),
            "freq_minutes": int(freq_minutes_input),
            "n_homes": int(n_homes_input),
            "ev_pen": float(ev_pen_input),
            "bess_pen": float(bess_pen_input),
            "pv_pen": float(pv_pen_input),
            "hvac_pen": float(hvac_pen_input),
            "hvac_mode": hvac_mode_input,
            "hvac_response_s": float(hvac_response_s_input),
            "cloudiness": float(cloudiness_input),
            "climate_shift_c": float(climate_shift_c_input),
            "solar_scale": float(solar_scale_input),
            "scarcity": float(scarcity_input),
            "setpoint_c": float(setpoint_c_input),
            "comfort_band_c": float(comfort_band_c_input),
            "seed": int(seed_input),
            "market_source": market_source_input,
            "market_lookback_days": int(market_lookback_days_input),
            "entsoe_key": entsoe_key_input_form,
            "fingrid_key": fingrid_key_input_form,
            "persist_market_data": bool(persist_market_data_input),
        }
        if next_scenario != st.session_state["scenario_inputs"]:
            st.session_state["scenario_inputs"] = next_scenario
            st.session_state["mpc_result_stale"] = True
            st.session_state.pop("default_mpc_models", None)
            st.session_state.pop("default_mpc_signature", None)
            st.session_state.pop("live_market_session", None)
            st.session_state.pop("lower_mpc_armed", None)
            st.session_state.pop("live_lower_result", None)
            st.session_state.pop("training_market_df", None)
            st.session_state.pop("operational_market_df", None)
            st.session_state.pop("market_metadata", None)
            st.session_state.pop("market_data_signature", None)

    scenario = st.session_state["scenario_inputs"]
    start_date = scenario["start_date"]
    days = int(scenario["days"])
    freq_minutes = int(scenario["freq_minutes"])
    n_homes = int(scenario["n_homes"])
    ev_pen = float(scenario["ev_pen"])
    bess_pen = float(scenario["bess_pen"])
    pv_pen = float(scenario["pv_pen"])
    hvac_pen = float(scenario["hvac_pen"])
    hvac_mode = str(scenario["hvac_mode"])
    hvac_response_s = float(scenario["hvac_response_s"])
    cloudiness = float(scenario["cloudiness"])
    climate_shift_c = float(scenario["climate_shift_c"])
    solar_scale = float(scenario["solar_scale"])
    scarcity = float(scenario["scarcity"])
    setpoint_c = float(scenario["setpoint_c"])
    comfort_band_c = float(scenario["comfort_band_c"])
    seed = int(scenario["seed"])
    market_source = str(scenario["market_source"])
    market_lookback_days = int(scenario["market_lookback_days"])
    entsoe_key_input = str(scenario.get("entsoe_key", ""))
    fingrid_key_input = str(scenario.get("fingrid_key", ""))
    persist_market_data = bool(scenario["persist_market_data"])
    use_real_market = market_source == "Real market APIs"
    real_market_ready = use_real_market and bool(entsoe_key_input or fingrid_key_input)
    if use_real_market and not (entsoe_key_input or fingrid_key_input):
        st.sidebar.warning("Enter at least one API key and click Apply Scenario to enable real market ingestion.")

    training_market_df = st.session_state.get("training_market_df", pd.DataFrame())
    operational_market_df = st.session_state.get("operational_market_df", pd.DataFrame())
    market_metadata = st.session_state.get("market_metadata", {})
    if real_market_ready:
        market_signature = market_data_signature(scenario)
        if st.session_state.get("market_data_signature") != market_signature:
            with st.spinner("Fetching TSO market data through the market service..."):
                try:
                    market_cache = None if persist_market_data else CacheManager(ttl_seconds=900, cache_dir=None)
                    market_service = MarketDataService(entsoe_key_input.strip() or None, fingrid_key_input.strip() or None, cache=market_cache)
                    combined_market_df, market_metadata = market_service.fetch_market_data(
                        lookback_days=int(market_lookback_days),
                        include_forecast_day=True,
                    )
                    training_market_df, operational_market_df = market_service.split_training_and_operational(combined_market_df)
                    st.session_state["training_market_df"] = training_market_df
                    st.session_state["operational_market_df"] = operational_market_df
                    st.session_state["market_metadata"] = market_metadata
                    st.session_state["market_data_signature"] = market_signature
                except Exception as exc:
                    st.sidebar.warning(f"Market service could not load external data: {exc}")
                    training_market_df = pd.DataFrame()
                    operational_market_df = pd.DataFrame()
                    market_metadata = {"errors": [str(exc)], "mode_used": "synthetic"}
                    st.session_state["training_market_df"] = training_market_df
                    st.session_state["operational_market_df"] = operational_market_df
                    st.session_state["market_metadata"] = market_metadata
                    st.session_state["market_data_signature"] = market_signature
    else:
        training_market_df = pd.DataFrame()
        operational_market_df = pd.DataFrame()
        market_metadata = {}

    portfolio_args = {
        "start_date": str(start_date),
        "days": days,
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
    }
    portfolio_signature = scenario_portfolio_signature(scenario)
    if portfolio_bundle_is_current(st.session_state, portfolio_signature):
        bundle = st.session_state["portfolio_bundle"]
    else:
        spinner_text = (
            "Generating the flexibility portfolio and applying cached TSO market data..."
            if real_market_ready
            else "Generating synthetic households, weather, and balancing market conditions..."
        )
        try:
            with st.spinner(spinner_text):
                bundle = generate_portfolio_runtime(
                    **portfolio_args,
                    market_data_mode="synthetic",
                    entsoe_api_key=None,
                    fingrid_api_key=None,
                    market_lookback_days=None,
                    persist_market_data=False,
                    use_cache=not real_market_ready,
                )
        except RuntimeError as exc:
            st.error(f"Real market data could not be loaded: {exc}")
            with st.spinner("Falling back to synthetic market data..."):
                bundle = generate_portfolio_runtime(**portfolio_args, market_data_mode="synthetic", use_cache=True)
        st.session_state["portfolio_bundle"] = bundle
        st.session_state["portfolio_signature"] = portfolio_signature
    df = bundle["data"].copy()
    preview_4s = bundle["preview_4s"].copy()
    device_roster = bundle.get("device_roster", pd.DataFrame())
    summary = dict(bundle["summary"])
    if real_market_ready:
        if isinstance(operational_market_df, pd.DataFrame) and not operational_market_df.empty:
            df = merge_market_data(df, operational_market_df)
            preview_4s = merge_market_data(preview_4s, operational_market_df)
        operational_market_columns = set(operational_market_df.columns) if isinstance(operational_market_df, pd.DataFrame) else set()
        replay_columns = []
        if {"afrr_up_act_frac", "afrr_down_act_frac"} & operational_market_columns:
            replay_columns.append("afrr_signal_norm")
        service_status = {
            "mode_requested": "real",
            "mode_used": "market_service" if isinstance(operational_market_df, pd.DataFrame) and not operational_market_df.empty else "synthetic",
            "entsoe": market_metadata.get("entsoe_status", "not_configured") if isinstance(market_metadata, dict) else "not_configured",
            "fingrid": market_metadata.get("fingrid_status", "not_configured") if isinstance(market_metadata, dict) else "not_configured",
            "real_time_preview_source": "real_activation" if replay_columns else "synthetic",
            "real_time_preview_columns": replay_columns,
            "columns_loaded": list((market_metadata.get("data_completeness", {}) if isinstance(market_metadata, dict) else {}).keys()),
            "observations_loaded": {
                "training_rows": int(market_metadata.get("training_rows", 0) or 0) if isinstance(market_metadata, dict) else 0,
                "forecast_rows": int(market_metadata.get("forecast_rows", 0) or 0) if isinstance(market_metadata, dict) else 0,
            },
            "errors": list(market_metadata.get("errors", [])) if isinstance(market_metadata, dict) else [],
            "query_start_utc": str(market_metadata.get("training_period", "")).split(" to ")[0] if isinstance(market_metadata, dict) else "",
            "query_end_utc": str(market_metadata.get("training_period", "")).split(" to ")[-1] if isinstance(market_metadata, dict) else "",
            "training_observations": int(market_metadata.get("training_rows", 0) or 0) if isinstance(market_metadata, dict) else 0,
            "training_days": float(market_lookback_days),
            "timescaledb": "local_service_cache" if persist_market_data else "memory_service_cache",
            "rows_persisted": int(market_metadata.get("total_rows", 0) or 0) if isinstance(market_metadata, dict) else 0,
            "generated_at_utc": str(market_metadata.get("generated_for_date", "")) if isinstance(market_metadata, dict) else "",
        }
        summary["market_data_status"] = service_status
    market_data_status = summary.get("market_data_status", {})
    replay_source_label = market_replay_source_label(market_data_status)
    market_columns = [
        col
        for col in [
            "spot_price_eur_per_mwh",
            "fcrn_capacity_eur_per_mw_h",
            "afrr_up_capacity_eur_per_mw_h",
            "afrr_down_capacity_eur_per_mw_h",
            "afrr_up_energy_eur_per_mwh",
            "afrr_down_energy_eur_per_mwh",
            "afrr_up_act_frac",
            "afrr_down_act_frac",
        ]
        if col in df.columns
    ]

    tabs = st.tabs(
        [
            "Live Ops",
            "Markets & Data",
            "Fleet" if not dev_mode else "Fleet / Response Models",
            "Forecasting & Compliance",
            "MPC Optimizer & Market Participation",
            "Settlement & Impact",
            "Advanced / Export",
        ]
    )
    with tabs[0]:
        st.markdown(
            f"""
            <div class="ops-header">
                <div>
                    <div class="ops-kicker">MPC market console</div>
                    <div class="ops-title">Coverly VPP Operations</div>
                    <div class="ops-subtitle">Residential flexibility, market readiness, and MPC control state for Fingrid reserve products.</div>
                </div>
                <div class="ops-header-meta">
                    <span class="ops-pill">{escape(runtime_context["branch"])}</span>
                    <span class="ops-pill">{escape(runtime_context["commit"])}</span>
                    <span class="ops-pill">{escape(runtime_context["build"])}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_status_grid(
            [
                {"label": "Data source", "value": str(market_data_status.get("mode_used", "synthetic")).title(), "note": "Training inputs"},
                {"label": "Replay feed", "value": replay_source_label, "note": "Market simulation"},
                {"label": "Model rows", "value": f"{len(df):,}", "note": f"{freq_minutes}-minute resolution"},
                {"label": "Real observations", "value": f"{int(market_data_status.get('training_observations', 0) or 0):,}", "note": "Loaded from APIs"},
                {"label": "Timescale rows", "value": f"{int(market_data_status.get('rows_persisted', 0) or 0):,}", "note": "Local cache"},
            ]
        )
        if market_data_status.get("errors"):
            st.warning("Some market API signals were unavailable. Open Training Data Explorer for the exact API messages and fallback status.")
        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown('<div class="section-label">Portfolio operating envelope</div>', unsafe_allow_html=True)
            overview_fig = go.Figure()
            overview_fig.add_trace(go.Scatter(x=df.index, y=df["net_load_baseline_kw"] / 1000.0, name="Net load", line={"color": RESOURCE_COLORS["Net"], "width": 2.5}))
            overview_fig.add_trace(go.Scatter(x=df.index, y=df["fast_sym_kw"] / 1000.0, name="Fast reserve", line={"color": RESOURCE_COLORS["Frequency"]}))
            overview_fig.add_trace(go.Scatter(x=df.index, y=df["flex_up_kw"] / 1000.0, name="Upward flexibility", line={"color": RESOURCE_COLORS["PV"]}))
            overview_fig.update_layout(yaxis_title="MW")
            st.plotly_chart(apply_chart_style(overview_fig, template, height=360, title="Portfolio availability and load"), use_container_width=True)
            if dev_mode:
                st.plotly_chart(plot_sankey(summary, template), use_container_width=True, key="dev_home_sankey")
        with right:
            st.markdown('<div class="section-label">Readiness snapshot</div>', unsafe_allow_html=True)
            render_status_grid(
                [
                    {"label": "Fast reserve", "value": f"{summary['peak_fast_mw']:.2f} MW", "note": "FCR-capable headroom", "tone": "good"},
                    {"label": "Upward flex", "value": f"{summary['peak_total_up_mw']:.2f} MW", "note": "Total portfolio envelope", "tone": "good"},
                    {"label": "Homes for 1 MW", "value": f"{summary['homes_for_1mw_total']:,}", "note": "aFRR availability"},
                    {"label": "Homes for fast 1 MW", "value": f"{summary['homes_for_1mw_fast']:,}", "note": "Fast reserve screen"},
                    {"label": "PV fleet", "value": f"{summary['pv_capacity_mw']:.2f} MW", "note": "Installed capacity"},
                    {"label": "Storage energy", "value": f"{summary['bess_energy_mwh'] + summary['ev_energy_mwh']:.2f} MWh", "note": "Accessible BESS + EV"},
                    {"label": "Devices", "value": f"{int(summary.get('device_count', len(device_roster))):,}", "note": "Traceable roster"},
                    {"label": "Gateways", "value": f"{int(summary.get('gateway_count', 0)):,}", "note": "Simulated edge controllers"},
                ],
                class_name="readiness-grid",
            )
            st.markdown(
                """
                <div class="flexi-card">
                    <h4>Market checkpoints</h4>
                    <p class="small-note">
                        Minimum bids, response speed, accuracy, endurance, baseline state, and market-price inputs are tracked across the MPC workflow.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if dev_mode:
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
        st.subheader("Data Explorer")
        if real_market_ready:
            st.caption(
                f"Portfolio and weather remain scenario based. Market columns use the real API window "
                f"{market_data_status.get('query_start_utc')} to {market_data_status.get('query_end_utc')}."
            )
        else:
            st.caption(f"All data is generated locally at a {freq_minutes}-minute resolution. Adjust sidebar parameters to regenerate the portfolio.")
        export_col, explain_col = st.columns([1, 1.3])
        export_col.download_button(
            "Download active training data as CSV",
            data=df.to_csv().encode("utf-8"),
            file_name="coverly_training_data.csv",
            mime="text/csv",
        )
        explain_col.info(
            f"Market mode: {market_data_status.get('mode_used', 'synthetic')}. "
            f"Realtime replay: {replay_source_label}. "
            f"ENTSO-E: {market_data_status.get('entsoe', 'not_configured')}. "
            f"Fingrid: {market_data_status.get('fingrid', 'not_configured')}. "
            f"TimescaleDB: {market_data_status.get('timescaledb', 'not_configured')}."
        )

        if market_columns:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("ENTSO-E", str(market_data_status.get("entsoe", "not_configured")).replace("_", " ").title())
            k2.metric("Fingrid", str(market_data_status.get("fingrid", "not_configured")).replace("_", " ").title())
            k3.metric("Lookback days", f"{market_data_status.get('training_days', 0) or 0}")
            k4.metric("DB rows persisted", f"{int(market_data_status.get('rows_persisted', 0) or 0):,}")
            market_meta = pd.DataFrame(
                [
                    {
                        "Signal": col,
                        "Raw observations": market_data_status.get("observations_loaded", {}).get(col, 0),
                        "Rows in model frame": int(df[col].notna().sum()),
                    }
                    for col in market_columns
                ]
            )
            st.dataframe(styled_dataframe(market_meta), use_container_width=True, hide_index=True)
            st.plotly_chart(
                apply_chart_style(data_quality_figure(df, market_columns, template), template, height=300, title="Active-frame market data completeness"),
                use_container_width=True,
            )

            price_cols = [col for col in market_columns if col.endswith(("eur_per_mwh", "eur_per_mw_h"))]
            if price_cols:
                market_fig = signal_line_figure(df, price_cols, "Market price signals used by forecasting and MPC", "EUR")
                st.plotly_chart(
                    apply_chart_style(market_fig, template, height=360),
                    use_container_width=True,
                )

            act_cols = [col for col in market_columns if col.endswith("_act_frac")]
            if act_cols:
                act_fig = signal_line_figure(df, act_cols, "Normalized activation signals used by forecasting", "Fraction")
                st.plotly_chart(
                    apply_chart_style(act_fig, template, height=320),
                    use_container_width=True,
                )
        else:
            st.info("No external market columns were loaded. The market and activation series below are synthetic scenario data.")

        if market_data_status.get("errors"):
            with st.expander("Market data API messages", expanded=False):
                for error in market_data_status["errors"]:
                    st.warning(error)

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
        st.subheader("Fleet Response Models" if dev_mode else "Fleet Response Summary")
        c1, c2, c3, c4 = st.columns(4)
        tau_bess_s = c1.slider("Visual-only BESS time constant (s)", 1.0, 5.0, 2.0, 0.5, disabled=not dev_mode)
        tau_ev_s = c2.slider("Visual-only EV time constant (s)", 1.0, 8.0, 4.0, 0.5, disabled=not dev_mode)
        tau_hvac1_min = c3.slider("Visual-only HVAC fast mode (min)", 5.0, 15.0, 8.0, 1.0, disabled=not dev_mode)
        tau_hvac2_min = c4.slider("Visual-only HVAC slow mode (min)", 30.0, 60.0, 45.0, 1.0, disabled=not dev_mode)
        st.caption(
            f"HVAC is currently modeled as {hvac_mode}. "
            + ("Inverter mode uses the electrical response for the fast HVAC trace. " if hvac_mode == HVAC_MODES[1] else "Conventional mode uses the slower thermal second-order response. ")
            + "These sliders are response-chart diagnostics only; optimizer response uses the scenario-level delivered HVAC response time and fixed BESS/EV device classes."
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

        if dev_mode:
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
                key="dev_response_animation",
            )

            bode_cols = st.columns(2)
            with bode_cols[0]:
                bode_fig = go.Figure()
                for resource in ["BESS", "EV", "HVAC"]:
                    bode = bode_points(resource, tau_bess_s, tau_ev_s, tau_hvac1_min * 60.0, tau_hvac2_min * 60.0, hvac_mode, hvac_response_s)
                    bode_fig.add_trace(go.Scatter(x=bode["omega"], y=bode["mag_db"], name=resource))
                bode_fig.update_layout(xaxis_type="log", xaxis_title="rad/s", yaxis_title="dB")
                st.plotly_chart(apply_chart_style(bode_fig, template, height=360, title="Bode magnitude"), use_container_width=True, key="dev_bode_magnitude")
            with bode_cols[1]:
                phase_fig = go.Figure()
                for resource in ["BESS", "EV", "HVAC"]:
                    bode = bode_points(resource, tau_bess_s, tau_ev_s, tau_hvac1_min * 60.0, tau_hvac2_min * 60.0, hvac_mode, hvac_response_s)
                    phase_fig.add_trace(go.Scatter(x=bode["omega"], y=bode["phase_deg"], name=resource))
                phase_fig.update_layout(xaxis_type="log", xaxis_title="rad/s", yaxis_title="Degrees")
                st.plotly_chart(apply_chart_style(phase_fig, template, height=360, title="Bode phase"), use_container_width=True, key="dev_bode_phase")

        st.info(
            "Interpretation: BESS and EV fleets can closely track the required reserve trajectory in seconds. HVAC thermal flexibility adds volume, but its slower second-order behavior makes it more suitable for aFRR or hybrid strategies than for pure FCR-N."
        )

    with tabs[3]:
        st.markdown(
            """
            <div class="model-banner">
                <div>
                    <div class="model-banner-title">Forecasting Lab</div>
                    <div class="model-banner-note">Train point and quantile forecasts, then promote the selected model into MPC.</div>
                </div>
                <span class="model-pill">LightGBM default</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        preferred_targets = [
            "net_load_baseline_kw",
            "pv_available_kw",
            "spot_price_eur_per_mwh",
            "fcrn_capacity_eur_per_mw_h",
            "afrr_up_capacity_eur_per_mw_h",
            "afrr_down_capacity_eur_per_mw_h",
            "afrr_up_energy_eur_per_mwh",
            "afrr_down_energy_eur_per_mwh",
            "fcr_signed_act",
            "afrr_up_act_frac",
            "afrr_down_act_frac",
        ]
        target_options = numeric_signal_options(df, preferred_targets)
        preferred_predictors = [
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
        predictor_pool = numeric_signal_options(df, preferred_predictors)
        forecaster_plugins = PLUGIN_REGISTRY.list_forecasters()
        plugin_options = list(forecaster_plugins)
        selected_default_plugin = plugin_options[preferred_forecaster_index(plugin_options)] if plugin_options else DEFAULT_FORECASTER_PLUGIN
        lgbm_status = lightgbm_runtime_status(plugin_options)
        render_status_grid(
            [
                {
                    "label": "Default model",
                    "value": forecaster_display_name(selected_default_plugin, forecaster_plugins) if selected_default_plugin in forecaster_plugins else selected_default_plugin,
                    "note": "Selector resets to this build default",
                    "tone": "good" if selected_default_plugin.startswith("lightgbm") else "warn",
                },
                {
                    "label": "LightGBM runtime",
                    "value": lgbm_status["label"],
                    "note": lgbm_status["detail"],
                    "tone": lgbm_status["tone"],
                },
                {
                    "label": "Available models",
                    "value": str(len(plugin_options)),
                    "note": ", ".join(forecaster_display_name(key, forecaster_plugins) for key in plugin_options),
                },
                {
                    "label": "Probabilistic output",
                    "value": "P05-P95",
                    "note": "Bands are plotted after training and used by MPC risk policy",
                    "tone": "good",
                },
            ],
            class_name="model-status-grid",
        )
        if "lightgbm_quantile" not in plugin_options:
            st.error("LightGBM is not registered in this running app. Check that Streamlit is running this repository's app.py, not the older app.py in the parent Playground folder.")
        elif lgbm_status["tone"] == "warn":
            st.warning("The LightGBM option is visible, but this virtual environment cannot import native lightgbm. Run `pip install -r requirements.txt` inside `.venv` to enable the native LightGBM backend.")
        col1, col2, col3, col4, col5 = st.columns([1.35, 1.55, 1.65, 0.65, 0.95])
        target = col1.selectbox("Target", target_options, index=0)
        default_predictors = [
            col
            for col in ["temp_out_c", "irradiance_wm2", "base_load_kw", "pv_available_kw", "net_load_baseline_kw", "fcr_signed_act"]
            if col in predictor_pool and col != target
        ]
        if target in market_columns:
            default_predictors = [col for col in market_columns if col in predictor_pool and col != target][:5] + default_predictors[:2]
        predictors = col2.multiselect(
            "Predictors",
            predictor_pool,
            default=default_predictors,
        )
        selected_forecaster_plugin = col3.selectbox(
            "Forecasting model",
            plugin_options,
            index=preferred_forecaster_index(plugin_options),
            format_func=lambda key: forecaster_display_name(key, forecaster_plugins),
            key=f"forecasting_model_select_{APP_BUILD_ID}",
        )
        lags = col4.slider("Lag depth", 1, 8, 4)
        max_horizon_steps = max(4, min(96, int((24 * 60) / max(freq_minutes, 1))))
        horizon_steps = col5.slider(f"Forecast horizon ({freq_minutes} min steps)", 1, max_horizon_steps, min(4, max_horizon_steps))
        preview_fig = signal_line_figure(df, [target], f"Selected target history: {target}")
        st.plotly_chart(apply_chart_style(preview_fig, template, height=260), use_container_width=True)
        if real_market_ready and target in market_columns:
            st.info(
                f"This forecast target is fed by real market ingestion when available. "
                f"Raw observations for this signal: {market_data_status.get('observations_loaded', {}).get(target, 0):,}."
            )

        if st.button("Train forecast", type="primary", key=f"train_forecast_{target}"):
            with st.spinner("Training the forecaster..."):
                forecaster = PLUGIN_REGISTRY.get_forecaster(selected_forecaster_plugin, seed=int(seed))
                comparison = forecaster.fit_from_frame(df, target, predictors, lags, horizon_steps)
                try:
                    spec = forecaster.to_forecast_spec()
                except ValueError:
                    spec = forecaster
            st.session_state["last_forecast_spec"] = spec
            st.session_state["last_forecaster_instance"] = forecaster
            st.session_state["last_forecaster_plugin"] = selected_forecaster_plugin
            st.session_state["last_forecast_comparison"] = comparison
            st.session_state["last_forecast_market_status"] = market_data_status
            st.session_state["last_forecast_row_count"] = int(len(df))

        if "last_forecast_spec" in st.session_state:
            spec = st.session_state["last_forecast_spec"]
            forecaster = st.session_state.get("last_forecaster_instance")
            comparison = st.session_state["last_forecast_comparison"]
            forecast_market_status = st.session_state.get("last_forecast_market_status", {})
            spec_target = getattr(spec, "target", target)
            spec_metrics = getattr(spec, "metrics", {})
            spec_horizon_steps = getattr(spec, "horizon_steps", horizon_steps)
            if spec_target != target:
                st.info(f"The displayed trained model is for `{spec_target}`. Click Train forecast to retrain for `{target}`.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{spec_metrics['mae']:.2f}")
            m2.metric("RMSE", f"{spec_metrics['rmse']:.2f}")
            m3.metric("Rows used", f"{st.session_state.get('last_forecast_row_count', len(df)):,}")
            trained_plugin = st.session_state.get("last_forecaster_plugin", DEFAULT_FORECASTER_PLUGIN)
            m4.metric("Model", forecaster_display_name(trained_plugin, forecaster_plugins) if trained_plugin in forecaster_plugins else trained_plugin)
            if forecaster is not None and hasattr(forecaster, "get_metadata"):
                model_metadata = forecaster.get_metadata()
                backend = str(model_metadata.get("backend", "model"))
                st.caption(f"Trained backend: `{backend}`. Quantile-aware models produce the P05-P95 uncertainty bands shown below.")
            st.caption(
                f"Training frame uses {st.session_state.get('last_forecast_row_count', len(df)):,} rows at {freq_minutes}-minute resolution. "
                f"Real market fetch window: {forecast_market_status.get('query_start_utc') or 'not used'} to "
                f"{forecast_market_status.get('query_end_utc') or 'not used'}; raw market observations loaded: "
                f"{int(forecast_market_status.get('training_observations', 0) or 0):,}."
            )

            pred_fig = forecast_performance_figure(comparison, str(spec_target))
            st.plotly_chart(apply_chart_style(pred_fig, template, height=380, title=f"Forecast performance: {spec_target}"), use_container_width=True)
            quantile_cols = [col for col in comparison.columns if col.startswith("prediction_p")]
            if quantile_cols:
                with st.expander("Forecast quantiles", expanded=False):
                    st.dataframe(styled_dataframe(comparison[["actual", "prediction", *quantile_cols]].tail(48).round(3)), use_container_width=True, height=260)

            if forecaster is not None and hasattr(forecaster, "get_feature_importance"):
                importance = forecaster.get_feature_importance()
            elif hasattr(spec, "model") and hasattr(spec.model, "feature_importances_"):
                importance = pd.DataFrame({"feature": spec.model.feature_names_in_, "importance": spec.model.feature_importances_}).sort_values(
                    "importance", ascending=False
                )
            else:
                importance = pd.DataFrame(columns=["feature", "importance"])
            if not importance.empty:
                imp_fig = px.bar(importance.head(12), x="importance", y="feature", orientation="h", template=template, title="Top feature importances")
                st.plotly_chart(apply_chart_style(imp_fig, template, height=380, title="Top feature importances"), use_container_width=True)

            try:
                serializable_spec = forecaster.to_forecast_spec() if forecaster is not None and hasattr(forecaster, "to_forecast_spec") else spec
                model_bytes = serialize_forecast_spec(serializable_spec)
                st.download_button("Download trained model", data=model_bytes, file_name=f"{spec_target}_forecast.pkl", mime="application/octet-stream")
            except Exception:
                st.caption("This forecaster does not expose a legacy serializable model payload.")
            if spec_target in SUPPORTED_MPC_TARGETS and spec_horizon_steps == 1:
                if st.button("Use this model inside the MPC loop"):
                    custom = st.session_state.get("custom_mpc_models", {})
                    custom[spec_target] = forecaster or spec
                    st.session_state["custom_mpc_models"] = custom
                    st.success(f"{spec_target} is now promoted into the MPC forecasting registry.")
            else:
                st.caption("To promote a model into the MPC loop, train a 1-step model for one of the MPC targets.")

    with tabs[4]:
        st.subheader("MPC Optimizer & Market Participation")
        st.info(SIMULATION_ONLY_NOTICE)
        st.caption(
            f"The outer MPC, synthetic portfolio, and forecasting loop run at a {freq_minutes}-minute interval. "
            "First run the upper MPC socket, then start the simulated market. The lower 4-second MPC is auto-armed against the live market clock."
        )
        opt1, opt2, opt3, opt4 = st.columns(4)
        market_mode = opt1.selectbox("Market product", ["Combined", "FCR-N", "aFRR"], help="Combined allows the MPC to split the portfolio across both products.")
        resource_mode = opt2.selectbox("Resource strategy", ["Hybrid portfolio", "Fast only"], help="Fast only uses BESS + EV. Hybrid also activates HVAC and PV for aFRR.")
        horizon_options = [15, 30, 60, 120, 240, 360, 720, 1440, 2880]
        dispatch_options = [15, 30, 60, 120, 240, 360, 720, 1440, 2880, 4320]
        max_window_minutes = max(int(days * 24 * 60), 15)
        horizon_options = [minutes for minutes in horizon_options if minutes <= max_window_minutes]
        dispatch_options = [minutes for minutes in dispatch_options if minutes <= max_window_minutes]
        format_window = lambda minutes: f"{minutes} min" if minutes < 60 else f"{minutes // 60:g} h"
        horizon_minutes = opt3.select_slider("Upper planning horizon", options=horizon_options, value=60 if 60 in horizon_options else horizon_options[0], format_func=format_window)
        dispatch_minutes = opt4.select_slider("Market simulation window", options=dispatch_options, value=60 if 60 in dispatch_options else dispatch_options[0], format_func=format_window)
        horizon_hours = horizon_minutes / 60.0
        dispatch_hours = dispatch_minutes / 60.0

        w1, w2, w3 = st.columns(3)
        w1.caption("Battery/EV wear. Unit: EUR/MWh throughput.")
        degradation_w = w1.slider(
            "Battery wear cost (EUR/MWh)",
            0.0,
            150.0,
            35.0,
            5.0,
            help="Throughput cost for BESS reserve activation. EV cycling uses 40% of this value in the upper MPC.",
        )
        w2.caption("Indoor comfort slack. Unit: EUR/degC-hour.")
        comfort_w = w2.slider(
            "Comfort cost (EUR/degC-hour)",
            0.0,
            1200.0,
            480.0,
            20.0,
            help="Cost of violating the indoor comfort band. It is multiplied by temperature slack in degC and interval length in hours.",
        )
        w3.caption("Missed EV energy promise. Unit: EUR/MWh shortfall.")
        departure_w = w3.slider(
            "EV shortfall cost (EUR/MWh)",
            0.0,
            5000.0,
            1000.0,
            50.0,
            help="Penalty for missing EV required energy. One MWh equals 1000 kWh, so 1000 EUR/MWh is 1 EUR/kWh short.",
        )
        risk_policy_labels = {values["label"]: key for key, values in RISK_POLICY_PRESETS.items()}
        risk_label = st.selectbox("Bid risk policy", list(risk_policy_labels), index=0)
        risk_policy = risk_policy_labels[risk_label]
        risk_defaults = RISK_POLICY_PRESETS[risk_policy]
        st.caption(
            f"{risk_label}: P{int(float(risk_defaults['risk_quantile']) * 100)} deliverability, "
            f"{100 * float(risk_defaults['reserve_buffer_pct']):.0f}% reserve buffer. "
            "Non-delivery and activation-volatility values are reported as audit exposure, not used to kill bids."
        )
        with st.expander("Advanced risk controls"):
            r2, r3, r4 = st.columns(3)
            risk_quantile = r2.slider("Reliable bid quantile", 0.50, 0.95, float(risk_defaults["risk_quantile"]), 0.05)
            reserve_buffer_pct = r3.slider("Reserve buffer", 0.0, 0.30, float(risk_defaults["reserve_buffer_pct"]), 0.01)
            non_delivery_penalty = r4.slider(
                "Non-delivery audit cost (EUR/MWh)",
                0.0,
                3000.0,
                float(risk_defaults["non_delivery"]),
                50.0,
                help="Audit exposure for reserve capacity that may not be delivered. It is reported, not subtracted inside the bid objective.",
            )
            rr1, rr2 = st.columns(2)
            activation_uncertainty_w = rr1.slider(
                "Activation volatility audit cost (EUR/MWh-eq)",
                0.0,
                500.0,
                float(risk_defaults["activation_uncertainty"]),
                10.0,
                help="Audit exposure for uncertain activation intensity. It is reported, not subtracted inside the bid objective.",
            )
            asset_fatigue_w = rr2.slider(
                "Customer fatigue cost (EUR/MWh-eq)",
                0.0,
                500.0,
                float(risk_defaults["asset_fatigue"]),
                5.0,
                help="Objective cost for repeated use of customer devices. Resource multipliers are BESS 1.00, EV 0.65, HVAC 0.45, PV 0.20.",
            )
        forecaster_plugins = PLUGIN_REGISTRY.list_forecasters()
        optimizer_plugins = PLUGIN_REGISTRY.list_optimizers()
        mpc_forecaster_options = list(forecaster_plugins)
        p1, p2 = st.columns(2)
        mpc_forecaster_plugin = p1.selectbox(
            "MPC forecaster",
            mpc_forecaster_options,
            index=preferred_forecaster_index(mpc_forecaster_options),
            format_func=lambda key: forecaster_display_name(key, forecaster_plugins),
            key=f"mpc_forecaster_select_{APP_BUILD_ID}",
        )
        optimizer_plugin = p2.selectbox(
            "Optimizer",
            list(optimizer_plugins),
            index=list(optimizer_plugins).index(DEFAULT_OPTIMIZER_PLUGIN) if DEFAULT_OPTIMIZER_PLUGIN in optimizer_plugins else 0,
            format_func=lambda key: f"{key} ({optimizer_plugins[key]})",
        )

        st.caption(
            f"Upper MPC checks {int(max(round(horizon_hours / (freq_minutes / 60.0)), 1))} forecast/control step(s) per horizon and "
            f"{int(max(round(dispatch_hours / (freq_minutes / 60.0)), 1))} market interval(s) over the selected simulation window. "
            f"{risk_label} means the bid is derated toward P{int(risk_quantile * 100)} deliverability before market participation is accepted."
        )
        live_market_session = st.session_state.get("live_market_session", {})
        market_status = live_market_status(live_market_session)
        market_delay_seconds = st.slider("Market start delay after pressure signal (seconds)", 0, 120, 10, 1)
        btn_upper, btn_market, btn_lower, btn_stop = st.columns([1, 1, 1, 1])
        existing_upper_result = st.session_state.get("upper_mpc_result", st.session_state.get("mpc_result", {}))
        existing_history = existing_upper_result.get("history", pd.DataFrame()) if isinstance(existing_upper_result, dict) else pd.DataFrame()
        upper_market_ready = (
            isinstance(existing_history, pd.DataFrame)
            and not existing_history.empty
            and not bool(st.session_state.get("mpc_result_stale", False))
            and st.session_state.get("mpc_phase") in {"upper", "market_scheduled", "market_live", "lower_waiting", "lower_live", "lower_stopped", "lower"}
        )
        socket_metrics = upper_socket_metrics(existing_upper_result)
        upper_socket_ready = bool(upper_market_ready and socket_metrics["ready"])
        if live_market_session and upper_market_ready and not socket_metrics["ready"]:
            st.session_state.pop("live_market_session", None)
            st.session_state.pop("market_simulator", None)
            st.session_state.pop("live_market_ticks", None)
            st.session_state.pop("lower_mpc_armed", None)
            st.session_state.pop("live_lower_result", None)
            st.session_state["mpc_phase"] = "upper"
            live_market_session = {}
            market_status = live_market_status(live_market_session)
            st.warning("Stopped the live market session because the current upper MPC result has no accepted reserve socket.")
        market_started = bool(live_market_session) and market_status["status"] in {"scheduled", "live", "closed"}
        lower_already_armed = bool(st.session_state.get("lower_mpc_armed", False))
        run_upper_mpc = btn_upper.button("Run Upper MPC Layer", type="primary")
        start_market_pressure = btn_market.button("Start Simulated Market", disabled=not upper_socket_ready or market_status["status"] in {"scheduled", "live"})
        activate_lower_mpc = btn_lower.button(
            "Lower MPC Auto-Armed" if lower_already_armed else "Activate Lower MPC",
            disabled=not market_started or not upper_socket_ready or lower_already_armed,
        )
        stop_live_market = btn_stop.button("Stop Simulated Market", disabled=not bool(live_market_session) or market_status["status"] == "closed")
        if upper_market_ready and not socket_metrics["ready"]:
            st.warning(
                "No executable market period is available yet: "
                f"{socket_metrics['reason']} Accepted slots = {int(socket_metrics['accepted_slots'])}, "
                f"maximum committed reserve = {float(socket_metrics['max_committed_kw']):.1f} kW. "
                "The market and lower MPC are disabled because there is no cleared upper socket to execute."
            )
        mpc_progress_bar = st.progress(0.0)
        mpc_progress_text = st.empty()
        base_penalty_weights = {
            "degradation": degradation_w,
            "comfort": comfort_w,
            "departure": departure_w,
            "risk_policy": risk_policy,
            "risk_quantile": risk_quantile,
            "reserve_buffer_pct": reserve_buffer_pct,
            "non_delivery": non_delivery_penalty,
            "activation_uncertainty": activation_uncertainty_w,
            "asset_fatigue": asset_fatigue_w,
        }
        if run_upper_mpc:
            with st.spinner("Solving risk-aware upper MPC socket..."):
                mpc_tracker = make_progress_tracker(mpc_progress_bar, mpc_progress_text, "MPC dispatch")
                mpc_tracker(1, 100, "Preparing default forecasting models")
                models = ensure_default_models(df, int(seed), mpc_forecaster_plugin)
                optimizer = PLUGIN_REGISTRY.get_optimizer(optimizer_plugin)
                mpc_tracker(8, 100, "Forecasting models ready")
                fleet_meta = {
                    "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
                    "setpoint_c": setpoint_c,
                    "comfort_band_c": comfort_band_c,
                    "n_hvac": summary["n_hvac"],
                    "hvac_mode": hvac_mode,
                    "hvac_response_s": hvac_response_s,
                    "fcr_hvac_share_cap": 0.20,
                }
                st.session_state["mpc_config"] = {
                    "market_mode": market_mode,
                    "resource_mode": resource_mode,
                    "horizon_hours": horizon_hours,
                    "dispatch_hours": dispatch_hours,
                    "horizon_minutes": horizon_minutes,
                    "dispatch_minutes": dispatch_minutes,
                    "forecaster_plugin": mpc_forecaster_plugin,
                    "optimizer_plugin": optimizer_plugin,
                    "risk_policy": risk_policy,
                    "risk_profile": risk_label,
                    "risk_quantile": risk_quantile,
                    "reserve_buffer_pct": reserve_buffer_pct,
                    "non_delivery_penalty": non_delivery_penalty,
                    "activation_uncertainty_weight": activation_uncertainty_w,
                    "asset_fatigue_weight": asset_fatigue_w,
                    "execute_lower_mpc": False,
                    "inner_controller_mode": "mpc",
                    "inner_dt_seconds": 4,
                    "inner_mpc_horizon_seconds": 4,
                    "rotation_strategy": "usage_aware",
                    "gateway_mode": "simulated_centralized",
                }
                vpp_service = VPPOptimizerService(fleet_meta)
                st.session_state["mpc_result"] = vpp_service.run_upper_mpc(
                    portfolio_df=df,
                    operational_market_df=operational_market_df if real_market_ready else pd.DataFrame(),
                    models=models,
                    market_mode=market_mode,
                    resource_mode=resource_mode,
                    horizon_hours=horizon_hours,
                    dispatch_hours=dispatch_hours,
                    penalty_weights=base_penalty_weights,
                    preview_4s=preview_4s,
                    optimizer=optimizer,
                    device_roster=device_roster,
                    inner_controller_mode="mpc",
                    inner_dt_seconds=4,
                    inner_mpc_horizon_seconds=4,
                    rotation_strategy="usage_aware",
                    gateway_mode="simulated_centralized",
                    execute_lower_mpc=False,
                    progress_callback=lambda current, total, message: mpc_tracker(
                        8 + int(round((current / max(total, 1)) * 92)),
                        100,
                        message,
                    ),
                )
                st.session_state["vpp_optimizer_service_boundary"] = {
                    "service": "VPPOptimizerService",
                    "market_data_source": "operational_market_df" if real_market_ready else "synthetic_portfolio",
                    "operational_market_rows": int(len(operational_market_df)) if isinstance(operational_market_df, pd.DataFrame) else 0,
                }
                st.session_state["upper_mpc_result"] = st.session_state["mpc_result"]
                st.session_state.pop("live_market_session", None)
                st.session_state.pop("market_simulator", None)
                st.session_state.pop("live_market_ticks", None)
                st.session_state.pop("lower_mpc_armed", None)
                st.session_state.pop("live_lower_result", None)
                st.session_state["mpc_result_stale"] = False
                st.session_state["mpc_phase"] = "upper"
                mpc_tracker(100, 100, "Upper MPC socket ready")
                st.rerun()

        if start_market_pressure:
            if not upper_socket_ready:
                st.warning(
                    "Market Pressure was not started because the upper MPC did not clear a nonzero accepted reserve socket. "
                    "Run Upper MPC again with settings that produce accepted reserve capacity."
                )
                st.stop()
            now_s = time.time()
            st.session_state["live_market_session"] = {
                "created_at_wall_s": now_s,
                "start_at_wall_s": now_s + float(market_delay_seconds),
                "delay_seconds": float(market_delay_seconds),
                "duration_seconds": float(dispatch_minutes * 60),
                "dt_seconds": 4,
                "market_mode": market_mode,
                "resource_mode": resource_mode,
                "market_data_source": str(market_data_status.get("mode_used", "synthetic")),
                "real_time_preview_source": str(market_data_status.get("real_time_preview_source", "synthetic")),
                "real_time_preview_columns": list(market_data_status.get("real_time_preview_columns", [])),
            }
            market_feed_for_simulator = build_live_market_feed(
                preview_4s,
                st.session_state.get("upper_mpc_result", st.session_state.get("mpc_result", {})),
                int(dispatch_minutes * 60),
                dt_seconds=4,
            )
            st.session_state["market_simulator"] = MarketSimulator(
                preview_signals=market_feed_for_simulator,
                config=MarketSimulatorConfig(dt_seconds=4),
                start_wall_time_s=now_s + float(market_delay_seconds),
            )
            st.session_state["live_market_ticks"] = []
            st.session_state["lower_mpc_armed"] = True
            st.session_state.pop("live_lower_result", None)
            st.session_state["mpc_phase"] = "lower_waiting"
            st.rerun()

        if stop_live_market:
            if live_market_session:
                stopped_session = dict(live_market_session)
                stopped_session["stopped_at_wall_s"] = time.time()
                stopped_session["stopped_reason"] = "operator_stop"
                st.session_state["live_market_session"] = stopped_session
            st.session_state.pop("lower_mpc_armed", None)
            st.session_state["mpc_phase"] = "lower_stopped" if st.session_state.get("live_lower_result") else ("upper" if upper_market_ready else "idle")
            st.rerun()

        if activate_lower_mpc:
            if not upper_socket_ready:
                st.warning("Lower MPC was not armed because the upper socket is empty or did not clear the market gate.")
                st.stop()
            st.session_state["lower_mpc_armed"] = True
            st.session_state["mpc_phase"] = "lower_waiting" if market_status["status"] == "scheduled" else "market_live"
            st.rerun()

        def render_live_market_tile() -> bool:
            live_market_session = st.session_state.get("live_market_session", {})
            market_status = live_market_status(live_market_session)
            lower_armed = bool(st.session_state.get("lower_mpc_armed", False))
            if live_market_session and st.session_state.get("live_lower_result"):
                lower_armed = True
                st.session_state["lower_mpc_armed"] = True
            runtime_market_feed = (
                sync_market_simulator_feed(
                    preview_4s,
                    st.session_state.get("upper_mpc_result", {}),
                    live_market_session,
                    int(float(live_market_session.get("duration_seconds", dispatch_minutes * 60))) if live_market_session else int(dispatch_minutes * 60),
                    dt_seconds=4,
                )
                if live_market_session
                else preview_4s
            )
            if live_market_session and market_status["status"] == "live" and st.session_state.get("mpc_phase") in {"market_scheduled", "lower_waiting"}:
                st.session_state["mpc_phase"] = "market_live" if not lower_armed else "lower_waiting"
            if lower_armed and not upper_socket_ready:
                st.session_state["lower_mpc_armed"] = False
                st.session_state.pop("live_lower_result", None)
                st.warning("Lower MPC was disarmed because the current upper socket has no accepted committed reserve.")
                lower_armed = False
            if lower_armed and live_market_session and market_status["status"] in {"live", "closed"}:
                fleet_meta = {
                    "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
                    "setpoint_c": setpoint_c,
                    "comfort_band_c": comfort_band_c,
                    "n_hvac": summary["n_hvac"],
                    "hvac_mode": hvac_mode,
                    "hvac_response_s": hvac_response_s,
                }
                upper_socket = st.session_state.get("upper_mpc_result", st.session_state.get("mpc_result", {}))
                elapsed_tick = int(float(market_status["elapsed_seconds"]) // 4) + 1
                lower_result = cached_lower_mpc_from_upper(
                    df=df,
                    upper_result=upper_socket,
                    fleet_meta=fleet_meta,
                    market_mode=str(live_market_session.get("market_mode", market_mode)),
                    resource_mode=str(live_market_session.get("resource_mode", resource_mode)),
                    preview_4s=runtime_market_feed,
                    device_roster=device_roster,
                    inner_dt_seconds=4,
                    elapsed_tick=elapsed_tick,
                )
                st.session_state["live_lower_result"] = lower_result
                st.session_state["mpc_config"] = {
                    **st.session_state.get("mpc_config", {}),
                    "execute_lower_mpc": True,
                    "gateway_mode": "live_market_coupled",
                }
                st.session_state["mpc_result_stale"] = False
                st.session_state["mpc_phase"] = "lower_live"

            if live_market_session:
                status_label = "Stopped" if live_market_session.get("stopped_reason") else str(market_status["status"]).replace("_", " ").title()
                live_cols = st.columns(4)
                live_cols[0].metric("Market process", status_label)
                live_cols[1].metric("Start countdown", f"{float(market_status['countdown_seconds']):.0f} s")
                live_cols[2].metric("Market elapsed", f"{float(market_status['elapsed_seconds']):.0f} s")
                lower_ticks = len(result_frame(st.session_state.get("live_lower_result", {}), "tracking_4s"))
                live_cols[3].metric("Lower MPC", f"Live ({lower_ticks} ticks)" if st.session_state.get("live_lower_result") else ("Armed" if lower_armed else "Waiting"))
                visible_market_feed = live_visible_frame(runtime_market_feed, live_market_session)
                if visible_market_feed.empty and market_status["status"] == "scheduled":
                    st.info("Market pressure countdown is active. The live market plot will start at the scheduled wall-clock time.")
                else:
                    market_fig = go.Figure()
                    market_fig.add_trace(go.Scatter(x=visible_market_feed.index, y=visible_market_feed["fcr_signal_norm"], name="FCR signal", line={"color": RESOURCE_COLORS["Frequency"]}))
                    market_fig.add_trace(go.Scatter(x=visible_market_feed.index, y=visible_market_feed["afrr_signal_norm"], name="aFRR signal", line={"color": RESOURCE_COLORS["Net"], "dash": "dot"}))
                    market_fig.update_layout(yaxis_title="normalized signal")
                    runtime_source = str(live_market_session.get("real_time_preview_source", market_data_status.get("real_time_preview_source", "synthetic")))
                    runtime_columns = live_market_session.get("real_time_preview_columns", market_data_status.get("real_time_preview_columns", []))
                    if runtime_source == "real_activation":
                        st.caption(
                            "Market-pressure replay uses loaded historical activation overlay "
                            f"({', '.join(runtime_columns) or 'activation'}); this is still a replay, not a live Fingrid dispatch feed."
                        )
                    else:
                        st.caption("Simulated market-pressure replay for MVP/demo use; this is not a live Fingrid activation feed.")
                    st.plotly_chart(
                        apply_chart_style(market_fig, template, height=260, title="Simulated Market Pressure"),
                        use_container_width=True,
                        key="live_market_pressure_signal",
                    )
                lower_rendered = render_lower_live_trace(
                    st.session_state.get("live_lower_result", {}),
                    live_market_session,
                    template,
                    key_prefix="fragment_lower_live",
                    fallback_result=st.session_state.get("upper_mpc_result", {}),
                    title="Live lower MPC replay",
                    include_debug_table=False,
                )
                if lower_armed and not lower_rendered and market_status["status"] == "live":
                    st.info("Lower MPC is armed and will draw the live request/delivery plot as soon as the first 4-second solve finishes.")
                return bool(market_status["status"] in {"scheduled", "live"})
            return False

        current_live_session = st.session_state.get("live_market_session", {})
        current_live_status = live_market_status(current_live_session)
        if hasattr(st, "fragment"):
            @st.fragment(run_every="4s" if current_live_session and current_live_status["status"] in {"scheduled", "live"} else None)
            def live_market_fragment() -> None:
                render_live_market_tile()

            live_market_fragment()
        else:
            render_live_market_tile()
            if current_live_session and current_live_status["status"] in {"scheduled", "live"}:
                st.info("Live auto-refresh requires Streamlit fragments. Upgrade Streamlit to keep updates scoped to the live plots.")
        active_result = st.session_state.get(
            "mpc_result",
            {
                "history": pd.DataFrame(),
                "tracking_4s": pd.DataFrame(),
                "summary": {},
                "compliance": {},
                "first_schedule": pd.DataFrame(),
                "household_contributions": pd.DataFrame(),
                "appliance_contributions": pd.DataFrame(),
                "upper_device_schedule": pd.DataFrame(),
                "gateway_commands": pd.DataFrame(),
                "usage_fatigue_summary": pd.DataFrame(),
            },
        )
        upper_view_result = st.session_state.get("upper_mpc_result", {})
        if result_history(upper_view_result).empty:
            upper_view_result = active_result
        lower_view_result = st.session_state.get("live_lower_result", {})
        if result_history(lower_view_result).empty and result_summary(active_result).get("execute_lower_mpc"):
            lower_view_result = active_result
        result = upper_view_result
        result_df = result_history(upper_view_result)
        if result_df.empty:
            result = active_result
            result_df = result_history(active_result)
        upper_result_df = result_history(upper_view_result)
        lower_result_df = result_history(lower_view_result)
        upper_plot_df = upper_result_df if not upper_result_df.empty else result_df

        if st.session_state.get("mpc_result_stale") and not result_df.empty:
            st.warning("Scenario inputs changed. The charts below still show the previous MPC run until you run dispatch again.")

        if result_df.empty:
            st.info("Configure the scenario and click Run Upper MPC Layer. Market pressure and lower MPC attach only after their own buttons are pressed.")
        else:
            s = result_summary(result)
            upper_compliance = upper_view_result.get("compliance", {}) if isinstance(upper_view_result, dict) else {}
            lower_compliance = lower_view_result.get("compliance", {}) if isinstance(lower_view_result, dict) else {}
            lower_tracking_for_compliance = result_frame(lower_view_result, "tracking_4s")
            if not lower_tracking_for_compliance.empty and lower_compliance:
                c = dict(upper_compliance)
                c.update(lower_compliance)
                lower_summary = result_summary(lower_view_result)
                s = {**s, **lower_summary}
                compliance_source = "lower 4-second replay audit"
            else:
                c = dict(upper_compliance)
                compliance_source = "upper bid-screen; lower replay pending"
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total revenue", fmt_money(float(s.get("total_revenue_eur", 0.0))))
            m2.metric("Delivered up energy", f"{float(s.get('delivered_up_mwh', 0.0)):.2f} MWh")
            m3.metric("Requirement score", f"{float(s.get('requirement_score_pct', 0.0)):.0f}%")
            m4.metric("Comfort violations", f"{float(s.get('comfort_violations_h', 0.0)):.1f} h")
            upper_socket_view_metrics = upper_socket_metrics(upper_view_result)
            mpc_status_cols = st.columns(5)
            solver_mode = result_df["solver_status"].mode().iloc[0] if "solver_status" in result_df and not result_df["solver_status"].empty else "unknown"
            inner_mode = result_df["inner_solver_status"].mode().iloc[0] if "inner_solver_status" in result_df and not result_df["inner_solver_status"].empty else "unknown"
            mpc_status_cols[0].metric("Upper MPC status", solver_mode)
            mpc_status_cols[1].metric("MPC intervals", f"{len(result_df):,}")
            mpc_status_cols[2].metric("Upper socket", "Ready" if upper_socket_view_metrics["ready"] else "Empty")
            mpc_status_cols[3].metric("4-sec MPC status", inner_mode if s.get("execute_lower_mpc") else "Not started")
            mpc_status_cols[4].metric("Mean shortfall", f"{float(result_df.get('shortfall_kw', pd.Series([0.0])).mean()):.1f} kW")
            risk_cols = st.columns(4)
            risk_cols[0].metric("Risk policy", st.session_state.get("mpc_config", {}).get("risk_profile", "P80"))
            risk_cols[1].metric("Accepted market slots", f"{int(s.get('market_gate_participation_intervals', 0)):,}")
            risk_cols[2].metric("Risk-adjusted profit", fmt_money(float(s.get("risk_adjusted_profit_eur", 0.0))))
            risk_cols[3].metric("Delivery risk cost", fmt_money(float(s.get("non_delivery_risk_cost_eur", 0.0))))
            if not upper_socket_view_metrics["ready"]:
                st.warning(
                    f"Upper socket is empty: {upper_socket_view_metrics['reason']} "
                    f"Accepted slots = {int(upper_socket_view_metrics['accepted_slots'])}, "
                    f"maximum committed reserve = {float(upper_socket_view_metrics['max_committed_kw']):.1f} kW, "
                    f"mean buffer = {float(upper_socket_view_metrics['mean_buffer_kw']):.1f} kW. "
                    "A flat lower MPC plot is expected until the upper market decision clears nonzero reserve."
                )
            debug_cols = [
                "market_gate_status",
                "market_gate_reason",
                "fcr_bid_kw",
                "afrr_up_bid_kw",
                "afrr_down_bid_kw",
                "capacity_revenue_eur",
                "activation_revenue_eur",
                "degradation_cost_eur",
                "comfort_penalty_eur",
                "non_delivery_risk_cost_eur",
                "activation_uncertainty_cost_eur",
                "asset_fatigue_cost_eur",
                "risk_adjusted_profit_eur",
                "fcrn_capacity_eur_per_mw_h",
                "afrr_up_capacity_eur_per_mw_h",
                "afrr_down_capacity_eur_per_mw_h",
            ]
            available_debug_cols = [col for col in debug_cols if col in upper_plot_df.columns]
            if available_debug_cols:
                with st.expander("Upper MPC market debug", expanded=not upper_socket_view_metrics["ready"]):
                    st.caption("This is the upper-layer market gate audit for each interval. It shows whether the block is caused by zero bids, minimum bid size, or risk-adjusted economics.")
                    st.dataframe(styled_dataframe(upper_plot_df[available_debug_cols].head(24).round(3)), use_container_width=True, height=280)
            st.info(
                f"MPC workflow: `{st.session_state.get('mpc_config', {}).get('forecaster_plugin', DEFAULT_FORECASTER_PLUGIN)}` forecasts each horizon, "
                f"`{st.session_state.get('mpc_config', {}).get('optimizer_plugin', DEFAULT_OPTIMIZER_PLUGIN)}` chooses reserve bids and resource dispatch, "
                "the market clock runs as a separate pressure process, and the lower 4-second tracker attaches to that live market state inside the upper socket."
            )
            forecast_trace = result_frame(upper_view_result, "forecast_trace").copy()
            if not forecast_trace.empty:
                trace_targets = [target_name for target_name in SUPPORTED_MPC_TARGETS if target_name in forecast_trace.columns]
                if trace_targets:
                    revision_controls = st.columns([1.2, 0.8, 1.0])
                    default_trace_target = "fcrn_capacity_eur_per_mw_h" if "fcrn_capacity_eur_per_mw_h" in trace_targets else trace_targets[0]
                    revision_target = revision_controls[0].selectbox(
                        "MPC forecast target",
                        trace_targets,
                        index=trace_targets.index(default_trace_target),
                        key="mpc_forecast_revision_target",
                    )
                    iteration_count = (
                        int(pd.to_numeric(forecast_trace["mpc_iteration"], errors="coerce").dropna().nunique())
                        if "mpc_iteration" in forecast_trace
                        else 0
                    )
                    max_iterations = revision_controls[1].slider(
                        "Forecast issues",
                        2,
                        max(2, min(12, iteration_count)),
                        min(6, max(2, min(12, iteration_count))),
                        key="mpc_forecast_revision_count",
                    )
                    if {"mpc_iteration", "horizon_step"}.issubset(forecast_trace.columns):
                        latest_rows = forecast_trace.sort_values(["mpc_iteration", "horizon_step"]).tail(1)
                    else:
                        latest_rows = pd.DataFrame()
                    latest_observations = int(latest_rows["observations_available"].iloc[0]) if "observations_available" in latest_rows and not latest_rows.empty else 0
                    revision_controls[2].metric("Latest observed rows", f"{latest_observations:,}")
                    revision_fig = forecast_revision_figure(forecast_trace, revision_target, max_iterations)
                    st.plotly_chart(
                        apply_chart_style(revision_fig, template, height=360, title="MPC forecast revisions"),
                        use_container_width=True,
                    )

            left, right = st.columns([1.45, 1.0])
            with left:
                st.plotly_chart(
                    apply_chart_style(upper_mpc_decision_figure(upper_plot_df), template, height=520, title="Upper MPC decision console"),
                    use_container_width=True,
                )
                waterfall_cols = {"raw_fcr_symmetric_kw", "fcr_dynamic_cap_total_kw", "fcr_energy_cap_total_kw", "reserve_buffer_kw", "fcr_bid_kw"}
                if waterfall_cols.issubset(upper_plot_df.columns):
                    if "market_gate_status" in upper_plot_df and (upper_plot_df["market_gate_status"].astype(str) == "Participate").any():
                        waterfall_row = upper_plot_df[upper_plot_df["market_gate_status"].astype(str) == "Participate"].iloc[0]
                    else:
                        waterfall_row = upper_plot_df.iloc[0]
                    raw_kw = float(waterfall_row["raw_fcr_symmetric_kw"])
                    dynamic_kw = min(raw_kw, float(waterfall_row["fcr_dynamic_cap_total_kw"]))
                    energy_kw = min(dynamic_kw, float(waterfall_row["fcr_energy_cap_total_kw"]))
                    buffer_kw = float(waterfall_row["reserve_buffer_kw"])
                    granular_kw = float(waterfall_row["fcr_bid_kw"])
                    bid_waterfall = go.Figure(
                        go.Waterfall(
                            measure=["absolute", "relative", "relative", "relative", "absolute"],
                            x=["Raw symmetric", "Dynamic cap", "Energy cap", "Risk buffer", "Final bid"],
                            y=[
                                raw_kw / 1000.0,
                                (dynamic_kw - raw_kw) / 1000.0,
                                (energy_kw - dynamic_kw) / 1000.0,
                                -buffer_kw / 1000.0,
                                granular_kw / 1000.0,
                            ],
                        )
                    )
                    bid_waterfall.update_layout(yaxis_title="MW")
                    st.plotly_chart(apply_chart_style(bid_waterfall, template, height=320, title="Why the FCR-N bid was reduced"), use_container_width=True)
                    st.caption(f"Waterfall uses one market interval: {pd.Timestamp(waterfall_row.name)}.")
                bidirectional_rows = []
                for resource, fcr_col, up_col, down_col in [
                    ("BESS", "bess_fcr_kw", "bess_afrr_up_kw", "bess_afrr_down_kw"),
                    ("EV", "ev_fcr_kw", "ev_afrr_up_kw", "ev_afrr_down_kw"),
                    ("HVAC", "hvac_fcr_kw", "hvac_afrr_up_kw", "hvac_afrr_down_kw"),
                ]:
                    if {fcr_col, up_col, down_col}.issubset(upper_plot_df.columns):
                        bidirectional_rows.append({"Resource": resource, "Direction": "FCR-N symmetric", "MW": float(upper_plot_df[fcr_col].mean()) / 1000.0})
                        bidirectional_rows.append({"Resource": resource, "Direction": "aFRR up", "MW": float(upper_plot_df[up_col].mean()) / 1000.0})
                        bidirectional_rows.append({"Resource": resource, "Direction": "aFRR down", "MW": -float(upper_plot_df[down_col].mean()) / 1000.0})
                if "pv_afrr_down_kw" in upper_plot_df:
                    bidirectional_rows.append({"Resource": "PV", "Direction": "aFRR down", "MW": -float(upper_plot_df["pv_afrr_down_kw"].mean()) / 1000.0})
                if bidirectional_rows:
                    bidirectional_fig = px.bar(
                        pd.DataFrame(bidirectional_rows),
                        x="Resource",
                        y="MW",
                        color="Direction",
                        barmode="relative",
                        template=template,
                        title="Risk-adjusted bidirectional resource commitment",
                    )
                    bidirectional_fig.update_layout(yaxis_title="MW, up positive / down negative")
                    st.plotly_chart(apply_chart_style(bidirectional_fig, template, height=320), use_container_width=True)
            with right:
                render_status_grid(
                    [
                        {
                            "label": "Average FCR bid",
                            "value": f"{float(upper_plot_df['fcr_bid_kw'].mean()):.0f} kW",
                            "note": "Minimum threshold 100 kW",
                            "tone": "good" if float(upper_plot_df["fcr_bid_kw"].mean()) >= 100.0 else "warn",
                        },
                        {
                            "label": "Average aFRR bid",
                            "value": f"{float(np.maximum(upper_plot_df['afrr_up_bid_kw'], upper_plot_df['afrr_down_bid_kw']).mean()):.0f} kW",
                            "note": "Minimum threshold 1000 kW",
                            "tone": "good" if float(np.maximum(upper_plot_df["afrr_up_bid_kw"], upper_plot_df["afrr_down_bid_kw"]).mean()) >= 1000.0 else "warn",
                        },
                    ],
                    class_name="readiness-grid",
                )
                st.markdown(
                    f"""
                    <div class="flexi-card">
                        <h4>Compliance snapshot</h4>
                        <p class="small-note">
                            FCR response estimate: <b>{float(c.get('fcr_response_s', 0.0)):.1f} s</b><br/>
                            aFRR response estimate: <b>{float(c.get('afrr_response_s', 0.0)):.0f} s</b><br/>
                            aFRR min/max ratio: <b>{float(c.get('afrr_min_accuracy_ratio', 0.0)):.2f} / {float(c.get('afrr_max_accuracy_ratio', 0.0)):.2f}</b><br/>
                            FCR stability screen: <b>{float(c.get('fcr_stability_margin_pct', 0.0)):.1f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            first_schedule = result_frame(upper_view_result, "first_schedule").copy()
            if not first_schedule.empty:
                first_schedule.index.name = "timestamp"
                st.caption("First upper MPC horizon snapshot. Revenue, delivery risk, activation uncertainty, and fatigue costs are shown before the lower 4-second MPC is started.")
                st.dataframe(styled_dataframe(first_schedule.head(12).round(2)), use_container_width=True, height=260)

            if {"market_gate_status", "risk_adjusted_profit_eur", "reserve_buffer_kw"}.issubset(upper_plot_df.columns):
                gate_cols = ["market_gate_status", "market_gate_reason", "risk_adjusted_profit_eur", "reserve_buffer_kw", "non_delivery_risk_cost_eur", "activation_uncertainty_cost_eur", "asset_fatigue_cost_eur"]
                st.markdown("### Market gate decisions")
                st.dataframe(
                    styled_dataframe(upper_plot_df[[col for col in gate_cols if col in upper_plot_df.columns]].head(24).round(3)),
                    use_container_width=True,
                    height=260,
                )

            rule_items = [
                ("FCR-N minimum 0.1 MW", "fcr_min_bid_ok"),
                ("aFRR minimum 1 MW", "afrr_min_bid_ok"),
                ("FCR-N fast response", "fcr_response_ok"),
                ("FCR-N 4-sec trace response", "fcr_actual_dynamic_response_ok"),
                ("FCR-N stability screen", "fcr_stability_margin_ok"),
                ("FCR-N prequalification evidence pack", "fcr_prequalification_evidence_ok"),
                ("aFRR starts within 30 s", "afrr_start_within_30s_ok"),
                ("aFRR full activation within 5 min", "afrr_full_activation_5min_ok"),
                ("aFRR 90-110% tracking envelope", "accuracy_ok"),
                ("aFRR energy reporting +/-10% per ISP", "afrr_energy_reporting_ok"),
                ("Storage 1 h endurance per direction", "storage_endurance_ok"),
                ("Baseline methodology available", "baseline_method_ok"),
            ]
            def compliance_status(key: str, passed: bool) -> str:
                if key in {"afrr_start_within_30s_ok", "afrr_full_activation_5min_ok", "accuracy_ok"}:
                    if key == "afrr_full_activation_5min_ok" and bool(c.get("afrr_full_activation_pending", False)):
                        return "Pending 5-min replay evidence"
                    if not bool(c.get("afrr_tracking_audit_evaluated", True)):
                        return "Capability pass; replay pending" if passed else "Pending lower replay"
                if key == "afrr_energy_reporting_ok" and not bool(c.get("afrr_energy_reporting_evaluated", True)):
                    return "Pending complete 15-min ISP"
                if key == "fcr_actual_dynamic_response_ok" and not bool(c.get("fcr_tracking_response_evaluated", True)):
                    return "Pending 180-s FCR replay"
                return "Pass" if passed else "Needs attention"

            rule_rows = []
            for rule, key in rule_items:
                passed = bool(c.get(key, False))
                rule_rows.append({"Rule": rule, "Pass": passed, "Status": compliance_status(key, passed)})
            st.caption(f"Compliance source: {compliance_source}. Timing rows can pass on resource capability while live replay evidence is still accumulating.")
            rule_df = pd.DataFrame(rule_rows)
            st.dataframe(styled_dataframe(rule_df), use_container_width=True, hide_index=True)

            household_contrib = result_frame(result, "household_contributions")
            appliance_contrib = result_frame(result, "appliance_contributions")
            upper_device_schedule = result_frame(upper_view_result, "upper_device_schedule")
            if upper_device_schedule.empty:
                upper_device_schedule = result_frame(result, "upper_device_schedule")
            usage_fatigue = result_frame(result, "usage_fatigue_summary")
            if not household_contrib.empty and not appliance_contrib.empty:
                st.markdown("### Usage-aware household selection")
                select_left, select_right = st.columns([1.15, 1.0])
                top_households = household_contrib.head(12).copy()
                select_left.plotly_chart(
                    apply_chart_style(
                        px.bar(
                            top_households,
                            x="household_id",
                            y="total_energy_kwh",
                            color="active_seconds",
                            labels={
                                "household_id": "Household",
                                "total_energy_kwh": "Energy contribution (kWh)",
                                "active_seconds": "Active seconds",
                            },
                            template=template,
                        ),
                        template,
                        height=340,
                        title="Top participating households",
                    ),
                    use_container_width=True,
                )
                device_mix = (
                    appliance_contrib.groupby("device_type", as_index=False)["total_energy_kwh"]
                    .sum()
                    .sort_values("total_energy_kwh", ascending=False)
                )
                select_right.plotly_chart(
                    apply_chart_style(
                        px.bar(
                            device_mix,
                            x="device_type",
                            y="total_energy_kwh",
                            color="device_type",
                            labels={"device_type": "Appliance", "total_energy_kwh": "Energy contribution (kWh)"},
                            template=template,
                        ),
                        template,
                        height=340,
                        title="Selected appliance mix",
                    ),
                    use_container_width=True,
                )
                table_cols = [
                    "household_id",
                    "device_id",
                    "device_type",
                    "gateway_id",
                    "total_energy_kwh",
                    "active_seconds",
                    "activation_count",
                    "max_abs_command_kw",
                ]
                st.dataframe(
                    styled_dataframe(appliance_contrib[[col for col in table_cols if col in appliance_contrib.columns]].head(30).round(3)),
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )
            if not upper_device_schedule.empty and "outer_interval" in upper_device_schedule:
                schedule_mix = (
                    upper_device_schedule.groupby(["outer_interval", "device_type"], as_index=False)["device_id"]
                    .nunique()
                    .rename(columns={"device_id": "selected_devices"})
                )
                st.plotly_chart(
                    apply_chart_style(
                        px.bar(
                            schedule_mix,
                            x="outer_interval",
                            y="selected_devices",
                            color="device_type",
                            labels={"outer_interval": "Outer MPC interval", "selected_devices": "Selected devices"},
                            template=template,
                        ),
                        template,
                        height=320,
                        title="Upper MPC selected devices over time",
                    ),
                    use_container_width=True,
                )
            if not usage_fatigue.empty:
                fatigue_cols = ["household_id", "device_id", "device_type", "usage_hours", "cooldown_seconds", "throughput_kwh", "fatigue_score"]
                st.caption("Usage and fatigue counters drive the rotation strategy so the same appliance is not selected for the full dispatch horizon.")
                st.dataframe(
                    styled_dataframe(usage_fatigue[[col for col in fatigue_cols if col in usage_fatigue.columns]].head(20).round(3)),
                    use_container_width=True,
                    hide_index=True,
                    height=260,
                )

            lower_tracking_full = (
                pd.DataFrame()
                if st.session_state.get("live_market_session")
                else result_frame(lower_view_result, "tracking_4s")
            )
            if not lower_tracking_full.empty:
                st.markdown("### Lower MPC 4-second Optimization Trace")
                live_lower_tracking = live_visible_frame(lower_tracking_full, st.session_state.get("live_market_session", {}))
                lower_visible = live_lower_tracking if not live_lower_tracking.empty else lower_tracking_full
                lm1, lm2, lm3, lm4 = st.columns(4)
                metric_frame = lower_visible
                lm1.metric("Visible 4-sec ticks", f"{len(metric_frame):,}")
                lm2.metric("Visible mean abs error", f"{float(metric_frame['tracking_error_kw'].abs().mean()):.2f} kW")
                lm3.metric("Visible max latency", f"{float(metric_frame.get('control_latency_ms', pd.Series([0.0])).max()):.1f} ms")
                lm4.metric("Visible recovery ticks", f"{int(metric_frame.get('recovery_mode', pd.Series(dtype=bool)).astype(bool).sum()):,}")

                st.plotly_chart(
                    apply_chart_style(lower_mpc_execution_figure(lower_visible), template, height=560, title="Lower MPC execution console"),
                    use_container_width=True,
                    key="lower_live_execution_console",
                )

                afrr_energy_audit = result_frame(lower_view_result, "afrr_energy_audit")
                if afrr_energy_audit.empty:
                    afrr_energy_audit = result_frame(result, "afrr_energy_audit")
                if not afrr_energy_audit.empty:
                    st.markdown("### aFRR settlement-period energy audit")
                    energy_audit_fig = go.Figure()
                    energy_audit_fig.add_trace(go.Bar(x=afrr_energy_audit["isp_start"], y=afrr_energy_audit["afrr_reported_energy_mwh"], name="BSP reported aFRR energy"))
                    energy_audit_fig.add_trace(go.Bar(x=afrr_energy_audit["isp_start"], y=afrr_energy_audit["afrr_fingrid_calculated_energy_mwh"], name="Fingrid-calculated proxy"))
                    energy_audit_fig.add_trace(
                        go.Scatter(
                            x=afrr_energy_audit["isp_start"],
                            y=afrr_energy_audit["afrr_energy_reporting_diff_pct"],
                            name="absolute diff %",
                            yaxis="y2",
                            line={"color": RESOURCE_COLORS["Frequency"], "width": 3},
                        )
                    )
                    energy_audit_fig.update_layout(
                        barmode="group",
                        yaxis_title="MWh",
                        yaxis2={"title": "diff ratio", "overlaying": "y", "side": "right", "range": [0, max(0.12, float(afrr_energy_audit["afrr_energy_reporting_diff_pct"].max()) * 1.2)]},
                    )
                    st.plotly_chart(apply_chart_style(energy_audit_fig, template, height=320, title="aFRR delivered energy reconciliation"), use_container_width=True, key="lower_live_afrr_energy_audit")

                lower_cols = [
                    "inner_solver_status",
                    "optimization_strategy",
                    "requested_signed_kw",
                    "requested_up_kw",
                    "requested_down_kw",
                    "delivered_signed_kw",
                    "delivered_up_kw",
                    "delivered_down_kw",
                    "ideal_delivered_kw",
                    "telemetry_error_kw",
                    "raw_request_kw",
                    "socket_up_kw",
                    "socket_down_kw",
                    "socket_violation_up_kw",
                    "socket_violation_down_kw",
                    "tracking_error_kw",
                    "tracking_tolerance_kw",
                    "active_capacity_kw",
                    "buffer_capacity_kw",
                    "buffer_used_kw",
                    "fast_bridge_used_kw",
                    "afrr_accuracy_ratio",
                    "afrr_reporting_error_kw",
                    "fcr_response_ratio",
                    "recovery_mode",
                    "error_rising",
                    "control_latency_ms",
                    "control_deadline_met",
                ]
                st.dataframe(
                    styled_dataframe(lower_tracking_full[[col for col in lower_cols if col in lower_tracking_full.columns]].head(80).round(3)),
                    use_container_width=True,
                    height=320,
                )

    with tabs[5]:
        st.subheader("Settlement & Impact")
        result = st.session_state.get("upper_mpc_result", st.session_state.get("mpc_result"))
        lower_result_for_viz = st.session_state.get("live_lower_result", {})
        if not result or result["history"].empty:
            st.info("Run the MPC tab to populate the simulation results.")
        else:
            result_df = result["history"]
            lower_tracking_source = result_frame(lower_result_for_viz, "tracking_4s")
            if lower_tracking_source.empty:
                lower_tracking_source = result.get("tracking_4s", pd.DataFrame())
            visible_tracking_df = live_visible_frame(lower_tracking_source, st.session_state.get("live_market_session", {}))
            tracking_df = visible_tracking_df if not visible_tracking_df.empty else lower_tracking_source
            viz_df = tracking_df if not tracking_df.empty else result_df
            contribution_df = viz_df
            s = result["summary"]
            ts_left, ts_right = st.columns([1.45, 1.0])
            ts_left.plotly_chart(
                apply_chart_style(
                    line_dual_axis(viz_df, template),
                    template,
                    height=420,
                    title="Optimized load, baseline, and frequency",
                ),
                use_container_width=True,
            )

            contribution_fig = go.Figure()
            contribution_fig.add_trace(go.Scatter(x=contribution_df.index, y=contribution_df["bess_up_kw"] / 1000.0, name="BESS up", stackgroup="up", line={"color": RESOURCE_COLORS["BESS"]}))
            contribution_fig.add_trace(go.Scatter(x=contribution_df.index, y=contribution_df["ev_up_kw"] / 1000.0, name="EV up", stackgroup="up", line={"color": RESOURCE_COLORS["EV"]}))
            contribution_fig.add_trace(go.Scatter(x=contribution_df.index, y=contribution_df["hvac_up_kw"] / 1000.0, name="HVAC up", stackgroup="up", line={"color": RESOURCE_COLORS["HVAC"]}))
            contribution_fig.add_trace(go.Scatter(x=contribution_df.index, y=-contribution_df["bess_down_kw"] / 1000.0, name="BESS down", stackgroup="down", line={"color": RESOURCE_COLORS["BESS"], "dash": "dot"}))
            contribution_fig.add_trace(go.Scatter(x=contribution_df.index, y=-contribution_df["ev_down_kw"] / 1000.0, name="EV down", stackgroup="down", line={"color": RESOURCE_COLORS["EV"], "dash": "dot"}))
            contribution_fig.add_trace(go.Scatter(x=contribution_df.index, y=-contribution_df["hvac_down_kw"] / 1000.0, name="HVAC down", stackgroup="down", line={"color": RESOURCE_COLORS["HVAC"], "dash": "dot"}))
            contribution_fig.add_trace(go.Scatter(x=contribution_df.index, y=-contribution_df["pv_down_kw"] / 1000.0, name="PV curtailment", stackgroup="down", line={"color": RESOURCE_COLORS["PV"]}))
            contribution_fig.update_layout(yaxis_title="MW, up positive / down negative")
            ts_right.plotly_chart(apply_chart_style(contribution_fig, template, height=420, title="Bidirectional resource contribution"), use_container_width=True)
            if not tracking_df.empty:
                st.caption(f"Simulation plots are using {len(tracking_df):,} live 4-second lower-MPC ticks. Upper market history still contains {len(result_df):,} outer interval(s).")

            row1 = st.columns(3)
            row1[0].metric("Capacity revenue", fmt_money(s["capacity_revenue_eur"]))
            row1[1].metric("Activation revenue", fmt_money(s["activation_revenue_eur"]))
            row1[2].metric("CO2 avoided (proxy)", f"{s['co2_avoided_kg']:.0f} kg")

            charts = st.columns(3)
            revenue_fig = bar_comparison(s["resource_revenue"], "Revenue by resource")
            fast_slow_fig = bar_comparison(s["fast_vs_slow"], "Fast vs slow revenue contribution")
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
            upper_bid_heatmap_df = result_df
            heat_cols[0].plotly_chart(
                apply_chart_style(availability_heatmap(upper_bid_heatmap_df, "fcr_bid_kw", "Upper bid heatmap (market intervals)", "PuBu"), template, height=320),
                use_container_width=True,
                key="settlement_upper_bid_heatmap",
            )
            if not tracking_df.empty:
                lower_success = tracking_df.assign(success=(tracking_df["delivered_up_kw"] >= 0.9 * tracking_df["requested_up_kw"]).astype(int))
                heat_cols[1].plotly_chart(
                    apply_chart_style(availability_heatmap(lower_success, "success", "Lower tracking success heatmap (4-sec ticks)", "Greens"), template, height=320),
                    use_container_width=True,
                    key="settlement_lower_success_heatmap",
                )
            else:
                bid_success = result_df.assign(success=(result_df.get("market_gate_status", pd.Series(index=result_df.index, dtype=object)).astype(str) == "Participate").astype(int))
                heat_cols[1].plotly_chart(
                    apply_chart_style(availability_heatmap(bid_success, "success", "Upper market-gate success heatmap (market intervals)", "Greens"), template, height=320),
                    use_container_width=True,
                    key="settlement_upper_success_heatmap",
                )

            live_settlement_session = st.session_state.get("live_market_session", {})

            def render_settlement_live_lower_replay() -> None:
                fresh_lower_result = st.session_state.get("live_lower_result", {})
                fresh_tracking = result_frame(fresh_lower_result, "tracking_4s")
                current_status = live_market_status(live_settlement_session)
                if fresh_tracking.empty:
                    if current_status["status"] == "scheduled":
                        st.info("Market pressure countdown is active. Settlement lower-MPC plots will attach when the live market starts.")
                    else:
                        st.info("Waiting for live lower-MPC tracking ticks from the MPC tab.")
                    with st.expander("Debug: Lower result structure", expanded=False):
                        st.write("patch: settlement_live_fragment")
                        st.write(f"live_market_status: {current_status}")
                        st.write(f"live_lower_result type: {type(fresh_lower_result).__name__}")
                        if isinstance(fresh_lower_result, dict):
                            st.write(f"live_lower_result keys: {list(fresh_lower_result.keys())}")
                            for key, value in fresh_lower_result.items():
                                if isinstance(value, pd.DataFrame):
                                    st.write(f"{key}: DataFrame shape {value.shape}")
                                else:
                                    st.write(f"{key}: {type(value).__name__}")
                        st.write(f"fresh_tracking shape: {getattr(fresh_tracking, 'shape', None)}")
                    return

                visible_tracking = live_visible_frame(fresh_tracking, live_settlement_session)
                displayed_tracking = visible_tracking if not visible_tracking.empty else fresh_tracking
                displayed_lower_result = {**fresh_lower_result, "tracking_4s": displayed_tracking}
                st.caption(
                    f"Settlement live replay is using {len(displayed_tracking):,} current 4-second lower-MPC ticks. "
                    f"Stored lower trace contains {len(fresh_tracking):,} tick(s)."
                )
                render_lower_live_trace(
                    displayed_lower_result,
                    {},
                    template,
                    key_prefix="settlement_fragment_lower",
                    fallback_result=result,
                    title="Settlement live lower MPC replay",
                    include_debug_table=False,
                )
                gateway_commands = result_frame(fresh_lower_result, "gateway_commands")
                if gateway_commands.empty:
                    gateway_commands = result.get("gateway_commands", pd.DataFrame())
                if not gateway_commands.empty:
                    device_types = sorted(gateway_commands["device_type"].dropna().unique())
                    device_filter = st.multiselect(
                        "Gateway command appliance filter",
                        device_types,
                        default=device_types,
                        key="settlement_live_gateway_filter",
                    )
                    command_view = gateway_commands[gateway_commands["device_type"].isin(device_filter)] if device_filter else gateway_commands
                    command_cols = [
                        "outer_interval",
                        "household_id",
                        "device_id",
                        "device_type",
                        "selection_role",
                        "gateway_id",
                        "up_energy_kwh",
                        "down_energy_kwh",
                        "active_seconds",
                        "activation_count",
                        "max_abs_command_kw",
                    ]
                    st.dataframe(
                        styled_dataframe(command_view[[col for col in command_cols if col in command_view.columns]].head(40).round(3)),
                        use_container_width=True,
                        hide_index=True,
                        height=310,
                    )

            if live_settlement_session and hasattr(st, "fragment"):
                settlement_status = live_market_status(live_settlement_session)

                @st.fragment(run_every="4s" if settlement_status["status"] in {"scheduled", "live"} else None)
                def settlement_live_lower_fragment() -> None:
                    render_settlement_live_lower_replay()

                settlement_live_lower_fragment()
            elif not tracking_df.empty:
                st.markdown("### Centralized 4-second MPC")
                st.caption("This view shows the centralized lower-layer MPC following the reserve request through simulated gateway commands.")
                window_ticks = min(len(tracking_df), max(450, int(3600 / 4)))
                tracking_window = tracking_df.tail(window_ticks)
                st.plotly_chart(
                    apply_chart_style(lower_mpc_execution_figure(tracking_window), template, height=560, title="Lower MPC execution console"),
                    use_container_width=True,
                    key="settlement_lower_execution_console",
                )

                gateway_commands = result_frame(lower_result_for_viz, "gateway_commands")
                if gateway_commands.empty:
                    gateway_commands = result.get("gateway_commands", pd.DataFrame())
                if not gateway_commands.empty:
                    device_filter = st.multiselect(
                        "Gateway command appliance filter",
                        sorted(gateway_commands["device_type"].dropna().unique()),
                        default=sorted(gateway_commands["device_type"].dropna().unique()),
                    )
                    command_view = gateway_commands[gateway_commands["device_type"].isin(device_filter)] if device_filter else gateway_commands
                    command_cols = [
                        "outer_interval",
                        "household_id",
                        "device_id",
                        "device_type",
                        "selection_role",
                        "gateway_id",
                        "up_energy_kwh",
                        "down_energy_kwh",
                        "active_seconds",
                        "activation_count",
                        "max_abs_command_kw",
                    ]
                    st.dataframe(
                        styled_dataframe(command_view[[col for col in command_cols if col in command_view.columns]].head(40).round(3)),
                        use_container_width=True,
                        hide_index=True,
                        height=310,
                    )
            else:
                if st.session_state.get("live_market_session"):
                    with st.expander("Debug: Lower result structure", expanded=False):
                        st.write("patch: settlement_live_trace_debug")
                        st.write(f"live_market_status: {live_market_status(st.session_state.get('live_market_session', {}))}")
                        st.write(f"live_lower_result type: {type(lower_result_for_viz).__name__}")
                        if isinstance(lower_result_for_viz, dict):
                            st.write(f"live_lower_result keys: {list(lower_result_for_viz.keys())}")
                            for key, value in lower_result_for_viz.items():
                                if isinstance(value, pd.DataFrame):
                                    st.write(f"{key}: DataFrame shape {value.shape}")
                                else:
                                    st.write(f"{key}: {type(value).__name__}")
                        st.write(f"lower_tracking_source shape: {getattr(lower_tracking_source, 'shape', None)}")
                        st.write(f"visible_tracking_df shape: {getattr(visible_tracking_df, 'shape', None)}")
                        st.write(f"tracking_df shape: {getattr(tracking_df, 'shape', None)}")
                else:
                    st.info("The upper market decision is available. Start the centralized 4-second MPC from the MPC tab to generate requested-vs-delivered tracking and gateway commands.")

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
                    mpc_config = st.session_state.get("mpc_config", {})
                    whatif_forecaster_plugin = mpc_config.get("forecaster_plugin", DEFAULT_FORECASTER_PLUGIN)
                    whatif_optimizer_plugin = mpc_config.get("optimizer_plugin", DEFAULT_OPTIMIZER_PLUGIN)
                    models = ensure_default_models(alt_df, int(seed), whatif_forecaster_plugin)
                    whatif_optimizer = PLUGIN_REGISTRY.get_optimizer(whatif_optimizer_plugin)
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
                            "fcr_hvac_share_cap": 0.20,
                        },
                        mpc_config["market_mode"],
                        mpc_config["resource_mode"],
                        mpc_config["horizon_hours"],
                        mpc_config["dispatch_hours"],
                        {
                            "degradation": degradation_w,
                            "comfort": comfort_w,
                            "departure": departure_w,
                            "risk_policy": str(mpc_config.get("risk_policy", "investor_balanced")),
                            "risk_quantile": float(mpc_config.get("risk_quantile", 0.80)),
                            "reserve_buffer_pct": float(mpc_config.get("reserve_buffer_pct", 0.08)),
                            "non_delivery": float(mpc_config.get("non_delivery_penalty", 900.0)),
                            "activation_uncertainty": float(mpc_config.get("activation_uncertainty_weight", 100.0)),
                            "asset_fatigue": float(mpc_config.get("asset_fatigue_weight", 75.0)),
                        },
                        preview_4s=preview_4s,
                        optimizer=whatif_optimizer,
                        device_roster=device_roster,
                        inner_controller_mode=mpc_config.get("inner_controller_mode", "mpc"),
                        inner_dt_seconds=int(mpc_config.get("inner_dt_seconds", 4)),
                        inner_mpc_horizon_seconds=int(mpc_config.get("inner_mpc_horizon_seconds", 4)),
                        rotation_strategy=mpc_config.get("rotation_strategy", "usage_aware"),
                        gateway_mode=mpc_config.get("gateway_mode", "simulated_centralized"),
                        execute_lower_mpc=bool(mpc_config.get("execute_lower_mpc", True)),
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
        result = st.session_state.get(
            "mpc_result",
            {"history": pd.DataFrame(), "tracking_4s": pd.DataFrame(), "summary": {}, "compliance": {}, "gateway_commands": pd.DataFrame()},
        )
        tracking_export = result.get("tracking_4s", pd.DataFrame())

        export_left, export_right = st.columns([1, 1])
        export_left.download_button("Download simulation bundle (JSON)", data=json.dumps(summary, indent=2).encode("utf-8"), file_name="coverly_summary.json", mime="application/json")
        export_right.download_button(
            "Download centralized 4-second MPC trace (CSV)",
            data=(tracking_export if not tracking_export.empty else preview_4s).to_csv().encode("utf-8"),
            file_name="coverly_4s_tracking.csv",
            mime="text/csv",
        )

        if result.get("summary"):
            pdf_bytes = make_pdf_summary(result["summary"], result["compliance"], config)
            st.download_button("Export MPC formulation summary as PDF", data=pdf_bytes, file_name="coverly_mpc_summary.pdf", mime="application/pdf")

        st.markdown("### External module artifacts")
        run_root = Path("runs")
        run_dirs = sorted([path for path in run_root.glob("*") if path.is_dir()], reverse=True) if run_root.exists() else []
        if run_dirs:
            selected_run = st.selectbox("Local run artifact", [str(path) for path in run_dirs])
            selected_path = Path(selected_run)
            summary_files = list(selected_path.glob("*_summary.json"))
            if summary_files:
                artifact_summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
                st.json(artifact_summary, expanded=False)
            forecast_file = selected_path / "forecast_comparison.csv"
            history_file = selected_path / "history.csv"
            if forecast_file.exists():
                artifact_forecast = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
                artifact_fig = signal_line_figure(artifact_forecast, ["actual", "prediction"], "External forecast artifact")
                st.plotly_chart(apply_chart_style(artifact_fig, template, height=340), use_container_width=True)
            if history_file.exists():
                artifact_history = pd.read_csv(history_file, index_col=0, parse_dates=True)
                artifact_cols = [col for col in ["fcr_bid_kw", "afrr_up_bid_kw", "afrr_down_bid_kw", "net_revenue_eur"] if col in artifact_history.columns]
                artifact_fig = signal_line_figure(artifact_history, artifact_cols, "External optimization artifact")
                st.plotly_chart(apply_chart_style(artifact_fig, template, height=340), use_container_width=True)
        else:
            st.info("Run `python scripts\\run_forecasting.py ...` or `python scripts\\run_optimization.py ...` to create local artifacts for visualization here.")

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
                    penalty_weights={"degradation": 35.0, "comfort": 480.0, "departure": 1000.0},
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
