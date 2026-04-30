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

import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from flexihome.core.engine import (
    FINGRID_RULES,
    ForecastSpec,
    HVAC_MODES,
    SUPPORTED_MPC_TARGETS,
    bode_points,
    default_hvac_response_seconds,
    fmt_kw,
    fmt_money,
    generate_synthetic_portfolio as generate_synthetic_portfolio_core,
    make_pdf_summary,
    response_model_profiles,
    run_mpc_controller,
    sensitivity_scan,
    serialize_forecast_spec,
    train_default_mpc_models,
    train_forecaster,
)

st.set_page_config(
    page_title="FlexiHome",
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


def data_quality_figure(df: pd.DataFrame, columns: List[str], template: str) -> go.Figure:
    rows = []
    for col in columns:
        if col in df.columns:
            rows.append({"Signal": col, "Completeness %": float(df[col].notna().mean() * 100.0)})
    frame = pd.DataFrame(rows) if rows else pd.DataFrame({"Signal": [], "Completeness %": []})
    fig = px.bar(frame, x="Signal", y="Completeness %", color="Completeness %", range_y=[0, 100], template=template)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Complete rows (%)")
    return fig


































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

    st.sidebar.header("Market Data Source")
    market_source = st.sidebar.radio(
        "Training and market signals",
        ["Synthetic", "Real market APIs"],
        help="Synthetic keeps the dashboard offline. Real market APIs uses ENTSO-E and Fingrid data where available.",
    )
    use_real_market = market_source == "Real market APIs"
    market_lookback_days = st.sidebar.slider("Real market lookback days", 1, 120, 30, 1, disabled=not use_real_market)
    entsoe_key_input = st.sidebar.text_input("ENTSO-E security token", type="password", disabled=not use_real_market)
    fingrid_key_input = st.sidebar.text_input("Fingrid Open Data API key", type="password", disabled=not use_real_market)
    persist_market_data = st.sidebar.toggle("Persist fetched market data to TimescaleDB", value=True, disabled=not use_real_market)
    real_market_ready = use_real_market and bool(entsoe_key_input or fingrid_key_input)
    if use_real_market and not (entsoe_key_input or fingrid_key_input):
        st.sidebar.warning("Enter at least one API key to enable real market ingestion.")

    spinner_text = (
        "Fetching real market data and generating the flexibility portfolio..."
        if real_market_ready
        else "Generating synthetic households, weather, and balancing market conditions..."
    )
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
    }
    try:
        with st.spinner(spinner_text):
            bundle = generate_portfolio_runtime(
                **portfolio_args,
                market_data_mode="real" if real_market_ready else "synthetic",
                entsoe_api_key=entsoe_key_input.strip() or None,
                fingrid_api_key=fingrid_key_input.strip() or None,
                market_lookback_days=int(market_lookback_days) if real_market_ready else None,
                persist_market_data=bool(persist_market_data) if real_market_ready else False,
                use_cache=not real_market_ready,
            )
    except RuntimeError as exc:
        st.error(f"Real market data could not be loaded: {exc}")
        with st.spinner("Falling back to synthetic market data..."):
            bundle = generate_portfolio_runtime(**portfolio_args, market_data_mode="synthetic", use_cache=True)
    df = bundle["data"]
    preview_4s = bundle["preview_4s"]
    summary = bundle["summary"]
    market_data_status = summary.get("market_data_status", {})
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
            "Home",
            "Training Data Explorer",
            "Response Models",
            "Forecasting Lab",
            "MPC Optimizer & Market Participation",
            "Simulation & Impact Visualizations",
            "Advanced / Export",
        ]
    )

    with tabs[0]:
        st.title("FlexiHome: Aggregated Residential Flexibility for Fingrid Balancing Markets")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Data source", str(market_data_status.get("mode_used", "synthetic")).title())
        h2.metric("Model rows", f"{len(df):,}")
        h3.metric("Real observations", f"{int(market_data_status.get('training_observations', 0) or 0):,}")
        h4.metric("Timescale rows", f"{int(market_data_status.get('rows_persisted', 0) or 0):,}")
        if market_data_status.get("errors"):
            st.warning("Some market API signals were unavailable. Open Training Data Explorer for the exact API messages and fallback status.")
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
            c1.metric("Market data", str(market_data_status.get("mode_used", "synthetic")).title())
            c2.metric("Raw market observations", f"{int(market_data_status.get('training_observations', 0) or 0):,}")
            st.markdown(
                """
                <div class="flexi-card">
                    <h4>Fingrid checkpoints embedded in the app</h4>
                    <p class="small-note">
                        The optimizer and compliance view track minimum bid sizes, response speed, expected accuracy,
                        storage endurance, and the presence of an explicit baseline. Capacity and activation prices
                        can be synthetic or loaded from ENTSO-E/Fingrid APIs from the sidebar.
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
            file_name="flexihome_training_data.csv",
            mime="text/csv",
        )
        explain_col.info(
            f"Market mode: {market_data_status.get('mode_used', 'synthetic')}. "
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
        col1, col2, col3, col4 = st.columns([1.2, 1.5, 0.8, 0.8])
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
        lags = col3.slider("Lag depth", 1, 8, 4)
        max_horizon_steps = max(4, min(96, int((24 * 60) / max(freq_minutes, 1))))
        horizon_steps = col4.slider(f"Forecast horizon ({freq_minutes} min steps)", 1, max_horizon_steps, min(4, max_horizon_steps))
        preview_fig = signal_line_figure(df, [target], f"Selected target history: {target}")
        st.plotly_chart(apply_chart_style(preview_fig, template, height=260), use_container_width=True)
        if real_market_ready and target in market_columns:
            st.info(
                f"This forecast target is fed by real market ingestion when available. "
                f"Raw observations for this signal: {market_data_status.get('observations_loaded', {}).get(target, 0):,}."
            )

        if st.button("Train selected XGBoost forecast", type="primary", key=f"train_forecast_{target}"):
            with st.spinner("Training the forecaster..."):
                spec, comparison = train_forecaster(df, target, predictors, lags, horizon_steps, int(seed))
            st.session_state["last_forecast_spec"] = spec
            st.session_state["last_forecast_comparison"] = comparison
            st.session_state["last_forecast_market_status"] = market_data_status
            st.session_state["last_forecast_row_count"] = int(len(df))

        if "last_forecast_spec" in st.session_state:
            spec: ForecastSpec = st.session_state["last_forecast_spec"]
            comparison = st.session_state["last_forecast_comparison"]
            forecast_market_status = st.session_state.get("last_forecast_market_status", {})
            if spec.target != target:
                st.info(f"The displayed trained model is for `{spec.target}`. Click Train selected XGBoost forecast to retrain for `{target}`.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{spec.metrics['mae']:.2f}")
            m2.metric("RMSE", f"{spec.metrics['rmse']:.2f}")
            m3.metric("Rows used", f"{st.session_state.get('last_forecast_row_count', len(df)):,}")
            m4.metric("Market source", str(forecast_market_status.get("mode_used", "synthetic")).title())
            st.caption(
                f"Training frame uses {st.session_state.get('last_forecast_row_count', len(df)):,} rows at {freq_minutes}-minute resolution. "
                f"Real market fetch window: {forecast_market_status.get('query_start_utc') or 'not used'} to "
                f"{forecast_market_status.get('query_end_utc') or 'not used'}; raw market observations loaded: "
                f"{int(forecast_market_status.get('training_observations', 0) or 0):,}."
            )

            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=comparison.index, y=comparison["actual"], name=f"Actual {spec.target}", line={"color": RESOURCE_COLORS["Base"]}))
            pred_fig.add_trace(go.Scatter(x=comparison.index, y=comparison["prediction"], name=f"Predicted {spec.target}", line={"color": RESOURCE_COLORS["Net"], "dash": "dot"}))
            pred_fig.update_layout(yaxis_title=spec.target)
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
        st.caption(
            f"The outer MPC, synthetic portfolio, and forecasting loop run at a {freq_minutes}-minute interval. "
            "Inside each MPC interval, a 4-second tracking controller updates BESS, EV, and HVAC dispatch against the reserve signals."
        )
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
                    preview_4s=preview_4s,
                    progress_callback=lambda current, total, message: mpc_tracker(
                        8 + int(round((current / max(total, 1)) * 92)),
                        100,
                        message,
                    ),
                )
                mpc_tracker(100, 100, "MPC dispatch finished")
        result = st.session_state.get(
            "mpc_result",
            {"history": pd.DataFrame(), "tracking_4s": pd.DataFrame(), "summary": {}, "compliance": {}, "first_schedule": pd.DataFrame()},
        )
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
            mpc_status_cols = st.columns(4)
            solver_mode = result_df["solver_status"].mode().iloc[0] if "solver_status" in result_df and not result_df["solver_status"].empty else "unknown"
            mpc_status_cols[0].metric("Optimizer status", solver_mode)
            mpc_status_cols[1].metric("MPC intervals", f"{len(result_df):,}")
            mpc_status_cols[2].metric("Market input", str(market_data_status.get("mode_used", "synthetic")).title())
            mpc_status_cols[3].metric("Mean net revenue", fmt_money(float(result_df["net_revenue_eur"].mean())))
            st.info(
                "MPC workflow: XGBoost forecasts each horizon, a PuLP/CBC MILP chooses reserve bids and resource dispatch, "
                "then only the first interval is applied before the horizon rolls forward."
            )

            left, right = st.columns([1.45, 1.0])
            with left:
                dispatch_fig = go.Figure()
                dispatch_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["fcr_bid_kw"] / 1000.0, name="FCR-N bid", stackgroup="one"))
                dispatch_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["afrr_up_bid_kw"] / 1000.0, name="aFRR up bid", stackgroup="one"))
                dispatch_fig.add_trace(go.Scatter(x=result_df.index, y=result_df["afrr_down_bid_kw"] / 1000.0, name="aFRR down bid", stackgroup="two"))
                dispatch_fig.update_layout(yaxis_title="MW")
                st.plotly_chart(apply_chart_style(dispatch_fig, template, height=400, title="MPC reserve schedule"), use_container_width=True)
                if {"fcrn_capacity_eur_per_mw_h", "afrr_up_capacity_eur_per_mw_h", "afrr_down_capacity_eur_per_mw_h"}.issubset(result_df.columns):
                    price_overlay = make_subplots(specs=[[{"secondary_y": True}]])
                    price_overlay.add_trace(
                        go.Scatter(x=result_df.index, y=result_df["net_revenue_eur"], name="Net revenue EUR", line={"color": RESOURCE_COLORS["Net"], "width": 3}),
                        secondary_y=False,
                    )
                    price_overlay.add_trace(
                        go.Scatter(x=result_df.index, y=result_df["fcrn_capacity_eur_per_mw_h"], name="FCR-N price", line={"dash": "dot"}),
                        secondary_y=True,
                    )
                    price_overlay.add_trace(
                        go.Scatter(x=result_df.index, y=result_df["afrr_up_capacity_eur_per_mw_h"], name="aFRR up price", line={"dash": "dash"}),
                        secondary_y=True,
                    )
                    price_overlay.update_yaxes(title_text="EUR", secondary_y=False)
                    price_overlay.update_yaxes(title_text="EUR/MW/h", secondary_y=True)
                    st.plotly_chart(apply_chart_style(price_overlay, template, height=330, title="Revenue response to market prices"), use_container_width=True)
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
            tracking_df = result.get("tracking_4s", pd.DataFrame())
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

            if not tracking_df.empty:
                st.markdown("### 4-second inner-loop tracking")
                st.caption("This view shows the fast inner controller tracking the reserve request inside the slower outer MPC interval.")
                tracking_window = tracking_df.iloc[: min(len(tracking_df), max(450, int(3600 / 4)))]
                tracking_fig = go.Figure()
                tracking_fig.add_trace(
                    go.Scatter(
                        x=tracking_window.index,
                        y=(tracking_window["requested_up_kw"] - tracking_window["requested_down_kw"]) / 1000.0,
                        name="Requested reserve (MW)",
                        line={"color": RESOURCE_COLORS["Frequency"], "dash": "dot"},
                    )
                )
                tracking_fig.add_trace(
                    go.Scatter(
                        x=tracking_window.index,
                        y=(tracking_window["delivered_up_kw"] - tracking_window["delivered_down_kw"]) / 1000.0,
                        name="Delivered reserve (MW)",
                        line={"color": RESOURCE_COLORS["Net"], "width": 3},
                    )
                )
                tracking_fig.add_trace(
                    go.Scatter(x=tracking_window.index, y=tracking_window["bess_up_kw"] / 1000.0, name="BESS up", line={"color": RESOURCE_COLORS["BESS"]})
                )
                tracking_fig.add_trace(
                    go.Scatter(x=tracking_window.index, y=-tracking_window["ev_down_kw"] / 1000.0, name="EV down", line={"color": RESOURCE_COLORS["EV"]})
                )
                tracking_fig.add_trace(
                    go.Scatter(x=tracking_window.index, y=tracking_window["hvac_up_kw"] / 1000.0, name="HVAC up", line={"color": RESOURCE_COLORS["HVAC"]})
                )
                tracking_fig.update_layout(yaxis_title="MW")
                st.plotly_chart(apply_chart_style(tracking_fig, template, height=380, title="Inner-loop tracking response (first interval window)"), use_container_width=True)

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
                        preview_4s=preview_4s,
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
        result = st.session_state.get("mpc_result", {"history": pd.DataFrame(), "tracking_4s": pd.DataFrame(), "summary": {}, "compliance": {}})
        tracking_export = result.get("tracking_4s", pd.DataFrame())

        export_left, export_right = st.columns([1, 1])
        export_left.download_button("Download simulation bundle (JSON)", data=json.dumps(summary, indent=2).encode("utf-8"), file_name="flexihome_summary.json", mime="application/json")
        export_right.download_button(
            "Download 4-second tracking results (CSV)",
            data=(tracking_export if not tracking_export.empty else preview_4s).to_csv().encode("utf-8"),
            file_name="flexihome_4s_tracking.csv",
            mime="text/csv",
        )

        if result.get("summary"):
            pdf_bytes = make_pdf_summary(result["summary"], result["compliance"], config)
            st.download_button("Export MPC formulation summary as PDF", data=pdf_bytes, file_name="flexihome_mpc_summary.pdf", mime="application/pdf")

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
