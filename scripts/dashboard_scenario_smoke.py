"""Smoke-test dashboard scenario apply semantics.

The dashboard should keep using the last applied portfolio bundle across normal
Streamlit reruns. Editing controls inside the form must not change the applied
scenario or trigger portfolio regeneration until the Apply Scenario button
updates session state.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from streamlit.elements.plotly_chart import PlotlyMixin

from app import (
    build_live_market_feed,
    default_scenario_inputs,
    live_market_status,
    live_visible_frame,
    portfolio_bundle_is_current,
    scenario_portfolio_signature,
    upper_socket_metrics,
)
from flexihome.core.market_simulator import MarketSimulator, MarketSimulatorConfig


def main() -> None:
    scenario = default_scenario_inputs()
    signature = scenario_portfolio_signature(scenario)
    session_state = {}

    if portfolio_bundle_is_current(session_state, signature):
        raise SystemExit("Empty session must not look like it has a current portfolio bundle.")

    session_state["portfolio_bundle"] = {"data": "existing-bundle"}
    session_state["portfolio_signature"] = signature
    if not portfolio_bundle_is_current(session_state, signature):
        raise SystemExit("Applied scenario should reuse the existing portfolio bundle.")

    pending_widget_values = dict(scenario)
    pending_widget_values["n_homes"] = int(pending_widget_values["n_homes"]) + 100
    if not portfolio_bundle_is_current(session_state, scenario_portfolio_signature(scenario)):
        raise SystemExit("Unsubmitted form edits must not invalidate the applied scenario bundle.")

    applied_scenario = dict(pending_widget_values)
    applied_signature = scenario_portfolio_signature(applied_scenario)
    if applied_signature == signature:
        raise SystemExit("Changing an applied scenario field must change the portfolio signature.")
    if portfolio_bundle_is_current(session_state, applied_signature):
        raise SystemExit("A newly applied scenario must require portfolio regeneration.")

    source = Path("app.py").read_text(encoding="utf-8")
    required_fragments = [
        'with st.sidebar.form("scenario_form")',
        'st.form_submit_button("Apply Scenario"',
        'st.session_state["portfolio_bundle"]',
        'st.session_state["portfolio_signature"]',
        'button("Run Upper MPC Layer"',
        'button("Start Simulated Market"',
        '"Activate Lower MPC"',
        'button("Stop Simulated Market"',
        "st.fragment(",
        "cached_lower_mpc_from_upper(",
        "render_lower_live_trace(",
        'key_prefix="fragment_lower_live"',
        'st.session_state["lower_mpc_armed"] = True',
        "Compliance source:",
        "Capability pass; replay pending",
        "Pending complete 15-min ISP",
        'run_lower_mpc_from_upper_result(',
        "MarketSimulator(",
        "sync_market_simulator_feed(",
        "upper_socket_ready",
        "upper_socket_metrics(",
        "MarketDataService(",
        "VPPOptimizerService(",
        "operational_market_df",
        "training_market_df",
        "merge_market_data(",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise SystemExit(f"Dashboard scenario form/cache guard is incomplete: {missing}")
    if 'st.session_state["mpc_result"] = lower_result' in source:
        raise SystemExit("Dashboard must not overwrite the upper MPC result with the live lower-MPC attachment.")
    if "time.sleep(1)" in source:
        raise SystemExit("Dashboard live refresh must not force a one-second full app rerun.")
    if "time.sleep(4)" in source or "live_refresh_requested" in source:
        raise SystemExit("Dashboard live refresh must stay scoped to Streamlit fragments, not a timed full-page rerun.")
    for fragment in [
        '[data-stale="true"]',
        "chart_theme_tokens(",
        "plot_bgcolor=tokens",
        "legend_y = -0.30",
        "Reserve tracking",
        "Resource dispatch",
        "Control error and latency",
        'barmode="relative"',
        'st.session_state["dashboard_theme_mode"]',
        "plot_sankey(summary, template)",
        '"stopped_at_wall_s"',
        '"stopped_reason"',
        "PlotlyMixin.plotly_chart",
        "plotly_download_config(",
        '"toImageButtonOptions"',
        '"displayModeBar"',
        'container.popover("Export / view figure")',
        'fig.to_html(',
        'fig.to_json(',
        'download_button(',
        '"Viewable HTML"',
        '"Plotly figure"',
    ]:
        if fragment not in source:
            raise SystemExit(f"Dashboard chart polish regression guard is missing: {fragment}")
    if not getattr(PlotlyMixin, "_coverly_download_exporter_installed", False):
        raise SystemExit("Streamlit Plotly renderer must be globally wrapped with figure export controls.")
    if "PlotlyMixin.plotly_chart = downloadable_plotly_chart" not in source:
        raise SystemExit("Every Streamlit Plotly chart must pass through the global downloadable renderer.")
    if source.count(".plotly_chart(") + source.count("st.plotly_chart(") < 35:
        raise SystemExit("Dashboard plot coverage guard detected fewer Plotly renders than expected.")
    blocked_chart_apis = ["st.line_chart(", "st.bar_chart(", "st.area_chart(", "st.altair_chart(", "st.vega_lite_chart(", "st.pyplot("]
    blocked = [api for api in blocked_chart_apis if api in source]
    if blocked:
        raise SystemExit(f"Dashboard plots must use the export-wrapped Plotly renderer, found: {blocked}")
    if 'st.session_state.pop("live_lower_result", None)\n            st.session_state["mpc_phase"] = "upper" if upper_market_ready else "idle"' in source:
        raise SystemExit("Stop Simulated Market must preserve the latest live lower-MPC result for post-stop review.")
    if '[data-testid="stDataFrame"] canvas' in source or '[data-testid="stDataFrame"],\n            [data-testid="stTable"]' in source:
        raise SystemExit("Interactive st.dataframe canvases must not be forced dark; that hides cell text in Streamlit.")
    if "Power state and tracking diagnostics" in source or "Accuracy envelope and control latency" in source:
        raise SystemExit("Lower-MPC default chart should stay compact; detailed diagnostics must not be restored as default stacked line panels.")
    if "Lower MPC has not produced a tracking trace yet" in source:
        raise SystemExit("Dashboard must not leave a stale static lower-MPC pending message while the live fragment owns replay rendering.")
    if 'if st.session_state.get("live_market_session")\n                else result_frame(lower_view_result, "tracking_4s")' not in source:
        raise SystemExit("Static lower-MPC trace should be suppressed while a live market session is rendered by the fragment.")
    if "visible_tracking_df = live_visible_frame(lower_tracking_source" not in source:
        raise SystemExit("Settlement tab should read the visible 4-second lower trace separately from the stored source trace.")
    if "tracking_df = visible_tracking_df if not visible_tracking_df.empty else lower_tracking_source" not in source:
        raise SystemExit("Settlement tab must not show the start-lower-MPC notice when live lower ticks already exist.")
    if 'with st.expander("Debug: Lower result structure"' not in source:
        raise SystemExit("Settlement tab should expose live_lower_result structure while a live session has no visible trace.")
    if "settlement_live_lower_fragment()" not in source or "patch: settlement_live_fragment" not in source:
        raise SystemExit("Settlement tab should use an auto-refreshing fragment for live lower-MPC replay state.")
    if "real_time_preview_source" not in source or "Replay feed" not in source:
        raise SystemExit("Dashboard must surface whether the 4-second market replay uses synthetic or real activation data.")

    session = {"start_at_wall_s": 110.0, "duration_seconds": 20.0, "dt_seconds": 4}
    if live_market_status(session, now_s=100.0)["status"] != "scheduled":
        raise SystemExit("Live market session should be scheduled before the start wall-clock.")
    if live_market_status(session, now_s=112.0)["status"] != "live":
        raise SystemExit("Live market session should become live at the start wall-clock.")
    if live_market_status(session, now_s=135.0)["status"] != "closed":
        raise SystemExit("Live market session should close after its duration.")
    stopped_session = {**session, "stopped_at_wall_s": 118.0}
    stopped_status = live_market_status(stopped_session, now_s=112.0)
    if stopped_status["status"] != "closed" or stopped_status["elapsed_seconds"] != 8.0:
        raise SystemExit("Operator-stopped live market sessions should remain visible as closed replay snapshots.")
    frame = pd.DataFrame({"signal": range(10)}, index=pd.date_range("2026-01-01", periods=10, freq="4s"))
    if len(live_visible_frame(frame, session, now_s=119.0)) != 3:
        raise SystemExit("Live visible frame should reveal rows according to elapsed market seconds.")

    preview_index = pd.date_range("2026-01-15 00:00:00", periods=260, freq="4s")
    preview = pd.DataFrame(
        {
            "frequency_hz": [50.0 - 0.001 * i for i in range(len(preview_index))],
            "fcr_signal_norm": [0.001 * i for i in range(len(preview_index))],
            "afrr_signal_norm": [-0.001 * i for i in range(len(preview_index))],
        },
        index=preview_index,
    )
    socket_start = pd.Timestamp("2026-01-15 00:15:00")
    upper_result = {"history": pd.DataFrame({"fcr_bid_kw": [100.0]}, index=[socket_start])}
    aligned_feed = build_live_market_feed(preview, upper_result, duration_seconds=16, dt_seconds=4)
    if aligned_feed.index[0] != socket_start:
        raise SystemExit("Live market feed should start at the first accepted upper-MPC interval timestamp.")
    expected_first = preview.loc[socket_start]
    if float(aligned_feed.iloc[0]["fcr_signal_norm"]) != float(expected_first["fcr_signal_norm"]):
        raise SystemExit("Live market feed must use the preview signal aligned to the upper-MPC socket, not preview tick zero.")
    simulator = MarketSimulator(aligned_feed, MarketSimulatorConfig(dt_seconds=4), start_wall_time_s=200.0)
    first_signal = simulator.get_current_signal(200.0)
    if first_signal is None:
        raise SystemExit("Market simulator should emit the first tick when the market starts.")
    if abs(float(first_signal["frequency_hz"]) - float(expected_first["frequency_hz"])) > 1e-12:
        raise SystemExit("Market simulator must preserve the aligned preview frequency instead of forcing 50 Hz.")

    empty_socket = {
        "history": pd.DataFrame(
            {
                "fcr_bid_kw": [0.0],
                "afrr_up_bid_kw": [0.0],
                "afrr_down_bid_kw": [0.0],
                "market_gate_status": ["Wait"],
                "reserve_buffer_kw": [0.0],
            }
        ),
        "summary": {"market_gate_participation_intervals": 0},
    }
    empty_metrics = upper_socket_metrics(empty_socket)
    if empty_metrics["ready"] or empty_metrics["accepted_slots"] != 0 or empty_metrics["max_committed_kw"] != 0.0:
        raise SystemExit("An upper MPC result with zero accepted reserve must not enable the live market.")

    cleared_socket = {
        "history": pd.DataFrame(
            {
                "fcr_bid_kw": [120.0, 140.0],
                "afrr_up_bid_kw": [0.0, 0.0],
                "afrr_down_bid_kw": [0.0, 0.0],
                "market_gate_status": ["Participate", "Wait"],
                "reserve_buffer_kw": [24.0, 28.0],
            }
        ),
        "summary": {"market_gate_participation_intervals": 1},
    }
    cleared_metrics = upper_socket_metrics(cleared_socket)
    if not cleared_metrics["ready"] or cleared_metrics["accepted_slots"] != 1 or cleared_metrics["max_committed_kw"] <= 0.0:
        raise SystemExit("A nonzero accepted upper socket should enable the live market and lower MPC.")

    print("dashboard scenario apply smoke checks passed")


if __name__ == "__main__":
    main()
