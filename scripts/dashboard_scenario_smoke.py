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
        'button("Activate Lower MPC"',
        'button("Stop Simulated Market"',
        "st.fragment(",
        "cached_lower_mpc_from_upper(",
        'run_lower_mpc_from_upper_result(',
        "MarketSimulator(",
        "sync_market_simulator_feed(",
        "upper_socket_ready",
        "upper_socket_metrics(",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise SystemExit(f"Dashboard scenario form/cache guard is incomplete: {missing}")
    if 'st.session_state["mpc_result"] = lower_result' in source:
        raise SystemExit("Dashboard must not overwrite the upper MPC result with the live lower-MPC attachment.")
    if "time.sleep(1)" in source:
        raise SystemExit("Dashboard live refresh must not force a one-second full app rerun.")

    session = {"start_at_wall_s": 110.0, "duration_seconds": 20.0, "dt_seconds": 4}
    if live_market_status(session, now_s=100.0)["status"] != "scheduled":
        raise SystemExit("Live market session should be scheduled before the start wall-clock.")
    if live_market_status(session, now_s=112.0)["status"] != "live":
        raise SystemExit("Live market session should become live at the start wall-clock.")
    if live_market_status(session, now_s=135.0)["status"] != "closed":
        raise SystemExit("Live market session should close after its duration.")
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
