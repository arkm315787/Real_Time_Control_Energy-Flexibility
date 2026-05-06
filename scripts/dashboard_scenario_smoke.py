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

from app import default_scenario_inputs, live_market_status, live_visible_frame, portfolio_bundle_is_current, scenario_portfolio_signature


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
        'button("Start Market Pressure"',
        'button("Activate Lower MPC"',
        'run_lower_mpc_from_upper_result(',
    ]
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise SystemExit(f"Dashboard scenario form/cache guard is incomplete: {missing}")

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

    print("dashboard scenario apply smoke checks passed")


if __name__ == "__main__":
    main()
