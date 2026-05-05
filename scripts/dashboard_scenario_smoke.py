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

from app import default_scenario_inputs, portfolio_bundle_is_current, scenario_portfolio_signature


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
    ]
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise SystemExit(f"Dashboard scenario form/cache guard is incomplete: {missing}")

    print("dashboard scenario apply smoke checks passed")


if __name__ == "__main__":
    main()
