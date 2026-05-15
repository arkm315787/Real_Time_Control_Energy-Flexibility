"""Smoke-check pending vs failed compliance states for live dashboard audits."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.core.engine import _enrich_tracking_for_market_audit, _market_technical_audit


def tracking_frame(periods: int, *, request_kw: float = 1000.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-15 00:00:00", periods=periods, freq="4s")
    seconds = pd.Series(range(periods), index=index, dtype=float) * 4.0
    ratio = (seconds / 300.0).clip(upper=1.0)
    delivered = request_kw * ratio
    return _enrich_tracking_for_market_audit(
        pd.DataFrame(
            {
                "requested_signed_kw": request_kw,
                "delivered_signed_kw": delivered,
                "fcr_local_request_kw": 0.0,
                "afrr_signal_request_kw": request_kw,
                "fcr_delivered_kw": 0.0,
                "afrr_delivered_kw": delivered,
            },
            index=index,
        )
    )


def assert_state(name: str, condition: bool) -> None:
    if not condition:
        raise SystemExit(name)


def main() -> None:
    result_df = pd.DataFrame({"fcr_bid_kw": [0.0]}, index=[pd.Timestamp("2026-01-15 00:00:00")])
    fleet_meta = {"hvac_response_s": 20.0}

    no_trace, _, _ = _market_technical_audit(result_df, pd.DataFrame(), fleet_meta, "aFRR", 4)
    assert_state("No trace should not fail aFRR start capability.", bool(no_trace["afrr_start_within_30s_ok"]))
    assert_state("No trace should mark full activation as pending.", bool(no_trace["afrr_full_activation_pending"]))
    assert_state("No trace should defer energy reporting.", not bool(no_trace["afrr_energy_reporting_evaluated"]))

    partial, _, _ = _market_technical_audit(result_df, tracking_frame(51), fleet_meta, "aFRR", 4)
    assert_state("Partial replay should pass aFRR start.", bool(partial["afrr_start_within_30s_ok"]))
    assert_state("Partial replay should keep 5-minute activation pending.", bool(partial["afrr_full_activation_pending"]))
    assert_state("Partial replay should defer 15-minute ISP reporting.", not bool(partial["afrr_energy_reporting_evaluated"]))
    assert_state("Partial replay should not mark ramp tracking as failed.", bool(partial["accuracy_ok"]))

    complete, _, energy = _market_technical_audit(result_df, tracking_frame(225), fleet_meta, "aFRR", 4)
    assert_state("Complete replay should evaluate full activation.", not bool(complete["afrr_full_activation_pending"]))
    assert_state("Complete replay should pass full activation.", bool(complete["afrr_full_activation_5min_ok"]))
    assert_state("Complete replay should evaluate energy reporting.", bool(complete["afrr_energy_reporting_evaluated"]))
    assert_state("Complete replay should pass energy reporting.", bool(complete["afrr_energy_reporting_ok"]))
    assert_state("Energy audit should mark the ISP complete.", bool(energy["period_complete"].all()))

    print("compliance pending-state smoke checks passed")


if __name__ == "__main__":
    main()
