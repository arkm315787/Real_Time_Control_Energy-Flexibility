"""Smoke-test the FastAPI layer without starting a network server."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.api.optimizer import app


SMALL_PORTFOLIO = {
    "days": 1,
    "n_homes": 80,
    "freq_minutes": 15,
    "ev_pen": 0.2,
    "bess_pen": 0.2,
    "pv_pen": 0.3,
    "hvac_pen": 0.5,
}


def main() -> None:
    client = TestClient(app)

    health = client.get("/health")
    health.raise_for_status()
    print("health:", health.json())

    plugins = client.get("/plugins")
    plugins.raise_for_status()
    plugin_payload = plugins.json()
    print("plugins:", plugin_payload)
    if "xgboost_default" not in plugin_payload["forecasters"]:
        raise SystemExit("Expected xgboost_default forecaster plugin")
    if "pulp_default" not in plugin_payload["optimizers"]:
        raise SystemExit("Expected pulp_default optimizer plugin")

    forecast = client.post(
        "/forecast",
        json={
            "forecaster_plugin": "xgboost_default",
            "target": "net_load_baseline_kw",
            "lags": 2,
            "horizon_steps": 1,
            "portfolio_config": SMALL_PORTFOLIO,
        },
    )
    forecast.raise_for_status()
    print("forecast metrics:", forecast.json()["metrics"])

    job = client.post(
        "/optimize",
        json={
            "market_mode": "Combined",
            "resource_mode": "Hybrid portfolio",
            "forecaster_plugin": "xgboost_default",
            "optimizer_plugin": "pulp_default",
            "horizon_hours": 1,
            "dispatch_hours": 1,
            "portfolio_config": SMALL_PORTFOLIO,
        },
    )
    job.raise_for_status()
    job_payload = job.json()
    print("queued:", job_payload["optimization_id"])

    result = client.get(job_payload["results_url"])
    result.raise_for_status()
    result_payload = result.json()
    summary = result_payload["result"]["summary"]
    row_counts = result_payload["result"]["row_counts"]
    print("result status:", result_payload["status"])
    print("total revenue:", summary["total_revenue_eur"])
    print("row counts:", row_counts)

    if result_payload["status"] != "completed":
        raise SystemExit(f"Expected completed optimization, got {result_payload['status']}")
    if row_counts["history"] < 1:
        raise SystemExit("Expected optimization history rows")
    for key in (
        "tracking_4s",
        "household_contributions",
        "appliance_contributions",
        "upper_device_schedule",
        "inner_mpc_trace",
        "gateway_commands",
        "usage_fatigue_summary",
    ):
        if row_counts.get(key, 0) < 1:
            raise SystemExit(f"Expected non-empty {key} rows")
    if summary.get("inner_controller_mode") != "mpc":
        raise SystemExit("Expected centralized lower controller mode to be mpc")


if __name__ == "__main__":
    main()
