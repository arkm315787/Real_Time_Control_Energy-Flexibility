"""Smoke-test the TimescaleDB optimization store.

Set FLEXIHOME_TIMESCALE_DSN before running this script. It creates the schema,
writes one tiny optimization job, updates it through the normal lifecycle, and
prints the stored result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.api.store import TimescaleOptimizationStore
from flexihome.api.services import new_optimization_id


def main() -> None:
    dsn = os.getenv("FLEXIHOME_TIMESCALE_DSN", "").strip()
    if not dsn:
        print("FLEXIHOME_TIMESCALE_DSN is not set; skipping TimescaleDB smoke test.")
        return

    store = TimescaleOptimizationStore(
        dsn=dsn,
        schema=os.getenv("FLEXIHOME_TIMESCALE_SCHEMA", "flexihome").strip() or "flexihome",
    )
    optimization_id = new_optimization_id()
    store.create(
        optimization_id,
        {
            "market_mode": "Combined",
            "resource_mode": "Hybrid portfolio",
            "portfolio_config": {"days": 1, "n_homes": 80},
        },
    )
    store.mark_running(optimization_id)
    store.complete(
        optimization_id,
        {
            "summary": {"total_revenue_eur": 12.34},
            "compliance": {"baseline_method_ok": True},
            "row_counts": {"history": 1, "tracking_4s": 1, "first_schedule": 1},
        },
    )
    job = store.get(optimization_id)
    if job is None or job.status != "completed":
        raise SystemExit("TimescaleDB store did not persist the completed job.")

    print("health:", store.health())
    print("stored job:", job.optimization_id, job.status, job.result["summary"])
    print("metrics:", store.metrics())


if __name__ == "__main__":
    main()
