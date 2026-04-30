"""Smoke test for optional ENTSO-E and Fingrid market-data ingestion."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.core.data import generate_synthetic_portfolio
from flexihome.core.market_data import load_local_env


def main() -> None:
    load_local_env()
    has_entsoe = bool(os.getenv("ENTSOE_API_KEY") or os.getenv("ENTSOE_SECURITY_TOKEN"))
    has_fingrid = bool(os.getenv("FINGRID_API_KEY") or os.getenv("FINGRID_OPENDATA_API_KEY"))
    print(f"ENTSO-E key configured: {has_entsoe}")
    print(f"Fingrid key configured: {has_fingrid}")

    bundle = generate_synthetic_portfolio(
        start_date="2026-01-15",
        days=2,
        n_homes=120,
        ev_pen=0.25,
        bess_pen=0.20,
        pv_pen=0.35,
        hvac_pen=0.60,
        cloudiness=0.55,
        climate_shift_c=0.0,
        solar_scale=1.0,
        seed=7,
        setpoint_c=21.0,
        comfort_band_c=1.0,
        scarcity=1.0,
        freq_minutes=15,
        hvac_mode="Inverter / variable-speed",
    )

    status = bundle["summary"].get("market_data_status", {})
    print(f"market data mode used: {status.get('mode_used')}")
    print(f"ENTSO-E status: {status.get('entsoe')}")
    print(f"Fingrid status: {status.get('fingrid')}")
    print(f"columns loaded: {', '.join(status.get('columns_loaded', [])) or 'none'}")
    print(f"observations loaded: {status.get('observations_loaded', {})}")
    if status.get("errors"):
        print("errors:")
        for error in status["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
