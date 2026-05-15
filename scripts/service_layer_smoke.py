"""Smoke checks for the market/VPP service boundary.

The test avoids live API calls by overriding the market fetch hook. It verifies
that TSO-style market data can be cached, split into training/operational
frames, and aligned onto an independent VPP simulation horizon.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.cache_manager import CacheManager
from services.market_service import MarketDataService
from services.vpp_optimizer_service import VPPOptimizerService
from utils.data_sync import merge_market_data


class FakeMarketDataService(MarketDataService):
    def _fetch_from_apis(self, start_date: date, end_date: date) -> tuple[pd.DataFrame, list[str]]:
        index = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(hours=23), freq="h")
        if index.empty:
            return pd.DataFrame(), []
        frame = pd.DataFrame(
            {
                "spot_price_eur_per_mwh": [50.0 + idx % 7 for idx in range(len(index))],
                "fcrn_capacity_eur_per_mw_h": [900.0 + idx % 5 for idx in range(len(index))],
                "afrr_up_capacity_eur_per_mw_h": [1200.0 + idx % 11 for idx in range(len(index))],
                "afrr_down_capacity_eur_per_mw_h": [1100.0 + idx % 13 for idx in range(len(index))],
                "afrr_up_act_frac": [0.2 if idx % 2 else 0.0 for idx in range(len(index))],
                "afrr_down_act_frac": [0.1 if idx % 3 else 0.0 for idx in range(len(index))],
            },
            index=index,
        )
        return frame, []


def main() -> None:
    service = FakeMarketDataService(
        entsoe_key="demo-entsoe",
        fingrid_key="demo-fingrid",
        cache=CacheManager(ttl_seconds=60, cache_dir=None),
    )
    combined, metadata = service.fetch_market_data(lookback_days=2, include_forecast_day=True)
    if combined.empty or metadata["total_rows"] != len(combined):
        raise SystemExit("Market service did not return a consistent market bundle.")

    training, operational = service.split_training_and_operational(combined)
    if training.empty or operational.empty:
        raise SystemExit("Market service split should produce training and operational frames.")

    target = pd.DataFrame(
        {"base_load_kw": [1000.0] * 8},
        index=pd.date_range("2026-01-15", periods=8, freq="15min"),
    )
    merged = merge_market_data(target, operational)
    if "spot_price_eur_per_mwh" not in merged or merged["spot_price_eur_per_mwh"].isna().all():
        raise SystemExit("Operational market data should align onto the VPP optimization horizon.")
    if "afrr_signal_norm" not in merged or float(merged["afrr_signal_norm"].abs().max()) <= 0.0:
        raise SystemExit("Operational activation data should drive the 4-second aFRR replay signal.")

    optimizer_service = VPPOptimizerService({"n_hvac": 10, "bess_energy_cap_mwh": 1.0})
    if optimizer_service.get_compliance_status() != {}:
        raise SystemExit("Fresh VPP optimizer service should not expose stale compliance state.")

    print("service layer smoke checks passed")


if __name__ == "__main__":
    main()
