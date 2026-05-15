"""Smoke test for optional ENTSO-E and Fingrid market-data ingestion."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.core.data import generate_synthetic_portfolio
import flexihome.core.market_data as market_data_module
from flexihome.core.market_data import load_local_env


def assert_real_activation_updates_preview() -> None:
    index = pd.date_range("2026-01-15", periods=4, freq="15min")
    preview_index = pd.date_range(index[0], periods=240, freq="4s")
    prices = pd.DataFrame(
        {
            "fcrn_capacity_eur_per_mw_h": 10.0,
            "afrr_up_capacity_eur_per_mw_h": 8.0,
            "afrr_down_capacity_eur_per_mw_h": 6.0,
            "afrr_up_energy_eur_per_mwh": 90.0,
            "afrr_down_energy_eur_per_mwh": 50.0,
        },
        index=index,
    )
    activation = pd.DataFrame(
        {
            "fcr_signed_act": 0.0,
            "fcr_up_act_frac": 0.0,
            "fcr_down_act_frac": 0.0,
            "afrr_signal_norm": 0.0,
            "afrr_up_act_frac": 0.0,
            "afrr_down_act_frac": 0.0,
            "frequency_hz": 50.0,
        },
        index=index,
    )
    preview = pd.DataFrame(
        {
            "frequency_hz": 50.0,
            "fcr_signal_norm": 0.0,
            "afrr_signal_norm": 0.0,
        },
        index=preview_index,
    )

    original_fetch = market_data_module.fetch_fingrid_dataset
    original_datasets = market_data_module.configured_fingrid_datasets

    def fake_datasets():
        return {
            "afrr_up_act_frac": ("FLEXIHOME_FINGRID_AFRR_UP_ACTIVATION_DATASET_ID", 349),
            "afrr_down_act_frac": ("FLEXIHOME_FINGRID_AFRR_DOWN_ACTIVATION_DATASET_ID", 350),
        }

    def fake_fetch(dataset_id, start, end, api_key, timeout_seconds=12.0):
        if dataset_id == 349:
            return pd.Series([0.1, 0.4, 0.8, 0.2], index=index, name="value")
        if dataset_id == 350:
            return pd.Series([0.0, 0.0, 0.0, 0.0], index=index, name="value")
        return pd.Series(dtype=float, name="value")

    try:
        market_data_module.configured_fingrid_datasets = fake_datasets
        market_data_module.fetch_fingrid_dataset = fake_fetch
        _, _, preview_out, status = market_data_module.overlay_real_market_data(
            index=index,
            prices=prices,
            activation=activation,
            preview_4s=preview,
            mode="auto",
            fingrid_api_key="offline-smoke-key",
            persist_to_timescale=False,
        )
    finally:
        market_data_module.fetch_fingrid_dataset = original_fetch
        market_data_module.configured_fingrid_datasets = original_datasets

    if status.get("real_time_preview_source") != "real_activation":
        raise SystemExit("Real Fingrid activation must mark the 4-second replay feed as real_activation.")
    if "afrr_signal_norm" not in status.get("real_time_preview_columns", []):
        raise SystemExit("Real Fingrid activation must mark afrr_signal_norm as a replay column.")
    if float(preview_out["afrr_signal_norm"].abs().max()) <= 0.0:
        raise SystemExit("Real Fingrid activation must change the 4-second replay aFRR signal.")


def main() -> None:
    load_local_env()
    assert_real_activation_updates_preview()
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
    print(f"realtime replay source: {status.get('real_time_preview_source')}")
    print(f"realtime replay columns: {', '.join(status.get('real_time_preview_columns', [])) or 'none'}")
    print(f"observations loaded: {status.get('observations_loaded', {})}")
    print(f"training observations: {status.get('training_observations', 0)}")
    print(f"query window: {status.get('query_start_utc')} to {status.get('query_end_utc')}")
    print(f"TimescaleDB status: {status.get('timescaledb')}; rows persisted: {status.get('rows_persisted', 0)}")
    if status.get("errors"):
        print("errors:")
        for error in status["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
