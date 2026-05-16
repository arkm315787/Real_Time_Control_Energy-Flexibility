"""Smoke-test Coverly plugin registration and default plugin execution."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexihome.core.base.registry import get_global_registry
from flexihome.core.data import generate_synthetic_portfolio
from flexihome.pipeline import register_pipeline_stages
from flexihome.plugins import DEFAULT_FORECASTER_PLUGIN, DEFAULT_OPTIMIZER_PLUGIN, register_default_plugins


FORECAST_PREDICTORS = [
    "temp_out_c",
    "irradiance_wm2",
    "base_load_kw",
    "pv_available_kw",
    "net_load_baseline_kw",
    "fcr_signed_act",
]

OPTIMIZER_COLUMNS = [
    "dt_h",
    "bess_up_kw",
    "bess_down_kw",
    "ev_up_kw",
    "ev_down_kw",
    "hvac_up_kw",
    "hvac_down_kw",
    "hvac_fast_kw",
    "pv_down_kw",
    "ev_soc_ref_mwh",
    "ev_soc_min_mwh",
    "ev_soc_max_mwh",
    "indoor_temp_c_ref",
    "fcr_up_act_frac",
    "fcr_down_act_frac",
    "afrr_up_act_frac",
    "afrr_down_act_frac",
    "fcrn_capacity_eur_per_mw_h",
    "afrr_up_capacity_eur_per_mw_h",
    "afrr_down_capacity_eur_per_mw_h",
    "afrr_up_energy_eur_per_mwh",
    "afrr_down_energy_eur_per_mwh",
]


def main() -> None:
    registry = register_default_plugins(get_global_registry())
    register_pipeline_stages(registry)
    print("forecasters:", registry.list_forecasters())
    print("optimizers:", registry.list_optimizers())
    print("pipeline stages:", registry.list_pipeline_stages())

    bundle = generate_synthetic_portfolio(
        start_date="2026-01-15",
        days=1,
        n_homes=80,
        ev_pen=0.2,
        bess_pen=0.2,
        pv_pen=0.3,
        hvac_pen=0.5,
        cloudiness=0.55,
        climate_shift_c=0.0,
        solar_scale=1.0,
        seed=42,
        setpoint_c=21.0,
        comfort_band_c=1.0,
        scarcity=1.0,
        freq_minutes=15,
        hvac_mode="Inverter / variable-speed",
    )
    df = bundle["data"]

    for forecaster_key in (DEFAULT_FORECASTER_PLUGIN, "lightgbm_quantile"):
        forecaster = registry.get_forecaster(forecaster_key, seed=42)
        comparison = forecaster.fit_from_frame(
            df=df,
            target="net_load_baseline_kw",
            predictors=FORECAST_PREDICTORS,
            lags=2,
            horizon_steps=1,
        )
        if comparison.empty or not forecaster.metrics:
            raise SystemExit(f"{forecaster_key} did not produce validation predictions and metrics.")
        sample_features = None
        for feature in getattr(forecaster, "feature_columns", []):
            if sample_features is None:
                sample_features = {}
            sample_features[feature] = float(df[feature.split("_lag")[0]].iloc[-1]) if feature.split("_lag")[0] in df else 0.0
        if sample_features is None:
            raise SystemExit(f"{forecaster_key} did not expose fitted feature columns.")
        sample_frame = pd.DataFrame([sample_features], index=df.tail(1).index)
        _, uncertainty = forecaster.predict(sample_frame, horizon_steps=1)
        required_quantiles = {"p05", "p20", "p50", "p80", "p95"}
        if uncertainty is None or not required_quantiles.issubset(set(uncertainty.columns)):
            raise SystemExit(f"{forecaster_key} must expose quantile forecast bands.")
        print(f"{forecaster_key} forecast metrics:", forecaster.metrics)

    optimizer = registry.get_optimizer(DEFAULT_OPTIMIZER_PLUGIN)
    forecast_horizon = df[OPTIMIZER_COLUMNS].iloc[1:5].copy()
    result = optimizer.solve(
        forecast_horizon=forecast_horizon,
        current_state={
            "bess_soc_mwh": float(df["bess_soc_ref_mwh"].iloc[0]),
            "ev_soc_delta_mwh": 0.0,
            "temp_delta_c": 0.0,
        },
        fleet_metadata={
            "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
            "setpoint_c": 21.0,
            "comfort_band_c": 1.0,
            "n_hvac": int(bundle["summary"]["n_hvac"]),
            "hvac_mode": "Inverter / variable-speed",
            "hvac_response_s": 20.0,
        },
        market_mode="Combined",
        resource_mode="Hybrid portfolio",
        penalty_weights={"degradation": 35.0, "comfort": 480.0, "departure": 1000.0},
    )
    if result.schedule.empty:
        raise SystemExit("Optimizer plugin returned an empty schedule.")
    print("optimizer status:", result.status)
    print("schedule rows:", len(result.schedule))


if __name__ == "__main__":
    main()
