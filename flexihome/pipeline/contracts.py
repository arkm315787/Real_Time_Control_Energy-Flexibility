"""Explicit data and artifact contracts for Coverly pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from flexihome.core.engine import SUPPORTED_MPC_TARGETS


DEFAULT_MPC_PREDICTORS = [
    "temp_out_c",
    "irradiance_wm2",
    "base_load_kw",
    "pv_available_kw",
    "net_load_baseline_kw",
    "fcrn_capacity_eur_per_mw_h",
    "afrr_up_capacity_eur_per_mw_h",
    "afrr_down_capacity_eur_per_mw_h",
    "afrr_up_act_frac",
    "afrr_down_act_frac",
    "fcr_signed_act",
]


MARKET_DATA_COLUMNS = [
    "spot_price_eur_per_mwh",
    "fcrn_capacity_eur_per_mw_h",
    "afrr_up_capacity_eur_per_mw_h",
    "afrr_down_capacity_eur_per_mw_h",
    "afrr_up_energy_eur_per_mwh",
    "afrr_down_energy_eur_per_mwh",
    "afrr_up_act_frac",
    "afrr_down_act_frac",
    "fcr_signed_act",
]


OPTIMIZER_INPUT_COLUMNS = [
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


@dataclass(frozen=True)
class DataFrameContract:
    """Schema contract for stage dataframe inputs or outputs."""

    name: str
    required_columns: Sequence[str]
    optional_columns: Sequence[str] = field(default_factory=tuple)
    description: str = ""

    def validate(self, frame: pd.DataFrame) -> None:
        if frame is None:
            raise ValueError(f"{self.name} requires a dataframe, got None.")
        missing = [column for column in self.required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{self.name} missing required columns: {missing}")

    def describe(self, frame: pd.DataFrame | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
            "description": self.description,
        }
        if frame is not None:
            payload["rows"] = int(len(frame))
            payload["available_columns"] = list(map(str, frame.columns))
        return payload


@dataclass(frozen=True)
class ArtifactContract:
    """Contract for a named artifact stored in PipelineContext.artifacts."""

    name: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class StageIOContract:
    """Explicit input/output contract for one pipeline stage."""

    stage_name: str
    input_data: DataFrameContract | None = None
    output_data: DataFrameContract | None = None
    required_artifacts: Sequence[ArtifactContract] = field(default_factory=tuple)
    output_artifacts: Sequence[ArtifactContract] = field(default_factory=tuple)

    def validate_input_artifacts(self, artifacts: Dict[str, Any]) -> None:
        missing = [item.name for item in self.required_artifacts if item.required and item.name not in artifacts]
        if missing:
            raise ValueError(f"{self.stage_name} missing required artifacts: {missing}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "input_data": self.input_data.describe() if self.input_data else None,
            "output_data": self.output_data.describe() if self.output_data else None,
            "required_artifacts": [asdict(item) for item in self.required_artifacts],
            "output_artifacts": [asdict(item) for item in self.output_artifacts],
        }


@dataclass
class PipelineRunConfig:
    """Configuration used by the default market-data to optimization pipeline."""

    start_date: str = "2026-01-15"
    days: int = 1
    n_homes: int = 80
    freq_minutes: int = 15
    ev_pen: float = 0.2
    bess_pen: float = 0.2
    pv_pen: float = 0.3
    hvac_pen: float = 0.5
    cloudiness: float = 0.55
    climate_shift_c: float = 0.0
    solar_scale: float = 1.0
    seed: int = 42
    setpoint_c: float = 21.0
    comfort_band_c: float = 1.0
    scarcity: float = 1.0
    hvac_mode: str = "Inverter / variable-speed"
    hvac_response_s: float | None = None
    market_data_mode: str = "synthetic"
    market_lookback_days: int | None = None
    persist_market_data: bool = False
    forecaster_plugin: str = "xgboost_default"
    optimizer_plugin: str = "pulp_default"
    inner_controller_mode: str = "mpc"
    inner_dt_seconds: int = 4
    inner_mpc_horizon_seconds: int = 4
    rotation_strategy: str = "usage_aware"
    gateway_mode: str = "simulated_centralized"
    execute_lower_mpc: bool = True
    market_mode: str = "Combined"
    resource_mode: str = "Hybrid portfolio"
    horizon_hours: float = 1
    dispatch_hours: float = 1
    risk_policy: str = "investor_balanced"
    fcr_hvac_share_cap: float = 0.20
    degradation_weight: float = 35.0
    comfort_weight: float = 480.0
    departure_weight: float = 1000.0
    risk_quantile: float = 0.80
    reserve_buffer_pct: float = 0.08
    non_delivery_penalty: float = 900.0
    activation_uncertainty_weight: float = 100.0
    asset_fatigue_weight: float = 75.0
    output_root: str = "runs"
    forecast_targets: Sequence[str] = field(default_factory=lambda: tuple(SUPPORTED_MPC_TARGETS))
    predictors: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_MPC_PREDICTORS))
    lags: int = 4
    horizon_steps: int = 1

    def portfolio_kwargs(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "days": self.days,
            "n_homes": self.n_homes,
            "ev_pen": self.ev_pen,
            "bess_pen": self.bess_pen,
            "pv_pen": self.pv_pen,
            "hvac_pen": self.hvac_pen,
            "cloudiness": self.cloudiness,
            "climate_shift_c": self.climate_shift_c,
            "solar_scale": self.solar_scale,
            "seed": self.seed,
            "setpoint_c": self.setpoint_c,
            "comfort_band_c": self.comfort_band_c,
            "scarcity": self.scarcity,
            "freq_minutes": self.freq_minutes,
            "hvac_mode": self.hvac_mode,
            "hvac_response_s": self.hvac_response_s,
            "market_data_mode": self.market_data_mode,
            "market_lookback_days": self.market_lookback_days,
            "persist_market_data": self.persist_market_data,
        }

    def penalty_weights(self) -> Dict[str, float]:
        return {
            "degradation": float(self.degradation_weight),
            "comfort": float(self.comfort_weight),
            "departure": float(self.departure_weight),
            "risk_policy": str(self.risk_policy),
            "risk_quantile": float(self.risk_quantile),
            "reserve_buffer_pct": float(self.reserve_buffer_pct),
            "non_delivery": float(self.non_delivery_penalty),
            "activation_uncertainty": float(self.activation_uncertainty_weight),
            "asset_fatigue": float(self.asset_fatigue_weight),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(Path(self.output_root))
        payload["forecast_targets"] = list(self.forecast_targets)
        payload["predictors"] = list(self.predictors)
        return payload


@dataclass
class PipelineManifest:
    """Versioned manifest for one pipeline run."""

    run_id: str
    pipeline_name: str
    config: Dict[str, Any]
    dag: List[Dict[str, Any]]
    contracts: List[Dict[str, Any]]
    artifacts: List[Dict[str, Any]]
    data_versions: Dict[str, str]
    started_at: str
    finished_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PORTFOLIO_DATA_CONTRACT = DataFrameContract(
    name="portfolio_timeseries",
    required_columns=tuple(sorted(set(DEFAULT_MPC_PREDICTORS + OPTIMIZER_INPUT_COLUMNS))),
    optional_columns=tuple(MARKET_DATA_COLUMNS),
    description="Time-indexed portfolio, flexibility, activation, and market-price data.",
)


FORECAST_OUTPUT_CONTRACT = StageIOContract(
    stage_name="forecasting",
    input_data=PORTFOLIO_DATA_CONTRACT,
    required_artifacts=(
        ArtifactContract("feature_columns", "Numeric feature columns selected from the portfolio dataframe."),
    ),
    output_artifacts=(
        ArtifactContract("forecast_models", "Fitted forecaster plugin instances keyed by target."),
        ArtifactContract("forecast_metrics", "Forecast validation metrics keyed by target."),
    ),
)


def validate_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
