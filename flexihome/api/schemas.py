"""Typed API request and response schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


MarketMode = Literal["FCR-N", "aFRR", "Combined"]
ResourceMode = Literal["Fast only", "Hybrid portfolio"]
HvacMode = Literal["Conventional / thermostat-led", "Inverter / variable-speed"]
RiskPolicy = Literal["investor_balanced", "technical_conservative", "stress_test"]
OptimizationStatus = Literal["queued", "running", "completed", "failed"]


class PortfolioConfig(BaseModel):
    """Synthetic residential portfolio inputs used by the optimizer."""

    start_date: date = date(2026, 1, 15)
    days: int = Field(7, ge=1, le=365)
    n_homes: int = Field(1500, ge=1, le=100000)
    ev_pen: float = Field(0.30, ge=0.0, le=1.0)
    bess_pen: float = Field(0.22, ge=0.0, le=1.0)
    pv_pen: float = Field(0.45, ge=0.0, le=1.0)
    hvac_pen: float = Field(0.82, ge=0.0, le=1.0)
    cloudiness: float = Field(0.55, ge=0.0, le=1.5)
    climate_shift_c: float = Field(0.0, ge=-20.0, le=20.0)
    solar_scale: float = Field(1.0, ge=0.0, le=3.0)
    seed: int = 42
    setpoint_c: float = Field(21.0, ge=10.0, le=30.0)
    comfort_band_c: float = Field(1.0, gt=0.0, le=10.0)
    scarcity: float = Field(1.0, ge=0.0, le=5.0)
    freq_minutes: int = Field(15, ge=1, le=60)
    hvac_mode: HvacMode = "Inverter / variable-speed"
    hvac_response_s: Optional[float] = Field(None, gt=0.0)


class OptimizationRequest(BaseModel):
    """Request schema for MPC optimization."""

    market_mode: MarketMode = "Combined"
    resource_mode: ResourceMode = "Hybrid portfolio"
    forecaster_plugin: str = Field("xgboost_default", min_length=1)
    optimizer_plugin: str = Field("pulp_default", min_length=1)
    inner_controller_mode: Literal["mpc"] = "mpc"
    inner_dt_seconds: int = Field(4, ge=1, le=60)
    inner_mpc_horizon_seconds: int = Field(4, ge=4, le=20)
    rotation_strategy: Literal["usage_aware"] = "usage_aware"
    gateway_mode: Literal["simulated_centralized"] = "simulated_centralized"
    execute_lower_mpc: bool = True
    horizon_hours: float = Field(2.0, ge=0.25, le=72)
    dispatch_hours: float = Field(4.0, ge=0.25, le=168)
    risk_policy: RiskPolicy = "investor_balanced"
    fcr_hvac_share_cap: float = Field(0.20, ge=0.0, le=1.0, description="Comfort-policy cap for HVAC share of FCR-N capacity; not a Fingrid rule.")
    portfolio_config: PortfolioConfig = Field(default_factory=PortfolioConfig)
    degradation_weight: float = Field(35.0, ge=0.0, description="Battery wear cost in EUR/MWh of activated storage throughput.")
    comfort_weight: float = Field(480.0, ge=0.0, description="Indoor comfort violation cost in EUR/degC-hour.")
    departure_weight: float = Field(1000.0, ge=0.0, description="EV missing-energy penalty in EUR/MWh of departure shortfall.")
    risk_quantile: float = Field(0.80, ge=0.50, le=0.95)
    reserve_buffer_pct: float = Field(0.08, ge=0.0, le=0.40)
    non_delivery_penalty: float = Field(900.0, ge=0.0, description="Audit-only non-delivery exposure in EUR/MWh; not a bid-killing objective term.")
    activation_uncertainty_weight: float = Field(100.0, ge=0.0, description="Audit-only activation-volatility exposure in EUR/MWh-equivalent.")
    asset_fatigue_weight: float = Field(75.0, ge=0.0, description="Customer/device fatigue cost in EUR/MWh-equivalent.")


class OptimizationResponse(BaseModel):
    """Summary response for queued and completed optimization jobs."""

    optimization_id: str
    status: OptimizationStatus
    total_revenue_eur: float = 0.0
    delivered_up_mwh: float = 0.0
    delivered_down_mwh: float = 0.0
    requirement_score_pct: float = 0.0
    compliance_checks: Dict[str, bool] = Field(default_factory=dict)
    timestamp: datetime
    results_url: str
    error: Optional[str] = None


class OptimizationResultResponse(BaseModel):
    """Full optimization result payload."""

    optimization_id: str
    status: OptimizationStatus
    request: Dict
    result: Dict = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ForecastRequest(BaseModel):
    """Standalone forecasting request."""

    forecaster_plugin: str = Field("xgboost_default", min_length=1)
    target: str = "net_load_baseline_kw"
    predictors: Optional[List[str]] = None
    lags: int = Field(4, ge=1, le=48)
    horizon_steps: int = Field(1, ge=1, le=96)
    portfolio_config: PortfolioConfig = Field(default_factory=PortfolioConfig)


class ForecastResponse(BaseModel):
    """Standalone forecast response."""

    forecaster_plugin: str
    target: str
    metrics: Dict[str, float]
    horizon_steps: int
    predictions: List[Dict]
    generated_at: datetime

