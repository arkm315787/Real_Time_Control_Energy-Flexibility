"""Typed API request and response schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


MarketMode = Literal["FCR-N", "aFRR", "Combined"]
ResourceMode = Literal["Fast only", "Hybrid portfolio"]
HvacMode = Literal["Conventional / thermostat-led", "Inverter / variable-speed"]
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
    horizon_hours: int = Field(2, ge=1, le=72)
    dispatch_hours: int = Field(4, ge=1, le=168)
    portfolio_config: PortfolioConfig = Field(default_factory=PortfolioConfig)
    degradation_weight: float = Field(18.0, ge=0.0)
    comfort_weight: float = Field(120.0, ge=0.0)
    departure_weight: float = Field(160.0, ge=0.0)


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

    target: str = "net_load_baseline_kw"
    predictors: Optional[List[str]] = None
    lags: int = Field(4, ge=1, le=48)
    horizon_steps: int = Field(1, ge=1, le=96)
    portfolio_config: PortfolioConfig = Field(default_factory=PortfolioConfig)


class ForecastResponse(BaseModel):
    """Standalone forecast response."""

    target: str
    metrics: Dict[str, float]
    horizon_steps: int
    predictions: List[Dict]
    generated_at: datetime

