"""Data models for VPP optimization state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class VPPFleetMeta:
    bess_energy_cap_mwh: float
    setpoint_c: float
    comfort_band_c: float
    n_hvac: int
    hvac_mode: str
    hvac_response_s: float
    fcr_hvac_share_cap: float = 0.20

    def as_engine_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


@dataclass
class VPPOptimizationConfig:
    market_mode: str = "Combined"
    resource_mode: str = "Hybrid portfolio"
    horizon_hours: float = 24.0
    dispatch_hours: float = 24.0
    inner_dt_seconds: int = 4
    inner_mpc_horizon_seconds: int = 4
    rotation_strategy: str = "usage_aware"
    gateway_mode: str = "simulated_centralized"
