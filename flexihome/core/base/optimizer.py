"""Base optimization plugin contracts for FlexiHome."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd


@dataclass
class OptimizationResult:
    """Standard result returned by optimizer plugins."""

    status: str
    controls: Dict[str, float]
    schedule: pd.DataFrame
    objective_value: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schedule"] = self.schedule
        return payload

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "controls": self.controls,
            "schedule": self.schedule,
            "objective_value": self.objective_value,
            "metadata": dict(self.metadata),
        }


class BaseOptimizer(ABC):
    """Abstract interface for pluggable optimization solvers."""

    def __init__(self, name: str, version: str = "1.0.0", **metadata: Any) -> None:
        self.name = name
        self.version = version
        self.metadata = dict(metadata)
        self.created_at = datetime.now(timezone.utc)

    @abstractmethod
    def solve(
        self,
        forecast_horizon: pd.DataFrame,
        current_state: Dict[str, float],
        fleet_metadata: Dict[str, float],
        market_mode: str,
        resource_mode: str,
        penalty_weights: Dict[str, float],
    ) -> OptimizationResult:
        """Solve one MPC horizon and return the first controls plus full schedule."""

    @abstractmethod
    def get_constraints(self) -> List[str]:
        """Describe the major constraint families used by this optimizer."""

    def validate_inputs(
        self,
        forecast_horizon: pd.DataFrame,
        current_state: Dict[str, float],
        fleet_metadata: Dict[str, float],
    ) -> bool:
        """Validate common optimizer inputs before a solver-specific formulation runs."""

        required_forecast_columns = {
            "dt_h",
            "bess_up_kw",
            "bess_down_kw",
            "ev_up_kw",
            "ev_down_kw",
            "hvac_up_kw",
            "hvac_down_kw",
            "pv_down_kw",
            "fcrn_capacity_eur_per_mw_h",
            "afrr_up_capacity_eur_per_mw_h",
            "afrr_down_capacity_eur_per_mw_h",
        }
        missing_forecast = required_forecast_columns.difference(forecast_horizon.columns)
        if missing_forecast:
            raise ValueError(f"Missing optimizer forecast columns: {sorted(missing_forecast)}")

        required_state = {"bess_soc_mwh", "ev_soc_delta_mwh", "temp_delta_c"}
        missing_state = required_state.difference(current_state)
        if missing_state:
            raise ValueError(f"Missing optimizer state values: {sorted(missing_state)}")

        required_fleet = {"bess_energy_cap_mwh", "setpoint_c", "comfort_band_c", "n_hvac"}
        missing_fleet = required_fleet.difference(fleet_metadata)
        if missing_fleet:
            raise ValueError(f"Missing fleet metadata values: {sorted(missing_fleet)}")
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return plugin metadata useful for audit logs and API diagnostics."""

        return {
            "name": self.name,
            "version": self.version,
            "class": self.__class__.__name__,
            "constraints": self.get_constraints(),
            "created_at": self.created_at.isoformat(),
            **self.metadata,
        }
