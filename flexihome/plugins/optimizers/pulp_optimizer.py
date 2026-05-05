"""Production PuLP/CBC optimizer plugin."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from flexihome.core.base.optimizer import BaseOptimizer, OptimizationResult


class PuLPOptimizer(BaseOptimizer):
    """PuLP/CBC MPC optimizer with the existing heuristic fallback."""

    def __init__(self, name: str = "pulp_default", version: str = "1.0.0", **metadata) -> None:
        super().__init__(name=name, version=version, solver="PuLP/CBC", fallback="heuristic", **metadata)

    def solve(
        self,
        forecast_horizon: pd.DataFrame,
        current_state: Dict[str, float],
        fleet_metadata: Dict[str, float],
        market_mode: str,
        resource_mode: str,
        penalty_weights: Dict[str, float],
    ) -> OptimizationResult:
        from flexihome.core.engine import solve_mpc_step

        self.validate_inputs(forecast_horizon, current_state, fleet_metadata)
        payload = solve_mpc_step(
            forecast_df=forecast_horizon,
            state=current_state,
            market_mode=market_mode,
            resource_mode=resource_mode,
            penalty_weights=penalty_weights,
            fleet_meta=fleet_metadata,
        )
        return OptimizationResult(
            status=str(payload.get("status", "unknown")),
            controls=dict(payload.get("controls", {})),
            schedule=payload.get("schedule", pd.DataFrame()),
            objective_value=payload.get("objective_value"),
            metadata={"optimizer_plugin": self.name, **self.get_metadata()},
        )

    def get_constraints(self) -> list[str]:
        return [
            "BESS energy and power bounds",
            "EV energy window and departure slack",
            "HVAC comfort band slack",
            "FCR-N and aFRR market enablement",
            "Response-time eligibility for market products",
            "Risk quantile derating and reserve buffer",
            "Resource-mode gating for fast-only portfolios",
            "Capacity, activation, degradation, comfort, non-delivery, activation-risk, and fatigue objective terms",
        ]
