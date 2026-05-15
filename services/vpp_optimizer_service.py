"""VPP Optimizer Service: aggregator-side optimization and compliance."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import pandas as pd

from flexihome.core.engine import run_lower_mpc_from_upper_result, run_mpc_controller
from utils.data_sync import merge_market_data

logger = logging.getLogger(__name__)


class VPPOptimizerService:
    """Runs VPP-side optimization independent of market-data ingestion."""

    def __init__(self, fleet_meta: Dict[str, object]) -> None:
        self.fleet_meta = dict(fleet_meta)
        self.upper_mpc_result: Dict[str, object] | None = None
        self.lower_mpc_result: Dict[str, object] | None = None
        self.optimization_frame: pd.DataFrame | None = None

    def run_upper_mpc(
        self,
        portfolio_df: pd.DataFrame,
        operational_market_df: pd.DataFrame | None,
        models: Dict[str, object],
        market_mode: str = "Combined",
        resource_mode: str = "Hybrid portfolio",
        horizon_hours: float = 24.0,
        dispatch_hours: float = 24.0,
        penalty_weights: Dict[str, object] | None = None,
        preview_4s: pd.DataFrame | None = None,
        optimizer: object | None = None,
        device_roster: pd.DataFrame | None = None,
        inner_controller_mode: str = "mpc",
        inner_dt_seconds: int = 4,
        inner_mpc_horizon_seconds: int = 4,
        rotation_strategy: str = "usage_aware",
        gateway_mode: str = "simulated_centralized",
        execute_lower_mpc: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Dict[str, object]:
        """Run the upper MPC using externally supplied market signals."""

        optimization_df = merge_market_data(portfolio_df, operational_market_df if operational_market_df is not None else pd.DataFrame())
        self.optimization_frame = optimization_df
        logger.info("Upper MPC optimizing %s intervals", len(optimization_df))
        self.upper_mpc_result = run_mpc_controller(
            df=optimization_df,
            models=models,
            fleet_meta=self.fleet_meta,
            market_mode=market_mode,
            resource_mode=resource_mode,
            horizon_hours=horizon_hours,
            dispatch_hours=dispatch_hours,
            penalty_weights=penalty_weights or {},
            preview_4s=preview_4s,
            optimizer=optimizer,
            device_roster=device_roster if device_roster is not None else pd.DataFrame(),
            inner_controller_mode=inner_controller_mode,
            inner_dt_seconds=int(inner_dt_seconds),
            inner_mpc_horizon_seconds=int(inner_mpc_horizon_seconds),
            rotation_strategy=rotation_strategy,
            gateway_mode=gateway_mode,
            execute_lower_mpc=bool(execute_lower_mpc),
            progress_callback=progress_callback,
        )
        return self.upper_mpc_result

    def run_lower_mpc(
        self,
        upper_result: Dict[str, object],
        market_feed_4s: pd.DataFrame,
        elapsed_tick: int,
        portfolio_df: pd.DataFrame | None = None,
        market_mode: str = "Combined",
        resource_mode: str = "Hybrid portfolio",
        device_roster: pd.DataFrame | None = None,
        inner_dt_seconds: int = 4,
        inner_mpc_horizon_seconds: int = 4,
    ) -> Dict[str, object]:
        """Run the live lower MPC from an already-cleared upper socket."""

        base_df = portfolio_df if portfolio_df is not None else self.optimization_frame
        if base_df is None or base_df.empty:
            raise ValueError("Lower MPC requires a portfolio frame aligned with the upper result.")
        self.lower_mpc_result = run_lower_mpc_from_upper_result(
            df=base_df,
            upper_result=upper_result,
            fleet_meta=self.fleet_meta,
            market_mode=market_mode,
            resource_mode=resource_mode,
            preview_4s=market_feed_4s,
            device_roster=device_roster if device_roster is not None else pd.DataFrame(),
            inner_controller_mode="mpc",
            inner_dt_seconds=int(inner_dt_seconds),
            inner_mpc_horizon_seconds=int(inner_mpc_horizon_seconds),
            rotation_strategy="usage_aware",
            gateway_mode="live_market_coupled",
            elapsed_seconds=int(elapsed_tick) * int(inner_dt_seconds),
        )
        return self.lower_mpc_result

    def get_compliance_status(self) -> Dict[str, object]:
        if self.lower_mpc_result:
            return dict(self.lower_mpc_result.get("compliance", {}))
        if self.upper_mpc_result:
            return dict(self.upper_mpc_result.get("compliance", {}))
        return {}
