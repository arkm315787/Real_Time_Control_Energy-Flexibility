"""Optimization pipeline stage."""

from __future__ import annotations

import pandas as pd

from flexihome.core.base import BasePipelineStage, PipelineContext
from flexihome.core.base.registry import get_global_registry
from flexihome.core.engine import default_hvac_response_seconds, run_mpc_controller
from flexihome.pipeline.contracts import OPTIMIZER_INPUT_COLUMNS, PORTFOLIO_DATA_CONTRACT, ArtifactContract, PipelineRunConfig, StageIOContract
from flexihome.plugins import register_default_plugins


class OptimizationStage(BasePipelineStage):
    """Run the configured MPC optimizer plugin over fitted forecast models."""

    contract = StageIOContract(
        stage_name="optimization",
        input_data=PORTFOLIO_DATA_CONTRACT,
        required_artifacts=(
            ArtifactContract("forecast_models", "Fitted forecast models keyed by MPC target."),
            ArtifactContract("portfolio_summary", "Portfolio summary generated during ingestion."),
            ArtifactContract("preview_4s", "Four-second reserve activation preview."),
            ArtifactContract("device_roster", "Traceable household/appliance roster for centralized dispatch."),
        ),
        output_artifacts=(
            ArtifactContract("optimization_result", "Full MPC result including history and tracking frames."),
            ArtifactContract("fleet_metadata", "Fleet metadata passed into the optimizer."),
        ),
    )

    def __init__(self, config: PipelineRunConfig, name: str | None = None, version: str = "1.0.0") -> None:
        super().__init__(name or "optimization", version=version)
        self.config = config

    def execute(self, context: PipelineContext) -> PipelineContext:
        assert context.data is not None
        registry = register_default_plugins(get_global_registry())
        optimizer = registry.get_optimizer(self.config.optimizer_plugin)
        summary = context.artifacts.get("portfolio_summary", {})
        hvac_response_s = self.config.hvac_response_s or default_hvac_response_seconds(self.config.hvac_mode)
        fleet_metadata = {
            "bess_energy_cap_mwh": float(context.data["bess_energy_cap_mwh"].iloc[0]),
            "setpoint_c": float(self.config.setpoint_c),
            "comfort_band_c": float(self.config.comfort_band_c),
            "n_hvac": int(summary.get("n_hvac", 0)),
            "hvac_mode": self.config.hvac_mode,
            "hvac_response_s": float(hvac_response_s),
        }
        result = run_mpc_controller(
            df=context.data,
            models=context.artifacts["forecast_models"],
            fleet_meta=fleet_metadata,
            market_mode=self.config.market_mode,
            resource_mode=self.config.resource_mode,
            horizon_hours=int(self.config.horizon_hours),
            dispatch_hours=int(self.config.dispatch_hours),
            penalty_weights=self.config.penalty_weights(),
            preview_4s=context.artifacts.get("preview_4s"),
            optimizer=optimizer,
            device_roster=context.artifacts.get("device_roster"),
            inner_controller_mode=self.config.inner_controller_mode,
            inner_dt_seconds=int(self.config.inner_dt_seconds),
            inner_mpc_horizon_seconds=int(self.config.inner_mpc_horizon_seconds),
            rotation_strategy=self.config.rotation_strategy,
            gateway_mode=self.config.gateway_mode,
        )
        result.setdefault("summary", {})["forecaster_plugin"] = self.config.forecaster_plugin
        result.setdefault("summary", {})["optimizer_plugin"] = self.config.optimizer_plugin
        result.setdefault("summary", {})["market_data_source"] = context.artifacts.get("market_data_status", {}).get("mode_used", "synthetic")
        context.with_artifact("optimization_result", result)
        context.with_artifact("fleet_metadata", fleet_metadata)
        context.with_metadata(self.name, {"optimizer_plugin": self.config.optimizer_plugin, "history_rows": len(result.get("history", []))})
        return context

    def validate_inputs(self, context: PipelineContext) -> bool:
        self.contract.validate_input_artifacts(context.artifacts)
        if not isinstance(context.data, pd.DataFrame):
            raise ValueError("OptimizationStage requires context.data as a pandas DataFrame.")
        self.contract.input_data.validate(context.data)
        missing = [column for column in OPTIMIZER_INPUT_COLUMNS if column not in context.data.columns]
        if missing:
            raise ValueError(f"OptimizationStage missing optimizer input columns: {missing}")
        return True

    def get_required_columns(self) -> list[str]:
        return list(OPTIMIZER_INPUT_COLUMNS)

    def get_metadata(self) -> dict:
        payload = super().get_metadata()
        payload["contract"] = self.contract.to_dict()
        return payload
