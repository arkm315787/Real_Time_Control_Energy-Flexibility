"""Forecasting pipeline stage."""

from __future__ import annotations

import pandas as pd

from flexihome.core.base import BasePipelineStage, PipelineContext
from flexihome.core.base.registry import get_global_registry
from flexihome.pipeline.contracts import FORECAST_OUTPUT_CONTRACT, PipelineRunConfig, validate_columns
from flexihome.plugins import register_default_plugins


class ForecastingStage(BasePipelineStage):
    """Train configured forecaster plugins for the requested MPC targets."""

    contract = FORECAST_OUTPUT_CONTRACT

    def __init__(self, config: PipelineRunConfig, name: str | None = None, version: str = "1.0.0") -> None:
        super().__init__(name or "forecasting", version=version)
        self.config = config

    def execute(self, context: PipelineContext) -> PipelineContext:
        assert context.data is not None
        registry = register_default_plugins(get_global_registry())
        models = {}
        metrics = {}
        comparisons = {}
        predictors = context.artifacts.get("feature_columns", list(self.config.predictors))
        targets = list(self.config.forecast_targets)
        validate_columns(context.data, targets, "ForecastingStage targets")
        for target in targets:
            forecaster = registry.get_forecaster(
                self.config.forecaster_plugin,
                name=f"{self.config.forecaster_plugin}_{target}",
                seed=int(self.config.seed),
            )
            comparison = forecaster.fit_from_frame(
                df=context.data,
                target=target,
                predictors=predictors,
                lags=int(self.config.lags),
                horizon_steps=int(self.config.horizon_steps),
            )
            models[target] = forecaster
            metrics[target] = dict(forecaster.metrics)
            comparisons[target] = comparison.tail(96)
        context.with_artifact("forecast_models", models)
        context.with_artifact("forecast_metrics", metrics)
        context.with_artifact("forecast_comparisons", comparisons)
        context.with_metadata(self.name, {"forecaster_plugin": self.config.forecaster_plugin, "targets": targets})
        return context

    def validate_inputs(self, context: PipelineContext) -> bool:
        self.contract.validate_input_artifacts(context.artifacts)
        if not isinstance(context.data, pd.DataFrame):
            raise ValueError("ForecastingStage requires context.data as a pandas DataFrame.")
        self.contract.input_data.validate(context.data)
        return True

    def get_required_columns(self) -> list[str]:
        return list(self.contract.input_data.required_columns)

    def get_metadata(self) -> dict:
        payload = super().get_metadata()
        payload["contract"] = self.contract.to_dict()
        return payload
