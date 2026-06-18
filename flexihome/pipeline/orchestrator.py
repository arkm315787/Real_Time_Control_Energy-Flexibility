"""Pipeline sequencing for Residential VPP production workflows."""

from __future__ import annotations

from typing import Any, Dict, List

from flexihome.core.base import PipelineContext, PipelineOrchestrator
from flexihome.pipeline.contracts import PipelineRunConfig
from flexihome.pipeline.stages import (
    FeatureEngineeringStage,
    ForecastingStage,
    MarketDataIngestionStage,
    OptimizationStage,
    ResultsSerializationStage,
)


class ContractPipelineOrchestrator(PipelineOrchestrator):
    """Pipeline orchestrator that publishes DAG and contract metadata."""

    def describe_dag(self) -> List[Dict[str, Any]]:
        nodes = [stage.name for stage in self.stages]
        edges = [{"from": nodes[idx], "to": nodes[idx + 1]} for idx in range(len(nodes) - 1)]
        return [{"nodes": nodes, "edges": edges}]

    def describe_contracts(self) -> List[Dict[str, Any]]:
        contracts = []
        for stage in self.stages:
            contract = getattr(stage, "contract", None)
            if contract is not None:
                contracts.append(contract.to_dict())
        return contracts

    def run(self, initial_data=None) -> PipelineContext:
        context = initial_data if isinstance(initial_data, PipelineContext) else PipelineContext(data=initial_data)
        context.with_metadata("dag", self.describe_dag())
        context.with_metadata("contracts", self.describe_contracts())
        return super().run(context)


def build_default_pipeline(config: PipelineRunConfig | None = None) -> ContractPipelineOrchestrator:
    config = config or PipelineRunConfig()
    return (
        ContractPipelineOrchestrator("flexihome_market_forecast_optimization")
        .add_stage(MarketDataIngestionStage(config))
        .add_stage(FeatureEngineeringStage(config))
        .add_stage(ForecastingStage(config))
        .add_stage(OptimizationStage(config))
        .add_stage(ResultsSerializationStage(config, pipeline_name="flexihome_market_forecast_optimization"))
    )


def default_pipeline_dag() -> List[Dict[str, Any]]:
    return build_default_pipeline(PipelineRunConfig()).describe_dag()
