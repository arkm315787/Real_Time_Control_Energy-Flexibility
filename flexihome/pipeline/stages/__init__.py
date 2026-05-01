"""Pipeline stage implementations."""

from flexihome.core.base.registry import PluginRegistry, get_global_registry

from .feature_engineering import FeatureEngineeringStage
from .forecasting_stage import ForecastingStage
from .market_data_ingestion import MarketDataIngestionStage
from .optimization_stage import OptimizationStage
from .results_serialization import ResultsSerializationStage


def register_pipeline_stages(registry: PluginRegistry | None = None, force: bool = False) -> PluginRegistry:
    registry = registry or get_global_registry()
    for key, stage_cls in {
        "market_data_ingestion": MarketDataIngestionStage,
        "feature_engineering": FeatureEngineeringStage,
        "forecasting": ForecastingStage,
        "optimization": OptimizationStage,
        "results_serialization": ResultsSerializationStage,
    }.items():
        try:
            registry.register_pipeline_stage(key, stage_cls, force=force)
        except KeyError:
            if force:
                raise
    return registry


__all__ = [
    "FeatureEngineeringStage",
    "ForecastingStage",
    "MarketDataIngestionStage",
    "OptimizationStage",
    "ResultsSerializationStage",
    "register_pipeline_stages",
]
