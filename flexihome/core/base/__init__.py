"""Base plugin interfaces for FlexiHome."""

from .data_pipeline import BasePipelineStage, PipelineContext, PipelineOrchestrator
from .forecaster import BaseForecaster, ForecastMetrics
from .optimizer import BaseOptimizer, OptimizationResult
from .registry import PluginRegistry, get_global_registry

__all__ = [
    "BaseForecaster",
    "BaseOptimizer",
    "BasePipelineStage",
    "ForecastMetrics",
    "OptimizationResult",
    "PipelineContext",
    "PipelineOrchestrator",
    "PluginRegistry",
    "get_global_registry",
]
