"""Shared FlexiHome core logic."""

from .engine import (
    FINGRID_RULES,
    HVAC_MODES,
    SUPPORTED_MPC_TARGETS,
    ForecastSpec,
    default_hvac_response_seconds,
    generate_synthetic_portfolio,
    run_mpc_controller,
    train_default_mpc_models,
    train_forecaster,
)
from .base import (
    BaseForecaster,
    BaseOptimizer,
    BasePipelineStage,
    ForecastMetrics,
    OptimizationResult,
    PipelineContext,
    PipelineOrchestrator,
    PluginRegistry,
    get_global_registry,
)

__all__ = [
    "BaseForecaster",
    "BaseOptimizer",
    "BasePipelineStage",
    "FINGRID_RULES",
    "HVAC_MODES",
    "SUPPORTED_MPC_TARGETS",
    "ForecastSpec",
    "ForecastMetrics",
    "OptimizationResult",
    "PipelineContext",
    "PipelineOrchestrator",
    "PluginRegistry",
    "default_hvac_response_seconds",
    "generate_synthetic_portfolio",
    "get_global_registry",
    "run_mpc_controller",
    "train_default_mpc_models",
    "train_forecaster",
]

