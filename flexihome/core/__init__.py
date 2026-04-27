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

__all__ = [
    "FINGRID_RULES",
    "HVAC_MODES",
    "SUPPORTED_MPC_TARGETS",
    "ForecastSpec",
    "default_hvac_response_seconds",
    "generate_synthetic_portfolio",
    "run_mpc_controller",
    "train_default_mpc_models",
    "train_forecaster",
]

