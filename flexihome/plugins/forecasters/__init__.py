"""Forecasting plugins bundled with Residential VPP."""

from flexihome.core.base.registry import PluginRegistry, get_global_registry

from .lightgbm_forecaster import LightGBMQuantileForecaster
from .xgboost_forecaster import XGBoostForecaster


def register_forecasters(registry: PluginRegistry | None = None, force: bool = False) -> PluginRegistry:
    registry = registry or get_global_registry()
    for key, plugin_cls in (
        ("xgboost_default", XGBoostForecaster),
        ("xgboost_v1", XGBoostForecaster),
        ("lightgbm_quantile", LightGBMQuantileForecaster),
        ("lightgbm_probabilistic", LightGBMQuantileForecaster),
    ):
        try:
            registry.register_forecaster(key, plugin_cls, force=force)
        except KeyError:
            if force:
                raise
    return registry


__all__ = ["LightGBMQuantileForecaster", "XGBoostForecaster", "register_forecasters"]
