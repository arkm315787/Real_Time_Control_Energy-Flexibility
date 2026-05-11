"""Forecasting plugins bundled with Coverly."""

from flexihome.core.base.registry import PluginRegistry, get_global_registry

from .xgboost_forecaster import XGBoostForecaster


def register_forecasters(registry: PluginRegistry | None = None, force: bool = False) -> PluginRegistry:
    registry = registry or get_global_registry()
    for key in ("xgboost_default", "xgboost_v1"):
        try:
            registry.register_forecaster(key, XGBoostForecaster, force=force)
        except KeyError:
            if force:
                raise
    return registry


__all__ = ["XGBoostForecaster", "register_forecasters"]
