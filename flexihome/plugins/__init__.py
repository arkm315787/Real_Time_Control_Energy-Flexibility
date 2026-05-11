"""Bundled Coverly plugin registration."""

from flexihome.core.base.registry import PluginRegistry, get_global_registry

from .forecasters import register_forecasters
from .optimizers import register_optimizers

DEFAULT_FORECASTER_PLUGIN = "xgboost_default"
DEFAULT_OPTIMIZER_PLUGIN = "pulp_default"


def register_default_plugins(registry: PluginRegistry | None = None, force: bool = False) -> PluginRegistry:
    registry = registry or get_global_registry()
    register_forecasters(registry, force=force)
    register_optimizers(registry, force=force)
    return registry


__all__ = [
    "DEFAULT_FORECASTER_PLUGIN",
    "DEFAULT_OPTIMIZER_PLUGIN",
    "register_default_plugins",
]
