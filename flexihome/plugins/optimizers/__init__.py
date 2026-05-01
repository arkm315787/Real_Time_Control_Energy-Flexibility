"""Optimization plugins bundled with FlexiHome."""

from flexihome.core.base.registry import PluginRegistry, get_global_registry

from .pulp_optimizer import PuLPOptimizer


def register_optimizers(registry: PluginRegistry | None = None, force: bool = False) -> PluginRegistry:
    registry = registry or get_global_registry()
    for key in ("pulp_default", "pulp_v1"):
        try:
            registry.register_optimizer(key, PuLPOptimizer, force=force)
        except KeyError:
            if force:
                raise
    return registry


__all__ = ["PuLPOptimizer", "register_optimizers"]
