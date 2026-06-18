"""Runtime plugin registry for Residential VPP forecasters, optimizers, and stages."""

from __future__ import annotations

from threading import RLock
from typing import Any, Dict, Type

from .data_pipeline import BasePipelineStage
from .forecaster import BaseForecaster
from .optimizer import BaseOptimizer


class PluginRegistry:
    """Central registry for swappable Residential VPP model and pipeline plugins."""

    def __init__(self) -> None:
        self._forecasters: Dict[str, Type[BaseForecaster]] = {}
        self._optimizers: Dict[str, Type[BaseOptimizer]] = {}
        self._pipeline_stages: Dict[str, Type[BasePipelineStage]] = {}
        self._lock = RLock()

    def register_forecaster(self, key: str, plugin_cls: Type[BaseForecaster], force: bool = False) -> None:
        self._register(self._forecasters, "forecaster", key, plugin_cls, BaseForecaster, force)

    def register_optimizer(self, key: str, plugin_cls: Type[BaseOptimizer], force: bool = False) -> None:
        self._register(self._optimizers, "optimizer", key, plugin_cls, BaseOptimizer, force)

    def register_pipeline_stage(self, key: str, plugin_cls: Type[BasePipelineStage], force: bool = False) -> None:
        self._register(self._pipeline_stages, "pipeline stage", key, plugin_cls, BasePipelineStage, force)

    def get_forecaster(self, key: str, *args: Any, **kwargs: Any) -> BaseForecaster:
        return self._instantiate(self._forecasters, "forecaster", key, *args, **kwargs)

    def get_optimizer(self, key: str, *args: Any, **kwargs: Any) -> BaseOptimizer:
        return self._instantiate(self._optimizers, "optimizer", key, *args, **kwargs)

    def get_pipeline_stage(self, key: str, *args: Any, **kwargs: Any) -> BasePipelineStage:
        return self._instantiate(self._pipeline_stages, "pipeline stage", key, *args, **kwargs)

    def list_forecasters(self) -> Dict[str, str]:
        return self._list(self._forecasters)

    def list_optimizers(self) -> Dict[str, str]:
        return self._list(self._optimizers)

    def list_pipeline_stages(self) -> Dict[str, str]:
        return self._list(self._pipeline_stages)

    def clear(self) -> None:
        with self._lock:
            self._forecasters.clear()
            self._optimizers.clear()
            self._pipeline_stages.clear()

    def _register(
        self,
        bucket: Dict[str, Type],
        plugin_type: str,
        key: str,
        plugin_cls: Type,
        base_cls: Type,
        force: bool,
    ) -> None:
        normalized = self._normalize_key(key)
        if not isinstance(plugin_cls, type) or not issubclass(plugin_cls, base_cls):
            raise TypeError(f"{plugin_type} plugin '{key}' must inherit from {base_cls.__name__}.")
        with self._lock:
            if normalized in bucket and not force:
                raise KeyError(f"{plugin_type} plugin '{normalized}' is already registered.")
            bucket[normalized] = plugin_cls

    def _instantiate(self, bucket: Dict[str, Type], plugin_type: str, key: str, *args: Any, **kwargs: Any) -> Any:
        normalized = self._normalize_key(key)
        with self._lock:
            plugin_cls = bucket.get(normalized)
        if plugin_cls is None:
            available = ", ".join(sorted(bucket)) or "none"
            raise KeyError(f"Unknown {plugin_type} plugin '{normalized}'. Available: {available}")
        kwargs.setdefault("name", normalized)
        return plugin_cls(*args, **kwargs)

    def _list(self, bucket: Dict[str, Type]) -> Dict[str, str]:
        with self._lock:
            return {key: cls.__name__ for key, cls in sorted(bucket.items())}

    @staticmethod
    def _normalize_key(key: str) -> str:
        normalized = str(key or "").strip()
        if not normalized:
            raise ValueError("Plugin key cannot be empty.")
        return normalized


_GLOBAL_REGISTRY = PluginRegistry()


def get_global_registry() -> PluginRegistry:
    return _GLOBAL_REGISTRY
