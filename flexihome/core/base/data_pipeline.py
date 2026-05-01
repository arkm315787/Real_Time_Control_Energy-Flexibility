"""Composable data pipeline contracts for FlexiHome."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd


@dataclass
class PipelineContext:
    """Mutable context passed between pipeline stages."""

    data: pd.DataFrame | None = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def with_artifact(self, name: str, value: Any) -> "PipelineContext":
        self.artifacts[name] = value
        return self

    def with_metadata(self, name: str, value: Any) -> "PipelineContext":
        self.metadata[name] = value
        return self


class BasePipelineStage(ABC):
    """Abstract interface for composable data pipeline stages."""

    def __init__(self, name: str, version: str = "1.0.0") -> None:
        self.name = name
        self.version = version

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Run this pipeline stage and return the updated context."""

    @abstractmethod
    def validate_inputs(self, context: PipelineContext) -> bool:
        """Validate that the context contains this stage's required inputs."""

    def get_required_columns(self) -> List[str]:
        return []

    def get_required_artifacts(self) -> List[str]:
        return []

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "class": self.__class__.__name__,
            "required_columns": self.get_required_columns(),
            "required_artifacts": self.get_required_artifacts(),
        }


class PipelineOrchestrator:
    """Sequential pipeline runner for modular FlexiHome processing."""

    def __init__(self, name: str = "flexihome_pipeline") -> None:
        self.name = name
        self.stages: List[BasePipelineStage] = []

    def add_stage(self, stage: BasePipelineStage) -> "PipelineOrchestrator":
        self.stages.append(stage)
        return self

    def run(self, initial_data: pd.DataFrame | PipelineContext | None = None) -> PipelineContext:
        if isinstance(initial_data, PipelineContext):
            context = initial_data
        else:
            context = PipelineContext(data=initial_data)
        context.with_metadata("pipeline", self.name)

        for stage in self.stages:
            stage.validate_inputs(context)
            context = stage.execute(context)
            context.metadata.setdefault("stages", []).append(stage.get_metadata())
        context.with_metadata("finished_at", datetime.now(timezone.utc).isoformat())
        return context
