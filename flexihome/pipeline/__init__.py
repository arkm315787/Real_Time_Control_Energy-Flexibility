"""Local pipeline helpers for running Residential VPP modules outside Streamlit."""

from .contracts import (
    ArtifactContract,
    DataFrameContract,
    PipelineManifest,
    PipelineRunConfig,
    StageIOContract,
)
from .orchestrator import ContractPipelineOrchestrator, build_default_pipeline, default_pipeline_dag
from .stages import register_pipeline_stages

__all__ = [
    "ArtifactContract",
    "ContractPipelineOrchestrator",
    "DataFrameContract",
    "PipelineManifest",
    "PipelineRunConfig",
    "StageIOContract",
    "build_default_pipeline",
    "default_pipeline_dag",
    "register_pipeline_stages",
]
