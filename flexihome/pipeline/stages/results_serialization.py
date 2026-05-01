"""Result serialization and artifact manifest pipeline stage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from flexihome.api.serialization import serialize_optimization_result
from flexihome.core.base import BasePipelineStage, PipelineContext
from flexihome.pipeline.artifacts import artifact_record, dataframe_fingerprint, new_run_dir, write_frame, write_json
from flexihome.pipeline.contracts import ArtifactContract, PipelineManifest, PipelineRunConfig, StageIOContract


class ResultsSerializationStage(BasePipelineStage):
    """Write pipeline outputs and a versioned manifest to disk."""

    contract = StageIOContract(
        stage_name="results_serialization",
        required_artifacts=(
            ArtifactContract("optimization_result", "Full MPC result including history and tracking frames."),
            ArtifactContract("forecast_metrics", "Forecast validation metrics keyed by target."),
            ArtifactContract("market_data_status", "Market-data source and ingestion diagnostics."),
        ),
        output_artifacts=(
            ArtifactContract("run_dir", "Directory containing all persisted artifacts."),
            ArtifactContract("manifest", "Versioned artifact manifest with hashes and schema."),
        ),
    )

    def __init__(
        self,
        config: PipelineRunConfig,
        pipeline_name: str = "flexihome_default_pipeline",
        name: str | None = None,
        version: str = "1.0.0",
    ) -> None:
        super().__init__(name or "results_serialization", version=version)
        self.config = config
        self.pipeline_name = pipeline_name

    def execute(self, context: PipelineContext) -> PipelineContext:
        run_dir = new_run_dir(self.config.output_root, "pipeline")
        result = context.artifacts["optimization_result"]
        artifact_records = []

        if context.data is not None:
            training_path = run_dir / "training_data.csv"
            write_frame(training_path, context.data)
            artifact_records.append(artifact_record("training_data", training_path, "dataframe", context.data))

        forecast_metrics_path = run_dir / "forecast_metrics.json"
        write_json(forecast_metrics_path, context.artifacts.get("forecast_metrics", {}))
        forecast_metrics_record = artifact_record("forecast_metrics", forecast_metrics_path, "json")
        artifact_records.append(forecast_metrics_record)

        for target, comparison in context.artifacts.get("forecast_comparisons", {}).items():
            if isinstance(comparison, pd.DataFrame) and not comparison.empty:
                comparison_path = run_dir / f"forecast_comparison_{target}.csv"
                write_frame(comparison_path, comparison)
                artifact_records.append(artifact_record(f"forecast_comparison_{target}", comparison_path, "dataframe", comparison))

        serialized = serialize_optimization_result(result)
        summary_path = run_dir / "optimization_summary.json"
        write_json(
            summary_path,
            {
                "summary": serialized.get("summary", {}),
                "compliance": serialized.get("compliance", {}),
                "row_counts": serialized.get("row_counts", {}),
                "market_data_status": context.artifacts.get("market_data_status", {}),
            },
        )
        artifact_records.append(artifact_record("optimization_summary", summary_path, "json"))

        for name in ("history", "tracking_4s", "first_schedule"):
            frame = result.get(name)
            if isinstance(frame, pd.DataFrame):
                path = run_dir / f"{name}.csv"
                write_frame(path, frame)
                artifact_records.append(artifact_record(name, path, "dataframe", frame))

        data_versions = {
            "portfolio_timeseries": dataframe_fingerprint(context.data) if context.data is not None else "",
            "forecast_metrics": forecast_metrics_record["sha256"],
        }
        if isinstance(result.get("history"), pd.DataFrame):
            data_versions["optimization_history"] = dataframe_fingerprint(result["history"])

        manifest = PipelineManifest(
            run_id=run_dir.name,
            pipeline_name=self.pipeline_name,
            config=self.config.to_dict(),
            dag=context.metadata.get("dag", []),
            contracts=context.metadata.get("contracts", []),
            artifacts=artifact_records,
            data_versions=data_versions,
            started_at=str(context.started_at.isoformat()),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest_path = run_dir / "manifest.json"
        write_json(manifest_path, manifest.to_dict())
        artifact_records.append(artifact_record("manifest", manifest_path, "json"))
        context.with_artifact("run_dir", Path(run_dir))
        context.with_artifact("manifest", manifest.to_dict())
        context.with_artifact("artifact_records", artifact_records)
        context.with_metadata(self.name, {"run_dir": str(run_dir), "artifact_count": len(artifact_records)})
        return context

    def validate_inputs(self, context: PipelineContext) -> bool:
        self.contract.validate_input_artifacts(context.artifacts)
        return True

    def get_metadata(self) -> dict:
        payload = super().get_metadata()
        payload["contract"] = self.contract.to_dict()
        return payload
