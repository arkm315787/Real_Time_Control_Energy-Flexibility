"""Feature engineering contract stage."""

from __future__ import annotations

import pandas as pd

from flexihome.core.base import BasePipelineStage, PipelineContext
from flexihome.pipeline.contracts import DEFAULT_MPC_PREDICTORS, PORTFOLIO_DATA_CONTRACT, ArtifactContract, PipelineRunConfig, StageIOContract


class FeatureEngineeringStage(BasePipelineStage):
    """Validate and publish feature columns used by forecasting plugins."""

    contract = StageIOContract(
        stage_name="feature_engineering",
        input_data=PORTFOLIO_DATA_CONTRACT,
        output_data=PORTFOLIO_DATA_CONTRACT,
        output_artifacts=(
            ArtifactContract("feature_columns", "Selected numeric predictor columns."),
            ArtifactContract("feature_contract", "Input/output schema description for the feature frame."),
        ),
    )

    def __init__(self, config: PipelineRunConfig, name: str | None = None, version: str = "1.0.0") -> None:
        super().__init__(name or "feature_engineering", version=version)
        self.config = config

    def execute(self, context: PipelineContext) -> PipelineContext:
        assert context.data is not None
        frame = context.data.copy()
        predictors = [column for column in self.config.predictors or DEFAULT_MPC_PREDICTORS if column in frame.columns]
        numeric_columns = list(frame.select_dtypes(include=["number", "bool"]).columns)
        feature_columns = [column for column in predictors if column in numeric_columns]
        if not feature_columns:
            raise ValueError("No numeric feature columns are available for forecasting.")
        context.data = frame
        context.with_artifact("feature_columns", feature_columns)
        context.with_artifact("numeric_columns", numeric_columns)
        context.with_artifact("feature_contract", self.contract.output_data.describe(frame))
        context.with_metadata(self.name, {"feature_columns": feature_columns, "numeric_column_count": len(numeric_columns)})
        return context

    def validate_inputs(self, context: PipelineContext) -> bool:
        self.contract.validate_input_artifacts(context.artifacts)
        if not isinstance(context.data, pd.DataFrame):
            raise ValueError("FeatureEngineeringStage requires context.data as a pandas DataFrame.")
        self.contract.input_data.validate(context.data)
        return True

    def get_required_columns(self) -> list[str]:
        return list(self.contract.input_data.required_columns)

    def get_metadata(self) -> dict:
        payload = super().get_metadata()
        payload["contract"] = self.contract.to_dict()
        return payload
