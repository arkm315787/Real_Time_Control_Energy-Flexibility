"""Market-data and portfolio ingestion pipeline stage."""

from __future__ import annotations

from flexihome.core.base import BasePipelineStage, PipelineContext
from flexihome.core.data import generate_synthetic_portfolio
from flexihome.pipeline.contracts import PORTFOLIO_DATA_CONTRACT, ArtifactContract, PipelineRunConfig, StageIOContract


class MarketDataIngestionStage(BasePipelineStage):
    """Generate the portfolio frame and apply configured market-data overlays."""

    contract = StageIOContract(
        stage_name="market_data_ingestion",
        output_data=PORTFOLIO_DATA_CONTRACT,
        output_artifacts=(
            ArtifactContract("portfolio_bundle", "Raw generated portfolio bundle."),
            ArtifactContract("preview_4s", "Four-second activation preview frame."),
            ArtifactContract("portfolio_summary", "Portfolio and market-data summary metadata."),
            ArtifactContract("market_data_status", "Market-data source and ingestion diagnostics."),
        ),
    )

    def __init__(self, config: PipelineRunConfig, name: str | None = None, version: str = "1.0.0") -> None:
        super().__init__(name or "market_data_ingestion", version=version)
        self.config = config

    def execute(self, context: PipelineContext) -> PipelineContext:
        bundle = generate_synthetic_portfolio(**self.config.portfolio_kwargs())
        data = bundle["data"]
        self.contract.output_data.validate(data)
        context.data = data
        context.with_artifact("portfolio_bundle", bundle)
        context.with_artifact("preview_4s", bundle.get("preview_4s"))
        context.with_artifact("portfolio_summary", bundle.get("summary", {}))
        context.with_artifact("market_data_status", bundle.get("summary", {}).get("market_data_status", {}))
        context.with_metadata(self.name, {"rows": len(data), "market_data_mode": bundle.get("summary", {}).get("market_data_source")})
        return context

    def validate_inputs(self, context: PipelineContext) -> bool:
        return True

    def get_required_columns(self) -> list[str]:
        return []

    def get_metadata(self) -> dict:
        payload = super().get_metadata()
        payload["contract"] = self.contract.to_dict()
        return payload
