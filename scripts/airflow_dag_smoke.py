"""Static smoke checks for Residential VPP Airflow DAG definitions.

This avoids requiring Airflow in the normal development environment while still
checking that production DAG files, retry/SLA settings, and plugin parameters
are present.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAGS = ROOT / "airflow" / "dags"


REQUIRED_DAGS = {
    "market_data_pipeline.py": "flexihome_market_data_pipeline",
    "forecasting_pipeline.py": "flexihome_forecasting_pipeline",
    "optimization_pipeline.py": "flexihome_optimization_pipeline",
    "results_pipeline.py": "flexihome_results_pipeline",
    "flexihome_pipeline.py": "flexihome_local_pipeline",
}


def assert_contains(path: Path, *patterns: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [pattern for pattern in patterns if pattern not in text]
    if missing:
        raise SystemExit(f"{path.name} missing required patterns: {missing}")


def main() -> None:
    common = DAGS / "flexihome_airflow_common.py"
    assert_contains(common, '"retries": 2', '"sla": timedelta(hours=1)', "retry_delay")

    for filename, dag_id in REQUIRED_DAGS.items():
        path = DAGS / filename
        if not path.exists():
            raise SystemExit(f"Missing DAG file: {path}")
        assert_contains(path, dag_id, "DEFAULT_ARGS", "dagrun_timeout", "max_active_runs")

    assert_contains(DAGS / "market_data_pipeline.py", "schedule=\"@hourly\"", "run_market_data_ingestion.py", "validate_market_artifacts")
    assert_contains(DAGS / "forecasting_pipeline.py", "forecaster_plugin", "run_forecasting.py", "validate_forecast_artifacts")
    assert_contains(DAGS / "optimization_pipeline.py", "forecaster_plugin", "optimizer_plugin", "run_optimization.py")
    assert_contains(DAGS / "results_pipeline.py", "run_pipeline.py", "validate_results_manifest")

    print("Airflow DAG smoke checks passed.")


if __name__ == "__main__":
    main()
