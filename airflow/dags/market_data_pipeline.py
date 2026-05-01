"""Production Airflow DAG for incoming market-data ingestion."""

from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from flexihome_airflow_common import DEFAULT_ARGS, PRODUCTION_TAGS, RUNS_ROOT, START_DATE, project_command


with DAG(
    dag_id="flexihome_market_data_pipeline",
    description="Hourly incoming data pipeline: source checks, market-data aggregation, feature validation.",
    default_args=DEFAULT_ARGS,
    start_date=START_DATE,
    schedule="@hourly",
    catchup=False,
    dagrun_timeout=timedelta(hours=1),
    max_active_runs=1,
    tags=[*PRODUCTION_TAGS, "incoming-data", "market-data"],
) as dag:
    fetch_entso_e = BashOperator(
        task_id="fetch_entso_e_configuration",
        bash_command=project_command(
            "python -c \"import os; "
            "print('ENTSO-E security token configured:', bool(os.getenv('ENTSOE_API_KEY') or os.getenv('ENTSOE_SECURITY_TOKEN')))\""
        ),
    )

    fetch_fingrid = BashOperator(
        task_id="fetch_fingrid_configuration",
        bash_command=project_command(
            "python -c \"import os; "
            "print('Fingrid Open Data key configured:', bool(os.getenv('FINGRID_API_KEY') or os.getenv('FINGRID_OPENDATA_API_KEY')))\""
        ),
    )

    aggregate = BashOperator(
        task_id="aggregate_market_frame",
        bash_command=project_command(
            "python scripts/run_market_data_ingestion.py "
            "--market-data-mode {{ dag_run.conf.get('market_data_mode', 'auto') }} "
            "--market-lookback-days {{ dag_run.conf.get('market_lookback_days', 30) }} "
            "--days {{ dag_run.conf.get('days', 2) }} "
            "--n-homes {{ dag_run.conf.get('n_homes', 80) }} "
            "--freq-minutes {{ dag_run.conf.get('freq_minutes', 15) }} "
            f"--output-root {RUNS_ROOT}"
        ),
    )

    validate = BashOperator(
        task_id="validate_market_artifacts",
        bash_command=project_command(
            f"python scripts/validate_pipeline_manifest.py --root {RUNS_ROOT} --prefix market_data --require-artifact market_data_frame"
        ),
    )

    fetch_entso_e >> fetch_fingrid >> aggregate >> validate
