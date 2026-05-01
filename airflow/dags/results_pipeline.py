"""Production Airflow DAG for outgoing serialized results."""

from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from flexihome_airflow_common import DEFAULT_ARGS, PRODUCTION_TAGS, RUNS_ROOT, START_DATE, project_command


with DAG(
    dag_id="flexihome_results_pipeline",
    description="Run the full contract pipeline and validate outgoing result manifests.",
    default_args=DEFAULT_ARGS,
    start_date=START_DATE,
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=2),
    max_active_runs=1,
    tags=[*PRODUCTION_TAGS, "outgoing-results", "manifest"],
) as dag:
    serialize_results = BashOperator(
        task_id="serialize_results_manifest",
        bash_command=project_command(
            "python scripts/run_pipeline.py "
            "--market-data-mode {{ dag_run.conf.get('market_data_mode', 'synthetic') }} "
            "--market-lookback-days {{ dag_run.conf.get('market_lookback_days', 30) }} "
            "--forecaster-plugin {{ dag_run.conf.get('forecaster_plugin', 'xgboost_default') }} "
            "--optimizer-plugin {{ dag_run.conf.get('optimizer_plugin', 'pulp_default') }} "
            "--market-mode {{ dag_run.conf.get('market_mode', 'Combined') }} "
            "--resource-mode \"{{ dag_run.conf.get('resource_mode', 'Hybrid portfolio') }}\" "
            "--horizon-hours {{ dag_run.conf.get('horizon_hours', 24) }} "
            "--dispatch-hours {{ dag_run.conf.get('dispatch_hours', 24) }} "
            "--days {{ dag_run.conf.get('days', 7) }} "
            "--n-homes {{ dag_run.conf.get('n_homes', 1500) }} "
            "--freq-minutes {{ dag_run.conf.get('freq_minutes', 15) }} "
            f"--output-root {RUNS_ROOT}"
        ),
    )

    validate_results = BashOperator(
        task_id="validate_results_manifest",
        bash_command=project_command(
            f"python scripts/validate_pipeline_manifest.py --root {RUNS_ROOT} --prefix pipeline --require-file manifest.json"
        ),
    )

    serialize_results >> validate_results
