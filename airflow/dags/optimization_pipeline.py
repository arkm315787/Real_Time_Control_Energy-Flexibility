"""Production Airflow DAG for parameterized optimization jobs."""

from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from flexihome_airflow_common import DEFAULT_ARGS, PRODUCTION_TAGS, RUNS_ROOT, START_DATE, project_command


with DAG(
    dag_id="flexihome_optimization_pipeline",
    description="Run registry-selected forecasting and optimizer plugins for MPC optimization.",
    default_args=DEFAULT_ARGS,
    start_date=START_DATE,
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=2),
    max_active_runs=1,
    tags=[*PRODUCTION_TAGS, "optimization", "plugins"],
) as dag:
    run_optimizer = BashOperator(
        task_id="run_parameterized_optimizer",
        bash_command=project_command(
            "python scripts/run_optimization.py "
            "--market-mode {{ dag_run.conf.get('market_mode', 'Combined') }} "
            "--resource-mode \"{{ dag_run.conf.get('resource_mode', 'Hybrid portfolio') }}\" "
            "--forecaster-plugin {{ dag_run.conf.get('forecaster_plugin', 'xgboost_default') }} "
            "--optimizer-plugin {{ dag_run.conf.get('optimizer_plugin', 'pulp_default') }} "
            "--horizon-hours {{ dag_run.conf.get('horizon_hours', 24) }} "
            "--dispatch-hours {{ dag_run.conf.get('dispatch_hours', 24) }} "
            "--days {{ dag_run.conf.get('days', 7) }} "
            "--n-homes {{ dag_run.conf.get('n_homes', 1500) }} "
            "--freq-minutes {{ dag_run.conf.get('freq_minutes', 15) }} "
            "--market-data-mode {{ dag_run.conf.get('market_data_mode', 'synthetic') }} "
            "--market-lookback-days {{ dag_run.conf.get('market_lookback_days', 30) }} "
            f"--output-root {RUNS_ROOT}"
        ),
    )

    validate_optimization_artifacts = BashOperator(
        task_id="validate_optimization_artifacts",
        bash_command=project_command(
            f"python scripts/validate_pipeline_manifest.py --root {RUNS_ROOT} --prefix optimization --require-file optimization_summary.json"
        ),
    )

    run_optimizer >> validate_optimization_artifacts
