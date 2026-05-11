"""Legacy compact Airflow DAG for local Coverly forecasting and optimization runs."""

from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from flexihome_airflow_common import DEFAULT_ARGS, PRODUCTION_TAGS, RUNS_ROOT, START_DATE, project_command


with DAG(
    dag_id="flexihome_local_pipeline",
    description="Run Coverly forecasting and MPC optimization modules as local artifacts.",
    default_args=DEFAULT_ARGS,
    start_date=START_DATE,
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=2),
    max_active_runs=1,
    tags=[*PRODUCTION_TAGS, "local", "forecasting", "optimization"],
) as dag:
    forecast = BashOperator(
        task_id="run_forecasting_module",
        bash_command=project_command(
            "python scripts/run_forecasting.py "
            "--target {{ dag_run.conf.get('target', 'fcrn_capacity_eur_per_mw_h') }} "
            "--forecaster-plugin {{ dag_run.conf.get('forecaster_plugin', 'xgboost_default') }} "
            "--days {{ dag_run.conf.get('days', 2) }} "
            "--n-homes {{ dag_run.conf.get('n_homes', 80) }} "
            "--market-data-mode {{ dag_run.conf.get('market_data_mode', 'synthetic') }} "
            "--market-lookback-days {{ dag_run.conf.get('market_lookback_days', 30) }} "
            f"--output-root {RUNS_ROOT}"
        ),
    )

    optimize = BashOperator(
        task_id="run_optimization_module",
        bash_command=project_command(
            "python scripts/run_optimization.py "
            "--market-mode {{ dag_run.conf.get('market_mode', 'Combined') }} "
            "--resource-mode \"{{ dag_run.conf.get('resource_mode', 'Hybrid portfolio') }}\" "
            "--forecaster-plugin {{ dag_run.conf.get('forecaster_plugin', 'xgboost_default') }} "
            "--optimizer-plugin {{ dag_run.conf.get('optimizer_plugin', 'pulp_default') }} "
            "--horizon-hours {{ dag_run.conf.get('horizon_hours', 1) }} "
            "--dispatch-hours {{ dag_run.conf.get('dispatch_hours', 1) }} "
            "--days {{ dag_run.conf.get('days', 2) }} "
            "--n-homes {{ dag_run.conf.get('n_homes', 80) }} "
            "--market-data-mode {{ dag_run.conf.get('market_data_mode', 'synthetic') }} "
            "--market-lookback-days {{ dag_run.conf.get('market_lookback_days', 30) }} "
            f"--output-root {RUNS_ROOT}"
        ),
    )

    forecast >> optimize
