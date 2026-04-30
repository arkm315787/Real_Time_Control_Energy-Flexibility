"""Airflow DAG for local FlexiHome forecasting and optimization runs."""

from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator


DEFAULT_ARGS = {
    "owner": "flexihome",
    "retries": 0,
}


with DAG(
    dag_id="flexihome_local_pipeline",
    description="Run FlexiHome forecasting and MPC optimization modules as local artifacts.",
    default_args=DEFAULT_ARGS,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["flexihome", "forecasting", "optimization"],
) as dag:
    forecast = BashOperator(
        task_id="run_forecasting_module",
        bash_command=(
            "cd /opt/airflow/project && "
            "python scripts/run_forecasting.py "
            "--target {{ dag_run.conf.get('target', 'fcrn_capacity_eur_per_mw_h') }} "
            "--days {{ dag_run.conf.get('days', 2) }} "
            "--n-homes {{ dag_run.conf.get('n_homes', 80) }} "
            "--market-data-mode {{ dag_run.conf.get('market_data_mode', 'synthetic') }} "
            "--market-lookback-days {{ dag_run.conf.get('market_lookback_days', 30) }} "
            "--output-root /opt/airflow/project/runs"
        ),
    )

    optimize = BashOperator(
        task_id="run_optimization_module",
        bash_command=(
            "cd /opt/airflow/project && "
            "python scripts/run_optimization.py "
            "--market-mode {{ dag_run.conf.get('market_mode', 'Combined') }} "
            "--resource-mode \"{{ dag_run.conf.get('resource_mode', 'Hybrid portfolio') }}\" "
            "--horizon-hours {{ dag_run.conf.get('horizon_hours', 1) }} "
            "--dispatch-hours {{ dag_run.conf.get('dispatch_hours', 1) }} "
            "--days {{ dag_run.conf.get('days', 2) }} "
            "--n-homes {{ dag_run.conf.get('n_homes', 80) }} "
            "--market-data-mode {{ dag_run.conf.get('market_data_mode', 'synthetic') }} "
            "--market-lookback-days {{ dag_run.conf.get('market_lookback_days', 30) }} "
            "--output-root /opt/airflow/project/runs"
        ),
    )

    forecast >> optimize
