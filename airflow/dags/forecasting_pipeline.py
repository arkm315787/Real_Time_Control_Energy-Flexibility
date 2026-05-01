"""Production Airflow DAG for parameterized forecasting jobs."""

from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from flexihome_airflow_common import DEFAULT_ARGS, PRODUCTION_TAGS, RUNS_ROOT, START_DATE, project_command


with DAG(
    dag_id="flexihome_forecasting_pipeline",
    description="Train and validate a registry-selected forecasting plugin for a configured target.",
    default_args=DEFAULT_ARGS,
    start_date=START_DATE,
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=1),
    max_active_runs=2,
    tags=[*PRODUCTION_TAGS, "forecasting", "plugins"],
) as dag:
    train_forecaster = BashOperator(
        task_id="train_parameterized_forecaster",
        bash_command=project_command(
            "python scripts/run_forecasting.py "
            "--target {{ dag_run.conf.get('target', 'fcrn_capacity_eur_per_mw_h') }} "
            "--forecaster-plugin {{ dag_run.conf.get('forecaster_plugin', 'xgboost_default') }} "
            "--lags {{ dag_run.conf.get('lags', 4) }} "
            "--horizon-steps {{ dag_run.conf.get('horizon_steps', 1) }} "
            "--days {{ dag_run.conf.get('days', 7) }} "
            "--n-homes {{ dag_run.conf.get('n_homes', 1500) }} "
            "--freq-minutes {{ dag_run.conf.get('freq_minutes', 15) }} "
            "--market-data-mode {{ dag_run.conf.get('market_data_mode', 'synthetic') }} "
            "--market-lookback-days {{ dag_run.conf.get('market_lookback_days', 30) }} "
            f"--output-root {RUNS_ROOT}"
        ),
    )

    validate_forecast_artifacts = BashOperator(
        task_id="validate_forecast_artifacts",
        bash_command=project_command(
            f"python scripts/validate_pipeline_manifest.py --root {RUNS_ROOT} --prefix forecast --require-file forecast_summary.json"
        ),
    )

    train_forecaster >> validate_forecast_artifacts
