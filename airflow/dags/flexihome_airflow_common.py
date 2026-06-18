"""Shared Airflow defaults and command helpers for Residential VPP DAGs."""

from __future__ import annotations

from datetime import timedelta

import pendulum


PROJECT_ROOT = "/opt/airflow/project"
RUNS_ROOT = f"{PROJECT_ROOT}/runs"
START_DATE = pendulum.datetime(2026, 1, 1, tz="UTC")

DEFAULT_ARGS = {
    "owner": "residential_vpp",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=1),
}

PRODUCTION_TAGS = ["residential_vpp", "production"]


def project_command(command: str) -> str:
    return f"cd {PROJECT_ROOT} && {command}"
