"""FastAPI API layer for the FlexiHome optimizer."""

from __future__ import annotations

import asyncio
import os
from typing import List

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from flexihome.core.base.registry import get_global_registry
from flexihome.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResultResponse,
)
from flexihome.api.services import (
    model_to_dict,
    new_optimization_id,
    response_from_job,
    run_forecast,
    run_optimization_job,
)
from flexihome.api.store import build_optimization_store
from flexihome.pipeline import register_pipeline_stages
from flexihome.plugins import register_default_plugins


register_default_plugins()
register_pipeline_stages()
app = FastAPI(
    title="FlexiHome Optimizer",
    version="1.0.0",
    description="Machine-readable API for FlexiHome forecasting and MPC optimization.",
)
STORE = build_optimization_store()


@app.get("/")
async def root() -> dict:
    return {
        "service": "FlexiHome Optimizer",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
) -> OptimizationResponse:
    """Submit a portfolio optimization request and poll `/results/{id}`."""

    _validate_plugin(request.forecaster_plugin, get_global_registry().list_forecasters(), "forecaster")
    _validate_plugin(request.optimizer_plugin, get_global_registry().list_optimizers(), "optimizer")
    optimization_id = new_optimization_id()
    job = STORE.create(optimization_id, model_to_dict(request))
    background_tasks.add_task(run_optimization_job, STORE, optimization_id, request)
    return response_from_job(job)


@app.get("/results/{optimization_id}", response_model=OptimizationResultResponse)
async def get_results(optimization_id: str) -> OptimizationResultResponse:
    """Retrieve full optimization results for a queued, running, or completed job."""

    job = STORE.get(optimization_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Optimization not found")
    return OptimizationResultResponse(
        optimization_id=job.optimization_id,
        status=job.status,
        request=job.request,
        result=job.result or {},
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@app.get("/optimizations", response_model=List[OptimizationResponse])
async def list_optimizations() -> List[OptimizationResponse]:
    """List recent optimization jobs."""

    return [response_from_job(job) for job in STORE.list()]


@app.post("/forecast", response_model=ForecastResponse)
async def forecast_next_horizon(request: ForecastRequest) -> ForecastResponse:
    """Generate a standalone forecast for one MPC target."""

    try:
        payload = await asyncio.to_thread(run_forecast, request)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ForecastResponse(**payload)


@app.get("/plugins")
async def list_plugins() -> dict:
    """List registered model and pipeline plugins."""

    registry = get_global_registry()
    return {
        "forecasters": registry.list_forecasters(),
        "optimizers": registry.list_optimizers(),
        "pipeline_stages": registry.list_pipeline_stages(),
    }


@app.get("/health")
async def health_check() -> dict:
    """Return service health for local and production readiness checks."""

    store_health = STORE.health()
    airflow_health = _airflow_health()
    degraded = store_health.get("timescaledb") == "unavailable" or airflow_health.get("airflow") == "unavailable"
    status = "degraded" if degraded else "healthy"
    return {
        "status": status,
        "fastapi": "ok",
        **store_health,
        **airflow_health,
    }


def _airflow_health() -> dict:
    url = os.getenv("FLEXIHOME_AIRFLOW_URL", "").strip().rstrip("/")
    if not url:
        return {"airflow": "not_configured"}
    try:
        response = requests.get(f"{url}/health", timeout=2)
        if response.ok:
            return {"airflow": "connected", "airflow_url": url}
        return {"airflow": "unavailable", "airflow_url": url, "airflow_error": f"HTTP {response.status_code}"}
    except requests.RequestException as exc:
        return {"airflow": "unavailable", "airflow_url": url, "airflow_error": str(exc)[:240]}


def _validate_plugin(name: str, available: dict, plugin_type: str) -> None:
    if name not in available:
        choices = ", ".join(sorted(available)) or "none"
        raise HTTPException(status_code=400, detail=f"Unknown {plugin_type} plugin '{name}'. Available: {choices}")


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics() -> str:
    """Return a small Prometheus-style metrics payload."""

    metrics = STORE.metrics()
    lines = [
        "# HELP flexihome_optimization_jobs Number of optimization jobs by status.",
        "# TYPE flexihome_optimization_jobs gauge",
    ]
    for status, value in metrics.items():
        if status == "total":
            continue
        lines.append(f'flexihome_optimization_jobs{{status="{status}"}} {value}')
    lines.append(f"flexihome_optimization_jobs_total {metrics['total']}")
    return "\n".join(lines) + "\n"

