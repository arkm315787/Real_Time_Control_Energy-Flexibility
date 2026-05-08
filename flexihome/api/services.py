"""Application services behind the FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict
from uuid import uuid4

from flexihome.api.schemas import (
    ForecastRequest,
    OptimizationRequest,
    OptimizationResponse,
    PortfolioConfig,
)
from flexihome.api.serialization import dataframe_to_records, json_safe, serialize_optimization_result
from flexihome.api.store import OptimizationStore, StoredJob
from flexihome.core.base.registry import get_global_registry
from flexihome.core.data import generate_synthetic_portfolio
from flexihome.core.engine import (
    SUPPORTED_MPC_TARGETS,
    default_hvac_response_seconds,
)
from flexihome.core.mpc import run_mpc_controller
from flexihome.plugins import DEFAULT_FORECASTER_PLUGIN, DEFAULT_OPTIMIZER_PLUGIN, register_default_plugins


DEFAULT_PREDICTORS = [
    "temp_out_c",
    "irradiance_wm2",
    "base_load_kw",
    "pv_available_kw",
    "net_load_baseline_kw",
    "fcrn_capacity_eur_per_mw_h",
    "afrr_up_capacity_eur_per_mw_h",
    "afrr_down_capacity_eur_per_mw_h",
    "afrr_up_act_frac",
    "afrr_down_act_frac",
    "fcr_signed_act",
]


def plugin_registry():
    return register_default_plugins(get_global_registry())


def new_optimization_id() -> str:
    return f"opt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"


def model_to_dict(model) -> Dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def response_from_job(job: StoredJob) -> OptimizationResponse:
    summary = (job.result or {}).get("summary", {})
    compliance = (job.result or {}).get("compliance", {})
    compliance_checks = {key: bool(value) for key, value in compliance.items() if key.endswith("_ok")}
    return OptimizationResponse(
        optimization_id=job.optimization_id,
        status=job.status,
        total_revenue_eur=float(summary.get("total_revenue_eur", 0.0) or 0.0),
        delivered_up_mwh=float(summary.get("delivered_up_mwh", 0.0) or 0.0),
        delivered_down_mwh=float(summary.get("delivered_down_mwh", 0.0) or 0.0),
        requirement_score_pct=float(summary.get("requirement_score_pct", 0.0) or 0.0),
        compliance_checks=compliance_checks,
        timestamp=job.updated_at,
        results_url=f"/results/{job.optimization_id}",
        error=job.error,
    )


def portfolio_kwargs(config: PortfolioConfig) -> Dict:
    payload = model_to_dict(config)
    payload["start_date"] = str(payload["start_date"])
    return payload


def fleet_meta_from_bundle(bundle: Dict, config: PortfolioConfig, fcr_hvac_share_cap: float = 0.20) -> Dict[str, float]:
    df = bundle["data"]
    hvac_response_s = config.hvac_response_s or default_hvac_response_seconds(config.hvac_mode)
    return {
        "bess_energy_cap_mwh": float(df["bess_energy_cap_mwh"].iloc[0]),
        "setpoint_c": float(config.setpoint_c),
        "comfort_band_c": float(config.comfort_band_c),
        "n_hvac": int(bundle["summary"]["n_hvac"]),
        "hvac_mode": config.hvac_mode,
        "hvac_response_s": float(hvac_response_s),
        "fcr_hvac_share_cap": float(fcr_hvac_share_cap),
    }


def run_optimization_job(
    store: OptimizationStore,
    optimization_id: str,
    request: OptimizationRequest,
) -> None:
    store.mark_running(optimization_id)
    try:
        bundle = generate_synthetic_portfolio(**portfolio_kwargs(request.portfolio_config))
        df = bundle["data"]
        registry = plugin_registry()
        forecaster_plugin = request.forecaster_plugin or DEFAULT_FORECASTER_PLUGIN
        optimizer_plugin = request.optimizer_plugin or DEFAULT_OPTIMIZER_PLUGIN
        models = train_mpc_forecasters(df, seed=int(request.portfolio_config.seed), forecaster_plugin=forecaster_plugin)
        optimizer = registry.get_optimizer(optimizer_plugin)
        result = run_mpc_controller(
            df=df,
            models=models,
            fleet_meta=fleet_meta_from_bundle(bundle, request.portfolio_config, request.fcr_hvac_share_cap),
            market_mode=request.market_mode,
            resource_mode=request.resource_mode,
            horizon_hours=float(request.horizon_hours),
            dispatch_hours=float(request.dispatch_hours),
            penalty_weights={
                "degradation": float(request.degradation_weight),
                "comfort": float(request.comfort_weight),
                "departure": float(request.departure_weight),
                "risk_policy": str(request.risk_policy),
                "risk_quantile": float(request.risk_quantile),
                "reserve_buffer_pct": float(request.reserve_buffer_pct),
                "non_delivery": float(request.non_delivery_penalty),
                "activation_uncertainty": float(request.activation_uncertainty_weight),
                "asset_fatigue": float(request.asset_fatigue_weight),
            },
            preview_4s=bundle["preview_4s"],
            optimizer=optimizer,
            device_roster=bundle.get("device_roster"),
            inner_controller_mode=request.inner_controller_mode,
            inner_dt_seconds=int(request.inner_dt_seconds),
            inner_mpc_horizon_seconds=int(request.inner_mpc_horizon_seconds),
            rotation_strategy=request.rotation_strategy,
            gateway_mode=request.gateway_mode,
            execute_lower_mpc=bool(request.execute_lower_mpc),
        )
        market_data_status = bundle.get("summary", {}).get("market_data_status", {})
        result.setdefault("summary", {})["market_data_source"] = market_data_status.get("mode_used", "synthetic")
        result.setdefault("summary", {})["forecaster_plugin"] = forecaster_plugin
        result.setdefault("summary", {})["optimizer_plugin"] = optimizer_plugin
        result["market_data_status"] = market_data_status
        store.complete(optimization_id, serialize_optimization_result(result))
    except Exception as exc:
        store.fail(optimization_id, str(exc))


def train_mpc_forecasters(df, seed: int, forecaster_plugin: str) -> Dict:
    registry = plugin_registry()
    models: Dict = {}
    for target in SUPPORTED_MPC_TARGETS:
        forecaster = registry.get_forecaster(
            forecaster_plugin,
            name=f"{forecaster_plugin}_{target}",
            seed=int(seed),
        )
        forecaster.fit_from_frame(
            df=df,
            target=target,
            predictors=DEFAULT_PREDICTORS,
            lags=4,
            horizon_steps=1,
        )
        models[target] = forecaster
    return models


def run_forecast(request: ForecastRequest) -> Dict:
    if request.target not in SUPPORTED_MPC_TARGETS:
        supported = ", ".join(SUPPORTED_MPC_TARGETS)
        raise ValueError(f"Unsupported target '{request.target}'. Supported targets: {supported}")
    bundle = generate_synthetic_portfolio(**portfolio_kwargs(request.portfolio_config))
    df = bundle["data"]
    predictors = request.predictors or DEFAULT_PREDICTORS
    registry = plugin_registry()
    forecaster_plugin = request.forecaster_plugin or DEFAULT_FORECASTER_PLUGIN
    forecaster = registry.get_forecaster(
        forecaster_plugin,
        name=f"{forecaster_plugin}_{request.target}",
        seed=int(request.portfolio_config.seed),
    )
    comparison = forecaster.fit_from_frame(
        df=df,
        target=request.target,
        predictors=predictors,
        lags=int(request.lags),
        horizon_steps=int(request.horizon_steps),
    )
    return {
        "forecaster_plugin": forecaster_plugin,
        "target": forecaster.target,
        "metrics": json_safe(forecaster.metrics),
        "horizon_steps": forecaster.horizon_steps,
        "predictions": dataframe_to_records(comparison.tail(48)),
        "generated_at": datetime.now(timezone.utc),
    }

