# FlexiHome VPP

FlexiHome is a development-stage Virtual Power Plant platform for residential energy flexibility. The current system models how aggregated homes with BESS, EV, HVAC, and PV resources can be forecasted, selected, and dispatched for Fingrid-oriented reserve products.

This branch, `codex-fastapi-api-layer`, is the active development branch. It contains the Streamlit operator dashboard, FastAPI optimizer service, plugin/pipeline architecture, Airflow DAGs, and the centralized two-layer MPC control prototype.

## Product Direction

FlexiHome is moving from a research dashboard toward a startup-grade VPP software foundation:

- centralized portfolio optimization for residential flexibility
- traceable household and appliance participation
- simulated gateway-level telemetry and commands
- clean API artifacts for downstream services
- professional dashboard views for market, control, and contribution analysis
- modular forecasting, optimization, pipeline, and orchestration components

The current version is still a simulation and development environment. It does not implement physical device adapters, residential local controllers, metering certification, market prequalification, settlement, or production cybersecurity controls.

## Current Control Architecture

FlexiHome uses a centralized two-layer control structure.

```text
Market and portfolio data
        |
        v
Forecasting models
        |
        v
15-minute outer MPC
  - reserve bidding
  - aggregate resource allocation
  - revenue, degradation, comfort, and state penalties
        |
        v
Usage-aware household/appliance selection
  - BESS, EV, HVAC, PV device roster
  - activation counters
  - throughput counters
  - consecutive-use limits
  - cooldown rotation
        |
        v
Centralized 4-second lower MPC
  - 20-second look-ahead horizon
  - simulated gateway telemetry
  - gateway command summaries
  - requested vs delivered reserve trace
```

The lower layer is implemented as MPC, not a reactive tracking controller. If the lower solver cannot solve an interval, the cycle is logged with failed status and shortfall. The controller does not switch to a greedy or reactive fallback path.

## Implemented Plan Status

| Plan item | Status | Notes |
| --- | --- | --- |
| Keep 15-minute outer MPC | Implemented | Existing receding-horizon optimizer remains the supervisory layer. |
| Add centralized 4-second lower MPC | Implemented | `CentralizedVPPController` runs with `inner_dt_seconds=4` and `inner_mpc_horizon_seconds=20`. |
| Remove active reactive fallback | Implemented | Failed lower-MPC cycles log failed status and shortfall instead of switching controller mode. |
| Add household/appliance roster | Implemented | Synthetic roster includes household, device, device type, gateway, protocol, power limits, state, response time, and usage counters. |
| Add usage-aware rotation | Implemented | Devices are penalized by usage, activations, throughput, and cooldown; cooldown and max-consecutive limits are enforced. |
| Add centralized gateway abstraction | Implemented for simulation | Gateway IDs and protocol labels are simulated. No local residential hardware or Modbus adapter is implemented. |
| Extend API defaults | Implemented | Request defaults use `inner_controller_mode="mpc"`, 4-second step, 20-second horizon, usage-aware rotation, and simulated centralized gateway mode. |
| Extend API/result artifacts | Implemented | Results include household/appliance contributions, upper device schedule, inner MPC trace, gateway commands, and usage/fatigue summary. |
| Export contribution CSVs | Implemented | CLI and pipeline serialization write the new MPC and contribution artifacts. |
| Dashboard Apply Scenario form | Implemented | Scenario controls are inside an Apply form, the last applied portfolio bundle is reused across reruns, and stale MPC results are marked when inputs change. |
| Dashboard MPC visualizations | Implemented | Upper selection, appliance mix, fatigue, lower reserve tracking, allocation, diagnostics, and gateway command tables are available. |
| CI smoke coverage | Implemented | CI runs API, plugin, pipeline, Airflow, forecasting, optimization, centralized MPC, and dashboard scenario-apply smoke tests. |

## Key Modules

- `app.py`  
  Streamlit dashboard for scenario setup, forecasting, MPC simulation, contribution analysis, and exports.

- `flexihome/core/engine.py`  
  Shared simulation, forecasting, outer MPC, and orchestration logic used by the dashboard, API, scripts, and pipeline.

- `flexihome/core/centralized_controller.py`  
  Centralized household roster generation, usage-aware selection, simulated gateway dispatch, and 4-second lower MPC.

- `flexihome/api/`  
  FastAPI schemas, endpoints, job services, serialization, and optional TimescaleDB persistence.

- `flexihome/pipeline/`  
  Contract-driven pipeline stages for market ingestion, features, forecasting, optimization, and result serialization.

- `flexihome/plugins/`  
  Swappable forecasting and optimization plugin defaults.

- `airflow/dags/`  
  Production-style DAG definitions for scheduled pipeline operation.

- `scripts/`  
  Local run scripts and smoke tests for API, plugins, pipelines, forecasting, optimization, and centralized MPC.

## Main Artifacts

Optimization runs produce the established aggregate outputs plus the new traceable MPC artifacts:

- `history.csv`
- `tracking_4s.csv`
- `first_schedule.csv`
- `household_contributions.csv`
- `appliance_contributions.csv`
- `upper_device_schedule.csv`
- `inner_mpc_trace.csv`
- `gateway_commands.csv`
- `usage_fatigue_summary.csv`
- `optimization_summary.json`

The API result payload includes the same data families:

- `history`
- `tracking_4s`
- `first_schedule`
- `household_contributions`
- `appliance_contributions`
- `upper_device_schedule`
- `inner_mpc_trace`
- `gateway_commands`
- `usage_fatigue_summary`
- `summary`
- `compliance`

## Dashboard Views

The dashboard is designed as an operational development tool, not a marketing page.

- Home: portfolio size, market data status, traceable device count, simulated gateway count
- Training Data Explorer: portfolio, market, weather, and flexibility time series
- Response Models: BESS, EV, PV, and HVAC response dynamics
- Forecasting Lab: model training, evaluation, and feature importance
- MPC Optimizer and Market Participation: outer MPC configuration, progress, solver status, upper selection, household/appliance contribution, and fatigue
- Simulation and Impact Visualizations: optimized load, reserve tracking, lower MPC allocation, diagnostics, and gateway commands
- Advanced / Export: JSON, CSV, PDF, and local run artifact inspection

Scenario controls are applied through an `Apply Scenario` form. Changing sliders does not immediately regenerate the portfolio; the dashboard keeps showing the last applied scenario, reuses the last applied portfolio bundle across normal reruns, and marks MPC results stale when inputs change.

## API Defaults

The optimization API defaults are aligned with the centralized MPC plan:

```json
{
  "inner_controller_mode": "mpc",
  "inner_dt_seconds": 4,
  "inner_mpc_horizon_seconds": 20,
  "rotation_strategy": "usage_aware",
  "gateway_mode": "simulated_centralized"
}
```

Only these lower-controller values are currently supported in this branch. This is intentional: the active path should remain centralized lower-layer MPC.

## Local Setup

Use Windows Command Prompt or PowerShell from the repository root.

```cmd
git clone https://github.com/arkm315787/Real_Time_Control_Energy-Flexibility.git
cd Real_Time_Control_Energy-Flexibility
git switch codex-fastapi-api-layer
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

If the repository is already cloned:

```cmd
cd /d "C:\Users\Kasutaja\OneDrive - Tallinna Tehnikaülikool\Documents\Playground\Real_Time_Control_Energy-Flexibility"
git fetch origin
git switch codex-fastapi-api-layer
git pull
.venv\Scripts\activate
```

## Run The Dashboard

```cmd
python -m streamlit run app.py
```

Open the printed local URL, usually:

```text
http://localhost:8501
```

## Run The FastAPI Service

```cmd
python scripts\run_api.py --reload
```

Useful endpoints:

- `GET /health`
- `GET /plugins`
- `POST /forecast`
- `POST /optimize`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`

The API uses an in-memory job store by default. TimescaleDB persistence is optional.

## Run A Local Optimization

```cmd
python scripts\run_optimization.py --days 1 --n-homes 80 --horizon-hours 1 --dispatch-hours 1 --output-root runs
```

The run writes CSV and JSON artifacts under `runs/optimization_*`.

## Run Checks

Core local checks:

```cmd
python -m py_compile app.py flexihome/core/engine.py flexihome/core/centralized_controller.py flexihome/api/services.py
python scripts\centralized_mpc_smoke.py
python scripts\dashboard_scenario_smoke.py
python scripts\api_smoke.py
python scripts\plugin_smoke.py
python scripts\pipeline_smoke.py
python scripts\run_optimization.py --days 1 --n-homes 80 --horizon-hours 1 --dispatch-hours 1 --output-root runs
```

GitHub CI also runs:

- compile checks
- API smoke test
- plugin smoke test
- centralized 4-second MPC smoke test
- dashboard scenario apply smoke test
- pipeline smoke test
- Airflow DAG smoke test
- forecasting module smoke test
- optimization module smoke test

## Optional Market Data

The system runs offline with synthetic market data by default. For real market overlays, set the relevant API keys before starting the dashboard or API:

```cmd
set ENTSOE_API_KEY=<your-entsoe-security-token>
set FINGRID_API_KEY=<your-fingrid-open-data-key>
```

Fingrid requests use `https://data.fingrid.fi/api` with the `x-api-key` header. The connector includes pacing and retry handling to reduce accidental API throttling.

## Optional TimescaleDB

Start the local database:

```cmd
docker compose up -d timescaledb
```

Configure the API:

```cmd
set FLEXIHOME_JOB_STORE=timescale
set FLEXIHOME_TIMESCALE_DSN=postgresql://flexihome:flexihome@localhost:5432/flexihome
python scripts\run_api.py --reload
```

If the DSN is configured but unavailable, the API falls back to in-memory storage unless `FLEXIHOME_TIMESCALE_STRICT=1` is set.

## Airflow Orchestration

Airflow DAGs are included for production-style orchestration:

- `flexihome_market_data_pipeline`
- `flexihome_forecasting_pipeline`
- `flexihome_optimization_pipeline`
- `flexihome_results_pipeline`
- `flexihome_local_pipeline`

Start Airflow dependencies with:

```cmd
docker compose up -d
```

Validate DAG definitions locally:

```cmd
python scripts\airflow_dag_smoke.py
```

## Development Roadmap

Important work that remains before this can become an operational VPP platform:

- real telemetry ingestion layer
- secure gateway/device command adapter design
- production Modbus, inverter, EVSE, thermostat, or aggregator protocol integrations
- household enrollment and asset registry
- metering, baseline validation, and settlement workflows
- market prequalification evidence and compliance reporting
- cybersecurity model, device authorization, audit logs, and command safety checks
- persistent time-series telemetry and command history
- fuller browser/UI regression tests for the Streamlit dashboard
- solver scalability benchmarking for larger portfolios
- deployment packaging, observability, and operator runbooks

## Branch Strategy

The latest implementation is on:

```text
codex-fastapi-api-layer
```

Branch relationship at the time this README was updated:

```text
main
  -> feature/plugin-architecture
      -> codex-fastapi-api-layer
```

`main` is currently stale, not redundant. The recommended cleanup is to merge `codex-fastapi-api-layer` into `main`, then delete `feature/plugin-architecture` and optionally delete `codex-fastapi-api-layer` after the merge.

## License And Usage

This repository is a development-stage VPP software prototype. It is suitable for research, product development, demos, and controlled simulation. It is not yet approved for real residential device control or live market operation.
