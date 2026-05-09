# FlexiHome VPP

FlexiHome is a development-stage Virtual Power Plant platform for residential energy flexibility. The current system models how aggregated homes with BESS, EV, HVAC, and PV resources can be forecasted, selected, and dispatched for Fingrid-oriented reserve products.

This branch, `codex/vpp-mvp-market-corrections`, is the active VPP MVP market/control branch. It builds on the `codex-fastapi-api-layer` API and pipeline foundation, then adds Fingrid-oriented market participation, risk-aware bidding, VPP technical audit screens, and the production-style 4-second centralized controller simulation.

## Product Direction

FlexiHome is moving from a research dashboard toward a startup-grade VPP software foundation:

- centralized portfolio optimization for residential flexibility
- traceable household and appliance participation
- simulated gateway-level telemetry and commands
- market-product eligibility, bid granularity, and socket-stacking checks
- VPP technical compliance evidence for simulation review
- clean API artifacts for downstream services
- professional dashboard views for market, control, and contribution analysis
- modular forecasting, optimization, pipeline, and orchestration components

The current version is still a simulation and development environment. It does not implement physical device adapters, residential local controllers, metering certification, market prequalification, settlement, or production cybersecurity controls.

The compliance layer is therefore a prequalification-prep screen, not a certification claim. It now audits aFRR 30-second start, 5-minute activation, 90-110% tracking envelope, 15-minute delivered-energy reconciliation, FCR-N 60/180-second step response, and a simulated FCR stability margin screen. Formal Fingrid prequalification still requires approved measurement data, TSO-reviewed test reports, telemetry, and settlement processes.

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
  - response-time eligibility for each market product
  - risk quantile and reserve buffer for physical derating
  - degradation, comfort, and asset-fatigue costs in the optimizer
  - non-delivery and activation-volatility costs reported for audit
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
  - one-step proportional tracking
  - active roster plus standby buffer roster
  - rising-error recovery with fast-resource priority
  - simulated gateway telemetry
  - gateway command summaries
  - requested vs delivered reserve trace
```

The lower layer is implemented as a one-step proportional tracking allocator, not a second scheduler. The dashboard does not run it automatically: first run the upper MPC layer, start the independent Market Pressure process, and then arm the centralized 4-second tracker. If the lower MPC is armed before the market start timestamp, it waits until the live market clock begins before attaching. The upper layer selects an active resource roster and a standby buffer roster from devices that satisfy availability, response-time, energy, usage, and cooldown limits. The executable plan is derated to the selected active roster when a paper bid exceeds physically selected capacity. Every 4-second tick, the lower layer clips the TSO request to the committed product limit, then distributes the request proportionally across the active BESS, EV, HVAC, and PV pools. If the predicted tracking error exceeds the 10% tolerance and is rising, the lower layer enters recovery mode and allocates the residual error to standby buffer capacity using fast-resource priority: BESS, then EV, then HVAC, then PV.

## Implemented Plan Status

| Plan item | Status | Notes |
| --- | --- | --- |
| Keep 15-minute outer MPC | Implemented | Existing receding-horizon optimizer remains the supervisory layer. |
| Add centralized 4-second lower MPC | Implemented | `CentralizedVPPController` runs with `inner_dt_seconds=4` and a one-step proportional tracking horizon. |
| Remove active reactive fallback | Implemented | Failed lower-MPC cycles log failed status and shortfall instead of switching controller mode. |
| Add household/appliance roster | Implemented | Synthetic roster includes household, device, device type, gateway, protocol, power limits, state, response time, and usage counters. |
| Add usage-aware rotation | Implemented | Devices are penalized by usage, activations, throughput, and cooldown; cooldown and max-consecutive limits are enforced. The upper selection separates active and buffer devices. |
| Add centralized gateway abstraction | Implemented for simulation | Gateway IDs and protocol labels are simulated. No local residential hardware or Modbus adapter is implemented. |
| Extend API defaults | Implemented | Request defaults use `inner_controller_mode="mpc"`, 4-second step, one-step lower tracking, usage-aware rotation, and simulated centralized gateway mode. |
| Extend API/result artifacts | Implemented | Results include household/appliance contributions, upper device schedule, inner MPC trace, gateway commands, and usage/fatigue summary. |
| Export contribution CSVs | Implemented | CLI and pipeline serialization write the new MPC and contribution artifacts. |
| Dashboard Apply Scenario form | Implemented | Scenario controls are inside an Apply form, the last applied portfolio bundle is reused across reruns, and stale MPC results are marked when inputs change. |
| Dashboard MPC visualizations | Implemented | Upper selection, appliance mix, fatigue, lower reserve tracking, allocation, diagnostics, and gateway command tables are available. |
| Risk-aware market bidding | Implemented for MVP | Upper MPC derates availability by bid quantile and buffer, prices non-delivery risk, activation uncertainty, and fatigue, and marks market slots as Participate/Wait. |
| Decoupled market and lower execution | Implemented | Dashboard upper decisions create a socket, Market Pressure runs on a wall-clock countdown, and the lower MPC attaches after the market starts. |
| Product technical audit | Implemented for simulation | 4-second traces are audited for aFRR timing, aFRR 90-110% envelope, settlement-period energy difference, FCR-N 60/180-second response, and FCR stability screening. |
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
- `afrr_energy_audit.csv`
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
- `afrr_energy_audit`
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

The MPC tab now follows a market-operator workflow:

- choose a market product, dispatch window, planning horizon, and risk policy
- run the risk-aware upper MPC layer to create the executable reserve socket
- inspect bidirectional BESS/EV/HVAC/PV commitments, risk-adjusted profit, buffers, and Participate/Wait decisions
- start the independent Market Pressure process with a wall-clock countdown delay
- arm the lower MPC; if it is armed early, it waits for the market start timestamp before attaching
- reveal the live market and lower MPC traces according to elapsed wall-clock time
- inspect whether the lower layer stayed in normal proportional mode or entered buffer recovery mode

## API Defaults

The optimization API defaults are aligned with the centralized MPC plan:

```json
{
  "inner_controller_mode": "mpc",
  "inner_dt_seconds": 4,
  "inner_mpc_horizon_seconds": 4,
  "rotation_strategy": "usage_aware",
  "gateway_mode": "simulated_centralized",
  "execute_lower_mpc": true,
  "risk_quantile": 0.8,
  "reserve_buffer_pct": 0.08
}
```

Only these lower-controller values are currently supported in this branch. This is intentional: the active path should remain centralized lower-layer MPC.

## Optimization Weight Units

The upper MPC uses unit-aware economic weights:

| Dashboard control | Default | Unit | Meaning |
| --- | ---: | --- | --- |
| Battery wear cost | 35 | EUR/MWh throughput | BESS activation cycling cost; EV activation uses 40% of this value. |
| Comfort cost | 480 | EUR/degC-hour | Indoor comfort-band violation cost, multiplied by temperature slack and interval hours. |
| EV shortfall cost | 1000 | EUR/MWh shortfall | Penalty for missing EV required departure energy. |
| Non-delivery cost | 900 | EUR/MWh exposure | Market penalty/risk premium for reserve capacity that may not be delivered. |
| Activation volatility cost | 100 | EUR/MWh-equivalent | Heuristic cost for bidding during uncertain activation periods. |
| Customer fatigue cost | 75 | EUR/MWh-equivalent | Heuristic cost for repeated customer/device use. |

Comfort is time-normalized as `comfort_weight * degC_slack * dt_h`, so changing the dashboard timestep does not silently change the value of comfort protection.

## Local Setup

Use Python 3.11 from Windows Command Prompt or PowerShell.

```cmd
git clone https://github.com/arkm315787/Real_Time_Control_Energy-Flexibility.git
cd Real_Time_Control_Energy-Flexibility
git switch codex/vpp-mvp-market-corrections
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

If the repository is already cloned:

```cmd
cd /d "C:\path\to\Real_Time_Control_Energy-Flexibility"
git fetch origin
git switch codex/vpp-mvp-market-corrections
git pull
.venv\Scripts\activate
```

The main runtime dependencies are in `requirements.txt`. The file intentionally pins `numpy<2` and compatible Pandas, SciPy, and control-system ranges because the VPP technical-audit path imports `python-control` and Matplotlib-backed response tooling. Without the NumPy upper bound, local installs can resolve to NumPy 2.x and fail before the dashboard or smoke tests start.

Use `requirements-dev.txt` for the FastAPI smoke test client. Use `requirements-airflow.txt` only inside the Airflow Docker image. Install `requirements-lstm.txt` only if you want to try the optional LSTM forecaster in the Forecasting Lab.

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
python scripts\run_optimization.py --days 1 --n-homes 1200 --horizon-hours 1 --dispatch-hours 1 --output-root runs
```

The run writes CSV and JSON artifacts under `runs/optimization_*`.

## Run Checks

Core local checks:

```cmd
python -m py_compile app.py flexihome/core/engine.py flexihome/core/centralized_controller.py flexihome/core/market_data.py flexihome/api/services.py
python scripts\centralized_mpc_smoke.py
python scripts\risk_aware_mpc_smoke.py
python scripts\compliance_pending_state_smoke.py
python scripts\vpp_mvp_multi_scenario_smoke.py
python scripts\vpp_technical_compliance_smoke.py
python scripts\dashboard_scenario_smoke.py
python scripts\api_smoke.py
python scripts\plugin_smoke.py
python scripts\pipeline_smoke.py
python scripts\run_optimization.py --days 1 --n-homes 1200 --horizon-hours 1 --dispatch-hours 1 --output-root runs
```

GitHub CI also runs:

- compile checks
- API smoke test
- plugin smoke test
- centralized 4-second MPC smoke test
- risk-aware upper MPC smoke test
- compliance pending-state smoke test
- multi-scenario VPP MVP smoke test
- five-context technical compliance smoke test
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

The latest VPP MVP market/control implementation is on:

```text
codex/vpp-mvp-market-corrections
```

Branch relationship at the time this README was updated:

```text
main
  -> feature/plugin-architecture
      -> codex-fastapi-api-layer
          -> codex/vpp-mvp-market-corrections
```

`codex-fastapi-api-layer` is the API/pipeline base for this branch. `main` is currently stale, not redundant. The recommended cleanup is to merge the VPP branch forward into `main` when it is accepted, then delete superseded intermediate branches only after their changes are safely included.

## License And Usage

This repository is a development-stage VPP software prototype. It is suitable for research, product development, demos, and controlled simulation. It is not yet approved for real residential device control or live market operation.
