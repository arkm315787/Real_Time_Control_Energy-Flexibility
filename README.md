# FlexiHome: Aggregated Residential Flexibility for Fingrid Balancing Markets

`FlexiHome` is a Streamlit learning dashboard for exploring how aggregated residential resources can participate in Finnish balancing markets, with a strong focus on Fingrid `FCR-N` and `aFRR`.

The application combines:
- synthetic Finland-style household and weather data
- EV, BESS, PV, and HVAC flexibility models
- forecasting models for MPC inputs
- a receding-horizon MPC-style optimizer
- interactive educational visualizations for response speed, compliance, revenue, and what-if analysis

This branch contains the Streamlit dashboard plus the FastAPI optimizer service and their Python dependencies.

## Dashboard Scope

The dashboard is designed for:
- students learning demand response, reserve markets, and distributed flexibility
- researchers exploring aggregated residential flexibility
- engineers who want a self-contained prototype for balancing-market education
- energy professionals who need an intuitive visual explanation of fast and slow flexibility

The core educational questions it addresses are:
- How much reserve can an aggregated residential portfolio provide?
- Which resources are fast enough for `FCR-N`?
- How do `EV`, `BESS`, `PV`, and `HVAC` differ in activation behavior?
- How does forecasting affect MPC dispatch quality?
- How do market settings, weather, comfort bands, and portfolio size change results?

## Main Features

### 1. Synthetic portfolio generation

The app generates a configurable portfolio of households with:
- base residential load
- PV generation
- EV charging and available flexibility
- BESS headroom and SOC reference
- HVAC baseline demand and flexible thermal margin
- synthetic FCR/aFRR prices and activation signals

Adjustable controls include:
- aggregated homes
- EV, BESS, PV, and HVAC penetration
- simulation length
- simulation timestep
- weather severity and cloudiness
- comfort band and setpoint
- market scarcity / price stress
- HVAC technology mode

### 2. Response model comparison

The `Response Models` tab compares resource dynamics using:
- step response plots
- animated activation playback
- Bode magnitude and phase plots

The app distinguishes between:
- very fast BESS response
- fast EV response
- PV curtailment response
- HVAC behavior, including an inverter/variable-speed mode

### 3. Forecasting lab

The `Forecasting Lab` tab lets you train forecasting models for signals used by the MPC, such as:
- net baseline load
- PV availability
- FCR-N capacity prices
- aFRR up/down capacity prices
- activation fractions

It includes:
- feature selection
- lag depth selection
- horizon selection
- performance plots
- feature importance plots
- model export

### 4. MPC optimizer

The `MPC Optimizer & Market Participation` tab simulates a receding-horizon control process that:
- forecasts the next horizon
- optimizes reserve bids and allocations
- applies only the first move
- shifts forward and repeats

It supports:
- `FCR-N`
- `aFRR`
- `Combined`
- `Fast only`
- `Hybrid portfolio`

The MPC tracks:
- BESS energy state
- EV charging flexibility
- HVAC thermal flexibility
- capacity and activation revenue
- degradation and comfort penalties
- compliance metrics against Fingrid-style thresholds

### 5. Impact and comparison views

The dashboard also provides:
- stacked contribution plots
- reserve schedule plots
- heatmaps
- resource revenue comparison
- fast vs slow contribution comparison
- compliance indicators
- what-if scenario reruns
- sensitivity scans

## Repository Files

The project now has two entry points: the Streamlit dashboard and a FastAPI service that shares the same core optimizer logic.

- [app.py](./app.py)
  - the main Streamlit application
- [flexihome/core](./flexihome/core)
  - Streamlit-free data generation, forecasting, response-model, MPC logic, and base plugin interfaces
- [flexihome/core/base](./flexihome/core/base)
  - abstract forecaster, optimizer, pipeline-stage, orchestrator, and plugin-registry contracts
- [flexihome/pipeline](./flexihome/pipeline)
  - explicit market-data, feature, forecasting, optimization, and serialization pipeline stages with contracts and manifests
- [flexihome/plugins](./flexihome/plugins)
  - bundled production plugins, currently `xgboost_default` for forecasting and `pulp_default` for optimization
- [flexihome/api](./flexihome/api)
  - FastAPI schemas, endpoints, in-memory/TimescaleDB job stores, and result serialization
- [compose.yaml](./compose.yaml)
  - local Docker Compose TimescaleDB service for API persistence
- [requirements.txt](./requirements.txt)
  - base dependencies required to run the dashboard and API service
- [requirements-lstm.txt](./requirements-lstm.txt)
  - optional deep-learning dependencies for TensorFlow/LSTM-related extensions

## Plugin Architecture

FlexiHome now has a plugin layer for production step-by-step extension:

- `BaseForecaster` supports `fit`, `predict`, `get_feature_importance`, common evaluation metrics, and a shared lagged-frame helper.
- `BaseOptimizer` supports swappable MPC solvers through a common `solve(...)` contract.
- `BasePipelineStage` and `PipelineOrchestrator` provide a modular path for ingestion, feature engineering, forecasting, optimization, and result publication.
- `PluginRegistry` centralizes runtime discovery and instantiation.

Bundled defaults are registered at startup:

```python
from flexihome.core.base.registry import get_global_registry
from flexihome.plugins import register_default_plugins

registry = register_default_plugins(get_global_registry())
registry.list_forecasters()
registry.list_optimizers()
```

The FastAPI service exposes the registered plugins at:

- `GET /plugins`

`POST /forecast` accepts `forecaster_plugin`, and `POST /optimize` accepts both `forecaster_plugin` and `optimizer_plugin`. Defaults are `xgboost_default` and `pulp_default`, so existing API calls continue to work.

## Data Pipeline

The production-oriented pipeline is explicit and contract-driven:

```text
market_data_ingestion -> feature_engineering -> forecasting -> optimization -> results_serialization
```

The implementation lives under `flexihome/pipeline/`:

- `contracts.py` defines dataframe schemas, artifact contracts, run configuration, and manifest structure.
- `orchestrator.py` builds the default DAG and publishes DAG/contract metadata into the run context.
- `stages/market_data_ingestion.py` creates the portfolio dataframe and applies synthetic, mixed, or real market data.
- `stages/feature_engineering.py` validates and publishes model feature columns.
- `stages/forecasting_stage.py` trains registry-selected forecaster plugins for MPC targets.
- `stages/optimization_stage.py` runs the registry-selected optimizer plugin.
- `stages/results_serialization.py` writes CSV/JSON artifacts plus `manifest.json` with file hashes, dataframe hashes, schemas, and run config.

Run the whole pipeline locally:

```cmd
python scripts\run_pipeline.py --days 1 --n-homes 80 --horizon-hours 1 --dispatch-hours 1 --output-root runs
```

## How To Run

## Complete Windows CMD Runbook

These steps run the complete local system: the FastAPI optimizer service and the Streamlit dashboard.

### 1. Get the repository

If you do not have the repository yet:

```cmd
cd /d "%USERPROFILE%\Documents"
git clone https://github.com/arkm315787/Real_Time_Control_Energy-Flexibility.git
cd Real_Time_Control_Energy-Flexibility
git switch codex-fastapi-api-layer
```

If you already cloned it:

```cmd
cd /d "C:\Users\Kasutaja\OneDrive - Tallinna Tehnikaülikool\Documents\Playground\Real_Time_Control_Energy-Flexibility"
git fetch origin
git switch codex-fastapi-api-layer
git pull
```
### To activate an already existing virtual environment in Command Prompt (cmd), follow these steps:
Run this command from your project folder:
cmd
.venv\Scripts\activate
### 2. Create and activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

For normal dashboard and API usage:

```cmd
pip install -r requirements.txt
```

For development checks, including the API smoke test:

```cmd
pip install -r requirements-dev.txt
```

### 4. Run quick checks

```cmd
python -m py_compile app.py flexihome\core\engine.py flexihome\api\optimizer.py flexihome\api\services.py
python scripts\market_data_smoke.py
python scripts\plugin_smoke.py
python scripts\pipeline_smoke.py
python scripts\airflow_dag_smoke.py
python scripts\api_smoke.py
python scripts\timescale_store_smoke.py
```

The API smoke test should print health, forecast metrics, `result status: completed`, and nonzero result row counts.
The market-data smoke test runs without keys and reports `synthetic`; with keys configured it reports which ENTSO-E/Fingrid columns were loaded.
The TimescaleDB smoke test skips itself unless `FLEXIHOME_TIMESCALE_DSN` is configured.

### 5. Run FastAPI

Keep this command running in Terminal 1:

```cmd
python scripts\run_api.py --reload
```

Then open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

If Windows blocks port `8000`, the launcher prints the replacement port. Use that printed port in the browser and `curl` commands, for example `http://127.0.0.1:8010/docs`.

CMD health check:

```cmd
curl http://127.0.0.1:8000/health
```

CMD optimization check:

```cmd
curl -X POST "http://127.0.0.1:8000/optimize" -H "Content-Type: application/json" -d "{\"market_mode\":\"Combined\",\"resource_mode\":\"Hybrid portfolio\",\"horizon_hours\":1,\"dispatch_hours\":1,\"portfolio_config\":{\"days\":1,\"n_homes\":80,\"freq_minutes\":15,\"ev_pen\":0.2,\"bess_pen\":0.2,\"pv_pen\":0.3,\"hvac_pen\":0.5}}"
```

Copy the `results_url` from the response, then run:

```cmd
curl http://127.0.0.1:8000/results/YOUR_OPTIMIZATION_ID
```

For example, if the response says `"results_url":"/results/opt_20260427T020919_ebe5ac42"`, run:

```cmd
curl http://127.0.0.1:8000/results/opt_20260427T020919_ebe5ac42
```

### 6. Run Streamlit dashboard

Open a second CMD window, activate the same virtual environment, and run:

```cmd
cd /d "C:\Users\Kasutaja\OneDrive - Tallinna Tehnikaülikool\Documents\Playground\Real_Time_Control_Energy-Flexibility"
.venv\Scripts\activate
python -m streamlit run app.py
```

Then open:

- `http://localhost:8501`

### 7. Stop local servers

In each CMD window, press:

```text
Ctrl + C
```

## Option A: Base dashboard only

This short path is useful if you only want the dashboard working quickly and do not need optional LSTM/TensorFlow extras.

### Command Prompt

```cmd
cd /d "C:\Users\Kasutaja\OneDrive - Tallinna Tehnikaülikool\Documents\Playground\Real_Time_Control_Energy-Flexibility"
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open:

- `http://localhost:8501`

## Option B: FastAPI optimizer service

Run the API service in a separate terminal from the dashboard:

```cmd
python scripts\run_api.py --reload
```

Then open:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`

Useful verification calls:

```powershell
Invoke-RestMethod http://localhost:8000/health

$body = @{
  market_mode = "Combined"
  resource_mode = "Hybrid portfolio"
  horizon_hours = 1
  dispatch_hours = 1
  portfolio_config = @{
    days = 1
    n_homes = 500
    freq_minutes = 15
  }
} | ConvertTo-Json -Depth 5

$job = Invoke-RestMethod -Method Post -Uri http://localhost:8000/optimize -Body $body -ContentType "application/json"
$job
Invoke-RestMethod "http://localhost:8000$($job.results_url)"
```

The API uses a process-local in-memory job store unless TimescaleDB is configured. The store boundary is shared, so the same endpoints work with either backend.

## Optional TimescaleDB persistence

By default, the API uses the in-memory job store and `/health` reports:

```json
{"job_store":"in_memory","timescaledb":"not_configured"}
```

To persist optimization jobs in TimescaleDB, point the API at a PostgreSQL/TimescaleDB database before starting the service.

### Docker Compose local database

The repository includes `compose.yaml` for a local TimescaleDB container. Start it from the repository root:

```cmd
copy .env.example .env
docker compose up -d timescaledb
```

PowerShell equivalent:

```powershell
Copy-Item .env.example .env
docker compose up -d timescaledb
```

The default local DSN is:

```text
postgresql://flexihome:flexihome@127.0.0.1:5432/flexihome
```

Run the persistence smoke test:

```cmd
set FLEXIHOME_TIMESCALE_DSN=postgresql://flexihome:flexihome@127.0.0.1:5432/flexihome
set FLEXIHOME_TIMESCALE_SCHEMA=flexihome
python scripts\timescale_store_smoke.py
```

Then start the API in the same terminal:

```cmd
python scripts\run_api.py --reload
```

Stop the local database when you are done:

```cmd
docker compose down
```

### CMD

```cmd
set FLEXIHOME_TIMESCALE_DSN=postgresql://USER:PASSWORD@HOST:5432/flexihome
set FLEXIHOME_TIMESCALE_SCHEMA=flexihome
python scripts\timescale_store_smoke.py
python scripts\run_api.py --reload
```

### PowerShell

```powershell
$env:FLEXIHOME_TIMESCALE_DSN = "postgresql://USER:PASSWORD@HOST:5432/flexihome"
$env:FLEXIHOME_TIMESCALE_SCHEMA = "flexihome"
python scripts\timescale_store_smoke.py
python scripts\run_api.py --reload
```

When configured, the API creates:

- `flexihome.optimization_jobs`
  - latest status, request JSON, result JSON, error, and timestamps
- `flexihome.optimization_job_events`
  - Timescale hypertable for queued/running/completed/failed lifecycle events

If the DSN is configured but unavailable, the API falls back to the in-memory store and `/health` reports `status: degraded`. Set `FLEXIHOME_TIMESCALE_STRICT=1` if you want startup to fail instead. The default database connection timeout is 5 seconds; override it with `FLEXIHOME_TIMESCALE_CONNECT_TIMEOUT`.

## Production Airflow Orchestration

Airflow is now the production orchestration layer for FlexiHome. The Streamlit app remains the operator/research dashboard, the FastAPI service remains the machine-readable request surface, and Airflow coordinates scheduled data ingestion, model training, optimization, and outgoing result publication.

The repository includes these Airflow services in `compose.yaml`:

- `airflow-postgres` for Airflow metadata
- `airflow-init` for Airflow DB migration and admin user creation
- `airflow-webserver` for the Airflow UI
- `airflow-scheduler` for DAG scheduling

### DAG Inventory

Airflow DAG files live in `airflow/dags/`.

| DAG | Schedule | Role | Main artifacts |
| --- | --- | --- | --- |
| `flexihome_market_data_pipeline` | `@hourly` | Incoming data pipeline: source checks, market-data aggregation, feature validation | `runs/market_data_*` |
| `flexihome_forecasting_pipeline` | Manual/API trigger | Parameterized forecasting model training | `runs/forecast_*` |
| `flexihome_optimization_pipeline` | Manual/API trigger | Parameterized MPC optimization | `runs/optimization_*` |
| `flexihome_results_pipeline` | Manual/API trigger | Full contract pipeline and outgoing result manifest validation | `runs/pipeline_*` |
| `flexihome_local_pipeline` | Manual/API trigger | Compact backward-compatible local forecast + optimization DAG | `runs/forecast_*`, `runs/optimization_*` |

All production DAGs share:

- `retries=2`
- `retry_delay=5 minutes`
- `sla=1 hour` through task default args
- explicit `dagrun_timeout`
- `max_active_runs` limits to prevent overlapping heavy optimization jobs
- runtime parameters through `dag_run.conf`

### Step 1: Configure Secrets And Runtime Mode

Create a local `.env` if needed:

```cmd
copy .env.example .env
```

For synthetic offline operation, no API keys are required. For real market ingestion, add:

```text
ENTSOE_API_KEY=<your-entsoe-security-token>
FINGRID_API_KEY=<your-fingrid-open-data-key>
FLEXIHOME_MARKET_DATA_MODE=auto
FLEXIHOME_FINGRID_MIN_INTERVAL_SECONDS=2.1
FLEXIHOME_FINGRID_MAX_RETRIES=4
```

For persisted API/market observations, also set:

```text
FLEXIHOME_TIMESCALE_DSN=postgresql://flexihome:flexihome@timescaledb:5432/flexihome
FLEXIHOME_TIMESCALE_SCHEMA=flexihome
```

Use Docker service hostnames inside Airflow containers. For example, use `timescaledb`, not `127.0.0.1`, from Airflow.

### Step 2: Start Airflow

Build and start TimescaleDB plus Airflow:

```cmd
docker compose up -d --build timescaledb airflow-postgres airflow-init airflow-webserver airflow-scheduler
```

Open the Airflow dashboard:

- `http://127.0.0.1:8080`
- default user: `admin`
- default password: `admin`

Check container health and DAG import status:

```cmd
docker compose ps
docker compose logs airflow-scheduler --tail 100
docker compose exec airflow-scheduler airflow dags list
```

### Step 3: Run The Incoming Market-Data Pipeline

This DAG is scheduled hourly and can also be triggered manually:

```cmd
docker compose exec airflow-scheduler airflow dags trigger flexihome_market_data_pipeline --conf "{\"market_data_mode\":\"auto\",\"market_lookback_days\":30,\"days\":2,\"n_homes\":80,\"freq_minutes\":15}"
```

Task flow:

```text
fetch_entso_e_configuration -> fetch_fingrid_configuration -> aggregate_market_frame -> validate_market_artifacts
```

The aggregation task writes `runs/market_data_*` with:

- `market_data_frame.csv`
- `market_data_summary.json`
- `manifest.json`

The manifest includes source status, schema, artifact hashes, and a dataframe fingerprint.

### Step 4: Run A Parameterized Forecasting Pipeline

Trigger a forecast model run:

```cmd
docker compose exec airflow-scheduler airflow dags trigger flexihome_forecasting_pipeline --conf "{\"target\":\"fcrn_capacity_eur_per_mw_h\",\"forecaster_plugin\":\"xgboost_default\",\"lags\":4,\"horizon_steps\":1,\"days\":7,\"n_homes\":1500,\"market_data_mode\":\"synthetic\"}"
```

Important parameters:

- `target`: forecast target column
- `forecaster_plugin`: registered forecasting plugin, such as `xgboost_default`
- `lags`: lag depth
- `horizon_steps`: forecast horizon
- `market_data_mode`: `synthetic`, `auto`, or `real`

Artifacts are written to `runs/forecast_*`:

- `training_data.csv`
- `forecast_comparison.csv`
- `forecast_summary.json`
- `model.pkl`

### Step 5: Run A Parameterized Optimization Pipeline

Trigger an MPC optimization run:

```cmd
docker compose exec airflow-scheduler airflow dags trigger flexihome_optimization_pipeline --conf "{\"market_mode\":\"Combined\",\"resource_mode\":\"Hybrid portfolio\",\"forecaster_plugin\":\"xgboost_default\",\"optimizer_plugin\":\"pulp_default\",\"horizon_hours\":24,\"dispatch_hours\":24,\"days\":7,\"n_homes\":1500,\"market_data_mode\":\"synthetic\"}"
```

Important parameters:

- `market_mode`: `Combined`, `FCR-N`, or `aFRR`
- `resource_mode`: `Hybrid portfolio` or `Fast only`
- `forecaster_plugin`: registered forecaster used inside the MPC loop
- `optimizer_plugin`: registered optimizer, such as `pulp_default`
- `horizon_hours`: MPC planning horizon
- `dispatch_hours`: closed-loop simulation horizon

Artifacts are written to `runs/optimization_*`:

- `training_data.csv`
- `history.csv`
- `tracking_4s.csv`
- `first_schedule.csv`
- `optimization_summary.json`

### Step 6: Run The Outgoing Results Pipeline

The results DAG runs the full contract pipeline and validates that a publishable manifest exists:

```cmd
docker compose exec airflow-scheduler airflow dags trigger flexihome_results_pipeline --conf "{\"market_data_mode\":\"synthetic\",\"forecaster_plugin\":\"xgboost_default\",\"optimizer_plugin\":\"pulp_default\",\"market_mode\":\"Combined\",\"resource_mode\":\"Hybrid portfolio\",\"horizon_hours\":24,\"dispatch_hours\":24,\"days\":7,\"n_homes\":1500}"
```

Artifacts are written to `runs/pipeline_*`:

- `training_data.csv`
- `forecast_metrics.json`
- `forecast_comparison_<target>.csv`
- `optimization_summary.json`
- `history.csv`
- `tracking_4s.csv`
- `first_schedule.csv`
- `manifest.json`

The `manifest.json` is the production handoff document. It contains DAG metadata, stage contracts, run configuration, artifact paths, file hashes, dataframe schema snapshots, and data-version fingerprints.

### Step 7: Inspect And Operate DAGs

List tasks:

```cmd
docker compose exec airflow-scheduler airflow tasks list flexihome_market_data_pipeline
docker compose exec airflow-scheduler airflow tasks list flexihome_forecasting_pipeline
docker compose exec airflow-scheduler airflow tasks list flexihome_optimization_pipeline
docker compose exec airflow-scheduler airflow tasks list flexihome_results_pipeline
```

Watch task logs from the UI or CLI:

```cmd
docker compose exec airflow-scheduler airflow dags state flexihome_results_pipeline RUN_ID
```

Replace `RUN_ID` with the run id shown in the Airflow UI.

### Step 8: Connect FastAPI Health To Airflow

The FastAPI `/health` endpoint reports Airflow status when configured:

```cmd
set FLEXIHOME_AIRFLOW_URL=http://127.0.0.1:8080
python scripts\run_api.py --reload
curl http://127.0.0.1:8000/health
```

In Docker-to-Docker production-style deployment, point API containers at the Airflow webserver hostname instead of `127.0.0.1`.

### Step 9: Validate DAG Definitions Locally

The development environment does not need Airflow installed to statically validate DAG coverage:

```cmd
python scripts\airflow_dag_smoke.py
python scripts\run_market_data_ingestion.py --market-data-mode synthetic --days 1 --n-homes 80 --output-root runs
python scripts\validate_pipeline_manifest.py --root runs --prefix market_data --require-artifact market_data_frame
```

These checks confirm that DAG files exist, retry/SLA configuration is present, plugin parameters are exposed, and market-data manifests are valid.

### Production Hardening Notes

This Docker Compose Airflow setup is a local production-style orchestrator, not a hardened cloud deployment. For an operational VPP deployment, move Airflow to Kubernetes/Helm or managed Airflow, store credentials in a secrets manager, configure remote logs, add alerting on SLA misses and task failures, and use immutable application images rather than mounting the whole project directory.

## Optional Real Market Data

By default the dashboard and API still run with synthetic market data, so the project works offline. In the Streamlit sidebar, use **Market Data Source** to switch between `Synthetic` and `Real market APIs`. The dashboard accepts ENTSO-E and Fingrid keys as password fields at runtime; these values are not written to tracked files.

The integration currently overlays:
- ENTSO-E Finland day-ahead spot price as `spot_price_eur_per_mwh`
- Fingrid FCR-N hourly market price as `fcrn_capacity_eur_per_mw_h`
- Fingrid aFRR up/down capacity prices as `afrr_up_capacity_eur_per_mw_h` and `afrr_down_capacity_eur_per_mw_h`
- Fingrid aFRR up/down energy prices as `afrr_up_energy_eur_per_mwh` and `afrr_down_energy_eur_per_mwh`
- Fingrid aFRR activation-volume datasets normalized into `afrr_up_act_frac` and `afrr_down_act_frac`

Create a local `.env` from the example if you have not already:

```cmd
copy .env.example .env
```

Then add your keys to `.env`, or set them in the terminal before running the app. Do not commit `.env`.

CMD:

```cmd
set ENTSOE_API_KEY=<your-entsoe-security-token>
set FINGRID_API_KEY=<your-fingrid-open-data-key>
set FLEXIHOME_MARKET_DATA_MODE=auto
```

PowerShell:

```powershell
$env:ENTSOE_API_KEY = "<your-entsoe-security-token>"
$env:FINGRID_API_KEY = "<your-fingrid-open-data-key>"
$env:FLEXIHOME_MARKET_DATA_MODE = "auto"
```

Run the market-data smoke test:

```cmd
python scripts\market_data_smoke.py
```

`FLEXIHOME_MARKET_DATA_MODE=auto` is recommended for development: it uses real data that is available and keeps synthetic fallback for unavailable columns. Use `FLEXIHOME_MARKET_DATA_MODE=real` only when you want the simulator to fail if no external market data can be loaded.

When TimescaleDB is configured and market-data persistence is enabled, fetched raw market observations are inserted into `flexihome.market_data_observations` by default. Override the table with `FLEXIHOME_MARKET_DATA_TABLE`. The dashboard also shows:
- which source was used: synthetic, mixed, or real
- ENTSO-E and Fingrid connection status
- raw observations loaded per signal
- requested API lookback window used for forecasting
- TimescaleDB persistence status and inserted row count

Fingrid API calls use the current endpoint `https://data.fingrid.fi/api` with the API key in the `x-api-key` header. The old `api.fingrid.fi/v1` endpoint is not used.
The connector paces Fingrid calls with `FLEXIHOME_FINGRID_MIN_INTERVAL_SECONDS=2.1` by default and retries HTTP `429` responses. This follows Fingrid's public API throttling guidance and prevents one dashboard refresh from exhausting the per-key request cadence.

The default Fingrid dataset IDs are in `.env.example`. Override them only if Fingrid changes dataset numbering or if you want to experiment with a different reserve-market signal.

## Local Module Workflow

The dashboard should be treated as the visualization and operator-review surface. Forecasting and optimization can run as separate local modules from VS Code, Terminal, Docker, or CI so heavier processing does not block Streamlit.

```mermaid
flowchart LR
    A["ENTSO-E / Fingrid APIs"] --> B["Market-data ingestion"]
    C["Synthetic scenario generator"] --> D["Training frame"]
    B --> D
    D --> E["Forecasting module"]
    D --> F["Optimization module"]
    E --> G["Run artifacts under runs/"]
    F --> G
    G --> H["Streamlit dashboard visualization"]
    G --> I["FastAPI / TimescaleDB records"]
```

Standalone forecasting:

```cmd
python scripts\run_forecasting.py --target fcrn_capacity_eur_per_mw_h --market-data-mode synthetic --forecaster-plugin xgboost_default --days 7 --output-root runs
```

Standalone optimization:

```cmd
python scripts\run_optimization.py --market-mode Combined --resource-mode "Hybrid portfolio" --horizon-hours 24 --dispatch-hours 24 --market-data-mode synthetic --forecaster-plugin xgboost_default --optimizer-plugin pulp_default --output-root runs
```

Full contract pipeline:

```cmd
python scripts\run_pipeline.py --market-data-mode synthetic --forecaster-plugin xgboost_default --optimizer-plugin pulp_default --days 7 --n-homes 1500 --horizon-hours 24 --dispatch-hours 24 --output-root runs
```

For real market data, set `ENTSOE_API_KEY` and/or `FINGRID_API_KEY`, then use `--market-data-mode auto` or `--market-data-mode real`. Each run writes CSV/JSON/model artifacts under `runs\forecast_*` or `runs\optimization_*`; `runs/` is intentionally gitignored.
The full pipeline writes `runs\pipeline_*` with `manifest.json`, artifact hashes, dataframe schema snapshots, and data-version fingerprints.
Open the dashboard's **Advanced / Export** tab to inspect those external run artifacts without retraining or resolving the optimization inside Streamlit.

Recommended engineering ownership:
- Forecasting engineer: add or replace model trainers behind `scripts\run_forecasting.py` or a new module with the same artifact contract: `forecast_summary.json`, `forecast_comparison.csv`, and `model.pkl`.
- Optimization engineer: add solvers behind `scripts\run_optimization.py` or a new module with the same artifact contract: `optimization_summary.json`, `history.csv`, `tracking_4s.csv`, and `first_schedule.csv`.
- Dashboard/API engineer: consume artifacts and expose visual/API status without embedding heavy training or solver logic into the UI.
- Data engineer: maintain ENTSO-E/Fingrid ingestion, TimescaleDB schemas, throttling, and dataset IDs.

GitHub CI is configured in `.github/workflows/ci.yml` to compile core modules, run the API smoke test, and smoke-test the standalone forecasting and optimization runners.

## Option C: Use a short virtual environment path

This is useful on Windows if you want optional TensorFlow/LSTM dependencies and want to avoid long-path problems.

```cmd
py -3 -m venv C:\fhv
C:\fhv\Scripts\activate
cd /d "C:\Users\Kasutaja\OneDrive - Tallinna Tehnikaülikool\Documents\Playground\Real_Time_Control_Energy-Flexibility"
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

## Quick Local Checks

```cmd
pip install -r requirements-dev.txt
python -m py_compile app.py flexihome\core\engine.py flexihome\api\optimizer.py flexihome\api\services.py
python scripts\plugin_smoke.py
python scripts\pipeline_smoke.py
python scripts\airflow_dag_smoke.py
python scripts\api_smoke.py
python scripts\timescale_store_smoke.py
python scripts\run_api.py
```

## Optional LSTM dependencies

The dashboard runs without TensorFlow. If you later add or enable LSTM-specific workflow extensions, install:

```cmd
pip install -r requirements-lstm.txt
```

If Windows long-path errors appear during TensorFlow installation, use a short environment path such as `C:\fhv`, or enable Windows long paths.

## Dashboard Workflow

### Home

The home page introduces:
- Fingrid balancing-market context
- aggregation logic
- fast reserve headroom
- total flexibility
- homes needed to meet reserve thresholds

### Synthetic Data Explorer

Use this tab to:
- inspect baseline and flexible power trajectories
- compare available upward and downward flexibility
- view distributions of flexibility
- inspect heatmaps of reserve availability

### Response Models

Use this tab to learn the difference between:
- instantaneous or near-instantaneous response
- ramp-limited response
- thermal second-order behavior
- inverter HVAC versus conventional thermal behavior

### Forecasting Lab

Use this tab to:
- choose a target
- select predictors
- set lag depth
- train a forecasting model
- inspect error metrics and feature importance

### MPC Optimizer & Market Participation

Use this tab to:
- select market product
- choose resource strategy
- set horizon and dispatch duration
- adjust degradation, comfort, and departure penalties
- run the reserve dispatch simulation

The app reports:
- total revenue
- delivered energy
- requirement score
- comfort violations
- compliance metrics
- first-horizon schedule snapshot

### Simulation & Impact Visualizations

Use this tab to compare:
- baseline versus optimized net load
- resource contribution stacks
- revenue by resource
- fast versus slow flexibility value
- bid success heatmaps
- what-if scenario reruns

### Advanced / Export

Use this tab to:
- export summary bundles
- export the 4-second preview trace
- export an MPC formulation summary as PDF
- run a compact sensitivity scan

## Current Modeling Notes

The app is intentionally educational rather than production-grade market software.

Important characteristics:
- It uses synthetic data by default, with optional ENTSO-E and Fingrid real-market overlays through environment variables.
- The MPC is a receding-horizon supervisory controller.
- The default dispatch logic is simplified for teaching clarity.
- The high-resolution `aFRR` signal preview is synthetic; hourly/15-minute aFRR price and activation columns can come from Fingrid when configured.
- The dashboard emphasizes explainability and interactive learning over strict operational deployment fidelity.

## Fingrid-Oriented Assumptions Embedded In The App

The app includes simplified educational checks for:
- `FCR-N` minimum bid size
- `aFRR` minimum bid size
- response-speed heuristics
- delivery accuracy
- storage endurance
- baseline methodology presence

These checks are not a legal or operational certification tool. They are there to help users understand likely bottlenecks and tradeoffs.

## Important Windows Notes

### 1. Streamlit cache and stale UI

If styling or UI changes do not appear immediately:

```text
Ctrl + Shift + R
```

If needed, stop Streamlit and restart:

```cmd
Ctrl + C
python -m streamlit run app.py
```

### 2. TensorFlow long-path issues

If installing TensorFlow fails with deep include-path errors, use a short environment path:

```cmd
py -3 -m venv C:\fhv
C:\fhv\Scripts\activate
```

Then install dependencies from there.

## Troubleshooting

### `TypeError: got an unexpected keyword argument 'squared'`

Cause:
- older/newer `scikit-learn` signature mismatch in `mean_squared_error`

Status:
- fixed in this branch

### Dashboard visuals look stale

Try:
- hard refresh
- restart Streamlit
- confirm you are running the current branch files

### MPC feels slow

This is expected for large settings because the app performs:
- repeated forecasting
- repeated horizon optimization
- rolling closed-loop simulation

If you want faster learning runs, use:
- shorter dispatch horizon
- shorter MPC horizon
- coarser timestep

### Optional LSTM install fails

Cause:
- TensorFlow on Windows can hit long-path limits

Fix:
- use `requirements.txt` only for the base dashboard
- use a short virtual environment path for optional TensorFlow installs

### FastAPI fails with `[WinError 10013]`

Cause:
- Windows refused the selected port, often because port `8000` is reserved by the OS, a VPN, Hyper-V/WinNAT, security software, or another service.

Fix:

```cmd
python scripts\run_api.py --reload
```

The launcher will skip blocked ports and print the working URL. You can also request a specific alternate port:

```cmd
python scripts\run_api.py --port 8010 --reload
```

To see Windows excluded TCP port ranges:

```cmd
netsh interface ipv4 show excludedportrange protocol=tcp
```

## Suggested Run Configurations For Learning

### Fast overview

- timestep: `15 min`
- MPC horizon: `2 h`
- dispatch horizon: `4 h`

### Balanced exploration

- timestep: `10 min`
- MPC horizon: `3 h`
- dispatch horizon: `4-6 h`

### Fine-grained demo

- timestep: `5 min`
- MPC horizon: `1-2 h`
- dispatch horizon: `2 h`

## Branch Information

The `main` branch contains the Streamlit dashboard. The `codex-fastapi-api-layer` branch adds the FastAPI optimizer service and refactors the shared optimizer logic into the `flexihome` package.

## License / Usage

This repository branch is intended as a research and educational dashboard artifact. If you plan to use it operationally, you should:
- review the optimization assumptions
- validate the forecasting logic on real data
- replace synthetic market and activation data with real sources
- harden the telemetry, control, and compliance layers

## Summary

`FlexiHome` is a self-contained educational dashboard for exploring residential flexibility aggregation in Fingrid balancing markets. It combines synthetic data, forecasting, response models, and MPC-style optimization in a format that is highly visual, interactive, and suitable for learning and demonstration.
