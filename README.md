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
  - Streamlit-free data generation, forecasting, response-model, and MPC logic
- [flexihome/api](./flexihome/api)
  - FastAPI schemas, endpoints, in-memory/TimescaleDB job stores, and result serialization
- [compose.yaml](./compose.yaml)
  - local Docker Compose TimescaleDB service for API persistence
- [requirements.txt](./requirements.txt)
  - base dependencies required to run the dashboard and API service
- [requirements-lstm.txt](./requirements-lstm.txt)
  - optional deep-learning dependencies for TensorFlow/LSTM-related extensions

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

The default Fingrid dataset IDs are in `.env.example`. Override them only if Fingrid changes dataset numbering or if you want to experiment with a different reserve-market signal.

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
