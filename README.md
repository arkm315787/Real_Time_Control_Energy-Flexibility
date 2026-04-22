# FlexiHome: Aggregated Residential Flexibility for Fingrid Balancing Markets

`FlexiHome` is a Streamlit learning dashboard for exploring how aggregated residential resources can participate in Finnish balancing markets, with a strong focus on Fingrid `FCR-N` and `aFRR`.

The application combines:
- synthetic Finland-style household and weather data
- EV, BESS, PV, and HVAC flexibility models
- forecasting models for MPC inputs
- a receding-horizon MPC-style optimizer
- interactive educational visualizations for response speed, compliance, revenue, and what-if analysis

This branch contains the Streamlit dashboard implementation and its Python dependencies.

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

This branch currently focuses on the dashboard files:

- [app.py](./app.py)
  - the main Streamlit application
- [requirements.txt](./requirements.txt)
  - base dependencies required to run the dashboard
- [requirements-lstm.txt](./requirements-lstm.txt)
  - optional deep-learning dependencies for TensorFlow/LSTM-related extensions

## How To Run

## Option A: Base dashboard only

This is the recommended path if you want the dashboard working quickly and do not need optional LSTM/TensorFlow extras.

### Command Prompt

```cmd
cd /d "C:\Users\Kasutaja\OneDrive - Tallinna Tehnikaülikool\Documents\Playground"
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open:

- `http://localhost:8501`

## Option B: Use a short virtual environment path

This is useful on Windows if you want optional TensorFlow/LSTM dependencies and want to avoid long-path problems.

```cmd
py -3 -m venv C:\fhv
C:\fhv\Scripts\activate
cd /d "C:\Users\Kasutaja\OneDrive - Tallinna Tehnikaülikool\Documents\Playground"
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
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
- It uses synthetic data rather than external APIs.
- The MPC is a receding-horizon supervisory controller.
- The default dispatch logic is simplified for teaching clarity.
- The `aFRR` signal preview is synthetic.
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

This README was added to the branch:

- `codex/flexihome-dashboard`

That branch contains the Streamlit dashboard files prepared for GitHub publication.

## License / Usage

This repository branch is intended as a research and educational dashboard artifact. If you plan to use it operationally, you should:
- review the optimization assumptions
- validate the forecasting logic on real data
- replace synthetic market and activation data with real sources
- harden the telemetry, control, and compliance layers

## Summary

`FlexiHome` is a self-contained educational dashboard for exploring residential flexibility aggregation in Fingrid balancing markets. It combines synthetic data, forecasting, response models, and MPC-style optimization in a format that is highly visual, interactive, and suitable for learning and demonstration.
