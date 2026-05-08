# FlexiHome Production Application Guide

**Repository:** `Real_Time_Control_Energy-Flexibility`  
**Current working branch:** `codex-fastapi-api-layer`  
**Also aligned remote branch:** `feature/plugin-architecture`  
**Audience:** non-programmers, researchers, energy professionals, project managers, and engineers who need to explain the system clearly  
**Purpose:** explain the whole application from the user dashboard to data generation, real market data, forecasting, optimization, API service, pipelines, database, Airflow orchestration, Docker containers, GitHub workflow, and production operation.

---

## Table Of Contents

1. Executive Summary
2. What Problem FlexiHome Solves
3. The Big Picture In Plain Language
4. Main Application Components
5. End-To-End Workflow
6. Dashboard Walkthrough
7. Data: What It Is, Where It Comes From, And How It Moves
8. Real Market Data Integration
9. Forecasting Explained
10. Optimization Explained
11. The 4-Second Tracking Controller
12. Compliance And Fingrid-Oriented Checks
13. Plugin Architecture
14. Explicit Data Pipeline Architecture
15. FastAPI Service
16. Database And Persistence
17. Airflow Production Orchestration
18. Docker Containers
19. Local Launch Procedures
20. Production-Style Operating Procedure
21. GitHub, Branches, CI, And Collaboration
22. Repository Map
23. Smoke Tests And Validation
24. Common Questions And Explanations
25. Glossary
26. Appendix A: Command Cheat Sheet
27. Appendix B: Example API Calls
28. Appendix C: How To Explain This Project To Someone Else

---

## 1. Executive Summary

FlexiHome is an application for studying and operating aggregated residential energy flexibility. In simple words, it asks:

> If many homes have flexible devices such as electric vehicles, batteries, solar panels, and smart heating or cooling systems, can those homes act together like a small virtual power plant and help the electricity grid?

The application has grown from an educational dashboard into a production-style software stack. It now includes:

- A **Streamlit dashboard** for interactive exploration.
- A **shared modeling core** that creates household portfolios, market signals, forecasts, and optimization results.
- A **FastAPI service** for machine-readable API access.
- A **plugin architecture** so forecasters and optimizers can be swapped without rewriting the application.
- A **contract-based pipeline layer** for data ingestion, feature engineering, forecasting, optimization, and result serialization.
- **TimescaleDB/PostgreSQL persistence** for optimization jobs and optionally market observations.
- **Airflow DAGs** for production-style scheduled and manual orchestration.
- **Docker Compose** services for TimescaleDB and Airflow.
- **GitHub CI checks** that compile and smoke-test important parts of the application.

At a high level, FlexiHome produces or fetches data, forecasts future grid and market conditions, chooses how the flexible devices should behave, checks whether the result looks compliant with simplified Fingrid reserve-market rules, and presents the result visually.

---

## 2. What Problem FlexiHome Solves

Electricity grids need balance. At every moment, generation and consumption must be close. If demand rises suddenly or generation drops, grid operators need flexible resources that can respond.

Traditionally, this flexibility comes from large power plants, industrial loads, or grid-scale batteries. FlexiHome explores another idea:

> A large number of ordinary homes can be aggregated into a coordinated portfolio. Together, their devices can provide measurable grid support.

The residential resources in this application are:

| Resource | What It Means | How It Can Help |
| --- | --- | --- |
| EV | Electric vehicle charging | Can slow down, pause, or sometimes shift charging |
| BESS | Battery energy storage system | Can charge or discharge quickly |
| PV | Solar generation | Can be curtailed downward when needed |
| HVAC | Heating, ventilation, air conditioning | Can shift heating/cooling within comfort limits |

The market products modeled are Fingrid-oriented:

| Product | Plain Meaning | Key Idea |
| --- | --- | --- |
| FCR-N | Fast frequency containment reserve | Very fast local response around 50 Hz |
| aFRR | Automatic frequency restoration reserve | Centralized activation signal, slower than FCR-N but still controlled tightly |
| Combined | Uses both products | Portfolio can split resources across FCR-N and aFRR |

The project is not a certification tool. It is a prototype/research platform. It helps users understand what is possible, what constraints matter, what data is needed, and how a production architecture can be built step by step.

---

## 3. The Big Picture In Plain Language

Think of the system as a decision factory.

1. **Inputs arrive.**
   - Weather assumptions.
   - Household portfolio size.
   - EV, battery, solar, and HVAC penetration.
   - Market prices and activation signals, either synthetic or from real APIs.

2. **The application creates a virtual neighborhood.**
   - It calculates expected home load.
   - It calculates solar production.
   - It estimates how many EVs are connected.
   - It estimates battery state of charge.
   - It estimates heating/cooling flexibility.

3. **Forecasting predicts important future signals.**
   - Future baseline load.
   - Future PV availability.
   - Future reserve prices.
   - Future activation fractions.

4. **Optimization chooses reserve bids and resource dispatch.**
   - How much FCR-N to bid.
   - How much aFRR up/down to bid.
   - Which resource supplies which part.
   - How to avoid damaging batteries or violating comfort.

5. **A tracking controller simulates fast real-time response.**
   - The outer optimization may work at 15-minute intervals.
   - Inside each interval, a 4-second tracking controller follows reserve signals.

6. **Results are shown, saved, or returned through APIs.**
   - Dashboard charts.
   - API JSON.
   - CSV and JSON artifacts.
   - Pipeline manifests.
   - TimescaleDB records when configured.

The whole system can be run manually by a researcher, requested through the API, or orchestrated by Airflow as a production-style workflow.

---

## 4. Main Application Components

The application is built from several connected parts.

```mermaid
flowchart TB
    User["Human user"] --> Dashboard["Streamlit dashboard"]
    Operator["Operator or external system"] --> API["FastAPI service"]
    Scheduler["Airflow scheduler"] --> Pipelines["Production pipeline scripts"]

    Dashboard --> Core["Shared FlexiHome core"]
    API --> Core
    Pipelines --> Core

    Core --> DataGen["Portfolio and synthetic data generator"]
    Core --> MarketData["ENTSO-E / Fingrid market data connector"]
    Core --> Forecasting["Forecasting plugins"]
    Core --> Optimizer["Optimization plugins"]
    Core --> Tracking["4-second tracking simulation"]

    API --> Store["In-memory or TimescaleDB job store"]
    MarketData --> Timescale["TimescaleDB market observations"]
    Pipelines --> Artifacts["runs/ artifacts and manifests"]
    AirflowDB["Airflow metadata PostgreSQL"] --> Scheduler
```

### Component List

| Component | Location | Role |
| --- | --- | --- |
| Dashboard | `app.py` | Interactive user interface |
| Core engine | `flexihome/core/engine.py` | Data generation, forecasting helpers, MPC, response models |
| Market data connector | `flexihome/core/market_data.py` | ENTSO-E and Fingrid data fetching, pacing, fallback |
| API | `flexihome/api/optimizer.py` | HTTP endpoints for forecast, optimization, health, plugins |
| API schemas | `flexihome/api/schemas.py` | Input/output validation rules |
| Job store | `flexihome/api/store.py` | In-memory or TimescaleDB job persistence |
| Plugins | `flexihome/plugins/` | Swappable forecaster and optimizer implementations |
| Base interfaces | `flexihome/core/base/` | Contracts for forecasters, optimizers, pipeline stages |
| Pipelines | `flexihome/pipeline/` | Explicit data pipeline stages and manifests |
| Airflow DAGs | `airflow/dags/` | Production orchestration workflows |
| Docker Compose | `compose.yaml` | Local TimescaleDB and Airflow services |
| Scripts | `scripts/` | Smoke tests and command-line runners |
| GitHub CI | `.github/workflows/ci.yml` | Automated validation on push and pull request |

---

## 5. End-To-End Workflow

The most complete workflow looks like this:

```mermaid
flowchart LR
    A["Start app or Airflow DAG"] --> B["Choose scenario settings"]
    B --> C["Generate residential portfolio"]
    C --> D{"Use real market APIs?"}
    D -->|No| E["Use synthetic price and activation data"]
    D -->|Yes| F["Fetch ENTSO-E and Fingrid data"]
    F --> G["Normalize and align market signals"]
    E --> H["Training dataframe"]
    G --> H
    H --> I["Feature engineering"]
    I --> J["Train forecast models"]
    J --> K["Run MPC optimizer"]
    K --> L["Simulate 4-second tracking"]
    L --> M["Compliance and revenue summary"]
    M --> N{"Output channel"}
    N --> O["Dashboard charts"]
    N --> P["FastAPI JSON"]
    N --> Q["runs/ CSV and JSON artifacts"]
    N --> R["TimescaleDB records"]
```

A non-programmer can understand the sequence as:

1. Set up a virtual group of homes.
2. Decide whether market conditions are synthetic or real.
3. Build a table with one row per time step.
4. Teach a forecasting model using that table.
5. Ask the optimizer to choose the best reserve behavior.
6. Simulate whether the homes can actually follow the requested signal.
7. Show or save the result.

---

## 6. Dashboard Walkthrough

The dashboard is the easiest place to see the application working. It is built with Streamlit and runs from `app.py`.

### 6.1 How To Launch The Dashboard

From the repository root:

```cmd
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

If another Streamlit process is already using that port, Streamlit may choose another port and print it in the terminal.

### 6.2 Sidebar Controls

The sidebar controls define the scenario. When a user changes these controls, the application regenerates or refetches the underlying dataset.

#### Dashboard Theme

| Control | Choices | Meaning |
| --- | --- | --- |
| Dashboard theme | Dark, Light | Changes chart and dashboard styling |

#### Portfolio Setup

| Control | Range / Choices | Default | Plain Explanation |
| --- | --- | --- | --- |
| Simulation start | Date | 2026-01-15 | First timestamp in the simulation |
| Simulation length | 1 to 365 days | 7 | How many days of data to simulate |
| Simulation timestep | 1 to 15 minutes | 15 | Time resolution of the main data table |
| Aggregated homes | 500 to 5000 | 1500 | Number of homes in the virtual portfolio |
| EV penetration | 0.0 to 1.0 | 0.30 | Share of homes with EV flexibility |
| BESS penetration | 0.0 to 1.0 | 0.22 | Share of homes with batteries |
| PV penetration | 0.0 to 1.0 | 0.45 | Share of homes with solar panels |
| Smart HVAC penetration | 0.0 to 1.0 | 0.82 | Share of homes with controllable HVAC |
| HVAC technology | Conventional or inverter | Inverter | Response type for HVAC |
| HVAC electrical response | 2 to 180 seconds | Depends on mode | How quickly HVAC electric power can react |

What the penetration controls mean:

- `0.30 EV penetration` means about 30 percent of homes have EV charging flexibility.
- `0.22 BESS penetration` means about 22 percent of homes have a battery system.
- These are not individual real houses. They are scenario assumptions.

#### Synthetic Finland Context

| Control | Range | Default | Plain Explanation |
| --- | --- | --- | --- |
| Cloudiness / variability | 0.1 to 1.0 | 0.55 | More clouds reduce and randomize solar production |
| Temperature anomaly | -8 C to +8 C | 0 C | Makes the simulated weather colder or warmer |
| Solar resource multiplier | 0.5 to 1.5 | 1.0 | Scales available sunlight |
| Market scarcity / price stress | 0.6 to 1.8 | 1.0 | Higher value creates more stressed, valuable market conditions |
| HVAC setpoint | 19 C to 23 C | 21 C | Desired indoor temperature |
| Comfort band | 0.5 C to 2.5 C | 1 C | Allowed comfort tolerance around the setpoint |
| Random seed | Number | 42 | Makes random scenario generation repeatable |

Random seed is useful for experiments. If the same settings and same seed are used, the synthetic data should be repeatable.

#### Market Data Source

| Control | Choices / Range | Meaning |
| --- | --- | --- |
| Training and market signals | Synthetic or Real market APIs | Choose offline generated signals or real market data |
| Real market lookback days | 1 to 120 | How much historical API data to fetch |
| ENTSO-E security token | Password field | Optional real day-ahead electricity price source |
| Fingrid Open Data API key | Password field | Optional real reserve-market data source |
| Persist fetched market data to TimescaleDB | On/off | Saves raw fetched market observations if database is configured |

If real market APIs are selected but no key is entered, the sidebar warns the user. If real data fails, the dashboard can fall back to synthetic data.

### 6.3 Dashboard Tabs

The dashboard has seven main tabs:

1. Home
2. Training Data Explorer
3. Response Models
4. Forecasting Lab
5. MPC Optimizer & Market Participation
6. Simulation & Impact Visualizations
7. Advanced / Export

### 6.4 Home Tab

The Home tab gives the high-level project story and headline metrics.

It shows:

- Data source: synthetic, mixed, or real.
- Number of model rows.
- Number of real observations loaded.
- Number of TimescaleDB rows persisted.
- Fast reserve headroom.
- Total upward flexibility.
- Homes needed for 1 MW aFRR.
- Homes needed for fast 1 MW.
- PV fleet size.
- Accessible storage energy.

It also shows a Sankey-style flow explaining how portfolio resources contribute to flexibility.

Plain explanation:

> The Home tab answers: "How big is my virtual portfolio, and roughly how much useful flexibility can it provide?"

### 6.5 Training Data Explorer Tab

This tab helps users inspect the actual data table used by forecasting and optimization.

It includes:

- A download button for the active training data as CSV.
- Market-source diagnostics.
- Market data completeness chart.
- Market price signals.
- Activation signals.
- Baseline and resource traces.
- Flexibility headroom plots.
- Distribution of flexible capacity.
- Heatmaps for fast response and HVAC flexibility.

Important data columns shown or used include:

| Column | Meaning |
| --- | --- |
| `base_load_kw` | Normal household electricity load |
| `pv_available_kw` | Solar power available |
| `ev_baseline_kw` | Expected EV charging without optimization |
| `hvac_baseline_kw` | Expected HVAC power without optimization |
| `net_load_baseline_kw` | Net household load after solar and baseline device behavior |
| `flex_up_kw` | Ability to reduce net load or inject more power |
| `flex_down_kw` | Ability to increase net load or absorb power |
| `fast_sym_kw` | Fast symmetric flexibility useful for FCR-N |

Plain explanation:

> This tab is like looking under the hood. It shows the table the model sees.

### 6.6 Response Models Tab

This tab explains how quickly different resources respond.

Controls:

| Control | Range | Meaning |
| --- | --- | --- |
| BESS time constant | 1 to 5 seconds | Battery response speed |
| EV time constant | 1 to 8 seconds | EV charger response speed |
| HVAC fast thermal mode tau1 | 5 to 15 minutes | Faster thermal response component |
| HVAC slow thermal mode tau2 | 30 to 60 minutes | Slower thermal response component |

Charts:

- Step response plot.
- Animated reserve activation playback.
- Bode magnitude plot.
- Bode phase plot.

Plain explanation:

> Batteries and EV chargers can change power quickly. HVAC adds useful volume, but comfort and thermal inertia mean it is slower.

### 6.7 Forecasting Lab Tab

This tab lets the user train a forecast model.

Controls:

| Control | Meaning |
| --- | --- |
| Target | The variable to predict |
| Predictors | Input variables used to make the prediction |
| Forecasting model | Plugin selected from the registry |
| Lag depth | How many previous time steps are used |
| Forecast horizon | How far ahead to predict |

Supported targets include:

- `net_load_baseline_kw`
- `pv_available_kw`
- `fcrn_capacity_eur_per_mw_h`
- `afrr_up_capacity_eur_per_mw_h`
- `afrr_down_capacity_eur_per_mw_h`
- `afrr_up_act_frac`
- `afrr_down_act_frac`
- `fcr_signed_act`

The tab shows:

- Selected target history.
- MAE and RMSE forecast metrics.
- Actual versus predicted chart.
- Feature importance.
- Model download button.
- Button to promote a trained one-step model into the MPC loop.

Plain explanation:

> Forecasting is the part where the application learns patterns from historical or generated data so it can make a reasonable guess about what happens next.

### 6.8 MPC Optimizer & Market Participation Tab

This is the main decision-making tab.

Controls:

| Control | Choices / Range | Meaning |
| --- | --- | --- |
| Market product | Combined, FCR-N, aFRR | Which reserve market product to simulate |
| Resource strategy | Hybrid portfolio, Fast only | Whether to use slow resources like HVAC/PV |
| MPC horizon | 4 to 48 hours | How far ahead the optimizer plans |
| Dispatch simulation horizon | 6 to 72 hours, limited by data length | How long the closed-loop simulation runs |
| Degradation weight | 1 to 60 | Penalty for using batteries/EVs too aggressively |
| Comfort violation weight | 10 to 300 | Penalty for violating indoor comfort |
| EV departure shortfall weight | 10 to 300 | Penalty for undercharging EVs |
| MPC forecaster | Registered forecaster plugin | Model used inside MPC |
| Optimizer | Registered optimizer plugin | Solver used for dispatch decisions |

Button:

- `Run MPC dispatch`

After running, it shows:

- Total revenue.
- Delivered up energy.
- Requirement score.
- Comfort violations.
- Optimizer status.
- Number of MPC intervals.
- Market input source.
- Mean net revenue.
- Reserve schedule chart.
- Revenue response to market prices.
- Compliance snapshot.
- First MPC horizon schedule table.
- Rule pass/fail table.

Plain explanation:

> The optimizer chooses how the virtual home fleet should participate in reserve markets while balancing revenue, battery wear, comfort, EV charging requirements, and grid rules.

### 6.9 Simulation & Impact Visualizations Tab

This tab shows the effects of the optimization.

It includes:

- Baseline versus optimized load.
- Frequency view.
- Resource contribution stack.
- Revenue by resource.
- Fast versus slow flexibility comparison.
- Delivered up/down energy.
- Bid and success heatmaps.
- 4-second inner-loop tracking plot.
- What-if scenario playground.

What-if controls:

| Control | Meaning |
| --- | --- |
| Price multiplier | Makes prices lower or higher |
| Cold snap impact | Makes temperature colder |
| Thermal response multiplier | Changes HVAC flexibility response |

Plain explanation:

> This tab answers: "What changed because of optimization, and what happens if the situation changes?"

### 6.10 Advanced / Export Tab

This tab supports exporting and inspecting results.

It includes:

- Download simulation summary JSON.
- Download 4-second tracking CSV.
- Export MPC formulation summary as PDF.
- Inspect external module artifacts under `runs/`.
- Run sensitivity analysis.
- View embedded market rule references.

Sensitivity analysis scans:

- BESS penetration.
- Market price multiplier.
- Revenue and requirement score changes.

Plain explanation:

> This tab is for saving results, comparing offline runs, and checking how sensitive conclusions are to key assumptions.

---

## 7. Data: What It Is, Where It Comes From, And How It Moves

The main data object is a time-indexed table. Each row represents one time step, for example every 15 minutes.

If the simulation length is 7 days and the timestep is 15 minutes:

- 24 hours/day x 4 rows/hour x 7 days = 672 rows.

Each row contains weather, load, device flexibility, market price, activation, and state information.

### 7.1 Synthetic Data

Synthetic data is generated locally by `generate_synthetic_portfolio()` in `flexihome/core/engine.py`.

The generator creates:

- Weather:
  - Outdoor temperature.
  - Solar irradiance.
  - Cloud variability.
- Household load:
  - Morning peak.
  - Evening peak.
  - Weekend effect.
  - Temperature sensitivity.
- Solar:
  - PV capacity based on PV penetration.
  - Solar production based on irradiance.
- EV:
  - Number of EVs.
  - Connected fraction.
  - Baseline charging.
  - Up/down flexibility.
  - State-of-charge limits.
- BESS:
  - Battery count.
  - Power capacity.
  - Energy capacity.
  - State of charge.
  - Up/down capability.
- HVAC:
  - Baseline heating/cooling.
  - Up/down flexibility.
  - Indoor temperature reference.
  - Comfort constraints.
- Market:
  - Synthetic FCR-N price.
  - Synthetic aFRR up/down capacity prices.
  - Synthetic aFRR up/down energy prices.
  - Synthetic activation fractions.

### 7.2 Real Market Data

When real data is enabled, synthetic market signals can be replaced or overlaid by:

- ENTSO-E Finland day-ahead spot prices.
- Fingrid FCR-N capacity price.
- Fingrid aFRR up/down capacity prices.
- Fingrid aFRR up/down energy prices.
- Fingrid aFRR activation volume data normalized into activation fractions.

The portfolio itself still remains scenario-based. Real market data affects market prices and activation signals.

### 7.3 Important Data Columns

| Column | Category | Meaning |
| --- | --- | --- |
| `temp_out_c` | Weather | Outdoor temperature |
| `irradiance_wm2` | Weather | Solar irradiance |
| `base_load_kw` | Load | Normal household power demand |
| `pv_available_kw` | PV | Solar production available |
| `pv_down_kw` | PV flexibility | Curtailable solar amount |
| `ev_connected_frac` | EV | Share of EVs connected |
| `ev_baseline_kw` | EV | Expected charging load |
| `ev_up_kw` | EV flexibility | Ability to reduce charging or provide upward response |
| `ev_down_kw` | EV flexibility | Ability to increase charging |
| `ev_soc_ref_mwh` | EV state | EV energy state reference |
| `bess_up_kw` | Battery flexibility | Battery discharge capability |
| `bess_down_kw` | Battery flexibility | Battery charge capability |
| `bess_soc_ref_mwh` | Battery state | Battery state of charge |
| `hvac_baseline_kw` | HVAC | Expected HVAC power |
| `hvac_up_kw` | HVAC flexibility | Ability to reduce HVAC load |
| `hvac_down_kw` | HVAC flexibility | Ability to increase HVAC load |
| `indoor_temp_c_ref` | Comfort | Indoor temperature reference |
| `fast_sym_kw` | Reserve | Fast symmetric reserve capability |
| `flex_up_kw` | Reserve | Total upward flexibility |
| `flex_down_kw` | Reserve | Total downward flexibility |
| `net_load_baseline_kw` | System | Net load before optimization |
| `dt_h` | Time | Timestep in hours |

### 7.4 Data Flow

```mermaid
flowchart TD
    Settings["Sidebar or API request settings"] --> Generator["Synthetic portfolio generator"]
    Generator --> Portfolio["Portfolio dataframe"]
    RealChoice{"Real market APIs enabled?"}
    Settings --> RealChoice
    RealChoice -->|No| SyntheticMarket["Synthetic prices and activation"]
    RealChoice -->|Yes| MarketAPIs["ENTSO-E and Fingrid APIs"]
    MarketAPIs --> Overlay["Normalize, align, and overlay signals"]
    SyntheticMarket --> Portfolio
    Overlay --> Portfolio
    Portfolio --> ForecastData["Feature dataset"]
    ForecastData --> ForecastModels["Forecast models"]
    Portfolio --> OptimizerInputs["MPC optimizer inputs"]
    ForecastModels --> OptimizerInputs
    OptimizerInputs --> Results["Optimization and tracking results"]
```

---

## 8. Real Market Data Integration

Real market data is handled in `flexihome/core/market_data.py`.

### 8.1 ENTSO-E

ENTSO-E is used for Finland day-ahead spot prices.

Configuration:

```text
ENTSOE_API_KEY=<your-token>
```

The application also recognizes:

```text
ENTSOE_SECURITY_TOKEN=<your-token>
```

### 8.2 Fingrid

Fingrid Open Data is used for reserve-market signals.

Configuration:

```text
FINGRID_API_KEY=<your-key>
```

The application also recognizes:

```text
FINGRID_OPENDATA_API_KEY=<your-key>
```

Default dataset IDs are stored in `.env.example` and read through environment variables. They include FCR-N price, aFRR up/down prices, aFRR energy prices, and aFRR activation datasets.

### 8.3 API Pacing And Reliability

The Fingrid connector paces calls using:

```text
FLEXIHOME_FINGRID_MIN_INTERVAL_SECONDS=2.1
FLEXIHOME_FINGRID_MAX_RETRIES=4
```

This prevents the dashboard or pipeline from making too many requests too quickly.

### 8.4 Synthetic, Real, And Auto Modes

| Mode | Meaning |
| --- | --- |
| `synthetic` | Use generated signals only |
| `real` | Require real API data; fail if unavailable |
| `auto` | Use real data where available and synthetic fallback where needed |

For production exploration, `auto` is useful. For strict validation, `real` is useful because it fails when real data cannot be loaded.

---

## 9. Forecasting Explained

Forecasting means predicting a future value from past and current data.

In FlexiHome, forecasting is used because the optimizer needs to plan ahead. It cannot only know the present. It needs estimates of future:

- Net baseline load.
- PV availability.
- FCR-N price.
- aFRR up/down prices.
- Activation fractions.

### 9.1 How Forecast Training Works

The model training process:

1. Choose a target, such as `net_load_baseline_kw`.
2. Choose predictors, such as temperature, irradiance, load, and market prices.
3. Add lagged values. A lag is a previous value, for example "what was the load 15 minutes ago?"
4. Split the data into training and validation sections.
5. Fit the model.
6. Predict the validation section.
7. Measure error with MAE and RMSE.

### 9.2 MAE And RMSE In Plain Language

| Metric | Meaning |
| --- | --- |
| MAE | Average absolute error. If MAE is 10 kW, predictions are off by about 10 kW on average. |
| RMSE | Error measure that punishes large mistakes more strongly. |
| Bias | Whether predictions are usually too high or too low. |
| MAPE | Percent error where meaningful. |

### 9.3 Current Production Forecaster

The current bundled forecaster plugin is:

```text
xgboost_default
```

It uses XGBoost, a machine-learning model that builds many decision trees and combines them.

Plain explanation:

> XGBoost looks for patterns such as "when it is cold and evening, load is usually higher" or "when solar irradiance is high, PV output is higher."

### 9.4 Plugin-Based Forecasting

Forecasting is now plugin-based. This means another team member can add a new model without rewriting the dashboard, API, or pipeline.

Possible future plugins:

- LSTM neural network.
- Prophet time-series model.
- ARIMA model.
- LightGBM model.
- A simple rule-based baseline.

The dashboard and API select the plugin by name.

---

## 10. Optimization Explained

Optimization is the decision-making part.

The optimizer asks:

> Given the forecast, device limits, market prices, comfort limits, and battery/EV constraints, what reserve bids and resource allocations should we choose?

### 10.1 What The Optimizer Decides

For every interval in the planning horizon, it decides:

- BESS contribution to FCR-N.
- EV contribution to FCR-N.
- HVAC contribution to FCR-N.
- BESS aFRR up/down.
- EV aFRR up/down.
- HVAC aFRR up/down.
- PV downward curtailment.
- Battery state-of-charge trajectory.
- EV energy deviation.
- Indoor temperature deviation.

### 10.2 Objective Function In Plain Language

The optimizer tries to maximize value:

```text
value = capacity revenue + activation revenue - degradation cost - comfort penalty - EV shortfall penalty
```

In words:

- It likes earning money from reserve capacity.
- It likes earning money from activation.
- It dislikes wearing out batteries and EVs.
- It dislikes making occupants uncomfortable.
- It dislikes leaving EVs undercharged.

### 10.3 Current Production Optimizer

The current bundled optimizer plugin is:

```text
pulp_default
```

It uses PuLP/CBC. PuLP is a Python optimization modeling library, and CBC is an open-source solver.

If the formal solver fails, the application has a heuristic fallback. That means the system can still produce a reasonable schedule instead of completely stopping.

### 10.4 Receding-Horizon MPC

MPC means Model Predictive Control.

FlexiHome uses a receding-horizon pattern:

1. Forecast the next horizon, for example 24 hours.
2. Optimize the schedule for that horizon.
3. Apply only the first interval.
4. Move forward one interval.
5. Forecast again.
6. Optimize again.

This is similar to driving a car with navigation:

- You plan the route ahead.
- You only drive the next small part.
- Then you update the plan as conditions change.

```mermaid
sequenceDiagram
    participant Data as Current data
    participant Forecast as Forecaster
    participant MPC as Optimizer
    participant Track as 4-second tracker
    participant State as Updated states

    Data->>Forecast: Predict next horizon
    Forecast->>MPC: Forecasted load, PV, prices, activation
    MPC->>MPC: Choose bids and resource dispatch
    MPC->>Track: Apply first control interval
    Track->>State: Update BESS, EV, HVAC states
    State->>Data: Roll forward and repeat
```

---

## 11. The 4-Second Tracking Controller

The outer MPC typically works at the dashboard timestep, often 15 minutes. But aFRR signals can update every 4 seconds.

FlexiHome therefore simulates an inner controller:

- It receives the planned reserve allocation from the optimizer.
- It receives a 4-second activation signal.
- It calculates requested up/down response.
- It simulates actual resource response.
- It updates states such as battery energy, EV state, and indoor temperature.

This gives a more realistic view of whether a schedule can be followed quickly enough.

### 11.1 Why This Matters

A schedule can look good at 15-minute resolution but fail at fast control resolution. The 4-second tracker tests the faster behavior.

### 11.2 Tracked Outputs

The tracker produces:

- Requested up/down kW.
- Delivered up/down kW.
- BESS, EV, HVAC, and PV contributions.
- Frequency estimate.
- Battery state of charge.
- EV state.
- Indoor temperature.
- Optimized net load.

These are exported in `tracking_4s.csv` and shown in the dashboard.

---

## 12. Compliance And Fingrid-Oriented Checks

The application includes simplified checks inspired by Fingrid reserve requirements.

Important embedded checks:

| Check | Meaning |
| --- | --- |
| FCR-N minimum bid | Average FCR bid should meet the threshold |
| aFRR minimum bid | Average aFRR bid should meet the threshold |
| FCR-N dynamic response | Simulated local droop response should reach roughly 63% in 60 seconds and 95% in 180 seconds |
| aFRR response speed | Resources should reach full activation within the 5-minute audit window |
| Accuracy | Delivered response should be within a reasonable request/delivery band |
| Storage endurance | Storage should have enough energy behind the bid |
| Baseline method available | Required baseline columns exist |

Current simplified constants:

| Product | Rule Used In App |
| --- | --- |
| FCR-N | 0.1 MW minimum bid, 0.1 MW bid granularity, local droop activation, 63%/60 s and 95%/180 s dynamic-response screen |
| aFRR | 1 MW minimum bid, 1 MW bid granularity, centralized 4-second activation tracking, 5-minute full-activation audit |
| aFRR accuracy | 90 to 110 percent delivery ratio |
| Storage endurance | 1 hour per direction heuristic |

Important note:

> These checks are educational and engineering-oriented. They are not official market certification.

---

## 13. Plugin Architecture

The plugin architecture lets the application use different forecasting models or optimization solvers without changing the rest of the system.

```mermaid
flowchart TB
    Registry["Plugin registry"] --> Forecasters["Forecaster plugins"]
    Registry --> Optimizers["Optimizer plugins"]
    Registry --> Stages["Pipeline stage plugins"]

    Forecasters --> XGB["xgboost_default"]
    Optimizers --> PuLP["pulp_default"]
    Stages --> Ingestion["market_data_ingestion"]
    Stages --> Features["feature_engineering"]
    Stages --> ForecastStage["forecasting"]
    Stages --> OptStage["optimization"]
    Stages --> Results["results_serialization"]

    Dashboard["Dashboard selector"] --> Registry
    API["API request fields"] --> Registry
    Airflow["Airflow DAG conf"] --> Registry
```

### 13.1 Base Interfaces

Location:

```text
flexihome/core/base/
```

Main interfaces:

| Interface | Purpose |
| --- | --- |
| `BaseForecaster` | Standard contract for models that learn and predict |
| `BaseOptimizer` | Standard contract for solvers that choose dispatch decisions |
| `BasePipelineStage` | Standard contract for pipeline steps |
| `PluginRegistry` | Central registry of available plugins |

### 13.2 Current Plugins

| Plugin | Type | Purpose |
| --- | --- | --- |
| `xgboost_default` | Forecaster | Production XGBoost forecasting implementation |
| `xgboost_v1` | Forecaster alias | Versioned alias |
| `pulp_default` | Optimizer | Production PuLP/CBC optimizer |
| `pulp_v1` | Optimizer alias | Versioned alias |

### 13.3 Why This Helps Production

Before plugins:

- Adding a new model meant editing core code.
- Replacing the solver meant touching shared engine functions.
- Different team members could easily conflict.

After plugins:

- New models can be added as separate files.
- New solvers can be registered by name.
- Dashboard, API, scripts, and Airflow can choose plugins through parameters.

---

## 14. Explicit Data Pipeline Architecture

The pipeline layer is under:

```text
flexihome/pipeline/
```

It formalizes the production dataflow.

```mermaid
flowchart LR
    A["market_data_ingestion"] --> B["feature_engineering"]
    B --> C["forecasting"]
    C --> D["optimization"]
    D --> E["results_serialization"]
```

### 14.1 Stage 1: Market Data Ingestion

File:

```text
flexihome/pipeline/stages/market_data_ingestion.py
```

Responsibilities:

- Generate the portfolio.
- Fetch or synthesize market signals.
- Produce the main dataframe.
- Store market-data status.
- Store 4-second preview signals.

Outputs:

- `portfolio_bundle`
- `preview_4s`
- `portfolio_summary`
- `market_data_status`

### 14.2 Stage 2: Feature Engineering

File:

```text
flexihome/pipeline/stages/feature_engineering.py
```

Responsibilities:

- Validate required columns.
- Identify numeric columns.
- Select model feature columns.
- Publish feature contract metadata.

Outputs:

- `feature_columns`
- `numeric_columns`
- `feature_contract`

### 14.3 Stage 3: Forecasting

File:

```text
flexihome/pipeline/stages/forecasting_stage.py
```

Responsibilities:

- Get the selected forecaster plugin.
- Train models for the required MPC targets.
- Store metrics and comparisons.

Outputs:

- `forecast_models`
- `forecast_metrics`
- `forecast_comparisons`

### 14.4 Stage 4: Optimization

File:

```text
flexihome/pipeline/stages/optimization_stage.py
```

Responsibilities:

- Get the selected optimizer plugin.
- Build fleet metadata.
- Run the MPC controller.
- Store the full result.

Outputs:

- `optimization_result`
- `fleet_metadata`

### 14.5 Stage 5: Results Serialization

File:

```text
flexihome/pipeline/stages/results_serialization.py
```

Responsibilities:

- Write CSV and JSON outputs.
- Write forecast metrics.
- Write optimization summary.
- Write tracking data.
- Write manifest.

Outputs:

- `run_dir`
- `manifest`
- `artifact_records`

### 14.6 Manifest

The manifest is a production handoff document. It records:

- Run ID.
- Pipeline name.
- Configuration.
- DAG metadata.
- Stage contracts.
- Artifact list.
- File hashes.
- Dataframe hashes.
- Schema snapshots.
- Start and finish timestamps.

This helps answer:

- What data was used?
- What code path produced it?
- Which files belong to the run?
- Did the data change?
- What plugin choices were used?

---

## 15. FastAPI Service

The API turns the application into a machine-readable service.

Launch:

```cmd
python scripts\run_api.py --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

### 15.1 API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Service information |
| `/optimize` | POST | Submit an optimization job |
| `/results/{optimization_id}` | GET | Retrieve full job results |
| `/optimizations` | GET | List recent jobs |
| `/forecast` | POST | Train and return a forecast response |
| `/plugins` | GET | List registered forecasters, optimizers, and pipeline stages |
| `/health` | GET | Report API, store, TimescaleDB, and Airflow status |
| `/metrics` | GET | Prometheus-style job metrics |

### 15.2 API Request Validation

Validation lives in:

```text
flexihome/api/schemas.py
```

Examples:

- `days` must be between 1 and 365.
- `n_homes` must be between 1 and 100000.
- Penetrations must be between 0 and 1.
- `horizon_hours` must be between 1 and 72.
- `dispatch_hours` must be between 1 and 168.
- `market_mode` must be `FCR-N`, `aFRR`, or `Combined`.
- `resource_mode` must be `Fast only` or `Hybrid portfolio`.

### 15.3 API Job Lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Store
    participant Worker
    participant Core

    Client->>API: POST /optimize
    API->>Store: create queued job
    API->>Worker: schedule background task
    API-->>Client: return optimization_id and results_url
    Worker->>Store: mark running
    Worker->>Core: generate data, train forecasts, optimize
    Core-->>Worker: result
    Worker->>Store: mark completed
    Client->>API: GET /results/{id}
    API->>Store: fetch job
    API-->>Client: full result JSON
```

---

## 16. Database And Persistence

The application can work without a database, but production needs persistence.

### 16.1 In-Memory Store

Default behavior:

- Optimization jobs are stored in memory.
- Data disappears when the API process stops.
- Good for development and smoke tests.

### 16.2 TimescaleDB Store

If `FLEXIHOME_TIMESCALE_DSN` is configured, the API attempts to use TimescaleDB/PostgreSQL.

Tables:

| Table | Purpose |
| --- | --- |
| `flexihome.optimization_jobs` | Latest job status, request, result, error, timestamps |
| `flexihome.optimization_job_events` | Time-series event log for queued/running/completed/failed lifecycle |
| `flexihome.market_data_observations` | Optional raw market observations |

### 16.3 Why TimescaleDB

TimescaleDB is PostgreSQL with time-series features. It is suitable because:

- Market data is time-indexed.
- Job lifecycle events are time-indexed.
- Future telemetry will likely be time-indexed.

### 16.4 Health Behavior

If a TimescaleDB DSN is configured but unavailable:

- The API can fall back to in-memory store.
- `/health` reports degraded status.
- Strict failure can be enabled with `FLEXIHOME_TIMESCALE_STRICT=1`.

---

## 17. Airflow Production Orchestration

Airflow coordinates scheduled and repeatable production workflows.

### 17.1 DAG Inventory

| DAG | Schedule | Role |
| --- | --- | --- |
| `flexihome_market_data_pipeline` | Hourly | Incoming market-data pipeline |
| `flexihome_forecasting_pipeline` | Manual | Train parameterized forecast model |
| `flexihome_optimization_pipeline` | Manual | Run parameterized optimization |
| `flexihome_results_pipeline` | Manual | Run full contract pipeline and validate outgoing manifest |
| `flexihome_local_pipeline` | Manual | Backward-compatible compact local pipeline |

### 17.2 Shared Airflow Defaults

File:

```text
airflow/dags/flexihome_airflow_common.py
```

Production defaults:

- Owner: `flexihome`
- Retries: 2
- Retry delay: 5 minutes
- SLA: 1 hour
- Start date: 2026-01-01 UTC

### 17.3 Market Data DAG

File:

```text
airflow/dags/market_data_pipeline.py
```

Task flow:

```text
fetch_entso_e_configuration -> fetch_fingrid_configuration -> aggregate_market_frame -> validate_market_artifacts
```

It writes:

```text
runs/market_data_*
```

### 17.4 Forecasting DAG

File:

```text
airflow/dags/forecasting_pipeline.py
```

Important parameters:

- `target`
- `forecaster_plugin`
- `lags`
- `horizon_steps`
- `days`
- `n_homes`
- `market_data_mode`

It writes:

```text
runs/forecast_*
```

### 17.5 Optimization DAG

File:

```text
airflow/dags/optimization_pipeline.py
```

Important parameters:

- `market_mode`
- `resource_mode`
- `forecaster_plugin`
- `optimizer_plugin`
- `horizon_hours`
- `dispatch_hours`
- `days`
- `n_homes`

It writes:

```text
runs/optimization_*
```

### 17.6 Results DAG

File:

```text
airflow/dags/results_pipeline.py
```

It runs the full contract pipeline and validates `manifest.json`.

It writes:

```text
runs/pipeline_*
```

### 17.7 Why Separate DAGs

Separate DAGs are useful because:

- Incoming data can run hourly.
- Forecast model training can be run when a model needs refreshing.
- Optimization can be run on demand or on a separate schedule.
- Results serialization can be controlled as a final publication step.

This is closer to production than one large all-in-one script.

---

## 18. Docker Containers

Docker Compose provides local production-style services.

File:

```text
compose.yaml
```

### 18.1 Services

| Service | Container | Role |
| --- | --- | --- |
| `timescaledb` | `flexihome-timescaledb` | Stores optimization jobs and market observations |
| `airflow-postgres` | `flexihome-airflow-postgres` | Airflow metadata database |
| `airflow-init` | `flexihome-airflow-init` | Initializes Airflow database and admin user |
| `airflow-webserver` | `flexihome-airflow-webserver` | Airflow UI |
| `airflow-scheduler` | `flexihome-airflow-scheduler` | Runs scheduled DAG tasks |

### 18.2 Ports

| Service | Host URL |
| --- | --- |
| TimescaleDB/PostgreSQL | `127.0.0.1:5432` |
| Airflow UI | `127.0.0.1:8080` |

### 18.3 Volumes

| Volume | Purpose |
| --- | --- |
| `timescaledb-data` | Persist TimescaleDB data |
| `airflow-postgres-data` | Persist Airflow metadata |
| `./airflow/dags` | Mount DAG definitions into Airflow |
| `./airflow/logs` | Store Airflow logs |
| `./` | Mount project code into Airflow containers |

### 18.4 Airflow Docker Image

File:

```text
Dockerfile.airflow
```

Base image:

```text
apache/airflow:2.10.5-python3.11
```

It installs:

```text
requirements-airflow.txt
```

This gives Airflow enough Python packages to run the FlexiHome scripts.

---

## 19. Local Launch Procedures

### 19.1 Start Dashboard Only

```cmd
cd /d "C:\Users\Kasutaja\OneDrive - Tallinna Tehnikaülikool\Documents\Playground\Real_Time_Control_Energy-Flexibility"
.venv\Scripts\activate
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

### 19.2 Start API Only

```cmd
python scripts\run_api.py --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

### 19.3 Start TimescaleDB

```cmd
copy .env.example .env
docker compose up -d timescaledb
```

Then set:

```cmd
set FLEXIHOME_TIMESCALE_DSN=postgresql://flexihome:flexihome@127.0.0.1:5432/flexihome
set FLEXIHOME_TIMESCALE_SCHEMA=flexihome
python scripts\timescale_store_smoke.py
```

### 19.4 Start Airflow

```cmd
docker compose up -d --build timescaledb airflow-postgres airflow-init airflow-webserver airflow-scheduler
```

Open:

```text
http://127.0.0.1:8080
```

Default local login:

```text
user: admin
password: admin
```

### 19.5 Run Full Pipeline Locally

```cmd
python scripts\run_pipeline.py --market-data-mode synthetic --forecaster-plugin xgboost_default --optimizer-plugin pulp_default --days 7 --n-homes 1500 --horizon-hours 24 --dispatch-hours 24 --output-root runs
```

---

## 20. Production-Style Operating Procedure

This is a practical sequence for operating the system.

### Step 1: Pull Latest Code

```cmd
git fetch origin
git switch codex-fastapi-api-layer
git pull
```

### Step 2: Configure Environment

Create or update `.env`.

For synthetic operation:

```text
FLEXIHOME_MARKET_DATA_MODE=synthetic
```

For real market operation:

```text
ENTSOE_API_KEY=<token>
FINGRID_API_KEY=<key>
FLEXIHOME_MARKET_DATA_MODE=auto
```

For database persistence:

```text
FLEXIHOME_TIMESCALE_DSN=postgresql://flexihome:flexihome@timescaledb:5432/flexihome
FLEXIHOME_TIMESCALE_SCHEMA=flexihome
```

### Step 3: Start Infrastructure

```cmd
docker compose up -d --build timescaledb airflow-postgres airflow-init airflow-webserver airflow-scheduler
```

### Step 4: Verify Health

```cmd
docker compose ps
docker compose exec airflow-scheduler airflow dags list
python scripts\airflow_dag_smoke.py
```

### Step 5: Trigger Incoming Data

```cmd
docker compose exec airflow-scheduler airflow dags trigger flexihome_market_data_pipeline --conf "{\"market_data_mode\":\"auto\",\"market_lookback_days\":30,\"days\":2,\"n_homes\":80,\"freq_minutes\":15}"
```

### Step 6: Trigger Forecasting

```cmd
docker compose exec airflow-scheduler airflow dags trigger flexihome_forecasting_pipeline --conf "{\"target\":\"fcrn_capacity_eur_per_mw_h\",\"forecaster_plugin\":\"xgboost_default\",\"lags\":4,\"horizon_steps\":1,\"days\":7,\"n_homes\":1500,\"market_data_mode\":\"synthetic\"}"
```

### Step 7: Trigger Optimization

```cmd
docker compose exec airflow-scheduler airflow dags trigger flexihome_optimization_pipeline --conf "{\"market_mode\":\"Combined\",\"resource_mode\":\"Hybrid portfolio\",\"forecaster_plugin\":\"xgboost_default\",\"optimizer_plugin\":\"pulp_default\",\"horizon_hours\":24,\"dispatch_hours\":24,\"days\":7,\"n_homes\":1500,\"market_data_mode\":\"synthetic\"}"
```

### Step 8: Trigger Results Publication

```cmd
docker compose exec airflow-scheduler airflow dags trigger flexihome_results_pipeline --conf "{\"market_data_mode\":\"synthetic\",\"forecaster_plugin\":\"xgboost_default\",\"optimizer_plugin\":\"pulp_default\",\"market_mode\":\"Combined\",\"resource_mode\":\"Hybrid portfolio\",\"horizon_hours\":24,\"dispatch_hours\":24,\"days\":7,\"n_homes\":1500}"
```

### Step 9: Inspect Artifacts

Look under:

```text
runs/
```

Important directories:

- `market_data_*`
- `forecast_*`
- `optimization_*`
- `pipeline_*`

### Step 10: Explain Results In The Dashboard

Open Streamlit:

```cmd
python -m streamlit run app.py
```

Use Advanced / Export to inspect external artifacts.

---

## 21. GitHub, Branches, CI, And Collaboration

The repository remote is:

```text
https://github.com/arkm315787/Real_Time_Control_Energy-Flexibility.git
```

Current branch:

```text
codex-fastapi-api-layer
```

The branch:

```text
feature/plugin-architecture
```

has also been aligned with the same production architecture commits.

### 21.1 What GitHub Does For This Project

GitHub is used to:

- Store the project code remotely.
- Track code history.
- Share branches.
- Run CI checks.
- Support future pull requests and reviews.

### 21.2 GitHub CI

File:

```text
.github/workflows/ci.yml
```

CI runs on:

- Push.
- Pull request.

Checks include:

- Install dependencies.
- Compile core modules.
- API smoke test.
- Plugin architecture smoke test.
- Pipeline DAG smoke test.
- Airflow DAG smoke test.
- Forecast module smoke test.
- Optimization module smoke test.

This is important because it catches broken imports, broken API behavior, broken plugin registration, broken pipeline DAG metadata, and broken local module runners.

### 21.3 Normal GitHub Workflow

Typical team workflow:

1. Create a branch.
2. Make changes.
3. Run smoke tests locally.
4. Commit changes.
5. Push branch.
6. Open pull request.
7. Let GitHub CI run.
8. Review and merge.

### 21.4 Important Branch History

Recent architecture commits:

- `f88c34b Add plugin and pipeline architecture`
- `711fdeb Add production Airflow pipelines`

These commits moved the project from dashboard/API prototype toward a production-style stack.

---

## 22. Repository Map

```text
Real_Time_Control_Energy-Flexibility/
├── app.py
├── README.md
├── compose.yaml
├── Dockerfile.airflow
├── requirements.txt
├── requirements-dev.txt
├── requirements-airflow.txt
├── .env.example
├── .github/workflows/ci.yml
├── airflow/
│   └── dags/
│       ├── flexihome_airflow_common.py
│       ├── flexihome_pipeline.py
│       ├── market_data_pipeline.py
│       ├── forecasting_pipeline.py
│       ├── optimization_pipeline.py
│       └── results_pipeline.py
├── flexihome/
│   ├── api/
│   ├── core/
│   │   └── base/
│   ├── pipeline/
│   │   └── stages/
│   └── plugins/
└── scripts/
```

### Important Script Families

| Script | Purpose |
| --- | --- |
| `run_api.py` | Start FastAPI service |
| `run_forecasting.py` | Train one forecast model and write artifacts |
| `run_optimization.py` | Run optimization and write artifacts |
| `run_pipeline.py` | Run full contract pipeline |
| `run_market_data_ingestion.py` | Run incoming data ingestion slice |
| `api_smoke.py` | Test API without network server |
| `plugin_smoke.py` | Test plugin registration and execution |
| `pipeline_smoke.py` | Test explicit pipeline and manifest |
| `airflow_dag_smoke.py` | Static test for Airflow DAG coverage |
| `market_data_smoke.py` | Test market-data fallback/status |
| `timescale_store_smoke.py` | Test TimescaleDB store if DSN is configured |
| `validate_pipeline_manifest.py` | Validate latest artifact directory |

---

## 23. Smoke Tests And Validation

Smoke tests are quick checks that prove the main system parts still work.

Recommended local checks:

```cmd
python -m py_compile app.py flexihome\core\engine.py flexihome\api\optimizer.py flexihome\api\services.py
python scripts\market_data_smoke.py
python scripts\plugin_smoke.py
python scripts\pipeline_smoke.py
python scripts\airflow_dag_smoke.py
python scripts\api_smoke.py
python scripts\timescale_store_smoke.py
```

### What Each Test Proves

| Test | Proves |
| --- | --- |
| `py_compile` | Python files can be imported/parsed |
| `market_data_smoke.py` | Synthetic fallback and market status work |
| `plugin_smoke.py` | Forecaster, optimizer, and stage plugins register and run |
| `pipeline_smoke.py` | Full contract pipeline runs and writes manifest |
| `airflow_dag_smoke.py` | Airflow DAG files, retry/SLA settings, and plugin parameters exist |
| `api_smoke.py` | API endpoints work without starting a server |
| `timescale_store_smoke.py` | TimescaleDB works when configured |

---

## 24. Common Questions And Explanations

### Is FlexiHome controlling real homes?

No. It currently simulates a virtual residential portfolio. It is designed for research, education, architecture development, and production planning.

### Does real market data mean the whole dataset is real?

No. The portfolio and devices are scenario-based. Real data is used for market prices and activation signals where configured.

### Why use forecasting?

The optimizer needs to plan ahead. Forecasting gives it expected future values.

### Why use optimization?

There are many constraints and tradeoffs. Optimization chooses a schedule that balances revenue, comfort, energy limits, and reserve requirements.

### Why use Airflow?

Airflow turns manual scripts into scheduled, observable production workflows.

### Why use TimescaleDB?

TimescaleDB persists time-indexed data, job history, and market observations.

### Why use plugins?

Plugins let the team replace models and solvers without rewriting the dashboard, API, or pipeline.

### What is the difference between dashboard, API, scripts, and Airflow?

| Surface | User |
| --- | --- |
| Dashboard | Human exploring scenarios |
| API | Software system requesting jobs |
| Scripts | Developer or operator running specific tasks |
| Airflow | Production scheduler coordinating tasks |

---

## 25. Glossary

| Term | Meaning |
| --- | --- |
| Aggregation | Combining many small resources into one larger portfolio |
| aFRR | Automatic frequency restoration reserve |
| API | A structured way for software to communicate |
| BESS | Battery energy storage system |
| DAG | Directed acyclic graph, an Airflow workflow |
| Dispatch | The actual power action assigned to resources |
| ENTSO-E | European electricity data platform |
| FCR-N | Frequency containment reserve for normal operation |
| Fingrid | Finnish transmission system operator |
| Forecast | Prediction of future values |
| HVAC | Heating, ventilation, and air conditioning |
| MPC | Model Predictive Control |
| Plugin | Replaceable component selected by name |
| PV | Solar photovoltaic generation |
| Reserve | Flexible capacity offered to help the grid |
| SLA | Service-level agreement, timing expectation in Airflow |
| Streamlit | Python framework for interactive dashboards |
| TimescaleDB | Time-series database built on PostgreSQL |
| VPP | Virtual power plant |

---

## 26. Appendix A: Command Cheat Sheet

### Activate Environment

```cmd
.venv\Scripts\activate
```

### Install Dependencies

```cmd
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run Dashboard

```cmd
python -m streamlit run app.py
```

### Run API

```cmd
python scripts\run_api.py --reload
```

### Run Full Pipeline

```cmd
python scripts\run_pipeline.py --days 1 --n-homes 80 --horizon-hours 1 --dispatch-hours 1 --output-root runs
```

### Run Airflow

```cmd
docker compose up -d --build timescaledb airflow-postgres airflow-init airflow-webserver airflow-scheduler
```

### List Airflow DAGs

```cmd
docker compose exec airflow-scheduler airflow dags list
```

### Trigger Results DAG

```cmd
docker compose exec airflow-scheduler airflow dags trigger flexihome_results_pipeline --conf "{\"market_data_mode\":\"synthetic\",\"forecaster_plugin\":\"xgboost_default\",\"optimizer_plugin\":\"pulp_default\",\"market_mode\":\"Combined\",\"resource_mode\":\"Hybrid portfolio\",\"horizon_hours\":24,\"dispatch_hours\":24,\"days\":7,\"n_homes\":1500}"
```

### Run Smoke Tests

```cmd
python scripts\market_data_smoke.py
python scripts\plugin_smoke.py
python scripts\pipeline_smoke.py
python scripts\airflow_dag_smoke.py
python scripts\api_smoke.py
python scripts\timescale_store_smoke.py
```

---

## 27. Appendix B: Example API Calls

### Health

```powershell
Invoke-RestMethod http://localhost:8000/health
```

### List Plugins

```powershell
Invoke-RestMethod http://localhost:8000/plugins
```

### Forecast

```powershell
$body = @{
  forecaster_plugin = "xgboost_default"
  target = "net_load_baseline_kw"
  lags = 4
  horizon_steps = 1
  portfolio_config = @{
    days = 1
    n_homes = 80
    freq_minutes = 15
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://localhost:8000/forecast -Body $body -ContentType "application/json"
```

### Optimization

```powershell
$body = @{
  market_mode = "Combined"
  resource_mode = "Hybrid portfolio"
  forecaster_plugin = "xgboost_default"
  optimizer_plugin = "pulp_default"
  horizon_hours = 1
  dispatch_hours = 1
  portfolio_config = @{
    days = 1
    n_homes = 80
    freq_minutes = 15
  }
} | ConvertTo-Json -Depth 5

$job = Invoke-RestMethod -Method Post -Uri http://localhost:8000/optimize -Body $body -ContentType "application/json"
$job
Invoke-RestMethod "http://localhost:8000$($job.results_url)"
```

---

## 28. Appendix C: How To Explain This Project To Someone Else

Here is a simple explanation:

> FlexiHome is a virtual power plant prototype for residential flexibility. It simulates many homes with EVs, batteries, solar panels, and smart HVAC. It can use synthetic market conditions or real ENTSO-E and Fingrid market data. It forecasts future load, solar, prices, and activation signals. Then it uses an optimizer to decide how the home portfolio should bid into FCR-N and aFRR reserve markets while respecting comfort, battery, EV, and response-speed constraints. The results can be explored in a dashboard, requested through an API, stored in a database, or orchestrated in production-style Airflow DAGs. The system now has plugins, explicit pipeline contracts, artifact manifests, Docker services, and GitHub CI checks, so it is moving from an educational app toward a production-ready architecture.

For a very short version:

> FlexiHome shows how thousands of small home devices can be coordinated like one flexible energy resource for grid balancing markets.

For a technical but still understandable version:

> The system generates or ingests time-series data, trains forecasting models, runs a receding-horizon MPC optimizer, simulates 4-second reserve tracking, validates simplified market-compliance rules, and publishes results through Streamlit, FastAPI, Airflow, artifacts, and optional TimescaleDB storage.

---

## Closing Note

FlexiHome is now best understood as three layers:

1. **User and operator layer:** dashboard, API, Airflow UI.
2. **Decision layer:** forecasting, optimization, tracking, compliance.
3. **Production layer:** plugins, pipelines, manifests, database, Docker, GitHub CI.

The dashboard is the visual explanation surface. The API is the machine-to-machine surface. Airflow is the production orchestration surface. The shared core is the modeling brain. The database and artifacts are the memory. GitHub is the collaboration and validation backbone.
