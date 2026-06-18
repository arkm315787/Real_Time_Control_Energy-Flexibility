# Residential VPP: Docker, Airflow, and Application Architecture - Detailed Explanation

## 1. WHERE YOUR APPLICATION ACTUALLY LIVES

Your Residential VPP application exists in **THREE DIFFERENT PLACES**:

### 1.1 On Your Computer (Host Machine)
- **Location**: `C:\Users\Kasutaja\OneDrive - Tallinna Tehnikaülikool\Documents\Playground\Real_Time_Control_Energy-Flexibility\`
- **Files**: 
  - `app.py` — The Streamlit dashboard (interactive web UI)
  - `flexihome/` folder — Core Python code for forecasting, optimization, market data
  - `airflow/dags/` — Workflow automation scripts
  - `scripts/` — Utility scripts for running the app
  - `requirements.txt` — Python dependencies

**This is the SOURCE CODE.** It lives on your computer unmodified.

### 1.2 Inside Docker Containers (Isolated Linux Environments)
When you run `docker compose up`, Docker creates **isolated Linux environments** (containers) and copies/mounts your code into them:

| Container | What's Inside | Purpose |
|-----------|--------------|---------|
| **timescaledb** | PostgreSQL + TimescaleDB | Database to store market data and optimization results |
| **airflow-postgres** | PostgreSQL | Database to store Airflow metadata (which DAGs ran, when, success/failure) |
| **airflow-init** | Python + Airflow + your code | One-time setup container; migrates Airflow DB, creates admin user |
| **airflow-webserver** | Python + Airflow + your code | Hosts the Airflow UI at `http://127.0.0.1:8080` |
| **airflow-scheduler** | Python + Airflow + your code | Runs your DAGs on a schedule |

### 1.3 In Memory (When Running)
When you run `python -m streamlit run app.py` on your computer (NOT in Docker):
- Python loads `app.py` and `flexihome/` modules
- Creates objects (dataframes, forecasting models, optimizations)
- Holds them in RAM while you interact with the dashboard at `http://localhost:8501`

**This Streamlit app is NOT in Docker** — it runs directly on your computer.

---

## 2. HOW DOCKER ENABLES YOUR APPLICATION

Docker is a **containerization technology**. It solves the problem: *"It works on my machine but not on others."*

### 2.1 What Docker Does For You

**Without Docker:**
- You'd need to manually install PostgreSQL, Airflow, Python dependencies on your computer
- If you share your project with a teammate, they'd have to do the same
- Different OS (Windows vs Mac vs Linux) might have subtle differences
- Airflow setup is complex with many configuration files

**With Docker:**
- Docker packages everything (OS, Python, PostgreSQL, Airflow, dependencies) into a **container image**
- Any container with that image runs **identically everywhere** — Windows, Mac, Linux, cloud servers
- You just run `docker compose up` and everything starts

### 2.2 Why Your Project Uses Docker

Your `compose.yaml` defines 5 containers because:

1. **TimescaleDB container** (timescaledb)
   - Runs PostgreSQL in an isolated environment
   - Stores market data, optimization results
   - Persists data in a **volume** (so data survives if container restarts)

2. **Airflow infrastructure** (3-4 containers)
   - Airflow itself is complex to set up
   - Docker handles all that complexity
   - Your code is **mounted** into the containers (so you can edit locally and see changes immediately)

---

## 3. THE ROLE OF AIRFLOW

Airflow is a **workflow orchestration tool**. It automates the data pipeline.

### 3.1 What Airflow Does

**Without Airflow:**
- You manually run Python scripts: `python run_market_data.py`, then `python run_forecasting.py`, then `python run_optimization.py`
- If a step fails, you have to manually restart it
- You have to remember to run them in the right order
- Hard to schedule them (e.g., "run every hour")

**With Airflow:**
- Define a **DAG** (directed acyclic graph) — a workflow as code
- Airflow automatically runs tasks in order, on a schedule
- Built-in retry logic, error notifications, logging
- You can monitor everything via a web UI

### 3.2 Your Airflow DAGs

These are defined in `airflow/dags/`:

| DAG Name | Schedule | What It Does |
|----------|----------|-------------|
| `flexihome_market_data_pipeline` | Every hour | Fetch real market data from ENTSO-E/Fingrid APIs, validate it, store in TimescaleDB |
| `flexihome_forecasting_pipeline` | Manual/API trigger | Train forecasting models (XGBoost) for demand, prices, etc. |
| `flexihome_optimization_pipeline` | Manual/API trigger | Run MPC optimizer to decide reserve bids and resource dispatch |
| `flexihome_results_pipeline` | Manual/API trigger | Run full contract pipeline, validate, export results |
| `flexihome_local_pipeline` | Manual/API trigger | Compact backward-compatible local forecast + optimization |

### 3.3 Example: How `flexihome_market_data_pipeline` Works

**DAG Definition (in code):**
```python
task_fetch_entsoe -> task_fetch_fingrid -> task_aggregate -> task_validate -> task_store
```

**When Airflow runs it (hourly):**

1. **task_fetch_entsoe** 
   - Python code calls ENTSO-E API
   - Downloads spot electricity prices for Finland
   - Saves to CSV

2. **task_fetch_fingrid**
   - Python code calls Fingrid API
   - Downloads FCR-N and aFRR reserve prices
   - Saves to CSV

3. **task_aggregate**
   - Loads both CSV files
   - Combines them into one dataframe
   - Adds synthetic data for missing values

4. **task_validate**
   - Checks data quality (no NaNs, expected columns exist, values in range)
   - If validation fails, Airflow stops and alerts you

5. **task_store**
   - Writes aggregated dataframe to TimescaleDB
   - Creates timestamped file under `runs/market_data_*`
   - Stores manifest.json with metadata

### 3.4 The Airflow Webserver (`http://127.0.0.1:8080`)

At this URL, you can:
- See all DAGs and their schedules
- Manually trigger a DAG run
- View the DAG graph (visual diagram of tasks)
- Check logs from each task
- Monitor performance, failures, retries

---

## 4. HOW DOCKER AND AIRFLOW WORK TOGETHER

```
┌─────────────────────────────────────────────────────────────┐
│ Your Computer (Host Machine)                               │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Docker Desktop                                       │  │
│  │                                                      │  │
│  │  ┌────────────────┐  ┌───────────────────────────┐ │  │
│  │  │ TimescaleDB    │  │ Airflow Containers      │ │  │
│  │  │ Container      │  │ ┌──────────────────────┐ │ │  │
│  │  │ (PostgreSQL)   │◄─►│ Webserver            │ │ │  │
│  │  │ Port 5432      │  │ (http://127.0.0.1:8080)│ │ │  │
│  │  └────────────────┘  └──────────────────────────┘ │ │  │
│  │                       ┌──────────────────────────┐ │ │  │
│  │                       │ Scheduler              │ │ │  │
│  │                       │ (runs DAGs on schedule)│ │ │  │
│  │                       └──────────────────────────┘ │ │  │
│  │                       ┌──────────────────────────┐ │ │  │
│  │                       │ Init Container         │ │ │  │
│  │                       │ (setup)                │ │ │  │
│  │                       └──────────────────────────┘ │ │  │
│  │                                                      │  │
│  │  All containers share:                             │  │
│  │  - Your code from ./airflow/dags/ (bind mount)    │  │
│  │  - Your code from ./ (bind mount to /opt/airflow)│  │
│  │  - Environment variables from .env                │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                         ▲                                  │
│                         │ (via Docker network)            │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Streamlit Dashboard (NOT in Docker)               │  │
│  │ python -m streamlit run app.py                    │  │
│  │ http://localhost:8501                             │  │
│  │                                                    │  │
│  │ Reads from TimescaleDB in container ──────────────┼──┼─►
│  │ Runs forecasting/optimization in Python process  │  │
│  │ (uses local ./flexihome/ code)                  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. DETAILED DATA FLOW

### 5.1 Market Data Ingestion (Airflow DAG)

```
ENTSO-E API (Europe)
└─► Airflow scheduler wakes up at :00
    └─► task_fetch_entsoe (in container)
        └─► Calls ENTSO-E API for spot prices
            └─► Saves data
                └─► task_aggregate
                    └─► Combines ENTSO-E + Fingrid data
                        └─► task_store
                            └─► Writes to TimescaleDB container
                                └─► Data persisted in /var/lib/docker/volumes/
```

### 5.2 Running Optimization via Streamlit (NOT in Airflow)

```
Your browser (http://localhost:8501)
└─► Click "Run MPC dispatch" button
    └─► Streamlit process on your computer calls Python code
        └─► ./flexihome/core/engine.py (run_mpc_controller)
            └─► Reads from TimescaleDB (connects to container on 127.0.0.1:5432)
                └─► Runs XGBoost forecasting models
                    └─► Runs PuLP optimization solver
                        └─► Results displayed in dashboard
                            └─► Optionally saved to ./runs/ (your computer)
```

### 5.3 Scheduled Market Data Via Airflow (Automation)

```
Airflow scheduler checks: Is it time to run flexihome_market_data_pipeline?
└─► Yes (every hour)
    └─► Airflow-scheduler container launches tasks
        └─► Each task = Python function inside the container
            └─► Reads .env for API keys (ENTSOE_API_KEY, FINGRID_API_KEY)
                └─► Calls external APIs
                    └─► Stores results in TimescaleDB (also in container)
                        └─► Logs written to ./airflow/logs/ (mounted from host)
                            └─► You can read logs on your computer
```

---

## 6. WHERE THE CODE ACTUALLY RUNS

### Scenario A: Streamlit Dashboard (Interactive Visualization)

**Where code runs:** Your computer's RAM + Docker network connection
**Code location:** `./app.py`, `./flexihome/`
**Command:** `python -m streamlit run app.py`
**Execution:**
- Python interpreter on your computer loads `app.py`
- Runs at `http://localhost:8501`
- When you click buttons, functions execute in your Python process
- Fast response (no network delay to containers)

**Example:**
```python
# You click "Run MPC dispatch" in Streamlit
# This code runs on YOUR COMPUTER:
result = run_mpc_controller(
    df=df,  # loaded in your Python process
    models=models,  # trained in your Python process
    ...
)
# Results shown in your browser within seconds
```

### Scenario B: Airflow DAG Execution (Scheduled Automation)

**Where code runs:** Inside Airflow containers (Linux)
**Code location:** Mounted from `./airflow/dags/` into `/opt/airflow/dags/` in container
**Command:** Automatic (scheduler triggers it)
**Execution:**
- Airflow-scheduler container wakes up every hour
- Reads `flexihome_market_data_pipeline.py` from the mounted volume
- Spawns task processes inside the container
- Each task runs Python code, may call external APIs
- Results stored in TimescaleDB (also in container)

**Example:**
```python
# Inside airflow-scheduler container, this code runs:
def task_fetch_entsoe():
    api_key = os.environ['ENTSOE_API_KEY']  # from .env
    response = requests.get('https://api.entsoe.eu/...')  # external API call
    df = pd.DataFrame(response.json())
    df.to_csv('market_data.csv')  # saved in container
    return df

# Airflow runs this automatically every hour
```

### Scenario C: Database Access (Shared Resource)

**Where TimescaleDB runs:** Inside a Docker container
**How you access it:**
- **From Streamlit (your computer):** 
  ```python
  conn = psycopg2.connect('postgresql://flexihome:flexihome@127.0.0.1:5432/flexihome')
  ```
  Port 5432 exposed by `ports: - "127.0.0.1:5432:5432"` in compose.yaml

- **From Airflow containers:**
  ```python
  conn = psycopg2.connect('postgresql://flexihome:flexihome@timescaledb:5432/flexihome')
  ```
  Uses container network (timescaledb is the DNS name inside the network)

**Result:** Both can read/write to the same database

---

## 7. WHY DOCKER CAN'T RUN THE WHOLE APPLICATION

Docker containers run in isolated Linux environments. They're great for **background services** (databases, scheduled jobs), but NOT for interactive dashboards on Windows/Mac.

### Why?

1. **Docker is Headless (No GUI)**
   - Containers run in the background
   - No display screen
   - Streamlit needs to display a web browser interface
   - Can't do that from inside a container on your personal computer

2. **Windows/Mac Docker Desktop Runs Linux VMs**
   - Docker Desktop on Windows/Mac actually runs a lightweight Linux virtual machine
   - Containers run inside that Linux VM
   - A Streamlit app inside that VM can't talk to your Windows browser easily
   - (It's technically possible but adds network complexity)

3. **Streamlit is Interactive**
   - User clicks buttons → Python code runs immediately
   - Needs fast local execution (milliseconds)
   - Container network calls add latency
   - Defeats the purpose of an interactive dashboard

### What Docker CAN Do Well

✅ **Background Services:**
- TimescaleDB database
- Airflow scheduler running DAGs
- FastAPI optimizer service
- Batch processing jobs

### What Should Run Locally (Not in Docker)

✅ **Interactive Applications:**
- Streamlit dashboard (`app.py`)
- Jupyter notebooks
- Local scripts you're developing

---

## 8. ARCHITECTURE SUMMARY

```
┌─────────────────────────────────────────────────────────────────────┐
│ YOUR COMPUTER (Windows/Mac)                                         │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐│
│ │ Docker Desktop (Runs Linux VM inside)                          ││
│ │                                                                 ││
│ │  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐  ││
│ │  │ TimescaleDB     │  │ Airflow      │  │ Airflow         │  ││
│ │  │ Container       │  │ Scheduler    │  │ Webserver       │  ││
│ │  │ (Database)      │  │ (Automation) │  │ (UI at :8080)   │  ││
│ │  │                 │◄─┤              │  │                 │  ││
│ │  │ Stores:         │  │ Runs DAGs    │  │ Monitor DAGs    │  ││
│ │  │ - Market data   │  │ every hour   │  │ + logs          │  ││
│ │  │ - Opt. results  │  │              │  │                 │  ││
│ │  └─────────────────┘  └──────────────┘  └─────────────────┘  ││
│ │                                                                 ││
│ │  Volumes (persistent storage):                                ││
│ │  - timescaledb-data → stays even if container restarts       ││
│ │  - airflow-postgres-data → Airflow metadata                  ││
│ │                                                                 ││
│ └─────────────────────────────────────────────────────────────────┘│
│  Port :5432    Port :8080                                          │
│      ▲              ▲                                              │
│      │              │                                              │
│  ┌───┴──────────────┴───────────────────────────────────────────┐ │
│  │ Streamlit Dashboard                                         │ │
│  │ (Your Computer Python Process)                             │ │
│  │                                                             │ │
│  │ python -m streamlit run app.py                             │ │
│  │ http://localhost:8501                                      │ │
│  │                                                             │ │
│  │ - Reads from TimescaleDB container via :5432             │ │
│  │ - Runs forecasting/optimization code from ./flexihome/   │ │
│  │ - Displays interactive dashboard                          │ │
│  │ - Saves results to ./runs/ folder                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Your File System:                                                  │
│  ./app.py                 ← Streamlit code (runs locally)         │
│  ./flexihome/           ← Core logic (runs locally)             │
│  ./airflow/dags/          ← Mounted into containers              │
│  ./airflow/logs/          ← Container writes logs here           │
│  ./runs/                  ← Results from optimization             │
│  .env                     ← Secrets & config                     │
│  compose.yaml             ← Docker configuration                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. WHAT HAPPENS STEP-BY-STEP WHEN YOU START

### Step 1: You run `docker compose up -d`

```bash
$ docker compose up -d
```

**What happens:**
- Docker reads `compose.yaml`
- Pulls images: `timescale/timescaledb:latest-pg17`, `postgres:16`, builds `residential_vpp-airflow:local`
- Creates a Docker network: `real_time_control_energy-flexibility_default`
- Starts 5 containers in order:
  1. airflow-postgres (needed before airflow-init)
  2. timescaledb (independent, starts immediately)
  3. airflow-init (waits for airflow-postgres to be healthy, runs once, exits)
  4. airflow-webserver (waits for airflow-init to complete)
  5. airflow-scheduler (waits for airflow-init to complete)

**Result:**
- TimescaleDB is ready at `127.0.0.1:5432`
- Airflow webserver is ready at `127.0.0.1:8080`
- Airflow scheduler is watching for DAGs to run

### Step 2: You open Streamlit in a DIFFERENT terminal

```bash
$ python -m streamlit run app.py
```

**What happens:**
- Python interpreter loads `app.py` in YOUR COMPUTER's RAM
- Streamlit starts a web server at `127.0.0.1:8501`
- You open browser to `http://localhost:8501`
- Streamlit connects to TimescaleDB (in container) via port 5432

**Result:**
- Interactive dashboard is live
- You can adjust sliders, run optimization, see results

### Step 3: Airflow Automatically Runs DAGs

**Every hour:**
- Airflow scheduler wakes up
- Checks if `flexihome_market_data_pipeline` should run (yes, every hour)
- Spawns tasks inside airflow-scheduler container
- Tasks call ENTSO-E/Fingrid APIs
- Data written to TimescaleDB container
- Logs written to `./airflow/logs/`

**When you manually trigger (from Airflow UI at :8080):**
- Click "Trigger" on a DAG
- Scheduler runs tasks
- Results stored in database

### Step 4: You Stop Everything

```bash
$ docker compose down
```

**What happens:**
- Containers shut down gracefully
- Networks removed
- **Volumes persist** (timescaledb-data, airflow-postgres-data)
- When you `docker compose up` again, all the old data is still there

---

## 10. DOCKER'S IMPACT ON YOUR APPLICATION

### What Docker Provides

✅ **Reproducibility**
- Same code + same Docker image = same behavior everywhere
- Teammate runs your project: `docker compose up` → works exactly like yours

✅ **Isolation**
- Database, scheduler, webserver all run in isolated containers
- If one crashes, others keep running
- Can restart single service without affecting others

✅ **Simplified Deployment**
- No need to manually install PostgreSQL, Airflow, Python packages
- `docker compose up` handles everything

✅ **Persistence**
- Volumes keep database data even if containers restart
- Logs saved to host filesystem

✅ **Networking**
- Containers communicate via Docker network
- Streamlit on your computer → connects to TimescaleDB in container via 127.0.0.1:5432

### What Docker Doesn't Do

❌ **Doesn't run your interactive Streamlit app** (should run locally for speed)

❌ **Doesn't replace Python** (Docker runs Python, but Python code still executes)

❌ **Doesn't make slow code fast** (optimization logic runs same speed, just isolated)

---

## 11. AIRFLOW'S IMPACT ON YOUR APPLICATION

### What Airflow Provides

✅ **Automation**
- Hourly market data ingestion (no manual script runs)
- Scheduled forecasting model retraining
- Scheduled optimization runs
- All without you clicking anything

✅ **Reliability**
- Automatic retries if a task fails
- Doesn't move to next task until previous one succeeds
- Built-in monitoring, alerting

✅ **Auditability**
- Every DAG run logged
- You can see exactly when it ran, how long, what failed
- Full visibility into data pipeline

✅ **Scalability**
- Can scale to multiple machines (though LocalExecutor only handles single machine)
- Could switch to Kubernetes/Celery executor for distributed execution

### What Airflow Doesn't Do

❌ **Doesn't replace the Streamlit dashboard** (Streamlit for interactive exploration, Airflow for automation)

❌ **Doesn't speed up individual tasks** (if optimization takes 5 minutes, Airflow runs it in 5 minutes)

❌ **Doesn't decide resource allocation** (DAGs define the workflow, your code decides dispatch)

---

## 12. REAL-WORLD SCENARIO

### Morning: You start the system

```bash
docker compose up -d
python -m streamlit run app.py
```

**In Docker:**
- TimescaleDB starts, ready to store data
- Airflow scheduler starts, watches for tasks

**In your browser:**
- Open http://localhost:8501
- Streamlit dashboard loads
- Adjust sliders: "I want to simulate 2000 homes with more solar"
- Click "Run MPC dispatch"
- Optimization runs on your computer, results shown in seconds
- You save results

### Hourly: Airflow Runs Market Data Fetch

**Without user interaction:**
- Airflow scheduler wakes up
- Runs `flexihome_market_data_pipeline` DAG
- Fetches latest prices from ENTSO-E
- Fetches latest FCR-N prices from Fingrid
- Stores in TimescaleDB container
- Updates are logged to `./airflow/logs/`

**Next time you open Streamlit dashboard:**
- Fresh market data is available
- Your forecasting models can use latest prices

### Later: You Run a Forecasting DAG

```bash
# You manually trigger from Airflow UI at http://localhost:8080
# Or from CLI:
docker compose exec airflow-scheduler \
  airflow dags trigger flexihome_forecasting_pipeline \
  --conf '{"target":"fcrn_capacity_eur_per_mw_h",...}'
```

**In container:**
- Forecasting task reads from TimescaleDB
- Trains XGBoost model
- Saves model to `./runs/forecast_*` (mounted volume, visible on your computer)
- Stores metadata in airflow-postgres

**You can see:**
- When you open Streamlit, click "Advanced/Export" tab
- View the trained model results
- Visualize forecasting accuracy

### End of Day: Stop Everything

```bash
docker compose down
```

**What happens:**
- Containers stop
- All database data saved in volumes
- Next day: `docker compose up -d` → system comes back with all old data

---

## 13. KEY TAKEAWAYS

1. **Docker = Container + Isolation + Reproducibility**
   - Runs background services (databases, schedulers)
   - NOT for interactive dashboards (use local Python)

2. **Your Code = Lives Everywhere**
   - Source: `C:\Users\...\Real_Time_Control_Energy-Flexibility\`
   - Inside Containers: Mounted from source (changes sync automatically)
   - In Memory: When Python runs it

3. **Streamlit App (Interactive) = Runs on Your Computer**
   - Connects to TimescaleDB in container
   - Runs forecasting/optimization logic locally
   - Displays results in browser

4. **Airflow = Automation + Scheduling**
   - Runs inside containers
   - Orchestrates data pipelines
   - Executes DAGs on schedule
   - Monitors all runs, logs everything

5. **Data Flow**
   ```
   ENTSO-E/Fingrid APIs
   ↓ (hourly via Airflow DAG in container)
   TimescaleDB container
   ↑ (Streamlit reads from here via 127.0.0.1:5432)
   Streamlit on your computer
   ↓ (you interact with dashboard)
   Forecasting & Optimization (Python on your computer)
   ↓ (results stored back to TimescaleDB)
   Visualization (Streamlit displays charts)
   ```

6. **Why Both Docker + Local Python?**
   - Docker: Reliable background services + scheduling
   - Local Python: Interactive exploration + fast iteration
   - Together: Production-grade data pipeline + interactive learning tool

---

## 14. FILE STRUCTURE CLARITY

```
Real_Time_Control_Energy-Flexibility/
│
├── app.py                          # ← Streamlit dashboard (YOUR COMPUTER)
├── requirements.txt                # ← Python dependencies (for LOCAL Python)
├── Dockerfile.airflow              # ← Blueprint for Airflow image
│
├── flexihome/
│   ├── core/
│   │   ├── engine.py              # ← Optimization, forecasting logic (YOUR COMPUTER + AIRFLOW)
│   │   ├── base/
│   │   │   ├── registry.py        # ← Plugin system
│   ├── api/                        # ← Optional FastAPI service
│   ├── plugins/                    # ← XGBoost, PuLP solvers
│   ├── pipeline/                   # ← Data pipeline stages
│
├── airflow/
│   ├── dags/
│   │   ├── flexihome_market_data_pipeline.py    # ← DAG (AIRFLOW CONTAINER)
│   │   ├── flexihome_forecasting_pipeline.py    # ← DAG (AIRFLOW CONTAINER)
│   │   ├── flexihome_optimization_pipeline.py   # ← DAG (AIRFLOW CONTAINER)
│   ├── logs/                       # ← Task logs written here (MOUNTED from container)
│
├── scripts/
│   ├── run_api.py                  # ← Optional FastAPI server (YOUR COMPUTER)
│   ├── run_pipeline.py             # ← Standalone pipeline (YOUR COMPUTER)
│   ├── run_market_data_ingestion.py # ← Standalone ingestion (YOUR COMPUTER)
│
├── runs/                           # ← Output artifacts (Optimization/Forecasting results)
│
├── .env                            # ← Secrets (API keys, DB credentials)
├── .env.example                    # ← Template for .env
├── compose.yaml                    # ← Docker configuration (defines containers)
└── docker-compose.yml              # ← Alternative name (usually compose.yaml in Docker v2+)
```

**Legend:**
- ← YOUR COMPUTER = Runs on your Windows/Mac
- ← AIRFLOW CONTAINER = Runs inside Docker container (Linux)
- ← MOUNTED from container = Written by container, visible on your computer
- ← DOCKER CONFIGURATION = Tells Docker how to build & run containers
