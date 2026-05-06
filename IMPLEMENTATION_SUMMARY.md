# Three-Gap Resolution Implementation Summary

## Gap 1: Market as Independent Simulation

Status: implemented for the Streamlit live session.

Files:
- `flexihome/core/market_simulator.py`
- `app.py`

What it does:
- `MarketSimulator` generates 4-second market pressure signals from wall-clock time.
- The dashboard creates a simulator when `Start Market Pressure` is pressed.
- `sync_market_simulator_feed()` polls the simulator on Streamlit reruns and overlays the live ticks onto the 4-second feed used by the lower controller and the market plot.
- The market remains separate from the upper MPC solve; upper MPC prepares the socket, and the live market process starts later.

Current limitation:
- Streamlit reruns are still the execution driver. This is suitable for the interactive dashboard, but a production service should move the 4-second tick loop into a background worker or process.

## Gap 2: Lower MPC Live Attachment

Status: integrated as an incremental live replay against elapsed wall-clock time.

What it does:
- Lower MPC can be armed before market start.
- Once the market is live, lower MPC only solves the number of 4-second ticks that have elapsed.
- It consumes the live simulator feed where available and falls back to the preview signal for future/non-polled ticks.
- Partial live summaries are scaled by solved 4-second ticks, not by full 15-minute intervals.

Current limitation:
- The app still recomputes the live lower result during reruns from elapsed time. It does not yet run a persistent background control worker.

## Gap 3: Socket Enforcement

Status: implemented and visible in telemetry.

Files:
- `flexihome/core/engine.py`
- `flexihome/core/centralized_controller.py`
- `app.py`

What it does:
- Upper MPC writes `socket_up_kw` and `socket_down_kw` into each market interval.
- Lower MPC clamps every TSO request to those socket bounds.
- The 4-second tracking table logs:
  - `raw_request_kw`
  - `product_clamped_request_kw`
  - `socket_up_kw`
  - `socket_down_kw`
  - `socket_violation_up_kw`
  - `socket_violation_down_kw`
- Lower MPC also keeps measured-delivery columns separate from ideal model response:
  - `ideal_delivered_kw`
  - `delivered_signed_kw`
  - `telemetry_error_kw`
  - `tracking_error_kw`

## Dashboard Fixes

- `Mean abs tracking error` again uses absolute error instead of signed mean error.
- Signed error is kept separately in `mean_signed_error_kw`.
- Live simulation plots use the 4-second lower tracking frame when lower MPC is active.

## Remaining Production Step

The next production-grade step is to move the lower MPC tick execution out of Streamlit reruns and into a dedicated 4-second worker with durable state, heartbeat monitoring, command acknowledgements, and gateway telemetry ingestion.
