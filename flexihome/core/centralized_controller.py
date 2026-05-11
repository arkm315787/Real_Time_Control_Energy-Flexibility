"""Centralized household roster and 4-second MPC dispatch for Coverly."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

DEVICE_ORDER = ["BESS", "EV", "HVAC", "PV"]
FCR_N_MIN_BID_KW = 100.0
FCR_N_BID_GRANULARITY_KW = 100.0
FCR_N_HVAC_SHARE_CAP = 0.20
AFRR_MIN_BID_KW = 1000.0
AFRR_BID_GRANULARITY_KW = 1000.0
RECOVERY_PRIORITY = ["BESS", "EV", "HVAC", "PV"]
FCR_N_RESPONSE_63_SECONDS = 60.0
FCR_N_RESPONSE_95_SECONDS = 180.0
COMBINED_STACKING_SPLIT_MODEL = "co_optimized_split_capacity"
COMBINED_STACKING_SHARED_MODEL = "shared_capacity_max_socket"

DEVICE_DEFAULTS = {
    "BESS": {
        "up_power_kw": 5.0,
        "down_power_kw": 5.0,
        "capacity_mwh": 0.012,
        "response_time_s": 2.0,
        "protocol": "Modbus/CAN via simulated gateway",
        "max_consecutive_seconds": 2 * 3600,
        "cooldown_seconds": 15 * 60,
        "degradation_cost": 1.00,
    },
    "EV": {
        "up_power_kw": 7.0,
        "down_power_kw": 7.0,
        "capacity_mwh": 0.060,
        "response_time_s": 4.0,
        "protocol": "ISO 15118 via simulated gateway",
        "max_consecutive_seconds": 1 * 3600,
        "cooldown_seconds": 15 * 60,
        "degradation_cost": 0.75,
    },
    "HVAC": {
        "up_power_kw": 4.6,
        "down_power_kw": 4.6,
        "capacity_mwh": 0.0,
        "response_time_s": 20.0,
        "protocol": "Gateway thermostat interface",
        "max_consecutive_seconds": 30 * 60,
        "cooldown_seconds": 15 * 60,
        "degradation_cost": 0.45,
    },
    "PV": {
        "up_power_kw": 0.0,
        "down_power_kw": 5.6,
        "capacity_mwh": 0.0,
        "response_time_s": 5.0,
        "protocol": "Inverter gateway interface",
        "max_consecutive_seconds": 2 * 3600,
        "cooldown_seconds": 15 * 60,
        "degradation_cost": 0.15,
    },
}


@dataclass(frozen=True)
class InnerControllerConfig:
    mode: str = "mpc"
    dt_seconds: int = 4
    horizon_seconds: int = 4
    rotation_strategy: str = "usage_aware"
    gateway_mode: str = "simulated_centralized"
    max_devices_per_type: int = 100000
    tracking_tolerance_pct: float = 0.10
    telemetry_error_pct: float = 0.004
    meter_quantization_kw: float = 0.1

    @property
    def horizon_steps(self) -> int:
        return max(1, int(round(self.horizon_seconds / max(self.dt_seconds, 1))))


def generate_household_device_roster(
    n_homes: int,
    ev_pen: float,
    bess_pen: float,
    pv_pen: float,
    hvac_pen: float,
    seed: int,
    hvac_response_s: float,
) -> pd.DataFrame:
    """Create a traceable synthetic household/appliance roster."""

    rng = np.random.default_rng(seed + 4471)
    homes = np.arange(1, int(n_homes) + 1)
    records: List[Dict[str, object]] = []

    def selected_homes(penetration: float) -> np.ndarray:
        count = int(round(len(homes) * float(np.clip(penetration, 0.0, 1.0))))
        if count <= 0:
            return np.array([], dtype=int)
        return np.sort(rng.choice(homes, size=count, replace=False))

    for device_type, chosen_homes in {
        "BESS": selected_homes(bess_pen),
        "EV": selected_homes(ev_pen),
        "PV": selected_homes(pv_pen),
        "HVAC": selected_homes(hvac_pen),
    }.items():
        defaults = dict(DEVICE_DEFAULTS[device_type])
        if device_type == "HVAC":
            defaults["response_time_s"] = float(hvac_response_s)
        for household_id in chosen_homes:
            device_id = f"h{int(household_id):05d}_{device_type.lower()}"
            gateway_id = f"gw_{(int(household_id) - 1) // 40 + 1:04d}"
            soc_fraction = float(rng.uniform(0.35, 0.85)) if defaults["capacity_mwh"] else 0.0
            records.append(
                {
                    "household_id": int(household_id),
                    "device_id": device_id,
                    "device_type": device_type,
                    "gateway_id": gateway_id,
                    "protocol": defaults["protocol"],
                    "up_power_kw": float(defaults["up_power_kw"]),
                    "down_power_kw": float(defaults["down_power_kw"]),
                    "capacity_mwh": float(defaults["capacity_mwh"]),
                    "state_mwh": float(defaults["capacity_mwh"] * soc_fraction),
                    "temp_delta_c": 0.0,
                    "response_time_s": float(defaults["response_time_s"]),
                    "max_consecutive_seconds": int(defaults["max_consecutive_seconds"]),
                    "cooldown_seconds_required": int(defaults["cooldown_seconds"]),
                    "degradation_cost": float(defaults["degradation_cost"]),
                    "online": True,
                    "availability_phase": float(rng.random()),
                    "usage_seconds": 0.0,
                    "consecutive_seconds": 0.0,
                    "cooldown_seconds": 0.0,
                    "throughput_kwh": 0.0,
                    "activation_count": 0,
                    "current_dispatch_kw": 0.0,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "household_id",
                "device_id",
                "device_type",
                "gateway_id",
                "protocol",
                "up_power_kw",
                "down_power_kw",
                "capacity_mwh",
                "state_mwh",
                "response_time_s",
            ]
        )
    return pd.DataFrame(records).sort_values(["household_id", "device_type"]).reset_index(drop=True)


def representative_roster_from_frame(df: pd.DataFrame, fleet_meta: Dict[str, float]) -> pd.DataFrame:
    """Build a small traceable roster when old callers do not provide one."""

    first = df.iloc[0]
    n_bess = max(int(round(float(first.get("bess_energy_cap_mwh", 0.0)) / DEVICE_DEFAULTS["BESS"]["capacity_mwh"])), 0)
    n_hvac = max(int(fleet_meta.get("n_hvac", 0)), 0)
    n_ev = max(int(round(float(first.get("ev_connected_count", 0.0)) / max(float(first.get("ev_connected_frac", 1.0)), 0.1))), 0)
    n_pv = max(int(round(float(first.get("pv_available_kw", 0.0)) / 2.5)), 0)
    n_homes = max(n_bess, n_hvac, n_ev, n_pv, 1)
    return generate_household_device_roster(
        n_homes=n_homes,
        ev_pen=min(n_ev / n_homes, 1.0),
        bess_pen=min(n_bess / n_homes, 1.0),
        pv_pen=min(n_pv / n_homes, 1.0),
        hvac_pen=min(n_hvac / n_homes, 1.0),
        seed=42,
        hvac_response_s=float(fleet_meta.get("hvac_response_s", DEVICE_DEFAULTS["HVAC"]["response_time_s"])),
    )


class CentralizedVPPController:
    """Centralized gateway-level 4-second MPC over selected household devices."""

    def __init__(
        self,
        device_roster: pd.DataFrame,
        fleet_meta: Dict[str, float],
        config: InnerControllerConfig | None = None,
    ) -> None:
        self.config = config or InnerControllerConfig()
        self.fleet_meta = dict(fleet_meta)
        self.devices = device_roster.copy()
        if self.devices.empty:
            self.devices = pd.DataFrame(columns=["device_id", "device_type"])
        if "device_id" not in self.devices:
            raise ValueError("Centralized controller requires a device_id column.")
        for column, default in {
            "usage_seconds": 0.0,
            "consecutive_seconds": 0.0,
            "cooldown_seconds": 0.0,
            "throughput_kwh": 0.0,
            "activation_count": 0,
            "current_dispatch_kw": 0.0,
            "temp_delta_c": 0.0,
            "online": True,
        }.items():
            if column not in self.devices:
                self.devices[column] = default
        self.devices = self.devices.set_index("device_id", drop=False)
        self.previous_pool_dispatch = {device_type: 0.0 for device_type in DEVICE_ORDER}
        self.pool_actual_dispatch = {device_type: 0.0 for device_type in DEVICE_ORDER}
        self.pool_actual_dispatch_fcr = {device_type: 0.0 for device_type in DEVICE_ORDER}
        self.pool_actual_dispatch_afrr = {device_type: 0.0 for device_type in DEVICE_ORDER}
        self.last_abs_tracking_error_kw = 0.0

    def execute_interval(
        self,
        row: pd.Series,
        fine_signals: pd.DataFrame,
        plan: Dict[str, float],
        interval_index: int,
        market_mode: str,
        resource_mode: str,
    ) -> tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        selected = self._select_upper_devices(row, plan, interval_index, resource_mode)
        self._derate_plan_to_selected(plan, selected)
        tracking_rows: List[Dict[str, float]] = []
        command_accumulator: Dict[str, Dict[str, object]] = {}
        outer_dt_seconds = int(round(float(row["dt_h"]) * 3600.0))

        for _, (ts, signal_row) in enumerate(fine_signals.iterrows()):
            tick_started = time.perf_counter()
            available = self._availability(row, selected.index)
            request_context = self._request_context(signal_row, plan, market_mode)
            request_now = float(request_context["request_kw"])
            committed_up_kw, committed_down_kw = self._commitment_limits(plan, market_mode)
            active_caps = self._pool_caps(selected, available, resource_mode, roles={"active"})
            buffer_caps = self._pool_caps(selected, available, resource_mode, roles={"buffer"})
            normal_solution = self._allocate_tracking_delta(
                signal_row,
                plan,
                active_caps,
                {device_type: {"up": 0.0, "down": 0.0} for device_type in DEVICE_ORDER},
                market_mode,
                recovery_mode=False,
                apply_state=False,
            )
            recovery_request_kw = float(normal_solution.get("afrr_request_kw", 0.0))
            recovery_predicted_kw = float(normal_solution.get("afrr_predicted_power_kw", 0.0))
            if abs(recovery_request_kw) <= 0.001:
                recovery_request_kw = request_now
                recovery_predicted_kw = float(normal_solution.get("predicted_power_kw", 0.0))
            normal_error = recovery_predicted_kw - recovery_request_kw
            tolerance_kw = max(abs(request_now) * float(self.config.tracking_tolerance_pct), 1.0)
            error_rising = abs(normal_error) >= self.last_abs_tracking_error_kw + 0.5
            recovery_mode = (
                abs(recovery_request_kw) > 0.001
                and abs(normal_error) > tolerance_kw
                and (error_rising or self.last_abs_tracking_error_kw > tolerance_kw)
            )
            solution = (
                self._allocate_tracking_delta(signal_row, plan, active_caps, buffer_caps, market_mode, recovery_mode=True, apply_state=True)
                if recovery_mode
                else self._allocate_tracking_delta(
                    signal_row,
                    plan,
                    active_caps,
                    {device_type: {"up": 0.0, "down": 0.0} for device_type in DEVICE_ORDER},
                    market_mode,
                    recovery_mode=False,
                    apply_state=True,
                )
            )

            if solution["status"] != "Optimal":
                delivered_by_type = {device_type: 0.0 for device_type in DEVICE_ORDER}
                command_by_type = {device_type: 0.0 for device_type in DEVICE_ORDER}
            else:
                delivered_by_type = solution["delivered_by_type"]
                command_by_type = solution["command_by_type"]

            commands = self._apportion_commands(selected, available, delivered_by_type, target_by_type=command_by_type)
            ideal_delivered = float(solution.get("predicted_power_kw", 0.0))
            measured_by_type = self._metered_delivery(ts, delivered_by_type, request_now)
            delivered = float(sum(measured_by_type.values()))
            if solution["status"] == "Optimal":
                self.pool_actual_dispatch.update(measured_by_type)
            self._apply_device_commands(commands, command_accumulator)
            tracking_error = delivered - request_now
            self.last_abs_tracking_error_kw = abs(tracking_error)
            control_latency_ms = (time.perf_counter() - tick_started) * 1000.0

            tracking_rows.append(
                {
                    "timestamp": ts,
                    "inner_solver_status": solution["status"],
                    "inner_controller_mode": self.config.mode,
                    "lower_tracking_mode": "proportional_buffer_tracker",
                    "fcr_control_source": "local_frequency_measurement",
                    "afrr_control_source": "tso_activation_signal",
                    "gateway_mode": self.config.gateway_mode,
                    "frequency_hz": float(signal_row["frequency_hz"]),
                    "fcr_signal_norm": float(signal_row["fcr_signal_norm"]),
                    "afrr_signal_norm": float(signal_row["afrr_signal_norm"]),
                    "committed_up_kw": committed_up_kw,
                    "committed_down_kw": committed_down_kw,
                    "raw_request_kw": float(request_context["raw_request_kw"]),
                    "fcr_local_request_kw": float(request_context["fcr_request_kw"]),
                    "afrr_signal_request_kw": float(request_context["afrr_request_kw"]),
                    "product_clamped_request_kw": float(request_context["product_clamped_request_kw"]),
                    "socket_up_kw": float(request_context["socket_up_kw"]),
                    "socket_down_kw": float(request_context["socket_down_kw"]),
                    "socket_violation_up_kw": float(request_context["socket_violation_up_kw"]),
                    "socket_violation_down_kw": float(request_context["socket_violation_down_kw"]),
                    "requested_signed_kw": request_now,
                    "requested_up_kw": max(request_now, 0.0),
                    "requested_down_kw": max(-request_now, 0.0),
                    "delivered_signed_kw": delivered,
                    "delivered_up_kw": max(delivered, 0.0),
                    "delivered_down_kw": max(-delivered, 0.0),
                    "fleet_power_before_kw": float(solution.get("actual_before_kw", 0.0)),
                    "fleet_power_after_kw": delivered,
                    "ideal_delivered_kw": ideal_delivered,
                    "fcr_delivered_kw": float(solution.get("fcr_predicted_power_kw", 0.0)),
                    "afrr_delivered_kw": float(solution.get("afrr_predicted_power_kw", 0.0)),
                    "telemetry_error_kw": delivered - ideal_delivered,
                    "target_command_kw": float(solution.get("target_command_kw", 0.0)),
                    "optimized_net_load_kw": float(row["net_load_baseline_kw"] - max(delivered, 0.0) + max(-delivered, 0.0)),
                    "baseline_net_load_kw": float(row["net_load_baseline_kw"]),
                    "tracking_delta_kw": float(solution.get("tracking_delta_kw", request_now - delivered)),
                    "tracking_error_kw": tracking_error,
                    "tracking_tolerance_kw": tolerance_kw,
                    "tracking_tolerance_pct": float(self.config.tracking_tolerance_pct),
                    "error_rising": bool(error_rising),
                    "recovery_mode": bool(recovery_mode),
                    "shortfall_kw": max(abs(request_now) - abs(delivered), 0.0),
                    "reserve_buffer_kw": float(plan.get("reserve_buffer_kw", 0.0)),
                    "fast_bridge_used_kw": float(solution.get("fast_bridge_used_kw", 0.0)),
                    "buffer_used_kw": float(solution.get("buffer_used_kw", 0.0)),
                    "active_capacity_kw": float(solution.get("active_capacity_kw", 0.0)),
                    "buffer_capacity_kw": float(solution.get("buffer_capacity_kw", 0.0)),
                    "active_selected_devices": int((selected.get("selection_role", pd.Series(dtype=object)) == "active").sum()),
                    "buffer_selected_devices": int((selected.get("selection_role", pd.Series(dtype=object)) == "buffer").sum()),
                    "control_latency_ms": float(control_latency_ms),
                    "control_deadline_ms": float(self.config.dt_seconds * 1000.0),
                    "control_deadline_met": bool(control_latency_ms <= self.config.dt_seconds * 1000.0),
                    "optimization_strategy": str(solution.get("optimization_strategy", "proportional_allocation")),
                    "bess_up_kw": max(measured_by_type.get("BESS", 0.0), 0.0),
                    "bess_down_kw": max(-measured_by_type.get("BESS", 0.0), 0.0),
                    "ev_up_kw": max(measured_by_type.get("EV", 0.0), 0.0),
                    "ev_down_kw": max(-measured_by_type.get("EV", 0.0), 0.0),
                    "hvac_up_kw": max(measured_by_type.get("HVAC", 0.0), 0.0),
                    "hvac_down_kw": max(-measured_by_type.get("HVAC", 0.0), 0.0),
                    "pv_down_kw": max(-measured_by_type.get("PV", 0.0), 0.0),
                    "selected_devices": int(len(selected)),
                    "inner_mpc_horizon_seconds": int(self.config.horizon_seconds),
                }
            )

        self._decrement_cooldowns(outer_dt_seconds)
        upper_schedule = selected.reset_index(drop=True)
        upper_schedule.insert(0, "timestamp", row.name)
        upper_schedule.insert(1, "outer_interval", interval_index)
        command_summary = pd.DataFrame(command_accumulator.values())
        if not command_summary.empty:
            command_summary.insert(0, "timestamp", row.name)
            command_summary.insert(1, "outer_interval", interval_index)

        tracking_df = pd.DataFrame(tracking_rows).set_index("timestamp")
        aggregated = self._aggregate_interval(row, tracking_df)
        return aggregated, tracking_df, upper_schedule, command_summary

    def select_upper_schedule(
        self,
        row: pd.Series,
        plan: Dict[str, float],
        interval_index: int,
        resource_mode: str,
    ) -> pd.DataFrame:
        selected = self._select_upper_devices(row, plan, interval_index, resource_mode)
        self._derate_plan_to_selected(plan, selected)
        if selected.empty:
            return selected.reset_index(drop=True)
        upper_schedule = selected.reset_index(drop=True)
        upper_schedule.insert(0, "timestamp", row.name)
        upper_schedule.insert(1, "outer_interval", interval_index)
        return upper_schedule

    def contribution_frames(self, command_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if command_summary is None or command_summary.empty:
            empty = pd.DataFrame()
            return empty, empty, self._fatigue_summary()
        appliance = (
            command_summary.groupby(["household_id", "device_id", "device_type", "gateway_id"], as_index=False)
            .agg(
                up_energy_kwh=("up_energy_kwh", "sum"),
                down_energy_kwh=("down_energy_kwh", "sum"),
                active_seconds=("active_seconds", "sum"),
                activation_count=("activation_count", "sum"),
                max_abs_command_kw=("max_abs_command_kw", "max"),
            )
            .assign(total_energy_kwh=lambda frame: frame["up_energy_kwh"] + frame["down_energy_kwh"])
            .sort_values("total_energy_kwh", ascending=False)
        )
        household = (
            appliance.groupby("household_id", as_index=False)
            .agg(
                up_energy_kwh=("up_energy_kwh", "sum"),
                down_energy_kwh=("down_energy_kwh", "sum"),
                total_energy_kwh=("total_energy_kwh", "sum"),
                active_seconds=("active_seconds", "sum"),
                appliance_count=("device_id", "nunique"),
            )
            .sort_values("total_energy_kwh", ascending=False)
        )
        return household, appliance, self._fatigue_summary()

    def _select_upper_devices(
        self,
        row: pd.Series,
        plan: Dict[str, float],
        interval_index: int,
        resource_mode: str,
    ) -> pd.DataFrame:
        if self.devices.empty:
            return self.devices.copy()
        availability = self._availability(row, self.devices.index)
        devices = self.devices.join(availability[["available_up_kw", "available_down_kw"]], how="left")
        devices[["available_up_kw", "available_down_kw"]] = devices[["available_up_kw", "available_down_kw"]].fillna(0.0)
        devices["usage_score"] = (
            devices["usage_seconds"] / np.maximum(devices["max_consecutive_seconds"], 1)
            + 0.02 * devices["activation_count"].astype(float)
            + 0.01 * devices["throughput_kwh"].astype(float)
            + np.where(devices["cooldown_seconds"] > 0, 100.0, 0.0)
        )

        active_requirements = self._requirements_by_type(plan, resource_mode)
        buffer_requirements = self._buffer_requirements_by_type(plan, active_requirements, resource_mode)
        selected_parts = []
        selected_ids: set[str] = set()

        def select_for_requirements(requirements: Dict[str, Dict[str, float]], role: str) -> None:
            for device_type in DEVICE_ORDER:
                req = requirements[device_type]
                if req["up"] <= 0.0 and req["down"] <= 0.0:
                    continue
                candidates = devices[
                    (devices["device_type"] == device_type)
                    & ~devices["device_id"].astype(str).isin(selected_ids)
                    & devices["online"].astype(bool)
                    & (devices["cooldown_seconds"].astype(float) <= 0.0)
                    & (devices["consecutive_seconds"].astype(float) < devices["max_consecutive_seconds"].astype(float))
                    & ((devices["available_up_kw"] > 0.001) | (devices["available_down_kw"] > 0.001))
                ].copy()
                if candidates.empty:
                    continue
                if role == "buffer":
                    candidates = candidates.sort_values(["response_time_s", "usage_score", "household_id"])
                else:
                    candidates = candidates.sort_values(["usage_score", "response_time_s", "household_id"])
                selected = []
                up_remaining = float(req["up"])
                down_remaining = float(req["down"])
                for _, candidate in candidates.iterrows():
                    if len(selected) >= self.config.max_devices_per_type:
                        break
                    up_cap = min(max(up_remaining, 0.0), float(candidate["available_up_kw"]))
                    down_cap = min(max(down_remaining, 0.0), float(candidate["available_down_kw"]))
                    if up_cap <= 0.001 and down_cap <= 0.001:
                        continue
                    row_payload = candidate.to_dict()
                    row_payload["upper_up_cap_kw"] = up_cap
                    row_payload["upper_down_cap_kw"] = down_cap
                    row_payload["upper_plan_up_kw"] = float(req["up"])
                    row_payload["upper_plan_down_kw"] = float(req["down"])
                    row_payload["selection_reason"] = self.config.rotation_strategy
                    row_payload["selection_role"] = role
                    selected.append(row_payload)
                    selected_ids.add(str(candidate["device_id"]))
                    up_remaining -= up_cap
                    down_remaining -= down_cap
                    if up_remaining <= 0.001 and down_remaining <= 0.001:
                        break
                if selected:
                    selected_parts.append(pd.DataFrame(selected))

        select_for_requirements(active_requirements, "active")
        select_for_requirements(buffer_requirements, "buffer")

        if not selected_parts:
            return pd.DataFrame(columns=list(self.devices.columns) + ["upper_up_cap_kw", "upper_down_cap_kw"])
        selected_df = pd.concat(selected_parts, ignore_index=True).set_index("device_id", drop=False)
        return selected_df

    def _derate_plan_to_selected(self, plan: Dict[str, float], selected: pd.DataFrame) -> None:
        if selected.empty:
            for key in [
                "bess_fcr_kw",
                "ev_fcr_kw",
                "hvac_fcr_kw",
                "bess_afrr_up_kw",
                "bess_afrr_down_kw",
                "ev_afrr_up_kw",
                "ev_afrr_down_kw",
                "hvac_afrr_up_kw",
                "hvac_afrr_down_kw",
                "pv_afrr_down_kw",
            ]:
                plan[key] = 0.0
            return
        active = selected[selected.get("selection_role", "active") == "active"] if "selection_role" in selected else selected
        caps = {
            device_type: {
                "up": float(active.loc[active["device_type"] == device_type, "upper_up_cap_kw"].sum()),
                "down": float(active.loc[active["device_type"] == device_type, "upper_down_cap_kw"].sum()),
            }
            for device_type in DEVICE_ORDER
        }

        for prefix, device_type in [("bess", "BESS"), ("ev", "EV"), ("hvac", "HVAC")]:
            fcr_key = f"{prefix}_fcr_kw"
            up_key = f"{prefix}_afrr_up_kw"
            down_key = f"{prefix}_afrr_down_kw"
            fcr = float(plan.get(fcr_key, 0.0))
            up = float(plan.get(up_key, 0.0))
            down = float(plan.get(down_key, 0.0))
            up_req = fcr + up
            down_req = fcr + down
            ratio = 1.0
            if up_req > 0.001:
                ratio = min(ratio, caps[device_type]["up"] / up_req)
            if down_req > 0.001:
                ratio = min(ratio, caps[device_type]["down"] / down_req)
            ratio = float(np.clip(ratio, 0.0, 1.0))
            plan[fcr_key] = fcr * ratio
            plan[up_key] = up * ratio
            plan[down_key] = down * ratio

        pv_down = float(plan.get("pv_afrr_down_kw", 0.0))
        if pv_down > 0.001:
            plan["pv_afrr_down_kw"] = min(pv_down, caps["PV"]["down"])
        self._snap_derated_plan_to_market_segments(plan)

    def _floor_market_segment(self, total_kw: float, min_kw: float, granularity_kw: float) -> float:
        total = float(total_kw)
        if total + 1e-6 < min_kw:
            return total
        segments = int(np.floor((total + 1e-6) / max(granularity_kw, 1e-9)))
        return float(max(segments, 0) * granularity_kw)

    def _scale_plan_keys(self, plan: Dict[str, float], keys: List[str], target_total_kw: float) -> None:
        current_total = float(sum(float(plan.get(key, 0.0)) for key in keys))
        if current_total <= 1e-9:
            for key in keys:
                plan[key] = 0.0
            return
        ratio = float(np.clip(target_total_kw / current_total, 0.0, 1.0))
        for key in keys:
            plan[key] = float(plan.get(key, 0.0)) * ratio

    def _enforce_fcr_hvac_share_policy(self, plan: Dict[str, float]) -> None:
        share_cap = float(np.clip(plan.get("fcr_hvac_share_cap", FCR_N_HVAC_SHARE_CAP), 0.0, 1.0))
        if share_cap >= 0.999:
            return
        fast_fcr = float(plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0))
        max_hvac_fcr = 0.0 if share_cap <= 0.0 else fast_fcr * share_cap / max(1.0 - share_cap, 1e-9)
        plan["hvac_fcr_kw"] = min(float(plan.get("hvac_fcr_kw", 0.0)), max_hvac_fcr)

    def _snap_derated_plan_to_market_segments(self, plan: Dict[str, float]) -> None:
        """Keep selected-roster derating from creating non-biddable aggregates."""

        fcr_keys = ["bess_fcr_kw", "ev_fcr_kw", "hvac_fcr_kw"]
        afrr_up_keys = ["bess_afrr_up_kw", "ev_afrr_up_kw", "hvac_afrr_up_kw"]
        afrr_down_keys = ["bess_afrr_down_kw", "ev_afrr_down_kw", "hvac_afrr_down_kw", "pv_afrr_down_kw"]

        self._enforce_fcr_hvac_share_policy(plan)
        fcr_target = self._floor_market_segment(sum(float(plan.get(key, 0.0)) for key in fcr_keys), FCR_N_MIN_BID_KW, FCR_N_BID_GRANULARITY_KW)
        afrr_up_target = self._floor_market_segment(sum(float(plan.get(key, 0.0)) for key in afrr_up_keys), AFRR_MIN_BID_KW, AFRR_BID_GRANULARITY_KW)
        afrr_down_target = self._floor_market_segment(sum(float(plan.get(key, 0.0)) for key in afrr_down_keys), AFRR_MIN_BID_KW, AFRR_BID_GRANULARITY_KW)

        self._scale_plan_keys(plan, fcr_keys, fcr_target)
        self._scale_plan_keys(plan, afrr_up_keys, afrr_up_target)
        self._scale_plan_keys(plan, afrr_down_keys, afrr_down_target)
        plan["fcr_bid_kw"] = float(fcr_target)
        plan["afrr_up_bid_kw"] = float(afrr_up_target)
        plan["afrr_down_bid_kw"] = float(afrr_down_target)
        plan["fcr_bid_mw"] = float(fcr_target / 1000.0)
        plan["afrr_up_bid_mw"] = float(afrr_up_target / 1000.0)
        plan["afrr_down_bid_mw"] = float(afrr_down_target / 1000.0)
        plan["fcr_bid_segments"] = int(round(fcr_target / FCR_N_BID_GRANULARITY_KW)) if fcr_target >= FCR_N_MIN_BID_KW else 0
        plan["afrr_up_segments"] = int(round(afrr_up_target / AFRR_BID_GRANULARITY_KW)) if afrr_up_target >= AFRR_MIN_BID_KW else 0
        plan["afrr_down_segments"] = int(round(afrr_down_target / AFRR_BID_GRANULARITY_KW)) if afrr_down_target >= AFRR_MIN_BID_KW else 0

    def _requirements_by_type(self, plan: Dict[str, float], resource_mode: str) -> Dict[str, Dict[str, float]]:
        requirements = {
            "BESS": {
                "up": float(plan.get("bess_fcr_kw", 0.0) + plan.get("bess_afrr_up_kw", 0.0)),
                "down": float(plan.get("bess_fcr_kw", 0.0) + plan.get("bess_afrr_down_kw", 0.0)),
            },
            "EV": {
                "up": float(plan.get("ev_fcr_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0)),
                "down": float(plan.get("ev_fcr_kw", 0.0) + plan.get("ev_afrr_down_kw", 0.0)),
            },
            "HVAC": {
                "up": float(plan.get("hvac_fcr_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0)),
                "down": float(plan.get("hvac_fcr_kw", 0.0) + plan.get("hvac_afrr_down_kw", 0.0)),
            },
            "PV": {"up": 0.0, "down": float(plan.get("pv_afrr_down_kw", 0.0))},
        }
        if resource_mode == "Fast only":
            requirements["HVAC"] = {"up": 0.0, "down": 0.0}
            requirements["PV"] = {"up": 0.0, "down": 0.0}
        else:
            hvac_alpha = self._response_alpha("HVAC")
            slow_up_bridge = max(requirements["HVAC"]["up"] * (1.0 - hvac_alpha), 0.0)
            slow_down_bridge = max(requirements["HVAC"]["down"] * (1.0 - hvac_alpha) + requirements["PV"]["down"] * 0.50, 0.0)
            for device_type, weight in {"BESS": 0.65, "EV": 0.35}.items():
                requirements[device_type]["up"] += slow_up_bridge * weight
            for device_type, weight in {"BESS": 0.55, "EV": 0.45}.items():
                requirements[device_type]["down"] += slow_down_bridge * weight
        return requirements

    def _buffer_requirements_by_type(
        self,
        plan: Dict[str, float],
        active_requirements: Dict[str, Dict[str, float]],
        resource_mode: str,
    ) -> Dict[str, Dict[str, float]]:
        requirements = {device_type: {"up": 0.0, "down": 0.0} for device_type in DEVICE_ORDER}
        buffer_kw = max(float(plan.get("reserve_buffer_kw", 0.0)), 0.0)
        if buffer_kw <= 0.0:
            return requirements
        up_total = sum(req["up"] for req in active_requirements.values())
        down_total = sum(req["down"] for req in active_requirements.values())
        allowed_up = ["BESS", "EV", "HVAC"]
        allowed_down = ["BESS", "EV", "PV", "HVAC"]
        if resource_mode == "Fast only":
            allowed_up = ["BESS", "EV"]
            allowed_down = ["BESS", "EV"]

        def assign(direction: str, allowed: List[str], total: float) -> None:
            if total <= 0.0:
                return
            weights = {
                device_type: 1.0 / max(
                    float(self.fleet_meta.get("hvac_response_s", DEVICE_DEFAULTS[device_type]["response_time_s"]))
                    if device_type == "HVAC"
                    else float(DEVICE_DEFAULTS[device_type]["response_time_s"]),
                    1e-6,
                )
                for device_type in allowed
            }
            weight_sum = sum(weights.values())
            for device_type, weight in weights.items():
                requirements[device_type][direction] = buffer_kw * weight / max(weight_sum, 1e-9)

        assign("up", allowed_up, up_total)
        assign("down", allowed_down, down_total)
        return requirements

    def _availability(self, row: pd.Series, device_ids: Iterable[str]) -> pd.DataFrame:
        device_ids = list(device_ids)
        if not device_ids:
            return pd.DataFrame(columns=["available_up_kw", "available_down_kw"])
        devices = self.devices.loc[device_ids].copy()
        total_power = self.devices.groupby("device_type")[["up_power_kw", "down_power_kw"]].sum().replace(0.0, np.nan)

        def total_for(device_type: str, column: str) -> float:
            try:
                value = float(total_power.loc[device_type, column])
            except (KeyError, TypeError, ValueError):
                value = 0.0
            return value if np.isfinite(value) and value > 0.0 else 1.0

        up_values = []
        down_values = []
        connected_values = []
        for _, device in devices.iterrows():
            device_type = str(device["device_type"])
            up_factor = 1.0
            down_factor = 1.0
            connected = bool(device.get("online", True))
            if device_type == "BESS":
                up_factor = float(row.get("bess_up_kw", 0.0)) / total_for("BESS", "up_power_kw")
                down_factor = float(row.get("bess_down_kw", 0.0)) / total_for("BESS", "down_power_kw")
                cap = max(float(device["capacity_mwh"]), 1e-6)
                energy_up = max((float(device["state_mwh"]) - 0.10 * cap) * 1000.0 * 0.95, 0.0)
                energy_down = max((0.95 * cap - float(device["state_mwh"])) * 1000.0 / 0.95, 0.0)
                up_values.append(min(float(device["up_power_kw"]) * up_factor, energy_up))
                down_values.append(min(float(device["down_power_kw"]) * down_factor, energy_down))
            elif device_type == "EV":
                connected = connected and float(device.get("availability_phase", 0.0)) <= float(row.get("ev_connected_frac", 1.0))
                up_factor = float(row.get("ev_up_kw", 0.0)) / total_for("EV", "up_power_kw")
                down_factor = float(row.get("ev_down_kw", 0.0)) / total_for("EV", "down_power_kw")
                cap = max(float(device["capacity_mwh"]), 1e-6)
                energy_up = max((float(device["state_mwh"]) - 0.20 * cap) * 1000.0 * 0.95, 0.0)
                energy_down = max((0.95 * cap - float(device["state_mwh"])) * 1000.0 / 0.95, 0.0)
                up_values.append(min(float(device["up_power_kw"]) * up_factor, energy_up) if connected else 0.0)
                down_values.append(min(float(device["down_power_kw"]) * down_factor, energy_down) if connected else 0.0)
            elif device_type == "HVAC":
                up_factor = float(row.get("hvac_up_kw", 0.0)) / total_for("HVAC", "up_power_kw")
                down_factor = float(row.get("hvac_down_kw", 0.0)) / total_for("HVAC", "down_power_kw")
                up_values.append(float(device["up_power_kw"]) * up_factor if connected else 0.0)
                down_values.append(float(device["down_power_kw"]) * down_factor if connected else 0.0)
            else:
                down_factor = float(row.get("pv_down_kw", 0.0)) / total_for("PV", "down_power_kw")
                up_values.append(0.0)
                down_values.append(float(device["down_power_kw"]) * down_factor if connected else 0.0)
            connected_values.append(connected)

        return pd.DataFrame(
            {
                "device_id": device_ids,
                "available_up_kw": np.clip(up_values, 0.0, None),
                "available_down_kw": np.clip(down_values, 0.0, None),
                "connected": connected_values,
            }
        ).set_index("device_id")

    def _pool_caps(
        self,
        selected: pd.DataFrame,
        available: pd.DataFrame,
        resource_mode: str,
        roles: set[str] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        caps = {device_type: {"up": 0.0, "down": 0.0} for device_type in DEVICE_ORDER}
        if selected.empty:
            return caps
        working = selected.join(available[["available_up_kw", "available_down_kw"]], rsuffix="_now")
        if roles is not None and "selection_role" in working:
            working = working[working["selection_role"].isin(roles)]
        for device_type, group in working.groupby("device_type"):
            if resource_mode == "Fast only" and device_type in {"HVAC", "PV"}:
                continue
            caps[device_type]["up"] = float(np.minimum(group["upper_up_cap_kw"], group["available_up_kw_now"]).sum())
            caps[device_type]["down"] = float(np.minimum(group["upper_down_cap_kw"], group["available_down_kw_now"]).sum())
        return caps

    def _request_context(self, signal_row: pd.Series, plan: Dict[str, float], market_mode: str) -> Dict[str, float]:
        fcr_signal = float(signal_row["fcr_signal_norm"]) if market_mode in {"FCR-N", "Combined"} else 0.0
        afrr_signal = float(signal_row["afrr_signal_norm"]) if market_mode in {"aFRR", "Combined"} else 0.0
        up_limit, down_limit = self._commitment_limits(plan, market_mode)
        raw_fcr_request = fcr_signal * self._fcr_bid_kw(plan)
        raw_afrr_request = max(afrr_signal, 0.0) * self._afrr_up_bid_kw(plan) - max(-afrr_signal, 0.0) * self._afrr_down_bid_kw(plan)
        raw_request = raw_fcr_request + raw_afrr_request
        clamped = float(np.clip(raw_request, -max(down_limit, 0.0), max(up_limit, 0.0)))
        socket_up = max(float(plan.get("socket_up_kw", up_limit)), 0.0)
        socket_down = max(float(plan.get("socket_down_kw", down_limit)), 0.0)
        request = float(np.clip(clamped, -socket_down, socket_up))
        split_scale = request / raw_request if abs(raw_request) > 1e-9 else 0.0
        fcr_request = raw_fcr_request * split_scale
        afrr_request = raw_afrr_request * split_scale
        return {
            "raw_request_kw": float(raw_request),
            "raw_fcr_request_kw": float(raw_fcr_request),
            "raw_afrr_request_kw": float(raw_afrr_request),
            "product_clamped_request_kw": clamped,
            "request_kw": request,
            "fcr_request_kw": float(fcr_request),
            "afrr_request_kw": float(afrr_request),
            "socket_up_kw": socket_up,
            "socket_down_kw": socket_down,
            "socket_violation_up_kw": max(clamped - socket_up, 0.0),
            "socket_violation_down_kw": max(-socket_down - clamped, 0.0),
        }

    def _request_kw(self, signal_row: pd.Series, plan: Dict[str, float], market_mode: str) -> float:
        return float(self._request_context(signal_row, plan, market_mode)["request_kw"])

    def _fcr_bid_kw(self, plan: Dict[str, float]) -> float:
        return float(plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0) + plan.get("hvac_fcr_kw", 0.0))

    def _afrr_up_bid_kw(self, plan: Dict[str, float]) -> float:
        return float(plan.get("bess_afrr_up_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0))

    def _afrr_down_bid_kw(self, plan: Dict[str, float]) -> float:
        return float(
            plan.get("bess_afrr_down_kw", 0.0)
            + plan.get("ev_afrr_down_kw", 0.0)
            + plan.get("hvac_afrr_down_kw", 0.0)
            + plan.get("pv_afrr_down_kw", 0.0)
        )

    def _truthy(self, value: object, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "ok"}
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        return bool(value)

    def _combined_split_capacity_enabled(self, plan: Dict[str, float]) -> bool:
        return (
            str(plan.get("combined_stacking_model", "")).strip() == COMBINED_STACKING_SPLIT_MODEL
            and self._truthy(plan.get("combined_shared_capacity_ok", plan.get("combined_split_capacity_ok", True)), default=True)
        )

    def _commitment_limits(self, plan: Dict[str, float], market_mode: str) -> tuple[float, float]:
        fcr_bid = float(plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0) + plan.get("hvac_fcr_kw", 0.0))
        afrr_up = float(plan.get("bess_afrr_up_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0))
        afrr_down = float(
            plan.get("bess_afrr_down_kw", 0.0)
            + plan.get("ev_afrr_down_kw", 0.0)
            + plan.get("hvac_afrr_down_kw", 0.0)
            + plan.get("pv_afrr_down_kw", 0.0)
        )
        if market_mode == "FCR-N":
            up_limit = fcr_bid
            down_limit = fcr_bid
        elif market_mode == "aFRR":
            up_limit = afrr_up
            down_limit = afrr_down
        else:
            if self._combined_split_capacity_enabled(plan):
                up_limit = fcr_bid + afrr_up
                down_limit = fcr_bid + afrr_down
            else:
                up_limit = max(fcr_bid, afrr_up)
                down_limit = max(fcr_bid, afrr_down)
        return float(max(up_limit, 0.0)), float(max(down_limit, 0.0))

    def _response_alpha(self, device_type: str) -> float:
        response_seconds = float(DEVICE_DEFAULTS[device_type]["response_time_s"])
        if device_type == "HVAC":
            response_seconds = float(self.fleet_meta.get("hvac_response_s", response_seconds))
        return float(np.clip(self.config.dt_seconds / max(response_seconds, 1e-6), 0.0, 1.0))

    def _plan_requirement_weights(self, plan: Dict[str, float], market_mode: str, direction: str) -> Dict[str, float]:
        fcr_active = market_mode in {"FCR-N", "Combined"}
        afrr_active = market_mode in {"aFRR", "Combined"}
        if direction == "up":
            return {
                "BESS": (float(plan.get("bess_fcr_kw", 0.0)) if fcr_active else 0.0) + (float(plan.get("bess_afrr_up_kw", 0.0)) if afrr_active else 0.0),
                "EV": (float(plan.get("ev_fcr_kw", 0.0)) if fcr_active else 0.0) + (float(plan.get("ev_afrr_up_kw", 0.0)) if afrr_active else 0.0),
                "HVAC": (float(plan.get("hvac_fcr_kw", 0.0)) if fcr_active else 0.0) + (float(plan.get("hvac_afrr_up_kw", 0.0)) if afrr_active else 0.0),
                "PV": 0.0,
            }
        return {
            "BESS": (float(plan.get("bess_fcr_kw", 0.0)) if fcr_active else 0.0) + (float(plan.get("bess_afrr_down_kw", 0.0)) if afrr_active else 0.0),
            "EV": (float(plan.get("ev_fcr_kw", 0.0)) if fcr_active else 0.0) + (float(plan.get("ev_afrr_down_kw", 0.0)) if afrr_active else 0.0),
            "HVAC": (float(plan.get("hvac_fcr_kw", 0.0)) if fcr_active else 0.0) + (float(plan.get("hvac_afrr_down_kw", 0.0)) if afrr_active else 0.0),
            "PV": float(plan.get("pv_afrr_down_kw", 0.0)) if afrr_active else 0.0,
        }

    def _proportional_commands(
        self,
        requested_abs_kw: float,
        caps: Dict[str, Dict[str, float]],
        direction: str,
        weights: Dict[str, float],
        order: List[str] | None = None,
    ) -> Dict[str, float]:
        commands = {device_type: 0.0 for device_type in DEVICE_ORDER}
        remaining = max(float(requested_abs_kw), 0.0)
        cap_by_type = {device_type: max(float(caps[device_type][direction]), 0.0) for device_type in DEVICE_ORDER}
        active_types = [device_type for device_type in DEVICE_ORDER if cap_by_type[device_type] > 0.001]
        if remaining <= 0.001 or not active_types:
            return commands
        if order:
            for device_type in order:
                if remaining <= 0.001:
                    break
                take = min(cap_by_type.get(device_type, 0.0), remaining)
                commands[device_type] += take
                remaining -= take
            return commands

        weighted_types = [device_type for device_type in active_types if weights.get(device_type, 0.0) > 0.0]
        if not weighted_types:
            weighted_types = active_types
            weights = {device_type: cap_by_type[device_type] for device_type in active_types}

        open_types = set(weighted_types)
        while remaining > 0.001 and open_types:
            total_weight = sum(max(float(weights.get(device_type, 0.0)), 0.0) for device_type in open_types)
            if total_weight <= 0.0:
                total_weight = sum(cap_by_type[device_type] - commands[device_type] for device_type in open_types)
                local_weights = {device_type: cap_by_type[device_type] - commands[device_type] for device_type in open_types}
            else:
                local_weights = weights
            allocated_this_round = 0.0
            saturated = []
            for device_type in list(open_types):
                headroom = cap_by_type[device_type] - commands[device_type]
                if headroom <= 0.001:
                    saturated.append(device_type)
                    continue
                share = remaining * max(float(local_weights.get(device_type, 0.0)), 0.0) / max(total_weight, 1e-9)
                take = min(headroom, share)
                commands[device_type] += take
                allocated_this_round += take
                if headroom - take <= 0.001:
                    saturated.append(device_type)
            remaining -= allocated_this_round
            for device_type in saturated:
                open_types.discard(device_type)
            if allocated_this_round <= 0.001:
                break
        return commands

    def _delivered_from_commands(self, command_by_type: Dict[str, float], actual_before_by_type: Dict[str, float]) -> Dict[str, float]:
        delivered = {}
        for device_type in DEVICE_ORDER:
            alpha = self._response_alpha(device_type)
            command = float(command_by_type.get(device_type, 0.0))
            actual_before = float(actual_before_by_type.get(device_type, 0.0))
            delivered[device_type] = actual_before + alpha * (command - actual_before)
        return delivered

    def _metered_delivery(self, timestamp: pd.Timestamp, delivered_by_type: Dict[str, float], request_now: float) -> Dict[str, float]:
        if abs(float(request_now)) <= 1e-9:
            return {device_type: 0.0 for device_type in DEVICE_ORDER}
        ts = pd.Timestamp(timestamp)
        seconds = ts.hour * 3600.0 + ts.minute * 60.0 + ts.second + ts.microsecond / 1_000_000.0
        base_error = max(float(self.config.telemetry_error_pct), 0.0)
        quantization = max(float(self.config.meter_quantization_kw), 0.0)
        response_weight = {"BESS": 0.55, "EV": 0.80, "HVAC": 1.45, "PV": 0.95}
        measured: Dict[str, float] = {}
        for idx, device_type in enumerate(DEVICE_ORDER, start=1):
            value = float(delivered_by_type.get(device_type, 0.0))
            if abs(value) <= 1e-9:
                measured[device_type] = 0.0
                continue
            ripple = 0.65 * np.sin(seconds / 19.0 + idx * 0.91) + 0.35 * np.cos(seconds / 43.0 + idx * 1.73)
            factor = 1.0 + base_error * response_weight[device_type] * ripple
            metered = value * factor
            if quantization > 0.0:
                metered = round(metered / quantization) * quantization
            measured[device_type] = float(metered)
        return measured

    def _allocate_tracking_delta(
        self,
        signal_row: pd.Series,
        plan: Dict[str, float],
        active_caps: Dict[str, Dict[str, float]],
        buffer_caps: Dict[str, Dict[str, float]],
        market_mode: str,
        recovery_mode: bool,
        apply_state: bool = False,
    ) -> Dict[str, object]:
        actual_before_by_type = {device_type: float(self.pool_actual_dispatch.get(device_type, 0.0)) for device_type in DEVICE_ORDER}
        request_context = self._request_context(signal_row, plan, market_mode)
        request_now = float(request_context["request_kw"])
        fcr_request_kw = float(request_context["fcr_request_kw"])
        afrr_request_kw = float(request_context["afrr_request_kw"])
        direction = "up" if request_now >= 0.0 else "down"
        sign = 1.0 if direction == "up" else -1.0
        request_abs = abs(request_now)
        plan_weights = self._plan_requirement_weights(plan, market_mode, direction)

        def split_predicted(predicted_kw: float) -> tuple[float, float]:
            request_total = fcr_request_kw + afrr_request_kw
            if abs(request_total) <= 1e-9:
                return 0.0, 0.0
            return (
                float(predicted_kw * fcr_request_kw / request_total),
                float(predicted_kw * afrr_request_kw / request_total),
            )

        active_abs = self._proportional_commands(request_abs, active_caps, direction, plan_weights)
        buffer_abs = {device_type: 0.0 for device_type in DEVICE_ORDER}
        command_by_type = {device_type: sign * active_abs[device_type] for device_type in DEVICE_ORDER}
        delivered_by_type = self._delivered_from_commands(command_by_type, actual_before_by_type)
        predicted_power_kw = sum(delivered_by_type.values())
        fcr_predicted_power_kw, afrr_predicted_power_kw = split_predicted(predicted_power_kw)
        fast_bridge_used_kw = 0.0

        if request_abs > 0.001 and np.sign(request_now - predicted_power_kw) == sign:
            bridge_remaining = abs(request_now - predicted_power_kw)
            fast_order = ["BESS", "EV"] if direction == "up" else ["BESS", "EV", "PV"]
            for device_type in fast_order:
                headroom = max(float(active_caps[device_type][direction]) - active_abs[device_type], 0.0)
                if headroom <= 0.001:
                    continue
                alpha = max(self._response_alpha(device_type), 1e-6)
                command_kw = min(headroom, bridge_remaining / alpha)
                if command_kw <= 0.001:
                    continue
                active_abs[device_type] += command_kw
                command_by_type[device_type] += sign * command_kw
                fast_bridge_used_kw += command_kw
                bridge_remaining -= alpha * command_kw
                if bridge_remaining <= 0.001:
                    break
            if fast_bridge_used_kw > 0.001:
                delivered_by_type = self._delivered_from_commands(command_by_type, actual_before_by_type)
                predicted_power_kw = sum(delivered_by_type.values())
                fcr_predicted_power_kw, afrr_predicted_power_kw = split_predicted(predicted_power_kw)

        residual_kw = request_now - predicted_power_kw
        if recovery_mode and abs(residual_kw) > 0.001 and np.sign(residual_kw) == sign:
            remaining_response_kw = abs(residual_kw)
            for device_type in RECOVERY_PRIORITY:
                alpha = max(self._response_alpha(device_type), 1e-6)
                available_command_kw = max(float(buffer_caps[device_type][direction]), 0.0)
                command_kw = min(available_command_kw, remaining_response_kw / alpha)
                if command_kw <= 0.001:
                    continue
                buffer_abs[device_type] = command_kw
                remaining_response_kw -= alpha * command_kw
                if remaining_response_kw <= 0.001:
                    break
            for device_type in DEVICE_ORDER:
                command_by_type[device_type] += sign * buffer_abs[device_type]
            delivered_by_type = self._delivered_from_commands(command_by_type, actual_before_by_type)
            predicted_power_kw = sum(delivered_by_type.values())
            fcr_predicted_power_kw, afrr_predicted_power_kw = split_predicted(predicted_power_kw)

        if apply_state:
            for device_type in DEVICE_ORDER:
                self.previous_pool_dispatch[device_type] = command_by_type[device_type]
                self.pool_actual_dispatch[device_type] = delivered_by_type[device_type]
        target_command_kw = sum(command_by_type.values())
        return {
            "status": "Optimal",
            "command_by_type": command_by_type,
            "delivered_by_type": delivered_by_type,
            "actual_before_kw": sum(actual_before_by_type.values()),
            "predicted_power_kw": predicted_power_kw,
            "fcr_request_kw": fcr_request_kw,
            "afrr_request_kw": afrr_request_kw,
            "fcr_predicted_power_kw": fcr_predicted_power_kw,
            "afrr_predicted_power_kw": afrr_predicted_power_kw,
            "target_command_kw": target_command_kw,
            "tracking_delta_kw": request_now - sum(actual_before_by_type.values()),
            "fast_bridge_used_kw": float(fast_bridge_used_kw),
            "buffer_used_kw": sum(buffer_abs.values()),
            "active_capacity_kw": sum(active_caps[device_type][direction] for device_type in DEVICE_ORDER),
            "buffer_capacity_kw": sum(buffer_caps[device_type][direction] for device_type in DEVICE_ORDER),
            "optimization_strategy": "proportional_buffer_recovery" if recovery_mode else "proportional_active_tracking",
        }

    def _apportion_commands(
        self,
        selected: pd.DataFrame,
        available: pd.DataFrame,
        dispatch_by_type: Dict[str, float],
        target_by_type: Dict[str, float] | None = None,
    ) -> pd.DataFrame:
        records = []
        if selected.empty:
            return pd.DataFrame(columns=["device_id", "command_kw"])
        working = selected.join(available[["available_up_kw", "available_down_kw"]], rsuffix="_now")
        for device_type, command_kw in dispatch_by_type.items():
            if abs(command_kw) <= 1e-6:
                continue
            direction = "up" if command_kw > 0 else "down"
            remaining = abs(float(command_kw))
            cap_col = "available_up_kw_now" if direction == "up" else "available_down_kw_now"
            group = working[working["device_type"] == device_type].copy()
            if "selection_role" in group:
                group["role_order"] = np.where(group["selection_role"] == "active", 0, 1)
                group = group.sort_values(["role_order", "usage_score", "household_id"])
            else:
                group = group.sort_values(["usage_score", "household_id"])
            target_ratio = 1.0
            if target_by_type is not None and abs(command_kw) > 1e-6:
                target_ratio = float(target_by_type.get(device_type, command_kw)) / float(command_kw)
            for _, device in group.iterrows():
                cap = min(float(device.get(cap_col, 0.0)), float(device.get(f"upper_{direction}_cap_kw", 0.0)))
                if cap <= 0.001 or remaining <= 0.001:
                    continue
                dispatched = min(cap, remaining)
                signed_dispatch = dispatched if direction == "up" else -dispatched
                records.append(
                    {
                        **device.to_dict(),
                        "command_kw": signed_dispatch,
                        "target_command_kw": signed_dispatch * target_ratio,
                    }
                )
                remaining -= dispatched
                if remaining <= 0.001:
                    break
        if not records:
            return pd.DataFrame(columns=list(selected.columns) + ["command_kw"])
        return pd.DataFrame(records)

    def _apply_device_commands(self, commands: pd.DataFrame, accumulator: Dict[str, Dict[str, object]]) -> None:
        dt_h = self.config.dt_seconds / 3600.0
        active_devices = set()
        if commands is not None and not commands.empty:
            for _, command in commands.iterrows():
                device_id = str(command["device_id"])
                command_kw = float(command["command_kw"])
                if device_id not in self.devices.index:
                    continue
                active_devices.add(device_id)
                device_type = str(command["device_type"])
                up_kw = max(command_kw, 0.0)
                down_kw = max(-command_kw, 0.0)
                if device_type in {"BESS", "EV"}:
                    cap = max(float(self.devices.at[device_id, "capacity_mwh"]), 1e-6)
                    state = float(self.devices.at[device_id, "state_mwh"])
                    state += dt_h / 1000.0 * (0.95 * down_kw - up_kw / 0.95)
                    self.devices.at[device_id, "state_mwh"] = float(np.clip(state, 0.05 * cap, 0.98 * cap))
                elif device_type == "HVAC":
                    self.devices.at[device_id, "temp_delta_c"] = float(
                        0.995 * float(self.devices.at[device_id, "temp_delta_c"]) + 0.16 * dt_h * (down_kw - up_kw)
                    )
                self.devices.at[device_id, "current_dispatch_kw"] = command_kw
                self.devices.at[device_id, "usage_seconds"] = float(self.devices.at[device_id, "usage_seconds"]) + self.config.dt_seconds
                self.devices.at[device_id, "consecutive_seconds"] = (
                    float(self.devices.at[device_id, "consecutive_seconds"]) + self.config.dt_seconds
                )
                self.devices.at[device_id, "throughput_kwh"] = float(self.devices.at[device_id, "throughput_kwh"]) + abs(command_kw) * dt_h
                self.devices.at[device_id, "activation_count"] = int(self.devices.at[device_id, "activation_count"]) + 1
                if float(self.devices.at[device_id, "consecutive_seconds"]) >= float(self.devices.at[device_id, "max_consecutive_seconds"]):
                    self.devices.at[device_id, "cooldown_seconds"] = float(self.devices.at[device_id, "cooldown_seconds_required"])
                    self.devices.at[device_id, "consecutive_seconds"] = 0.0

                payload = accumulator.setdefault(
                    device_id,
                    {
                        "household_id": int(command["household_id"]),
                        "device_id": device_id,
                        "device_type": device_type,
                        "gateway_id": str(command["gateway_id"]),
                        "protocol": str(command.get("protocol", "")),
                        "selection_role": str(command.get("selection_role", "active")),
                        "up_energy_kwh": 0.0,
                        "down_energy_kwh": 0.0,
                        "active_seconds": 0.0,
                        "activation_count": 0,
                        "max_abs_command_kw": 0.0,
                    },
                )
                payload["up_energy_kwh"] = float(payload["up_energy_kwh"]) + up_kw * dt_h
                payload["down_energy_kwh"] = float(payload["down_energy_kwh"]) + down_kw * dt_h
                payload["active_seconds"] = float(payload["active_seconds"]) + self.config.dt_seconds
                payload["activation_count"] = int(payload["activation_count"]) + 1
                payload["max_abs_command_kw"] = max(float(payload["max_abs_command_kw"]), abs(command_kw))

        inactive = self.devices.index.difference(list(active_devices))
        if len(inactive):
            self.devices.loc[inactive, "current_dispatch_kw"] = 0.0
            self.devices.loc[inactive, "consecutive_seconds"] = 0.0

    def _decrement_cooldowns(self, elapsed_seconds: int) -> None:
        if "cooldown_seconds" in self.devices:
            self.devices["cooldown_seconds"] = np.maximum(self.devices["cooldown_seconds"].astype(float) - elapsed_seconds, 0.0)

    def _aggregate_interval(self, row: pd.Series, tracking_df: pd.DataFrame) -> Dict[str, float]:
        delivered_signed = tracking_df["delivered_up_kw"] - tracking_df["delivered_down_kw"]
        requested_signed = tracking_df["requested_up_kw"] - tracking_df["requested_down_kw"]
        return {
            "frequency_hz": float(tracking_df["frequency_hz"].mean()),
            "requested_up_kw": float(tracking_df["requested_up_kw"].mean()),
            "requested_down_kw": float(tracking_df["requested_down_kw"].mean()),
            "delivered_up_kw": float(tracking_df["delivered_up_kw"].mean()),
            "delivered_down_kw": float(tracking_df["delivered_down_kw"].mean()),
            "bess_up_kw": float(tracking_df["bess_up_kw"].mean()),
            "bess_down_kw": float(tracking_df["bess_down_kw"].mean()),
            "ev_up_kw": float(tracking_df["ev_up_kw"].mean()),
            "ev_down_kw": float(tracking_df["ev_down_kw"].mean()),
            "hvac_up_kw": float(tracking_df["hvac_up_kw"].mean()),
            "hvac_down_kw": float(tracking_df["hvac_down_kw"].mean()),
            "pv_down_kw": float(tracking_df["pv_down_kw"].mean()),
            "hvac_fcr_up_kw": 0.0,
            "hvac_fcr_down_kw": 0.0,
            "bess_fcr_up_kw": 0.0,
            "bess_fcr_down_kw": 0.0,
            "ev_fcr_up_kw": 0.0,
            "ev_fcr_down_kw": 0.0,
            "bess_soc_mwh": float(self.devices.loc[self.devices["device_type"] == "BESS", "state_mwh"].sum()),
            "ev_soc_mwh": float(self.devices.loc[self.devices["device_type"] == "EV", "state_mwh"].sum()),
            "indoor_temp_c": float(row.get("indoor_temp_c_ref", 21.0) + self.devices.loc[self.devices["device_type"] == "HVAC", "temp_delta_c"].mean()),
            "optimized_net_load_kw": float(row["net_load_baseline_kw"] - max(delivered_signed.mean(), 0.0) + max(-delivered_signed.mean(), 0.0)),
            "baseline_net_load_kw": float(row["net_load_baseline_kw"]),
            "tracking_samples": int(len(tracking_df)),
            "tracking_error_kw": float(np.abs(delivered_signed - requested_signed).mean()),
            "mean_signed_error_kw": float((delivered_signed - requested_signed).mean()),
            "max_abs_error_per_tick_kw": float(np.abs(delivered_signed - requested_signed).max()),
            "socket_violation_up_kw": float(tracking_df.get("socket_violation_up_kw", pd.Series([0.0])).sum()),
            "socket_violation_down_kw": float(tracking_df.get("socket_violation_down_kw", pd.Series([0.0])).sum()),
            "shortfall_kw": float(tracking_df["shortfall_kw"].mean()),
            "inner_solver_status": tracking_df["inner_solver_status"].mode().iloc[0] if not tracking_df.empty else "unknown",
        }

    def _fatigue_summary(self) -> pd.DataFrame:
        if self.devices.empty:
            return pd.DataFrame()
        frame = self.devices.reset_index(drop=True)[
            [
                "household_id",
                "device_id",
                "device_type",
                "gateway_id",
                "usage_seconds",
                "consecutive_seconds",
                "cooldown_seconds",
                "throughput_kwh",
                "activation_count",
            ]
        ].copy()
        frame["usage_hours"] = frame["usage_seconds"] / 3600.0
        frame["fatigue_score"] = (
            frame["usage_seconds"] / np.maximum(self.devices.reset_index(drop=True)["max_consecutive_seconds"].astype(float), 1.0)
            + 0.02 * frame["activation_count"]
            + 0.01 * frame["throughput_kwh"]
        )
        return frame.sort_values("fatigue_score", ascending=False)
