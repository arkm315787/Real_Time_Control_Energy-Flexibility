"""Centralized household roster and 4-second MPC dispatch for FlexiHome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.optimize import linprog


DEVICE_ORDER = ["BESS", "EV", "HVAC", "PV"]

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
    horizon_seconds: int = 20
    rotation_strategy: str = "usage_aware"
    gateway_mode: str = "simulated_centralized"
    max_devices_per_type: int = 48

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
        tracking_rows: List[Dict[str, float]] = []
        command_accumulator: Dict[str, Dict[str, object]] = {}
        outer_dt_seconds = int(round(float(row["dt_h"]) * 3600.0))

        for tick_idx, (ts, signal_row) in enumerate(fine_signals.iterrows()):
            horizon = self._signal_horizon(fine_signals, tick_idx)
            available = self._availability(row, selected.index)
            caps = self._pool_caps(selected, available, resource_mode)
            request_now = self._request_kw(signal_row, plan, market_mode)
            solution = self._solve_pool_mpc(horizon, plan, caps, market_mode)

            if solution["status"] != "Optimal":
                dispatch_by_type = {device_type: 0.0 for device_type in DEVICE_ORDER}
            else:
                dispatch_by_type = solution["dispatch_by_type"]

            commands = self._apportion_commands(selected, available, dispatch_by_type)
            delivered = float(commands["command_kw"].sum()) if not commands.empty else 0.0
            self._apply_device_commands(commands, command_accumulator)

            tracking_rows.append(
                {
                    "timestamp": ts,
                    "inner_solver_status": solution["status"],
                    "inner_controller_mode": self.config.mode,
                    "gateway_mode": self.config.gateway_mode,
                    "frequency_hz": float(signal_row["frequency_hz"]),
                    "fcr_signal_norm": float(signal_row["fcr_signal_norm"]),
                    "afrr_signal_norm": float(signal_row["afrr_signal_norm"]),
                    "requested_up_kw": max(request_now, 0.0),
                    "requested_down_kw": max(-request_now, 0.0),
                    "delivered_up_kw": max(delivered, 0.0),
                    "delivered_down_kw": max(-delivered, 0.0),
                    "tracking_error_kw": delivered - request_now,
                    "shortfall_kw": max(abs(request_now) - abs(delivered), 0.0),
                    "bess_up_kw": max(dispatch_by_type.get("BESS", 0.0), 0.0),
                    "bess_down_kw": max(-dispatch_by_type.get("BESS", 0.0), 0.0),
                    "ev_up_kw": max(dispatch_by_type.get("EV", 0.0), 0.0),
                    "ev_down_kw": max(-dispatch_by_type.get("EV", 0.0), 0.0),
                    "hvac_up_kw": max(dispatch_by_type.get("HVAC", 0.0), 0.0),
                    "hvac_down_kw": max(-dispatch_by_type.get("HVAC", 0.0), 0.0),
                    "pv_down_kw": max(-dispatch_by_type.get("PV", 0.0), 0.0),
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

        requirements = self._requirements_by_type(plan, resource_mode)
        selected_parts = []
        for device_type in DEVICE_ORDER:
            req = requirements[device_type]
            if req["up"] <= 0.0 and req["down"] <= 0.0:
                continue
            candidates = devices[
                (devices["device_type"] == device_type)
                & devices["online"].astype(bool)
                & (devices["cooldown_seconds"].astype(float) <= 0.0)
                & (devices["consecutive_seconds"].astype(float) < devices["max_consecutive_seconds"].astype(float))
                & ((devices["available_up_kw"] > 0.001) | (devices["available_down_kw"] > 0.001))
            ].copy()
            if candidates.empty:
                continue
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
                selected.append(row_payload)
                up_remaining -= up_cap
                down_remaining -= down_cap
                if up_remaining <= 0.001 and down_remaining <= 0.001:
                    break
            if selected:
                selected_parts.append(pd.DataFrame(selected))

        if not selected_parts:
            return pd.DataFrame(columns=list(self.devices.columns) + ["upper_up_cap_kw", "upper_down_cap_kw"])
        selected_df = pd.concat(selected_parts, ignore_index=True).set_index("device_id", drop=False)
        return selected_df

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

    def _pool_caps(self, selected: pd.DataFrame, available: pd.DataFrame, resource_mode: str) -> Dict[str, Dict[str, float]]:
        caps = {device_type: {"up": 0.0, "down": 0.0} for device_type in DEVICE_ORDER}
        if selected.empty:
            return caps
        working = selected.join(available[["available_up_kw", "available_down_kw"]], rsuffix="_now")
        for device_type, group in working.groupby("device_type"):
            if resource_mode == "Fast only" and device_type in {"HVAC", "PV"}:
                continue
            caps[device_type]["up"] = float(np.minimum(group["upper_up_cap_kw"], group["available_up_kw_now"]).sum())
            caps[device_type]["down"] = float(np.minimum(group["upper_down_cap_kw"], group["available_down_kw_now"]).sum())
        return caps

    def _signal_horizon(self, fine_signals: pd.DataFrame, tick_idx: int) -> pd.DataFrame:
        horizon = fine_signals.iloc[tick_idx : tick_idx + self.config.horizon_steps].copy()
        if horizon.empty:
            horizon = fine_signals.iloc[[-1]].copy()
        while len(horizon) < self.config.horizon_steps:
            next_row = horizon.iloc[[-1]].copy()
            next_row.index = [horizon.index[-1] + pd.Timedelta(seconds=self.config.dt_seconds)]
            horizon = pd.concat([horizon, next_row])
        return horizon

    def _request_kw(self, signal_row: pd.Series, plan: Dict[str, float], market_mode: str) -> float:
        fcr_signal = float(signal_row["fcr_signal_norm"]) if market_mode in {"FCR-N", "Combined"} else 0.0
        afrr_signal = float(signal_row["afrr_signal_norm"]) if market_mode in {"aFRR", "Combined"} else 0.0
        fcr_bid = float(plan.get("bess_fcr_kw", 0.0) + plan.get("ev_fcr_kw", 0.0) + plan.get("hvac_fcr_kw", 0.0))
        afrr_up = float(plan.get("bess_afrr_up_kw", 0.0) + plan.get("ev_afrr_up_kw", 0.0) + plan.get("hvac_afrr_up_kw", 0.0))
        afrr_down = float(
            plan.get("bess_afrr_down_kw", 0.0)
            + plan.get("ev_afrr_down_kw", 0.0)
            + plan.get("hvac_afrr_down_kw", 0.0)
            + plan.get("pv_afrr_down_kw", 0.0)
        )
        return fcr_signal * fcr_bid + max(afrr_signal, 0.0) * afrr_up - max(-afrr_signal, 0.0) * afrr_down

    def _solve_pool_mpc(
        self,
        horizon: pd.DataFrame,
        plan: Dict[str, float],
        caps: Dict[str, Dict[str, float]],
        market_mode: str,
    ) -> Dict[str, object]:
        resource_count = len(DEVICE_ORDER)
        steps = len(horizon)
        up_start = 0
        down_start = up_start + resource_count * steps
        slack_pos_start = down_start + resource_count * steps
        slack_neg_start = slack_pos_start + steps
        move_pos_start = slack_neg_start + steps
        move_neg_start = move_pos_start + resource_count * steps
        n_vars = move_neg_start + resource_count * steps

        def r_index(start: int, resource_idx: int, step_idx: int) -> int:
            return start + step_idx * resource_count + resource_idx

        c = np.zeros(n_vars)
        bounds = [(0.0, None)] * n_vars
        for k in range(steps):
            c[slack_pos_start + k] = 10000.0
            c[slack_neg_start + k] = 10000.0
            for r, device_type in enumerate(DEVICE_ORDER):
                bounds[r_index(up_start, r, k)] = (0.0, max(float(caps[device_type]["up"]), 0.0))
                bounds[r_index(down_start, r, k)] = (0.0, max(float(caps[device_type]["down"]), 0.0))
                cost = DEVICE_DEFAULTS[device_type]["degradation_cost"]
                c[r_index(up_start, r, k)] = cost
                c[r_index(down_start, r, k)] = cost
                c[r_index(move_pos_start, r, k)] = 1.5
                c[r_index(move_neg_start, r, k)] = 1.5

        a_eq = []
        b_eq = []
        for k, (_, signal_row) in enumerate(horizon.iterrows()):
            row = np.zeros(n_vars)
            for r in range(resource_count):
                row[r_index(up_start, r, k)] = 1.0
                row[r_index(down_start, r, k)] = -1.0
            row[slack_pos_start + k] = -1.0
            row[slack_neg_start + k] = 1.0
            a_eq.append(row)
            b_eq.append(self._request_kw(signal_row, plan, market_mode))

        for k in range(steps):
            for r, device_type in enumerate(DEVICE_ORDER):
                row = np.zeros(n_vars)
                row[r_index(up_start, r, k)] = 1.0
                row[r_index(down_start, r, k)] = -1.0
                if k > 0:
                    row[r_index(up_start, r, k - 1)] = -1.0
                    row[r_index(down_start, r, k - 1)] = 1.0
                    rhs = 0.0
                else:
                    rhs = float(self.previous_pool_dispatch.get(device_type, 0.0))
                row[r_index(move_pos_start, r, k)] = -1.0
                row[r_index(move_neg_start, r, k)] = 1.0
                a_eq.append(row)
                b_eq.append(rhs)

        result = linprog(c, A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq), bounds=bounds, method="highs")
        if not result.success:
            return {"status": "Failed", "dispatch_by_type": {device_type: 0.0 for device_type in DEVICE_ORDER}}

        dispatch_by_type = {}
        for r, device_type in enumerate(DEVICE_ORDER):
            up = float(result.x[r_index(up_start, r, 0)])
            down = float(result.x[r_index(down_start, r, 0)])
            dispatch_by_type[device_type] = up - down
            self.previous_pool_dispatch[device_type] = up - down
        return {"status": "Optimal", "dispatch_by_type": dispatch_by_type}

    def _apportion_commands(self, selected: pd.DataFrame, available: pd.DataFrame, dispatch_by_type: Dict[str, float]) -> pd.DataFrame:
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
            group = working[working["device_type"] == device_type].sort_values(["usage_score", "household_id"]).copy()
            for _, device in group.iterrows():
                cap = min(float(device.get(cap_col, 0.0)), float(device.get(f"upper_{direction}_cap_kw", 0.0)))
                if cap <= 0.001 or remaining <= 0.001:
                    continue
                dispatched = min(cap, remaining)
                records.append({**device.to_dict(), "command_kw": dispatched if direction == "up" else -dispatched})
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
            "tracking_error_kw": float((delivered_signed - requested_signed).mean()),
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
