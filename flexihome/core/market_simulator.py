"""Independent market pressure signal generator for decoupled lower MPC."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class MarketSimulatorConfig:
    """Configuration for synthetic market pressure signal generation."""

    dt_seconds: int = 4
    fcr_volatility: float = 0.15
    afrr_volatility: float = 0.20
    signal_smoothing: float = 0.85


class MarketSimulator:
    """Generates independent 4-second pressure signals on wall-clock time."""

    def __init__(
        self,
        preview_signals: pd.DataFrame,
        config: MarketSimulatorConfig | None = None,
        start_wall_time_s: float | None = None,
    ) -> None:
        """
        Initialize market simulator.

        Args:
            preview_signals: Pre-computed market signals (fcr_signal_norm, afrr_signal_norm).
            config: Simulator configuration.
            start_wall_time_s: Wall-clock time when market starts (None = immediately).
        """
        self.config = config or MarketSimulatorConfig()
        self.preview_signals = preview_signals.reset_index(drop=True) if isinstance(preview_signals, pd.DataFrame) else pd.DataFrame()
        self.start_wall_time_s = start_wall_time_s
        self.last_tick_index = -1
        self.last_fcr_signal = 0.0
        self.last_afrr_signal = 0.0
        self.last_frequency_hz = 50.0
        self.tick_count = 0
        self.rng = np.random.default_rng(seed=42)

    def set_start_time(self, wall_time_s: float) -> None:
        """Update market start time."""
        self.start_wall_time_s = wall_time_s

    def get_current_signal(self, wall_time_now_s: float) -> Dict[str, float] | None:
        """
        Poll current market signal based on wall-clock time.

        Returns:
            Dict with 'fcr_signal_norm', 'afrr_signal_norm', 'timestamp', or None if market not started.
        """
        if self.start_wall_time_s is None:
            return None

        # Has market started yet?
        if wall_time_now_s < self.start_wall_time_s:
            return None

        # How much wall-clock time elapsed since market start?
        elapsed_s = wall_time_now_s - self.start_wall_time_s

        # Which 4-second tick are we in?
        tick_index = int(elapsed_s / float(self.config.dt_seconds))

        # If we're in a new tick, generate/fetch the signal
        if tick_index != self.last_tick_index:
            self.last_tick_index = tick_index

            # Use preview if available, otherwise generate synthetic
            if tick_index < len(self.preview_signals):
                row = self.preview_signals.iloc[tick_index]
                fcr_signal = float(row.get("fcr_signal_norm", 0.0))
                afrr_signal = float(row.get("afrr_signal_norm", 0.0))
                frequency_hz = float(row.get("frequency_hz", 50.0 - 0.1 * fcr_signal))
            else:
                # Synthetic generation for ticks beyond preview
                fcr_noise = float(self.rng.normal(0, self.config.fcr_volatility))
                afrr_noise = float(self.rng.normal(0, self.config.afrr_volatility))
                fcr_signal = float(
                    np.clip(
                        self.config.signal_smoothing * self.last_fcr_signal + (1 - self.config.signal_smoothing) * fcr_noise,
                        -1.0,
                        1.0,
                    )
                )
                afrr_signal = float(
                    np.clip(
                        self.config.signal_smoothing * self.last_afrr_signal + (1 - self.config.signal_smoothing) * afrr_noise,
                        -1.0,
                        1.0,
                    )
                )
                frequency_hz = 50.0 - 0.1 * fcr_signal

            self.last_fcr_signal = fcr_signal
            self.last_afrr_signal = afrr_signal
            self.last_frequency_hz = frequency_hz
            self.tick_count += 1

            return {
                "fcr_signal_norm": fcr_signal,
                "afrr_signal_norm": afrr_signal,
                "frequency_hz": frequency_hz,
                "tick_index": tick_index,
                "timestamp": pd.Timestamp.now(),
                "wall_elapsed_seconds": elapsed_s,
            }

        # Still in the same tick, return cached signal
        return {
            "fcr_signal_norm": self.last_fcr_signal,
            "afrr_signal_norm": self.last_afrr_signal,
            "frequency_hz": self.last_frequency_hz,
            "tick_index": tick_index,
            "timestamp": pd.Timestamp.now(),
            "wall_elapsed_seconds": elapsed_s,
        }

    def get_elapsed_seconds(self) -> float:
        """Get elapsed seconds since market start, or 0 if not started."""
        if self.start_wall_time_s is None:
            return 0.0
        return max(time.time() - self.start_wall_time_s, 0.0)

    def reset(self) -> None:
        """Reset simulator state."""
        self.last_tick_index = -1
        self.last_fcr_signal = 0.0
        self.last_afrr_signal = 0.0
        self.last_frequency_hz = 50.0
        self.tick_count = 0
