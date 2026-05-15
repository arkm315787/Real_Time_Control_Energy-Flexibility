"""Synchronize external market data with portfolio simulation frames."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


MARKET_SIGNAL_COLUMNS = {
    "spot_price_eur_per_mwh",
    "fcrn_capacity_eur_per_mw_h",
    "afrr_up_capacity_eur_per_mw_h",
    "afrr_down_capacity_eur_per_mw_h",
    "afrr_up_energy_eur_per_mwh",
    "afrr_down_energy_eur_per_mwh",
    "afrr_up_act_frac",
    "afrr_down_act_frac",
    "frequency_hz",
    "fcr_signal_norm",
    "afrr_signal_norm",
}


def data_completeness(df: pd.DataFrame) -> dict[str, float]:
    if df is None or df.empty:
        return {}
    return {col: float(df[col].notna().mean() * 100.0) for col in df.columns}


def _datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index, errors="coerce")
        work = work[work.index.notna()]
    if isinstance(work.index, pd.DatetimeIndex) and work.index.tz is not None:
        work.index = work.index.tz_convert(None)
    return work.sort_index()


def align_market_frame_to_index(market_df: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    if market_df is None or market_df.empty:
        return pd.DataFrame(index=target_index)
    work = _datetime_index(market_df)
    if work.empty:
        return pd.DataFrame(index=target_index)
    target = pd.DatetimeIndex(pd.to_datetime(target_index))
    if target.tz is not None:
        target = target.tz_convert(None)
    target_start = pd.Timestamp(target.min())
    target_end = pd.Timestamp(target.max())
    source_start = pd.Timestamp(work.index.min())
    source_end = pd.Timestamp(work.index.max())
    if source_end < target_start or source_start > target_end:
        repeated = {
            col: np.resize(work[col].dropna().to_numpy(), len(target))
            for col in work.columns
            if not work[col].dropna().empty
        }
        return pd.DataFrame(repeated, index=target)
    aligned = work.reindex(target)
    try:
        aligned = aligned.interpolate(method="time", limit_direction="both")
    except ValueError:
        aligned = aligned.ffill().bfill()
    return aligned.ffill().bfill()


def merge_market_data(portfolio_df: pd.DataFrame, market_df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if portfolio_df is None or portfolio_df.empty or market_df is None or market_df.empty:
        return portfolio_df.copy() if isinstance(portfolio_df, pd.DataFrame) else pd.DataFrame()
    merged = portfolio_df.copy()
    aligned = align_market_frame_to_index(market_df, merged.index)
    selected = list(columns) if columns is not None else [col for col in aligned.columns if col in MARKET_SIGNAL_COLUMNS or col not in merged.columns]
    for col in selected:
        if col in aligned.columns:
            merged[col] = aligned[col]
    if {"afrr_up_act_frac", "afrr_down_act_frac"}.issubset(aligned.columns):
        up = pd.to_numeric(aligned["afrr_up_act_frac"], errors="coerce")
        down = pd.to_numeric(aligned["afrr_down_act_frac"], errors="coerce")
        if up.notna().any() or down.notna().any():
            merged["afrr_signal_norm"] = (up.fillna(0.0) - down.fillna(0.0)).clip(-1.0, 1.0)
    return merged


def split_training_and_operational(combined_df: pd.DataFrame, training_ratio: float = 90 / 91) -> tuple[pd.DataFrame, pd.DataFrame]:
    if combined_df is None or combined_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    split_idx = max(min(int(len(combined_df) * training_ratio), len(combined_df)), 0)
    return combined_df.iloc[:split_idx].copy(), combined_df.iloc[split_idx:].copy()
