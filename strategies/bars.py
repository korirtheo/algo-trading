"""
Bar-interval helpers for the 1-min data pipeline.

The historical universe is now downloaded as 1-min bars (stored_data_1min/),
while legacy stored_data_*/ dirs contain pre-aggregated 2-min bars. The
strategy layer (simulate_day_combined) expects 2-min bars throughout
(candle_count semantics, opening-range windows, etc.).

This module provides:
  - detect_bar_size_minutes(df): infer the bar interval from the index spacing
  - resample_1min_to_2min(df, tz): deterministic 1-min -> 2-min aggregation
    aligned to 9:30 ET using the SAME slot logic as the live TradierStreamer
    (_bar_slot: minutes_since_open // 2), so backtest bars match live bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_bar_size_minutes(df: pd.DataFrame) -> int:
    """Infer the bar interval (minutes) from the index's median spacing.

    Handles 1-min, 2-min, and larger legacy bars. Returns 0 when it can't
    determine (caller keeps bars as-is).
    """
    if df is None or len(df) < 2:
        return 0
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    diffs = np.diff(idx.asi8)  # units depend on dtype resolution (us or ns)
    if len(diffs) == 0:
        return 0
    # Normalize to seconds regardless of dtype resolution.
    res = getattr(idx, "_resolution", 6)  # -6=ns, -3=us ... actually use dtype
    if str(idx.dtype).endswith("ns"):
        secs = diffs / 1e9
    else:  # us (and any other resolution is minute-scale at minimum)
        secs = diffs / 1e6
    minutes = float(np.median(secs)) / 60
    if minutes <= 0:
        return 0
    # round to nearest integer minute (handles slight skew from missing bars)
    return int(round(minutes))


def _slot_index(index, tz):
    """Map each timestamp to its 2-min slot aligned at 9:30 ET.

    slot = minutes_since_open // 2  (matches live/_bar_slot)
    Candle 1 = 9:30-9:31 -> slot 0, Candle 2 = 9:32-9:33 -> slot 1, ...
    """
    et = index.tz_convert(tz)
    minutes_since_open = (
        et.hour * 60 + et.minute
    ) - (9 * 60 + 30)
    return (minutes_since_open // 2).to_numpy()


def resample_1min_to_2min(df: pd.DataFrame, tz=None) -> pd.DataFrame:
    """Deterministically aggregate 1-min market-hours bars into 2-min bars.

    Slot-aligned to 9:30 ET so bar 1 covers 9:30-9:31 (matches the live
    streamer). open=first, high=max, low=min, close=last, volume=sum.

    Bars that straddle the 9:30 boundary (i.e. pre-9:30 minutes in the df)
    are dropped — callers pass market-hours-only data.
    """
    import zoneinfo

    if tz is None:
        tz = zoneinfo.ZoneInfo("America/New_York")

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(tz)

    slots = _slot_index(idx, tz)
    valid = slots >= 0  # drop any pre-9:30 remnants
    if not valid.any():
        return df.iloc[0:0]

    df = df.iloc[np.where(valid)[0]].copy()
    idx = idx[valid]
    slots = slots[valid]

    g = pd.DataFrame({
        "Open": df["Open"].to_numpy(),
        "High": df["High"].to_numpy(),
        "Low": df["Low"].to_numpy(),
        "Close": df["Close"].to_numpy(),
        "Volume": df["Volume"].to_numpy(),
        "_slot": slots,
        "_ts": idx,
    })

    grouped = g.groupby("_slot")
    out = pd.DataFrame({
        "Open": grouped["Open"].first(),
        "High": grouped["High"].max(),
        "Low": grouped["Low"].min(),
        "Close": grouped["Close"].last(),
        "Volume": grouped["Volume"].sum(),
        "_ts": grouped["_ts"].first(),
    })
    out = out.set_index("_ts").sort_index()
    out.index.name = df.index.name or "Datetime"
    return out[["Open", "High", "Low", "Close", "Volume"]]


def ensure_2min_bars(df: pd.DataFrame, tz=None) -> pd.DataFrame:
    """Return df as 2-min bars: pass-through if already 2-min, otherwise
    deterministically resample 1-min (or finer) -> 2-min. No-op on unknown."""
    size = detect_bar_size_minutes(df)
    if size <= 1:
        return resample_1min_to_2min(df, tz)
    return df
