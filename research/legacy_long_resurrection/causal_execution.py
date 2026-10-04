"""Causal execution primitives for legacy long-strategy resurrection.

Rules:
- A bar-close signal is known only after that bar completes.
- Default market execution is therefore the NEXT bar open.
- No exit may inspect the signal bar or fill bar before the fill exists.
- Every fill carries signal/fill timestamps for causal auditing.

This module intentionally contains no legacy performance assumptions.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Signal:
    strategy: str
    bar_index: int
    timestamp: object
    reference_price: float
    reason: str


@dataclass(frozen=True)
class Fill:
    strategy: str
    signal_bar_index: int
    fill_bar_index: int
    signal_timestamp: object
    fill_timestamp: object
    signal_price: float
    raw_fill_price: float
    fill_price: float


def next_bar_open_fill(signal: Signal, bars, entry_slippage_pct: float) -> Optional[Fill]:
    """Execute a close-confirmed signal at the next available bar open."""
    fill_i = signal.bar_index + 1
    if fill_i >= len(bars):
        return None
    row = bars.iloc[fill_i]
    raw = float(row["Open"])
    if raw <= 0:
        return None
    return Fill(
        strategy=signal.strategy,
        signal_bar_index=signal.bar_index,
        fill_bar_index=fill_i,
        signal_timestamp=signal.timestamp,
        fill_timestamp=bars.index[fill_i],
        signal_price=float(signal.reference_price),
        raw_fill_price=raw,
        fill_price=raw * (1.0 + entry_slippage_pct / 100.0),
    )


def assert_causal_fill(fill: Fill) -> None:
    assert fill.fill_bar_index > fill.signal_bar_index, (
        f"{fill.strategy}: fill must occur after close-confirmed signal"
    )
    assert fill.fill_timestamp > fill.signal_timestamp, (
        f"{fill.strategy}: non-increasing signal/fill timestamp"
    )


def first_exit_bar(fill: Fill) -> int:
    """Earliest complete OHLC bar whose high/low can be used for exit testing."""
    return fill.fill_bar_index
