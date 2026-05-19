"""
Strategy HALT: Halt-Resume Runner
==================================
Catches stocks halted intraday (T1/T2/T12/LUDP/LUDS) that print a
liquidity-driven run on resumption.

Entry: first post-resume 2-min bar with sufficient volume + green body,
within `max_entry_bars_after_resume` bars of the official resume time.

Exit:  hard stop / target1 (partial) / target2 (full) / trail / time / EOD.

Same module shape as strategies/low_float_squeeze.py so the engine can call
it uniformly.

NOTE: this strategy does NOT participate in the pre-market scan or in
simulate_day_combined. It is invoked separately by engine_combined once a
halt-resume ticker has been added intraday.
"""
import numpy as np


DEFAULT_PARAMS = {
    # Eligibility
    "min_resume_volume": 50_000,         # first post-resume candle volume
    "max_entry_bars_after_resume": 5,    # only enter within first 10 minutes
    "min_price": 1.0,
    "max_price": 50.0,
    "max_float": 50_000_000,             # bigger floats just don't run as hard
    "min_resume_gap_pct": 0.0,           # resume_price vs halt_price (>=0 = up only)

    # Position management
    "stop_pct": 10.0,
    "partial_sell_pct": 50.0,
    "target1_pct": 25.0,
    "target2_pct": 50.0,
    "trail_pct": 5.0,
    "trail_activate_pct": 10.0,
    "time_limit_minutes": 30,

    # Halt reason whitelist (NASDAQ codes)
    # T1=news pending, T2=news released, T12=other regulatory,
    # LUDP/LUDS=volatility halt up/down. Market-wide M1-M9 EXCLUDED.
    "tradable_reasons": ["T1", "T2", "T12", "LUDP", "LUDS"],
}


def is_eligible(halt_event, float_shares=None, params=None):
    """Filter halt events to the universe we want to trade.

    Args:
        halt_event: object exposing .reason, .resume_price, .halt_price
        float_shares: int or None. If None we are permissive — per spec
                      open-question (1): treat missing float as eligible.
        params: param overrides
    Returns:
        bool
    """
    p = params or DEFAULT_PARAMS

    reason = (halt_event.reason or "").strip().upper()
    if reason not in [r.upper() for r in p["tradable_reasons"]]:
        return False

    rp = halt_event.resume_price
    if rp is None or rp <= 0:
        return False
    if not (p["min_price"] <= rp <= p["max_price"]):
        return False

    if float_shares is not None and float_shares > p["max_float"]:
        return False

    if halt_event.halt_price and halt_event.halt_price > 0:
        gap_pct = (rp / halt_event.halt_price - 1) * 100
        if gap_pct < p["min_resume_gap_pct"]:
            return False

    return True


def create_state(ticker, halt_event, float_shares=None, params=None):
    """Init per-ticker state at resume. Mirrors low_float_squeeze.create_state."""
    p = params or DEFAULT_PARAMS
    return {
        "ticker": ticker,
        "strategy": "HALT",
        "halt_reason": (halt_event.reason or "").strip().upper(),
        "resume_ts": halt_event.resume_dt,
        "resume_price": halt_event.resume_price,
        "halt_price": halt_event.halt_price,
        "float_shares": float_shares,
        # Bar tracking
        "bars_since_resume": 0,
        "signal": False, "signal_price": None,
        # Position
        "entry_price": None,
        "entry_time": None,
        "exit_price": None,
        "exit_time": None,
        "exit_reason": None,
        "shares": 0,
        "position_cost": 0.0,
        "pnl": 0.0,
        "vol_capped": False,
        "done": False,
        # Exit management
        "highest_since_entry": 0.0,
        "trailing_active": False,
        "partial_taken": False,
        "partial_proceeds": 0.0,
    }


def check_signal(state, c_open, c_high, c_low, c_close, c_vol, params=None):
    """Fire entry on first post-resume bar with min volume AND green body.

    Returns True iff signal fires. Mutates state["signal"] / signal_price.
    """
    p = params or DEFAULT_PARAMS
    if state["done"] or state["entry_price"] is not None:
        return False

    state["bars_since_resume"] += 1
    if state["bars_since_resume"] > p["max_entry_bars_after_resume"]:
        state["done"] = True
        return False

    if c_vol < p["min_resume_volume"]:
        return False
    if c_close <= c_open:
        # Wait for the first green confirmation. A red first bar after a
        # T1 resume is the classic "news was bad" tell — skip it.
        return False

    state["signal"] = True
    state["signal_price"] = c_close
    return True


def check_exit(state, c_high, c_low, c_close, minutes_in_trade, minutes_to_close,
               slippage_pct=0.05, eod_exit_minutes=15, params=None):
    """Exit on stop / target1(partial) / target2 / trail / time / EOD.

    Mirrors strategies/low_float_squeeze.check_exit so the engine integration
    point is identical.

    Returns (should_exit: bool, exit_price: float|None, reason: str|None).
    """
    p = params or DEFAULT_PARAMS
    if state["entry_price"] is None:
        return False, None, None
    entry = state["entry_price"]

    # EOD forced exit
    if minutes_to_close <= eod_exit_minutes:
        return True, c_close, "EOD_CLOSE"

    # Track high-water mark
    if c_high > state["highest_since_entry"]:
        state["highest_since_entry"] = c_high

    # 1. Trail (if armed)
    if state["trailing_active"]:
        trail_stop = state["highest_since_entry"] * (1 - p["trail_pct"] / 100)
        if c_low <= trail_stop:
            return True, trail_stop, "TRAIL"
    else:
        # 2. Hard stop (before trail arms)
        stop_price = entry * (1 - p["stop_pct"] / 100)
        if c_low <= stop_price:
            return True, stop_price, "STOP"

    # 3. Arm trail once we have enough cushion
    if not state["trailing_active"]:
        unrealized_pct = (c_high / entry - 1) * 100
        if unrealized_pct >= p["trail_activate_pct"]:
            state["trailing_active"] = True

    # 4. Partial at target1
    if not state["partial_taken"] and p["partial_sell_pct"] > 0:
        tgt1 = entry * (1 + p["target1_pct"] / 100)
        if c_high >= tgt1:
            state["partial_taken"] = True
            return True, tgt1, "PARTIAL"

    # 5. Full target at target2
    tgt2 = entry * (1 + p["target2_pct"] / 100)
    if c_high >= tgt2:
        return True, tgt2, "TARGET"

    # 6. Time stop
    if minutes_in_trade >= p["time_limit_minutes"]:
        return True, c_close, "TIME_STOP"

    return False, None, None
