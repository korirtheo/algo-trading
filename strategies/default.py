"""
Strategy L: Low Float Squeeze
==============================
Gap-up + low float + HOD break with volume surge -> squeeze play.

Entry: New HOD + volume surge + price acceleration + above VWAP
Exit:  Trailing stop + hard stop + time stop + EOD

Params from standalone Optuna trial #486: $78.6M PnL, PF 4.33, 84.9% WR
"""
import numpy as np


# --- DEFAULT PARAMS (combined optimizer trial #432: 583 trades, 88.7% WR, $79.7M L PnL) ---
DEFAULT_PARAMS = {
    "max_float": 15_000_000,
    "min_gap_pct": 15.0,
    "earliest_candle": 12,
    "latest_candle": 150,
    "hod_break_required": True,
    "vol_surge_mult": 1.5,
    "min_price_accel_pct": 2.5,
    "require_above_vwap": True,
    "tier1_float": 2_000_000,
    "tier2_float": 5_000_000,
    "tier1_target1_pct": 45.0,
    "tier1_target2_pct": 60.0,
    "tier2_target1_pct": 30.0,
    "tier2_target2_pct": 50.0,
    "tier3_target1_pct": 8.0,
    "tier3_target2_pct": 20.0,
    "partial_sell_pct": 25.0,
    "stop_pct": 15.0,
    "trail_pct": 1.0,
    "trail_activate_pct": 1.0,
    "time_limit_minutes": 120,
}


def compute_vwap(highs, lows, closes, volumes):
    """Compute cumulative VWAP from arrays."""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)
    typical = (h + l + c) / 3.0
    cum_tp_vol = np.cumsum(typical * v)
    cum_vol = np.cumsum(v)
    cum_vol[cum_vol == 0] = 1e-9
    return cum_tp_vol / cum_vol


def get_tiered_targets(float_shares, params):
    """Return (target1_pct, target2_pct) based on float size."""
    if float_shares < params["tier1_float"]:
        return params["tier1_target1_pct"], params["tier1_target2_pct"]
    elif float_shares < params["tier2_float"]:
        return params["tier2_target1_pct"], params["tier2_target2_pct"]
    else:
        return params["tier3_target1_pct"], params["tier3_target2_pct"]


def is_eligible(ticker, gap_pct, float_shares, params=None):
    """Check if a ticker qualifies for L strategy."""
    p = params or DEFAULT_PARAMS
    if float_shares is None or float_shares > p["max_float"]:
        return False
    if gap_pct < p["min_gap_pct"]:
        return False
    return True


def create_state(ticker, gap_pct, float_shares, premarket_high, pm_volume, params=None):
    """Create initial state dict for tracking one ticker."""
    p = params or DEFAULT_PARAMS
    tgt1, tgt2 = get_tiered_targets(float_shares, p)
    return {
        "ticker": ticker,
        "float_shares": float_shares,
        "gap_pct": gap_pct,
        "premarket_high": premarket_high,
        "pm_volume": pm_volume,
        "strategy": "L",
        # Tracking
        "candle_count": 0,
        "open_price": None,
        "running_hod": 0.0,
        "signal": False,
        "signal_price": None,
        # Volume tracking for surge detection
        "recent_volumes": [],
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
        "target1_pct": tgt1,
        "target2_pct": tgt2,
        "highest_since_entry": 0.0,
        "trailing_active": False,
        "partial_taken": False,
        "partial_proceeds": 0.0,
    }


def check_signal(state, c_open, c_high, c_low, c_close, c_vol, vwap_value=None, params=None):
    """Check if entry signal fires on this bar.

    Args:
        state: mutable state dict (updated in place)
        c_open/c_high/c_low/c_close/c_vol: current bar OHLCV
        vwap_value: current VWAP value (optional, required if require_above_vwap)
        params: strategy params dict

    Returns:
        True if signal fires, False otherwise.
        Updates state["signal"] and state["signal_price"] if True.
    """
    p = params or DEFAULT_PARAMS

    if state["done"] or state["entry_price"] is not None:
        return False

    state["candle_count"] += 1

    # Track open price
    if state["candle_count"] == 1:
        state["open_price"] = c_open

    # Track volume for surge detection
    state["recent_volumes"].append(c_vol)

    # Update running HOD
    is_new_hod = False
    if c_high > state["running_hod"]:
        prev_hod = state["running_hod"]
        state["running_hod"] = c_high
        is_new_hod = prev_hod > 0  # Not first candle

    # Too early or too late
    if state["candle_count"] < p["earliest_candle"]:
        return False
    if state["candle_count"] > p["latest_candle"]:
        state["done"] = True
        return False

    # 1. HOD break check
    if p["hod_break_required"] and not is_new_hod:
        return False

    # 2. Volume surge check
    vols = state["recent_volumes"]
    candle_idx = len(vols) - 1
    if candle_idx >= 10:
        avg_vol = float(np.mean(vols[-11:-1]))  # last 10 before current
    elif candle_idx > 0:
        avg_vol = float(np.mean(vols[:-1]))
    else:
        avg_vol = 0

    if avg_vol > 0 and c_vol < avg_vol * p["vol_surge_mult"]:
        return False

    # 3. Price acceleration (green candle with min body %)
    if c_open > 0:
        body_pct = (c_close / c_open - 1) * 100
    else:
        body_pct = 0
    if body_pct < p["min_price_accel_pct"]:
        return False

    # 4. VWAP filter
    if p["require_above_vwap"] and vwap_value is not None:
        if c_close < vwap_value:
            return False

    # All conditions met
    state["signal"] = True
    state["signal_price"] = c_close
    return True


def check_exit(state, c_high, c_low, c_close, minutes_in_trade, minutes_to_close,
               slippage_pct=0.05, eod_exit_minutes=15, params=None):
    """Check if any exit condition is met.

    Args:
        state: mutable state dict
        c_high/c_low/c_close: current bar HLC
        minutes_in_trade: minutes since entry
        minutes_to_close: minutes until 4:00 PM ET
        slippage_pct: slippage %
        eod_exit_minutes: close positions this many minutes before close
        params: strategy params dict

    Returns:
        (should_exit, exit_price, exit_reason) or (False, None, None)
    """
    p = params or DEFAULT_PARAMS

    if state["entry_price"] is None:
        return False, None, None

    entry = state["entry_price"]

    # EOD forced exit
    if minutes_to_close <= eod_exit_minutes:
        return True, c_close, "EOD_CLOSE"

    # Update highest since entry
    if c_high > state["highest_since_entry"]:
        state["highest_since_entry"] = c_high

    # 1. Trailing stop (if active)
    if state["trailing_active"]:
        trail_stop = state["highest_since_entry"] * (1 - p["trail_pct"] / 100)
        if c_low <= trail_stop:
            return True, trail_stop, "TRAIL"
    else:
        # 2. Hard stop (before trail activates)
        stop_price = entry * (1 - p["stop_pct"] / 100)
        if c_low <= stop_price:
            return True, stop_price, "STOP"

    # 3. Activate trailing stop
    if not state["trailing_active"]:
        unrealized_pct = (c_high / entry - 1) * 100
        if unrealized_pct >= p["trail_activate_pct"]:
            state["trailing_active"] = True

    # 4. Partial sell at target1 (handled externally for live — returns signal)
    if not state["partial_taken"] and p["partial_sell_pct"] > 0:
        tgt1 = entry * (1 + state["target1_pct"] / 100)
        if c_high >= tgt1:
            # Signal partial sell — caller handles execution
            state["partial_taken"] = True
            return True, tgt1, "PARTIAL"

    # 5. Full target (target2)
    tgt2 = entry * (1 + state["target2_pct"] / 100)
    if c_high >= tgt2:
        return True, tgt2, "TARGET"

    # 6. Time stop
    if minutes_in_trade >= p["time_limit_minutes"]:
        return True, c_close, "TIME_STOP"

    return False, None, None
