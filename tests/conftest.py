"""
Shared fixtures and helpers for algo-trading test suite.

Provides:
  - Synthetic candle/pick generators for simulator integration tests
  - Pre-built param dicts for each strategy
  - Snapshot builders
  - Param-lock fixture for thread-safety during tests
"""
import sys
import os
import pandas as pd
import numpy as np
import pytest
from zoneinfo import ZoneInfo

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import test_green_candle_combined as tgc
from optimize_combined import (
    ALL_STRATS, set_strategy_params, _build_param_snapshot, _param_lock,
    DATA_DIRS,
)
from optimize_gl_split import (
    _map_split_to_standard, _build_split_snapshot, suggest_gl_split_params,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
BASE_PRICE = 10.0  # Base price for synthetic candles


# ---------------------------------------------------------------------------
# Synthetic candle generator
# ---------------------------------------------------------------------------
def make_candles(opens, highs, lows, closes, volumes=None, start_time="09:30",
                 date="2025-01-15", freq="1min"):
    """Build a market-hours DataFrame from OHLCV arrays.

    Each entry is one minute of data. Time starts at start_time on the given date.
    Returns a DataFrame with DatetimeIndex (US/Eastern) and columns:
      Open, High, Low, Close, Volume
    """
    n = len(opens)
    if volumes is None:
        volumes = [100_000] * n

    # Build timestamps starting from start_time
    h, m = map(int, start_time.split(":"))
    base = pd.Timestamp(f"{date} {start_time}", tz=ET)
    index = pd.date_range(base, periods=n, freq=freq)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=index)
    return df


def make_pick(ticker="TEST", gap_pct=50.0, premarket_high=None, prev_close=10.0,
              pm_volume=1_000_000, market_hour_candles=None, **extra):
    """Build a pick dict matching the format from load_picks_for_dir."""
    if premarket_high is None:
        premarket_high = prev_close * 1.5
    market_open = prev_close * (1 + gap_pct / 100)
    pick = {
        "ticker": ticker,
        "gap_pct": gap_pct,
        "market_open": market_open,
        "premarket_high": premarket_high,
        "prev_close": prev_close,
        "pm_volume": pm_volume,
        "market_hour_candles": market_hour_candles,
    }
    pick.update(extra)
    return pick


# ---------------------------------------------------------------------------
# G-strategy candle sequences for exit logic testing
# ---------------------------------------------------------------------------
def g_entry_candles(date="2025-01-15", base=BASE_PRICE):
    """Candles that trigger G entry: big gap + 2nd green + new high.

    Returns ~25 candles:
      - Candle 0: big gap up open, green body, establishes first_candle_high
      - Candle 1: green (2nd green), makes new high -> G signal fires
      - Candles 2-24: price action to trigger various exits
    """
    gap_mult = 1.50  # 50% gap
    o0 = base * gap_mult  # ~15.0
    c0 = o0 * 1.02        # green body ~2%
    h0 = c0 * 1.01
    l0 = o0 * 0.99

    # Candle 1: 2nd green, new high -> triggers G signal
    o1 = c0
    c1 = o1 * 1.03
    h1 = c1 * 1.02
    l1 = o1 * 0.99

    # Candles 2+: price drops to trigger stop, or rises for trail/target
    # We'll make them configurable via the sequence
    candles = [
        # candle 0: big gap green
        (o0, h0, l0, c0, 500_000),
        # candle 1: 2nd green, new high -> G entry fires
        (o1, h1, l1, c1, 600_000),
        # candle 2: slight dip (still above entry, no exit yet)
        (c1, c1 * 1.005, c1 * 0.995, c1 * 1.002, 400_000),
        # candle 3: more dip
        (c1 * 1.002, c1 * 1.003, c1 * 0.990, c1 * 0.995, 350_000),
        # candle 4: continues dropping
        (c1 * 0.995, c1 * 0.996, c1 * 0.975, c1 * 0.980, 300_000),
        # candle 5: further drop
        (c1 * 0.980, c1 * 0.981, c1 * 0.965, c1 * 0.970, 280_000),
        # candle 6-9: flat / slight recovery
        (c1 * 0.970, c1 * 0.975, c1 * 0.965, c1 * 0.972, 250_000),
        (c1 * 0.972, c1 * 0.978, c1 * 0.968, c1 * 0.975, 240_000),
        (c1 * 0.975, c1 * 0.980, c1 * 0.970, c1 * 0.978, 230_000),
        (c1 * 0.978, c1 * 0.982, c1 * 0.972, c1 * 0.980, 220_000),
    ]
    opens = [c[0] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    closes = [c[3] for c in candles]
    volumes = [c[4] for c in candles]
    return make_candles(opens, highs, lows, closes, volumes, date=date)


def g_stop_candles(date="2025-01-15", base=BASE_PRICE, stop_pct=5.0):
    """Candles that trigger G entry then a STOP exit.

    Price drops below entry_price * (1 - stop_pct/100).
    """
    gap_mult = 1.50
    o0 = base * gap_mult
    c0 = o0 * 1.02
    h0 = c0 * 1.01
    l0 = o0 * 0.99

    o1 = c0
    c1 = o1 * 1.03
    h1 = c1 * 1.02
    l1 = o1 * 0.99

    # Entry happens around c1. Stop at c1 * (1 - stop_pct/100)
    stop_level = c1 * (1 - stop_pct / 100)

    candles = [
        (o0, h0, l0, c0, 500_000),
        (o1, h1, l1, c1, 600_000),
        # candle 2: big drop to trigger stop
        (c1, c1 * 0.99, stop_level * 0.99, stop_level * 0.995, 400_000),
    ]
    opens = [c[0] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    closes = [c[3] for c in candles]
    volumes = [c[4] for c in candles]
    return make_candles(opens, highs, lows, closes, volumes, date=date)


def g_trail_candles(date="2025-01-15", base=BASE_PRICE,
                    trail_pct=3.0, trail_activate_pct=2.0):
    """Candles that trigger G entry, rise to activate trail, then drop to trigger trail exit.

    Entry at c1. Trail activates when unrealized >= trail_activate_pct.
    Trail stop = highest_since_entry * (1 - trail_pct/100).
    """
    gap_mult = 1.50
    o0 = base * gap_mult
    c0 = o0 * 1.02
    h0 = c0 * 1.01
    l0 = o0 * 0.99

    o1 = c0
    c1 = o1 * 1.03
    h1 = c1 * 1.02
    l1 = o1 * 0.99

    # Candle 2: rise above entry to activate trail
    peak = c1 * (1 + trail_activate_pct / 100 + 0.01)  # just above activate
    c2 = peak * 0.995
    h2 = peak
    l2 = c2 * 0.998

    # Candle 3: continue up (new highest)
    peak2 = peak * 1.01
    c3 = peak2 * 0.998
    h3 = peak2
    l3 = c3 * 0.995

    # Candle 4: drop to trigger trail
    trail_stop = peak2 * (1 - trail_pct / 100)
    c4 = trail_stop * 0.995
    h4 = trail_stop * 1.002
    l4 = trail_stop * 0.99  # below trail_stop

    candles = [
        (o0, h0, l0, c0, 500_000),
        (o1, h1, l1, c1, 600_000),
        (c2, h2, l2, c2, 400_000),
        (c3, h3, l3, c3, 350_000),
        (c4, h4, l4, c4, 300_000),
    ]
    opens = [c[0] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    closes = [c[3] for c in candles]
    volumes = [c[4] for c in candles]
    return make_candles(opens, highs, lows, closes, volumes, date=date)


def g_target_candles(date="2025-01-15", base=BASE_PRICE, target_pct=10.0):
    """Candles that trigger G entry then hit TARGET exit."""
    gap_mult = 1.50
    o0 = base * gap_mult
    c0 = o0 * 1.02
    h0 = c0 * 1.01
    l0 = o0 * 0.99

    o1 = c0
    c1 = o1 * 1.03
    h1 = c1 * 1.02
    l1 = o1 * 0.99

    # Candle 2: spike to target
    target = c1 * (1 + target_pct / 100)
    c2 = target * 0.99
    h2 = target * 1.001  # just above target
    l2 = c1 * 0.995

    candles = [
        (o0, h0, l0, c0, 500_000),
        (o1, h1, l1, c1, 600_000),
        (c2, h2, l2, c2, 400_000),
    ]
    opens = [c[0] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    closes = [c[3] for c in candles]
    volumes = [c[4] for c in candles]
    return make_candles(opens, highs, lows, closes, volumes, date=date)


# ---------------------------------------------------------------------------
# Two-G-trade day (G1 + G2) for split-param testing
# ---------------------------------------------------------------------------
def two_g_trades_candles(date="2025-01-15", base=BASE_PRICE,
                         g1_stop=5.0, g2_stop=3.0):
    """Day with two G entries: G1 stops out, then G2 enters and stops out.

    G1 enters candle 1, stops out candle 2.
    G2 enters candle 3 (re-entry), stops out candle 4.
    """
    gap_mult = 1.50
    o0 = base * gap_mult
    c0 = o0 * 1.02
    h0 = c0 * 1.01
    l0 = o0 * 0.99

    # Candle 1: 2nd green, new high -> G1 entry
    o1 = c0
    c1 = o1 * 1.03
    h1 = c1 * 1.02
    l1 = o1 * 0.99

    # Candle 2: G1 stop
    g1_stop_level = c1 * (1 - g1_stop / 100)
    c2 = g1_stop_level * 0.995
    h2 = c1 * 1.001
    l2 = g1_stop_level * 0.99  # below stop

    # Candle 3: new signal for G2 (need green + new high again)
    o3 = c2
    c3 = o3 * 1.05  # big green
    h3 = c3 * 1.02
    l3 = o3 * 0.995

    # Candle 4: G2 stop
    g2_stop_level = c3 * (1 - g2_stop / 100)
    c4 = g2_stop_level * 0.995
    h4 = c3 * 1.001
    l4 = g2_stop_level * 0.99

    candles = [
        (o0, h0, l0, c0, 500_000),
        (o1, h1, l1, c1, 600_000),
        (c2, h2, l2, c2, 400_000),
        (o3, h3, l3, c3, 500_000),
        (c4, h4, l4, c4, 400_000),
    ]
    opens = [c[0] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    closes = [c[3] for c in candles]
    volumes = [c[4] for c in candles]
    return make_candles(opens, highs, lows, closes, volumes, date=date)


# ---------------------------------------------------------------------------
# Standard strategy param builders
# ---------------------------------------------------------------------------
def make_hgaf_params(strat_letter, target_pct=10.0, time_limit=15,
                      stop_pct=5.0, trail_pct=3.0, trail_act=2.0,
                      gap_pct=30.0, enable=True):
    """Build params dict for H/G/A/F strategies (share exit pattern)."""
    prefix = strat_letter.lower()
    params = {
        f"enable_{s}": (s == strat_letter.lower()) if enable else False
        for s in ALL_STRATS if s != "x"
    }
    params[f"enable_{strat_letter.lower()}"] = enable
    params[f"{prefix}_min_gap_pct"] = gap_pct
    params[f"{prefix}_target_pct"] = target_pct
    params[f"{prefix}_time_limit_min"] = time_limit
    params[f"{prefix}_stop_pct"] = stop_pct
    params[f"{prefix}_trail_pct"] = trail_pct
    params[f"{prefix}_trail_activate_pct"] = trail_act
    params["priority_g"] = 1
    return params


def make_split_g_params(g1_stop=5.0, g1_trail=3.0, g1_trail_act=2.0,
                         g1_target=10.0, g1_time=15,
                         g2_stop=3.0, g2_trail=1.0, g2_trail_act=4.0,
                         g2_target=8.0, g2_time=10,
                         g_min_gap=30.0):
    """Build split-param dict for G1/G2 testing.

    Includes default L params since _map_split_to_standard() expects both.
    """
    params = {f"enable_{s}": False for s in ALL_STRATS if s != "x"}
    params["enable_g"] = True
    params["g_min_gap_pct"] = g_min_gap
    params["g_require_2nd_green"] = True
    params["g_require_2nd_new_high"] = True

    # G1 exit
    params["g1_target_pct"] = g1_target
    params["g1_time_limit_min"] = g1_time
    params["g1_stop_pct"] = g1_stop
    params["g1_trail_pct"] = g1_trail
    params["g1_trail_activate_pct"] = g1_trail_act

    # G2 exit
    params["g2_target_pct"] = g2_target
    params["g2_time_limit_min"] = g2_time
    params["g2_stop_pct"] = g2_stop
    params["g2_trail_pct"] = g2_trail
    params["g2_trail_activate_pct"] = g2_trail_act

    # Default L params (required by _map_split_to_standard)
    params["l_min_gap"] = 30.0
    params["l_max_float"] = 10_000_000
    params["l_earliest_candle"] = 5
    params["l_latest_candle"] = 120
    params["l_vol_surge_mult"] = 2.0
    params["l_min_price_accel_pct"] = 1.0
    params["l_tier1_float"] = 1_000_000
    params["l_tier2_float"] = 5_000_000
    for tier in ["1", "2", "3"]:
        params[f"l1_tier{tier}_target1_pct"] = 20.0
        params[f"l1_tier{tier}_target2_pct"] = 40.0
        params[f"l2_tier{tier}_target1_pct"] = 15.0
        params[f"l2_tier{tier}_target2_pct"] = 30.0
    params["l1_stop_pct"] = 10.0
    params["l1_partial_sell_pct"] = 25.0
    params["l1_trail_pct"] = 3.0
    params["l1_trail_activate_pct"] = 2.0
    params["l1_time_limit_min"] = 60
    params["l2_stop_pct"] = 8.0
    params["l2_partial_sell_pct"] = 25.0
    params["l2_trail_pct"] = 2.0
    params["l2_trail_activate_pct"] = 3.0
    params["l2_time_limit_min"] = 45

    params["priority_g"] = 1
    return params


def make_split_l_params(l1_stop=10.0, l1_trail=3.0, l1_trail_act=2.0,
                         l1_time=60,
                         l2_stop=8.0, l2_trail=2.0, l2_trail_act=3.0,
                         l2_time=45,
                         l_min_gap=30.0, l_max_float=10_000_000):
    """Build split-param dict for L1/L2 testing.

    Includes default G params since _map_split_to_standard() expects both.
    """
    params = {f"enable_{s}": False for s in ALL_STRATS if s != "x"}
    params["enable_l"] = True
    params["l_min_gap"] = l_min_gap
    params["l_max_float"] = l_max_float
    params["l_earliest_candle"] = 5
    params["l_latest_candle"] = 120
    params["l_vol_surge_mult"] = 2.0
    params["l_min_price_accel_pct"] = 1.0
    params["l_tier1_float"] = 1_000_000
    params["l_tier2_float"] = 5_000_000

    # L1 exit
    params["l1_tier1_target1_pct"] = 20.0
    params["l1_tier1_target2_pct"] = 40.0
    params["l1_tier2_target1_pct"] = 12.0
    params["l1_tier2_target2_pct"] = 30.0
    params["l1_tier3_target1_pct"] = 8.0
    params["l1_tier3_target2_pct"] = 20.0
    params["l1_stop_pct"] = l1_stop
    params["l1_partial_sell_pct"] = 25.0
    params["l1_trail_pct"] = l1_trail
    params["l1_trail_activate_pct"] = l1_trail_act
    params["l1_time_limit_min"] = l1_time

    # L2 exit
    params["l2_tier1_target1_pct"] = 15.0
    params["l2_tier1_target2_pct"] = 30.0
    params["l2_tier2_target1_pct"] = 10.0
    params["l2_tier2_target2_pct"] = 25.0
    params["l2_tier3_target1_pct"] = 6.0
    params["l2_tier3_target2_pct"] = 15.0
    params["l2_stop_pct"] = l2_stop
    params["l2_partial_sell_pct"] = 25.0
    params["l2_trail_pct"] = l2_trail
    params["l2_trail_activate_pct"] = l2_trail_act
    params["l2_time_limit_min"] = l2_time

    # Default G params (required by _map_split_to_standard)
    params["g_min_gap_pct"] = 30.0
    params["g_require_2nd_green"] = True
    params["g_require_2nd_new_high"] = True
    params["g1_target_pct"] = 10.0
    params["g1_time_limit_min"] = 15
    params["g1_stop_pct"] = 5.0
    params["g1_trail_pct"] = 3.0
    params["g1_trail_activate_pct"] = 2.0
    params["g2_target_pct"] = 8.0
    params["g2_time_limit_min"] = 10
    params["g2_stop_pct"] = 3.0
    params["g2_trail_pct"] = 1.0
    params["g2_trail_activate_pct"] = 4.0

    params["priority_l"] = 7
    return params


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def param_lock():
    """Acquire the global param lock for tests that mutate tgc globals."""
    _param_lock.acquire()
    yield
    _param_lock.release()


@pytest.fixture
def clean_tgc_globals():
    """Save and restore tgc globals around a test."""
    saved = {k: getattr(tgc, k) for k in dir(tgc) if k.isupper() and not k.startswith("_")}
    yield
    for k, v in saved.items():
        setattr(tgc, k, v)


@pytest.fixture
def slippage_on():
    """Ensure slippage settings are ON."""
    old_dyn = tgc.USE_DYNAMIC_SLIPPAGE
    old_mw = tgc.USE_MULTIWINDOW_SLIPPAGE
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    yield
    tgc.USE_DYNAMIC_SLIPPAGE = old_dyn
    tgc.USE_MULTIWINDOW_SLIPPAGE = old_mw
