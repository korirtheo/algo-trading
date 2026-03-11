"""
Combined Green Candle Strategy Backtest (H+G+A+F+D+V+M+R+P+W on shared balance)
=================================================================================
Strategy H: Gap>=35% + body>=4% + 2nd green + new hi + vol confirm -> +16% target, 15m
Strategy G: Gap>=30% + 2nd green + new hi       -> +11% target, 10m time stop
Strategy A: Gap>=15% + body>=4% + 2nd green + new hi -> +6% target, 12m time stop
Strategy F: Gap>=10% + 2nd green                 -> +8% target, 3m time stop (catch-all)
Strategy D: Gap>=10% + opening spike + dip + 5-candle high -> partial +12%, runner +18%
            Trailing stop 2% (activates at +2%), hard stop -9%, 30m time limit
Strategy M: Gap>=20% + morning spike + midday consolidation + breakout -> partial +6%
            Trailing stop 4% (activates at +5%), hard stop -10%, 120m time limit
Strategy V: Gap>=24% + below VWAP 4+ candles + reclaim with volume -> partial +9%, runner +17%
            Trailing stop 2% (activates at +2%), hard stop -10%, 100m time limit
Strategy R: Day 1 gap>=100%, Day 2 pullback+bounce above D1 close -> +19% target
            Trailing stop 4% (activates at +8%), hard stop -10%, 180m time limit
Strategy P: Gap>=10% + PM high breakout + pullback + bounce -> partial +9%, runner +13%
            Trailing stop 2% (activates at +2%), hard stop -12%, 40m time limit
Strategy W: Gap>=8% + morning run + consolidation + power hour breakout -> +6% target
            Trailing stop 1% (activates at +3.5%), hard stop -3.5%, EOD exit

Priority: H > G > A > F > D > V > M > R > P > W (highest conviction first).
Single shared cash balance, no position limits. Full balance sizing per trade.
No SPY regime filter.

Usage:
  python test_green_candle_combined.py
  python test_green_candle_combined.py --no-charts
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
from datetime import datetime

import json

from test_full import (
    load_all_picks,
    _is_warrant_or_unit,
    SLIPPAGE_PCT,
    STARTING_CASH,
    TOP_N,
    MARGIN_THRESHOLD,
    VOL_CAP_PCT,
    ET_TZ,
)

import io, sys as _sys
if hasattr(_sys.stdout, 'buffer'):
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8",
                                    errors="replace", line_buffering=True)

# --- FLOAT DATA (for strategy L) ---
FLOAT_DATA = {}
_float_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "float_data.json")
if os.path.exists(_float_path):
    with open(_float_path) as _f:
        _raw = json.load(_f)
    for _tk, _v in _raw.items():
        if isinstance(_v, dict) and _v.get("floatShares"):
            FLOAT_DATA[_tk] = _v["floatShares"]

# --- STRATEGY H CONFIG: High Conviction (filtered G) ---
H_MIN_GAP_PCT = 35.0
H_MIN_BODY_PCT = 4.0
H_REQUIRE_VOL_CONFIRM = True
H_TARGET_PCT = 16.0
H_TIME_LIMIT_MINUTES = 15
H_STOP_PCT = 0.0                 # 0 = no hard stop (legacy behavior)
H_TRAIL_PCT = 0.0                # 0 = no trailing stop
H_TRAIL_ACTIVATE_PCT = 0.0

# --- STRATEGY G CONFIG: Big Gap Runner ---
G_MIN_GAP_PCT = 30.0
G_MIN_BODY_PCT = 0.0
G_REQUIRE_2ND_GREEN = True
G_REQUIRE_2ND_NEW_HIGH = True
G_TARGET_PCT = 11.0
G_TIME_LIMIT_MINUTES = 10
G_STOP_PCT = 0.0
G_TRAIL_PCT = 0.0
G_TRAIL_ACTIVATE_PCT = 0.0

# --- STRATEGY A CONFIG: Quick Scalp ---
A_MIN_GAP_PCT = 15.0
A_MIN_BODY_PCT = 4.0
A_MAX_BODY_PCT = 999
A_REQUIRE_2ND_GREEN = True
A_REQUIRE_2ND_NEW_HIGH = True
A_TARGET_PCT = 6.0
A_TIME_LIMIT_MINUTES = 12
A_STOP_PCT = 0.0
A_TRAIL_PCT = 0.0
A_TRAIL_ACTIVATE_PCT = 0.0

# --- STRATEGY F CONFIG: Catch-All ---
F_MIN_GAP_PCT = 10.0
F_MIN_BODY_PCT = 0.0
F_REQUIRE_2ND_GREEN = True
F_REQUIRE_2ND_NEW_HIGH = False
F_TARGET_PCT = 8.0
F_TIME_LIMIT_MINUTES = 3
F_STOP_PCT = 0.0
F_TRAIL_PCT = 0.0
F_TRAIL_ACTIVATE_PCT = 0.0

# --- STRATEGY D CONFIG: Opening Dip Buy (Optuna v2 optimized) ---
# Fallback: fires when H/G/A/F don't classify, alongside P but fires earlier
D_MIN_GAP_PCT = 30.0             # Optuna v2: raised from 10%
D_MIN_SPIKE_PCT = 10.0           # Optuna v2: raised from 6%
D_SPIKE_WINDOW = 20              # Optuna v2: widened from 5
D_DIP_PCT = 6.0                  # Optuna v2: tightened from 8%
D_ENTRY_MODE = "5candle"         # "5candle" or "vwap"
D_MAX_ENTRY_CANDLE = 45          # Optuna v2: widened from 15
D_TARGET1_PCT = 12.0             # Target (no partial sell)
D_TARGET2_PCT = 12.0             # Same as target1 (partial_sell=0)
D_STOP_PCT = 9.0                 # Hard stop per entry
D_TIME_LIMIT_MINUTES = 70        # Optuna v2: widened from 30
D_PARTIAL_SELL_PCT = 0.0         # Optuna v2: no partial sell — trail only
D_TRAIL_PCT = 2.0                # Fixed trailing stop %
D_TRAIL_ACTIVATE_PCT = 2.0       # Start trailing after +2% unrealized

# --- STRATEGY M CONFIG: Midday Range Break ---
# Morning spike → midday consolidation → afternoon breakout
M_MIN_GAP_PCT = 10.0
M_MORNING_SPIKE_PCT = 8.0        # Morning high >= 8% above open
M_MORNING_CANDLES = 40           # Check spike in first 40 candles
M_RANGE_START_CANDLE = 55        # Consolidation starts at candle 55
M_CONSOLIDATION_LEN = 40         # Consolidation lasts 40 candles (ends c95)
M_MAX_RANGE_PCT = 7.0            # Consolidation range <= 7%
M_VOL_RATIO = 0.5                # Consolidation vol <= 0.5x morning vol
M_MAX_ENTRY_CANDLE = 150         # Must enter by candle 150
M_TARGET1_PCT = 6.0              # Partial target: sell 50% here
M_STOP_PCT = 10.0                # Hard stop per entry
M_TIME_LIMIT_MINUTES = 120       # 60 candles * 2 min
M_PARTIAL_SELL_PCT = 50.0        # Sell 50% at target1
M_TRAIL_PCT = 4.0                # Trailing stop %
M_TRAIL_ACTIVATE_PCT = 5.0       # Start trailing after +5% unrealized

# --- STRATEGY V CONFIG: VWAP Reclaim (Optuna v2 optimized) ---
# Gap-up sells off below VWAP, then reclaims with volume -> buy the reclaim
V_MIN_GAP_PCT = 14.0             # Optuna v2: lowered from 24%
V_MIN_BELOW_CANDLES = 7          # Optuna v2: raised from 4
V_MIN_BELOW_PCT = 1.0            # Must dip at least 1% below VWAP
V_VOL_SPIKE_RATIO = 3.5          # Optuna v2: raised from 1.0
V_MAX_ENTRY_CANDLE = 60          # Optuna v2: lowered from 80
V_TARGET1_PCT = 3.0              # Optuna v2: lowered from 9%
V_TARGET2_PCT = 19.0             # Optuna v2: raised from 17%
V_STOP_PCT = 9.0                 # Optuna v2: lowered from 10%
V_TIME_LIMIT_MINUTES = 100       # 50 candles * 2 min
V_PARTIAL_SELL_PCT = 25.0        # Sell 25% at target1
V_TRAIL_PCT = 3.0                # Optuna v2: raised from 2%
V_TRAIL_ACTIVATE_PCT = 5.0       # Optuna v2: raised from 2%

# --- STRATEGY R CONFIG: Multi-Day Runner ---
# Day 1: massive gap-up. Day 2: pullback then bounce continuation.
R_DAY1_MIN_GAP = 40.0
R_D2_PULLBACK_PCT = 10.0         # Optuna v2: raised from 3%
R_PULLBACK_WINDOW = 30           # Pullback within first 30 candles
R_BOUNCE_REF = "d2_open"         # Optuna v2: changed from d1_close
R_MAX_ENTRY_CANDLE = 55          # Optuna v2: raised from 20
R_TARGET1_PCT = 9.0              # Optuna v2: lowered from 19%
R_STOP_PCT = 9.0                 # Optuna v2: lowered from 10%
R_TRAIL_PCT = 6.0                # Optuna v2: raised from 4%
R_TRAIL_ACTIVATE_PCT = 6.0       # Optuna v2: lowered from 8%
R_TIME_LIMIT_MINUTES = 100       # Optuna v2: reduced from 180

# --- STRATEGY P CONFIG: PM High Breakout + Pullback + Bounce (Optuna v2 optimized) ---
# Only fires when H/G/A/F did not classify the stock (fallback)
# Advanced exits: trailing stop (no partial sell — full trail)
P_MIN_GAP_PCT = 10.0            # Minimum gap
P_CONFIRM_ABOVE = 3             # Optuna v2: 3 candles above PM high
P_CONFIRM_WINDOW = 3            # Optuna v2: 3 candle window
P_PULLBACK_PCT = 9.0            # Optuna v2: widened from 7%
P_PULLBACK_TIMEOUT = 10         # Optuna v2: tightened from 30
P_MAX_ENTRY_CANDLE = 105        # Optuna v2: raised from 75
P_TARGET1_PCT = 15.0            # Optuna v2: raised from 9%
P_TARGET2_PCT = 15.0            # Same as target1 (partial_sell=0)
P_STOP_PCT = 12.0               # Hard stop
P_TIME_LIMIT_MINUTES = 180      # Optuna v2: raised from 40
P_PARTIAL_SELL_PCT = 0.0        # Optuna v2: no partial sell — trail only
P_TRAIL_PCT = 2.0               # Fixed trailing stop %
P_TRAIL_ACTIVATE_PCT = 2.0      # Start trailing after +2% unrealized

# --- STRATEGY W CONFIG: Power Hour Breakout ---
# Gap-up + morning run + all-day consolidation + 3 PM+ volume breakout
W_MIN_GAP_PCT = 10.0
W_MIN_MORNING_RUN = 4.0          # Optuna v2: raised from 3%
W_CONSOL_START = 35              # Optuna v2: earlier start
W_MAX_RANGE_PCT = 10.0           # Optuna v2: tightened from 16%
W_MAX_VWAP_DEV_PCT = 5.0         # Optuna v2: tightened from 11%
W_EARLIEST_CANDLE = 165          # Optuna v2: later start
W_LATEST_CANDLE = 190            # Optuna v2: later end
W_VOL_SURGE_MULT = 2.0           # Optuna v2: lowered from 3x
W_VOL_VS_MORNING_MULT = 0.1      # Breakout vol >= 0.1x morning spike vol
W_MAX_HOD_BREAKS = 3             # Max HOD breaks before entry window
W_REQUIRE_ABOVE_VWAP = True      # Must be above VWAP at breakout
W_TARGET_PCT = 8.0               # Optuna v2: raised from 6%
W_STOP_PCT = 3.0                 # Optuna v2: tightened from 3.5%
W_TRAIL_PCT = 2.5                # Optuna v2: raised from 1%
W_TRAIL_ACTIVATE_PCT = 2.0       # Optuna v2: lowered from 3.5%

# --- STRATEGY L CONFIG: Low Float Squeeze (optimized trial #486) ---
L_MAX_FLOAT = 15_000_000        # Float shares threshold
L_MIN_GAP_PCT = 30.0            # Minimum gap %
L_EARLIEST_CANDLE = 8           # Don't enter too early
L_LATEST_CANDLE = 115           # Latest possible entry candle
L_HOD_BREAK_REQUIRED = True     # Must break to new HOD for entry
L_VOL_SURGE_MULT = 1.5          # Current candle vol >= Nx avg of last 10
L_MIN_PRICE_ACCEL_PCT = 1.0     # Min green candle body % for entry candle
L_REQUIRE_ABOVE_VWAP = True     # Must be above VWAP at entry
# Float-tiered targets
L_TIER1_FLOAT = 1_000_000       # Ultra-low float boundary
L_TIER2_FLOAT = 5_000_000       # Low float boundary
L_TIER1_TARGET1_PCT = 30.0
L_TIER1_TARGET2_PCT = 40.0
L_TIER2_TARGET1_PCT = 15.0
L_TIER2_TARGET2_PCT = 40.0
L_TIER3_TARGET1_PCT = 9.0
L_TIER3_TARGET2_PCT = 32.0
L_STOP_PCT = 14.0               # Hard stop (wide for low float volatility)
L_PARTIAL_SELL_PCT = 0.0        # No partial sell
L_TRAIL_PCT = 1.0               # Tight trailing stop %
L_TRAIL_ACTIVATE_PCT = 2.0      # Start trailing early at +2%
L_TIME_LIMIT_MINUTES = 70       # Time limit in minutes

# --- STRATEGY O CONFIG: Opening Range Breakout ---
O_MIN_GAP_PCT = 10.0
O_RANGE_CANDLES = 5              # candles to form opening range (5 = 10 min)
O_BREAKOUT_VOL_MULT = 1.5       # breakout candle vol vs avg range vol
O_MAX_ENTRY_CANDLE = 30
O_TARGET1_PCT = 8.0
O_TARGET2_PCT = 15.0
O_STOP_PCT = 0.0                # 0 = dynamic stop at range low
O_PARTIAL_SELL_PCT = 50.0
O_TRAIL_PCT = 2.0
O_TRAIL_ACTIVATE_PCT = 3.0
O_TIME_LIMIT_MINUTES = 60

# --- STRATEGY B CONFIG: Red-to-Green (R2G) ---
B_MIN_GAP_PCT = 15.0
B_MAX_DIP_PCT = 5.0             # max dip below open before abandoning
B_MIN_RECLAIM_VOL_MULT = 1.5    # volume surge on reclaim candle
B_MAX_ENTRY_CANDLE = 20
B_TARGET1_PCT = 6.0
B_TARGET2_PCT = 12.0
B_STOP_PCT = 4.0
B_PARTIAL_SELL_PCT = 50.0
B_TRAIL_PCT = 2.0
B_TRAIL_ACTIVATE_PCT = 3.0
B_TIME_LIMIT_MINUTES = 30

# --- STRATEGY K CONFIG: First Pullback Buy ---
K_MIN_GAP_PCT = 10.0
K_MIN_RUN_PCT = 5.0             # min morning run-up before pullback
K_RUN_WINDOW = 15               # candles to establish run
K_PULLBACK_PCT = 3.0            # min pullback % from run high
K_PULLBACK_VOL_RATIO = 0.5      # pullback vol <= ratio * run vol (orderly)
K_BOUNCE_VOL_MULT = 1.5         # bounce vol >= mult * pullback avg vol
K_MAX_ENTRY_CANDLE = 45
K_TARGET1_PCT = 8.0
K_TARGET2_PCT = 15.0
K_STOP_PCT = 5.0
K_PARTIAL_SELL_PCT = 50.0
K_TRAIL_PCT = 2.0
K_TRAIL_ACTIVATE_PCT = 3.0
K_TIME_LIMIT_MINUTES = 60

# --- STRATEGY C CONFIG: Micro Flag / Base Pattern ---
C_MIN_GAP_PCT = 10.0
C_MIN_SPIKE_PCT = 5.0           # initial spike before consolidation
C_MIN_BASE_CANDLES = 3          # min candles in tight base
C_MAX_BASE_CANDLES = 8          # max candles in base before abandon
C_MAX_BASE_RANGE_PCT = 3.0      # max range of base (tight consolidation)
C_BREAKOUT_VOL_MULT = 1.5       # vol surge on breakout
C_MAX_ENTRY_CANDLE = 60
C_TARGET1_PCT = 8.0
C_TARGET2_PCT = 15.0
C_STOP_PCT = 4.0
C_PARTIAL_SELL_PCT = 50.0
C_TRAIL_PCT = 2.0
C_TRAIL_ACTIVATE_PCT = 3.0
C_TIME_LIMIT_MINUTES = 60

# --- STRATEGY S CONFIG: Stuff-and-Break ---
S_MIN_GAP_PCT = 10.0
S_MIN_HOD_TESTS = 2             # min times HOD tested and rejected
S_HOD_TOLERANCE_PCT = 0.5       # within X% of HOD = "test"
S_REJECTION_PCT = 1.0           # must pull back X% to count as rejection
S_BREAKOUT_VOL_MULT = 1.5       # vol surge on final break
S_MAX_ENTRY_CANDLE = 90
S_TARGET1_PCT = 8.0
S_TARGET2_PCT = 15.0
S_STOP_PCT = 4.0
S_PARTIAL_SELL_PCT = 50.0
S_TRAIL_PCT = 2.0
S_TRAIL_ACTIVATE_PCT = 3.0
S_TIME_LIMIT_MINUTES = 90

# --- STRATEGY E CONFIG: Gap-and-Go RelVol ---
E_MIN_GAP_PCT = 15.0
E_MIN_PM_VOL_MULT = 5.0         # premarket vol >= X * typical daily avg
E_MAX_ENTRY_CANDLE = 5           # enter very early
E_TARGET1_PCT = 6.0
E_TARGET2_PCT = 12.0
E_STOP_PCT = 4.0
E_PARTIAL_SELL_PCT = 50.0
E_TRAIL_PCT = 2.0
E_TRAIL_ACTIVATE_PCT = 3.0
E_TIME_LIMIT_MINUTES = 20

# --- STRATEGY I CONFIG: P1 Immediate PM High Breakout ---
I_MIN_GAP_PCT = 10.0
I_MAX_ENTRY_CANDLE = 30          # must break PM high early
I_BREAKOUT_VOL_MULT = 1.5
I_TARGET1_PCT = 8.0
I_TARGET2_PCT = 15.0
I_STOP_PCT = 5.0
I_PARTIAL_SELL_PCT = 50.0
I_TRAIL_PCT = 2.0
I_TRAIL_ACTIVATE_PCT = 3.0
I_TIME_LIMIT_MINUTES = 60

# --- STRATEGY J CONFIG: P3 VWAP + PM High Breakout ---
J_MIN_GAP_PCT = 10.0
J_MAX_ENTRY_CANDLE = 90
J_VWAP_PROXIMITY_PCT = 2.0      # price within X% of VWAP at breakout
J_TARGET1_PCT = 8.0
J_TARGET2_PCT = 15.0
J_STOP_PCT = 5.0
J_PARTIAL_SELL_PCT = 50.0
J_TRAIL_PCT = 2.0
J_TRAIL_ACTIVATE_PCT = 3.0
J_TIME_LIMIT_MINUTES = 90

# --- STRATEGY N CONFIG: P4 HOD Reclaim ---
N_MIN_GAP_PCT = 10.0
N_MIN_HOD_AGE = 10              # HOD must be at least X candles old
N_PULLBACK_FROM_HOD_PCT = 3.0   # must pull back X% from HOD
N_MAX_ENTRY_CANDLE = 120
N_TARGET1_PCT = 8.0
N_TARGET2_PCT = 15.0
N_STOP_PCT = 5.0
N_PARTIAL_SELL_PCT = 50.0
N_TRAIL_PCT = 2.0
N_TRAIL_ACTIVATE_PCT = 3.0
N_TIME_LIMIT_MINUTES = 90

# --- SHARED CONFIG ---
EOD_EXIT_MINUTES = 15
FULL_BALANCE_SIZING = True      # Use full balance for each trade

# Strategy priority (lower = higher priority; Optuna can override)
STRAT_PRIORITY = {
    "H": 0, "G": 1, "A": 2, "F": 3, "D": 4, "V": 5, "P": 6,
    "M": 7, "R": 8, "W": 9,
    "O": 10, "B": 11, "K": 12, "C": 13, "S": 14, "E": 15,
    "I": 16, "J": 17, "N": 18, "L": 19
}
STRAT_KEYS = ["H","G","A","F","D","V","P","M","R","W","O","B","K","C","S","E","I","J","N","L"]



def _load_r_intraday(ticker, target_date, data_dirs):
    """Load intraday CSV for a ticker and filter to target_date market hours."""
    for d in data_dirs:
        csv_path = os.path.join(d, "intraday", f"{ticker}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if df.empty:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            day_data = df[df.index.date == pd.Timestamp(target_date).date()]
            if len(day_data) < 20:
                continue
            day_data = day_data.between_time("09:30", "15:58")
            if len(day_data) < 20:
                continue
            # Re-localize to UTC for consistency with other picks
            day_data.index = day_data.index.tz_localize("America/New_York").tz_convert("UTC")
            return day_data
        except Exception:
            continue
    return None


def _compute_vwap(mh):
    """Compute cumulative VWAP from market-hour candle DataFrame."""
    h = mh["High"].values.astype(float)
    l = mh["Low"].values.astype(float)
    c = mh["Close"].values.astype(float)
    v = mh["Volume"].values.astype(float)
    typical = (h + l + c) / 3.0
    cum_tp_vol = np.cumsum(typical * v)
    cum_vol = np.cumsum(v)
    cum_vol[cum_vol == 0] = 1e-9
    return cum_tp_vol / cum_vol


def _get_tiered_targets(float_shares):
    """Return (target1_pct, target2_pct) based on float size."""
    if float_shares < L_TIER1_FLOAT:
        return L_TIER1_TARGET1_PCT, L_TIER1_TARGET2_PCT
    elif float_shares < L_TIER2_FLOAT:
        return L_TIER2_TARGET1_PCT, L_TIER2_TARGET2_PCT
    else:
        return L_TIER3_TARGET1_PCT, L_TIER3_TARGET2_PCT


def _classify_candle2(gap_pct, body_pct, second_green, second_new_high, vol_confirm=False):
    """Classify on candle 2 for strategies H, G, A, F.
    Priority: H > G > A > F (highest conviction first)."""
    # H: high conviction - gap>=25%, body>=5%, 2nd green + new hi + vol confirm
    if (gap_pct >= H_MIN_GAP_PCT
            and body_pct >= H_MIN_BODY_PCT
            and second_green and second_new_high
            and (not H_REQUIRE_VOL_CONFIRM or vol_confirm)):
        return "H"
    # G: gap>=25%, 2nd green + new hi (strong runners)
    if (gap_pct >= G_MIN_GAP_PCT
            and body_pct >= G_MIN_BODY_PCT
            and (not G_REQUIRE_2ND_GREEN or second_green)
            and (not G_REQUIRE_2ND_NEW_HIGH or second_new_high)):
        return "G"
    # A: gap>=15%, 2nd green + new hi (quick scalp)
    if (gap_pct >= A_MIN_GAP_PCT
            and A_MIN_BODY_PCT <= body_pct <= A_MAX_BODY_PCT
            and (not A_REQUIRE_2ND_GREEN or second_green)
            and (not A_REQUIRE_2ND_NEW_HIGH or second_new_high)):
        return "A"
    # F: gap>=10%, 2nd green (catch-all)
    if (gap_pct >= F_MIN_GAP_PCT
            and body_pct >= F_MIN_BODY_PCT
            and (not F_REQUIRE_2ND_GREEN or second_green)
            and (not F_REQUIRE_2ND_NEW_HIGH or second_new_high)):
        return "F"
    return None


def simulate_day_combined(picks, cash, cash_account=False):
    """Simulate combined strategy for one day with single cash pool.
    cash: float, available cash.
    Returns: (states, cash, unsettled, selection_log)
    """
    cash_box = [float(cash)]
    unsettled_box = [0.0]
    selection_log = []

    def _receive_proceeds(amount):
        if cash_account:
            unsettled_box[0] += amount
        else:
            cash_box[0] += amount

    # Build unified timestamp index
    all_timestamps = set()
    for pick in picks:
        mh = pick.get("market_hour_candles")
        if mh is not None and len(mh) > 0:
            all_timestamps.update(mh.index.tolist())
    all_timestamps = sorted(all_timestamps)

    if not all_timestamps:
        return [], cash_box[0], unsettled_box[0], []

    # Initialize states
    states = []
    for pick in picks:
        mh = pick.get("market_hour_candles")
        if mh is None or len(mh) == 0:
            continue
        vwap = _compute_vwap(mh)
        ticker = pick["ticker"]
        float_shares = FLOAT_DATA.get(ticker)
        l_eligible = (float_shares is not None
                      and float_shares <= L_MAX_FLOAT
                      and pick["gap_pct"] >= L_MIN_GAP_PCT)
        l_tgt1, l_tgt2 = _get_tiered_targets(float_shares) if l_eligible else (0, 0)
        states.append({
            "ticker": ticker,
            "premarket_high": pick["premarket_high"],
            "gap_pct": pick["gap_pct"],
            "pm_volume": pick.get("pm_volume", 0),
            "mh": mh,
            "vwap": vwap,
            # State — H/G/A/F candle classification
            "candle_count": 0,
            "first_candle_ok": False,
            "first_candle_high": 0.0,
            "first_candle_body_pct": 0.0,
            "first_candle_volume": 0.0,      # For vol confirm (H)
            "signal": False,
            "signal_price": None,
            "open_price": None,
            "strategy": None,  # "H", "G", "A", "F", "D", or "P"
            # State — D (Opening Dip Buy fallback)
            "d_eligible": False,         # Set True after candle 2 if H/G/A/F failed
            "d_spike_high": 0.0,         # Highest high in spike window
            "d_spike_detected": False,   # Spike >= min_spike_pct above open
            "d_dip_detected": False,     # Low hit dip level
            "d_below_vwap": False,       # For VWAP entry mode
            "d_recent_closes": [],       # For 5-candle high tracking
            # State — D advanced exit management
            "d_highest_since_entry": 0.0,
            "d_trailing_active": False,
            "d_partial_taken": False,
            "d_partial_proceeds": 0.0,
            # State — M (Midday Range Break fallback)
            "m_eligible": False,
            "m_morning_spike_ok": False,
            "m_consol_checked": False,
            "m_consol_high": 0.0,
            "m_breakout_ready": False,
            # State — M advanced exit management
            "m_highest_since_entry": 0.0,
            "m_trailing_active": False,
            "m_partial_taken": False,
            "m_partial_proceeds": 0.0,
            # State — V (VWAP Reclaim fallback)
            "v_eligible": False,
            "v_below_count": 0,
            "v_max_below_pct": 0.0,
            "v_reclaim_ready": False,
            # State — V advanced exit management
            "v_highest_since_entry": 0.0,
            "v_trailing_active": False,
            "v_partial_taken": False,
            "v_partial_proceeds": 0.0,
            # State — R (Multi-Day Runner)
            "is_r_candidate": pick.get("is_r_candidate", False),
            "r_day1_close": pick.get("r_day1_close", 0),
            "r_eligible": pick.get("is_r_candidate", False),
            "r_pullback_seen": False,
            # State — R advanced exit management
            "r_highest_since_entry": 0.0,
            "r_trailing_active": False,
            # State — P (PM high breakout fallback)
            "p_eligible": False,         # Set True after candle 2 if H/G/A/F failed
            "p_recent_closes": [],       # Track closes above PM high
            "p_breakout_confirmed": False,
            "p_pullback_detected": False,
            "p_candles_since_confirm": 0,
            # State — P advanced exit management
            "p_highest_since_entry": 0.0,
            "p_trailing_active": False,
            "p_partial_taken": False,
            "p_partial_proceeds": 0.0,
            # State — W (Power Hour Breakout fallback)
            "w_eligible": False,
            "w_morning_run_ok": False,
            "w_morning_high": 0.0,
            "w_morning_spike_vol": 0.0,
            "w_consol_checked": False,
            "w_consol_high": 0.0,
            "w_consol_avg_vol": 0.0,
            "w_breakout_ready": False,
            # State — W exit management
            "w_highest_since_entry": 0.0,
            "w_trailing_active": False,
            # State — L (Low Float Squeeze)
            "l_eligible": l_eligible,
            "l_float_shares": float_shares or 0,
            "l_running_hod": 0.0,
            "l_target1_pct": l_tgt1,
            "l_target2_pct": l_tgt2,
            # State — L exit management
            "l_highest_since_entry": 0.0,
            "l_trailing_active": False,
            "l_partial_taken": False,
            "l_partial_proceeds": 0.0,
            # State — O (Opening Range Breakout)
            "o_eligible": pick["gap_pct"] >= O_MIN_GAP_PCT,
            "o_range_high": 0.0,
            "o_range_low": 999999.0,
            "o_range_formed": False,
            "o_range_avg_vol": 0.0,
            "o_highest_since_entry": 0.0,
            "o_trailing_active": False,
            "o_partial_taken": False,
            "o_partial_proceeds": 0.0,
            "o_dynamic_stop": 0.0,       # range low used as stop
            # State — B (Red-to-Green R2G)
            "b_eligible": pick["gap_pct"] >= B_MIN_GAP_PCT,
            "b_candle1_red": False,
            "b_open_price": 0.0,
            "b_dip_seen": False,
            "b_avg_vol_sum": 0.0,
            "b_avg_vol_count": 0,
            "b_highest_since_entry": 0.0,
            "b_trailing_active": False,
            "b_partial_taken": False,
            "b_partial_proceeds": 0.0,
            # State — K (First Pullback Buy)
            "k_eligible": False,
            "k_run_high": 0.0,
            "k_run_detected": False,
            "k_run_vol_sum": 0.0,
            "k_pullback_low": 999999.0,
            "k_pullback_detected": False,
            "k_pullback_vol_sum": 0.0,
            "k_pullback_candles": 0,
            "k_highest_since_entry": 0.0,
            "k_trailing_active": False,
            "k_partial_taken": False,
            "k_partial_proceeds": 0.0,
            # State — C (Micro Flag / Base Pattern)
            "c_eligible": False,
            "c_spike_detected": False,
            "c_spike_high": 0.0,
            "c_base_start_candle": 0,
            "c_base_candle_count": 0,
            "c_base_high": 0.0,
            "c_base_low": 999999.0,
            "c_highest_since_entry": 0.0,
            "c_trailing_active": False,
            "c_partial_taken": False,
            "c_partial_proceeds": 0.0,
            # State — S (Stuff-and-Break)
            "s_eligible": False,
            "s_hod": 0.0,
            "s_hod_candle": 0,
            "s_hod_tests": 0,
            "s_last_test_candle": 0,
            "s_rejected": False,
            "s_highest_since_entry": 0.0,
            "s_trailing_active": False,
            "s_partial_taken": False,
            "s_partial_proceeds": 0.0,
            # State — E (Gap-and-Go RelVol)
            "e_eligible": (pick["gap_pct"] >= E_MIN_GAP_PCT
                           and pick.get("pm_volume", 0) > 0),
            "e_pm_vol_ok": False,        # checked at candle 1
            "e_highest_since_entry": 0.0,
            "e_trailing_active": False,
            "e_partial_taken": False,
            "e_partial_proceeds": 0.0,
            # State — I (P1 Immediate PM High Breakout)
            "i_eligible": False,
            "i_highest_since_entry": 0.0,
            "i_trailing_active": False,
            "i_partial_taken": False,
            "i_partial_proceeds": 0.0,
            # State — J (P3 VWAP + PM High Breakout)
            "j_eligible": False,
            "j_highest_since_entry": 0.0,
            "j_trailing_active": False,
            "j_partial_taken": False,
            "j_partial_proceeds": 0.0,
            # State — N (P4 HOD Reclaim)
            "n_eligible": False,
            "n_hod": 0.0,
            "n_hod_candle": 0,
            "n_pullback_seen": False,
            "n_pullback_low": 999999.0,
            "n_highest_since_entry": 0.0,
            "n_trailing_active": False,
            "n_partial_taken": False,
            "n_partial_proceeds": 0.0,
            # State — H/G/A/F trail tracking
            "hgaf_trail_stop": 0.0,
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
        })

    # Create dedicated L-only states for cross-pool signal independence.
    # Without this, H/G/A/F claim signal=True first and L never gets checked
    # on the same ticker. L-only states skip H/G/A/F detection entirely.
    l_only_states = []
    for st in states:
        if st["l_eligible"]:
            l_st = dict(st)  # shallow copy (shares mh/vwap refs, which are read-only)
            l_st["l_only"] = True
            # Disable all non-L strategies
            l_st["d_eligible"] = False
            l_st["v_eligible"] = False
            l_st["m_eligible"] = False
            l_st["p_eligible"] = False
            l_st["w_eligible"] = False
            l_st["is_r_candidate"] = False
            l_st["r_eligible"] = False
            l_st["r_pullback_seen"] = False
            l_st["o_eligible"] = False
            l_st["b_eligible"] = False
            l_st["e_eligible"] = False
            l_st["k_eligible"] = False
            l_st["c_eligible"] = False
            l_st["s_eligible"] = False
            l_st["i_eligible"] = False
            l_st["j_eligible"] = False
            l_st["n_eligible"] = False
            # Reset per-instance state
            l_st["signal"] = False
            l_st["signal_price"] = None
            l_st["strategy"] = None
            l_st["candle_count"] = 0
            l_st["first_candle_ok"] = False
            l_st["first_candle_body_pct"] = 0.0
            l_st["done"] = False
            l_st["entry_price"] = None
            l_st["shares"] = 0
            l_st["position_cost"] = 0.0
            l_st["pnl"] = 0.0
            l_st["l_running_hod"] = 0.0
            l_st["l_highest_since_entry"] = 0.0
            l_st["l_trailing_active"] = False
            l_st["l_partial_taken"] = False
            l_st["l_partial_proceeds"] = 0.0
            # Fresh mutable lists (avoid cross-state mutation)
            l_st["d_recent_closes"] = []
            l_st["p_recent_closes"] = []
            l_only_states.append(l_st)
            # Prevent original state from also detecting L (avoid double entries)
            st["l_eligible"] = False
    states.extend(l_only_states)

    # Create dedicated O-only states (ORB works regardless of candle 1 color)
    o_only_states = []
    for st in states:
        if st.get("l_only"):
            continue
        if st["o_eligible"]:
            o_st = dict(st)
            o_st["o_only"] = True
            # Disable everything except O
            for k in ["d_eligible","v_eligible","m_eligible","p_eligible","w_eligible",
                       "k_eligible","c_eligible","s_eligible","i_eligible","j_eligible","n_eligible"]:
                o_st[k] = False
            o_st["l_eligible"] = False
            o_st["is_r_candidate"] = False
            o_st["r_eligible"] = False
            o_st["b_eligible"] = False
            o_st["e_eligible"] = False
            # Reset per-instance state
            o_st["signal"] = False
            o_st["signal_price"] = None
            o_st["strategy"] = None
            o_st["candle_count"] = 0
            o_st["first_candle_ok"] = False
            o_st["first_candle_body_pct"] = 0.0
            o_st["done"] = False
            o_st["entry_price"] = None
            o_st["shares"] = 0
            o_st["position_cost"] = 0.0
            o_st["pnl"] = 0.0
            o_st["o_range_high"] = 0.0
            o_st["o_range_low"] = 999999.0
            o_st["o_range_formed"] = False
            o_st["o_range_avg_vol"] = 0.0
            o_st["o_highest_since_entry"] = 0.0
            o_st["o_trailing_active"] = False
            o_st["o_partial_taken"] = False
            o_st["o_partial_proceeds"] = 0.0
            o_st["d_recent_closes"] = []
            o_st["p_recent_closes"] = []
            o_only_states.append(o_st)
            st["o_eligible"] = False  # prevent original from also detecting O
    states.extend(o_only_states)

    # Create dedicated B-only states (R2G specifically needs RED candle 1)
    b_only_states = []
    for st in states:
        if st.get("l_only") or st.get("o_only"):
            continue
        if st["b_eligible"]:
            b_st = dict(st)
            b_st["b_only"] = True
            for k in ["d_eligible","v_eligible","m_eligible","p_eligible","w_eligible",
                       "k_eligible","c_eligible","s_eligible","i_eligible","j_eligible","n_eligible"]:
                b_st[k] = False
            b_st["l_eligible"] = False
            b_st["o_eligible"] = False
            b_st["is_r_candidate"] = False
            b_st["r_eligible"] = False
            b_st["e_eligible"] = False
            b_st["signal"] = False
            b_st["signal_price"] = None
            b_st["strategy"] = None
            b_st["candle_count"] = 0
            b_st["first_candle_ok"] = False
            b_st["first_candle_body_pct"] = 0.0
            b_st["done"] = False
            b_st["entry_price"] = None
            b_st["shares"] = 0
            b_st["position_cost"] = 0.0
            b_st["pnl"] = 0.0
            b_st["b_candle1_red"] = False
            b_st["b_open_price"] = 0.0
            b_st["b_dip_seen"] = False
            b_st["b_avg_vol_sum"] = 0.0
            b_st["b_avg_vol_count"] = 0
            b_st["b_highest_since_entry"] = 0.0
            b_st["b_trailing_active"] = False
            b_st["b_partial_taken"] = False
            b_st["b_partial_proceeds"] = 0.0
            b_st["d_recent_closes"] = []
            b_st["p_recent_closes"] = []
            b_only_states.append(b_st)
            st["b_eligible"] = False
    states.extend(b_only_states)

    # Create dedicated E-only states (Gap-and-Go RelVol, fires early regardless of candle color)
    e_only_states = []
    for st in states:
        if st.get("l_only") or st.get("o_only") or st.get("b_only"):
            continue
        if st["e_eligible"]:
            e_st = dict(st)
            e_st["e_only"] = True
            for k in ["d_eligible","v_eligible","m_eligible","p_eligible","w_eligible",
                       "k_eligible","c_eligible","s_eligible","i_eligible","j_eligible","n_eligible"]:
                e_st[k] = False
            e_st["l_eligible"] = False
            e_st["o_eligible"] = False
            e_st["b_eligible"] = False
            e_st["is_r_candidate"] = False
            e_st["r_eligible"] = False
            e_st["signal"] = False
            e_st["signal_price"] = None
            e_st["strategy"] = None
            e_st["candle_count"] = 0
            e_st["first_candle_ok"] = False
            e_st["first_candle_body_pct"] = 0.0
            e_st["done"] = False
            e_st["entry_price"] = None
            e_st["shares"] = 0
            e_st["position_cost"] = 0.0
            e_st["pnl"] = 0.0
            e_st["e_pm_vol_ok"] = False
            e_st["e_highest_since_entry"] = 0.0
            e_st["e_trailing_active"] = False
            e_st["e_partial_taken"] = False
            e_st["e_partial_proceeds"] = 0.0
            e_st["d_recent_closes"] = []
            e_st["p_recent_closes"] = []
            e_only_states.append(e_st)
            st["e_eligible"] = False
    states.extend(e_only_states)

    for ts in all_timestamps:
        entry_candidates = []

        for st in states:
            if st["done"]:
                continue
            if ts not in st["mh"].index:
                continue

            candle = st["mh"].loc[ts]
            c_open = float(candle["Open"])
            c_high = float(candle["High"])
            c_low = float(candle["Low"])
            c_close = float(candle["Close"])

            # --- IN POSITION: check exits using strategy-specific params ---
            if st["entry_price"] is not None:
                try:
                    ts_et = ts.astimezone(ET_TZ)
                except Exception:
                    ts_et = ts
                minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)

                entry_et = st["entry_time"]
                try:
                    entry_et = entry_et.astimezone(ET_TZ)
                except Exception:
                    pass
                minutes_in_trade = (ts_et.hour * 60 + ts_et.minute) - (entry_et.hour * 60 + entry_et.minute)

                # --- Helper: close entire remaining position ---
                def _close_position(st, price, reason, ts_now):
                    sell_price = price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["shares"] * sell_price
                    partial_procs = (st.get("p_partial_proceeds", 0)
                                     + st.get("d_partial_proceeds", 0)
                                     + st.get("m_partial_proceeds", 0)
                                     + st.get("v_partial_proceeds", 0)
                                     + st.get("l_partial_proceeds", 0)
                                     + st.get("o_partial_proceeds", 0)
                                     + st.get("b_partial_proceeds", 0)
                                     + st.get("k_partial_proceeds", 0)
                                     + st.get("c_partial_proceeds", 0)
                                     + st.get("s_partial_proceeds", 0)
                                     + st.get("e_partial_proceeds", 0)
                                     + st.get("i_partial_proceeds", 0)
                                     + st.get("j_partial_proceeds", 0)
                                     + st.get("n_partial_proceeds", 0))
                    st["pnl"] = partial_procs + proceeds - st["position_cost"]
                    st["exit_price"] = price
                    st["exit_time"] = ts_now
                    st["exit_reason"] = reason
                    st["entry_price"] = None
                    st["shares"] = 0
                    st["done"] = True
                    _receive_proceeds(proceeds)

                # EOD forced exit (all strategies)
                if minutes_to_close <= EOD_EXIT_MINUTES:
                    _close_position(st, c_close, "EOD_CLOSE", ts)
                    continue

                # ===== STRATEGY P: Advanced exit (trailing + partial) =====
                if st["strategy"] == "P":
                    # Update highest since entry
                    if c_high > st["p_highest_since_entry"]:
                        st["p_highest_since_entry"] = c_high

                    # 1. Trailing stop (if active)
                    if st["p_trailing_active"]:
                        trail_stop = st["p_highest_since_entry"] * (1 - P_TRAIL_PCT / 100)
                        if c_low <= trail_stop:
                            _close_position(st, trail_stop, "TRAIL", ts)
                            continue
                    else:
                        # 2. Hard stop (before trail activates)
                        stop_price = st["entry_price"] * (1 - P_STOP_PCT / 100)
                        if c_low <= stop_price:
                            _close_position(st, stop_price, "STOP", ts)
                            continue

                    # 3. Activate trailing stop at +X% unrealized
                    if not st["p_trailing_active"]:
                        unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                        if unrealized_pct >= P_TRAIL_ACTIVATE_PCT:
                            st["p_trailing_active"] = True

                    # 4. Partial sell at target1
                    if not st["p_partial_taken"] and P_PARTIAL_SELL_PCT > 0:
                        tgt1 = st["entry_price"] * (1 + P_TARGET1_PCT / 100)
                        if c_high >= tgt1:
                            partial_shares = st["shares"] * (P_PARTIAL_SELL_PCT / 100)
                            sell_price = tgt1 * (1 - SLIPPAGE_PCT / 100)
                            partial_proceeds = partial_shares * sell_price
                            st["p_partial_proceeds"] += partial_proceeds
                            st["shares"] -= partial_shares
                            st["p_partial_taken"] = True
                            _receive_proceeds(partial_proceeds)
                            if st["shares"] <= 0.001:
                                # Sold everything (shouldn't happen with 25%)
                                st["pnl"] = st["p_partial_proceeds"] - st["position_cost"]
                                st["exit_price"] = tgt1
                                st["exit_time"] = ts
                                st["exit_reason"] = "TARGET"
                                st["entry_price"] = None
                                st["shares"] = 0
                                st["done"] = True
                                continue
                            # Fall through to check target2 on same candle

                    # 5. Runner target2 (full exit of remaining)
                    tgt2 = st["entry_price"] * (1 + P_TARGET2_PCT / 100)
                    if c_high >= tgt2:
                        _close_position(st, tgt2, "TARGET", ts)
                        continue

                    # 6. Time stop
                    if minutes_in_trade >= P_TIME_LIMIT_MINUTES:
                        _close_position(st, c_close, "TIME_STOP", ts)
                        continue

                    continue

                # ===== STRATEGY D: Dip Buy (trailing + partial) =====
                if st["strategy"] == "D":
                    # Update highest since entry
                    if c_high > st["d_highest_since_entry"]:
                        st["d_highest_since_entry"] = c_high

                    # 1. Trailing stop (if active)
                    if st["d_trailing_active"]:
                        trail_stop = st["d_highest_since_entry"] * (1 - D_TRAIL_PCT / 100)
                        if c_low <= trail_stop:
                            _close_position(st, trail_stop, "TRAIL", ts)
                            continue
                    else:
                        # 2. Hard stop (before trail activates)
                        stop_price = st["entry_price"] * (1 - D_STOP_PCT / 100)
                        if c_low <= stop_price:
                            _close_position(st, stop_price, "STOP", ts)
                            continue

                    # 3. Activate trailing stop at +X% unrealized
                    if not st["d_trailing_active"]:
                        unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                        if unrealized_pct >= D_TRAIL_ACTIVATE_PCT:
                            st["d_trailing_active"] = True

                    # 4. Partial sell at target1
                    if not st["d_partial_taken"] and D_PARTIAL_SELL_PCT > 0:
                        tgt1 = st["entry_price"] * (1 + D_TARGET1_PCT / 100)
                        if c_high >= tgt1:
                            partial_shares = st["shares"] * (D_PARTIAL_SELL_PCT / 100)
                            sell_price = tgt1 * (1 - SLIPPAGE_PCT / 100)
                            partial_proceeds = partial_shares * sell_price
                            st["d_partial_proceeds"] += partial_proceeds
                            st["shares"] -= partial_shares
                            st["d_partial_taken"] = True
                            _receive_proceeds(partial_proceeds)
                            if st["shares"] <= 0.001:
                                st["pnl"] = st["d_partial_proceeds"] - st["position_cost"]
                                st["exit_price"] = tgt1
                                st["exit_time"] = ts
                                st["exit_reason"] = "TARGET"
                                st["entry_price"] = None
                                st["shares"] = 0
                                st["done"] = True
                                continue
                            # Fall through to check target2 on same candle

                    # 5. Runner target2 (full exit of remaining)
                    tgt2 = st["entry_price"] * (1 + D_TARGET2_PCT / 100)
                    if c_high >= tgt2:
                        _close_position(st, tgt2, "TARGET", ts)
                        continue

                    # 6. Time stop
                    if minutes_in_trade >= D_TIME_LIMIT_MINUTES:
                        _close_position(st, c_close, "TIME_STOP", ts)
                        continue

                    continue

                # ===== STRATEGY M: Midday Break (trailing + partial) =====
                if st["strategy"] == "M":
                    # Update highest since entry
                    if c_high > st["m_highest_since_entry"]:
                        st["m_highest_since_entry"] = c_high

                    # 1. Trailing stop (if active)
                    if st["m_trailing_active"]:
                        trail_stop = st["m_highest_since_entry"] * (1 - M_TRAIL_PCT / 100)
                        if c_low <= trail_stop:
                            _close_position(st, trail_stop, "TRAIL", ts)
                            continue
                    else:
                        # 2. Hard stop (before trail activates)
                        stop_price = st["entry_price"] * (1 - M_STOP_PCT / 100)
                        if c_low <= stop_price:
                            _close_position(st, stop_price, "STOP", ts)
                            continue

                    # 3. Activate trailing stop at +X% unrealized
                    if not st["m_trailing_active"]:
                        unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                        if unrealized_pct >= M_TRAIL_ACTIVATE_PCT:
                            st["m_trailing_active"] = True

                    # 4. Partial sell at target1
                    if not st["m_partial_taken"] and M_PARTIAL_SELL_PCT > 0:
                        tgt1 = st["entry_price"] * (1 + M_TARGET1_PCT / 100)
                        if c_high >= tgt1:
                            partial_shares = st["shares"] * (M_PARTIAL_SELL_PCT / 100)
                            sell_price = tgt1 * (1 - SLIPPAGE_PCT / 100)
                            partial_proceeds = partial_shares * sell_price
                            st["m_partial_proceeds"] += partial_proceeds
                            st["shares"] -= partial_shares
                            st["m_partial_taken"] = True
                            _receive_proceeds(partial_proceeds)
                            if st["shares"] <= 0.001:
                                st["pnl"] = st["m_partial_proceeds"] - st["position_cost"]
                                st["exit_price"] = tgt1
                                st["exit_time"] = ts
                                st["exit_reason"] = "TARGET"
                                st["entry_price"] = None
                                st["shares"] = 0
                                st["done"] = True
                                continue

                    # 5. Time stop
                    if minutes_in_trade >= M_TIME_LIMIT_MINUTES:
                        _close_position(st, c_close, "TIME_STOP", ts)
                        continue

                    continue

                # ===== STRATEGY V: VWAP Reclaim (trailing + partial) =====
                if st["strategy"] == "V":
                    if c_high > st["v_highest_since_entry"]:
                        st["v_highest_since_entry"] = c_high

                    # 1. Trailing stop (if active)
                    if st["v_trailing_active"]:
                        trail_stop = st["v_highest_since_entry"] * (1 - V_TRAIL_PCT / 100)
                        if c_low <= trail_stop:
                            _close_position(st, trail_stop, "TRAIL", ts)
                            continue
                    else:
                        stop_price = st["entry_price"] * (1 - V_STOP_PCT / 100)
                        if c_low <= stop_price:
                            _close_position(st, stop_price, "STOP", ts)
                            continue

                    # 2. Activate trailing stop
                    if not st["v_trailing_active"]:
                        unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                        if unrealized_pct >= V_TRAIL_ACTIVATE_PCT:
                            st["v_trailing_active"] = True

                    # 3. Partial sell at target1
                    if not st["v_partial_taken"] and V_PARTIAL_SELL_PCT > 0:
                        tgt1 = st["entry_price"] * (1 + V_TARGET1_PCT / 100)
                        if c_high >= tgt1:
                            partial_shares = st["shares"] * (V_PARTIAL_SELL_PCT / 100)
                            sell_price = tgt1 * (1 - SLIPPAGE_PCT / 100)
                            partial_proceeds = partial_shares * sell_price
                            st["v_partial_proceeds"] += partial_proceeds
                            st["shares"] -= partial_shares
                            st["v_partial_taken"] = True
                            _receive_proceeds(partial_proceeds)
                            if st["shares"] <= 0.001:
                                st["pnl"] = st["v_partial_proceeds"] - st["position_cost"]
                                st["exit_price"] = tgt1
                                st["exit_time"] = ts
                                st["exit_reason"] = "TARGET"
                                st["entry_price"] = None
                                st["shares"] = 0
                                st["done"] = True
                                continue

                    # 4. Runner target2
                    tgt2 = st["entry_price"] * (1 + V_TARGET2_PCT / 100)
                    if c_high >= tgt2:
                        _close_position(st, tgt2, "TARGET", ts)
                        continue

                    # 5. Time stop
                    if minutes_in_trade >= V_TIME_LIMIT_MINUTES:
                        _close_position(st, c_close, "TIME_STOP", ts)
                        continue

                    continue

                # ===== STRATEGY L: Low Float Squeeze (trailing + partial + tiered targets) =====
                if st["strategy"] == "L":
                    if c_high > st["l_highest_since_entry"]:
                        st["l_highest_since_entry"] = c_high

                    # 1. Trailing stop (if active)
                    if st["l_trailing_active"]:
                        trail_stop = st["l_highest_since_entry"] * (1 - L_TRAIL_PCT / 100)
                        if c_low <= trail_stop:
                            _close_position(st, trail_stop, "TRAIL", ts)
                            continue
                    else:
                        # 2. Hard stop (before trail activates)
                        stop_price = st["entry_price"] * (1 - L_STOP_PCT / 100)
                        if c_low <= stop_price:
                            _close_position(st, stop_price, "STOP", ts)
                            continue

                    # 3. Activate trailing stop
                    if not st["l_trailing_active"]:
                        unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                        if unrealized_pct >= L_TRAIL_ACTIVATE_PCT:
                            st["l_trailing_active"] = True

                    # 4. Partial sell at target1 (float-tiered)
                    if not st["l_partial_taken"] and L_PARTIAL_SELL_PCT > 0:
                        tgt1 = st["entry_price"] * (1 + st["l_target1_pct"] / 100)
                        if c_high >= tgt1:
                            partial_shares = st["shares"] * (L_PARTIAL_SELL_PCT / 100)
                            sell_price = tgt1 * (1 - SLIPPAGE_PCT / 100)
                            partial_proceeds = partial_shares * sell_price
                            st["l_partial_proceeds"] += partial_proceeds
                            st["shares"] -= partial_shares
                            st["l_partial_taken"] = True
                            _receive_proceeds(partial_proceeds)
                            if st["shares"] <= 0.001:
                                st["pnl"] = st["l_partial_proceeds"] - st["position_cost"]
                                st["exit_price"] = tgt1
                                st["exit_time"] = ts
                                st["exit_reason"] = "TARGET"
                                st["entry_price"] = None
                                st["shares"] = 0
                                st["done"] = True
                                continue

                    # 5. Runner target2 (float-tiered)
                    tgt2 = st["entry_price"] * (1 + st["l_target2_pct"] / 100)
                    if c_high >= tgt2:
                        _close_position(st, tgt2, "TARGET", ts)
                        continue

                    # 6. Time stop
                    if minutes_in_trade >= L_TIME_LIMIT_MINUTES:
                        _close_position(st, c_close, "TIME_STOP", ts)
                        continue

                    continue

                # ===== STRATEGY R: Multi-Day Runner (trailing + target) =====
                if st["strategy"] == "R":
                    if c_high > st["r_highest_since_entry"]:
                        st["r_highest_since_entry"] = c_high

                    # 1. Trailing stop (if active)
                    if st["r_trailing_active"]:
                        trail_stop = st["r_highest_since_entry"] * (1 - R_TRAIL_PCT / 100)
                        if c_low <= trail_stop:
                            _close_position(st, trail_stop, "TRAIL", ts)
                            continue
                    else:
                        stop_price = st["entry_price"] * (1 - R_STOP_PCT / 100)
                        if c_low <= stop_price:
                            _close_position(st, stop_price, "STOP", ts)
                            continue

                    # 2. Activate trailing stop
                    if not st["r_trailing_active"]:
                        unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                        if unrealized_pct >= R_TRAIL_ACTIVATE_PCT:
                            st["r_trailing_active"] = True

                    # 3. Target
                    tgt = st["entry_price"] * (1 + R_TARGET1_PCT / 100)
                    if c_high >= tgt:
                        _close_position(st, tgt, "TARGET", ts)
                        continue

                    # 4. Time stop
                    if minutes_in_trade >= R_TIME_LIMIT_MINUTES:
                        _close_position(st, c_close, "TIME_STOP", ts)
                        continue

                    continue

                # ===== STRATEGY W: Power Hour Breakout (trailing + target) =====
                if st["strategy"] == "W":
                    if c_high > st["w_highest_since_entry"]:
                        st["w_highest_since_entry"] = c_high

                    # 1. Trailing stop (if active)
                    if st["w_trailing_active"]:
                        trail_stop = st["w_highest_since_entry"] * (1 - W_TRAIL_PCT / 100)
                        if c_low <= trail_stop:
                            _close_position(st, trail_stop, "TRAIL", ts)
                            continue
                    else:
                        # 2. Hard stop (before trail activates)
                        stop_price = st["entry_price"] * (1 - W_STOP_PCT / 100)
                        if c_low <= stop_price:
                            _close_position(st, stop_price, "STOP", ts)
                            continue

                    # 3. Activate trailing stop at +X% unrealized
                    if not st["w_trailing_active"]:
                        unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                        if unrealized_pct >= W_TRAIL_ACTIVATE_PCT:
                            st["w_trailing_active"] = True

                    # 4. Target (full exit)
                    tgt = st["entry_price"] * (1 + W_TARGET_PCT / 100)
                    if c_high >= tgt:
                        _close_position(st, tgt, "TARGET", ts)
                        continue

                    continue

                # ===== NEW STRATEGIES (O,B,K,C,S,E,I,J,N): generic exit logic =====
                if st["strategy"] in ("O","B","K","C","S","E","I","J","N"):
                    _strat = st["strategy"].lower()
                    # Strategy-specific params lookup
                    _exit_cfg = {
                        "o": (O_TARGET1_PCT, O_TARGET2_PCT, O_STOP_PCT, O_TRAIL_PCT, O_TRAIL_ACTIVATE_PCT, O_PARTIAL_SELL_PCT, O_TIME_LIMIT_MINUTES),
                        "b": (B_TARGET1_PCT, B_TARGET2_PCT, B_STOP_PCT, B_TRAIL_PCT, B_TRAIL_ACTIVATE_PCT, B_PARTIAL_SELL_PCT, B_TIME_LIMIT_MINUTES),
                        "k": (K_TARGET1_PCT, K_TARGET2_PCT, K_STOP_PCT, K_TRAIL_PCT, K_TRAIL_ACTIVATE_PCT, K_PARTIAL_SELL_PCT, K_TIME_LIMIT_MINUTES),
                        "c": (C_TARGET1_PCT, C_TARGET2_PCT, C_STOP_PCT, C_TRAIL_PCT, C_TRAIL_ACTIVATE_PCT, C_PARTIAL_SELL_PCT, C_TIME_LIMIT_MINUTES),
                        "s": (S_TARGET1_PCT, S_TARGET2_PCT, S_STOP_PCT, S_TRAIL_PCT, S_TRAIL_ACTIVATE_PCT, S_PARTIAL_SELL_PCT, S_TIME_LIMIT_MINUTES),
                        "e": (E_TARGET1_PCT, E_TARGET2_PCT, E_STOP_PCT, E_TRAIL_PCT, E_TRAIL_ACTIVATE_PCT, E_PARTIAL_SELL_PCT, E_TIME_LIMIT_MINUTES),
                        "i": (I_TARGET1_PCT, I_TARGET2_PCT, I_STOP_PCT, I_TRAIL_PCT, I_TRAIL_ACTIVATE_PCT, I_PARTIAL_SELL_PCT, I_TIME_LIMIT_MINUTES),
                        "j": (J_TARGET1_PCT, J_TARGET2_PCT, J_STOP_PCT, J_TRAIL_PCT, J_TRAIL_ACTIVATE_PCT, J_PARTIAL_SELL_PCT, J_TIME_LIMIT_MINUTES),
                        "n": (N_TARGET1_PCT, N_TARGET2_PCT, N_STOP_PCT, N_TRAIL_PCT, N_TRAIL_ACTIVATE_PCT, N_PARTIAL_SELL_PCT, N_TIME_LIMIT_MINUTES),
                    }
                    tgt1_pct, tgt2_pct, stop_pct, trail_pct, trail_act, partial_pct, time_lim = _exit_cfg[_strat]
                    hkey = f"{_strat}_highest_since_entry"
                    tkey = f"{_strat}_trailing_active"
                    pkey = f"{_strat}_partial_taken"
                    ppkey = f"{_strat}_partial_proceeds"

                    # 1. Update highest since entry
                    if c_high > st[hkey]:
                        st[hkey] = c_high

                    # 2. Trailing stop (if active)
                    if st[tkey]:
                        trail_stop = st[hkey] * (1 - trail_pct / 100)
                        if c_low <= trail_stop:
                            _close_position(st, trail_stop, "TRAIL", ts)
                            continue
                    else:
                        # 3. Hard stop — O uses dynamic stop (range low) if stop_pct==0
                        if _strat == "o" and stop_pct == 0:
                            dyn_stop = st.get("o_dynamic_stop", 0)
                            if dyn_stop > 0 and c_low <= dyn_stop:
                                _close_position(st, dyn_stop, "STOP", ts)
                                continue
                        elif stop_pct > 0:
                            stop_price = st["entry_price"] * (1 - stop_pct / 100)
                            if c_low <= stop_price:
                                _close_position(st, stop_price, "STOP", ts)
                                continue

                    # 4. Activate trailing stop
                    if not st[tkey]:
                        unrealized = (c_high / st["entry_price"] - 1) * 100
                        if unrealized >= trail_act:
                            st[tkey] = True

                    # 5. Partial sell at target1
                    if not st[pkey] and partial_pct > 0:
                        tgt1_price = st["entry_price"] * (1 + tgt1_pct / 100)
                        if c_high >= tgt1_price:
                            sell_shares = int(st["shares"] * partial_pct / 100)
                            if sell_shares > 0:
                                sell_price = tgt1_price * (1 - SLIPPAGE_PCT / 100)
                                proceeds = sell_shares * sell_price
                                st["shares"] -= sell_shares
                                st[ppkey] += proceeds
                                _receive_proceeds(proceeds)
                            st[pkey] = True

                    # 6. Full target at target2
                    tgt2_price = st["entry_price"] * (1 + tgt2_pct / 100)
                    if c_high >= tgt2_price:
                        _close_position(st, tgt2_price, "TARGET", ts)
                        continue

                    # 7. Time stop
                    if minutes_in_trade >= time_lim:
                        _close_position(st, c_close, "TIME_STOP", ts)
                        continue

                    continue

                # ===== STRATEGIES H/G/A/F: target + stop + trail + time stop =====
                strat_map = {
                    "H": (H_TARGET_PCT, H_TIME_LIMIT_MINUTES, H_STOP_PCT, H_TRAIL_PCT, H_TRAIL_ACTIVATE_PCT),
                    "G": (G_TARGET_PCT, G_TIME_LIMIT_MINUTES, G_STOP_PCT, G_TRAIL_PCT, G_TRAIL_ACTIVATE_PCT),
                    "A": (A_TARGET_PCT, A_TIME_LIMIT_MINUTES, A_STOP_PCT, A_TRAIL_PCT, A_TRAIL_ACTIVATE_PCT),
                    "F": (F_TARGET_PCT, F_TIME_LIMIT_MINUTES, F_STOP_PCT, F_TRAIL_PCT, F_TRAIL_ACTIVATE_PCT),
                }
                target_pct, time_limit, stop_pct, trail_pct, trail_act = strat_map.get(
                    st["strategy"], (A_TARGET_PCT, A_TIME_LIMIT_MINUTES, 0, 0, 0))

                # Target hit
                target_price = st["entry_price"] * (1 + target_pct / 100)
                if c_high >= target_price:
                    _close_position(st, target_price, "TARGET", ts)
                    continue

                # Hard stop
                if stop_pct > 0:
                    stop_price = st["entry_price"] * (1 - stop_pct / 100)
                    if c_low <= stop_price:
                        _close_position(st, stop_price, "STOP", ts)
                        continue

                # Trailing stop
                if trail_pct > 0:
                    unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                    if unrealized_pct >= trail_act:
                        trail_stop = c_high * (1 - trail_pct / 100)
                        if trail_stop > st.get("hgaf_trail_stop", 0):
                            st["hgaf_trail_stop"] = trail_stop
                    if st.get("hgaf_trail_stop", 0) > 0 and c_low <= st["hgaf_trail_stop"]:
                        _close_position(st, st["hgaf_trail_stop"], "TRAIL", ts)
                        continue

                # Time stop
                if minutes_in_trade >= time_limit:
                    _close_position(st, c_close, "TIME_STOP", ts)
                    continue

                continue

            # --- NOT IN POSITION: candle-by-candle signal detection ---
            st["candle_count"] += 1

            # --- R CANDIDATE: separate entry logic (Day 2 of massive gapper) ---
            if st["is_r_candidate"]:
                if st["candle_count"] > R_MAX_ENTRY_CANDLE or not st["r_eligible"]:
                    st["done"] = True
                    continue

                # Track open price from first candle
                if st["candle_count"] == 1:
                    st["open_price"] = c_open

                # Phase 1: Pullback from D2 open
                if not st["r_pullback_seen"] and st["candle_count"] <= R_PULLBACK_WINDOW:
                    pullback_level = st["open_price"] * (1 - R_D2_PULLBACK_PCT / 100)
                    if c_low <= pullback_level:
                        st["r_pullback_seen"] = True

                # Phase 2: Bounce above reference
                if st["r_pullback_seen"] and not st["signal"]:
                    ref_price = st["r_day1_close"] if R_BOUNCE_REF == "d1_close" else st["open_price"]
                    if c_close > ref_price:
                        st["strategy"] = "R"
                        st["signal"] = True
                        st["signal_price"] = c_close
                        entry_candidates.append(st)
                continue

            # CANDLE 1: Check first candle is green
            if st["candle_count"] == 1:
                if c_open > 0:
                    body_pct = (c_close / c_open - 1) * 100
                    st["first_candle_body_pct"] = body_pct
                    st["first_candle_high"] = c_high
                    st["first_candle_volume"] = float(candle["Volume"])
                    st["signal_price"] = c_close
                    st["open_price"] = c_open
                    if body_pct > 0:
                        st["first_candle_ok"] = True
                if not st["first_candle_ok"] and not st.get("l_only") and not st.get("o_only") and not st.get("b_only") and not st.get("e_only"):
                    st["done"] = True

                # B-only: on candle 1, check if it's RED (that's what R2G needs)
                if st.get("b_only") and st["candle_count"] == 1:
                    st["b_open_price"] = c_open
                    if c_close < c_open:  # RED candle 1
                        st["b_candle1_red"] = True
                    else:
                        st["done"] = True  # Not red → R2G doesn't apply

                # E-only: on candle 1, check premarket volume vs first candle volume
                if st.get("e_only") and st["candle_count"] == 1:
                    st["open_price"] = c_open
                    c_vol = float(candle["Volume"])
                    if c_vol > 0 and st["pm_volume"] > 0:
                        # Use first candle volume as proxy for "typical"
                        if st["pm_volume"] >= E_MIN_PM_VOL_MULT * c_vol:
                            st["e_pm_vol_ok"] = True

            # CANDLE 2: Classify for H, G, A, F (skip for L/O/B/E-only states)
            elif st["candle_count"] == 2 and st["first_candle_ok"] and not st.get("l_only") and not st.get("o_only") and not st.get("b_only") and not st.get("e_only"):
                second_green = c_close > c_open
                second_new_high = c_high > st["first_candle_high"]
                vol_confirm = float(candle["Volume"]) > st["first_candle_volume"]

                strategy = _classify_candle2(
                    st["gap_pct"], st["first_candle_body_pct"],
                    second_green, second_new_high, vol_confirm,
                )
                if strategy:
                    st["strategy"] = strategy
                    st["signal"] = True
                    entry_candidates.append(st)
                else:
                    # H/G/A/F rejected — enable D, V, M, P, W fallbacks if gap qualifies
                    eligible = False
                    if st["gap_pct"] >= D_MIN_GAP_PCT:
                        st["d_eligible"] = True
                        # Initialize D spike tracking retroactively
                        st["d_spike_high"] = max(st["first_candle_high"], c_high)
                        eligible = True
                    if st["gap_pct"] >= V_MIN_GAP_PCT:
                        st["v_eligible"] = True
                        eligible = True
                    if st["gap_pct"] >= M_MIN_GAP_PCT:
                        st["m_eligible"] = True
                        eligible = True
                    if st["gap_pct"] >= P_MIN_GAP_PCT:
                        st["p_eligible"] = True
                        eligible = True
                    if st["gap_pct"] >= W_MIN_GAP_PCT:
                        st["w_eligible"] = True
                        eligible = True
                    if st["gap_pct"] >= K_MIN_GAP_PCT:
                        st["k_eligible"] = True
                        eligible = True
                    if st["gap_pct"] >= C_MIN_GAP_PCT:
                        st["c_eligible"] = True
                        eligible = True
                    if st["gap_pct"] >= S_MIN_GAP_PCT:
                        st["s_eligible"] = True
                        eligible = True
                    if st["gap_pct"] >= I_MIN_GAP_PCT:
                        st["i_eligible"] = True
                        eligible = True
                    if st["gap_pct"] >= J_MIN_GAP_PCT:
                        st["j_eligible"] = True
                        eligible = True
                    if st["gap_pct"] >= N_MIN_GAP_PCT:
                        st["n_eligible"] = True
                        eligible = True
                    if st["l_eligible"]:  # Already set at init based on float + gap
                        eligible = True
                    if not eligible:
                        st["done"] = True

            # Retry fill on later candles (H/G/A/F)
            elif st["signal"] and st["entry_price"] is None and not st["done"]:
                if c_close > 0:
                    st["signal_price"] = c_close
                    entry_candidates.append(st)

            # --- All fallback strategies track simultaneously, first to fire wins ---
            elif (st["d_eligible"] or st["v_eligible"] or st["m_eligible"] or st["p_eligible"] or st["w_eligible"]
                  or st["l_eligible"] or st["k_eligible"] or st["c_eligible"] or st["s_eligible"]
                  or st["i_eligible"] or st["j_eligible"] or st["n_eligible"]
                  or st.get("o_only") or st.get("b_only") or st.get("e_only")) and st["entry_price"] is None and not st["done"]:
                # Check if ALL strategies have timed out
                d_timed_out = not st["d_eligible"] or st["candle_count"] > D_MAX_ENTRY_CANDLE
                v_timed_out = not st["v_eligible"] or st["candle_count"] > V_MAX_ENTRY_CANDLE
                m_timed_out = not st["m_eligible"] or st["candle_count"] > M_MAX_ENTRY_CANDLE
                p_timed_out = not st["p_eligible"] or st["candle_count"] > P_MAX_ENTRY_CANDLE
                w_timed_out = not st["w_eligible"] or st["candle_count"] > W_LATEST_CANDLE
                l_timed_out = not st["l_eligible"] or st["candle_count"] > L_LATEST_CANDLE
                k_timed_out = not st["k_eligible"] or st["candle_count"] > K_MAX_ENTRY_CANDLE
                c_timed_out = not st["c_eligible"] or st["candle_count"] > C_MAX_ENTRY_CANDLE
                s_timed_out = not st["s_eligible"] or st["candle_count"] > S_MAX_ENTRY_CANDLE
                o_timed_out = not st["o_eligible"] or st["candle_count"] > O_MAX_ENTRY_CANDLE
                b_timed_out = not st["b_eligible"] or st["candle_count"] > B_MAX_ENTRY_CANDLE
                e_timed_out = not st["e_eligible"] or st["candle_count"] > E_MAX_ENTRY_CANDLE
                i_timed_out = not st["i_eligible"] or st["candle_count"] > I_MAX_ENTRY_CANDLE
                j_timed_out = not st["j_eligible"] or st["candle_count"] > J_MAX_ENTRY_CANDLE
                n_timed_out = not st["n_eligible"] or st["candle_count"] > N_MAX_ENTRY_CANDLE
                if (d_timed_out and v_timed_out and m_timed_out and p_timed_out and w_timed_out
                    and l_timed_out and k_timed_out and c_timed_out and s_timed_out
                    and o_timed_out and b_timed_out and e_timed_out
                    and i_timed_out and j_timed_out and n_timed_out):
                    st["done"] = True
                    continue

                # --- D: Opening Dip Buy detection ---
                if st["d_eligible"] and st["candle_count"] <= D_MAX_ENTRY_CANDLE and not st["signal"]:
                    # Update spike high
                    if c_high > st["d_spike_high"]:
                        st["d_spike_high"] = c_high

                    # Check spike magnitude once we have enough candles
                    if not st["d_spike_detected"] and st["candle_count"] >= D_SPIKE_WINDOW:
                        open_p = st["open_price"]
                        if open_p and open_p > 0:
                            spike_pct = (st["d_spike_high"] / open_p - 1) * 100
                            if spike_pct >= D_MIN_SPIKE_PCT:
                                st["d_spike_detected"] = True
                            else:
                                st["d_eligible"] = False  # Spike too small

                    # After spike detected: wait for dip
                    if st["d_spike_detected"] and not st["d_dip_detected"]:
                        dip_level = st["d_spike_high"] * (1 - D_DIP_PCT / 100)
                        if c_low <= dip_level:
                            st["d_dip_detected"] = True

                    # After dip: look for entry signal
                    if st["d_dip_detected"]:
                        # Get candle index for VWAP lookup
                        candle_idx = st["candle_count"] - 1
                        vwap_val = st["vwap"][candle_idx] if candle_idx < len(st["vwap"]) else 0

                        # Track below-VWAP state
                        if c_close < vwap_val:
                            st["d_below_vwap"] = True

                        st["d_recent_closes"].append(c_close)
                        if len(st["d_recent_closes"]) > 5:
                            st["d_recent_closes"] = st["d_recent_closes"][-5:]

                        if D_ENTRY_MODE == "vwap":
                            # VWAP reclaim: was below, now close above
                            if st["d_below_vwap"] and c_close > vwap_val:
                                st["strategy"] = "D"
                                st["signal"] = True
                                st["signal_price"] = c_close
                                entry_candidates.append(st)
                        else:
                            # 5-candle high: close > max of last 5 closes
                            if len(st["d_recent_closes"]) >= 5:
                                prev_max = max(st["d_recent_closes"][:-1])
                                if c_close > prev_max:
                                    st["strategy"] = "D"
                                    st["signal"] = True
                                    st["signal_price"] = c_close
                                    entry_candidates.append(st)

                # --- M: Midday Range Break detection ---
                if st["m_eligible"] and st["candle_count"] <= M_MAX_ENTRY_CANDLE and not st["signal"]:
                    candle_idx = st["candle_count"] - 1  # 0-based index into mh

                    # Phase 1: Morning spike check (once)
                    if not st["m_morning_spike_ok"]:
                        if st["candle_count"] >= M_MORNING_CANDLES:
                            morning_highs = st["mh"].iloc[:M_MORNING_CANDLES]["High"].values.astype(float)
                            morning_high = float(np.max(morning_highs))
                            open_p = st["open_price"]
                            if open_p and open_p > 0:
                                spike_pct = (morning_high / open_p - 1) * 100
                                if spike_pct >= M_MORNING_SPIKE_PCT:
                                    st["m_morning_spike_ok"] = True
                                else:
                                    st["m_eligible"] = False

                    # Phase 2: Consolidation zone check (once)
                    elif not st["m_consol_checked"]:
                        consol_end_candle = M_RANGE_START_CANDLE + M_CONSOLIDATION_LEN
                        if st["candle_count"] >= consol_end_candle:
                            if len(st["mh"]) >= consol_end_candle:
                                consol_slice = st["mh"].iloc[M_RANGE_START_CANDLE:consol_end_candle]
                                c_highs = consol_slice["High"].values.astype(float)
                                c_lows = consol_slice["Low"].values.astype(float)
                                c_vols = consol_slice["Volume"].values.astype(float)

                                consol_high = float(np.max(c_highs))
                                consol_low = float(np.min(c_lows))

                                # Range check
                                range_pct = (consol_high - consol_low) / consol_high * 100 if consol_high > 0 else 999
                                if range_pct > M_MAX_RANGE_PCT:
                                    st["m_eligible"] = False
                                else:
                                    # Volume contraction check
                                    morning_avg_vol = float(np.mean(
                                        st["mh"].iloc[:min(30, len(st["mh"]))]["Volume"].values.astype(float)))
                                    consol_avg_vol = float(np.mean(c_vols))
                                    if morning_avg_vol > 0 and consol_avg_vol / morning_avg_vol > M_VOL_RATIO:
                                        st["m_eligible"] = False
                                    else:
                                        st["m_consol_high"] = consol_high
                                        st["m_consol_checked"] = True
                                        st["m_breakout_ready"] = True
                            else:
                                st["m_eligible"] = False

                    # Phase 3: Breakout above consolidation high
                    elif st["m_breakout_ready"]:
                        if c_close > st["m_consol_high"]:
                            st["strategy"] = "M"
                            st["signal"] = True
                            st["signal_price"] = c_close
                            entry_candidates.append(st)

                # --- V: VWAP Reclaim detection ---
                if st["v_eligible"] and st["candle_count"] <= V_MAX_ENTRY_CANDLE and not st["signal"]:
                    candle_idx = st["candle_count"] - 1
                    vwap_val = st["vwap"][candle_idx] if candle_idx < len(st["vwap"]) else 0

                    if vwap_val > 0:
                        # Track consecutive closes below VWAP
                        if c_close < vwap_val:
                            st["v_below_count"] += 1
                            # Track max depth below VWAP
                            below_pct = (vwap_val - c_close) / vwap_val * 100
                            if below_pct > st["v_max_below_pct"]:
                                st["v_max_below_pct"] = below_pct
                        else:
                            # Close above VWAP: check if reclaim conditions met
                            if (st["v_below_count"] >= V_MIN_BELOW_CANDLES
                                    and st["v_max_below_pct"] >= V_MIN_BELOW_PCT):
                                # Volume spike check
                                if candle_idx >= 10:
                                    recent_vols = st["mh"].iloc[candle_idx-10:candle_idx]["Volume"].values.astype(float)
                                    avg_vol = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 0
                                else:
                                    avg_vol = 0
                                cur_vol = float(candle["Volume"])
                                if avg_vol <= 0 or cur_vol >= avg_vol * V_VOL_SPIKE_RATIO:
                                    st["strategy"] = "V"
                                    st["signal"] = True
                                    st["signal_price"] = c_close
                                    entry_candidates.append(st)
                                else:
                                    st["v_below_count"] = 0  # Reset, keep tracking
                            else:
                                st["v_below_count"] = 0  # Reset, not enough below candles

                # --- P: PM high breakout + pullback + bounce ---
                if st["p_eligible"] and st["candle_count"] <= P_MAX_ENTRY_CANDLE and not st["signal"]:
                    pm_high = st["premarket_high"]

                    # Phase 1: Breakout confirmation
                    if not st["p_breakout_confirmed"]:
                        st["p_recent_closes"].append(c_close > pm_high)
                        if len(st["p_recent_closes"]) > P_CONFIRM_WINDOW:
                            st["p_recent_closes"] = st["p_recent_closes"][-P_CONFIRM_WINDOW:]
                        if sum(st["p_recent_closes"]) >= P_CONFIRM_ABOVE:
                            st["p_breakout_confirmed"] = True

                    # Phase 2: Pullback detection
                    elif not st["p_pullback_detected"]:
                        st["p_candles_since_confirm"] += 1
                        pullback_zone = pm_high * (1 + P_PULLBACK_PCT / 100)
                        if c_low <= pullback_zone:
                            st["p_pullback_detected"] = True
                            if c_close > pm_high:
                                st["strategy"] = "P"
                                st["signal"] = True
                                st["signal_price"] = c_close
                                entry_candidates.append(st)
                        elif st["p_candles_since_confirm"] >= P_PULLBACK_TIMEOUT:
                            if c_close > pm_high:
                                st["strategy"] = "P"
                                st["signal"] = True
                                st["signal_price"] = c_close
                                entry_candidates.append(st)
                            else:
                                st["p_eligible"] = False

                    # Phase 3: Bounce
                    else:
                        if c_close > pm_high:
                            st["strategy"] = "P"
                            st["signal"] = True
                            st["signal_price"] = c_close
                            entry_candidates.append(st)

                # --- W: Power Hour Breakout detection ---
                if st["w_eligible"] and st["candle_count"] <= W_LATEST_CANDLE and not st["signal"]:
                    candle_idx = st["candle_count"] - 1

                    # Phase 1: Morning run check (at candle 30)
                    if not st["w_morning_run_ok"]:
                        if st["candle_count"] >= 30:
                            morning_highs = st["mh"].iloc[:30]["High"].values.astype(float)
                            morning_high = float(np.max(morning_highs))
                            morning_vols = st["mh"].iloc[:30]["Volume"].values.astype(float)
                            open_p = st["open_price"]
                            if open_p and open_p > 0:
                                run_pct = (morning_high / open_p - 1) * 100
                                if run_pct >= W_MIN_MORNING_RUN:
                                    st["w_morning_run_ok"] = True
                                    st["w_morning_high"] = morning_high
                                    st["w_morning_spike_vol"] = float(np.max(morning_vols))
                                else:
                                    st["w_eligible"] = False

                    # Phase 2: Consolidation check (at earliest candle)
                    elif not st["w_consol_checked"]:
                        if st["candle_count"] >= W_EARLIEST_CANDLE:
                            if len(st["mh"]) >= W_EARLIEST_CANDLE:
                                consol_slice = st["mh"].iloc[W_CONSOL_START:W_EARLIEST_CANDLE]
                                c_highs_w = consol_slice["High"].values.astype(float)
                                c_lows_w = consol_slice["Low"].values.astype(float)
                                c_closes_w = consol_slice["Close"].values.astype(float)
                                c_vols_w = consol_slice["Volume"].values.astype(float)

                                w_consol_high = float(np.max(c_highs_w))
                                w_consol_low = float(np.min(c_lows_w))

                                # Range check
                                w_range_pct = (w_consol_high - w_consol_low) / w_consol_high * 100 if w_consol_high > 0 else 999
                                if w_range_pct > W_MAX_RANGE_PCT:
                                    st["w_eligible"] = False
                                else:
                                    # VWAP deviation check
                                    c_vwap_w = st["vwap"][W_CONSOL_START:W_EARLIEST_CANDLE]
                                    min_len = min(len(c_closes_w), len(c_vwap_w))
                                    if min_len > 0 and np.all(c_vwap_w[:min_len] > 0):
                                        vwap_devs = np.abs(c_closes_w[:min_len] - c_vwap_w[:min_len]) / c_vwap_w[:min_len] * 100
                                        max_dev = float(np.max(vwap_devs))
                                        if max_dev > W_MAX_VWAP_DEV_PCT:
                                            st["w_eligible"] = False
                                        else:
                                            # HOD breaks check
                                            hod_breaks = 0
                                            running_hod = float(st["mh"].iloc[0]["High"])
                                            for k in range(1, min(W_EARLIEST_CANDLE, len(st["mh"]))):
                                                kh = float(st["mh"].iloc[k]["High"])
                                                if kh > running_hod:
                                                    hod_breaks += 1
                                                    running_hod = kh
                                            if hod_breaks > W_MAX_HOD_BREAKS:
                                                st["w_eligible"] = False
                                            else:
                                                st["w_consol_checked"] = True
                                                st["w_consol_high"] = w_consol_high
                                                st["w_consol_avg_vol"] = float(np.mean(c_vols_w))
                                                st["w_breakout_ready"] = True
                                    else:
                                        st["w_eligible"] = False
                            else:
                                st["w_eligible"] = False

                    # Phase 3: Breakout scan
                    elif st["w_breakout_ready"]:
                        if c_close > st["w_consol_high"]:
                            cur_vol = float(candle["Volume"])
                            # Volume surge vs consolidation avg
                            vol_ok = (st["w_consol_avg_vol"] <= 0 or
                                      cur_vol >= st["w_consol_avg_vol"] * W_VOL_SURGE_MULT)
                            # Volume vs morning spike
                            morning_vol_ok = (st["w_morning_spike_vol"] <= 0 or
                                             cur_vol >= st["w_morning_spike_vol"] * W_VOL_VS_MORNING_MULT)
                            # VWAP filter
                            vwap_ok = True
                            if W_REQUIRE_ABOVE_VWAP and candle_idx < len(st["vwap"]):
                                vwap_ok = c_close >= st["vwap"][candle_idx]

                            if vol_ok and morning_vol_ok and vwap_ok:
                                st["strategy"] = "W"
                                st["signal"] = True
                                st["signal_price"] = c_close
                                entry_candidates.append(st)

                # --- O: Opening Range Breakout detection (O-only states) ---
                if st["o_eligible"] and st["candle_count"] <= O_MAX_ENTRY_CANDLE and not st["signal"]:
                    c_vol = float(candle["Volume"])
                    if not st["o_range_formed"]:
                        # Building opening range
                        if c_high > st["o_range_high"]:
                            st["o_range_high"] = c_high
                        if c_low < st["o_range_low"]:
                            st["o_range_low"] = c_low
                        st["o_range_avg_vol"] += c_vol
                        if st["candle_count"] >= O_RANGE_CANDLES:
                            st["o_range_formed"] = True
                            st["o_range_avg_vol"] /= O_RANGE_CANDLES
                            st["o_dynamic_stop"] = st["o_range_low"]
                    else:
                        # Range formed — look for breakout above range high
                        if c_close > st["o_range_high"]:
                            vol_ok = (st["o_range_avg_vol"] <= 0
                                      or c_vol >= O_BREAKOUT_VOL_MULT * st["o_range_avg_vol"])
                            if vol_ok:
                                st["strategy"] = "O"
                                st["signal"] = True
                                st["signal_price"] = c_close
                                entry_candidates.append(st)

                # --- B: Red-to-Green (R2G) detection (B-only states) ---
                if st["b_eligible"] and st.get("b_candle1_red") and st["candle_count"] <= B_MAX_ENTRY_CANDLE and not st["signal"]:
                    c_vol = float(candle["Volume"])
                    open_p = st["b_open_price"]
                    if open_p > 0:
                        # Track average volume
                        st["b_avg_vol_sum"] += c_vol
                        st["b_avg_vol_count"] += 1
                        avg_vol = st["b_avg_vol_sum"] / st["b_avg_vol_count"]

                        # Check if price dipped below open
                        if c_low < open_p:
                            st["b_dip_seen"] = True
                            # Check max dip not exceeded
                            dip_pct = (open_p - c_low) / open_p * 100
                            if dip_pct > B_MAX_DIP_PCT:
                                st["b_eligible"] = False

                        # After dip: look for reclaim back above open with volume
                        if st["b_dip_seen"] and c_close > open_p:
                            vol_ok = avg_vol <= 0 or c_vol >= B_MIN_RECLAIM_VOL_MULT * avg_vol
                            if vol_ok:
                                st["strategy"] = "B"
                                st["signal"] = True
                                st["signal_price"] = c_close
                                entry_candidates.append(st)

                # --- K: First Pullback Buy detection ---
                if st["k_eligible"] and st["candle_count"] <= K_MAX_ENTRY_CANDLE and not st["signal"]:
                    c_vol = float(candle["Volume"])
                    open_p = st["open_price"] or c_open
                    if not st["k_run_detected"]:
                        # Phase 1: Detect morning run-up
                        if c_high > st["k_run_high"]:
                            st["k_run_high"] = c_high
                        st["k_run_vol_sum"] += c_vol
                        if st["candle_count"] >= K_RUN_WINDOW and open_p > 0:
                            run_pct = (st["k_run_high"] / open_p - 1) * 100
                            if run_pct >= K_MIN_RUN_PCT:
                                st["k_run_detected"] = True
                            else:
                                st["k_eligible"] = False
                    elif not st["k_pullback_detected"]:
                        # Phase 2: Detect orderly pullback
                        if c_low < st["k_pullback_low"]:
                            st["k_pullback_low"] = c_low
                        st["k_pullback_vol_sum"] += c_vol
                        st["k_pullback_candles"] += 1
                        pullback_pct = (st["k_run_high"] - st["k_pullback_low"]) / st["k_run_high"] * 100
                        if pullback_pct >= K_PULLBACK_PCT:
                            # Check if pullback was orderly (low volume)
                            avg_run_vol = st["k_run_vol_sum"] / max(K_RUN_WINDOW, 1)
                            avg_pb_vol = st["k_pullback_vol_sum"] / max(st["k_pullback_candles"], 1)
                            if avg_run_vol <= 0 or avg_pb_vol <= K_PULLBACK_VOL_RATIO * avg_run_vol:
                                st["k_pullback_detected"] = True
                    else:
                        # Phase 3: Bounce with volume
                        avg_pb_vol = st["k_pullback_vol_sum"] / max(st["k_pullback_candles"], 1)
                        vol_ok = avg_pb_vol <= 0 or c_vol >= K_BOUNCE_VOL_MULT * avg_pb_vol
                        if c_close > st["k_pullback_low"] * (1 + K_PULLBACK_PCT / 100) and vol_ok:
                            st["strategy"] = "K"
                            st["signal"] = True
                            st["signal_price"] = c_close
                            entry_candidates.append(st)

                # --- C: Micro Flag / Base Pattern detection ---
                if st["c_eligible"] and st["candle_count"] <= C_MAX_ENTRY_CANDLE and not st["signal"]:
                    c_vol = float(candle["Volume"])
                    open_p = st["open_price"] or c_open
                    if not st["c_spike_detected"]:
                        # Phase 1: Initial spike
                        if c_high > st["c_spike_high"]:
                            st["c_spike_high"] = c_high
                        if open_p > 0 and st["c_spike_high"] > 0:
                            spike_pct = (st["c_spike_high"] / open_p - 1) * 100
                            if spike_pct >= C_MIN_SPIKE_PCT:
                                st["c_spike_detected"] = True
                                st["c_base_start_candle"] = st["candle_count"]
                    elif st["c_base_candle_count"] < C_MAX_BASE_CANDLES:
                        # Phase 2: Track consolidation (tight base)
                        st["c_base_candle_count"] += 1
                        if c_high > st["c_base_high"]:
                            st["c_base_high"] = c_high
                        if c_low < st["c_base_low"]:
                            st["c_base_low"] = c_low
                        # Check if range is still tight
                        if st["c_base_low"] > 0:
                            base_range = (st["c_base_high"] - st["c_base_low"]) / st["c_base_low"] * 100
                            if base_range > C_MAX_BASE_RANGE_PCT:
                                # Range broken — reset base tracking
                                st["c_spike_detected"] = False
                                st["c_spike_high"] = c_high
                                st["c_base_candle_count"] = 0
                                st["c_base_high"] = 0.0
                                st["c_base_low"] = 999999.0
                    else:
                        # Phase 3: Breakout above base high with volume
                        if c_close > st["c_base_high"]:
                            # Need min candles in base
                            if st["c_base_candle_count"] >= C_MIN_BASE_CANDLES:
                                vol_ok = c_vol >= C_BREAKOUT_VOL_MULT * (st.get("first_candle_volume", c_vol) or c_vol)
                                if vol_ok:
                                    st["strategy"] = "C"
                                    st["signal"] = True
                                    st["signal_price"] = c_close
                                    entry_candidates.append(st)
                            else:
                                st["c_eligible"] = False

                # --- S: Stuff-and-Break detection ---
                if st["s_eligible"] and st["candle_count"] <= S_MAX_ENTRY_CANDLE and not st["signal"]:
                    c_vol = float(candle["Volume"])
                    # Track HOD
                    if c_high > st["s_hod"]:
                        old_hod = st["s_hod"]
                        st["s_hod"] = c_high
                        st["s_hod_candle"] = st["candle_count"]
                        # If we already have rejections and price breaks to new HOD with volume
                        if st["s_hod_tests"] >= S_MIN_HOD_TESTS and old_hod > 0:
                            break_pct = (c_high - old_hod) / old_hod * 100
                            if break_pct > 0:
                                vol_ok = c_vol >= S_BREAKOUT_VOL_MULT * (st.get("first_candle_volume", c_vol) or c_vol)
                                if vol_ok:
                                    st["strategy"] = "S"
                                    st["signal"] = True
                                    st["signal_price"] = c_close
                                    entry_candidates.append(st)
                    elif st["s_hod"] > 0:
                        # Check if testing HOD (within tolerance)
                        hod_dist_pct = (st["s_hod"] - c_high) / st["s_hod"] * 100
                        if hod_dist_pct <= S_HOD_TOLERANCE_PCT and not st["s_rejected"]:
                            st["s_rejected"] = True  # Mark as testing
                            st["s_last_test_candle"] = st["candle_count"]
                        # Check for rejection (pulled back after test)
                        if st["s_rejected"]:
                            pullback_pct = (st["s_hod"] - c_low) / st["s_hod"] * 100
                            if pullback_pct >= S_REJECTION_PCT:
                                st["s_hod_tests"] += 1
                                st["s_rejected"] = False  # Reset for next test

                # --- E: Gap-and-Go RelVol detection (E-only states) ---
                if st["e_eligible"] and st.get("e_pm_vol_ok") and st["candle_count"] <= E_MAX_ENTRY_CANDLE and not st["signal"]:
                    # Enter on first green candle with elevated volume
                    if c_close > c_open:
                        st["strategy"] = "E"
                        st["signal"] = True
                        st["signal_price"] = c_close
                        entry_candidates.append(st)

                # --- I: P1 Immediate PM High Breakout detection ---
                if st["i_eligible"] and st["candle_count"] <= I_MAX_ENTRY_CANDLE and not st["signal"]:
                    c_vol = float(candle["Volume"])
                    pm_high = st["premarket_high"]
                    if pm_high > 0 and c_close > pm_high:
                        vol_ok = c_vol >= I_BREAKOUT_VOL_MULT * (st.get("first_candle_volume", c_vol) or c_vol)
                        if vol_ok:
                            st["strategy"] = "I"
                            st["signal"] = True
                            st["signal_price"] = c_close
                            entry_candidates.append(st)

                # --- J: P3 VWAP + PM High Breakout detection ---
                if st["j_eligible"] and st["candle_count"] <= J_MAX_ENTRY_CANDLE and not st["signal"]:
                    pm_high = st["premarket_high"]
                    candle_idx = st["candle_count"] - 1
                    vwap_val = st["vwap"][candle_idx] if candle_idx < len(st["vwap"]) else 0
                    if pm_high > 0 and vwap_val > 0:
                        # Price must be near VWAP AND breaking above PM high
                        vwap_dist = abs(c_close - vwap_val) / vwap_val * 100
                        if vwap_dist <= J_VWAP_PROXIMITY_PCT and c_close > pm_high:
                            st["strategy"] = "J"
                            st["signal"] = True
                            st["signal_price"] = c_close
                            entry_candidates.append(st)

                # --- N: P4 HOD Reclaim detection ---
                if st["n_eligible"] and st["candle_count"] <= N_MAX_ENTRY_CANDLE and not st["signal"]:
                    # Track HOD
                    if c_high > st["n_hod"]:
                        if st["n_hod"] > 0 and st["n_pullback_seen"]:
                            # HOD reclaim after pullback!
                            age = st["candle_count"] - st["n_hod_candle"]
                            if age >= N_MIN_HOD_AGE:
                                st["strategy"] = "N"
                                st["signal"] = True
                                st["signal_price"] = c_close
                                entry_candidates.append(st)
                        st["n_hod"] = c_high
                        st["n_hod_candle"] = st["candle_count"]
                        st["n_pullback_seen"] = False
                    elif st["n_hod"] > 0:
                        # Check for pullback from HOD
                        pullback_pct = (st["n_hod"] - c_low) / st["n_hod"] * 100
                        if pullback_pct >= N_PULLBACK_FROM_HOD_PCT:
                            st["n_pullback_seen"] = True

                # --- L: Low Float Squeeze detection ---
                # Always track HOD for L-eligible stocks
                if st["l_eligible"] and not st["signal"] and c_high > st["l_running_hod"]:
                    st["l_running_hod"] = c_high
                if st["l_eligible"] and st["candle_count"] >= L_EARLIEST_CANDLE and st["candle_count"] <= L_LATEST_CANDLE and not st["signal"]:
                    candle_idx = st["candle_count"] - 1

                    # Check if this candle made a new HOD (already tracked above)
                    is_new_hod = (c_high >= st["l_running_hod"]) and (st["l_running_hod"] > 0)

                    # 1. HOD break check
                    if not L_HOD_BREAK_REQUIRED or is_new_hod:
                        # 2. Volume surge check
                        c_vol = float(candle["Volume"])
                        if candle_idx >= 10:
                            recent_vols = st["mh"].iloc[candle_idx-10:candle_idx]["Volume"].values.astype(float)
                            avg_vol = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 0
                        else:
                            recent_vols = st["mh"].iloc[:candle_idx]["Volume"].values.astype(float)
                            avg_vol = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 0

                        vol_ok_l = avg_vol <= 0 or c_vol >= avg_vol * L_VOL_SURGE_MULT

                        # 3. Price acceleration (green candle body)
                        body_pct_l = (c_close / c_open - 1) * 100 if c_open > 0 else 0
                        accel_ok = body_pct_l >= L_MIN_PRICE_ACCEL_PCT

                        # 4. VWAP filter
                        vwap_ok_l = True
                        if L_REQUIRE_ABOVE_VWAP and candle_idx < len(st["vwap"]):
                            vwap_ok_l = c_close >= st["vwap"][candle_idx]

                        if vol_ok_l and accel_ok and vwap_ok_l:
                            st["strategy"] = "L"
                            st["signal"] = True
                            st["signal_price"] = c_close
                            entry_candidates.append(st)
                    else:
                        # Still track HOD even when not breaking
                        pass

        # --- PASS 2: Capital allocation (single pool) ---
        entry_candidates.sort(key=lambda s: (STRAT_PRIORITY.get(s["strategy"], 99), -s["first_candle_body_pct"]))

        filled_this_ts = []
        skipped_this_ts = []

        for st in entry_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue

            try:
                ts_et = ts.astimezone(ET_TZ)
            except Exception:
                ts_et = ts
            mins_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
            if mins_to_close <= EOD_EXIT_MINUTES:
                continue

            # Check cash
            if cash_box[0] < 100:
                skipped_this_ts.append(st["strategy"])
                continue
            trade_size = cash_box[0]  # FULL_BALANCE_SIZING

            fill_price = st["signal_price"]
            if fill_price is None or fill_price <= 0:
                continue

            # Volume cap: total exposure on this ticker <= VOL_CAP_PCT of dollar vol.
            # Uses shares * current_price (fill_price) for accurate market-value exposure.
            if VOL_CAP_PCT > 0:
                pre_entry = st["mh"].loc[st["mh"].index <= ts]
                vol_shares = pre_entry["Volume"].sum() if len(pre_entry) > 0 else 0
                dollar_vol = fill_price * vol_shares
                vol_limit = dollar_vol * (VOL_CAP_PCT / 100)
                # Subtract CURRENT market-value exposure on this ticker across ALL active states
                existing_exposure = 0.0
                for s in states:
                    if s["entry_price"] is not None and s["ticker"] == st["ticker"]:
                        existing_exposure += s["shares"] * fill_price
                remaining_room = vol_limit - existing_exposure
                if remaining_room < 50:
                    continue
                if trade_size > remaining_room:
                    trade_size = remaining_room
                    st["vol_capped"] = True
                if trade_size < 50:
                    continue

            entry_price = fill_price * (1 + SLIPPAGE_PCT / 100)
            st["entry_price"] = entry_price
            st["entry_time"] = ts
            st["position_cost"] = trade_size
            st["shares"] = trade_size / entry_price
            cash_box[0] -= trade_size
            filled_this_ts.append(st["strategy"])

            # Initialize strategy-specific exit tracking
            if st["strategy"] == "D":
                st["d_highest_since_entry"] = entry_price
            elif st["strategy"] == "V":
                st["v_highest_since_entry"] = entry_price
            elif st["strategy"] == "R":
                st["r_highest_since_entry"] = entry_price
            elif st["strategy"] == "M":
                st["m_highest_since_entry"] = entry_price
            elif st["strategy"] == "P":
                st["p_highest_since_entry"] = entry_price
            elif st["strategy"] == "W":
                st["w_highest_since_entry"] = entry_price
            elif st["strategy"] == "L":
                st["l_highest_since_entry"] = entry_price

        # Log selection decisions when strategies were skipped
        if filled_this_ts and skipped_this_ts:
            selection_log.append({
                "filled": list(filled_this_ts),
                "skipped": list(skipped_this_ts),
            })

    # EOD: close remaining
    for st in states:
        if st["entry_price"] is not None and st["shares"] > 0:
            last_ts = st["mh"].index[-1]
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["shares"] * sell_price
            partial_procs = (st.get("p_partial_proceeds", 0)
                             + st.get("d_partial_proceeds", 0)
                             + st.get("m_partial_proceeds", 0)
                             + st.get("v_partial_proceeds", 0)
                             + st.get("l_partial_proceeds", 0)
                             + st.get("o_partial_proceeds", 0)
                             + st.get("b_partial_proceeds", 0)
                             + st.get("k_partial_proceeds", 0)
                             + st.get("c_partial_proceeds", 0)
                             + st.get("s_partial_proceeds", 0)
                             + st.get("e_partial_proceeds", 0)
                             + st.get("i_partial_proceeds", 0)
                             + st.get("j_partial_proceeds", 0)
                             + st.get("n_partial_proceeds", 0))
            st["pnl"] = partial_procs + proceeds - st["position_cost"]
            st["exit_price"] = last_close
            st["exit_time"] = last_ts
            st["exit_reason"] = "EOD_CLOSE"
            st["entry_price"] = None
            st["shares"] = 0
            st["done"] = True
            _receive_proceeds(proceeds)

    return states, cash_box[0], unsettled_box[0], selection_log


# --- MAIN ----

if __name__ == "__main__":
    no_charts = "--no-charts" in sys.argv
    explain_mode = "--explain" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--no-charts", "--explain")]
    data_dirs = args if args else ["stored_data_combined"]

    STRAT_KEYS = ["H","G","A","F","D","V","P","M","R","W","O","B","K","C","S","E","I","J","N","L"]

    print(f"Combined Green Candle Strategy (H+G+A+F+D+V+M+R+P+W) Backtest")
    print(f"{'='*70}")
    print(f"  Strategy H: Gap>={H_MIN_GAP_PCT}% + body>={H_MIN_BODY_PCT}% + 2nd green + new hi + vol confirm")
    print(f"              Target: +{H_TARGET_PCT}% | Time Stop: {H_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy G: Gap>={G_MIN_GAP_PCT}% + 2nd green + new hi")
    print(f"              Target: +{G_TARGET_PCT}% | Time Stop: {G_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy A: Gap>{A_MIN_GAP_PCT}% + body>={A_MIN_BODY_PCT}% + 2nd green + new hi")
    print(f"              Target: +{A_TARGET_PCT}% | Time Stop: {A_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy F: Gap>{F_MIN_GAP_PCT}% + 2nd green (catch-all)")
    print(f"              Target: +{F_TARGET_PCT}% | Time Stop: {F_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy D: Gap>{D_MIN_GAP_PCT}% + spike>={D_MIN_SPIKE_PCT}% + dip {D_DIP_PCT}% + {D_ENTRY_MODE}")
    print(f"              Partial: sell {D_PARTIAL_SELL_PCT:.0f}% at +{D_TARGET1_PCT}% | Runner: +{D_TARGET2_PCT}%")
    print(f"              Trail: {D_TRAIL_PCT}% (activates +{D_TRAIL_ACTIVATE_PCT}%) | Stop: -{D_STOP_PCT}% | {D_TIME_LIMIT_MINUTES}m")
    print(f"  Strategy V: Gap>{V_MIN_GAP_PCT}% + {V_MIN_BELOW_CANDLES} closes below VWAP + reclaim w/ volume")
    print(f"              Partial: sell {V_PARTIAL_SELL_PCT:.0f}% at +{V_TARGET1_PCT}% | Runner: +{V_TARGET2_PCT}%")
    print(f"              Trail: {V_TRAIL_PCT}% (activates +{V_TRAIL_ACTIVATE_PCT}%) | Stop: -{V_STOP_PCT}% | {V_TIME_LIMIT_MINUTES}m")
    print(f"  Strategy M: Gap>{M_MIN_GAP_PCT}% + morning spike {M_MORNING_SPIKE_PCT}% + consolidation + breakout")
    print(f"              Partial: sell {M_PARTIAL_SELL_PCT:.0f}% at +{M_TARGET1_PCT}% | Trail: {M_TRAIL_PCT}%")
    print(f"              Trail activates +{M_TRAIL_ACTIVATE_PCT}% | Stop: -{M_STOP_PCT}% | {M_TIME_LIMIT_MINUTES}m")
    print(f"  Strategy R: Day1 gap>={R_DAY1_MIN_GAP:.0f}% + Day2 pullback {R_D2_PULLBACK_PCT:.0f}% + bounce > {R_BOUNCE_REF}")
    print(f"              Target: +{R_TARGET1_PCT:.0f}% | Trail: {R_TRAIL_PCT}% (at +{R_TRAIL_ACTIVATE_PCT}%) | Stop: -{R_STOP_PCT}% | {R_TIME_LIMIT_MINUTES}m")
    print(f"  Strategy P: Gap>{P_MIN_GAP_PCT}% + PM high breakout + pullback + bounce (fallback)")
    print(f"              Partial: sell {P_PARTIAL_SELL_PCT:.0f}% at +{P_TARGET1_PCT}% | Runner: +{P_TARGET2_PCT}%")
    print(f"              Trail: {P_TRAIL_PCT}% (activates +{P_TRAIL_ACTIVATE_PCT}%) | Stop: -{P_STOP_PCT}% | {P_TIME_LIMIT_MINUTES}m")
    print(f"  Strategy W: Gap>{W_MIN_GAP_PCT}% + morning run {W_MIN_MORNING_RUN}% + power hour breakout")
    print(f"              Target: +{W_TARGET_PCT}% | Trail: {W_TRAIL_PCT}% (at +{W_TRAIL_ACTIVATE_PCT}%) | Stop: -{W_STOP_PCT}%")
    print(f"  Strategy L: Low float (<{L_MAX_FLOAT/1e6:.0f}M) + gap>{L_MIN_GAP_PCT}% + HOD break + vol surge")
    print(f"              Tiered targets: T1(<{L_TIER1_FLOAT/1e6:.0f}M) +{L_TIER1_TARGET1_PCT}/{L_TIER1_TARGET2_PCT}% | T2(<{L_TIER2_FLOAT/1e6:.0f}M) +{L_TIER2_TARGET1_PCT}/{L_TIER2_TARGET2_PCT}% | T3 +{L_TIER3_TARGET1_PCT}/{L_TIER3_TARGET2_PCT}%")
    print(f"              Trail: {L_TRAIL_PCT}% (at +{L_TRAIL_ACTIVATE_PCT}%) | Stop: -{L_STOP_PCT}% | {L_TIME_LIMIT_MINUTES}m")
    priority_str = " > ".join(k for k, _ in sorted(STRAT_PRIORITY.items(), key=lambda x: x[1]))
    print(f"  Priority:   {priority_str}")
    print(f"  Single pool: ${STARTING_CASH:,} | Full balance sizing")
    print(f"  No SPY regime filter")
    print(f"  Data: {data_dirs}")
    print(f"{'='*70}\n")

    print("Loading data...")
    all_dates, daily_picks = load_all_picks(data_dirs)
    all_dates = [d for d in all_dates if "2024-01-01" <= d <= "2026-02-28"]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")

    # Pre-scan for R (Multi-Day Runner) candidates
    r_day2_picks = {}
    for idx in range(len(all_dates) - 1):
        d1, d2 = all_dates[idx], all_dates[idx + 1]
        for pick in daily_picks.get(d1, []):
            if pick["gap_pct"] < R_DAY1_MIN_GAP:
                continue
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) < 10:
                continue
            day1_close = float(mh["Close"].values[-1])
            d2_data = _load_r_intraday(pick["ticker"], d2, data_dirs)
            if d2_data is not None and len(d2_data) >= 20:
                r_pick = {
                    "ticker": pick["ticker"],
                    "gap_pct": pick["gap_pct"],
                    "premarket_high": 0,
                    "pm_volume": 0,
                    "market_hour_candles": d2_data,
                    "is_r_candidate": True,
                    "r_day1_close": day1_close,
                }
                r_day2_picks.setdefault(d2, []).append(r_pick)
    r_total = sum(len(v) for v in r_day2_picks.values())
    print(f"  R candidates: {r_total} across {len(r_day2_picks)} days\n")

    cash = float(STARTING_CASH)
    unsettled = 0.0
    all_results = []
    all_selection_logs = []

    print(f"\n  Starting Cash: ${cash:,.0f}")
    print()

    print(f"{'Date':<12} {'Strat':>5} {'Trades':>6} {'Win':>4} {'Loss':>5} "
          f"{'Day P&L':>12} {'Balance':>14}")
    print("-" * 72)

    for d in all_dates:
        # Settle overnight
        cash += unsettled
        unsettled = 0.0

        picks = daily_picks.get(d, [])
        # Inject R candidates (Day 2 of massive gappers)
        r_picks = r_day2_picks.get(d, [])
        all_picks = picks + r_picks
        cash_account = cash < MARGIN_THRESHOLD

        states, cash, unsettled, day_selection = simulate_day_combined(
            all_picks, cash, cash_account
        )
        all_selection_logs.extend(day_selection)

        day_pnl = 0.0
        day_trades = 0
        day_wins = 0
        day_losses = 0
        counts = {k: [0, 0] for k in STRAT_KEYS}

        for st in states:
            if st["exit_reason"] is not None:
                day_trades += 1
                day_pnl += st["pnl"]
                if st["pnl"] > 0:
                    day_wins += 1
                else:
                    day_losses += 1
                s = st["strategy"]
                if s in counts:
                    counts[s][0] += 1
                    if st["pnl"] > 0:
                        counts[s][1] += 1

        equity = cash + unsettled

        # Build strat label like "H1G1A2F1"
        parts = []
        for key in STRAT_KEYS:
            if counts[key][0] > 0:
                parts.append(f"{key}{counts[key][0]}")
        strat_label = "".join(parts) if parts else ""

        print(f"{d:<12} {strat_label:>5} {day_trades:>6} {day_wins:>4} {day_losses:>5} "
              f"${day_pnl:>+11,.0f} ${equity:>13,.0f}")

        # Per-trade detail with % profit
        traded_states = [s for s in states if s["exit_reason"] is not None]
        for st in traded_states:
            pct = (st["pnl"] / st["position_cost"] * 100) if st["position_cost"] > 0 else 0
            reason_short = {"TARGET": "T", "TIME_STOP": "TS", "EOD_CLOSE": "EOD",
                            "STOP": "SL", "TRAIL": "TR"}.get(
                st["exit_reason"], st["exit_reason"][:3])
            vc_tag = " VC" if st.get("vol_capped") else ""
            print(f"  -> [{st['strategy']}] {st['ticker']:<6} {reason_short:<3}  "
                  f"${st['pnl']:>+10,.0f}  ({pct:>+6.2f}%){vc_tag}")

        # --explain: show per-ticker decisions
        if explain_mode:
            for st in states:
                tk = st["ticker"]
                gap = st["gap_pct"]
                traded = st["exit_reason"] is not None
                if traded:
                    continue  # already shown above
                # Build reason why not traded
                reasons = []
                if st.get("is_r_candidate"):
                    reasons.append(f"R-candidate (day2)")
                # Check candle 1
                cc = st.get("candle_count", 0)
                if cc == 0:
                    reasons.append("no candles")
                elif st.get("first_candle_body_pct", 0) == 0 and not st.get("o_eligible") and not st.get("b_eligible") and not st.get("e_eligible") and not st.get("l_eligible"):
                    reasons.append("candle1 red/doji")
                # Check what was eligible
                elig = []
                for code in STRAT_KEYS:
                    key = f"{code.lower()}_eligible"
                    if st.get(key):
                        elig.append(code)
                if elig:
                    reasons.append(f"eligible: {','.join(elig)}")
                else:
                    reasons.append("no strategy eligible")
                # Check if signal fired but no cash
                if st.get("signal") and st.get("entry_price") is None:
                    reasons.append("SIGNAL but no cash")
                elif st.get("done") and not st.get("signal"):
                    reasons.append("timed out / no signal")
                elif not st.get("done") and not st.get("signal"):
                    reasons.append("no signal")
                print(f"     [{tk:<6} gap={gap:>5.1f}%] NOT TRADED: {' | '.join(reasons)}")

        all_results.append({
            "date": d, "picks": picks, "states": states,
            "day_pnl": day_pnl, "equity": equity, "regime_skip": False,
            "trades": day_trades, "wins": day_wins, "losses": day_losses,
            **{f"{k.lower()}_trades": counts[k][0] for k in STRAT_KEYS},
            **{f"{k.lower()}_wins": counts[k][1] for k in STRAT_KEYS},
        })

    # --- Summary ---
    final_equity = cash + unsettled
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    strat_totals = {k: sum(r[f"{k.lower()}_trades"] for r in all_results) for k in STRAT_KEYS}
    strat_wins = {k: sum(r[f"{k.lower()}_wins"] for r in all_results) for k in STRAT_KEYS}
    daily_pnls = [r["day_pnl"] for r in all_results if r["trades"] > 0]
    green = sum(1 for p in daily_pnls if p > 0) if daily_pnls else 0
    red = sum(1 for p in daily_pnls if p <= 0) if daily_pnls else 0
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if daily_pnls and np.std(daily_pnls) > 0 else 0

    all_exits = {}
    all_trade_pnls = []
    strat_pnls = {k: [] for k in STRAT_KEYS}
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None:
                reason_key = f"{st['strategy']}_{st['exit_reason']}"
                all_exits[reason_key] = all_exits.get(reason_key, 0) + 1
                all_trade_pnls.append(st["pnl"])
                s = st["strategy"]
                if s in strat_pnls:
                    strat_pnls[s].append(st["pnl"])

    avg_win = np.mean([p for p in all_trade_pnls if p > 0]) if total_wins > 0 else 0
    avg_loss = np.mean([p for p in all_trade_pnls if p <= 0]) if total_losses > 0 else 0

    # Per-trade % returns
    all_trade_pcts = []
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None and st["position_cost"] > 0:
                all_trade_pcts.append(st["pnl"] / st["position_cost"] * 100)

    # Daily equity % changes
    daily_eq_pcts = []
    for i in range(1, len(all_results)):
        prev_eq = all_results[i-1]["equity"]
        curr_eq = all_results[i]["equity"]
        if prev_eq > 0:
            daily_eq_pcts.append((curr_eq / prev_eq - 1) * 100)

    print(f"\n{'='*70}")
    print(f"  COMBINED STRATEGY SUMMARY (H+G+A+F+D+V+P)")
    print(f"{'='*70}")
    print(f"  Starting Cash:    ${STARTING_CASH:,}")
    print(f"  Ending Equity:    ${final_equity:,.0f}  ({(final_equity/STARTING_CASH - 1)*100:+.1f}%)")
    if unsettled > 0:
        print(f"    (Cash: ${cash:,.0f} + Unsettled: ${unsettled:,.0f})")
    print(f"  Trading Days:     {len(all_dates)}")
    print(f"  Total Trades:     {total_trades}")
    print(f"    Winners:        {total_wins} ({total_wins/max(total_trades,1)*100:.1f}%)")
    print(f"    Losers:         {total_losses}")
    print(f"  Avg Win:          ${avg_win:+,.0f}")
    print(f"  Avg Loss:         ${avg_loss:+,.0f}")
    pf = abs(avg_win * total_wins / (avg_loss * total_losses)) if total_losses > 0 and avg_loss != 0 else 0
    print(f"  Profit Factor:    {pf:.2f}" if pf > 0 else "")

    # Daily % metrics
    if daily_eq_pcts:
        print(f"\n  Daily % Change:")
        print(f"    Avg Daily:      {np.mean(daily_eq_pcts):+.2f}%")
        print(f"    Median Daily:   {np.median(daily_eq_pcts):+.2f}%")
        print(f"    Best Day:       {max(daily_eq_pcts):+.2f}%")
        print(f"    Worst Day:      {min(daily_eq_pcts):+.2f}%")
        print(f"    Std Dev:        {np.std(daily_eq_pcts):.2f}%")

    # Per-trade % metrics
    if all_trade_pcts:
        win_pcts = [p for p in all_trade_pcts if p > 0]
        loss_pcts = [p for p in all_trade_pcts if p <= 0]
        print(f"\n  Per-Trade % Return:")
        print(f"    Avg Trade:      {np.mean(all_trade_pcts):+.2f}%")
        print(f"    Median Trade:   {np.median(all_trade_pcts):+.2f}%")
        if win_pcts:
            print(f"    Avg Winner:     {np.mean(win_pcts):+.2f}%")
        if loss_pcts:
            print(f"    Avg Loser:      {np.mean(loss_pcts):+.2f}%")

    # Per-strategy breakdown
    strat_info = [
        ("H (High Conviction)", "H"), ("G (Big Gap Runner)", "G"),
        ("A (Quick Scalp)", "A"), ("F (Catch-All)", "F"),
        ("D (Dip Buy)", "D"), ("V (VWAP Reclaim)", "V"),
        ("P (PM High Breakout)", "P"), ("M (Midday Breakout)", "M"),
        ("R (Multi-Day Runner)", "R"), ("W (Power Hour)", "W"),
        ("O (Opening Range Breakout)", "O"), ("B (Red-to-Green)", "B"),
        ("K (First Pullback)", "K"), ("C (Micro Flag)", "C"),
        ("S (Stuff-and-Break)", "S"), ("E (Gap-and-Go)", "E"),
        ("I (PM High Immediate)", "I"), ("J (VWAP+PMH Breakout)", "J"),
        ("N (HOD Reclaim)", "N"), ("L (Low Float Squeeze)", "L"),
    ]
    for label, key in strat_info:
        total_s = strat_totals[key]
        wins_s = strat_wins[key]
        pnls_s = strat_pnls[key]
        print(f"\n  {'---'*17}")
        print(f"  STRATEGY {label}")
        print(f"  {'---'*17}")
        print(f"    Trades:  {total_s}")
        print(f"    Winners: {wins_s} ({wins_s/max(total_s,1)*100:.1f}%)")
        if pnls_s:
            w = [p for p in pnls_s if p > 0]
            l = [p for p in pnls_s if p <= 0]
            print(f"    Total PnL: ${sum(pnls_s):+,.0f}")
            if w:
                print(f"    Avg Win:   ${np.mean(w):+,.0f}")
            if l:
                print(f"    Avg Loss:  ${np.mean(l):+,.0f}")

    print(f"\n  Exit Reasons:")
    for reason, count in sorted(all_exits.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20} {count:>4} ({count/max(total_trades,1)*100:.1f}%)")

    if daily_pnls:
        print(f"\n  Green Days:       {green}/{len(daily_pnls)} ({green/len(daily_pnls)*100:.1f}%)")
        print(f"  Red Days:         {red}/{len(daily_pnls)}")
        print(f"  Best Day:         ${max(daily_pnls):+,.0f}")
        print(f"  Worst Day:        ${min(daily_pnls):+,.0f}")
        print(f"  Avg P&L/Day:      ${np.mean(daily_pnls):+,.0f}")
        print(f"  Sharpe (ann.):    {sharpe:.2f}")

    # --- STRATEGY SELECTION ANALYSIS ---
    if all_selection_logs:
        print(f"\n  {'='*50}")
        print(f"  STRATEGY SELECTION ANALYSIS")
        print(f"  {'='*50}")
        skip_counts = {k: 0 for k in STRAT_KEYS}
        steal_counts = {k: {k2: 0 for k2 in STRAT_KEYS} for k in STRAT_KEYS}
        for entry in all_selection_logs:
            for sk in entry["skipped"]:
                if sk in skip_counts:
                    skip_counts[sk] += 1
                    for fl in entry["filled"]:
                        if fl in steal_counts[sk]:
                            steal_counts[sk][fl] += 1
        print(f"  Times each strategy was SKIPPED (no cash):")
        for k in STRAT_KEYS:
            if skip_counts[k] > 0:
                stealers = [(k2, c) for k2, c in steal_counts[k].items() if c > 0]
                stealers.sort(key=lambda x: -x[1])
                stealer_str = ", ".join(f"{s}({c}x)" for s, c in stealers[:5])
                print(f"    {k}: {skip_counts[k]:>4}x  (cash taken by: {stealer_str})")
        total_skips = sum(skip_counts.values())
        print(f"  Total skip events: {total_skips}")

    # --- STRESS TEST ---
    print(f"\n  {'='*50}")
    print(f"  STRESS TEST")
    print(f"  {'='*50}")

    # Max drawdown
    equities = [r["equity"] for r in all_results]
    peak = equities[0]
    max_dd_dollar = 0.0
    max_dd_pct = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = peak - eq
        dd_pct = dd / peak if peak > 0 else 0
        if dd > max_dd_dollar:
            max_dd_dollar = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    print(f"  Max Drawdown:     ${max_dd_dollar:,.0f} ({max_dd_pct*100:.1f}%)")

    # Consecutive losing days
    streak = 0
    max_streak = 0
    for pnl in daily_pnls:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"  Max Losing Streak: {max_streak} days")

    # Remove top 10% of winning days
    if daily_pnls:
        sorted_pnls = sorted(daily_pnls, reverse=True)
        n_remove = max(1, int(len(daily_pnls) * 0.10))
        remaining = sorted_pnls[n_remove:]
        rem_total = sum(remaining)
        rem_std = np.std(remaining) if len(remaining) > 1 else 1.0
        rem_sharpe = (np.mean(remaining) / rem_std) * np.sqrt(252) if rem_std > 0 else 0
        rem_pass = rem_total > 0
        print(f"  Remove Top 10%:   ${rem_total:+,.0f} (Sharpe {rem_sharpe:.2f}) "
              f"{'PASS' if rem_pass else 'FAIL'}")

    # Kelly criterion
    wins_pnls = [p for p in daily_pnls if p > 0]
    loss_pnls_abs = [abs(p) for p in daily_pnls if p < 0]
    if wins_pnls and loss_pnls_abs:
        avg_w = np.mean(wins_pnls)
        avg_l = np.mean(loss_pnls_abs)
        p_w = len(wins_pnls) / len(daily_pnls)
        b = avg_w / avg_l if avg_l > 0 else float('inf')
        kelly = (b * p_w - (1 - p_w)) / b if b > 0 else 0
        print(f"  Kelly Criterion:  {kelly*100:.1f}% {'PASS' if kelly > 0 else 'FAIL'}")
    else:
        kelly = 0
        print(f"  Kelly Criterion:  N/A")

    # Monte Carlo: remove random 10% of days, check if still profitable
    np.random.seed(42)
    n_sims = 1000
    n_remove_mc = max(1, int(len(daily_pnls) * 0.10))
    mc_profitable = 0
    all_daily = np.array(daily_pnls)
    for _ in range(n_sims):
        indices = np.random.choice(len(all_daily), size=len(all_daily) - n_remove_mc, replace=False)
        if all_daily[indices].sum() > 0:
            mc_profitable += 1
    mc_pct = mc_profitable / n_sims * 100
    mc_verdict = "PASS" if mc_pct > 80 else ("WEAK" if mc_pct > 50 else "FAIL")
    print(f"  Monte Carlo:      {mc_pct:.1f}% profitable ({mc_verdict})")

    # Volume-capped trade count
    vc_count = sum(1 for r in all_results for s in r["states"]
                   if s["exit_reason"] is not None and s.get("vol_capped"))
    if vc_count > 0:
        print(f"  Vol-Capped Trades: {vc_count}")

    stress_pass = (rem_total > 0 and kelly > 0 and mc_pct > 80) if daily_pnls else False
    print(f"\n  OVERALL: {'ALL PASS' if stress_pass else 'SOME CONCERNS'}")
    print(f"{'='*70}")

    # --- CHARTS ---
    if no_charts:
        print("\n  Skipping charts (--no-charts)")
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("charts", f"gc_combined_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nGenerating charts -> {run_dir}/")

        def _to_et(ts_val):
            try:
                return ts_val.astimezone(ET_TZ)
            except Exception:
                return ts_val

        COLORS_H = ["#9C27B0", "#AB47BC", "#BA68C8"]   # Purple for H
        COLORS_G = ["#4CAF50", "#66BB6A", "#81C784"]   # Greens for G
        COLORS_A = ["#2196F3", "#42A5F5", "#64B5F6"]   # Blues for A
        COLORS_F = ["#FF9800", "#FFA726", "#FFB74D"]    # Oranges for F
        COLORS_D = ["#00BCD4", "#26C6DA", "#4DD0E1"]    # Cyan for D
        COLORS_V = ["#607D8B", "#78909C", "#90A4AE"]    # Blue-grey for V
        COLORS_M = ["#795548", "#8D6E63", "#A1887F"]    # Brown for M
        COLORS_R = ["#009688", "#26A69A", "#4DB6AC"]    # Teal for R
        COLORS_P = ["#E91E63", "#EC407A", "#F06292"]    # Pink for P
        COLORS_W = ["#CDDC39", "#D4E157", "#DCE775"]    # Lime for W
        COLORS_L = ["#FF5722", "#FF7043", "#FF8A65"]    # Deep orange for L
        DAYS_PER_PAGE = 5

        num_pages = math.ceil(len(all_dates) / DAYS_PER_PAGE)
        for page in range(num_pages):
            start = page * DAYS_PER_PAGE
            end = min(start + DAYS_PER_PAGE, len(all_dates))
            page_results = all_results[start:end]
            n_rows = len(page_results)

            fig, axes = plt.subplots(
                n_rows, 2, figsize=(20, 4.5 * n_rows),
                gridspec_kw={"width_ratios": [2.5, 1]},
            )
            if n_rows == 1:
                axes = [axes]

            fig.suptitle(
                f"Combined H+G+A+F+D+V+M+P Page {page+1}/{num_pages}: "
                f"{page_results[0]['date']} to {page_results[-1]['date']}\n"
                f"H: +{H_TARGET_PCT}%/{H_TIME_LIMIT_MINUTES}m (purple) | "
                f"G: +{G_TARGET_PCT}%/{G_TIME_LIMIT_MINUTES}m (green) | "
                f"A: +{A_TARGET_PCT}%/{A_TIME_LIMIT_MINUTES}m (blue) | "
                f"F: +{F_TARGET_PCT}%/{F_TIME_LIMIT_MINUTES}m (orange)\n"
                f"D: +{D_TARGET1_PCT}/{D_TARGET2_PCT}%/{D_TIME_LIMIT_MINUTES}m (cyan) | "
                f"V: +{V_TARGET1_PCT}/{V_TARGET2_PCT}%/{V_TIME_LIMIT_MINUTES}m (grey) | "
                f"M: +{M_TARGET1_PCT}%/{M_TIME_LIMIT_MINUTES}m (brown)\n"
                f"R: +{R_TARGET1_PCT}%/{R_TIME_LIMIT_MINUTES}m (teal) | "
                f"P: +{P_TARGET1_PCT}/{P_TARGET2_PCT}%/{P_TIME_LIMIT_MINUTES}m (pink) | "
                f"L: tiered/{L_TIME_LIMIT_MINUTES}m (d.orange)",
                fontsize=11, fontweight="bold", y=1.01,
            )

            for i, res in enumerate(page_results):
                row_axes = axes[i] if n_rows > 1 else axes[0]
                ax_price, ax_pnl = row_axes[0], row_axes[1]

                traded = [s for s in res["states"] if s["exit_reason"] is not None]

                if traded:
                    color_idx = {k: 0 for k in STRAT_KEYS}
                    color_map = {"H": COLORS_H, "G": COLORS_G, "A": COLORS_A, "F": COLORS_F, "D": COLORS_D, "V": COLORS_V, "M": COLORS_M, "R": COLORS_R, "P": COLORS_P, "W": COLORS_W, "L": COLORS_L}
                    for si, st in enumerate(traded):
                        mh = st["mh"]
                        if mh.index.tz is not None:
                            et_times = mh.index.tz_convert(ET_TZ)
                        else:
                            et_times = mh.index.tz_localize("UTC").tz_convert(ET_TZ)

                        s = st["strategy"] or "A"
                        colors_list = color_map.get(s, COLORS_A)
                        color = colors_list[color_idx.get(s, 0) % len(colors_list)]
                        color_idx[s] = color_idx.get(s, 0) + 1

                        first_candle_close = float(mh.iloc[0]["Close"])
                        ref_price = first_candle_close
                        pct_change = (mh["Close"].values.astype(float) / ref_price - 1) * 100

                        strat_tag = st["strategy"]
                        vc_tag = " VC" if st.get("vol_capped") else ""
                        cost_k = st["position_cost"] / 1000
                        label = (f"[{strat_tag}] {st['ticker']} ${cost_k:.0f}K{vc_tag} "
                                 f"(gap {st['gap_pct']:.0f}%)")
                        ax_price.plot(et_times, pct_change, color=color, linewidth=1.2,
                                      label=label, alpha=0.85)

                        # BUY marker
                        if st["entry_time"] is not None:
                            et_buy = _to_et(st["entry_time"])
                            buy_pct = (st.get("signal_price", ref_price) / ref_price - 1) * 100
                            ax_price.plot(et_buy, buy_pct, marker="^", color=color,
                                          markersize=10, zorder=5)
                            ax_price.annotate(
                                f"BUY({strat_tag})", xy=(et_buy, buy_pct), xytext=(0, 12),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=6, fontweight="bold", color=color,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          ec=color, alpha=0.8, lw=0.5),
                            )

                        # SELL marker
                        if st["exit_time"] is not None and st["exit_price"] is not None:
                            et_sell = _to_et(st["exit_time"])
                            sell_pct = (st["exit_price"] / ref_price - 1) * 100
                            is_win = st["pnl"] > 0
                            marker = "v" if not is_win else "s"
                            sell_color = "#4CAF50" if is_win else "#f44336"
                            ax_price.plot(et_sell, sell_pct, marker=marker, color=sell_color,
                                          markersize=10, zorder=5,
                                          markeredgecolor="white", markeredgewidth=1)
                            reason_short = {"TARGET": "T", "STOP_LOSS": "SL", "STOP": "SL",
                                            "EOD_CLOSE": "EOD", "TIME_STOP": "TS",
                                            "TRAIL": "TR"}.get(
                                st["exit_reason"], st["exit_reason"][:3])
                            ax_price.annotate(
                                reason_short, xy=(et_sell, sell_pct),
                                xytext=(0, -14 if sell_pct >= 0 else 12),
                                textcoords="offset points", ha="center",
                                va="top" if sell_pct >= 0 else "bottom",
                                fontsize=6, fontweight="bold", color=sell_color,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          ec=sell_color, alpha=0.8, lw=0.5),
                            )

                    ax_price.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                    # Show target lines for active strategies this day
                    day_strats = set(s["strategy"] for s in traded)
                    target_lines = {
                        "H": (H_TARGET_PCT, "#9C27B0"), "G": (G_TARGET_PCT, "#4CAF50"),
                        "A": (A_TARGET_PCT, "#2196F3"), "F": (F_TARGET_PCT, "#FF9800"),
                        "D": (D_TARGET1_PCT, "#00BCD4"), "V": (V_TARGET1_PCT, "#607D8B"),
                        "M": (M_TARGET1_PCT, "#795548"),
                        "R": (R_TARGET1_PCT, "#009688"),
                        "P": (P_TARGET1_PCT, "#E91E63"),
                        "W": (W_TARGET_PCT, "#CDDC39"),
                        "L": (L_TIER3_TARGET1_PCT, "#FF5722"),
                    }
                    shown_targets = set()
                    for sk, (tv, tc_line) in target_lines.items():
                        if sk in day_strats and tv not in shown_targets:
                            ax_price.axhline(y=tv, color=tc_line, linestyle=":", alpha=0.4)
                            shown_targets.add(tv)
                    ax_price.set_title(f"{res['date']} - Price (% from 1st candle close)",
                                       fontsize=10, fontweight="bold")
                    ax_price.set_ylabel("% from Entry", fontsize=8)
                    ax_price.legend(fontsize=6, loc="upper left", ncol=2)
                    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p", tz=ET_TZ))
                    ax_price.tick_params(axis="x", rotation=30, labelsize=7)
                    ax_price.grid(alpha=0.3)

                    # P&L bar chart
                    tickers = []
                    for s in traded:
                        vc = " VC" if s.get("vol_capped") else ""
                        cost_k = s["position_cost"] / 1000
                        tickers.append(f"[{s['strategy']}] {s['ticker']} ${cost_k:.0f}K{vc}")
                    pnls = [s["pnl"] for s in traded]
                    pct_profits = [(s["pnl"] / s["position_cost"] * 100) if s["position_cost"] > 0 else 0 for s in traded]
                    reasons = [s["exit_reason"] for s in traded]
                    bar_colors = ["#4CAF50" if p > 0 else "#f44336" for p in pnls]
                    y_pos = range(len(tickers))
                    bars = ax_pnl.barh(y_pos, pnls, color=bar_colors, edgecolor="white", height=0.6)
                    ax_pnl.set_yticks(y_pos)
                    ax_pnl.set_yticklabels([f"{t} ({r[:3]})" for t, r in zip(tickers, reasons)], fontsize=7)
                    ax_pnl.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
                    ax_pnl.invert_yaxis()
                    for j, (bar, pnl, pct) in enumerate(zip(bars, pnls, pct_profits)):
                        x_pos = bar.get_width()
                        align = "left" if pnl >= 0 else "right"
                        offset = 5 if pnl >= 0 else -5
                        ax_pnl.annotate(f"${pnl:+.0f} ({pct:+.1f}%)", xy=(x_pos, j), xytext=(offset, 0),
                                        textcoords="offset points", ha=align, va="center",
                                        fontsize=7, fontweight="bold", color=bar_colors[j])
                    day_total = sum(pnls)
                    tc = "#4CAF50" if day_total >= 0 else "#f44336"
                    bal = res["equity"]
                    ax_pnl.set_title(f"P&L: ${day_total:+,.0f} | Bal: ${bal:,.0f}",
                                     fontsize=10, fontweight="bold", color=tc)
                    ax_pnl.set_xlabel("P&L ($)", fontsize=8)
                    ax_pnl.grid(alpha=0.3, axis="x")
                else:
                    bal = res["equity"]
                    ax_price.text(0.5, 0.5, f"{res['date']}\nNo signals",
                                  ha="center", va="center", fontsize=12,
                                  transform=ax_price.transAxes, color="gray")
                    ax_price.set_title(f"{res['date']} - No Trades", fontsize=10)
                    ax_pnl.text(0.5, 0.5, "0 trades", ha="center", va="center",
                                fontsize=10, transform=ax_pnl.transAxes, color="gray")
                    ax_pnl.set_title(f"P&L: $0 | Bal: ${bal:,.0f}", fontsize=10, color="gray")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            chart_path = os.path.join(run_dir, f"gc_page_{page+1:02d}.png")
            plt.savefig(chart_path, dpi=120, bbox_inches="tight")
            plt.close()
            sys.stdout.write(f"\r  Charts: page {page+1}/{num_pages}")
            sys.stdout.flush()

        print(f"\r  {num_pages} chart pages saved to {run_dir}/          ")

        # --- SUMMARY CHART ---
        fig2 = plt.figure(figsize=(22, 18))
        gs = fig2.add_gridspec(3, 3, width_ratios=[1.2, 1.2, 0.8],
                               height_ratios=[1.0, 1.0, 1.0], hspace=0.35, wspace=0.3)
        fig2.suptitle(
            f"Combined H+G+A+F+D+V+M+R+P Summary: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)",
            fontsize=16, fontweight="bold",
        )

        # 1. Daily P&L bars
        ax = fig2.add_subplot(gs[0, :])
        dates_list = [r["date"] for r in all_results]
        pnls_list = [r["day_pnl"] for r in all_results]
        bar_c = ["#4CAF50" if p >= 0 else "#f44336" for p in pnls_list]
        ax.bar(range(len(dates_list)), pnls_list, color=bar_c, edgecolor="none", width=0.8)
        ax.set_xticks(range(0, len(dates_list), max(1, len(dates_list) // 15)))
        ax.set_xticklabels([dates_list[i] for i in range(0, len(dates_list), max(1, len(dates_list) // 15))],
                           rotation=45, fontsize=8, ha="right")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        tp = sum(pnls_list)
        tc = "#4CAF50" if tp >= 0 else "#f44336"
        ax.set_title(f"Daily P&L | Total: ${tp:+,.0f}", fontsize=13, fontweight="bold", color=tc)
        ax.set_ylabel("P&L ($)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

        # 2. Equity curve
        ax = fig2.add_subplot(gs[1, :])
        equities = [r["equity"] for r in all_results]
        ax.plot(range(len(dates_list)), equities, color="#2196F3", linewidth=2, label="Combined H+G+A+F+D+V+M+R+P+W")
        ax.fill_between(range(len(dates_list)), STARTING_CASH, equities,
                        where=[e >= STARTING_CASH for e in equities], alpha=0.15, color="#4CAF50")
        ax.fill_between(range(len(dates_list)), STARTING_CASH, equities,
                        where=[e < STARTING_CASH for e in equities], alpha=0.15, color="#f44336")
        ax.axhline(y=STARTING_CASH, color="gray", linestyle="--", alpha=0.5,
                   label=f"Start: ${STARTING_CASH:,}")
        ax.axhline(y=MARGIN_THRESHOLD, color="#FF9800", linestyle=":", alpha=0.6,
                   label=f"Margin ${MARGIN_THRESHOLD/1000:.0f}K")

        # Milestone markers with vertical drop-lines and elevated labels
        milestones = [(50_000, "$50K", "#FF9800"), (100_000, "$100K", "#E91E63"),
                      (1_000_000, "$1M", "#9C27B0"), (10_000_000, "$10M", "#F44336")]
        y_max = max(equities)
        for mi, (mlevel, mlabel, mcolor) in enumerate(milestones):
            if y_max >= mlevel:
                ax.axhline(y=mlevel, color=mcolor, linestyle="--", alpha=0.3, linewidth=1)
                for midx, eq in enumerate(equities):
                    if eq >= mlevel:
                        # Vertical drop-line from label to the crossing point
                        label_y = y_max * (0.85 + mi * 0.04)  # stagger labels high
                        ax.plot([midx, midx], [mlevel, label_y], color=mcolor,
                                linestyle="-", linewidth=1, alpha=0.6)
                        ax.plot(midx, mlevel, marker="o", color=mcolor, markersize=6, zorder=5)
                        ax.annotate(f"{mlabel}\n{dates_list[midx]}",
                                    xy=(midx, label_y), xytext=(0, 6),
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=8, color=mcolor, fontweight="bold",
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                              ec=mcolor, alpha=0.9, lw=1))
                        break

        ax.set_xticks(range(0, len(dates_list), max(1, len(dates_list) // 15)))
        ax.set_xticklabels([dates_list[i] for i in range(0, len(dates_list), max(1, len(dates_list) // 15))],
                           rotation=45, fontsize=8, ha="right")
        ax.set_title(f"Equity: ${STARTING_CASH:,} -> ${equities[-1]:,.0f} ({(equities[-1]/STARTING_CASH-1)*100:+.1f}%)",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Balance ($)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 3. Exit reasons pie (split by strategy)
        ax = fig2.add_subplot(gs[2, 0])
        if all_exits:
            reason_colors = {
                "H_TARGET": "#9C27B0", "H_TIME_STOP": "#BA68C8", "H_EOD_CLOSE": "#CE93D8",
                "G_TARGET": "#4CAF50", "G_TIME_STOP": "#81C784", "G_EOD_CLOSE": "#C8E6C9",
                "A_TARGET": "#2196F3", "A_TIME_STOP": "#64B5F6", "A_EOD_CLOSE": "#BBDEFB",
                "F_TARGET": "#FF9800", "F_TIME_STOP": "#FFB74D", "F_EOD_CLOSE": "#FFE0B2",
                "D_TARGET": "#00BCD4", "D_STOP": "#4DD0E1", "D_TRAIL": "#26C6DA",
                "D_TIME_STOP": "#80DEEA", "D_EOD_CLOSE": "#B2EBF2",
                "V_TARGET": "#607D8B", "V_STOP": "#90A4AE", "V_TRAIL": "#78909C",
                "V_TIME_STOP": "#B0BEC5", "V_EOD_CLOSE": "#CFD8DC",
                "M_TARGET": "#795548", "M_STOP": "#A1887F", "M_TRAIL": "#8D6E63",
                "M_TIME_STOP": "#BCAAA4", "M_EOD_CLOSE": "#D7CCC8",
                "R_TARGET": "#009688", "R_STOP": "#4DB6AC", "R_TRAIL": "#26A69A",
                "R_TIME_STOP": "#80CBC4", "R_EOD_CLOSE": "#B2DFDB",
                "P_TARGET": "#E91E63", "P_STOP": "#F48FB1", "P_TRAIL": "#F06292",
                "P_TIME_STOP": "#F8BBD0", "P_EOD_CLOSE": "#FCE4EC",
                "W_TARGET": "#CDDC39", "W_STOP": "#DCE775", "W_TRAIL": "#D4E157",
                "W_TIME_STOP": "#E6EE9C", "W_EOD_CLOSE": "#F0F4C3",
            }
            labels = list(all_exits.keys())
            sizes = list(all_exits.values())
            colors = [reason_colors.get(r, "#999") for r in labels]
            ax.pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
                   colors=colors, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
        ax.set_title("Exit Reasons by Strategy", fontweight="bold")

        # 4. Stats + Stress Test
        ax = fig2.add_subplot(gs[2, 1])
        ax.axis("off")
        stats_text = (
            f"PERFORMANCE\n{'='*38}\n"
            f"Starting:     ${STARTING_CASH:,}\n"
            f"Ending:       ${final_equity:,.0f}\n"
            f"Return:       {(final_equity/STARTING_CASH-1)*100:+.1f}%\n"
            f"{'─'*38}\n"
            f"Total Trades: {total_trades}\n"
        )
        for k in STRAT_KEYS:
            st_t = strat_totals[k]
            st_w = strat_wins[k]
            st_pnl = sum(strat_pnls[k])
            stats_text += f"  {k}: {st_t:>3} ({st_w}W {st_w/max(st_t,1)*100:.0f}%) ${st_pnl:+,.0f}\n"
        stats_text += (
            f"{'─'*38}\n"
            f"Win Rate:     {total_wins/max(total_trades,1)*100:.1f}%\n"
            f"Avg Win:      ${avg_win:+,.0f}\n"
            f"Avg Loss:     ${avg_loss:+,.0f}\n"
            f"Profit Fac:   {pf:.2f}\n"
            f"Sharpe:       {sharpe:.2f}\n"
            f"Green Days:   {green}/{len(daily_pnls)} ({green/max(len(daily_pnls),1)*100:.0f}%)\n"
        )
        if daily_pnls:
            stats_text += (
                f"Best Day:     ${max(daily_pnls):+,.0f}\n"
                f"Worst Day:    ${min(daily_pnls):+,.0f}\n"
            )
        # Stress test section with pass/fail markers
        rem_mark = "OK" if rem_total > 0 else "XX"
        kelly_mark = "OK" if kelly > 0 else "XX"
        mc_mark = "OK" if mc_pct >= 90 else "XX"
        dd_mark = "OK" if max_dd_pct < 0.40 else "XX"
        overall_mark = "OK" if stress_pass else "XX"
        stats_text += (
            f"\n{'='*38}\n"
            f"STRESS TEST\n"
            f"{'='*38}\n"
            f"[{dd_mark}] Max DD:      ${max_dd_dollar:,.0f} ({max_dd_pct*100:.1f}%)\n"
            f"     Loss Streak:  {max_streak} days\n"
        )
        if daily_pnls:
            rem_sharpe = 0.0
            if np.std(daily_pnls) > 0:
                sorted_pnls = sorted(daily_pnls, reverse=True)
                cut = max(1, len(sorted_pnls) // 10)
                rem_pnls = sorted_pnls[cut:]
                if np.std(rem_pnls) > 0:
                    rem_sharpe = (np.mean(rem_pnls) / np.std(rem_pnls)) * np.sqrt(252)
            stats_text += (
                f"[{rem_mark}] Rm Top 10%: ${rem_total:+,.0f} (Sh {rem_sharpe:.2f})\n"
                f"[{kelly_mark}] Kelly:      {kelly*100:.1f}%\n"
                f"[{mc_mark}] Monte Carlo:{mc_pct:.0f}% profitable\n"
            )
        if vc_count > 0:
            stats_text += f"   Vol-Capped:  {vc_count}/{total_trades} trades\n"
        stats_text += f"\n[{overall_mark}] OVERALL: {'ALL PASS' if stress_pass else 'CONCERNS'}\n"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # 5. Config Parameters
        ax = fig2.add_subplot(gs[2, 2])
        ax.axis("off")
        config_text = (
            f"PARAMETERS\n{'='*32}\n"
            f"\nH (High Conviction)\n{'-'*32}\n"
            f"  Gap >= {H_MIN_GAP_PCT:.0f}%\n"
            f"  Body >= {H_MIN_BODY_PCT:.0f}%\n"
            f"  2nd green + new hi\n"
            f"  Vol confirm: {H_REQUIRE_VOL_CONFIRM}\n"
            f"  Target: +{H_TARGET_PCT:.0f}%  Stop: {H_TIME_LIMIT_MINUTES}m\n"
            f"\nG (Runner)\n{'-'*32}\n"
            f"  Gap >= {G_MIN_GAP_PCT:.0f}%\n"
            f"  2nd green + new hi\n"
            f"  Target: +{G_TARGET_PCT:.0f}%  Stop: {G_TIME_LIMIT_MINUTES}m\n"
            f"\nA (Scalp)\n{'-'*32}\n"
            f"  Gap >= {A_MIN_GAP_PCT:.0f}%\n"
            f"  Body >= {A_MIN_BODY_PCT:.0f}%\n"
            f"  2nd green + new hi\n"
            f"  Target: +{A_TARGET_PCT:.0f}%  Stop: {A_TIME_LIMIT_MINUTES}m\n"
            f"\nF (Catch-All)\n{'-'*32}\n"
            f"  Gap >= {F_MIN_GAP_PCT:.0f}%\n"
            f"  2nd green (no hi req)\n"
            f"  Target: +{F_TARGET_PCT:.0f}%  Stop: {F_TIME_LIMIT_MINUTES}m\n"
            f"\nD (Dip Buy)\n{'-'*32}\n"
            f"  Gap >= {D_MIN_GAP_PCT:.0f}%\n"
            f"  Spike >= {D_MIN_SPIKE_PCT:.0f}% in {D_SPIKE_WINDOW}c\n"
            f"  Dip: {D_DIP_PCT:.0f}% | Mode: {D_ENTRY_MODE}\n"
            f"  Max entry: c{D_MAX_ENTRY_CANDLE}\n"
            f"  Target1: +{D_TARGET1_PCT:.0f}% (sell {D_PARTIAL_SELL_PCT:.0f}%)\n"
            f"  Target2: +{D_TARGET2_PCT:.0f}% (runner)\n"
            f"  Stop: -{D_STOP_PCT:.0f}%  Trail: {D_TRAIL_PCT:.0f}% (at +{D_TRAIL_ACTIVATE_PCT:.0f}%)\n"
            f"  Time: {D_TIME_LIMIT_MINUTES}m\n"
            f"\nV (VWAP Reclaim)\n{'-'*32}\n"
            f"  Gap >= {V_MIN_GAP_PCT:.0f}%\n"
            f"  Below VWAP: {V_MIN_BELOW_CANDLES}+ closes\n"
            f"  Min depth: {V_MIN_BELOW_PCT:.0f}% below VWAP\n"
            f"  Vol spike: >= {V_VOL_SPIKE_RATIO:.1f}x\n"
            f"  Target1: +{V_TARGET1_PCT:.0f}% (sell {V_PARTIAL_SELL_PCT:.0f}%)\n"
            f"  Target2: +{V_TARGET2_PCT:.0f}% (runner)\n"
            f"  Stop: -{V_STOP_PCT:.0f}%  Trail: {V_TRAIL_PCT:.0f}% (at +{V_TRAIL_ACTIVATE_PCT:.0f}%)\n"
            f"  Time: {V_TIME_LIMIT_MINUTES}m\n"
            f"\nM (Midday Break)\n{'-'*32}\n"
            f"  Gap >= {M_MIN_GAP_PCT:.0f}%\n"
            f"  Morning spike >= {M_MORNING_SPIKE_PCT:.0f}%\n"
            f"  Consol: c{M_RANGE_START_CANDLE}-{M_RANGE_START_CANDLE+M_CONSOLIDATION_LEN}\n"
            f"  Range <= {M_MAX_RANGE_PCT:.0f}%, Vol <= {M_VOL_RATIO:.1f}x\n"
            f"  Target1: +{M_TARGET1_PCT:.0f}% (sell {M_PARTIAL_SELL_PCT:.0f}%)\n"
            f"  Stop: -{M_STOP_PCT:.0f}%  Trail: {M_TRAIL_PCT:.0f}% (at +{M_TRAIL_ACTIVATE_PCT:.0f}%)\n"
            f"  Time: {M_TIME_LIMIT_MINUTES}m\n"
            f"\nR (Multi-Day)\n{'-'*32}\n"
            f"  Day1 gap >= {R_DAY1_MIN_GAP:.0f}%\n"
            f"  D2 pullback: {R_D2_PULLBACK_PCT:.0f}%\n"
            f"  Bounce > {R_BOUNCE_REF}\n"
            f"  Target: +{R_TARGET1_PCT:.0f}%\n"
            f"  Stop: -{R_STOP_PCT:.0f}%  Trail: {R_TRAIL_PCT:.0f}% (at +{R_TRAIL_ACTIVATE_PCT:.0f}%)\n"
            f"  Time: {R_TIME_LIMIT_MINUTES}m\n"
            f"\nP (PM High Breakout)\n{'-'*32}\n"
            f"  Gap >= {P_MIN_GAP_PCT:.0f}%\n"
            f"  Confirm: {P_CONFIRM_ABOVE}/{P_CONFIRM_WINDOW} closes\n"
            f"  Pullback: {P_PULLBACK_PCT:.1f}%\n"
            f"  Max entry: c{P_MAX_ENTRY_CANDLE}\n"
            f"  Target1: +{P_TARGET1_PCT:.0f}% (sell {P_PARTIAL_SELL_PCT:.0f}%)\n"
            f"  Target2: +{P_TARGET2_PCT:.0f}% (runner)\n"
            f"  Stop: -{P_STOP_PCT:.0f}%  Trail: {P_TRAIL_PCT:.0f}% (at +{P_TRAIL_ACTIVATE_PCT:.0f}%)\n"
            f"  Time: {P_TIME_LIMIT_MINUTES}m\n"
            f"\nW (Power Hour)\n{'-'*32}\n"
            f"  Gap >= {W_MIN_GAP_PCT:.0f}%\n"
            f"  Morning run >= {W_MIN_MORNING_RUN:.0f}%\n"
            f"  Consol: c{W_CONSOL_START}-{W_EARLIEST_CANDLE}\n"
            f"  Range <= {W_MAX_RANGE_PCT:.0f}%\n"
            f"  VWAP dev <= {W_MAX_VWAP_DEV_PCT:.0f}%\n"
            f"  Vol surge >= {W_VOL_SURGE_MULT:.1f}x\n"
            f"  Target: +{W_TARGET_PCT:.0f}%\n"
            f"  Stop: -{W_STOP_PCT:.1f}%  Trail: {W_TRAIL_PCT:.0f}%\n"
            f"\nSHARED\n{'-'*32}\n"
            f"  Priority:  H>G>A>F>D>V>M>R>P>W\n"
            f"  Sizing:    100% balance\n"
            f"  Slippage:  {SLIPPAGE_PCT}%\n"
            f"  Vol cap:   {VOL_CAP_PCT}%\n"
            f"  Start:     ${STARTING_CASH:,}\n"
            f"  Margin:    ${MARGIN_THRESHOLD:,}\n"
            f"\n  {all_dates[0]} to {all_dates[-1]}\n"
            f"  {len(all_dates)} trading days\n"
        )
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.7))

        plt.tight_layout()
        summary_path = os.path.join(run_dir, "gc_summary.png")
        plt.savefig(summary_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Summary chart saved to {summary_path}")
