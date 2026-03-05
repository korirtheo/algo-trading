"""
Test: what if the $71M code didn't have VOL_CAP, T+1 settlement, or adaptive sizing?
Run the old two-pass sim with these features disabled to narrow down.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import test_full as tf
from regime_filters import RegimeFilter

DATA_DIRS = tf.DEFAULT_DATA_DIRS
ALL_DATES, daily_picks = tf.load_all_picks(DATA_DIRS)

regime = RegimeFilter(
    spy_ma_period=tf.SPY_SMA_PERIOD,
    enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
)
regime.load_data(ALL_DATES[0], ALL_DATES[-1])

# Import the old two-pass sim
from verify_root_cause2 import simulate_day_old_twopass

# Set old config
tf.SCALE_IN = 1
tf.SCALE_IN_TRIGGER_PCT = 19.0
tf.SCALE_IN_GATE_PARTIAL = True
tf.BREAKEVEN_STOP_PCT = 0.0
tf.ENTRY_CUTOFF_MINUTES = 0

TESTS = {
    "1: Old 2pass + all current features": {
        "VOL_CAP_PCT": 5.0,
        "EOD_EXIT_MINUTES": 30,
        "ADAPTIVE_SIZING": True,
    },
    "2: Old 2pass + no VOL_CAP": {
        "VOL_CAP_PCT": 0.0,
        "EOD_EXIT_MINUTES": 30,
        "ADAPTIVE_SIZING": True,
    },
    "3: Old 2pass + no VOL_CAP + no EOD_EXIT": {
        "VOL_CAP_PCT": 0.0,
        "EOD_EXIT_MINUTES": 0,
        "ADAPTIVE_SIZING": True,
    },
    "4: Old 2pass + no VOL_CAP + no EOD_EXIT + fixed 50%": {
        "VOL_CAP_PCT": 0.0,
        "EOD_EXIT_MINUTES": 0,
        "ADAPTIVE_SIZING": False,
    },
}

for label, cfg in TESTS.items():
    saves = {}
    for k, v in cfg.items():
        saves[k] = getattr(tf, k)
        setattr(tf, k, v)

    rolling_cash = tf.STARTING_CASH
    scale_ins = 0
    total_trades = 0
    wins = 0

    for date_str in ALL_DATES:
        should_trade, _, _ = regime.check(date_str)
        if not should_trade:
            continue
        picks = daily_picks.get(date_str, [])
        if not picks:
            continue
        # For test 4, force margin (no T+1) by always passing cash_account=False
        if not cfg.get("ADAPTIVE_SIZING", True):
            is_cash = False  # always margin = instant settlement
        else:
            is_cash = rolling_cash < tf.MARGIN_THRESHOLD
        states, ending_cash = simulate_day_old_twopass(picks, rolling_cash, cash_account=is_cash)
        for s in states:
            if s["entry_price"] is not None:
                total_trades += 1
                if s["pnl"] > 0:
                    wins += 1
                if s.get("scaled_in") and s.get("scale_in_time") is not None:
                    scale_ins += 1
        rolling_cash = ending_cash

    wr = wins / total_trades * 100 if total_trades > 0 else 0
    print(f"  {label}")
    print(f"    Final: ${rolling_cash:>15,.2f}  Trades: {total_trades}  WR: {wr:.1f}%  SI: {scale_ins}")

    for k, v in saves.items():
        setattr(tf, k, v)

print(f"\n  Target: $71,084,943 (old baseline)")
