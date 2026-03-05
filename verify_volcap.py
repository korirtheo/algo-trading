"""Quick test: does disabling VOL_CAP restore the $71M-range results?"""
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

TESTS = [
    ("VOL_CAP=5% (current)", {"VOL_CAP_PCT": 5.0, "SCALE_IN": 0}),
    ("VOL_CAP=0 (disabled), SI OFF", {"VOL_CAP_PCT": 0.0, "SCALE_IN": 0}),
    ("VOL_CAP=0, SI 19% gated (old cfg)", {"VOL_CAP_PCT": 0.0, "SCALE_IN": 1, "SCALE_IN_TRIGGER_PCT": 19.0, "SCALE_IN_GATE_PARTIAL": True}),
    ("VOL_CAP=5%, SI 19% gated", {"VOL_CAP_PCT": 5.0, "SCALE_IN": 1, "SCALE_IN_TRIGGER_PCT": 19.0, "SCALE_IN_GATE_PARTIAL": True}),
]

for label, cfg in TESTS:
    saves = {}
    for k, v in cfg.items():
        saves[k] = getattr(tf, k)
        setattr(tf, k, v)
    # Also force no cutoff, no breakeven
    saves2 = {"BREAKEVEN_STOP_PCT": getattr(tf, "BREAKEVEN_STOP_PCT"), "ENTRY_CUTOFF_MINUTES": getattr(tf, "ENTRY_CUTOFF_MINUTES")}
    tf.BREAKEVEN_STOP_PCT = 0.0
    tf.ENTRY_CUTOFF_MINUTES = 0

    rolling_cash = tf.STARTING_CASH
    si = 0
    trades = 0
    volcapped = 0
    for date_str in ALL_DATES:
        should_trade, _, _ = regime.check(date_str)
        if not should_trade:
            continue
        picks = daily_picks.get(date_str, [])
        if not picks:
            continue
        is_cash = rolling_cash < tf.MARGIN_THRESHOLD
        states, ending_cash = tf.simulate_day(picks, rolling_cash, cash_account=is_cash)
        for s in states:
            if s["entry_price"] is not None:
                trades += 1
                if s.get("scaled_in") and s.get("scale_in_time"):
                    si += 1
                if s.get("vol_capped"):
                    volcapped += 1
        rolling_cash = ending_cash

    print(f"  {label:<40} ${rolling_cash:>15,.2f}  trades={trades}  SI={si}  VolCap={volcapped}")

    for k, v in saves.items():
        setattr(tf, k, v)
    for k, v in saves2.items():
        setattr(tf, k, v)
