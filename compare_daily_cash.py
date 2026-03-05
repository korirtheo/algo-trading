"""Compare day-by-day cash between old_twopass reproduction and $71M actual."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import test_full as tf
from regime_filters import RegimeFilter
from verify_root_cause2 import simulate_day_old_twopass

DATA_DIRS = tf.DEFAULT_DATA_DIRS
ALL_DATES, daily_picks = tf.load_all_picks(DATA_DIRS)

regime = RegimeFilter(
    spy_ma_period=tf.SPY_SMA_PERIOD,
    enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
)
regime.load_data(ALL_DATES[0], ALL_DATES[-1])

# $71M run actual cash (extracted from run output)
ACTUAL_71M = {
    "2025-01-03": 10000.00,
    "2025-01-06": 11984.63,
    "2025-01-07": 12533.24,
    "2025-01-08": 11527.95,
    "2025-01-10": 11220.65,
    "2025-01-13": 11208.17,
    "2025-01-14": 14031.35,
    "2025-01-15": 12118.08,
    "2025-01-16": 13197.36,
    "2025-01-17": 12674.55,
    "2025-01-21": 12692.40,
    "2025-01-22": 12160.85,
    "2025-01-23": 14571.29,
    "2025-01-24": 14813.86,
    "2025-01-27": 18176.68,
}

# Run old_twopass with SI 19% / gate ON
tf.SCALE_IN = 1
tf.SCALE_IN_TRIGGER_PCT = 19.0
tf.SCALE_IN_GATE_PARTIAL = True
tf.BREAKEVEN_STOP_PCT = 0.0
tf.ENTRY_CUTOFF_MINUTES = 0

rolling_cash = tf.STARTING_CASH
print(f"{'Date':<12} {'$71M Actual':>14} {'Old_2pass':>14} {'Diff':>12} {'Diff%':>8}")
print("-" * 62)

for date_str in ALL_DATES:
    if date_str > "2025-01-27":
        break

    actual = ACTUAL_71M.get(date_str)
    if actual is not None:
        diff = rolling_cash - actual
        pct = diff / actual * 100 if actual else 0
        print(f"{date_str:<12} ${actual:>12,.2f} ${rolling_cash:>12,.2f} ${diff:>+10,.2f} {pct:>+7.1f}%")

    should_trade, _, _ = regime.check(date_str)
    if not should_trade:
        continue
    picks = daily_picks.get(date_str, [])
    if not picks:
        continue
    is_cash = rolling_cash < tf.MARGIN_THRESHOLD

    # Print detailed trades for first day where divergence > $100
    if actual and abs(rolling_cash - actual) > 100:
        pass  # Already diverged

    states, ending_cash = simulate_day_old_twopass(picks, rolling_cash, cash_account=is_cash)

    # Print trade details for Jan 3 (first day) to compare
    if date_str == "2025-01-03":
        print(f"\n  JAN 3 TRADES (starting ${rolling_cash:,.2f}):")
        for s in states:
            if s["entry_price"] is not None:
                pnl = s["total_exit_value"] - s["total_cash_spent"]
                si = "SI" if s.get("scaled_in") else ""
                ps = "T1" if s.get("partial_sold") else ""
                print(f"    {s['ticker']:<8} cost=${s['total_cash_spent']:,.0f} "
                      f"entry=${s['entry_price']:.2f} exit=${s.get('exit_price','N/A')} "
                      f"pnl=${pnl:+,.2f} {s['exit_reason']} {si} {ps}")
        print(f"  Day ending: ${ending_cash:,.2f}\n")

    rolling_cash = ending_cash
