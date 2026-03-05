"""Check regime filter for early January dates."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import test_full as tf
from regime_filters import RegimeFilter

regime = RegimeFilter(
    spy_ma_period=tf.SPY_SMA_PERIOD,
    enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
)

DATA_DIRS = tf.DEFAULT_DATA_DIRS
ALL_DATES, daily_picks = tf.load_all_picks(DATA_DIRS)
regime.load_data(ALL_DATES[0], ALL_DATES[-1])

print(f"SPY SMA period: {tf.SPY_SMA_PERIOD}")
print(f"Date range: {ALL_DATES[0]} to {ALL_DATES[-1]}")
print(f"Total dates: {len(ALL_DATES)}")
print()

# Check first 20 dates
for date_str in ALL_DATES[:20]:
    should_trade, reason, details = regime.check(date_str)
    picks = daily_picks.get(date_str, [])
    print(f"  {date_str}: trade={should_trade:>5}  reason={reason:<20} picks={len(picks)}  {details}")
