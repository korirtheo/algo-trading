"""
Verify same-candle entry+exit look-ahead bias.
Counts how often the OLD code would trigger partial sell on the entry candle.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import test_full as tf
from regime_filters import RegimeFilter

DATA_DIRS = tf.DEFAULT_DATA_DIRS

# Turn off scale-in to isolate the entry+exit bias
tf.SCALE_IN = 0
tf.BREAKEVEN_STOP_PCT = 0.0
tf.ENTRY_CUTOFF_MINUTES = 0

ALL_DATES, daily_picks = tf.load_all_picks(DATA_DIRS)

regime = RegimeFilter(
    spy_ma_period=tf.SPY_SMA_PERIOD,
    enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
)
regime.load_data(ALL_DATES[0], ALL_DATES[-1])

same_candle_sells = 0
total_entries = 0
same_candle_pnl = 0.0

for date_str in ALL_DATES:
    should_trade, _, _ = regime.check(date_str)
    if not should_trade:
        continue
    picks = daily_picks.get(date_str, [])
    if not picks:
        continue

    for pick in picks:
        mh = pick["market_hour_candles"]
        pm_high = pick["premarket_high"]

        # Simulate 3-phase entry detection
        recent_closes = []
        breakout_confirmed = False
        pullback_detected = False
        candles_since_confirm = 0

        for ts in sorted(mh.index):
            candle = mh.loc[ts]
            c_high = float(candle["High"])
            c_low = float(candle["Low"])
            c_close = float(candle["Close"])

            entered = False

            if not breakout_confirmed:
                recent_closes.append(c_close > pm_high)
                if len(recent_closes) > tf.CONFIRM_WINDOW:
                    recent_closes = recent_closes[-tf.CONFIRM_WINDOW:]
                if sum(recent_closes) >= tf.CONFIRM_ABOVE:
                    breakout_confirmed = True
            elif not pullback_detected:
                candles_since_confirm += 1
                pullback_zone = pm_high * (1 + tf.PULLBACK_PCT / 100)
                if c_low <= pullback_zone:
                    pullback_detected = True
                    if c_close > pm_high:
                        entered = True
                elif candles_since_confirm >= tf.PULLBACK_TIMEOUT:
                    if c_close > pm_high:
                        entered = True
            else:
                if c_close > pm_high:
                    entered = True

            if entered:
                total_entries += 1
                entry_price = c_close * (1 + tf.SLIPPAGE_PCT / 100)
                target_price = entry_price * (1 + tf.PARTIAL_SELL_PCT / 100)

                if c_high >= target_price:
                    same_candle_sells += 1
                    # Calculate what profit was locked on 90% of position
                    sell_price = target_price * (1 - tf.SLIPPAGE_PCT / 100)
                    pnl_pct = (sell_price - entry_price) / entry_price * 100
                    same_candle_pnl += pnl_pct
                break  # only first entry per stock

print(f"\n{'='*60}")
print(f"  SAME-CANDLE ENTRY+EXIT ANALYSIS (look-ahead bias)")
print(f"{'='*60}")
print(f"  Total entries detected:    {total_entries}")
print(f"  Same-candle partial sells: {same_candle_sells} ({same_candle_sells/total_entries*100:.1f}%)")
print(f"  Avg P&L on same-candle:    {same_candle_pnl/same_candle_sells:.1f}%" if same_candle_sells > 0 else "")
print(f"\n  In the old single-pass code, these entries would IMMEDIATELY")
print(f"  lock in +{tf.PARTIAL_SELL_PCT}% on 90% of position with ZERO risk.")
print(f"  This is look-ahead bias: buying at c_close, selling at c_high")
print(f"  which already happened earlier in the same candle.")
print(f"{'='*60}")
