"""Quick verification that scale-in logic is sound."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import test_full as tf
from regime_filters import RegimeFilter

DATA_DIRS = tf.DEFAULT_DATA_DIRS

# Enable scale-in fix
tf.BREAKEVEN_STOP_PCT = 0.0
tf.ENTRY_CUTOFF_MINUTES = 0
tf.SCALE_IN_TRIGGER_PCT = 14.0
tf.SCALE_IN_GATE_PARTIAL = False

ALL_DATES, daily_picks = tf.load_all_picks(DATA_DIRS)

regime = RegimeFilter(
    spy_ma_period=tf.SPY_SMA_PERIOD,
    enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
)
regime.load_data(ALL_DATES[0], ALL_DATES[-1])

rolling_cash = tf.STARTING_CASH
min_cash = float('inf')
negative_cash_days = []
scale_in_examples = []

for date_str in ALL_DATES:
    should_trade, _, _ = regime.check(date_str)
    if not should_trade:
        continue

    picks = daily_picks.get(date_str, [])
    if not picks:
        continue

    is_cash = rolling_cash < tf.MARGIN_THRESHOLD
    states, ending_cash = tf.simulate_day(picks, rolling_cash, cash_account=is_cash)

    # Check for negative cash within the day
    day_pnl = sum(s["pnl"] for s in states if s["entry_price"] is not None)

    # Track min cash
    if ending_cash < min_cash:
        min_cash = ending_cash
    if ending_cash < 0:
        negative_cash_days.append((date_str, ending_cash))

    # Track scale-in examples (first 10)
    for s in states:
        if s.get("scaled_in") and s["entry_price"] is not None and s["scale_in_time"] is not None and len(scale_in_examples) < 10:
            scale_in_examples.append({
                "date": date_str,
                "ticker": s["ticker"],
                "orig_entry": s["original_entry_price"],
                "scale_in_price": s["scale_in_price"],
                "avg_entry": s["entry_price"],
                "orig_size": s["original_position_size"],
                "total_spent": s["total_cash_spent"],
                "total_exit": s["total_exit_value"],
                "pnl": s["pnl"],
                "pnl_pct": s["pnl_pct"],
                "exit_reason": s["exit_reason"],
                "shares": s["shares"],
                "exit_price": s.get("exit_price"),
            })

    rolling_cash = ending_cash

print(f"Final Cash:     ${rolling_cash:,.2f}")
print(f"Min Cash:       ${min_cash:,.2f}")
print(f"Negative days:  {len(negative_cash_days)}")
if negative_cash_days:
    for d, c in negative_cash_days[:5]:
        print(f"  {d}: ${c:,.2f}")

print(f"\n{'='*90}")
print(f"  SCALE-IN TRADE EXAMPLES (first 10)")
print(f"{'='*90}")
print(f"  {'Date':<12} {'Ticker':<8} {'Entry':>8} {'SI@':>8} {'AvgEntry':>8} {'Orig$':>10} {'Total$':>10} {'Exit$':>10} {'P&L':>12} {'P&L%':>7} {'Exit':>12}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*7} {'-'*12}")

for e in scale_in_examples:
    print(f"  {e['date']:<12} {e['ticker']:<8} ${e['orig_entry']:>6.2f} ${e['scale_in_price']:>6.2f} ${e['avg_entry']:>6.2f} ${e['orig_size']:>9,.0f} ${e['total_spent']:>9,.0f} ${e['total_exit']:>9,.0f} ${e['pnl']:>+11,.2f} {e['pnl_pct']:>+6.1f}% {e['exit_reason']:>12}")

# Verify P&L math on examples
print(f"\n  P&L VERIFICATION:")
for e in scale_in_examples[:3]:
    expected_pnl = e["total_exit"] - e["total_spent"]
    print(f"  {e['ticker']}: exit_value(${e['total_exit']:,.2f}) - spent(${e['total_spent']:,.2f}) = ${expected_pnl:+,.2f} (reported: ${e['pnl']:+,.2f}) {'OK' if abs(expected_pnl - e['pnl']) < 0.01 else 'MISMATCH!'}")
