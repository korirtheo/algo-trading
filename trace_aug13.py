"""Trace all trades on August 13, 2025 in detail."""
import os, sys, pickle, pandas as pd, numpy as np
from zoneinfo import ZoneInfo

# Import config from test_full
sys.path.insert(0, os.path.dirname(__file__))
from test_full import (
    load_all_picks, simulate_day, STARTING_CASH, TRADE_PCT,
    STOP_LOSS_PCT, PARTIAL_SELL_PCT, PARTIAL_SELL_FRAC,
    PARTIAL_SELL_PCT_2, PARTIAL_SELL_FRAC_2, ATR_PERIOD, ATR_MULTIPLIER,
    SCALE_IN, SCALE_IN_TRIGGER_PCT, N_EXIT_TRANCHES, MIN_PM_VOLUME,
    SPY_SMA_PERIOD, SLIPPAGE_PCT, CONFIRM_ABOVE, CONFIRM_WINDOW,
    PULLBACK_PCT, PULLBACK_TIMEOUT
)

ET = ZoneInfo("America/New_York")
TARGET_DATE = "2025-08-13"

print("Loading picks...")
ALL_DATES, daily_picks = load_all_picks(["stored_data_oos"])

# Load regime filter
from regime_filters import RegimeFilter
regime = RegimeFilter(spy_ma_period=SPY_SMA_PERIOD, enable_vix=False, enable_spy_trend=True, enable_adaptive=False)
regime.load_data(ALL_DATES[0], ALL_DATES[-1])

# Roll cash forward to target date
rolling_cash = STARTING_CASH
for date_str in ALL_DATES:
    if date_str == TARGET_DATE:
        break
    should_trade, _, _ = regime.check(date_str)
    if not should_trade:
        continue
    picks = daily_picks[date_str]
    states, ending_cash = simulate_day(picks, rolling_cash)
    rolling_cash = ending_cash

print(f"\n{'='*80}")
print(f"  TRADE TRACE: {TARGET_DATE}")
print(f"  Starting cash: ${rolling_cash:,.2f}")
print(f"{'='*80}")

# Check regime
should_trade, _, regime_info = regime.check(TARGET_DATE)
print(f"  Regime: {'TRADE' if should_trade else 'SKIP'} ({regime_info})")
if not should_trade:
    print("  Day skipped by regime filter.")
    sys.exit(0)

picks = daily_picks[TARGET_DATE]
print(f"  Candidates: {len(picks)} stocks")
for i, p in enumerate(picks):
    print(f"    {i+1}. {p['ticker']:6s} gap={p['gap_pct']:+.1f}%  pm_high=${p['premarket_high']:.2f}  pm_vol={p.get('pm_volume',0):,.0f}")

# Run simulation
states, ending_cash = simulate_day(picks, rolling_cash)

# Print results
traded = [s for s in states if s["entry_price"] is not None]
not_traded = [s for s in states if s["entry_price"] is None]

print(f"\n  Trades taken: {len(traded)}")
print(f"  Not traded: {len(not_traded)} ({', '.join(s['ticker']+':'+s['exit_reason'] for s in not_traded)})")
print(f"\n{'─'*80}")

for i, s in enumerate(traded):
    entry_et = s["entry_time"].tz_convert(ET) if s["entry_time"] and s["entry_time"].tzinfo else s["entry_time"]
    exit_et = s["exit_time"].tz_convert(ET) if s["exit_time"] and s["exit_time"].tzinfo else s["exit_time"]

    print(f"\n  TRADE {i+1}: {s['ticker']}")
    print(f"  {'─'*40}")
    print(f"  Gap:           {s['gap_pct']:+.1f}%")
    print(f"  PM High:       ${s['premarket_high']:.2f}")
    print(f"  PM Volume:     {s['pm_volume']:,.0f}")
    print(f"  Entry Price:   ${s['entry_price']:.2f}  @ {entry_et.strftime('%I:%M %p ET') if entry_et else 'N/A'}")
    print(f"  Position Cost: ${s['position_cost']:,.2f}  ({s['shares']:.1f} shares)")

    if s.get("scaled_in") and s.get("scale_in_price"):
        si_et = s["scale_in_time"].tz_convert(ET) if s["scale_in_time"] and s["scale_in_time"].tzinfo else s["scale_in_time"]
        print(f"  Scale-in:      ${s['scale_in_price']:.2f}  @ {si_et.strftime('%I:%M %p ET') if si_et else 'N/A'}")

    if s.get("partial_sold") and s.get("partial_sell_price"):
        ps_et = s["partial_sell_time"].tz_convert(ET) if s["partial_sell_time"] and s["partial_sell_time"].tzinfo else s["partial_sell_time"]
        print(f"  Partial Sell:  ${s['partial_sell_price']:.2f}  @ {ps_et.strftime('%I:%M %p ET') if ps_et else 'N/A'}")

    exit_price_str = f"${s['exit_price']:.2f}" if s['exit_price'] is not None else "N/A"
    exit_time_str = exit_et.strftime('%I:%M %p ET') if exit_et else "N/A"
    print(f"  Exit Price:    {exit_price_str}  @ {exit_time_str}")
    print(f"  Exit Reason:   {s['exit_reason']}")
    print(f"  Total Value:   ${s['total_exit_value']:,.2f}")
    print(f"  P&L:           ${s['pnl']:+,.2f}  ({s['pnl_pct']:+.1f}%)")

day_pnl = sum(s["pnl"] for s in traded)
print(f"\n{'─'*80}")
print(f"  Day P&L:       ${day_pnl:+,.2f}")
print(f"  Ending Cash:   ${ending_cash:,.2f}")
print(f"{'='*80}")
