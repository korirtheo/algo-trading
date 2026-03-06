"""
Fill Reality Check: Can we actually fill these trades without moving the market?

Analyzes actual position sizes vs stock liquidity at time of entry.
Key metrics:
  - Position size as % of candle volume at entry
  - Position size as % of first 10 minutes total volume
  - Dollar volume available vs dollars we're trying to deploy
  - How many shares we need vs shares traded in the candle
"""
import sys
import numpy as np
from collections import defaultdict
from zoneinfo import ZoneInfo

from test_full import (
    load_all_picks,
    SLIPPAGE_PCT,
    STARTING_CASH,
    MARGIN_THRESHOLD,
    VOL_CAP_PCT,
    ET_TZ,
)
import test_green_candle_combined as tgcc

DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2025-01-01", "2026-02-28")

print("=" * 70)
print("  FILL REALITY CHECK")
print("=" * 70)

print("\nLoading data...")
all_dates, daily_picks = load_all_picks(DATA_DIRS)
all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
print(f"  {len(all_dates)} trading days\n")

# Run full sim to get actual position sizes
cash = STARTING_CASH
unsettled = 0.0
all_trades = []

for d in all_dates:
    if unsettled > 0:
        cash += unsettled
        unsettled = 0.0

    picks = daily_picks.get(d, [])
    cash_account = cash < MARGIN_THRESHOLD
    states, new_cash, new_unsettled = tgcc.simulate_day_combined(picks, cash, cash_account)

    for st in states:
        if st["exit_reason"] is not None:
            mh = st["mh"]
            position_cost = st["position_cost"]
            if position_cost <= 0:
                continue
            # entry_price gets cleared on exit, reconstruct from exit_price or cost/shares
            # Use exit_price as proxy for price level (close enough for volume analysis)
            entry_price = st.get("exit_price") or 0
            if entry_price <= 0:
                continue
            shares_needed = position_cost / entry_price

            # Volume at entry candle (candle 2 = index 1)
            if len(mh) >= 2:
                entry_candle_vol = float(mh.iloc[1]["Volume"])
                entry_candle_dollar_vol = entry_candle_vol * entry_price

                # Volume in first 5 candles (10 minutes)
                first_10min_vol = float(mh.iloc[:5]["Volume"].sum()) if len(mh) >= 5 else float(mh["Volume"].sum())
                first_10min_dollar = first_10min_vol * entry_price

                # Total day volume
                total_day_vol = float(mh["Volume"].sum())
                total_day_dollar = total_day_vol * entry_price

                # Our share of volume
                pct_of_entry_candle = (shares_needed / entry_candle_vol * 100) if entry_candle_vol > 0 else 999
                pct_of_10min = (shares_needed / first_10min_vol * 100) if first_10min_vol > 0 else 999
                pct_of_day = (shares_needed / total_day_vol * 100) if total_day_vol > 0 else 999

                all_trades.append({
                    "date": d,
                    "ticker": st["ticker"],
                    "strategy": st["strategy"],
                    "position_cost": position_cost,
                    "entry_price": entry_price,
                    "shares_needed": shares_needed,
                    "vol_capped": st.get("vol_capped", False),
                    "pnl": st["pnl"],
                    "pnl_pct": (st["pnl"] / position_cost * 100) if position_cost > 0 else 0,
                    # Volume metrics
                    "entry_candle_vol": entry_candle_vol,
                    "entry_candle_dollar_vol": entry_candle_dollar_vol,
                    "first_10min_vol": first_10min_vol,
                    "first_10min_dollar": first_10min_dollar,
                    "total_day_vol": total_day_vol,
                    "total_day_dollar": total_day_dollar,
                    "pct_of_entry_candle": pct_of_entry_candle,
                    "pct_of_10min": pct_of_10min,
                    "pct_of_day": pct_of_day,
                })

    cash = new_cash
    unsettled = new_unsettled

print(f"  Analyzed {len(all_trades)} trades\n")

# --- OVERALL STATS ---
print("=" * 70)
print("  POSITION SIZE vs MARKET VOLUME")
print("=" * 70)
print(f"\n  Your order as % of ENTRY CANDLE volume (2-min bar):")
pcts = [t["pct_of_entry_candle"] for t in all_trades]
print(f"    Median: {np.median(pcts):.1f}%  Mean: {np.mean(pcts):.1f}%  "
      f"P75: {np.percentile(pcts, 75):.1f}%  P90: {np.percentile(pcts, 90):.1f}%  "
      f"Max: {max(pcts):.1f}%")
print(f"    Trades > 10% of candle: {sum(1 for p in pcts if p > 10)}/{len(pcts)}")
print(f"    Trades > 25% of candle: {sum(1 for p in pcts if p > 25)}/{len(pcts)}")
print(f"    Trades > 50% of candle: {sum(1 for p in pcts if p > 50)}/{len(pcts)}")

print(f"\n  Your order as % of FIRST 10 MINUTES volume:")
pcts10 = [t["pct_of_10min"] for t in all_trades]
print(f"    Median: {np.median(pcts10):.1f}%  Mean: {np.mean(pcts10):.1f}%  "
      f"P75: {np.percentile(pcts10, 75):.1f}%  P90: {np.percentile(pcts10, 90):.1f}%  "
      f"Max: {max(pcts10):.1f}%")
print(f"    Trades > 5% of 10min:  {sum(1 for p in pcts10 if p > 5)}/{len(pcts10)}")
print(f"    Trades > 10% of 10min: {sum(1 for p in pcts10 if p > 10)}/{len(pcts10)}")

print(f"\n  Your order as % of FULL DAY volume:")
pcts_day = [t["pct_of_day"] for t in all_trades]
print(f"    Median: {np.median(pcts_day):.1f}%  Mean: {np.mean(pcts_day):.1f}%  "
      f"P75: {np.percentile(pcts_day, 75):.1f}%  P90: {np.percentile(pcts_day, 90):.1f}%  "
      f"Max: {max(pcts_day):.1f}%")

# --- DOLLAR SIZE ---
print(f"\n  Position size ($):")
costs = [t["position_cost"] for t in all_trades]
print(f"    Median: ${np.median(costs):,.0f}  Mean: ${np.mean(costs):,.0f}  "
      f"Max: ${max(costs):,.0f}")

print(f"\n  Entry candle dollar volume ($):")
dvols = [t["entry_candle_dollar_vol"] for t in all_trades]
print(f"    Median: ${np.median(dvols):,.0f}  Mean: ${np.mean(dvols):,.0f}")

# --- BY BALANCE TIER ---
print(f"\n{'='*70}")
print("  BREAKDOWN BY ACCOUNT SIZE TIER")
print(f"{'='*70}")

tiers = [
    ("$25K-$100K", 25_000, 100_000),
    ("$100K-$500K", 100_000, 500_000),
    ("$500K-$1M", 500_000, 1_000_000),
    ("$1M-$3M", 1_000_000, 3_000_000),
    ("$3M+", 3_000_000, float("inf")),
]

for label, lo, hi in tiers:
    tier_trades = [t for t in all_trades if lo <= t["position_cost"] < hi]
    if not tier_trades:
        continue

    n = len(tier_trades)
    avg_cost = np.mean([t["position_cost"] for t in tier_trades])
    avg_pct_candle = np.mean([t["pct_of_entry_candle"] for t in tier_trades])
    avg_pct_10min = np.mean([t["pct_of_10min"] for t in tier_trades])
    max_pct_candle = max(t["pct_of_entry_candle"] for t in tier_trades)
    vc_count = sum(1 for t in tier_trades if t["vol_capped"])
    over_10 = sum(1 for t in tier_trades if t["pct_of_entry_candle"] > 10)

    print(f"\n  {label}: {n} trades (avg ${avg_cost:,.0f})")
    print(f"    Avg % of entry candle: {avg_pct_candle:.1f}%  Max: {max_pct_candle:.1f}%")
    print(f"    Avg % of 10min vol:    {avg_pct_10min:.1f}%")
    print(f"    Vol-capped: {vc_count}  >10% of candle: {over_10}")

# --- WORST OFFENDERS ---
print(f"\n{'='*70}")
print("  TOP 20 LARGEST ORDERS (% of entry candle volume)")
print(f"{'='*70}")
sorted_trades = sorted(all_trades, key=lambda t: t["pct_of_entry_candle"], reverse=True)
print(f"  {'Date':<12} {'Ticker':<7} {'S':>1} {'Position':>12} {'Entry CandleVol':>16} "
      f"{'%Candle':>8} {'%10min':>7} {'%Day':>5} {'VC':>3} {'PnL%':>7}")
print(f"  {'-'*85}")
for t in sorted_trades[:20]:
    vc = "Y" if t["vol_capped"] else ""
    print(f"  {t['date']:<12} {t['ticker']:<7} {t['strategy']:>1} "
          f"${t['position_cost']:>11,.0f} ${t['entry_candle_dollar_vol']:>15,.0f} "
          f"{t['pct_of_entry_candle']:>7.1f}% {t['pct_of_10min']:>6.1f}% {t['pct_of_day']:>4.1f}% "
          f"{vc:>3} {t['pnl_pct']:>+6.1f}%")

# --- REALISTIC CAPACITY ANALYSIS ---
print(f"\n{'='*70}")
print("  REALISTIC CAPACITY: WHAT ACCOUNT SIZE CAN THIS STRATEGY HANDLE?")
print(f"{'='*70}")
print("  Rule of thumb: your order should be <5-10% of entry candle volume")
print("  to avoid significant market impact on small-cap gap stocks.\n")

# For each trade, compute max position that stays under 5% of entry candle vol
max_safe_positions = []
for t in all_trades:
    safe_5pct = t["entry_candle_dollar_vol"] * 0.05  # 5% of candle vol
    safe_10pct = t["entry_candle_dollar_vol"] * 0.10  # 10% of candle vol
    max_safe_positions.append({
        "safe_5pct": safe_5pct,
        "safe_10pct": safe_10pct,
        "actual": t["position_cost"],
    })

safe5 = [p["safe_5pct"] for p in max_safe_positions]
safe10 = [p["safe_10pct"] for p in max_safe_positions]

print(f"  Max safe position (<5% of candle):")
print(f"    Median: ${np.median(safe5):,.0f}  P10: ${np.percentile(safe5, 10):,.0f}  "
      f"P25: ${np.percentile(safe5, 25):,.0f}  Min: ${min(safe5):,.0f}")
print(f"  Max safe position (<10% of candle):")
print(f"    Median: ${np.median(safe10):,.0f}  P10: ${np.percentile(safe10, 10):,.0f}  "
      f"P25: ${np.percentile(safe10, 25):,.0f}  Min: ${min(safe10):,.0f}")

# Count how many trades exceed safe threshold at various account sizes
print(f"\n  Trades that WOULD exceed 10% of candle volume at account size:")
for acct_size in [25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]:
    exceeded = sum(1 for t in all_trades if acct_size > t["entry_candle_dollar_vol"] * 0.10)
    print(f"    ${acct_size/1000:.0f}K account: {exceeded}/{len(all_trades)} trades ({exceeded/len(all_trades)*100:.0f}%) exceed 10%")

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}")
