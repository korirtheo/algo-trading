"""
Fill Reality Check v2: Focus on vol-capped trades.
The vol cap limits us to 5% of realized dollar volume.
Question: Is 5% still too much for these small-cap gap stocks?
"""
import sys
import numpy as np
from zoneinfo import ZoneInfo

from test_full import (
    load_all_picks, SLIPPAGE_PCT, STARTING_CASH, MARGIN_THRESHOLD, VOL_CAP_PCT, ET_TZ,
)
import test_green_candle_combined as tgcc

DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2025-01-01", "2026-02-28")

print("Loading data...")
all_dates, daily_picks = load_all_picks(DATA_DIRS)
all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
print(f"  {len(all_dates)} trading days\n")

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
            exit_price = st.get("exit_price") or 0
            if position_cost <= 0 or exit_price <= 0 or len(mh) < 2:
                continue
            shares_needed = position_cost / exit_price

            # Volume up to and including entry candle
            entry_candle_vol = float(mh.iloc[1]["Volume"])
            cumulative_vol_at_entry = float(mh.iloc[:2]["Volume"].sum())
            cumulative_dollar_at_entry = cumulative_vol_at_entry * exit_price

            # What the vol cap allowed: 5% of cumulative dollar vol
            vol_cap_limit = cumulative_dollar_at_entry * (VOL_CAP_PCT / 100)

            # Shares in the entry candle
            pct_of_entry_candle_shares = (shares_needed / entry_candle_vol * 100) if entry_candle_vol > 0 else 999

            # First 10 min
            first_10min_vol = float(mh.iloc[:5]["Volume"].sum()) if len(mh) >= 5 else float(mh["Volume"].sum())
            pct_of_10min = (shares_needed / first_10min_vol * 100) if first_10min_vol > 0 else 999

            all_trades.append({
                "date": d,
                "ticker": st["ticker"],
                "strategy": st["strategy"],
                "position_cost": position_cost,
                "cash_available": cash,
                "vol_capped": st.get("vol_capped", False),
                "vol_cap_limit": vol_cap_limit,
                "shares_needed": shares_needed,
                "entry_candle_vol": entry_candle_vol,
                "entry_candle_dollar_vol": entry_candle_vol * exit_price,
                "pct_of_entry_candle": pct_of_entry_candle_shares,
                "pct_of_10min": pct_of_10min,
                "exit_price": exit_price,
                "pnl": st["pnl"],
                "pnl_pct": (st["pnl"] / position_cost * 100) if position_cost > 0 else 0,
            })

    cash = new_cash
    unsettled = new_unsettled

print(f"Analyzed {len(all_trades)} trades\n")

# Split by vol-capped vs not
vc_trades = [t for t in all_trades if t["vol_capped"]]
no_vc = [t for t in all_trades if not t["vol_capped"]]

print("=" * 70)
print("  HOW MUCH ARE WE ACTUALLY BUYING? (with 5% vol cap)")
print("=" * 70)

print(f"\n  Vol-capped trades: {len(vc_trades)}/{len(all_trades)} ({len(vc_trades)/len(all_trades)*100:.0f}%)")
print(f"  Non-capped trades: {len(no_vc)}/{len(all_trades)}")

# For vol-capped trades: position = 5% of cumulative volume
# So as % of entry candle alone, it'll be higher
print(f"\n  --- VOL-CAPPED TRADES (position = 5% of cumul. vol) ---")
if vc_trades:
    pcts = [t["pct_of_entry_candle"] for t in vc_trades]
    costs = [t["position_cost"] for t in vc_trades]
    limits = [t["vol_cap_limit"] for t in vc_trades]
    candle_dvols = [t["entry_candle_dollar_vol"] for t in vc_trades]
    print(f"  Position size:     median ${np.median(costs):,.0f}  mean ${np.mean(costs):,.0f}")
    print(f"  Vol cap limit:     median ${np.median(limits):,.0f}  mean ${np.mean(limits):,.0f}")
    print(f"  Entry candle $vol: median ${np.median(candle_dvols):,.0f}  mean ${np.mean(candle_dvols):,.0f}")
    print(f"  Position as % of entry candle:")
    print(f"    Median: {np.median(pcts):.1f}%  Mean: {np.mean(pcts):.1f}%  "
          f"P90: {np.percentile(pcts, 90):.1f}%  Max: {max(pcts):.1f}%")
    pcts10 = [t["pct_of_10min"] for t in vc_trades]
    print(f"  Position as % of first 10min:")
    print(f"    Median: {np.median(pcts10):.1f}%  Mean: {np.mean(pcts10):.1f}%  "
          f"P90: {np.percentile(pcts10, 90):.1f}%")

print(f"\n  --- NON-CAPPED TRADES (position = full cash, under vol cap) ---")
if no_vc:
    pcts = [t["pct_of_entry_candle"] for t in no_vc]
    costs = [t["position_cost"] for t in no_vc]
    candle_dvols = [t["entry_candle_dollar_vol"] for t in no_vc]
    print(f"  Position size:     median ${np.median(costs):,.0f}  mean ${np.mean(costs):,.0f}")
    print(f"  Entry candle $vol: median ${np.median(candle_dvols):,.0f}  mean ${np.mean(candle_dvols):,.0f}")
    print(f"  Position as % of entry candle:")
    print(f"    Median: {np.median(pcts):.1f}%  Mean: {np.mean(pcts):.1f}%  "
          f"P90: {np.percentile(pcts, 90):.1f}%  Max: {max(pcts):.1f}%")

# --- KEY QUESTION: How much slippage would this cause? ---
print(f"\n{'='*70}")
print("  ESTIMATED REAL-WORLD SLIPPAGE")
print(f"{'='*70}")
print("  Rule of thumb for small-cap stocks:")
print("    <1% of candle vol = invisible (~0.05% slip)")
print("    1-5% = minor impact (~0.1-0.3% slip)")
print("    5-15% = noticeable (~0.3-1% slip)")
print("    15-30% = significant (~1-3% slip)")
print("    >30% = you ARE the market (3%+ slip)")

buckets = [
    ("<1%", 0, 1),
    ("1-5%", 1, 5),
    ("5-15%", 5, 15),
    ("15-30%", 15, 30),
    (">30%", 30, 9999),
]
slip_estimates = [0.05, 0.2, 0.65, 2.0, 5.0]

print(f"\n  {'Bucket':>8} {'Trades':>7} {'%':>5} {'Est Slip':>9} {'Avg PnL%':>9} {'After Slip':>11}")
print(f"  {'-'*55}")

total_slip_cost = 0
for (label, lo, hi), est_slip in zip(buckets, slip_estimates):
    bucket_trades = [t for t in all_trades if lo <= t["pct_of_entry_candle"] < hi]
    if not bucket_trades:
        continue
    n = len(bucket_trades)
    avg_pnl = np.mean([t["pnl_pct"] for t in bucket_trades])
    # Slip on entry + exit = 2x
    real_slip = est_slip * 2
    after_slip = avg_pnl - real_slip
    slip_dollar = sum(t["position_cost"] * real_slip / 100 for t in bucket_trades)
    total_slip_cost += slip_dollar
    print(f"  {label:>8} {n:>7} {n/len(all_trades)*100:>4.0f}% {est_slip:>8.2f}% "
          f"{avg_pnl:>+8.2f}% {after_slip:>+10.2f}%")

print(f"\n  Total estimated slippage cost: ${total_slip_cost:,.0f}")
print(f"  Current simulated slippage:    ${sum(t['position_cost'] * 0.05/100 * 2 for t in all_trades):,.0f}")
print(f"  Additional real-world cost:    ${total_slip_cost - sum(t['position_cost'] * 0.05/100 * 2 for t in all_trades):,.0f}")

# --- VOL CAP COMPARISON: 3% vs 5% vs 7% ---
print(f"\n{'='*70}")
print("  WHAT IF WE LOWER VOL CAP TO REDUCE IMPACT?")
print(f"{'='*70}")
print("  Lowering vol cap = smaller orders = less market impact")
print("  But also = less capital deployed per trade\n")

# Re-run with different vol caps and show impact metrics
import test_full
for vc in [1, 2, 3, 5, 7]:
    test_full.VOL_CAP_PCT = float(vc)
    tgcc.VOL_CAP_PCT = float(vc)

    cash2 = STARTING_CASH
    uns2 = 0.0
    trades2 = []
    for d in all_dates:
        if uns2 > 0:
            cash2 += uns2
            uns2 = 0.0
        picks = daily_picks.get(d, [])
        ca = cash2 < MARGIN_THRESHOLD
        states2, cash2, uns2 = tgcc.simulate_day_combined(picks, cash2, ca)
        for st in states2:
            if st["exit_reason"] is not None and st["position_cost"] > 0:
                ep = st.get("exit_price") or 0
                if ep <= 0 or len(st["mh"]) < 2:
                    continue
                sn = st["position_cost"] / ep
                ecv = float(st["mh"].iloc[1]["Volume"])
                pct = (sn / ecv * 100) if ecv > 0 else 999
                trades2.append({
                    "pct_candle": pct,
                    "position_cost": st["position_cost"],
                    "vc": st.get("vol_capped", False),
                })

    final2 = cash2 + uns2
    vc_n = sum(1 for t in trades2 if t["vc"])
    pcts2 = [t["pct_candle"] for t in trades2]
    over5 = sum(1 for p in pcts2 if p > 5)
    over15 = sum(1 for p in pcts2 if p > 15)
    med_pct = np.median(pcts2) if pcts2 else 0
    p90_pct = np.percentile(pcts2, 90) if pcts2 else 0
    print(f"  VolCap {vc}%: ${final2:>13,.0f}  VC:{vc_n:>3}  "
          f"Med%candle: {med_pct:>5.1f}%  P90: {p90_pct:>5.1f}%  "
          f">5%: {over5:>3}  >15%: {over15:>3}")

# Restore
test_full.VOL_CAP_PCT = 5.0
tgcc.VOL_CAP_PCT = 5.0

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}")
