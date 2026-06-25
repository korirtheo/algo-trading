"""Patient G: G's eligibility, but waits up to 1 HOUR for candle 1's high
to be broken — catches "builders" (consolidation + late breakout) that
G's strict end-of-candle-2 timing misses.

Setup spec:
  Eligibility:   gap >= 20%, candle 1 green
  Entry trigger: first candle within first 30 bars (60 min) where High > candle 1 High
  Stop:          below entry by 8% OR below candle 1 low (whichever is closer to entry)
  Target:        +20%
  Trail:         5% trail after +5% activation
  Time stop:     30 min from entry
"""
import json
import os
import sys
import pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')

STARTING_CASH = 25_000
DATA_DIRS = ["stored_data_combined", "stored_data", "stored_data_2022", "stored_data_2023",
             "stored_data_mar_may_2026", "stored_data_jun_2026"]

# Load picks
all_picks = {}
for d in DATA_DIRS:
    p = os.path.join(d, "fulltest_picks_gap2_vol250k.pkl")
    if not os.path.exists(p): continue
    with open(p, "rb") as f:
        for date, picks in pickle.load(f).items():
            if date not in all_picks: all_picks[date] = picks

dates = sorted(all_picks.keys())
print(f"Total dates: {len(dates)}")

# Params
PG_MIN_GAP = 20.0
PG_MAX_ENTRY_CANDLE = 30   # 60 min (2-min bars)
PG_STOP_PCT = 8.0          # hard stop
PG_TARGET_PCT = 20.0
PG_TIME_LIMIT_MIN = 30
PG_TRAIL_PCT = 5.0
PG_TRAIL_ACTIVATE_PCT = 5.0


def simulate_pg_day(picks, cash):
    """Patient G simulation. One trade per day, ticker-priority by gap_pct desc."""
    # Eligible picks
    eligible = []
    for p in picks:
        if p.get("gap_pct", 0) < PG_MIN_GAP: continue
        mh = p.get("market_hour_candles")
        if mh is None or len(mh) < 3: continue
        # Candle 1 must be green
        c1 = mh.iloc[0]
        if c1["Close"] <= c1["Open"]: continue
        eligible.append(p)

    # Sort by gap descending
    eligible.sort(key=lambda p: -p["gap_pct"])

    for p in eligible:
        ticker = p["ticker"]
        mh = p["market_hour_candles"]
        c1_high = float(mh.iloc[0]["High"])
        c1_low = float(mh.iloc[0]["Low"])

        # Find first candle (after candle 1) where High > c1_high, within max_entry
        entry_idx = None
        for i in range(1, min(len(mh), PG_MAX_ENTRY_CANDLE + 1)):
            if mh.iloc[i]["High"] > c1_high:
                entry_idx = i
                break
        if entry_idx is None:
            continue

        # Enter at the breakout candle's close (conservative — could also enter at c1_high+0.01)
        entry_price = float(mh.iloc[entry_idx]["Close"])
        # Apply ~50bp slippage on entry (microcap reality)
        entry_price *= 1.005
        position_cost = cash
        shares = position_cost / entry_price

        # Stop = max(entry × (1 - PG_STOP_PCT/100), c1_low * 1.001)  — use whichever is tighter
        stop_price_pct = entry_price * (1 - PG_STOP_PCT / 100)
        stop_price = max(stop_price_pct, c1_low * 0.995)  # below c1_low or pct, tighter wins
        # Actually for "patient" entries we want a LOOSER stop. Use the pct only.
        stop_price = entry_price * (1 - PG_STOP_PCT / 100)
        target_price = entry_price * (1 + PG_TARGET_PCT / 100)
        trail_activated = False
        trail_high = entry_price

        max_hold_bars = PG_TIME_LIMIT_MIN // 2
        exit_price = None
        exit_reason = None
        for j in range(entry_idx + 1, min(len(mh), entry_idx + 1 + max_hold_bars)):
            bar = mh.iloc[j]
            # Stop hit
            if bar["Low"] <= stop_price:
                exit_price = stop_price * 0.995  # ~50bp slip on stop
                exit_reason = "STOP"; break
            # Target hit
            if bar["High"] >= target_price:
                exit_price = target_price * 0.998  # small slip
                exit_reason = "TARGET"; break
            # Trail logic
            if bar["High"] > trail_high:
                trail_high = float(bar["High"])
            if not trail_activated and (trail_high - entry_price) / entry_price * 100 >= PG_TRAIL_ACTIVATE_PCT:
                trail_activated = True
            if trail_activated:
                trail_stop = trail_high * (1 - PG_TRAIL_PCT / 100)
                if bar["Low"] <= trail_stop:
                    exit_price = trail_stop * 0.995
                    exit_reason = "TRAIL"; break

        if exit_reason is None:
            # Time stop
            j_end = min(len(mh) - 1, entry_idx + max_hold_bars)
            exit_price = float(mh.iloc[j_end]["Close"]) * 0.995
            exit_reason = "TIME"

        pnl = (exit_price - entry_price) * shares
        return [{
            "ticker": ticker, "entry_idx": entry_idx, "entry_price": entry_price,
            "exit_price": exit_price, "exit_reason": exit_reason,
            "pnl": pnl, "ret_pct": (exit_price/entry_price - 1) * 100,
            "gap_pct": p["gap_pct"],
        }]
    return []


# Run on all days
daily_results = {}
for d in dates:
    picks = all_picks.get(d, [])
    if not picks: continue
    trades = simulate_pg_day(picks, STARTING_CASH)
    if trades: daily_results[d] = trades

all_trades = [t for d in dates for t in daily_results.get(d, [])]
n = len(all_trades)
wins = [t for t in all_trades if t["pnl"] > 0]
total_pnl = sum(t["pnl"] for t in all_trades)
print(f"\n=== Patient G overall ===")
print(f"  Days fired:    {len(daily_results)}")
print(f"  Total trades:  {n}")
print(f"  WR:            {len(wins)/n*100 if n else 0:.1f}%")
print(f"  Avg ret/trade: {np.mean([t['ret_pct'] for t in all_trades]):.2f}%" if all_trades else "")
print(f"  Total $:       ${total_pnl:,.0f}")
if all_trades:
    print(f"  Avg per trade: ${total_pnl/n:+,.0f}")
    by_exit = defaultdict(list)
    for t in all_trades: by_exit[t["exit_reason"]].append(t)
    print(f"  Exit breakdown:")
    for reason, ts in by_exit.items():
        wn = sum(1 for t in ts if t["pnl"] > 0)
        print(f"    {reason}: n={len(ts):>3} WR={wn/len(ts)*100:.0f}% avg=${sum(t['pnl'] for t in ts)/len(ts):+,.0f}")

# Multi-year breakdown
print(f"\n=== Multi-year ===")
by_year = defaultdict(list)
for d, ts in daily_results.items():
    for t in ts: by_year[d[:4]].append(t)
for year in sorted(by_year.keys()):
    ts = by_year[year]
    wins_y = sum(1 for t in ts if t["pnl"] > 0)
    total = sum(t["pnl"] for t in ts)
    print(f"  {year}: n={len(ts):>3} WR={wins_y/len(ts)*100:>4.1f}% total=${total:>+10,.0f}  avg=${total/len(ts):>+8,.0f}")

# Orthogonality with G+L
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
with open("config/trial_w13_1202_deploy.json") as f: p_dep = json.load(f)["params"]
with open("config/trial_432_params.json") as f: baseline = json.load(f)
merged = {**baseline, **p_dep}
for s in ALL_STRATS: merged[f"enable_{s}"] = (s in {"g","l"})
set_strategy_params(merged)
tgc.USE_DYNAMIC_SLIPPAGE = True; tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True; tgc.SLIP_IMPACT_K = 3.0; tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15; tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0; tgc.NEWS_MODULATOR_ENABLED = False

g_fires_days = set(); g_silent_days = set()
for d in dates:
    picks = all_picks[d]
    if not picks: continue
    try:
        states, _, _, _ = tgc.simulate_day_combined(picks, 25000, cash_account=True)
        has_g = any(s.get("exit_reason") is not None and s.get("position_cost", 0) > 0 for s in states)
    except Exception:
        has_g = False
    (g_fires_days if has_g else g_silent_days).add(d)

incremental = [t for d, ts in daily_results.items() for t in ts if d in g_silent_days]
overlap = [t for d, ts in daily_results.items() for t in ts if d in g_fires_days]
print(f"\n=== Orthogonality (G+L vs Patient G) ===")
print(f"Patient G fires on G-silent days (INCREMENTAL): {len(incremental)} trades, "
      f"total=${sum(t['pnl'] for t in incremental):,.0f}, "
      f"WR={sum(1 for t in incremental if t['pnl'] > 0)/len(incremental)*100 if incremental else 0:.1f}%")
print(f"Patient G fires on G-fires days (overlap):     {len(overlap)} trades, "
      f"total=${sum(t['pnl'] for t in overlap):,.0f}, "
      f"WR={sum(1 for t in overlap if t['pnl'] > 0)/len(overlap)*100 if overlap else 0:.1f}%")

# Mar-Jun 2026 blind OOS only
mar_jun = [t for d, ts in daily_results.items() if "2026-03-01" <= d <= "2026-12-31" for t in ts]
print(f"\n=== Mar-Jun 2026 blind OOS ===")
print(f"Patient G:   n={len(mar_jun)} WR={sum(1 for t in mar_jun if t['pnl']>0)/len(mar_jun)*100 if mar_jun else 0:.1f}% "
      f"total=${sum(t['pnl'] for t in mar_jun):,.0f}")
print(f"  (W13 #1202 Mar-Jun reference: $229K, 78% WR — but using compounded equity)")

with open("results/patient_g_trades.json", "w") as f:
    json.dump([{"date": d, **t} for d in dates for t in daily_results.get(d, [])], f, indent=2, default=str)
print(f"\nSaved: results/patient_g_trades.json")
