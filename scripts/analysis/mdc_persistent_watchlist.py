"""MDC v2: Persistent Watchlist Member.

If a ticker is on today's watchlist AND was also on watchlist within last 3 days,
it's showing PERSISTENT interest (keeps gapping). Trade it.

This is MUCH broader than v1 (which required 25%+ Day 1 intraday pump).
Captures any sustained multi-day theme.

Setup:
  Eligibility:
    - On today's watchlist
    - Same ticker appeared on watchlist at least once in last 3 trading days
  Entry:
    - At end of 1st 2-min bar (9:32 ET)
  Exit:
    - Stop: -10%
    - Target: +15%
    - Trail: 3% after +5% activation
    - Time: 30 min
"""
import json
import os
import sys
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

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

LOOKBACK_DAYS = 3
STOP_PCT = 10.0
TARGET_PCT = 15.0
TRAIL_PCT = 3.0
TRAIL_ACTIVATE_PCT = 5.0
TIME_LIMIT_MIN = 30


def get_recent_watchlist_tickers(d_idx, lookback):
    """For each of the last `lookback` trading days BEFORE d_idx, gather all watchlist tickers."""
    tickers = set()
    for i in range(max(0, d_idx - lookback), d_idx):
        for p in all_picks.get(dates[i], []):
            tickers.add(p["ticker"])
    return tickers


def simulate_day(pick):
    """Simulate one position for a persistent-watchlist ticker."""
    mh = pick.get("market_hour_candles")
    if mh is None or len(mh) < 5:
        return None
    # Entry at end of first 2-min bar
    entry_price = float(mh.iloc[0]["Close"]) * 1.003  # 30bp slip
    shares = STARTING_CASH / entry_price
    stop_price = entry_price * (1 - STOP_PCT / 100)
    target_price = entry_price * (1 + TARGET_PCT / 100)
    trail_activated = False
    trail_high = entry_price
    max_bars = TIME_LIMIT_MIN // 2

    exit_price = None; exit_reason = None
    for j in range(1, min(len(mh), 1 + max_bars)):
        bar = mh.iloc[j]
        if bar["Low"] <= stop_price:
            exit_price = stop_price * 0.998; exit_reason = "STOP"; break
        if bar["High"] >= target_price:
            exit_price = target_price * 0.998; exit_reason = "TARGET"; break
        if bar["High"] > trail_high: trail_high = float(bar["High"])
        if not trail_activated and (trail_high - entry_price) / entry_price * 100 >= TRAIL_ACTIVATE_PCT:
            trail_activated = True
        if trail_activated:
            trail_stop = trail_high * (1 - TRAIL_PCT / 100)
            if bar["Low"] <= trail_stop:
                exit_price = trail_stop * 0.998; exit_reason = "TRAIL"; break
    if exit_reason is None:
        j_end = min(len(mh) - 1, max_bars)
        exit_price = float(mh.iloc[j_end]["Close"]) * 0.998; exit_reason = "TIME"
    pnl = (exit_price - entry_price) * shares
    return {
        "exit_reason": exit_reason, "pnl": pnl,
        "ret_pct": (exit_price / entry_price - 1) * 100,
    }


# Process each day
trades_by_date = defaultdict(list)
for i, d in enumerate(dates):
    if i < LOOKBACK_DAYS: continue
    recent_tickers = get_recent_watchlist_tickers(i, LOOKBACK_DAYS)
    today_picks = all_picks.get(d, [])
    for p in today_picks:
        if p["ticker"] not in recent_tickers: continue  # Only persistent ones
        result = simulate_day(p)
        if result:
            trades_by_date[d].append({
                "ticker": p["ticker"], "gap_pct": p.get("gap_pct"),
                **result,
            })

all_trades = [t for d, ts in trades_by_date.items() for t in ts]
n = len(all_trades)
wins = [t for t in all_trades if t["pnl"] > 0]
total_pnl = sum(t["pnl"] for t in all_trades)
print(f"\n=== MDC v2 (Persistent Watchlist, lookback={LOOKBACK_DAYS}d) ===")
print(f"  Days with trades: {len(trades_by_date)}")
print(f"  Total trades:     {n}")
print(f"  WR:               {len(wins)/n*100 if n else 0:.1f}%")
print(f"  Total $:          ${total_pnl:,.0f}")
if all_trades:
    print(f"  Avg ret/trade: {np.mean([t['ret_pct'] for t in all_trades]):.2f}%")
    print(f"  Avg $/trade:   ${total_pnl/n:+,.0f}")
    print(f"  Best:          {max(t['ret_pct'] for t in all_trades):.1f}%")
    print(f"  Worst:         {min(t['ret_pct'] for t in all_trades):.1f}%")

    by_exit = defaultdict(list)
    for t in all_trades: by_exit[t["exit_reason"]].append(t)
    print(f"\n  Exit breakdown:")
    for reason, ts in by_exit.items():
        w = sum(1 for t in ts if t["pnl"] > 0)
        print(f"    {reason}: n={len(ts):>3} WR={w/len(ts)*100:>4.1f}% total=${sum(t['pnl'] for t in ts):>+9,.0f}")

# Multi-year
print(f"\n=== Multi-year ===")
by_year = defaultdict(list)
for d, ts in trades_by_date.items():
    for t in ts: by_year[d[:4]].append(t)
for year in sorted(by_year.keys()):
    ts = by_year[year]
    if not ts: continue
    w = sum(1 for t in ts if t["pnl"] > 0)
    total = sum(t["pnl"] for t in ts)
    print(f"  {year}: n={len(ts):>4} WR={w/len(ts)*100:>4.1f}% total=${total:>+10,.0f}  avg=${total/len(ts):>+8,.0f}")

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

g_silent = set(); g_fires = set()
for d in dates:
    picks = all_picks[d]
    if not picks: continue
    try:
        states, _, _, _ = tgc.simulate_day_combined(picks, 25000, cash_account=True)
        has_g = any(s.get("exit_reason") is not None and s.get("position_cost", 0) > 0 for s in states)
    except Exception: has_g = False
    (g_fires if has_g else g_silent).add(d)

inc = [t for d, ts in trades_by_date.items() if d in g_silent for t in ts]
ovr = [t for d, ts in trades_by_date.items() if d in g_fires for t in ts]
print(f"\n=== Orthogonality with G+L ===")
print(f"MDC v2 on G-silent days (INCREMENTAL): n={len(inc)} "
      f"WR={sum(1 for t in inc if t['pnl']>0)/len(inc)*100 if inc else 0:.1f}% "
      f"total=${sum(t['pnl'] for t in inc):,.0f}")
print(f"MDC v2 on G-fires days (overlap):     n={len(ovr)} "
      f"WR={sum(1 for t in ovr if t['pnl']>0)/len(ovr)*100 if ovr else 0:.1f}% "
      f"total=${sum(t['pnl'] for t in ovr):,.0f}")

mar_jun = [t for d, ts in trades_by_date.items() if "2026-03-01" <= d <= "2026-12-31" for t in ts]
print(f"\n=== Mar-Jun 2026 blind OOS ===")
print(f"MDC v2: n={len(mar_jun)} "
      f"WR={sum(1 for t in mar_jun if t['pnl']>0)/len(mar_jun)*100 if mar_jun else 0:.1f}% "
      f"total=${sum(t['pnl'] for t in mar_jun):,.0f}")
