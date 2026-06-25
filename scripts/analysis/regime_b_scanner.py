"""Path B: Pre-market scanner regime detection.

Per-day features from existing pre-market scanner data (no new data needed):
  - n_candidates: how many tickers passed the scanner
  - max_gap_pct: largest gap
  - median_gap_pct: typical gap among watchlist
  - total_pm_dvol: combined pre-market dollar volume across watchlist
  - n_high_gappers: count of >= 30% gappers
  - max_pm_dvol: largest PM dollar volume in watchlist

Question: do these PRE-MARKET features predict our strategy's PnL that day?

If yes -> we can build a regime gate that filters days BEFORE market opens
(zero in-day cost).
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '.')

STARTING_CASH = 25_000
ALL_STRATS = ["h", "g", "a", "f", "d", "v", "p", "m", "r", "w", "o", "b", "k", "c", "s", "e", "i", "j", "n", "l", "x"]
DATA_DIRS = ["stored_data_combined", "stored_data", "stored_data_2022", "stored_data_2023",
             "stored_data_mar_may_2026", "stored_data_jun_2026"]

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

with open("config/trial_w13_1202_deploy.json") as f:
    p_dep = json.load(f)["params"]
with open("config/trial_432_params.json") as f:
    baseline = json.load(f)

merged = {**baseline, **p_dep}
for s in ALL_STRATS:
    merged[f"enable_{s}"] = (s in {"g", "l"})
set_strategy_params(merged)

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
dates = sorted([d for d in all_dates if "2022-01-01" <= d <= "2026-12-31"])
print(f"Analyzing {len(dates)} days...")

tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0
tgc.NEWS_MODULATOR_ENABLED = False

# Combine: per-day scanner features + per-day strategy PnL (independent $25K)
daily = []
for d in dates:
    picks = picks_by_date.get(d, [])
    if not picks:
        continue

    # Extract scanner features (computed from picks regardless of strategy)
    gaps = [p.get("gap_pct", 0) for p in picks if p.get("gap_pct") is not None]
    pm_dvols = [p.get("pm_dvol", p.get("pm_volume", 0) * p.get("prev_close", 0))
                for p in picks if p.get("pm_dvol") is not None or p.get("pm_volume") is not None]
    pm_dvols = [v for v in pm_dvols if v and v > 0]

    feat = {
        "date": d,
        "n_candidates": len(picks),
        "max_gap_pct": max(gaps) if gaps else 0,
        "median_gap_pct": float(np.median(gaps)) if gaps else 0,
        "mean_gap_pct": float(np.mean(gaps)) if gaps else 0,
        "n_high_gappers": sum(1 for g in gaps if g >= 30),
        "n_very_high_gappers": sum(1 for g in gaps if g >= 50),
        "total_pm_dvol": float(sum(pm_dvols)) if pm_dvols else 0,
        "max_pm_dvol": float(max(pm_dvols)) if pm_dvols else 0,
        "median_pm_dvol": float(np.median(pm_dvols)) if pm_dvols else 0,
    }

    # Strategy PnL for the day (independent $25K)
    try:
        states, end_cash, unset, _ = tgc.simulate_day_combined(picks, STARTING_CASH, cash_account=True)
    except Exception:
        feat["day_pnl"] = 0; feat["n_trades"] = 0
        daily.append(feat); continue
    day_trades = [s for s in states if s.get("exit_reason") is not None and s.get("position_cost", 0) > 0]
    feat["day_pnl"] = sum(t["pnl"] for t in day_trades)
    feat["n_trades"] = len(day_trades)
    feat["day_wins"] = sum(1 for t in day_trades if t["pnl"] > 0)
    daily.append(feat)

with_trades = [r for r in daily if r["n_trades"] > 0]
print(f"\nDays processed: {len(daily)}, with at least 1 trade: {len(with_trades)}")

# Correlations: each pre-market feature -> day PnL (only days with trades, to isolate effect)
features = ["n_candidates", "max_gap_pct", "median_gap_pct", "mean_gap_pct",
            "n_high_gappers", "n_very_high_gappers",
            "total_pm_dvol", "max_pm_dvol", "median_pm_dvol"]
print(f"\n=== Pearson(pre-market feature, same-day PnL) — n={len(with_trades)} ===")
print(f"{'Feature':<22s} {'Pearson':>9s}")
for f in features:
    x = np.array([r[f] for r in with_trades])
    y = np.array([r["day_pnl"] for r in with_trades])
    if x.std() == 0: continue
    p = float(np.corrcoef(x, y)[0, 1])
    print(f"{f:<22s} {p:>+8.3f}")

# Now bucket days by max_gap_pct (most likely "regime" indicator)
print(f"\n=== Per-day PnL bucketed by MAX gap of pre-market watchlist ===")
buckets = [
    ("No big gappers (max<20%)",  lambda r: r["max_gap_pct"] < 20),
    ("Some big (20-40%)",         lambda r: 20 <= r["max_gap_pct"] < 40),
    ("Many big (40-70%)",         lambda r: 40 <= r["max_gap_pct"] < 70),
    ("Squeeze regime (>=70%)",    lambda r: r["max_gap_pct"] >= 70),
]
print(f"{'Bucket':<35s} {'n_days':>7s} {'avg_pnl':>10s} {'med_pnl':>10s} {'avg_trades':>11s}")
for label, pred in buckets:
    days_in = [r for r in daily if pred(r)]
    if not days_in:
        print(f"{label:<35s}  (no days)"); continue
    avg_pnl = np.mean([r["day_pnl"] for r in days_in])
    med_pnl = np.median([r["day_pnl"] for r in days_in])
    avg_tr = np.mean([r["n_trades"] for r in days_in])
    print(f"{label:<35s} {len(days_in):>7d} ${avg_pnl:>+8,.0f}  ${med_pnl:>+8,.0f}  {avg_tr:>10.1f}")

# Most important: bucket by n_high_gappers (count of 30%+ gappers in watchlist)
print(f"\n=== Per-day PnL bucketed by #(30%+ gappers in watchlist) ===")
buckets2 = [
    ("0 high gappers", lambda r: r["n_high_gappers"] == 0),
    ("1 high gapper",  lambda r: r["n_high_gappers"] == 1),
    ("2-3 high",       lambda r: 2 <= r["n_high_gappers"] <= 3),
    ("4-7 high",       lambda r: 4 <= r["n_high_gappers"] <= 7),
    ("8+ high",        lambda r: r["n_high_gappers"] >= 8),
]
print(f"{'Bucket':<20s} {'n_days':>7s} {'avg_pnl':>10s} {'med_pnl':>10s} {'avg_trades':>11s}")
for label, pred in buckets2:
    days_in = [r for r in daily if pred(r)]
    if not days_in:
        print(f"{label:<20s}  (no days)"); continue
    avg_pnl = np.mean([r["day_pnl"] for r in days_in])
    med_pnl = np.median([r["day_pnl"] for r in days_in])
    avg_tr = np.mean([r["n_trades"] for r in days_in])
    print(f"{label:<20s} {len(days_in):>7d} ${avg_pnl:>+8,.0f}  ${med_pnl:>+8,.0f}  {avg_tr:>10.1f}")

# Filter test: skip days with insufficient high-gappers
print(f"\n=== Filter test on daily independent PnL (per-day fresh $25K) ===")
total_all = sum(r["day_pnl"] for r in daily)
print(f"Trade all days:                            ${total_all:>+12,.0f}  (n={len(daily)})")
for thresh in [1, 2, 3, 4, 5]:
    f_pnl = sum(r["day_pnl"] for r in daily if r["n_high_gappers"] >= thresh)
    f_n = sum(1 for r in daily if r["n_high_gappers"] >= thresh)
    print(f"Trade only if n_high_gappers >= {thresh}:        ${f_pnl:>+12,.0f}  (n={f_n})")

# Best filter combo
print(f"\n=== Hybrid filter tests ===")
for thresh_n, thresh_max in [(2, 20), (3, 30), (4, 40), (2, 50)]:
    f_pnl = sum(r["day_pnl"] for r in daily
                if r["n_high_gappers"] >= thresh_n and r["max_gap_pct"] >= thresh_max)
    f_n = sum(1 for r in daily if r["n_high_gappers"] >= thresh_n and r["max_gap_pct"] >= thresh_max)
    print(f"n_high>={thresh_n} AND max_gap>={thresh_max}: ${f_pnl:>+12,.0f}  (n={f_n})")

# Save daily data
with open("results/regime_b_daily.json", "w") as f:
    json.dump([{k: v if not isinstance(v, (np.integer, np.floating)) else float(v) for k, v in r.items()} for r in daily], f, indent=2, default=str)
print(f"\nSaved: results/regime_b_daily.json")
