"""Path A: Strategy-self-regime detection.

Replay W13 #1202 on 2022-2026 with FRESH $25K each day (no compounding).
For each day collect: n_trades, n_wins, day_pnl, day_return_pct.
Compute rolling 20d WR (lookback, no leakage) — does it predict next-day PnL?

If yes -> regime is persistent and detectable from strategy's own performance
If no  -> regime is too random for self-referential detection (need external signals)
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

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
print(f"Replaying W13 #1202 on {len(dates)} days (2022-2026)...")

tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0
tgc.NEWS_MODULATOR_ENABLED = False

# Per-day independent simulation (fresh $25K, no compounding)
daily = []
for d in dates:
    dp = picks_by_date.get(d, [])
    if not dp:
        continue
    try:
        states, end_cash, unset, _ = tgc.simulate_day_combined(dp, STARTING_CASH, cash_account=True)
    except Exception:
        continue
    day_trades = [s for s in states if s.get("exit_reason") is not None and s.get("position_cost", 0) > 0]
    wins = [t for t in day_trades if t["pnl"] > 0]
    losers = [t for t in day_trades if t["pnl"] <= 0]
    n_total = len(day_trades)
    day_pnl = sum(t["pnl"] for t in day_trades)
    day_pnl_pct = (day_pnl / STARTING_CASH) * 100 if STARTING_CASH else 0
    daily.append({
        "date": d, "n_trades": n_total, "n_wins": len(wins), "n_losers": len(losers),
        "day_pnl": day_pnl, "day_pnl_pct": day_pnl_pct,
        "day_wr": len(wins) / n_total * 100 if n_total > 0 else None,
        "avg_winner": np.mean([t["pnl"] for t in wins]) if wins else 0,
        "avg_loser": np.mean([t["pnl"] for t in losers]) if losers else 0,
    })

print(f"\nDays with at least 1 trade: {sum(1 for r in daily if r['n_trades'] > 0)} / {len(daily)}")
print(f"Days with 0 trades:         {sum(1 for r in daily if r['n_trades'] == 0)} / {len(daily)}")

# Compute rolling 20-day metrics (lookback, current day NOT included)
ROLL = 20
for i, row in enumerate(daily):
    window = daily[max(0, i - ROLL):i]  # PRIOR 20 days, no leakage
    total_trades = sum(r["n_trades"] for r in window)
    total_wins = sum(r["n_wins"] for r in window)
    total_pnl = sum(r["day_pnl"] for r in window)
    row["roll20_wr"] = total_wins / total_trades * 100 if total_trades > 0 else None
    row["roll20_pnl"] = total_pnl
    row["roll20_n_trades"] = total_trades

# Filter: only days WITH trades and where rolling lookback has enough trades
analyzable = [r for r in daily if r["n_trades"] > 0 and r["roll20_n_trades"] >= 20]
print(f"Analyzable days (have trades + 20+ prior trades): {len(analyzable)}")

# Bucket by rolling WR
buckets = [
    ("Cold (< 50%)",       lambda r: r["roll20_wr"] < 50),
    ("Mid  (50-60%)",      lambda r: 50 <= r["roll20_wr"] < 60),
    ("Warm (60-70%)",      lambda r: 60 <= r["roll20_wr"] < 70),
    ("Hot  (>= 70%)",      lambda r: r["roll20_wr"] >= 70),
]
print(f"\n=== Day-by-day PnL by rolling-20d WR bucket ===")
print(f"{'Bucket':<20s} {'n_days':>7s} {'avg_pnl':>10s} {'med_pnl':>10s} {'day_wr':>7s} {'win_$':>9s} {'lose_$':>9s}")
for label, pred in buckets:
    days_in = [r for r in analyzable if pred(r)]
    if not days_in:
        print(f"{label:<20s}  (no days)")
        continue
    avg_pnl = np.mean([r["day_pnl"] for r in days_in])
    med_pnl = np.median([r["day_pnl"] for r in days_in])
    overall_wins = sum(r["n_wins"] for r in days_in)
    overall_total = sum(r["n_trades"] for r in days_in)
    overall_wr = overall_wins / overall_total * 100 if overall_total > 0 else 0
    avg_win = np.mean([r["avg_winner"] for r in days_in if r["avg_winner"] > 0])
    avg_lose = np.mean([r["avg_loser"] for r in days_in if r["avg_loser"] < 0])
    print(f"{label:<20s} {len(days_in):>7d} ${avg_pnl:>+8,.0f}  ${med_pnl:>+8,.0f}  {overall_wr:>5.1f}%  ${avg_win:>+7,.0f}  ${avg_lose:>+7,.0f}")

# Correlation: rolling20_wr -> next-day pnl
roll_wr = np.array([r["roll20_wr"] for r in analyzable])
next_pnl = np.array([r["day_pnl"] for r in analyzable])
pearson = float(np.corrcoef(roll_wr, next_pnl)[0, 1])
print(f"\nPearson(roll20_wr, same_day_pnl): {pearson:+.3f}")

# Also test: rolling20_pnl -> next-day pnl
roll_pnl = np.array([r["roll20_pnl"] for r in analyzable])
pearson_pnl = float(np.corrcoef(roll_pnl, next_pnl)[0, 1])
print(f"Pearson(roll20_pnl, same_day_pnl): {pearson_pnl:+.3f}")

# Filter test: skip trading on cold days (< 50% rolling WR)
total_pnl_all = sum(r["day_pnl"] for r in analyzable)
total_pnl_no_cold = sum(r["day_pnl"] for r in analyzable if r["roll20_wr"] >= 50)
total_pnl_no_mid = sum(r["day_pnl"] for r in analyzable if r["roll20_wr"] >= 60)
total_pnl_only_hot = sum(r["day_pnl"] for r in analyzable if r["roll20_wr"] >= 70)
print(f"\n=== Filter test (sum of independent-day PnL across {len(analyzable)} days) ===")
print(f"All days traded:                 ${total_pnl_all:>+12,.0f}  (n={len(analyzable)})")
print(f"Skip Cold (< 50% rolling WR):    ${total_pnl_no_cold:>+12,.0f}  (n={sum(1 for r in analyzable if r['roll20_wr'] >= 50)})")
print(f"Trade only Warm+Hot (>= 60%):    ${total_pnl_no_mid:>+12,.0f}  (n={sum(1 for r in analyzable if r['roll20_wr'] >= 60)})")
print(f"Trade only Hot (>= 70%):         ${total_pnl_only_hot:>+12,.0f}  (n={sum(1 for r in analyzable if r['roll20_wr'] >= 70)})")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
dates_arr = [datetime.strptime(r["date"], "%Y-%m-%d") for r in analyzable]
ax1 = axes[0]
ax1.plot(dates_arr, [r["roll20_wr"] for r in analyzable], color='steelblue', linewidth=1.5, label='Roll-20d WR')
ax1.axhline(50, color='red', linestyle='--', alpha=0.4, label='50% threshold')
ax1.axhline(60, color='orange', linestyle='--', alpha=0.4)
ax1.axhline(70, color='green', linestyle='--', alpha=0.4)
ax1.set_ylabel('Rolling 20-day WR (%)')
ax1.set_title('W13 #1202 — Rolling Strategy WR over Time')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
colors = ['red' if p < 0 else 'green' for p in [r["day_pnl"] for r in analyzable]]
ax2.bar(dates_arr, [r["day_pnl"] for r in analyzable], color=colors, alpha=0.7, width=1.0)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('Per-day PnL ($)')
ax2.set_xlabel('Date')
ax2.set_title(f'Per-day PnL (each day independent $25K start) — Pearson(roll_wr, pnl) = {pearson:+.3f}')
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "results/regime_a_self_strategy.png"
fig.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nChart saved: {out_path}")

# Save raw daily data for later analysis
import json as _json
with open("results/regime_a_daily.json", "w") as f:
    _json.dump([{k: v if not isinstance(v, (np.integer, np.floating)) else float(v) for k, v in r.items()} for r in daily], f, indent=2, default=str)
print(f"Saved daily data: results/regime_a_daily.json")
