"""
Run 2026 OOS forward test on top 100 best + 100 random trials from oglhmafp_v5.
Shows how IS performance generalizes to out-of-sample data.

Usage: python run_v5_oos_forward.py
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psycopg2
import numpy as np
import time

STUDY_ID = 44
DB = "postgresql://postgres@127.0.0.1:5432/optuna_oglhmafp"
DATA_DIRS = ["stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data_jul_2026"]
STARTING_CASH = 25_000

# ── 1. Load trials ─────────────────────────────────────────────────────
print("Loading trials from PostgreSQL...")
conn = psycopg2.connect(DB)
cur = conn.cursor()

# Top 100 by score
cur.execute("""
    SELECT t.trial_id, t.number, v.value as score
    FROM trials t JOIN trial_values v ON t.trial_id = v.trial_id
    WHERE t.study_id = %s AND t.state = 'COMPLETE'
    ORDER BY v.value DESC LIMIT 100
""", (STUDY_ID,))
top100 = cur.fetchall()
print(f"  Top 100: trial #{top100[0][1]} (score=${top100[0][2]:,.0f}) to #{top100[-1][1]} (score=${top100[-1][2]:,.0f})")

# 100 random (excluding top 100)
cur.execute("""
    SELECT t.trial_id, t.number, v.value as score
    FROM trials t JOIN trial_values v ON t.trial_id = v.trial_id
    WHERE t.study_id = %s AND t.state = 'COMPLETE'
""", (STUDY_ID,))
all_trials = cur.fetchall()
all_ids = {t[0] for t in all_trials}
top_ids = {t[0] for t in top100}
remaining = [t for t in all_trials if t[0] not in top_ids]
random.seed(42)
random100 = random.sample(remaining, min(100, len(remaining)))
print(f"  Random 100: from {len(remaining)} non-top trials")

# Load params + user_attrs for all 200 trials
def load_trial(cur, trial_id, number):
    cur.execute("SELECT param_name, param_value FROM trial_params WHERE trial_id = %s", (trial_id,))
    params = {name: val for name, val in cur.fetchall()}
    cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id = %s", (trial_id,))
    for key, val in cur.fetchall():
        if key.startswith("enable_"):
            params[key] = val == "true" if isinstance(val, str) else val
    # Cast whole-number floats to int
    for k, v in params.items():
        if isinstance(v, float) and v == int(v):
            params[k] = int(v)
    # Get IS score
    cur.execute("SELECT value FROM trial_values WHERE trial_id = %s", (trial_id,))
    row = cur.fetchone()
    is_score = row[0] if row else 0
    return {"trial_id": trial_id, "number": number, "is_score": is_score, "params": params}

print("Loading params for 200 trials...")
trials_data = []
for tid, num, score in top100:
    trials_data.append(("top", load_trial(cur, tid, num)))
for tid, num, score in random100:
    trials_data.append(("random", load_trial(cur, tid, num)))
conn.close()

# ── 2. Load OOS data ───────────────────────────────────────────────────
print("Loading OOS data...")
from test_full import load_all_picks, MARGIN_THRESHOLD
all_dates, daily_picks = load_all_picks(DATA_DIRS)
oos_dates = [d for d in all_dates if d >= "2026-03-01"]
print(f"  OOS: {len(oos_dates)} days ({oos_dates[0]} to {oos_dates[-1]})")

# ── 3. Run backtests ──────────────────────────────────────────────────
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params, ALL_STRATS

tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True

results = []
total = len(trials_data)
t0 = time.time()

for idx, (group, trial) in enumerate(trials_data):
    num = trial["number"]
    params = trial["params"]
    is_score = trial["is_score"]

    if idx % 20 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / max(elapsed, 1)
        eta = (total - idx - 1) / max(rate, 0.01) / 60
        print(f"  [{idx+1}/{total}] ({group}) Trial #{num}  IS=${is_score:,.0f}  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta:.1f}min")

    # Apply params
    set_strategy_params(params)
    for s in ALL_STRATS:
        setattr(tgc, f"ENABLE_{s.upper()}", params.get(f"enable_{s}", False))

    # Run backtest
    cash = STARTING_CASH
    unsettled = 0.0
    daily = []
    all_trades = 0
    all_wins = 0
    strat_pnl = {}
    strat_trades = {}

    for d in oos_dates:
        picks = daily_picks.get(d, [])
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(picks, cash, is_live=False)
        except:
            end_c = cash
            unset = 0.0
            states = []
        pnl = end_c - cash
        cash = end_c
        if is_cash:
            cash += unset
        daily.append(pnl)

        for st in states:
            if st["exit_reason"] is not None:
                all_trades += 1
                s = st["strategy"]
                strat_pnl[s] = strat_pnl.get(s, 0) + st["pnl"]
                strat_trades[s] = strat_trades.get(s, 0) + 1
                if st["pnl"] > 0:
                    all_wins += 1

    equity = cash
    total_pnl = equity - STARTING_CASH
    daily_arr = np.array(daily)
    sharpe = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0
    wr = all_wins / max(all_trades, 1) * 100

    # Profit factor
    trade_pnls = []
    for st_list in [states]:
        pass
    gross_wins = sum(v for v in strat_pnl.values() if v > 0)
    gross_losses = abs(sum(v for v in strat_pnl.values() if v <= 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Max drawdown
    peak = STARTING_CASH
    max_dd = 0
    eq = STARTING_CASH
    for p in daily:
        eq += p
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    enabled = sorted([k.replace("enable_", "").upper() for k, v in params.items()
                      if k.startswith("enable_") and v is True])

    results.append({
        "group": group,
        "trial": num,
        "is_score": is_score,
        "oos_pnl": total_pnl,
        "oos_equity": equity,
        "oos_sharpe": sharpe,
        "oos_wr": wr,
        "oos_trades": all_trades,
        "oos_pf": pf,
        "oos_max_dd": max_dd,
        "enabled": enabled,
        "strat_pnl": strat_pnl,
        "strat_trades": strat_trades,
    })

elapsed = time.time() - t0
print(f"\nAll {total} backtests done in {elapsed/60:.1f} min")

# ── 4. Summary ─────────────────────────────────────────────────────────
top_results = [r for r in results if r["group"] == "top"]
rand_results = [r for r in results if r["group"] == "random"]

def stats(label, subset):
    pnls = [r["oos_pnl"] for r in subset]
    sharpes = [r["oos_sharpe"] for r in subset]
    trades = [r["oos_trades"] for r in subset]
    wrs = [r["oos_wr"] for r in subset]
    pfs = [r["oos_pf"] for r in subset if r["oos_pf"] != float("inf")]
    dds = [r["oos_max_dd"] for r in subset]
    winners = sum(1 for p in pnls if p > 0)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Trials:          {len(subset)}")
    print(f"  Winners:         {winners}/{len(subset)} ({100*winners/len(subset):.0f}%)")
    print(f"  OOS PnL:         mean=${np.mean(pnls):>+10,.0f}  median=${np.median(pnls):>+10,.0f}")
    print(f"                   best=${max(pnls):>+10,.0f}  worst=${min(pnls):>+10,.0f}")
    print(f"  OOS Sharpe:      mean={np.mean(sharpes):.2f}  median={np.median(sharpes):.2f}")
    print(f"  OOS WR:          mean={np.mean(wrs):.1f}%  median={np.median(wrs):.1f}%")
    print(f"  OOS PF:          mean={np.mean(pfs):.2f}  median={np.median(pfs):.2f}")
    print(f"  OOS Trades:      mean={np.mean(trades):.0f}  median={np.median(trades):.0f}")
    print(f"  OOS MaxDD:       mean={np.mean(dds):.1f}%  median={np.median(dds):.1f}%")

stats("TOP 100 (by IS score)", top_results)
stats("RANDOM 100", rand_results)

# ── 5. IS vs OOS correlation ──────────────────────────────────────────
top_is = [r["is_score"] for r in top_results]
top_oos = [r["oos_pnl"] for r in top_results]
corr = np.corrcoef(top_is, top_oos)[0, 1] if len(top_is) > 1 else 0

print(f"\n{'='*70}")
print(f"  IS vs OOS CORRELATION (top 100)")
print(f"{'='*70}")
print(f"  Pearson r = {corr:.4f}")
if corr < 0.3:
    print(f"  → Weak correlation — IS score is a poor OOS predictor")
elif corr < 0.6:
    print(f"  → Moderate correlation — some IS→OOS transfer")
else:
    print(f"  → Strong correlation — IS generalizes well")

# ── 6. Strategy combo distribution ────────────────────────────────────
from collections import Counter
print(f"\n{'='*70}")
print(f"  STRATEGY COMBOS (top 100)")
print(f"{'='*70}")
combos = Counter(tuple(r["enabled"]) for r in top_results)
for combo, cnt in combos.most_common(10):
    subset = [r for r in top_results if tuple(r["enabled"]) == combo]
    avg_oos = np.mean([r["oos_pnl"] for r in subset])
    print(f"  {list(combo):<30} {cnt:>3} trials  avg OOS: ${avg_oos:>+10,.0f}")

print(f"\n{'='*70}")
print(f"  STRATEGY COMBOS (random 100)")
print(f"{'='*70}")
combos_r = Counter(tuple(r["enabled"]) for r in rand_results)
for combo, cnt in combos_r.most_common(10):
    subset = [r for r in rand_results if tuple(r["enabled"]) == combo]
    avg_oos = np.mean([r["oos_pnl"] for r in subset])
    print(f"  {list(combo):<30} {cnt:>3} trials  avg OOS: ${avg_oos:>+10,.0f}")

# ── 7. Top 10 best OOS ────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TOP 10 BEST OOS (from all 200)")
print(f"{'='*70}")
by_oos = sorted(results, key=lambda r: -r["oos_pnl"])[:10]
for r in by_oos:
    print(f"  #{r['trial']:<5} ({r['group']:<6}) IS=${r['is_score']:>+12,.0f}  "
          f"OOS=${r['oos_pnl']:>+10,.0f}  Sharpe={r['oos_sharpe']:.2f}  "
          f"WR={r['oos_wr']:.0f}%  Trades={r['oos_trades']:<4}  "
          f"strats={r['enabled']}")

# ── 8. Top 10 worst OOS ──────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TOP 10 WORST OOS (from all 200)")
print(f"{'='*70}")
by_oos_worst = sorted(results, key=lambda r: r["oos_pnl"])[:10]
for r in by_oos_worst:
    print(f"  #{r['trial']:<5} ({r['group']:<6}) IS=${r['is_score']:>+12,.0f}  "
          f"OOS=${r['oos_pnl']:>+10,.0f}  Sharpe={r['oos_sharpe']:.2f}  "
          f"WR={r['oos_wr']:.0f}%  Trades={r['oos_trades']:<4}  "
          f"strats={r['enabled']}")

# ── 9. Save full results ──────────────────────────────────────────────
out = {
    "study": "oglhmafp_v5",
    "oos_period": f"{oos_dates[0]} to {oos_dates[-1]}",
    "n_days": len(oos_dates),
    "starting_cash": STARTING_CASH,
    "top100": [{"trial": r["trial"], "is_score": r["is_score"], "oos_pnl": r["oos_pnl"],
                "oos_sharpe": r["oos_sharpe"], "oos_wr": r["oos_wr"], "oos_trades": r["oos_trades"],
                "oos_pf": r["oos_pf"], "oos_max_dd": r["oos_max_dd"], "enabled": r["enabled"]}
               for r in top_results],
    "random100": [{"trial": r["trial"], "is_score": r["is_score"], "oos_pnl": r["oos_pnl"],
                   "oos_sharpe": r["oos_sharpe"], "oos_wr": r["oos_wr"], "oos_trades": r["oos_trades"],
                   "oos_pf": r["oos_pf"], "oos_max_dd": r["oos_max_dd"], "enabled": r["enabled"]}
                  for r in rand_results],
}
with open("results/v5_oos_forward_2026.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nFull results saved to results/v5_oos_forward_2026.json")
