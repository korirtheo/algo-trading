"""Wide forward test of gl_split_v2 on 2026 OOS (March onward).

Runs top-100 + random-100 trials through the full simulator with split params.
Reports OOS PnL, per-strategy breakdown, train vs OOS correlation.

Usage:
  python scripts/analysis/wide_forward_split.py
  python scripts/analysis/wide_forward_split.py --top 50 --random 50
  python scripts/analysis/wide_forward_split.py --oos-start 2026-01-01
"""
import sys, os, random, json, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import optuna
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from optimize_combined import (
    set_strategy_params, run_combined_backtest,
    _param_lock, STARTING_CASH, MARGIN_THRESHOLD,
)
from optimize_gl_split import _map_split_to_standard, _build_split_snapshot
from test_full import load_all_picks

# ─── Config ──────────────────────────────────────────────────────────────
STUDY = "gl_split_v2"
DB = "postgresql://postgres@127.0.0.1:5432/optuna_gl_split"
TOP_N = 100
RANDOM_N = 100
RANDOM_SEED = 42
OOS_DIRS = [
    "stored_data_mar_may_2026",
    "stored_data_jun_2026",
    "stored_data_jul_2026",
]


def fetch_trials(study_name, db, top_n, random_n):
    """Fetch top-N by score + random-N from the study."""
    study = optuna.load_study(study_name=study_name, storage=db)
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    # Sort by score descending
    completed.sort(key=lambda t: t.value, reverse=True)

    # Top N
    top_trials = completed[:top_n]

    # Random N from middle (skip top N, sample from rest)
    pool = completed[top_n:]
    rng = random.Random(RANDOM_SEED)
    rand_trials = rng.sample(pool, min(random_n, len(pool)))

    # Combine, deduplicate by trial number
    seen = set()
    selected = []
    for t in top_trials + rand_trials:
        if t.number not in seen:
            seen.add(t.number)
            selected.append(t)

    # Tag train rank
    rank_map = {t.number: i + 1 for i, t in enumerate(completed)}
    for t in selected:
        t._train_rank = rank_map[t.number]

    return selected, len(completed)


def run_one_trial(args):
    """Run a single trial's backtest. Designed for ProcessPoolExecutor."""
    trial_num, trial_score, params, oos_dates, oos_picks = args

    # Add enable flags
    params["enable_g"] = True
    params["enable_l"] = True
    for s in "vhafdrwobkcseijn":
        params[f"enable_{s}"] = False

    try:
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)
        result = run_combined_backtest(oos_picks, oos_dates, params_snapshot=snapshot)

        g = result["strats"].get("G", {"n": 0, "wins": 0, "pnl": 0})
        l = result["strats"].get("L", {"n": 0, "wins": 0, "pnl": 0})

        return {
            "number": trial_num,
            "train_score": trial_score,
            "oos_pnl": result["total_pnl"],
            "oos_pf": result["pf"],
            "oos_n": result["n"],
            "oos_equity": result["equity"],
            "g_n": g["n"], "g_wins": g["wins"], "g_pnl": g["pnl"],
            "l_n": l["n"], "l_wins": l["wins"], "l_pnl": l["pnl"],
            "g_wr": g["wins"] / g["n"] * 100 if g["n"] else 0,
            "l_wr": l["wins"] / l["n"] * 100 if l["n"] else 0,
        }
    except Exception as e:
        return {"number": trial_num, "train_score": trial_score, "error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=TOP_N)
    parser.add_argument("--random", type=int, default=RANDOM_N)
    parser.add_argument("--oos-start", type=str, default="2026-03-01")
    args = parser.parse_args()

    print(f"Loading trials from {STUDY}...")
    trials, total = fetch_trials(STUDY, DB, args.top, args.random)
    print(f"  Total completed: {total}")
    print(f"  Selected: {len(trials)} ({args.top} top + up to {args.random} random)")

    print(f"\nLoading OOS data (2026 from {args.oos_start})...")
    oos_dates, oos_picks = load_all_picks(OOS_DIRS)
    # Filter by date
    oos_dates = [d for d in oos_dates if d >= args.oos_start]
    print(f"  OOS days: {len(oos_dates)} ({oos_dates[0]} to {oos_dates[-1]})")

    # Build work items — pass pick data through pickling
    work = []
    for t in trials:
        work.append((t.number, t.value, dict(t.params), oos_dates, oos_picks))

    print(f"\nRunning backtests ({len(work)} trials)...")
    results = []
    t0 = time.time()

    # Run serially (the simulator isn't picklable for multiprocessing easily)
    for i, w in enumerate(work):
        r = run_one_trial(w)
        results.append(r)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(work) - i - 1) / rate
            print(f"  [{i+1}/{len(work)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # Separate errors
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"\n  {len(errs)} trials errored:")
        for e in errs[:5]:
            print(f"    #{e['number']}: {e['error']}")

    if not ok:
        print("No successful results!")
        return

    # ─── Analysis ────────────────────────────────────────────────────────
    top_results = sorted([r for r in ok if r["train_score"] >= sorted(
        [r2["train_score"] for r2 in ok], reverse=True
    )[min(args.top - 1, len(ok) - 1)]], key=lambda r: r["train_score"], reverse=True)

    # Actually, let's just split by train_rank
    top_set = set()
    for t in trials[:args.top]:
        top_set.add(t.number)

    top_r = [r for r in ok if r["number"] in top_set]
    rand_r = [r for r in ok if r["number"] not in top_set]

    print("\n" + "=" * 72)
    print("  WIDE FORWARD TEST RESULTS — gl_split_v2 on 2026 OOS")
    print("=" * 72)

    # Overall stats
    for label, group in [("TOP 100", top_r), ("RANDOM 100", rand_r)]:
        if not group:
            continue
        pnls = [r["oos_pnl"] for r in group]
        pfs = [r["oos_pf"] for r in group]
        ns = [r["oos_n"] for r in group]
        g_pnls = [r["g_pnl"] for r in group]
        l_pnls = [r["l_pnl"] for r in group]

        profitable = sum(1 for p in pnls if p > 0)
        print(f"\n── {label} ({len(group)} trials) ──")
        print(f"  OOS PnL:  median=${statistics.median(pnls):,.0f}  mean=${statistics.mean(pnls):,.0f}")
        print(f"           best=${max(pnls):,.0f}  worst=${min(pnls):,.0f}")
        print(f"           profitable: {profitable}/{len(group)} ({profitable/len(group)*100:.0f}%)")
        print(f"  OOS PF:   median={statistics.median(pfs):.2f}  mean={statistics.mean(pfs):.2f}")
        print(f"  OOS N:    median={statistics.median(ns):.0f}  mean={statistics.mean(ns):.0f}")
        print(f"  G PnL:    median=${statistics.median(g_pnls):,.0f}  mean=${statistics.mean(g_pnls):,.0f}")
        print(f"  L PnL:    median=${statistics.median(l_pnls):,.0f}  mean=${statistics.mean(l_pnls):,.0f}")

    # Correlation: train score vs OOS PnL
    if len(ok) > 2:
        train_scores = [r["train_score"] for r in ok]
        oos_pnls = [r["oos_pnl"] for r in ok]
        corr = np.corrcoef(train_scores, oos_pnls)[0, 1]
        print(f"\n  Train Score ↔ OOS PnL correlation: {corr:.3f}")

    # Top 20 by OOS PnL
    top20 = sorted(ok, key=lambda r: r["oos_pnl"], reverse=True)[:20]
    print(f"\n{'─' * 72}")
    print("  TOP 20 BY OOS PnL")
    print(f"{'─' * 72}")
    print(f"  {'#':>5}  {'Train':>10}  {'OOS PnL':>12}  {'PF':>6}  {'Trades':>6}  {'G PnL':>12}  {'L PnL':>12}  {'G WR':>6}  {'L WR':>6}")
    for r in top20:
        tag = " *" if r["number"] in top_set else ""
        print(f"  {r['number']:>5}  {r['train_score']:>10,.0f}  ${r['oos_pnl']:>11,.0f}  {r['oos_pf']:>6.2f}  {r['oos_n']:>6}  ${r['g_pnl']:>11,.0f}  ${r['l_pnl']:>11,.0f}  {r['g_wr']:>5.1f}%  {r['l_wr']:>5.1f}%{tag}")

    # Worst 10
    worst10 = sorted(ok, key=lambda r: r["oos_pnl"])[:10]
    print(f"\n{'─' * 72}")
    print("  WORST 10 BY OOS PnL")
    print(f"{'─' * 72}")
    print(f"  {'#':>5}  {'Train':>10}  {'OOS PnL':>12}  {'PF':>6}  {'Trades':>6}  {'G PnL':>12}  {'L PnL':>12}")
    for r in worst10:
        tag = " *" if r["number"] in top_set else ""
        print(f"  {r['number']:>5}  {r['train_score']:>10,.0f}  ${r['oos_pnl']:>11,.0f}  {r['oos_pf']:>6.2f}  {r['oos_n']:>6}  ${r['g_pnl']:>11,.0f}  ${r['l_pnl']:>11,.0f}{tag}")

    print(f"\n  * = was in top-100 by training score")
    print(f"  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
