"""Wide-forward OOS test for gl_1min_v2: top-50 by training score + 50 random.

Each trial runs the FULL G+L param set (enable flags corrected from user_attrs)
over the 1-min pick cache OOS window with the fixed (time-based) liquidity model.

Usage:
    python wide_forward_gl_1min_oos.py --workers 8 [--top 50] [--random 50]
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import optuna

CACHE = os.path.join("stored_data_1min", "fulltest_picks_gap2_vol250k.pkl")
STARTING_CASH = 25_000
STORAGE = "postgresql://postgres@127.0.0.1:5432/optuna_gl_1min"
STUDY = "gl_1min_v2"

_picks_by_date = None  # per-worker global, loaded once in initializer


def _load_cache(_):
    global _picks_by_date
    with open(CACHE, "rb") as f:
        _picks_by_date = pickle.load(f)


def _run_one(job):
    trial_number, params, date_lo, date_hi = job
    try:
        import test_green_candle_combined as tgc
        from optimize_combined import set_strategy_params, _build_param_snapshot

        tgc.USE_DYNAMIC_SLIPPAGE = True
        tgc.USE_MULTIWINDOW_SLIPPAGE = True
        tgc.USE_VOLATILITY_ADJUSTMENT = True
        tgc.USE_2MIN_SLIPPAGE = False
        # G2 studies were trained with first-bar-only fills; replicate the same
        # runtime behavior for OOS so params behave identically to training.
        import os as _os

        set_strategy_params(params)
        if _os.environ.get("G_FIRST_BAR_ONLY", "") == "1":
            tgc.G_FIRST_BAR_ONLY = True
        # PM credit overrides must be applied AFTER set_strategy_params (which
        # resets G_PM_* to registry defaults for untuned params).
        _pmfrac = _os.environ.get("PM_CREDIT_FRACTION", "")
        if _pmfrac:
            tgc.G_PM_CREDIT_FRACTION = float(_pmfrac)
        _pmgate = _os.environ.get("PM_GATE_SCALE", "")
        if _pmgate:
            tgc.G_PM_GATE_STRENGTH_SCALE = float(_pmgate)
        # Margin/cash-cap overrides (must be after set_strategy_params, which
        # resets non-registry globals). Replicates study-time leverage.
        _mm = _os.environ.get("MARGIN_MULTIPLIER", "")
        if _mm:
            tgc.MARGIN_MULTIPLIER = float(_mm)
        _pc = _os.environ.get("MAX_POSITION_PCT_OF_CASH", "")
        if _pc:
            tgc.MAX_POSITION_PCT_OF_CASH = float(_pc)
        snapshot = _build_param_snapshot()

        dates = sorted(d for d in _picks_by_date if date_lo <= d <= date_hi)
        cash = float(STARTING_CASH)
        all_trades = []
        for d in dates:
            picks = _picks_by_date.get(d, [])
            if not picks:
                continue
            cash_account = cash < tgc.MARGIN_THRESHOLD
            states, cash, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account, params=snapshot
            )
            effective_cash = cash + (unsettled if cash_account else 0)
            for st in states:
                if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                    pnl = st.get("pnl", 0)
                    if pnl is not None:
                        all_trades.append(pnl)
            cash = effective_cash

        n = len(all_trades)
        if n == 0:
            return {"trial": trial_number, "n": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0}
        total_pnl = sum(all_trades)
        wins = [p for p in all_trades if p > 0]
        losses = [p for p in all_trades if p <= 0]
        gross_win = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-9
        return {
            "trial": trial_number,
            "n": n,
            "pnl": round(total_pnl, 2),
            "pf": round(gross_win / gross_loss, 3),
            "wr": round(len(wins) / n * 100, 1),
        }
    except Exception as e:
        return {"trial": trial_number, "n": -1, "pnl": 0.0, "pf": 0.0, "wr": 0.0,
                "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--random", type=int, default=50)
    parser.add_argument("--study", default=STUDY, help="Study name to test")
    parser.add_argument("--date-lo", default="2026-03-02")
    parser.add_argument("--date-hi", default="2026-08-07")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Wide-Forward OOS: %s (fixed liquidity)" % args.study)
    print(f"  Top {args.top} by train score + {args.random} random, workers={args.workers}")
    print(f"  OOS window: {args.date_lo} -> {args.date_hi}")

    study = optuna.load_study(study_name=args.study, storage=STORAGE)
    completed = [
        t for t in study.trials
        if t.state.name == "COMPLETE" and t.value is not None and t.value > 0
    ]
    print(f"  {len(completed)} completed trials with positive score")

    from optimize_combined import merge_enable_from_user_attrs

    def params_for(trial):
        return merge_enable_from_user_attrs(trial)

    top = sorted(completed, key=lambda t: t.value, reverse=True)[:args.top]
    rest = [t for t in completed if t.number not in {x.number for x in top}]
    rng = random.Random(args.seed)
    rnd = rng.sample(rest, min(args.random, len(rest)))
    selected = top + rnd
    print(f"  Selected {len(top)} top + {len(rnd)} random = {len(selected)} trials")

    jobs = [(t.number, params_for(t), args.date_lo, args.date_hi) for t in selected]

    t0 = time.time()
    with Pool(args.workers, initializer=_load_cache, initargs=(None,)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_run_one, jobs), 1):
            results.append(r)
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                eta = (el / i) * (len(jobs) - i) if i > 0 else 0
                print(f"  {i}/{len(jobs)} done ({el:.0f}s, ~{eta:.0f}s left)", flush=True)

    # Save CSV
    out_csv = "wide_forward_gl_1min_oos.csv"
    with open(out_csv, "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["trial", "n", "pnl", "pf", "wr"])
        w.writeheader()
        for r in sorted(results, key=lambda x: x.get("trial", 0)):
            w.writerow({k: r.get(k, "") for k in ("trial", "n", "pnl", "pf", "wr")})
    print(f"\nSaved -> {out_csv}")

    # Summary stats
    valid = [r for r in results if r.get("n", 0) > 0]
    pnls = sorted(r["pnl"] for r in valid)
    n = len(pnls)
    med = pnls[n // 2] if n else 0
    mean = sum(pnls) / n if n else 0
    profitable = sum(1 for p in pnls if p > 0)
    best = max(pnls) if pnls else 0
    worst = min(pnls) if pnls else 0
    pf_list = [r["pf"] for r in valid]
    med_pf = sorted(pf_list)[len(pf_list) // 2] if pf_list else 0
    print("\n=== Wide-forward OOS summary ===")
    print(f"  valid trials: {n}")
    print(f"  median PnL:   ${med:,.0f}")
    print(f"  mean PnL:     ${mean:,.0f}")
    print(f"  % profitable: {profitable}/{n} = {profitable/n*100:.0f}%")
    print(f"  best:         ${best:,.0f}   worst: ${worst:,.0f}")
    print(f"  median PF:    {med_pf:.2f}")

    # Top 10 by OOS PnL with train score
    score_by_num = {t.number: t.value for t in completed}
    print("\n  Top 10 trials by OOS PnL:")
    for r in sorted(valid, key=lambda x: x["pnl"], reverse=True)[:10]:
        print(f"    #{r['trial']}: OOS ${r['pnl']:>12,.0f}  n={r['n']:3d}  pf={r['pf']:.2f}  "
              f"wr={r['wr']:.0f}%  train_score={score_by_num.get(r['trial'], 0):,.0f}")

    print(f"\n  Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
