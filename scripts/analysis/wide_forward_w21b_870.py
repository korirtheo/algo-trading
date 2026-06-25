"""Wide-forward test for W21b #870 basin (target=85, stop=24, time=33, trail=0.5).

Same methodology as #511's wide-forward (which gave Pearson +0.956, 82% beat W13).

Compares #870 basin vs:
  - W13 #1202 (replaced)
  - W21b #511 (currently deployed)
  - 50 random rank-100-500 trials (control)

Forward window: Mar-Jun 2026 (true OOS — training ended 2026-02-28).

Outputs:
  - Per-trial forward_pnl, dd, n_trades
  - Pearson train vs forward
  - % basin beating W13 baseline
  - % basin beating #511 deployed
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psycopg2

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W13_DEPLOY = "config/trial_w13_1202_deploy.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
DATA_DIRS = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
OUTDIR = "results"
ALL_STRATS = ["h", "g", "a", "f", "d", "v", "p", "m", "r", "w", "o", "b", "k", "c", "s", "e", "i", "j", "n", "l", "x"]
EXIT_PARAMS = ["g_target_pct", "g_stop_pct", "g_time_limit_min", "g_trail_pct", "g_trail_activate_pct"]


def fetch_trials(top_n=50, random_n=50, random_range=(100, 500), seed=42):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname="optuna_w21b")
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value FROM trials t
          JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > 0
           AND t.study_id=(SELECT study_id FROM studies WHERE study_name='w21b_target100')
         ORDER BY tv.value DESC
    """)
    sorted_trials = cur.fetchall()

    selected = list(sorted_trials[:top_n])
    lo, hi = random_range
    hi = min(hi, len(sorted_trials))
    rand_pool = sorted_trials[lo:hi] if hi > lo else []
    rng = random.Random(seed)
    rand_sample = rng.sample(rand_pool, min(random_n, len(rand_pool))) if rand_pool else []
    selected.extend(rand_sample)

    seen = set()
    trials = []
    for tid, num, score in selected:
        if tid in seen:
            continue
        seen.add(tid)
        cur.execute("SELECT param_name, param_value FROM trial_params WHERE trial_id=%s", (tid,))
        params = {}
        for name, val in cur.fetchall():
            if name == "g_time_limit_min":
                params[name] = int(float(val))
            else:
                params[name] = float(val)
        train_rank = next(i + 1 for i, r in enumerate(sorted_trials) if r[0] == tid)
        trials.append({"trial_id": tid, "number": num, "score": float(score),
                       "params": params, "train_rank": train_rank})
    c.close()
    return trials, len(sorted_trials)


def forward(args):
    trial_or_label, exit_params, label = args
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f:
        baseline = json.load(f)
    with open(W13_DEPLOY) as f:
        w13 = json.load(f)["params"]
    merged = {**baseline, **w13, **exit_params}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)

    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0
    tgc.NEWS_MODULATOR_ENABLED = False

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if "2026-03-01" <= d <= "2026-06-30"])

    cash = STARTING_CASH
    eq_curve = [cash]
    n_trades = 0
    worst = 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            eq_curve.append(cash)
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
        except Exception:
            eq_curve.append(cash)
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_trades += 1
                if st["pnl"] < worst:
                    worst = st["pnl"]
        cash = end_c + (unset if is_cash else 0)
        eq_curve.append(cash)
    eq = np.array(eq_curve)
    peak = np.maximum.accumulate(eq)
    dd_arr = (eq - peak) / peak * 100
    return {
        "label": label,
        "final": float(eq[-1]),
        "forward_pnl": float(eq[-1] - STARTING_CASH),
        "max_dd_pct": float(dd_arr.min()),
        "n_trades": n_trades,
        "worst_trade": float(worst),
    }


def main():
    print("=" * 92)
    print("  W21b #870 BASIN wide-forward test")
    print("  Forward window: 2026-03-01 -> 2026-06-30 (true OOS)")
    print("=" * 92)

    trials, total = fetch_trials(top_n=50, random_n=80, random_range=(100, 1200))
    print(f"  Total trials in W21b: {total}")
    print(f"  Selected: top 50 + random 80 from rank 100-1200 = {len(trials)}")

    # Build forward jobs
    jobs = []
    # Baselines first
    with open(W13_DEPLOY) as f:
        w13_full = json.load(f)["params"]
    with open(W21B_DEPLOY) as f:
        w21b_full = json.load(f)["params"]
    w13_exits = {k: w13_full[k] for k in EXIT_PARAMS}
    w21b511_exits = {k: w21b_full[k] for k in EXIT_PARAMS}
    jobs.append((None, w13_exits, "W13_#1202_baseline"))
    jobs.append((None, w21b511_exits, "W21b_#511_deployed"))
    for t in trials:
        exit_params = {k: t["params"][k] for k in EXIT_PARAMS if k in t["params"]}
        marker = "TOP" if t["train_rank"] <= 50 else "MID"
        label = f"{marker}_#{t['number']}_rank{t['train_rank']}"
        jobs.append((t, exit_params, label))

    print(f"  Total forward jobs: {len(jobs)} (incl. 2 baselines)")
    print(f"  Running 6 parallel workers...")

    forwards = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(forward, j): j[2] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                # Attach trial metadata
                label = r["label"]
                trial_meta = None
                for t in trials:
                    marker = "TOP" if t["train_rank"] <= 50 else "MID"
                    if label == f"{marker}_#{t['number']}_rank{t['train_rank']}":
                        trial_meta = t
                        break
                if trial_meta:
                    r["trial_number"] = trial_meta["number"]
                    r["train_score"] = trial_meta["score"]
                    r["train_rank"] = trial_meta["train_rank"]
                    r["params"] = trial_meta["params"]
                forwards.append(r)
                done += 1
                print(f"    [{done:>3}/{len(jobs)}] {label:<32}  fwd ${r['forward_pnl']:>+11,.0f}  DD {r['max_dd_pct']:>6.1f}%  n={r['n_trades']:>3}")
            except Exception as e:
                print(f"    failed [{futs[fut]}]: {e}")

    # Split baselines
    base_w13 = next(r for r in forwards if r["label"] == "W13_#1202_baseline")
    base_511 = next(r for r in forwards if r["label"] == "W21b_#511_deployed")
    trial_rows = [r for r in forwards if "trial_number" in r]
    top50 = [r for r in trial_rows if r["train_rank"] <= 50]
    mid50 = [r for r in trial_rows if r["train_rank"] > 50]

    # Pearson train vs forward (over trial rows only)
    if len(trial_rows) >= 5:
        xs = np.array([r["train_score"] for r in trial_rows])
        ys = np.array([r["forward_pnl"] for r in trial_rows])
        pearson = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 else 0
    else:
        pearson = 0

    pct_beat_w13 = sum(1 for r in top50 if r["forward_pnl"] > base_w13["forward_pnl"]) / len(top50) * 100
    pct_beat_511 = sum(1 for r in top50 if r["forward_pnl"] > base_511["forward_pnl"]) / len(top50) * 100
    top50_med = float(np.median([r["forward_pnl"] for r in top50]))
    top50_max = float(max(r["forward_pnl"] for r in top50))
    top50_min = float(min(r["forward_pnl"] for r in top50))
    mid50_med = float(np.median([r["forward_pnl"] for r in mid50])) if mid50 else 0

    print()
    print("=" * 92)
    print("  RESULTS")
    print("=" * 92)
    print(f"  W13 #1202 baseline:     fwd ${base_w13['forward_pnl']:>+11,.0f}  DD {base_w13['max_dd_pct']:>6.1f}%")
    print(f"  W21b #511 deployed:     fwd ${base_511['forward_pnl']:>+11,.0f}  DD {base_511['max_dd_pct']:>6.1f}%")
    print()
    print(f"  #870 BASIN (top 50):    median ${top50_med:>+11,.0f}  max ${top50_max:>+11,.0f}  min ${top50_min:>+11,.0f}")
    print(f"  Random control (mid 50): median ${mid50_med:>+11,.0f}")
    print()
    print(f"  Pearson train -> forward (all 100 trials): {pearson:+.3f}")
    print(f"  Top-50 basin beating W13 baseline: {pct_beat_w13:.1f}%")
    print(f"  Top-50 basin beating #511 deployed: {pct_beat_511:.1f}%")
    print()
    print(f"  Top 10 forward winners:")
    for i, r in enumerate(sorted(top50, key=lambda x: -x["forward_pnl"])[:10]):
        print(f"    {i+1:>2}. #{r['trial_number']:<5}  fwd ${r['forward_pnl']:>+11,.0f}  "
              f"target={r['params'].get('g_target_pct'):>5} stop={r['params'].get('g_stop_pct'):>5} "
              f"time={r['params'].get('g_time_limit_min'):>4} trail={r['params'].get('g_trail_pct'):>4}")

    summary = {
        "study": "W21b basin around #870",
        "forward_window": "2026-03-01 to 2026-06-30",
        "n_total_trials": total,
        "n_forward_tested": len(trial_rows),
        "baselines": {"W13_#1202": base_w13, "W21b_#511": base_511},
        "stats": {
            "pearson_train_vs_forward": pearson,
            "top50_median_forward": top50_med,
            "top50_max_forward": top50_max,
            "top50_min_forward": top50_min,
            "mid50_median_forward": mid50_med,
            "pct_top50_beat_W13": pct_beat_w13,
            "pct_top50_beat_511": pct_beat_511,
        },
        "forwards": sorted(trial_rows, key=lambda x: -x["forward_pnl"]),
    }
    out_path = f"{OUTDIR}/wide_forward_w21b_870_basin.json"
    os.makedirs(OUTDIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
