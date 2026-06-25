"""Forward-test a WIDE sample of W9 trials on 2026.

- Top 30 by training score
- Random 30 from middle-rank trials (ranks 50-500)
Goal: find any hidden gem trials that train mediocre but forward well,
mirroring the W7 #254 pattern (rank 2 by train, rank 1 by forward).
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
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
OUTDIR = "results/walk_forward_v9_cv"


def fetch_trials(db, top_n=30, random_n=30, random_range=(50, 500), seed=42):
    """Pull top_n by training score + random_n from the middle ranks."""
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value
          FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > 0
         ORDER BY tv.value DESC
    """)
    all_sorted = cur.fetchall()
    print(f"  Total trials w/ positive score: {len(all_sorted)}")

    # Top N
    selected = list(all_sorted[:top_n])
    # Random sample from middle ranks (avoiding the very best + the very worst)
    lo, hi = random_range
    hi = min(hi, len(all_sorted))
    middle = all_sorted[lo:hi]
    rng = random.Random(seed)
    rand_sample = rng.sample(middle, min(random_n, len(middle)))
    selected.extend(rand_sample)

    # Dedup by trial_id (in case overlap)
    seen = set()
    unique = []
    for row in selected:
        if row[0] in seen: continue
        seen.add(row[0])
        unique.append(row)
    print(f"  Total to forward-test: {len(unique)}")

    # Fetch params for each
    trials = []
    for tid, num, score in unique:
        cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
        ua = dict(cur.fetchall())
        cur.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s", (tid,))
        params = {}
        for name, val, dist in cur.fetchall():
            try:
                d = json.loads(dist) if dist else {}
                kind = d.get("name", "")
                if kind == "CategoricalDistribution":
                    params[name] = d["attributes"]["choices"][int(val)]
                elif "Int" in kind:
                    params[name] = int(val)
                else:
                    params[name] = float(val)
            except Exception:
                params[name] = val
        # Compute train rank too (1-indexed)
        train_rank = next(i+1 for i, r in enumerate(all_sorted) if r[0] == tid)
        trials.append({"trial_id": tid, "number": num, "score": float(score),
                        "user_attrs": ua, "params": params,
                        "train_rank": train_rank})
    c.close()
    return trials


def forward(trial):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(trial["params"])
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    daily_eq = [cash]
    n_trades = 0
    worst_trade = 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            daily_eq.append(cash); continue
        if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as nfp
            day_picks = nfp(day_picks, d, tgc.NEWS_MIN_ARTICLES, tgc.NEWS_REQUIRE_CATALYST)
        if not day_picks:
            daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_trades += 1
                if st["pnl"] < worst_trade: worst_trade = st["pnl"]
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = float((eq - peak).min() / peak[(eq - peak).argmin()] * 100) if len(peak) and peak[(eq - peak).argmin()] > 0 else 0
    return {
        "trial_number": trial["number"],
        "train_rank": trial["train_rank"],
        "score": float(trial["score"]),
        "user_attrs": trial["user_attrs"],
        "forward_final": float(eq[-1]),
        "forward_pnl": float(eq[-1] - STARTING_CASH),
        "forward_multi": float(eq[-1] / STARTING_CASH),
        "forward_max_dd_pct": dd_pct,
        "forward_n_trades": n_trades,
        "forward_worst_trade": float(worst_trade),
    }


def main():
    print("Fetching trials from optuna_w9...")
    trials = fetch_trials("optuna_w9", top_n=30, random_n=30, random_range=(50, 500), seed=42)

    os.makedirs(OUTDIR, exist_ok=True)

    print(f"\nForward-testing {len(trials)} trials on 2026 (6 parallel workers)...")
    forwards = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(forward, t): t["number"] for t in trials}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                forwards.append(r)
                done += 1
                marker = "  TOP" if r["train_rank"] <= 30 else "  MID"
                print(f"  [{done}/{len(trials)}]{marker} #{r['trial_number']:<5} train_rank={r['train_rank']:<4} "
                      f"train${r['score']/1000:.0f}K -> fwd ${r['forward_pnl']:>+9,.0f} ({r['forward_multi']:.2f}x) "
                      f"DD {r['forward_max_dd_pct']:>5.1f}%")
            except Exception as e:
                print(f"  forward failed: {e}")

    forwards.sort(key=lambda r: -r["forward_pnl"])

    out_path = f"{OUTDIR}/w9_wide_forward.json"
    with open(out_path, "w") as f:
        json.dump(forwards, f, indent=2)
    print(f"\nWrote {out_path}")

    # Summary
    print(f"\n{'='*92}")
    print(f"  RANKED BY FORWARD PnL (top 15)")
    print(f"{'='*92}")
    print(f"  {'rank':<4} {'trial':<7} {'train_rank':<11} {'train$':>10} {'fwd$':>10} {'fwd_x':>6} {'fwd_DD%':>7} {'fwd_n':>6}")
    for i, r in enumerate(forwards[:15]):
        print(f"  {i+1:<4} #{r['trial_number']:<5} {r['train_rank']:<11} ${r['score']/1000:>7.0f}K "
              f"${r['forward_pnl']:>+8,.0f} {r['forward_multi']:>5.2f}x {r['forward_max_dd_pct']:>6.1f}% {r['forward_n_trades']:>5}")

    if forwards:
        w = forwards[0]
        print(f"\n  WINNER BY FORWARD: #{w['trial_number']} (train rank {w['train_rank']})")
        print(f"    Forward 2026: ${w['forward_pnl']:+,.0f} ({w['forward_multi']:.2f}x)")
        print(f"    vs W7 #254 (+$199,553):  ${w['forward_pnl'] - 199553:+,.0f}")
        print(f"    vs W8 #1550 (+$36,668):  ${w['forward_pnl'] - 36668:+,.0f}")
        print(f"    vs #124 deployed:        ${w['forward_pnl'] - 22534:+,.0f}")


if __name__ == "__main__":
    main()
