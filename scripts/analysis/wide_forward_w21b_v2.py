"""Wide-forward on W21b v2 top trials.

Pulls top 50 + 50 random middle-rank trials from W21b v2 (full 2024+2025 data,
single-window objective). Tests each on:
  - 2022 (truly held out — never in W21b v2 training)
  - 2026 Mar-Jun (truly held out — training ends 2026-02-28)

Reports:
  - Per-trial PnL/PF/WR on each OOS window
  - Pearson train_score -> 2022 OOS forward
  - Pearson train_score -> 2026 Mar-Jun OOS forward
  - % of top 50 beating #511 (deployed) on each OOS
  - Identifies the regime-robust basin (good on BOTH OOS windows)
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
ALL_STRATS = ["h", "g", "a", "f", "d", "v", "p", "m", "r", "w", "o", "b", "k", "c", "s", "e", "i", "j", "n", "l", "x"]
EXIT_PARAMS = ["g_target_pct", "g_stop_pct", "g_time_limit_min", "g_trail_pct", "g_trail_activate_pct"]

WINDOWS = [
    {"name": "2022_OOS", "dirs": ["stored_data_2022"],
     "date_lo": "2022-01-01", "date_hi": "2022-12-31"},
    {"name": "2026_Mar_Jun_OOS",
     "dirs": ["stored_data", "stored_data_mar_may_2026",
              "stored_data_jun_2026", "stored_data_2026_gap_fill"],
     "date_lo": "2026-03-01", "date_hi": "2026-06-24"},
]


def fetch_trials(top_n=50, random_n=50, random_range=(100, 800), seed=42):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname="optuna_w21b_v2")
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value FROM trials t
          JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > 0
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
    exit_params, label, window_name, window_dirs, date_lo, date_hi = args
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

    dirs = [d for d in window_dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])

    cash = STARTING_CASH
    n_trades = 0
    g_n, g_w = 0, 0
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp:
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_trades += 1
                if st.get("strategy") == "G":
                    g_n += 1
                    if st["pnl"] > 0:
                        g_w += 1
        cash = end_c + (unset if is_cash else 0)
    return {
        "label": label, "window": window_name,
        "final": float(cash), "pnl": float(cash - STARTING_CASH),
        "n_trades": n_trades, "g_n": g_n, "g_wr": (g_w / g_n * 100) if g_n else 0,
    }


def main():
    print("=" * 100)
    print("  W21b v2 wide-forward — top 50 + random 50 on 2022 OOS + 2026 Mar-Jun OOS")
    print("=" * 100)

    trials, total = fetch_trials(top_n=50, random_n=50, random_range=(100, 800))
    print(f"  W21b v2 trials in DB: {total}, selected: {len(trials)}")

    with open(W13_DEPLOY) as f:
        w13_full = json.load(f)["params"]
    with open(W21B_DEPLOY) as f:
        w21b_full = json.load(f)["params"]
    w13_exits = {k: w13_full[k] for k in EXIT_PARAMS}
    w21b511_exits = {k: w21b_full[k] for k in EXIT_PARAMS}

    jobs = []
    for w in WINDOWS:
        jobs.append((w13_exits, "W13_baseline", w["name"], w["dirs"], w["date_lo"], w["date_hi"]))
        jobs.append((w21b511_exits, "W21b_#511", w["name"], w["dirs"], w["date_lo"], w["date_hi"]))
        for t in trials:
            ep = {k: t["params"][k] for k in EXIT_PARAMS if k in t["params"]}
            marker = "TOP" if t["train_rank"] <= 50 else "MID"
            label = f"{marker}_#{t['number']}_rank{t['train_rank']}"
            jobs.append((ep, label, w["name"], w["dirs"], w["date_lo"], w["date_hi"]))

    print(f"  Forward jobs: {len(jobs)} (across {len(WINDOWS)} OOS windows, 6 workers)")
    results = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(forward, j): j[1:3] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                done += 1
                if done % 20 == 0:
                    print(f"    [{done}/{len(jobs)}] done")
            except Exception as e:
                lbl, win = futs[fut]
                print(f"    FAIL {lbl} / {win}: {e}")

    # Index by (label, window)
    by_lw = {(r["label"], r["window"]): r for r in results}
    trial_results = {}  # number -> {window: pnl}
    for r in results:
        if r["label"].startswith(("TOP_", "MID_")):
            num = int(r["label"].split("#")[1].split("_")[0])
            trial_results.setdefault(num, {})[r["window"]] = r["pnl"]

    base_w13 = {w["name"]: by_lw[("W13_baseline", w["name"])]["pnl"] for w in WINDOWS}
    base_511 = {w["name"]: by_lw[("W21b_#511", w["name"])]["pnl"] for w in WINDOWS}

    print()
    print("=" * 100)
    print("  RESULTS — W21b v2 wide-forward")
    print("=" * 100)
    print(f"\nBaselines:")
    for w in WINDOWS:
        print(f"  {w['name']:<22}  W13: ${base_w13[w['name']]:>+10,.0f}  #511: ${base_511[w['name']]:>+10,.0f}")

    # Per-trial summary: pull train_score + 2022 OOS + 2026 OOS
    summary_rows = []
    for t in trials:
        n = t["number"]
        pn_2022 = trial_results.get(n, {}).get("2022_OOS", 0)
        pn_2026 = trial_results.get(n, {}).get("2026_Mar_Jun_OOS", 0)
        summary_rows.append({
            "trial": n, "rank": t["train_rank"], "score": t["score"],
            "params": t["params"], "p_2022": pn_2022, "p_2026": pn_2026,
        })

    # Pearson on train vs each OOS
    xs = np.array([r["score"] for r in summary_rows])
    ys_22 = np.array([r["p_2022"] for r in summary_rows])
    ys_26 = np.array([r["p_2026"] for r in summary_rows])
    pearson_22 = float(np.corrcoef(xs, ys_22)[0, 1]) if xs.std() > 0 else 0
    pearson_26 = float(np.corrcoef(xs, ys_26)[0, 1]) if xs.std() > 0 else 0

    top50 = [r for r in summary_rows if r["rank"] <= 50]
    beat_511_2022 = sum(1 for r in top50 if r["p_2022"] > base_511["2022_OOS"]) / len(top50) * 100
    beat_511_2026 = sum(1 for r in top50 if r["p_2026"] > base_511["2026_Mar_Jun_OOS"]) / len(top50) * 100

    # Identify regime-robust: beats #511 on BOTH OOS windows
    robust = [r for r in top50 if r["p_2022"] > base_511["2022_OOS"] and r["p_2026"] > base_511["2026_Mar_Jun_OOS"]]
    print(f"\nTop 50 trials forward stats:")
    print(f"  Pearson train_score -> 2022 OOS:    {pearson_22:+.3f}")
    print(f"  Pearson train_score -> 2026 OOS:    {pearson_26:+.3f}")
    print(f"  Top 50 beating #511 on 2022:        {beat_511_2022:.0f}% ({sum(1 for r in top50 if r['p_2022']>base_511['2022_OOS'])}/{len(top50)})")
    print(f"  Top 50 beating #511 on 2026 Mar-Jun: {beat_511_2026:.0f}% ({sum(1 for r in top50 if r['p_2026']>base_511['2026_Mar_Jun_OOS'])}/{len(top50)})")
    print(f"  REGIME-ROBUST (beats #511 on BOTH): {len(robust)}/{len(top50)} trials")

    if robust:
        print(f"\n  Top 10 regime-robust trials (sorted by 2022 OOS + 2026 OOS sum):")
        robust.sort(key=lambda r: -(r["p_2022"] + r["p_2026"]))
        print(f"  {'#':>5} {'rank':>4} {'train$':>11} {'2022 OOS':>11} {'2026 OOS':>11} {'target':>6} {'stop':>5} {'time':>5} {'trail':>5}")
        for r in robust[:10]:
            p = r["params"]
            print(f"  {r['trial']:>5} {r['rank']:>4} ${r['score']/1e6:>9.1f}M ${r['p_2022']:>+9,.0f} ${r['p_2026']:>+9,.0f} "
                  f"{p['g_target_pct']:>6} {p['g_stop_pct']:>5} {p['g_time_limit_min']:>5} {p['g_trail_pct']:>5}")
    else:
        print(f"\n  NO regime-robust trials. #511's basin is structural — don't swap.")

    # Save
    summary = {
        "n_total": total, "n_tested": len(trials),
        "baselines": {w["name"]: {"W13": base_w13[w["name"]], "511": base_511[w["name"]]} for w in WINDOWS},
        "pearson": {"2022_OOS": pearson_22, "2026_Mar_Jun_OOS": pearson_26},
        "pct_top50_beat_511": {"2022_OOS": beat_511_2022, "2026_Mar_Jun_OOS": beat_511_2026},
        "n_regime_robust": len(robust),
        "trials": [{k: r[k] for k in ("trial", "rank", "score", "params", "p_2022", "p_2026")} for r in summary_rows],
    }
    out_path = "results/wide_forward_w21b_v2.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
