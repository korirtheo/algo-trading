"""Wide forward-test sample from W7, W8, W9 — find what predicts 2026 forward.

For each study:
  - Sample 30 top-by-train + 30 random middle-rank trials
  - Forward-test each on 2026 (parallel)
  - Pull training-time features from user_attrs
Then correlate each feature with forward PnL.

Reveals whether:
  - A universal feature predicts forward across all 3 objectives, OR
  - Different objectives have different "right" features.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psycopg2

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
OUTDIR = "results"

STUDIES = [
    {"name": "W7",   "db": "optuna",      "objective": "legacy: total_pnl * min(pf, 3)",
      "forced_strats": None},
    {"name": "W8",   "db": "optuna_w8",   "objective": "CV-min: min(year_pnl * min(pf, 3))",
      "forced_strats": None},
    {"name": "W9",   "db": "optuna_w9",   "objective": "CV-geomean: geomean(year_pnl * min(pf, 3))",
      "forced_strats": None},
    {"name": "W10a", "db": "optuna_w10",  "objective": "legacy on G+L only, no shape filter",
      "forced_strats": {"g", "l"}},   # CRITICAL: trials don't have enable_* keys, force them
    # W10b deferred — has no trials yet, will be added once it runs
]
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]


def fetch_trials(db, top_n=30, random_n=30, random_range=(50, 500), seed=42):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value
          FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > 0
         ORDER BY tv.value DESC
    """)
    all_sorted = cur.fetchall()

    selected = list(all_sorted[:top_n])
    lo, hi = random_range
    hi = min(hi, len(all_sorted))
    middle = all_sorted[lo:hi] if hi > lo else []
    rng = random.Random(seed)
    rand_sample = rng.sample(middle, min(random_n, len(middle))) if middle else []
    selected.extend(rand_sample)

    seen = set()
    unique = []
    for row in selected:
        if row[0] in seen: continue
        seen.add(row[0])
        unique.append(row)

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
        train_rank = next(i+1 for i, r in enumerate(all_sorted) if r[0] == tid)
        trials.append({"trial_id": tid, "number": num, "score": float(score),
                        "user_attrs": ua, "params": params, "train_rank": train_rank})
    c.close()
    return trials, len(all_sorted)


def forward(trial):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(trial["params"])

    # CRITICAL FIX 2026-06-21: W10+ trials don't have enable_* keys because they
    # were forced via env var. set_strategy_params defaults missing enables to
    # True → wrong strategies fire. Apply forced_strats from study config.
    forced = trial.get("_forced_strats")
    if forced:
        for s in ALL_STRATS:
            merged[f"enable_{s}"] = (s in forced)

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
        "train_score": float(trial["score"]),
        "user_attrs": trial["user_attrs"],
        "forward_pnl": float(eq[-1] - STARTING_CASH),
        "forward_final": float(eq[-1]),
        "forward_multi": float(eq[-1] / STARTING_CASH),
        "forward_max_dd_pct": dd_pct,
        "forward_n_trades": n_trades,
        "forward_worst_trade": float(worst_trade),
    }


def correlate(rows, feature, target="forward_pnl"):
    xs, ys = [], []
    for r in rows:
        v = r.get(feature)
        t = r.get(target)
        if v is None or t is None: continue
        try:
            v = float(v); t = float(t)
        except (TypeError, ValueError): continue
        if not (np.isfinite(v) and np.isfinite(t)): continue
        xs.append(v); ys.append(t)
    if len(xs) < 5: return None, None
    xs = np.array(xs); ys = np.array(ys)
    p = float(np.corrcoef(xs, ys)[0, 1])
    def _rank(a):
        o = a.argsort(); r = np.empty_like(o, dtype=float); r[o] = np.arange(len(a)); return r
    s = float(np.corrcoef(_rank(xs), _rank(ys))[0, 1])
    return p, s


def extract_features(forward_result):
    """Pull numeric features from user_attrs for correlation analysis."""
    ua = forward_result["user_attrs"]
    out = {
        "forward_pnl": forward_result["forward_pnl"],
        "forward_max_dd_pct": forward_result["forward_max_dd_pct"],
        "forward_n_trades": forward_result["forward_n_trades"],
        "train_score": forward_result["train_score"],
        "train_rank": forward_result["train_rank"],
    }
    # Try to parse all numeric user_attrs
    numeric_keys = ["total_pnl", "sum_pnl", "min_pnl", "min_pf", "mean_pnl",
                     "pf", "wr", "n", "sharpe_pct", "equity",
                     "pnl_2022", "pnl_2023", "pnl_2024", "pnl_2025",
                     "pf_2022", "pf_2023", "pf_2024", "pf_2025",
                     "n_2022", "n_2023", "n_2024", "n_2025"]
    for k in numeric_keys:
        if k in ua:
            try:
                out[k] = float(ua[k])
            except (TypeError, ValueError):
                pass
    # Derived: per-year stats (only if all 4 years present)
    years = ["2022", "2023", "2024", "2025"]
    pnls = [out.get(f"pnl_{y}") for y in years]
    pfs = [out.get(f"pf_{y}") for y in years]
    if all(p is not None for p in pnls):
        out["yr_min_pnl"] = min(pnls)
        out["yr_max_pnl"] = max(pnls)
        out["yr_mean_pnl"] = statistics.mean(pnls)
        try:
            out["yr_std_pnl"] = statistics.stdev(pnls)
            out["yr_cv_pnl"] = out["yr_std_pnl"] / abs(out["yr_mean_pnl"]) if out["yr_mean_pnl"] else 999
        except statistics.StatisticsError:
            pass
    if all(p is not None for p in pfs):
        out["yr_min_pf"] = min(pfs)
        out["yr_max_pf"] = max(pfs)
        out["yr_mean_pf"] = statistics.mean(pfs)
    return out


def run_study(study, top_n=30, random_n=30):
    print(f"\n{'='*92}")
    print(f"  {study['name']}  objective: {study['objective']}  db: {study['db']}")
    if study.get("forced_strats"):
        print(f"  FORCED strats: {sorted(study['forced_strats'])}")
    print(f"{'='*92}")

    trials, total = fetch_trials(study["db"], top_n=top_n, random_n=random_n)
    # Attach forced_strats per-trial so the worker pool can see it
    for t in trials:
        t["_forced_strats"] = study.get("forced_strats")
    print(f"  Total trials in DB: {total}, selected: {len(trials)} "
          f"(top {top_n} + random {random_n} from middle)")

    print(f"  Forward-testing {len(trials)} trials (6 parallel workers)...")
    forwards = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(forward, t): t["number"] for t in trials}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                forwards.append(r)
                done += 1
                marker = "TOP" if r["train_rank"] <= top_n else "MID"
                print(f"    [{done:>2}/{len(trials)}] {marker} #{r['trial_number']:<5} rank={r['train_rank']:<4} "
                      f"-> fwd ${r['forward_pnl']:>+9,.0f}")
            except Exception as e:
                print(f"    failed: {e}")

    forwards.sort(key=lambda r: -r["forward_pnl"])

    out_path = f"{OUTDIR}/wide_forward_{study['name'].lower()}.json"
    os.makedirs(OUTDIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(forwards, f, indent=2)
    print(f"\n  Wrote {out_path}")

    # Top 10 by forward
    print(f"\n  Top 10 by FORWARD PnL:")
    print(f"  {'rank':<4} {'trial':<6} {'train_rank':<11} {'train$':>10} {'fwd$':>10} {'fwd_x':>6} {'fwd_DD%':>8}")
    for i, r in enumerate(forwards[:10]):
        print(f"  {i+1:<4} #{r['trial_number']:<5} {r['train_rank']:<11} ${r['train_score']/1000:>7.0f}K "
              f"${r['forward_pnl']:>+8,.0f} {r['forward_multi']:>5.2f}x {r['forward_max_dd_pct']:>7.1f}%")

    # Feature correlation
    feature_rows = [extract_features(r) for r in forwards]
    candidate_features = [
        "train_score", "train_rank", "total_pnl", "sum_pnl", "min_pnl",
        "min_pf", "pf", "wr", "n", "mean_pnl", "sharpe_pct",
        "pnl_2022", "pnl_2023", "pnl_2024", "pnl_2025",
        "pf_2022", "pf_2023", "pf_2024", "pf_2025",
        "yr_min_pnl", "yr_max_pnl", "yr_mean_pnl", "yr_cv_pnl",
        "yr_min_pf", "yr_max_pf",
    ]
    correlations = {}
    for f in candidate_features:
        p, s = correlate(feature_rows, f, "forward_pnl")
        if p is not None:
            correlations[f] = {"pearson": p, "spearman": s}

    print(f"\n  Correlation with forward_pnl (features available in {study['name']}):")
    print(f"  {'feature':<22} {'pearson':>9} {'spearman':>10}")
    print(f"  {'-'*22} {'-'*9} {'-'*10}")
    sorted_corrs = sorted(correlations.items(), key=lambda x: -abs(x[1]["pearson"]))
    for f, c in sorted_corrs[:15]:
        print(f"  {f:<22} {c['pearson']:>9.3f} {c['spearman']:>10.3f}")

    return {
        "study": study["name"],
        "objective": study["objective"],
        "db": study["db"],
        "n_total_trials": total,
        "n_forward_tested": len(forwards),
        "top10_by_forward": [
            {"rank": i+1, "trial": r["trial_number"], "train_rank": r["train_rank"],
              "train_score": r["train_score"], "forward_pnl": r["forward_pnl"],
              "forward_multi": r["forward_multi"], "forward_max_dd_pct": r["forward_max_dd_pct"]}
            for i, r in enumerate(forwards[:10])
        ],
        "correlations_with_forward_pnl": correlations,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Comma-separated study names to run (e.g., 'W10a')")
    args = ap.parse_args()

    TOP_N = 50
    RANDOM_N = 50
    only_names = set(args.only.split(",")) if args.only else None

    studies = [s for s in STUDIES if (not only_names or s["name"] in only_names)]
    print(f"Running studies: {[s['name'] for s in studies]}")

    summaries = []
    for study in studies:
        try:
            s = run_study(study, top_n=TOP_N, random_n=RANDOM_N)
            summaries.append(s)
        except Exception as e:
            print(f"  STUDY {study['name']} FAILED: {e}")
            import traceback; traceback.print_exc()

    out_summary = f"{OUTDIR}/wide_forward_comparison.json"
    with open(out_summary, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n=== Wrote {out_summary} ===")

    # Cross-study comparison
    print(f"\n{'='*92}")
    print(f"  CROSS-STUDY COMPARISON")
    print(f"{'='*92}")
    print(f"  {'study':<5} {'objective':<48} {'top1 fwd PnL':>14}")
    for s in summaries:
        top1 = s["top10_by_forward"][0]["forward_pnl"] if s["top10_by_forward"] else 0
        print(f"  {s['study']:<5} {s['objective']:<48} ${top1:>+12,.0f}")
    print()
    print(f"  Baselines: #124 deployed +$22,534  |  W7 #254 (original) +$199,553")

    # Which features predict forward across studies?
    print(f"\n  TOP-3 PREDICTIVE FEATURES PER STUDY (by abs(pearson)):")
    print(f"  {'study':<5} {'feature':<22} {'pearson':>9} {'spearman':>10}")
    for s in summaries:
        sorted_c = sorted(s["correlations_with_forward_pnl"].items(),
                          key=lambda x: -abs(x[1]["pearson"]))
        for i, (f, c) in enumerate(sorted_c[:3]):
            study_label = s["study"] if i == 0 else ""
            print(f"  {study_label:<5} {f:<22} {c['pearson']:>9.3f} {c['spearman']:>10.3f}")


if __name__ == "__main__":
    main()
