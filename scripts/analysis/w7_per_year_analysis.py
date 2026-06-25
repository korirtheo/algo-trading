"""Per-year decomposition for top W7 trials — empirical objective design.

For each of the top N trials, runs SEPARATE backtests on 2022, 2023, 2024,
2025 with fresh $25K starting cash. Also forward-tests on 2026 (blind).

Output:
  results/walk_forward_v7_news/per_year_analysis.csv   — full per-trial table
  results/walk_forward_v7_news/per_year_correlations.json — feature vs 2026 PnL

Then you can SEE which per-year aggregate best predicts 2026 generalization:
  - total_pnl (the current objective)
  - min(yearly pnl)
  - min(yearly pf)
  - stdev(pnl) / mean(pnl)   [coefficient of variation]
  - geomean(yearly pnl)
  - 25th percentile yearly pnl
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import csv
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psycopg2


STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
OUTDIR = "results/walk_forward_v7_news"

YEAR_DIRS = {
    "2022": ["stored_data_2022"],
    "2023": ["stored_data_2023"],
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"],
}


def fetch_top_n(n):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname="optuna")
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value
          FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > 0
         ORDER BY tv.value DESC LIMIT %s
    """, (n,))
    rows = cur.fetchall()
    trials = []
    for tid, num, score in rows:
        cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
        ua = dict(cur.fetchall())
        cur.execute(
            "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s",
            (tid,))
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
        trials.append({
            "number": num, "score": float(score), "user_attrs": ua,
            "params": params,
        })
    c.close()
    return trials


def backtest_year(year, params):
    """Single-year backtest with fresh $25K. Returns dict of stats."""
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(params)
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0

    dirs = [d for d in YEAR_DIRS[year] if os.path.exists(d)]
    if not dirs:
        return {"pnl": 0, "pf": 0, "n": 0, "wins": 0, "losses": 0,
                "max_dd_pct": 0, "max_dd_dollar": 0, "equity_end": STARTING_CASH}
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith(year)])

    cash = STARTING_CASH
    eq_curve = [cash]
    trades = []
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if day_picks and tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as nfp
            day_picks = nfp(day_picks, d, tgc.NEWS_MIN_ARTICLES, tgc.NEWS_REQUIRE_CATALYST)
        if not day_picks:
            eq_curve.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            eq_curve.append(cash); continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                trades.append(st["pnl"])
        cash = end_c + (unset if is_cash else 0)
        eq_curve.append(cash)

    eq = np.array(eq_curve)
    peak = np.maximum.accumulate(eq)
    dd_pct = float((eq - peak).min() / peak[(eq - peak).argmin()] * 100) if len(peak) and peak[(eq - peak).argmin()] > 0 else 0
    dd_dollar = float((eq - peak).min())

    if not trades:
        return {"pnl": 0, "pf": 0, "n": 0, "wins": 0, "losses": 0,
                "max_dd_pct": dd_pct, "max_dd_dollar": dd_dollar,
                "equity_end": float(eq[-1])}

    pnls = np.array(trades)
    pnl_total = float(pnls.sum())
    wins = float(pnls[pnls > 0].sum())
    losses = abs(float(pnls[pnls <= 0].sum()))
    pf = float(wins / losses) if losses > 1e-9 else 99.0
    n_wins = int((pnls > 0).sum())
    n_losses = int((pnls <= 0).sum())
    return {
        "pnl": pnl_total, "pf": pf, "n": len(trades),
        "wins": n_wins, "losses": n_losses,
        "max_dd_pct": dd_pct, "max_dd_dollar": dd_dollar,
        "equity_end": float(eq[-1]),
    }


def per_trial_worker(trial):
    """Worker: run 5 single-year backtests on one trial. Sequential within process."""
    out = {
        "number": trial["number"],
        "train_score": trial["score"],
        "train_pnl_attr": float(trial["user_attrs"].get("total_pnl", 0)),
        "train_pf_attr": float(trial["user_attrs"].get("pf", 0)),
    }
    for year in ["2022", "2023", "2024", "2025", "2026"]:
        try:
            r = backtest_year(year, trial["params"])
            for k, v in r.items():
                out[f"{year}_{k}"] = v
        except Exception as e:
            out[f"{year}_err"] = str(e)[:100]
    return out


def aggregate_features(row):
    """Compute per-trial aggregate features from per-year stats."""
    train_years = ["2022", "2023", "2024", "2025"]
    pnls = [row.get(f"{y}_pnl", 0) for y in train_years]
    pfs = [row.get(f"{y}_pf", 0) for y in train_years]
    ns = [row.get(f"{y}_n", 0) for y in train_years]

    feats = {}
    feats["sum_train_pnl"] = sum(pnls)
    feats["min_train_pnl"] = min(pnls)
    feats["max_train_pnl"] = max(pnls)
    feats["mean_train_pnl"] = statistics.mean(pnls)
    feats["min_train_pf"] = min(pfs)
    feats["mean_train_pf"] = statistics.mean(pfs)
    try:
        feats["std_train_pnl"] = statistics.stdev(pnls) if len(pnls) > 1 else 0
    except Exception:
        feats["std_train_pnl"] = 0
    feats["cv_train_pnl"] = (feats["std_train_pnl"] / abs(feats["mean_train_pnl"])
                              if feats["mean_train_pnl"] != 0 else 999)
    # Geometric mean of positive yearly PnLs (treat any nonpositive as 1 for geo-mean stability)
    pos = [max(p, 1.0) for p in pnls]
    feats["geomean_train_pnl"] = float(np.prod(pos) ** (1.0 / len(pos)))
    # 25th percentile yearly PnL
    feats["p25_train_pnl"] = float(np.percentile(pnls, 25))
    feats["min_train_n"] = min(ns)
    return feats


def correlate(rows, feature, target):
    """Pearson + Spearman of feature vs target across rows."""
    xs = np.array([r[feature] for r in rows], dtype=float)
    ys = np.array([r[target] for r in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]; ys = ys[mask]
    if len(xs) < 3:
        return None, None
    pearson = float(np.corrcoef(xs, ys)[0, 1])
    # Spearman via ranks
    def _rank(a):
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
    spearman = float(np.corrcoef(_rank(xs), _rank(ys))[0, 1])
    return pearson, spearman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=30)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    print(f"Pulling top {args.n} W7 trials from Postgres...")
    trials = fetch_top_n(args.n)
    print(f"Got {len(trials)}.  Train-score top 5:")
    for t in trials[:5]:
        print(f"  #{t['number']:<4}  train ${t['score']/1e6:.2f}M")

    print(f"\nFanning out across {args.workers} processes (5 yearly backtests per trial)...")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(per_trial_worker, t): t["number"] for t in trials}
        done = 0
        for fut in as_completed(futs):
            num = futs[fut]
            try:
                r = fut.result()
                rows.append(r)
                done += 1
                p22 = r.get("2022_pnl", 0); p23 = r.get("2023_pnl", 0)
                p24 = r.get("2024_pnl", 0); p25 = r.get("2025_pnl", 0)
                p26 = r.get("2026_pnl", 0)
                print(f"  [{done}/{len(trials)}] #{num:<4}  "
                      f"22 ${p22:>+8,.0f}  23 ${p23:>+8,.0f}  "
                      f"24 ${p24:>+8,.0f}  25 ${p25:>+8,.0f}  | 26 ${p26:>+8,.0f}")
            except Exception as e:
                print(f"  #{num}: FAILED ({str(e)[:80]})")

    # Compute aggregate features per row
    for r in rows:
        r.update(aggregate_features(r))

    # Save CSV
    csv_path = os.path.join(OUTDIR, "per_year_analysis.csv")
    keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # Correlation analysis vs 2026 PnL
    print(f"\n{'='*80}\n  Correlation of per-trial features vs 2026 PnL (n={len(rows)})\n{'='*80}")
    print(f"  {'feature':<25} {'pearson_r':>12} {'spearman_r':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    feats = [
        "sum_train_pnl",
        "min_train_pnl",
        "max_train_pnl",
        "mean_train_pnl",
        "min_train_pf",
        "mean_train_pf",
        "std_train_pnl",
        "cv_train_pnl",
        "geomean_train_pnl",
        "p25_train_pnl",
        "min_train_n",
        "train_score",
        "train_pnl_attr",
        "train_pf_attr",
    ]
    corrs = {}
    for f in feats:
        if any(f not in r for r in rows): continue
        p, s = correlate(rows, f, "2026_pnl")
        corrs[f] = {"pearson": p, "spearman": s}
        print(f"  {f:<25} {p:>12.3f} {s:>12.3f}" if p is not None else f"  {f:<25} {'n/a':>12} {'n/a':>12}")

    out_json = os.path.join(OUTDIR, "per_year_correlations.json")
    with open(out_json, "w") as f:
        json.dump({
            "n_trials": len(rows),
            "correlations_with_2026_pnl": corrs,
        }, f, indent=2)
    print(f"\nWrote {out_json}")

    # Show the table
    print(f"\n{'='*100}\n  Per-trial summary table (sorted by 2026 PnL)\n{'='*100}")
    rows.sort(key=lambda r: -r.get("2026_pnl", 0))
    print(f"  {'rank':<4} {'#':<5} {'train$':>10} {'2022$':>10} {'2023$':>10} {'2024$':>10} {'2025$':>10} | {'2026$':>10} {'min$':>10} {'minPF':>6}")
    for i, r in enumerate(rows):
        train = r["train_score"] / 1e6
        print(f"  {i+1:<4} #{r['number']:<4} ${train:>7.2f}M "
              f"${r.get('2022_pnl',0):>+8,.0f} ${r.get('2023_pnl',0):>+8,.0f} "
              f"${r.get('2024_pnl',0):>+8,.0f} ${r.get('2025_pnl',0):>+8,.0f} "
              f"| ${r.get('2026_pnl',0):>+8,.0f} ${r['min_train_pnl']:>+8,.0f} {r['min_train_pf']:>5.2f}")


if __name__ == "__main__":
    main()
