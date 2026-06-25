"""Multi-year ablation — measure G, V, A, L (and others) contribution
per year on #254 and #124. Each year backtested with fresh $25K.

Answers ChatGPT's Phase 1 question: is L always bad, or just bad in 2026?
Is G always dominant? Is V always positive?
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
OUTDIR = "results"

YEAR_DIRS = {
    "2022": ["stored_data_2022"],
    "2023": ["stored_data_2023"],
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"],
}

ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]


def enabled_strats(params):
    return [s for s in ALL_STRATS if params.get(f"enable_{s}") is True]


def make_ablation_params(base_params, disable=(), enable_only=None):
    p = dict(base_params)
    if enable_only is not None:
        for s in ALL_STRATS:
            p[f"enable_{s}"] = (s in enable_only)
    else:
        for s in disable:
            p[f"enable_{s}"] = False
    return p


def backtest_year(task):
    """task = {"label": str, "params": dict, "year": str}
    Single-year backtest with fresh $25K. Per-strategy attribution included.
    """
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(task["params"])
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0

    year = task["year"]
    dirs = [d for d in YEAR_DIRS[year] if os.path.exists(d)]
    if not dirs:
        return {**task, "pnl": 0, "n_trades": 0, "per_strategy": {}}
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith(year)])

    cash = STARTING_CASH
    n_trades = 0
    by_strat = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks: continue
        if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as nfp
            day_picks = nfp(day_picks, d, tgc.NEWS_MIN_ARTICLES, tgc.NEWS_REQUIRE_CATALYST)
        if not day_picks: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_trades += 1
                strat = st.get("strategy", "?")
                by_strat[strat]["n"] += 1
                by_strat[strat]["pnl"] += st["pnl"]
        cash = end_c + (unset if is_cash else 0)
    return {
        "label": task["label"],
        "year": year,
        "ablation": task.get("ablation", "BASE"),
        "pnl": float(cash - STARTING_CASH),
        "final": float(cash),
        "n_trades": n_trades,
        "per_strategy": {k: {"n": v["n"], "pnl": float(v["pnl"])} for k, v in by_strat.items()},
    }


def build_tasks(label, params, ablations):
    """For each year and each ablation, build a task."""
    tasks = []
    for year in ["2022", "2023", "2024", "2025", "2026"]:
        # BASE
        tasks.append({"label": label, "year": year, "ablation": "BASE",
                       "params": dict(params)})
        # Each ablation
        for ablation_label, ablation_disable in ablations:
            new_params = make_ablation_params(params, disable=ablation_disable)
            tasks.append({"label": label, "year": year,
                          "ablation": ablation_label,
                          "params": new_params})
    return tasks


def main():
    configs = [
        {
            "label": "#254",
            "path": "config/trial_254_w7_extracted.json",
            "ablations": [
                ("-G", ["g"]),
                ("-A", ["a"]),
                ("-V", ["v"]),
                ("-R", ["r"]),
                ("-L", ["l"]),
                ("-L-R", ["l", "r"]),
            ],
        },
        {
            "label": "#124",
            "path": "config/trial_124_microcap_pump_extracted.json",
            "ablations": [
                ("-H", ["h"]),
                ("-A", ["a"]),
                ("-C", ["c"]),
                ("-S", ["s"]),
                ("-I", ["i"]),
            ],
        },
    ]

    all_tasks = []
    for cfg in configs:
        if not os.path.exists(cfg["path"]): continue
        with open(cfg["path"]) as f: data = json.load(f)
        params = data.get("params", data)
        print(f"  {cfg['label']} enables: {','.join(s.upper() for s in enabled_strats(params))}")
        all_tasks.extend(build_tasks(cfg["label"], params, cfg["ablations"]))

    print(f"\n  Total tasks: {len(all_tasks)} (config × year × ablation)")
    print(f"  Running in parallel (6 workers)...\n")

    results = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(backtest_year, t): (t["label"], t["year"], t["ablation"]) for t in all_tasks}
        done = 0
        for fut in as_completed(futs):
            label, year, ablation = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                done += 1
                print(f"  [{done:>2}/{len(all_tasks)}] {label} {ablation:<6} {year}  pnl ${r['pnl']:>+9,.0f}  n={r['n_trades']:>4}")
            except Exception as e:
                print(f"  failed {label} {year} {ablation}: {e}")

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = f"{OUTDIR}/multi_year_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved {out_path}")

    # Build matrix: per config, per ablation, PnL by year
    print(f"\n\n{'='*92}")
    print(f"  PER-YEAR PnL BY ABLATION ($ fresh $25K each year)")
    print(f"{'='*92}")

    for cfg in configs:
        if not os.path.exists(cfg["path"]): continue
        label = cfg["label"]
        ablations = ["BASE"] + [a[0] for a in cfg["ablations"]]
        cfg_results = [r for r in results if r["label"] == label]

        print(f"\n  {label}:")
        print(f"  {'ablation':<10} {'2022':>10} {'2023':>10} {'2024':>10} {'2025':>10} {'2026':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for ab in ablations:
            row = f"  {ab:<10}"
            for y in ["2022", "2023", "2024", "2025", "2026"]:
                v = next((r["pnl"] for r in cfg_results if r["year"] == y and r["ablation"] == ab), None)
                row += f" ${v:>+8,.0f}" if v is not None else "       n/a"
            print(row)

        # Contribution = BASE - ablation
        print(f"\n  Contribution (BASE - ablation) — positive = strategy HELPS that year:")
        print(f"  {'strategy':<10} {'2022':>10} {'2023':>10} {'2024':>10} {'2025':>10} {'2026':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for ab_label, disabled in cfg["ablations"]:
            if len(disabled) > 1: continue  # skip combo ablations for contribution
            s = disabled[0].upper()
            row = f"  {s:<10}"
            for y in ["2022", "2023", "2024", "2025", "2026"]:
                base = next((r["pnl"] for r in cfg_results if r["year"] == y and r["ablation"] == "BASE"), None)
                ablated = next((r["pnl"] for r in cfg_results if r["year"] == y and r["ablation"] == ab_label), None)
                if base is not None and ablated is not None:
                    contrib = base - ablated
                    row += f" ${contrib:>+8,.0f}"
                else:
                    row += "       n/a"
            print(row)


if __name__ == "__main__":
    main()
