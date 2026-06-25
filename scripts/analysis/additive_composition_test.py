"""Additive composition test (ChatGPT's recommended pre-W10 experiment).

Tests these specific configs built from #254's parameters:
  - G only
  - G + L
  - G + V
  - G + V + L
  - #254 BASE (G+A+V+R+L) — reference

Each forward-tested on 2022, 2023, 2024, 2025, 2026 INDEPENDENTLY (fresh $25K).
Per-strategy attribution captured.

Goal: determine whether
  (a) G alone captures the alpha (W10 = G-only)
  (b) G+L gives consistent multi-year benefit (W10 = G+L)
  (c) G+V provides real diversification (W10 = G+V)
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
SOURCE_CONFIG = "config/trial_254_w7_extracted.json"
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


def make_composition_params(base_params, enabled):
    p = dict(base_params)
    for s in ALL_STRATS:
        p[f"enable_{s}"] = (s in enabled)
    return p


def backtest_year(task):
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
    daily_eq = [cash]
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
                strat = st.get("strategy", "?")
                by_strat[strat]["n"] += 1
                by_strat[strat]["pnl"] += st["pnl"]
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)

    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd = float((eq - peak).min() / peak[(eq - peak).argmin()] * 100) if len(peak) and peak[(eq - peak).argmin()] > 0 else 0
    return {
        "label": task["label"],
        "year": year,
        "pnl": float(cash - STARTING_CASH),
        "final": float(cash),
        "n_trades": n_trades,
        "dd_pct": dd,
        "per_strategy": {k: {"n": v["n"], "pnl": float(v["pnl"])} for k, v in by_strat.items()},
    }


def main():
    with open(SOURCE_CONFIG) as f: data = json.load(f)
    src_params = data.get("params", data)
    src_enabled = [s for s in ALL_STRATS if src_params.get(f"enable_{s}") is True]
    print(f"  Source: #254 enables {','.join(s.upper() for s in src_enabled)}")

    compositions = [
        ("BASE (G+A+V+R+L)", set(src_enabled)),  # reference
        ("G only",       {"g"}),
        ("G + L",        {"g", "l"}),
        ("G + V",        {"g", "v"}),
        ("G + V + L",    {"g", "v", "l"}),
        ("G + V + A",    {"g", "v", "a"}),
        ("G + V + L + A",{"g", "v", "l", "a"}),  # everything except R
    ]

    tasks = []
    for label, enabled in compositions:
        params = make_composition_params(src_params, enabled)
        for year in ["2022", "2023", "2024", "2025", "2026"]:
            tasks.append({"label": label, "year": year, "params": params,
                           "enabled": sorted(enabled)})

    print(f"\n  {len(tasks)} tasks ({len(compositions)} compositions × 5 years)")
    print(f"  Running in parallel (6 workers)...\n")

    results = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(backtest_year, t): (t["label"], t["year"]) for t in tasks}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                done += 1
                print(f"  [{done:>2}/{len(tasks)}] {r['label']:<22} {r['year']}  ${r['pnl']:>+11,.0f}  n={r['n_trades']:>4}  DD {r['dd_pct']:>5.1f}%")
            except Exception as e:
                print(f"  failed: {e}")

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = f"{OUTDIR}/additive_composition_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved {out_path}")

    # Pivot: composition × year
    print(f"\n\n{'='*92}")
    print(f"  PER-YEAR PnL BY COMPOSITION (fresh $25K each year)")
    print(f"{'='*92}")
    comp_labels = [c[0] for c in compositions]
    print(f"  {'config':<22} {'2022':>11} {'2023':>11} {'2024':>11} {'2025':>11} {'2026':>11} {'TOTAL':>13}")
    print(f"  {'-'*22} {'-'*11} {'-'*11} {'-'*11} {'-'*11} {'-'*11} {'-'*13}")
    for label in comp_labels:
        row = f"  {label:<22}"
        total = 0
        for y in ["2022", "2023", "2024", "2025", "2026"]:
            v = next((r["pnl"] for r in results if r["label"] == label and r["year"] == y), None)
            if v is not None:
                row += f" ${v:>+9,.0f}"
                total += v
            else:
                row += "        n/a"
        row += f"  ${total:>+11,.0f}"
        print(row)

    # Delta vs BASE per year
    print(f"\n  DELTA vs BASE (positive = composition WINS that year vs full #254):")
    print(f"  {'config':<22} {'2022':>11} {'2023':>11} {'2024':>11} {'2025':>11} {'2026':>11} {'TOTAL':>13}")
    print(f"  {'-'*22} {'-'*11} {'-'*11} {'-'*11} {'-'*11} {'-'*11} {'-'*13}")
    for label in comp_labels:
        if label == comp_labels[0]: continue
        row = f"  {label:<22}"
        total = 0
        for y in ["2022", "2023", "2024", "2025", "2026"]:
            base = next((r["pnl"] for r in results if r["label"] == comp_labels[0] and r["year"] == y), None)
            this = next((r["pnl"] for r in results if r["label"] == label and r["year"] == y), None)
            if base is not None and this is not None:
                delta = this - base
                row += f" ${delta:>+9,.0f}"
                total += delta
            else:
                row += "        n/a"
        row += f"  ${total:>+11,.0f}"
        print(row)

    # Total ranking
    print(f"\n  TOTAL PnL (sum of all 5 years), sorted best-first:")
    totals = []
    for label in comp_labels:
        total = sum(r["pnl"] for r in results if r["label"] == label)
        totals.append((label, total))
    totals.sort(key=lambda x: -x[1])
    for label, total in totals:
        print(f"    {label:<22}  ${total:>+12,.0f}")


if __name__ == "__main__":
    main()
