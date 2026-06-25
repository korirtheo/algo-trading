"""Apples-to-apples comparison of W13, W21b #511, #561, #870, W21c #579.

Runs each config on:
  - 2022 (cool small-cap regime, 251 days)
  - 2024 (FULL, all quarter dirs included)
  - 2025 (FULL)
  - 2026 Mar-Jun (true held-out OOS, 74 days)

Reports:
  - Per-year pnl, pf, wr, n
  - Composite (worst-year score = pnl * min(pf,3))
  - Held-out 2026 Mar-Jun fwd
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W13_DEPLOY = "config/trial_w13_1202_deploy.json"
ALL_STRATS = ["h", "g", "a", "f", "d", "v", "p", "m", "r", "w", "o", "b", "k", "c", "s", "e", "i", "j", "n", "l", "x"]
EXIT_PARAMS = ["g_target_pct", "g_stop_pct", "g_time_limit_min", "g_trail_pct", "g_trail_activate_pct"]

WINDOWS = [
    {"name": "2022", "dirs": ["stored_data_2022"],
     "date_lo": "2022-01-01", "date_hi": "2022-12-31"},
    {"name": "2024", "dirs": ["stored_data_combined", "stored_data_jan_mar_2024",
                              "stored_data_apr_jun_2024", "stored_data_jul_sep_2024",
                              "stored_data_oct_dec_2024"],
     "date_lo": "2024-01-01", "date_hi": "2024-12-31"},
    {"name": "2025", "dirs": ["stored_data_combined", "stored_data_jan_mar_2025",
                              "stored_data_apr_jun_2025", "stored_data_jul_2025",
                              "stored_data_oos"],
     "date_lo": "2025-01-01", "date_hi": "2025-12-31"},
    {"name": "2026_Mar_Jun_OOS", "dirs": ["stored_data", "stored_data_mar_may_2026",
                                          "stored_data_jun_2026", "stored_data_2026_gap_fill"],
     "date_lo": "2026-03-01", "date_hi": "2026-06-24"},
]

CONFIGS = [
    {"name": "W13_#1202", "exits": {"g_target_pct": 11.0, "g_stop_pct": 12.0,
        "g_time_limit_min": 24, "g_trail_pct": 1.0, "g_trail_activate_pct": 0.0}},
    {"name": "W21_#19", "exits": {"g_target_pct": 40.0, "g_stop_pct": 19.0,
        "g_time_limit_min": 21, "g_trail_pct": 1.0, "g_trail_activate_pct": 0.0}},
    {"name": "W21b_#511_DEPLOYED", "exits": {"g_target_pct": 62.0, "g_stop_pct": 25.0,
        "g_time_limit_min": 12, "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}},
    {"name": "W21b_#561", "exits": {"g_target_pct": 64.0, "g_stop_pct": 24.0,
        "g_time_limit_min": 9, "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}},
    {"name": "W21b_#870", "exits": {"g_target_pct": 85.0, "g_stop_pct": 24.0,
        "g_time_limit_min": 33, "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}},
    {"name": "W21c_#579_NEW", "exits": {"g_target_pct": 34.0, "g_stop_pct": 12.0,
        "g_time_limit_min": 33, "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}},
]


def run_one(args):
    config_name, window_name, exit_params, window_dirs, date_lo, date_hi = args
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
    all_trades = []
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
                all_trades.append({"pnl": st["pnl"], "strategy": st.get("strategy")})
        cash = end_c + (unset if is_cash else 0)

    n = len(all_trades)
    if n == 0:
        return {"config": config_name, "window": window_name, "n": 0, "pnl": 0, "pf": 0, "wr": 0, "equity": cash, "n_days": len(dates)}
    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
    losses = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0))
    pf = wins / losses if losses > 0 else 99.0
    wr = sum(1 for t in all_trades if t["pnl"] > 0) / n * 100
    return {"config": config_name, "window": window_name, "n": n, "pnl": total_pnl, "pf": pf, "wr": wr, "equity": cash, "n_days": len(dates)}


def main():
    jobs = []
    for c in CONFIGS:
        for w in WINDOWS:
            jobs.append((c["name"], w["name"], c["exits"], w["dirs"], w["date_lo"], w["date_hi"]))

    print(f"Running {len(jobs)} backtest jobs ({len(CONFIGS)} configs x {len(WINDOWS)} windows)")
    print(f"Configs: {[c['name'] for c in CONFIGS]}")
    print(f"Windows: {[w['name'] for w in WINDOWS]}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_one, j): (j[0], j[1]) for j in jobs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                print(f"  {r['config']:<22} {r['window']:<22}  n={r['n']:>5}  pnl=${r['pnl']:>13,.0f}  pf={r['pf']:>5.2f}  wr={r['wr']:>5.1f}%  ({r['n_days']} days)")
            except Exception as e:
                cn, wn = futs[fut]
                print(f"  FAIL {cn} / {wn}: {e}")

    # Build comparison table
    by_config = {}
    for r in results:
        by_config.setdefault(r["config"], {})[r["window"]] = r

    print()
    print("=" * 110)
    print(f"  APPLES-TO-APPLES COMPARISON")
    print("=" * 110)
    print()
    train_windows = ["2022", "2024", "2025"]
    print(f"{'Config':<24} | {'2022 pnl':>11} {'2024 pnl':>11} {'2025 pnl':>11} | {'COMPOSITE':>12} | {'2026 OOS':>11}")
    print("-" * 110)
    for c in CONFIGS:
        row = by_config.get(c["name"], {})
        scores = []
        for w in train_windows:
            r = row.get(w, {})
            pnl = r.get("pnl", 0); pf = r.get("pf", 0)
            if pf >= 0.5 and pnl > 0:
                scores.append(pnl * min(pf, 3.0))
            else:
                scores.append(-9999)
        composite = min(scores)
        oos = row.get("2026_Mar_Jun_OOS", {}).get("pnl", 0)
        print(f"{c['name']:<24} | ${row.get('2022',{}).get('pnl',0):>10,.0f} ${row.get('2024',{}).get('pnl',0):>10,.0f} ${row.get('2025',{}).get('pnl',0):>10,.0f} | ${composite:>11,.0f} | ${oos:>10,.0f}")

    out_path = "results/w21_versions_apples_to_apples.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "configs": CONFIGS, "windows": WINDOWS}, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
