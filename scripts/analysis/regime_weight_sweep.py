"""Sweep w_today (today vs 5-day-rolling weight) and find the best for
the swap_635 strategy (trial #6 normal + trial #635 squeeze + skip DEAD)."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime, compute_features

DATA_DIRS = [
    "stored_data_2021", "stored_data_2022", "stored_data_2023",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data", "stored_data_mar_may_2026",
]
NORMAL_PARAMS_PATH = "config/trial_6_extracted.json"
SQUEEZE_PARAMS_PATH = "config/trial_635_extracted.json"
BASELINE_PATH = "config/trial_432_params.json"
STARTING_CASH = 25_000
ROLLING_WINDOW = 5


def _read_merged(path):
    with open(path) as f:
        p = json.load(f)
    if isinstance(p, dict) and "params" in p:
        p = p["params"]
    with open(BASELINE_PATH) as f:
        base = json.load(f)
    merged = dict(base); merged.update(p)
    return merged


def load_and_apply(path):
    set_strategy_params(_read_merged(path))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.VOL_CAP_PCT = 5.0


def run_swap_with_weight(daily_picks, dates, w_today):
    """Run swap_635 (trial #6 on normal / trial #635 on squeeze / skip dead)
    using `w_today` to blend today vs 5d-rolling features for regime classify."""
    cash = STARTING_CASH
    per_regime = defaultdict(lambda: {"days": 0, "pnl": 0.0})
    per_day_pnl = []
    rolling_buf = deque(maxlen=ROLLING_WINDOW)  # holds last N days' feature dicts
    current_loaded = None

    def _ensure(name):
        nonlocal current_loaded
        if current_loaded == name:
            return
        load_and_apply(SQUEEZE_PARAMS_PATH if name == "squeeze" else NORMAL_PARAMS_PATH)
        current_loaded = name

    for date in dates:
        picks_today = daily_picks.get(date, [])
        f_today = compute_features(picks_today)

        # Rolling 5d means (using buffer of prior days, exclusive of today)
        if rolling_buf:
            roll = {
                k: float(np.mean([f[k] for f in rolling_buf]))
                for k in ("n_above_20", "n_above_50", "max_gap", "med_pm_volume")
            }
        else:
            roll = None

        regime = classify_regime(picks_today, rolling_5d_features=roll, w_today=w_today)
        per_regime[regime]["days"] += 1

        # Update buffer AFTER classification
        rolling_buf.append({k: f_today[k] for k in
                             ("n_above_20", "n_above_50", "max_gap", "med_pm_volume")})

        if regime == "dead":
            per_day_pnl.append(0.0)
            continue

        _ensure("squeeze" if regime == "squeeze" else "normal")

        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, ending_cash, unsettled, _ = tgc.simulate_day_combined(
                picks_today, cash, cash_account=is_cash
            )
        except Exception:
            per_day_pnl.append(0.0)
            continue

        day_pnl = ending_cash - cash
        cash = ending_cash
        if is_cash:
            cash += unsettled
        per_regime[regime]["pnl"] += day_pnl
        per_day_pnl.append(day_pnl)

    pnls = np.array([p for p in per_day_pnl if p != 0.0])
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0.0

    return {
        "w_today": w_today,
        "final_equity": cash,
        "total_pnl": cash - STARTING_CASH,
        "sharpe": sharpe,
        "n_traded_days": sum(1 for d in per_day_pnl if d != 0),
        "n_dead_days": per_regime["dead"]["days"],
        "n_squeeze_days": per_regime["squeeze"]["days"],
        "n_normal_days": per_regime["normal"]["days"],
        "squeeze_pnl": per_regime["squeeze"]["pnl"],
        "normal_pnl": per_regime["normal"]["pnl"],
    }


def main():
    print("=" * 80)
    print("REGIME WEIGHT SWEEP: trial #6 normal + trial #635 squeeze + skip dead")
    print(f"Variable: w_today (weight on today's features vs {ROLLING_WINDOW}-day rolling)")
    print("=" * 80)

    print("\nLoading data...")
    dates, daily_picks = load_all_picks(DATA_DIRS)
    print(f"  {len(dates)} trading days: {dates[0]} -> {dates[-1]}")

    weights = [0.00, 0.25, 0.50, 0.70, 0.85, 1.00]
    results = []
    for w in weights:
        print(f"\n--- Running w_today = {w:.2f} ---")
        r = run_swap_with_weight(daily_picks, dates, w)
        results.append(r)
        print(f"  -> Final ${r['final_equity']:,.0f}  Sharpe {r['sharpe']:.2f}  "
              f"squeeze {r['n_squeeze_days']} days / normal {r['n_normal_days']} / dead {r['n_dead_days']}")

    print("\n" + "=" * 80)
    print("SWEEP RESULTS (sorted by total PnL)")
    print("=" * 80)
    results.sort(key=lambda x: -x["total_pnl"])
    print(f"  {'w_today':>8} {'Final equity':>16} {'Sharpe':>8} {'dead':>5} {'normal':>7} "
          f"{'squeeze':>8} {'squeeze PnL':>16}")
    for r in results:
        marker = "  <-- BEST" if r == results[0] else ""
        print(f"  {r['w_today']:>8.2f} ${r['final_equity']:>15,.0f} {r['sharpe']:>8.2f} "
              f"{r['n_dead_days']:>5} {r['n_normal_days']:>7} {r['n_squeeze_days']:>8} "
              f"${r['squeeze_pnl']:>14,.0f}{marker}")

    # Save details
    out_dir = "results/regime_gate"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(results).to_csv(os.path.join(out_dir, f"weight_sweep_{ts}.csv"), index=False)
    print(f"\nSaved to {out_dir}/weight_sweep_{ts}.csv")


if __name__ == "__main__":
    main()
