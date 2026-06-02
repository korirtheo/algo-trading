"""Test swap_635 across multiple threshold settings to find the sweet spot.

Each setting runs the full 2021-2026 backtest with:
  - skip DEAD days
  - trial #6 on NORMAL days
  - trial #635 on SQUEEZE days

We override the module-level thresholds in strategies.regime_gate so the
classifier uses different cutoffs each run.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
import strategies.regime_gate as rg

DATA_DIRS = [
    "stored_data_2021", "stored_data_2022", "stored_data_2023",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data", "stored_data_mar_may_2026",
]
TRIAL_6 = "config/trial_6_extracted.json"
TRIAL_635 = "config/trial_635_extracted.json"
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000


def _read_merged(path):
    with open(path) as f:
        p = json.load(f)
    if isinstance(p, dict) and "params" in p:
        p = p["params"]
    with open(BASELINE) as f:
        base = json.load(f)
    merged = dict(base); merged.update(p)
    return merged


def load_and_apply(path):
    set_strategy_params(_read_merged(path))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.VOL_CAP_PCT = 5.0


def run_swap(daily_picks, dates):
    """Run swap (trial #6 normal / trial #635 squeeze / skip dead).
    Uses whatever thresholds are currently in rg module."""
    cash = STARTING_CASH
    per_day = []
    per_regime = defaultdict(lambda: {"days": 0, "pnl": 0.0})
    current = None

    def _ensure(name):
        nonlocal current
        if current == name:
            return
        load_and_apply(TRIAL_635 if name == "squeeze" else TRIAL_6)
        current = name

    for date in dates:
        picks = daily_picks.get(date, [])
        regime = rg.classify_regime(picks)
        per_regime[regime]["days"] += 1

        if regime == "dead":
            per_day.append({"date": date, "regime": regime, "pnl": 0.0})
            continue

        _ensure("squeeze" if regime == "squeeze" else "normal")

        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, ending_cash, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account=is_cash
            )
        except Exception:
            per_day.append({"date": date, "regime": regime, "pnl": 0.0})
            continue

        day_pnl = ending_cash - cash
        cash = ending_cash
        if is_cash:
            cash += unsettled
        per_regime[regime]["pnl"] += day_pnl
        per_day.append({"date": date, "regime": regime, "pnl": day_pnl})

    pnls = np.array([d["pnl"] for d in per_day if d["pnl"] != 0])
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if len(pnls) > 1 and pnls.std() > 0 else 0.0
    return {
        "final_equity": cash,
        "total_pnl": cash - STARTING_CASH,
        "sharpe": sharpe,
        "per_regime": dict(per_regime),
        "per_day": per_day,
    }


def main():
    print("=" * 95)
    print("REGIME THRESHOLD SWAP TEST")
    print("Comparing swap_635 (skip DEAD / trial #6 NORMAL / trial #635 SQUEEZE)")
    print("across 4 threshold settings to find the sweet spot")
    print("=" * 95)

    print("\nLoading picks...")
    dates, daily_picks = load_all_picks(DATA_DIRS)
    print(f"  {len(dates)} days")

    # Threshold configurations to test
    # (label, dead_n20, dead_max_gap, dead_med_vol, squeeze_n50, squeeze_max_gap)
    configs = [
        ("Current (s_n50>=3, mg>100)",         2, 30.0, 500_000.0, 3, 100.0),
        ("Medium (s_n50>=4, mg>150)",          2, 30.0, 500_000.0, 4, 150.0),
        ("Tight (s_n50>=5, mg>200)",           2, 30.0, 500_000.0, 5, 200.0),
        ("Very tight (s_n50>=6, mg>250)",      2, 30.0, 500_000.0, 6, 250.0),
    ]

    runs = []
    for label, d_n20, d_mg, d_mv, s_n50, s_mg in configs:
        # Override module constants
        rg.DEAD_N20_MAX = d_n20
        rg.DEAD_MAX_GAP_MAX = d_mg
        rg.DEAD_MED_PM_VOL_MAX = d_mv
        rg.SQUEEZE_N50_MIN = s_n50
        rg.SQUEEZE_MAX_GAP_MIN = s_mg

        print(f"\n--- {label} ---")
        r = run_swap(daily_picks, dates)
        r["label"] = label
        runs.append(r)
        per_r = r["per_regime"]
        print(f"  Final ${r['final_equity']:,.0f}  Sharpe {r['sharpe']:.2f}  "
              f"dead {per_r['dead']['days']} ({per_r['dead']['pnl']:.0f}$) / "
              f"normal {per_r['normal']['days']} ({per_r['normal']['pnl']:.0f}$) / "
              f"squeeze {per_r['squeeze']['days']} ({per_r['squeeze']['pnl']:.0f}$)")

    print("\n" + "=" * 95)
    print("RESULTS (sorted by Sharpe)")
    print("=" * 95)
    runs.sort(key=lambda x: -x["sharpe"])
    print(f"  {'Setting':<32} {'Final equity':>16} {'PnL':>14} {'Sharpe':>8} "
          f"{'dead':>5} {'normal':>7} {'squeeze':>8} {'squeeze PnL':>14}")
    for r in runs:
        pr = r["per_regime"]
        marker = "  <- BEST" if r == runs[0] else ""
        print(f"  {r['label']:<32} ${r['final_equity']:>15,.0f} ${r['total_pnl']:>13,.0f} "
              f"{r['sharpe']:>8.2f} {pr['dead']['days']:>5} {pr['normal']['days']:>7} "
              f"{pr['squeeze']['days']:>8} ${pr['squeeze']['pnl']:>12,.0f}{marker}")

    print("\nPer-regime PnL/day (lower better for dead, higher better for squeeze):")
    print(f"  {'Setting':<32} {'dead $/day':>12} {'normal $/day':>14} {'squeeze $/day':>15}")
    for r in runs:
        pr = r["per_regime"]
        d_ppd = pr['dead']['pnl'] / max(1, pr['dead']['days'])
        n_ppd = pr['normal']['pnl'] / max(1, pr['normal']['days'])
        s_ppd = pr['squeeze']['pnl'] / max(1, pr['squeeze']['days'])
        print(f"  {r['label']:<32} ${d_ppd:>11,.0f} ${n_ppd:>13,.0f} ${s_ppd:>14,.0f}")

    out_dir = "results/regime_gate"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    flat = []
    for r in runs:
        for d in r["per_day"]:
            flat.append({"label": r["label"], **d})
    pd.DataFrame(flat).to_csv(os.path.join(out_dir, f"threshold_swap_{ts}.csv"), index=False)
    print(f"\nSaved per-day to {out_dir}/threshold_swap_{ts}.csv")


if __name__ == "__main__":
    main()
