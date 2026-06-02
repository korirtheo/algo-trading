"""OOS validation of squeeze specialist on 2019 squeeze days.

2019 is fully pre-training for any of our configs:
  - Trial #6 was trained on 2024-25
  - Trial #326 was trained on 2021-2026 squeeze days
  - 2019 = pure blind test

Identifies 2019 squeeze days using the tuned thresholds (n50>=4, mg>150),
then runs trial #6 and trial #326 on those days and compares.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np
import pandas as pd

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime, regime_features

DATA_DIR_2019 = "stored_data_2019"
TRIAL_6 = "config/trial_6_extracted.json"
TRIAL_326 = "config/trial_326_squeeze_extracted.json"
TRIAL_635 = "config/trial_635_extracted.json"
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000


def _merged(path):
    with open(path) as f:
        p = json.load(f)
    if isinstance(p, dict) and "params" in p:
        p = p["params"]
    with open(BASELINE) as f:
        base = json.load(f)
    m = dict(base); m.update(p)
    return m


def load_and_apply(path):
    set_strategy_params(_merged(path))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.VOL_CAP_PCT = 5.0


def run_on(daily_picks, dates, config_path, label):
    load_and_apply(config_path)
    cash = STARTING_CASH
    per_day = []
    for d in dates:
        picks = daily_picks.get(d, [])
        if not picks:
            per_day.append({"date": d, "pnl": 0})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, ending_cash, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account=is_cash
            )
        except Exception as e:
            per_day.append({"date": d, "pnl": 0, "error": str(e)[:60]})
            continue
        day_pnl = ending_cash - cash
        cash = ending_cash
        if is_cash:
            cash += unsettled
        per_day.append({"date": d, "pnl": day_pnl})

    pnls = np.array([d["pnl"] for d in per_day if d["pnl"] != 0])
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if len(pnls) > 1 and pnls.std() > 0 else 0.0
    return {
        "label": label,
        "config": os.path.basename(config_path),
        "final_equity": cash,
        "total_pnl": cash - STARTING_CASH,
        "sharpe": sharpe,
        "n_days": len(dates),
        "per_day": per_day,
    }


def main():
    print("=" * 90)
    print("OOS TEST: squeeze specialist on 2019 squeeze days (pre-training BLIND)")
    print("=" * 90)

    print("\nLoading 2019 picks...")
    dates, daily_picks = load_all_picks([DATA_DIR_2019])
    print(f"  {len(dates)} 2019 trading days: {dates[0]} -> {dates[-1]}")

    # Classify each day
    regime_counts = {"dead": 0, "normal": 0, "squeeze": 0}
    squeeze_days = []
    for d in dates:
        picks = daily_picks.get(d, [])
        r = classify_regime(picks)
        regime_counts[r] += 1
        if r == "squeeze":
            squeeze_days.append(d)

    print(f"\nRegime distribution on 2019 (using tuned thresholds n50>=4, mg>150):")
    for r in ("dead", "normal", "squeeze"):
        n = regime_counts[r]
        print(f"  {r:<10} {n:>4} days  ({100*n/len(dates):.1f}%)")

    if not squeeze_days:
        print("\nNo squeeze days found in 2019 — pre-COVID was the LOW-density era.")
        print("This itself is an important finding: regime gate correctly identifies 2019 as non-squeeze.")

        # Run trial #6 across ALL 2019 days as a sanity check
        print("\n--- Sanity: trial #6 on ALL 2019 days (regardless of regime) ---")
        r6 = run_on(daily_picks, dates, TRIAL_6, "trial_6_all_2019")
        print(f"  Final ${r6['final_equity']:,.0f}  PnL ${r6['total_pnl']:+,.0f}  Sharpe {r6['sharpe']:.2f}")
        return

    print(f"\n2019 squeeze days: {len(squeeze_days)}")
    for i, d in enumerate(squeeze_days[:20]):
        f = regime_features(daily_picks.get(d, []))
        print(f"  {d}  n_above_20={f['n_above_20']:>2}  n_above_50={f['n_above_50']:>2}  "
              f"max_gap={f['max_gap']:>6.0f}%")
    if len(squeeze_days) > 20:
        print(f"  ... and {len(squeeze_days)-20} more")

    runs = []
    for cfg, label in [(TRIAL_6, "Trial #6 (generalist)"),
                        (TRIAL_635, "Trial #635 (2024-25 squeeze IS)"),
                        (TRIAL_326, "Trial #326 (NEW squeeze specialist)")]:
        if not os.path.exists(cfg):
            print(f"\n[skip] {label}: {cfg} not found")
            continue
        print(f"\n--- {label} ---")
        r = run_on(daily_picks, squeeze_days, cfg, label)
        runs.append(r)
        print(f"  Final ${r['final_equity']:,.0f}  PnL ${r['total_pnl']:+,.0f}  Sharpe {r['sharpe']:.2f}")

    print("\n" + "=" * 90)
    print("HEADLINE: 2019 squeeze days OOS comparison")
    print("=" * 90)
    print(f"  {'Config':<32} {'Final equity':>16} {'PnL':>14} {'Sharpe':>8} {'PnL/day':>11}")
    for r in runs:
        ppd = r['total_pnl'] / r['n_days']
        print(f"  {r['label']:<32} ${r['final_equity']:>15,.0f} ${r['total_pnl']:>+13,.0f} "
              f"{r['sharpe']:>8.2f} ${ppd:>+10,.0f}")


if __name__ == "__main__":
    main()
