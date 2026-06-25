"""OOS validation of squeeze specialist #541 (PF 2.731) on 2019 + 2020.

Trial #541 was trained on the 2021-2026 squeeze-day subset. 2019 and 2020
are fully pre-training (pure blind years). Compares against:
  - Trial #6: pre-regime generalist (trained on 2024-25)
  - Trial #326: prior squeeze specialist (PF 2.085)
  - Trial #635: 2024-25 squeeze IS-overfit reference
  - Trial #541: NEW squeeze specialist (PF 2.731)

Classifies each day with the tuned thresholds (n50>=4, mg>150), then runs
each config on the squeeze days only. Prints a single comparison table
for each year.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime, regime_features

TRIAL_6   = "config/trial_6_extracted.json"
TRIAL_326 = "config/trial_326_squeeze_extracted.json"
TRIAL_541 = "config/trial_541_squeeze_extracted.json"
TRIAL_635 = "config/trial_635_extracted.json"
BASELINE  = "config/trial_432_params.json"
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
        "final_equity": cash,
        "total_pnl": cash - STARTING_CASH,
        "sharpe": sharpe,
        "n_days": len(dates),
        "per_day": per_day,
    }


def year_block(data_dir, year_label):
    print("=" * 96)
    print(f"OOS TEST: squeeze specialists on {year_label} squeeze days (BLIND)")
    print("=" * 96)

    if not os.path.exists(data_dir):
        print(f"[skip] {data_dir} not found")
        return

    print(f"\nLoading {data_dir} picks...")
    dates, daily_picks = load_all_picks([data_dir])
    if not dates:
        print(f"  No dates loaded — check {data_dir}/")
        return
    print(f"  {len(dates)} trading days: {dates[0]} -> {dates[-1]}")

    regime_counts = {"dead": 0, "normal": 0, "squeeze": 0}
    squeeze_days = []
    for d in dates:
        picks = daily_picks.get(d, [])
        r = classify_regime(picks)
        regime_counts[r] += 1
        if r == "squeeze":
            squeeze_days.append(d)

    print(f"\nRegime distribution (tuned n50>=4, mg>150):")
    for r in ("dead", "normal", "squeeze"):
        n = regime_counts[r]
        print(f"  {r:<10} {n:>4} days  ({100*n/len(dates):.1f}%)")

    if not squeeze_days:
        print(f"\nNo squeeze days in {year_label} — regime gate would skip the whole year.")
        return

    print(f"\n{year_label} squeeze days: {len(squeeze_days)}")
    for d in squeeze_days[:15]:
        f = regime_features(daily_picks.get(d, []))
        print(f"  {d}  n_above_20={f['n_above_20']:>2}  n_above_50={f['n_above_50']:>2}  "
              f"max_gap={f['max_gap']:>6.0f}%")
    if len(squeeze_days) > 15:
        print(f"  ... and {len(squeeze_days)-15} more")

    runs = []
    for cfg, label in [(TRIAL_6,   "#6   (generalist)"),
                        (TRIAL_326, "#326 (old squeeze PF 2.09)"),
                        (TRIAL_635, "#635 (2024-25 squeeze IS)"),
                        (TRIAL_541, "#541 (NEW squeeze PF 2.73)")]:
        if not os.path.exists(cfg):
            print(f"\n[skip] {label}: {cfg} not found")
            continue
        print(f"\n--- {label} ---")
        r = run_on(daily_picks, squeeze_days, cfg, label)
        runs.append(r)
        print(f"  Final ${r['final_equity']:,.0f}  PnL ${r['total_pnl']:+,.0f}  "
              f"Sharpe {r['sharpe']:.2f}")

    print(f"\n{'-' * 96}")
    print(f"HEADLINE: {year_label} squeeze-day OOS comparison ({len(squeeze_days)} days)")
    print('-' * 96)
    print(f"  {'Config':<32} {'Final equity':>16} {'PnL':>14} {'Sharpe':>8} {'PnL/day':>11}")
    for r in runs:
        ppd = r['total_pnl'] / r['n_days']
        print(f"  {r['label']:<32} ${r['final_equity']:>15,.0f} ${r['total_pnl']:>+13,.0f} "
              f"{r['sharpe']:>8.2f} ${ppd:>+10,.0f}")
    print()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="2019,2020",
                   help="comma list, e.g. 2019,2020")
    args = p.parse_args()
    for y in args.years.split(","):
        y = y.strip()
        year_block(f"stored_data_{y}", y)


if __name__ == "__main__":
    main()
