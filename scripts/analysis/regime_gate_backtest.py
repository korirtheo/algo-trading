"""
Backtest comparison: trial #6 with vs without the regime gate over 2021-2026.

For each trading day:
  1. Classify regime from that day's pre-market scan output (picks pickle)
  2. With gate ON: skip day if regime == "dead"; trade normally otherwise
  3. With gate OFF: trade every day (baseline)

Then aggregate PnL, equity curves, day-counts per regime, and compare.

Usage:
    python -m scripts.analysis.regime_gate_backtest
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime, regime_features

DATA_DIRS = [
    "stored_data_2021", "stored_data_2022", "stored_data_2023",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data", "stored_data_mar_may_2026",
]
NORMAL_PARAMS_PATH = "config/trial_6_extracted.json"
# Two candidate squeeze specialists to compare:
#   trial #24   - best from current 2021-2026 study (IS for all 5 years)
#   trial #635  - best from earlier 2024-25 study (purer squeeze-regime fit)
SQUEEZE_24_PATH = "config/trial_24_2021_2026_extracted.json"
SQUEEZE_635_PATH = "config/trial_635_extracted.json"
BASELINE_PATH = "config/trial_432_params.json"
STARTING_CASH = 25_000


def _read_merged(path):
    with open(path) as f:
        p = json.load(f)
    if isinstance(p, dict) and "params" in p:
        p = p["params"]
    with open(BASELINE_PATH) as f:
        base = json.load(f)
    merged = dict(base)
    merged.update(p)
    return merged


def load_params(path=NORMAL_PARAMS_PATH):
    merged = _read_merged(path)
    set_strategy_params(merged)
    return merged


def enable_dynamic_slip():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.VOL_CAP_PCT = 5.0


def run_backtest(daily_picks, dates, *, mode):
    """Run a 2021-2026 backtest.

    mode:
      "baseline"    - trial #6 on every day, no gate
      "gate"        - trial #6 on normal+squeeze, skip DEAD
      "swap_24"     - trial #6 on normal, trial #24 on squeeze, skip DEAD
      "swap_635"    - trial #6 on normal, trial #635 on squeeze, skip DEAD
    """
    assert mode in ("baseline", "gate", "swap_24", "swap_635")

    cash = STARTING_CASH
    per_day = []
    per_regime = defaultdict(lambda: {"days": 0, "pnl": 0.0, "trades": 0})
    current_loaded = None  # which config is currently loaded into tgc

    def _ensure_config(name):
        nonlocal current_loaded
        if current_loaded == name:
            return
        if name == "squeeze_24":
            path = SQUEEZE_24_PATH
        elif name == "squeeze_635":
            path = SQUEEZE_635_PATH
        else:
            path = NORMAL_PARAMS_PATH
        load_params(path)
        enable_dynamic_slip()
        current_loaded = name

    for date in dates:
        picks_today = daily_picks.get(date, [])
        regime = classify_regime(picks_today)
        per_regime[regime]["days"] += 1

        # Decide whether to trade and which config to use
        skip = False
        which_config = "normal"
        if mode in ("gate", "swap_24", "swap_635") and regime == "dead":
            skip = True
        elif mode == "swap_24" and regime == "squeeze":
            which_config = "squeeze_24"
        elif mode == "swap_635" and regime == "squeeze":
            which_config = "squeeze_635"

        if skip:
            per_day.append({"date": date, "regime": regime, "skipped": True,
                            "trades": 0, "pnl": 0.0, "cash": cash,
                            "config": "none"})
            continue

        _ensure_config(which_config)

        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, ending_cash, unsettled, _ = tgc.simulate_day_combined(
                picks_today, cash, cash_account=is_cash
            )
        except Exception as e:
            per_day.append({"date": date, "regime": regime, "skipped": False,
                            "trades": 0, "pnl": 0.0, "cash": cash,
                            "config": which_config, "error": str(e)[:80]})
            continue

        day_pnl = ending_cash - cash
        n_trades = sum(1 for st in states if st.get("entry_price") is not None
                       and st.get("exit_price") is not None)
        cash = ending_cash
        if is_cash:
            cash += unsettled

        per_regime[regime]["pnl"] += day_pnl
        per_regime[regime]["trades"] += n_trades

        per_day.append({
            "date": date,
            "regime": regime,
            "skipped": False,
            "trades": n_trades,
            "pnl": day_pnl,
            "cash": cash,
            "config": which_config,
        })

    return {
        "mode": mode,
        "final_equity": cash,
        "total_pnl": cash - STARTING_CASH,
        "n_traded_days": sum(1 for d in per_day if not d["skipped"]),
        "n_skipped_days": sum(1 for d in per_day if d["skipped"]),
        "per_regime": dict(per_regime),
        "per_day": per_day,
    }


def main():
    print("=" * 78)
    print("REGIME GATE BACKTEST — trial #6 with vs without gate, 2021-2026")
    print("=" * 78)
    print()

    print("Loading params...")
    load_params()
    enable_dynamic_slip()

    print(f"Loading picks across {len(DATA_DIRS)} dirs...")
    dates, daily_picks = load_all_picks(DATA_DIRS)
    print(f"  {len(dates)} trading days: {dates[0]} -> {dates[-1]}")

    # Regime distribution first, before sim
    regime_counts = Counter()
    for d in dates:
        regime_counts[classify_regime(daily_picks[d])] += 1
    print(f"\nRegime distribution over {len(dates)} days:")
    for r in ("dead", "normal", "squeeze"):
        n = regime_counts[r]
        print(f"  {r:<10} {n:>5} days  ({100*n/len(dates):.1f}%)")

    print("\n--- Run 1/4: BASELINE (trial #6 every day, no gate) ---")
    baseline = run_backtest(daily_picks, dates, mode="baseline")

    print("--- Run 2/4: GATE (trial #6 + skip DEAD days) ---")
    gated = run_backtest(daily_picks, dates, mode="gate")

    print("--- Run 3/4: SWAP_24 (trial #6 on normal, trial #24 on squeeze, skip DEAD) ---")
    swap24 = run_backtest(daily_picks, dates, mode="swap_24")

    print("--- Run 4/4: SWAP_635 (trial #6 on normal, trial #635 on squeeze, skip DEAD) ---")
    swap635 = run_backtest(daily_picks, dates, mode="swap_635")

    print("\n" + "=" * 92)
    print("RESULTS")
    print("=" * 92)
    runs = [("Baseline", baseline), ("Gate", gated),
            ("Swap_24", swap24), ("Swap_635", swap635)]
    print(f"  {'Metric':<22}" + "".join(f" {label:>20}" for label, _ in runs))
    print(f"  {'Final equity':<22}" + "".join(f"   ${r['final_equity']:>17,.0f}" for _, r in runs))
    print(f"  {'Total PnL':<22}" + "".join(f"   ${r['total_pnl']:>17,.0f}" for _, r in runs))
    print(f"  {'% return':<22}" + "".join(f"   {r['total_pnl']/STARTING_CASH*100:>17.1f}%" for _, r in runs))
    print(f"  {'Trading days':<22}" + "".join(f"   {r['n_traded_days']:>18}" for _, r in runs))
    print(f"  {'Skipped days':<22}" + "".join(f"   {r['n_skipped_days']:>18}" for _, r in runs))

    print("\nSharpe ratios (daily, annualized):")
    for label, run in runs:
        pnls = np.array([d["pnl"] for d in run["per_day"] if not d["skipped"]])
        if len(pnls) > 1 and pnls.std() > 0:
            sharpe = pnls.mean() / pnls.std() * np.sqrt(252)
            print(f"  {label:<12}  Sharpe={sharpe:.2f}  Avg P&L/day=${pnls.mean():,.2f}  StDev=${pnls.std():,.2f}")

    print("\nPer-regime breakdown:")
    print(f"  {'Regime':<10} {'Days':>6} {'Baseline':>14} {'Gate':>14} {'Swap_24':>14} {'Swap_635':>14}")
    for r in ("dead", "normal", "squeeze"):
        b = baseline["per_regime"].get(r, {"days": 0, "pnl": 0.0})
        g = gated["per_regime"].get(r, {"days": 0, "pnl": 0.0})
        s24 = swap24["per_regime"].get(r, {"days": 0, "pnl": 0.0})
        s635 = swap635["per_regime"].get(r, {"days": 0, "pnl": 0.0})
        print(f"  {r:<10} {b['days']:>6} ${b['pnl']:>13,.0f} ${g['pnl']:>13,.0f} ${s24['pnl']:>13,.0f} ${s635['pnl']:>13,.0f}")

    # Save results
    out_dir = "results/regime_gate"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for label, run in runs:
        df = pd.DataFrame(run["per_day"])
        df.to_csv(os.path.join(out_dir, f"{label.lower()}_{ts}.csv"), index=False)
    print(f"\nPer-day CSVs saved to {out_dir}/")


if __name__ == "__main__":
    main()
