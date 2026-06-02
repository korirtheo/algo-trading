"""OOS test for trial #635 on squeeze days that fall OUTSIDE its 2024-25 training window.

Splits squeeze days into:
  IS  = 2024-01-01 -> 2025-12-31  (training period for #635)
  OOS = 2021-01-01 -> 2023-12-31  (pre-training, blind test)
  OOS_recent = 2026-01-01 -> 2026-05-15 (post-training, fresh OOS)

Runs trial #635 + trial #6 on each subset and compares PnL/Sharpe.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime

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


def run_on_subset(daily_picks, dates, config_path, label):
    """Run a single trial config on a specific date subset (squeeze days)."""
    load_and_apply(config_path)
    cash = STARTING_CASH
    per_day_pnl = []
    n_trades_total = 0
    n_wins = 0
    n_losses = 0

    for date in dates:
        picks_today = daily_picks.get(date, [])
        if not picks_today:
            continue
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
        per_day_pnl.append(day_pnl)

        for st in states:
            if st.get("entry_price") is not None and st.get("exit_price") is not None:
                n_trades_total += 1
                pnl = st.get("pnl", 0)
                if pnl > 0:
                    n_wins += 1
                else:
                    n_losses += 1

    pnls = np.array(per_day_pnl)
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 and len(pnls) > 1 else 0.0
    wr = 100 * n_wins / max(1, n_trades_total)

    return {
        "label": label,
        "config": os.path.basename(config_path),
        "n_days": len(dates),
        "n_days_with_data": len([d for d in dates if daily_picks.get(d)]),
        "final_equity": cash,
        "total_pnl": cash - STARTING_CASH,
        "sharpe": sharpe,
        "n_trades": n_trades_total,
        "wins": n_wins,
        "losses": n_losses,
        "wr": wr,
    }


def main():
    print("=" * 80)
    print("TRIAL #635 OOS TEST ON SQUEEZE DAYS")
    print("  IS         = squeeze days in 2024-2025 (training period for #635)")
    print("  OOS_pre    = squeeze days in 2021-2023 (BLIND — pre-training)")
    print("  OOS_recent = squeeze days in 2026 (forward, post-training)")
    print("=" * 80)

    print("\nLoading picks...")
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    print(f"  {len(all_dates)} days: {all_dates[0]} -> {all_dates[-1]}")

    # Bucket all dates by regime + year
    squeeze_pre = []   # 2021-2023 squeeze
    squeeze_is = []    # 2024-2025 squeeze
    squeeze_post = []  # 2026 squeeze

    for d in all_dates:
        picks = daily_picks.get(d, [])
        regime = classify_regime(picks)
        if regime != "squeeze":
            continue
        # d is a string like "2021-01-04"
        year = int(d[:4])
        if 2021 <= year <= 2023:
            squeeze_pre.append(d)
        elif 2024 <= year <= 2025:
            squeeze_is.append(d)
        elif year >= 2026:
            squeeze_post.append(d)

    print(f"\nSqueeze day count by period:")
    print(f"  OOS_pre  (2021-2023): {len(squeeze_pre)}")
    print(f"  IS       (2024-2025): {len(squeeze_is)}")
    print(f"  OOS_post (2026):      {len(squeeze_post)}")

    buckets = [
        ("OOS_pre",   squeeze_pre),
        ("IS",        squeeze_is),
        ("OOS_post",  squeeze_post),
        ("ALL",       squeeze_pre + squeeze_is + squeeze_post),
    ]

    runs = []
    for label, dates in buckets:
        if not dates:
            print(f"\n[{label}]: no days, skipping")
            continue
        print(f"\n--- Running on {label} ({len(dates)} squeeze days) ---")
        r6 = run_on_subset(daily_picks, dates, TRIAL_6, f"{label}_trial_6")
        r635 = run_on_subset(daily_picks, dates, TRIAL_635, f"{label}_trial_635")
        print(f"  trial #6:   ${r6['final_equity']:>14,.0f}  Sharpe {r6['sharpe']:.2f}  "
              f"trades {r6['n_trades']}  WR {r6['wr']:.1f}%")
        print(f"  trial #635: ${r635['final_equity']:>14,.0f}  Sharpe {r635['sharpe']:.2f}  "
              f"trades {r635['n_trades']}  WR {r635['wr']:.1f}%")
        runs.append((label, r6, r635))

    print("\n" + "=" * 80)
    print("HEADLINE: does trial #635 generalize to OOS squeeze days?")
    print("=" * 80)
    print(f"  {'Subset':<14} {'Days':>5} {'Trial #6 PnL':>16} {'Trial #635 PnL':>18} "
          f"{'#635 vs #6':>13} {'#635 Sharpe':>13}")
    for label, r6, r635 in runs:
        delta_pct = 100 * (r635['total_pnl'] - r6['total_pnl']) / max(1, abs(r6['total_pnl']))
        print(f"  {label:<14} {r6['n_days']:>5} ${r6['total_pnl']:>15,.0f} ${r635['total_pnl']:>17,.0f}"
              f" {delta_pct:>+12.0f}% {r635['sharpe']:>13.2f}")

    # Save details
    out_dir = "results/regime_gate"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    flat = []
    for label, r6, r635 in runs:
        flat.append(r6); flat.append(r635)
    pd.DataFrame(flat).to_csv(os.path.join(out_dir, f"oos_squeeze_{ts}.csv"), index=False)
    print(f"\nSaved to {out_dir}/oos_squeeze_{ts}.csv")


if __name__ == "__main__":
    main()
