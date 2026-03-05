"""
Quick comparison: run two configs on Dec 2025 - Jan 2026 only.
Prints daily P&L side-by-side to show where they diverge.
"""
import sys
import os
import numpy as np

# Reuse the optimizer's fast simulation
from optimize import load_all_picks, simulate_day_fast, DAILY_CASH

DATA_DIRS = ["stored_data_combined"]
DATE_START = "2025-11-01"  # include Nov for context
DATE_END = "2026-01-31"

# Old Phase 2 config ($71M run, bug-fixed)
OLD_CONFIG = {
    "stop_loss_pct": 16.0,
    "partial_sell_frac": 0.90,
    "partial_sell_pct": 15.0,
    "atr_period": 8,
    "atr_multiplier": 4.25,
    "confirm_above": 2,
    "confirm_window": 4,
    "pullback_pct": 4.0,
    "pullback_timeout": 24,
    "n_exit_tranches": 3,
    "min_pm_volume": 250_000,
    "scale_in": 1,
    "scale_in_trigger_pct": 14.0,
    "scale_in_frac": 0.50,
    "eod_exit_minutes": 30,
    "entry_cutoff_minutes": 0,
    "min_gap_pct": 2.0,
    "runner_mode": 0,
    "liq_vacuum": 0,
    "structural_stop": 0,
    "structural_stop_atr_mult": 1.5,
}

# Current Phase 3 Optuna config
NEW_CONFIG = {
    "stop_loss_pct": 11.0,
    "partial_sell_frac": 0.95,
    "partial_sell_pct": 19.0,
    "atr_period": 21,
    "atr_multiplier": 4.0,
    "confirm_above": 3,
    "confirm_window": 3,
    "pullback_pct": 5.5,
    "pullback_timeout": 20,
    "n_exit_tranches": 2,
    "min_pm_volume": 0,
    "scale_in": 1,
    "scale_in_trigger_pct": 5.0,
    "scale_in_frac": 1.0,
    "eod_exit_minutes": 35,
    "entry_cutoff_minutes": 0,
    "min_gap_pct": 12.0,
    "runner_mode": 0,
    "liq_vacuum": 0,
    "structural_stop": 1,
    "structural_stop_atr_mult": 1.25,
}

if __name__ == "__main__":
    print("Loading data...")
    daily_picks = load_all_picks(DATA_DIRS)

    # Filter to date range
    filtered = {d: v for d, v in daily_picks.items()
                if DATE_START <= d <= DATE_END}
    dates = sorted(filtered.keys())
    print(f"Period: {dates[0]} to {dates[-1]} ({len(dates)} days)\n")

    # Run both configs
    old_pnls = []
    new_pnls = []
    for d in dates:
        picks = filtered[d]
        old_pnl, _ = simulate_day_fast(picks, OLD_CONFIG)
        new_pnl, _ = simulate_day_fast(picks, NEW_CONFIG)
        old_pnls.append(old_pnl)
        new_pnls.append(new_pnl)

    # Print side-by-side
    old_cum = np.cumsum(old_pnls)
    new_cum = np.cumsum(new_pnls)

    print(f"{'Date':<14} {'Old P&L':>10} {'Old Cum':>12} {'New P&L':>10} {'New Cum':>12} {'Delta':>10}")
    print("-" * 72)

    for i, d in enumerate(dates):
        delta = old_cum[i] - new_cum[i]
        marker = ""
        if abs(old_pnls[i] - new_pnls[i]) > 500:
            marker = " <--"
        print(f"{d:<14} ${old_pnls[i]:>+9,.0f} ${old_cum[i]:>+11,.0f} "
              f"${new_pnls[i]:>+9,.0f} ${new_cum[i]:>+11,.0f} "
              f"${delta:>+9,.0f}{marker}")

    print("-" * 72)
    print(f"{'TOTAL':<14} ${sum(old_pnls):>+9,.0f} {'':>12} "
          f"${sum(new_pnls):>+9,.0f}")

    # Summary
    old_total = sum(old_pnls)
    new_total = sum(new_pnls)
    old_green = sum(1 for p in old_pnls if p > 0)
    new_green = sum(1 for p in new_pnls if p > 0)
    old_sharpe = (np.mean(old_pnls) / np.std(old_pnls)) * np.sqrt(252) if np.std(old_pnls) > 0 else 0
    new_sharpe = (np.mean(new_pnls) / np.std(new_pnls)) * np.sqrt(252) if np.std(new_pnls) > 0 else 0

    print(f"\n{'='*50}")
    print(f"  {'Metric':<20} {'Old (Phase 2)':>14} {'New (Phase 3)':>14}")
    print(f"  {'-'*48}")
    print(f"  {'Total P&L':<20} ${old_total:>+13,.0f} ${new_total:>+13,.0f}")
    print(f"  {'Sharpe':<20} {old_sharpe:>14.2f} {new_sharpe:>14.2f}")
    print(f"  {'Green days':<20} {old_green:>14} {new_green:>14}")
    print(f"  {'Best day':<20} ${max(old_pnls):>+13,.0f} ${max(new_pnls):>+13,.0f}")
    print(f"  {'Worst day':<20} ${min(old_pnls):>+13,.0f} ${min(new_pnls):>+13,.0f}")
    print(f"  {'Avg P&L/day':<20} ${np.mean(old_pnls):>+13,.0f} ${np.mean(new_pnls):>+13,.0f}")
    print(f"{'='*50}")
