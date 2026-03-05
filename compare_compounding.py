"""
Compare two configs with COMPOUNDING — simulates rolling cash like test_full.py
but uses the fast optimizer engine. Shows daily equity side by side.
"""
import sys
import numpy as np
from optimize import load_all_picks, simulate_day_fast, DAILY_CASH

DATA_DIRS = ["stored_data_combined"]
STARTING_CASH = 25_000
TRADE_PCT = 0.50

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

# Regime filter
from regime_filters import RegimeFilter, _download_and_cache
import pandas as pd


def run_compounding(daily_picks, dates, config, regime, starting_cash=25_000):
    """Run with compounding — scale daily cash by account size."""
    cash = starting_cash
    results = []

    for d in dates:
        # Regime check
        if regime is not None:
            should_trade, _, _ = regime.check(d)
            if not should_trade:
                results.append((d, 0.0, cash, 0))
                continue

        picks = daily_picks[d]

        # Simulate with flat $10K
        flat_pnl, _ = simulate_day_fast(picks, config)

        # Scale P&L by account size vs $10K baseline
        # This approximates compounding: if account is $100K, P&L is 10x
        scale = cash / DAILY_CASH
        compound_pnl = flat_pnl * scale

        cash += compound_pnl
        if cash < 0:
            cash = 0

        results.append((d, compound_pnl, cash, flat_pnl))

    return results


if __name__ == "__main__":
    print("Loading data...")
    daily_picks = load_all_picks(DATA_DIRS)
    all_dates = sorted(daily_picks.keys())

    # Load regime filter (SPY SMA 40)
    start_date, end_date = all_dates[0], all_dates[-1]
    rf = RegimeFilter(spy_ma_period=40, enable_vix=False, enable_spy_trend=True, enable_adaptive=False)
    rf.load_data(start_date, end_date)

    print(f"\nFull period: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")
    print(f"Running both configs with compounding from ${STARTING_CASH:,}...\n")

    old_results = run_compounding(daily_picks, all_dates, OLD_CONFIG, rf, STARTING_CASH)
    new_results = run_compounding(daily_picks, all_dates, NEW_CONFIG, rf, STARTING_CASH)

    # Print December focused
    print(f"\n{'='*90}")
    print(f"  DECEMBER 2025 - JANUARY 2026 (COMPOUNDING)")
    print(f"{'='*90}")
    print(f"{'Date':<14} {'Old P&L':>12} {'Old Balance':>14} {'New P&L':>12} {'New Balance':>14} {'Note'}")
    print("-" * 82)

    for i, d in enumerate(all_dates):
        if d < "2025-11-15":
            continue
        if d > "2026-01-31":
            break

        od, opnl, obal, oflat = old_results[i]
        nd, npnl, nbal, nflat = new_results[i]

        note = ""
        if opnl == 0 and npnl == 0:
            note = "SKIP (regime)"
        elif abs(opnl - npnl) > max(obal, nbal) * 0.02:
            note = "<-- BIG DIFF"

        print(f"{d:<14} ${opnl:>+11,.0f} ${obal:>13,.0f} "
              f"${npnl:>+11,.0f} ${nbal:>13,.0f} {note}")

    # Find where they diverge
    print(f"\n{'='*90}")
    print(f"  MONTHLY SUMMARY (COMPOUNDING)")
    print(f"{'='*90}")

    months = {}
    for i, d in enumerate(all_dates):
        month = d[:7]
        if month not in months:
            months[month] = {"old_pnl": 0, "new_pnl": 0, "old_bal": 0, "new_bal": 0}
        months[month]["old_pnl"] += old_results[i][1]
        months[month]["new_pnl"] += new_results[i][1]
        months[month]["old_bal"] = old_results[i][2]
        months[month]["new_bal"] = new_results[i][2]

    print(f"{'Month':<12} {'Old P&L':>14} {'Old Balance':>14} {'New P&L':>14} {'New Balance':>14}")
    print("-" * 70)
    for month in sorted(months.keys()):
        m = months[month]
        print(f"{month:<12} ${m['old_pnl']:>+13,.0f} ${m['old_bal']:>13,.0f} "
              f"${m['new_pnl']:>+13,.0f} ${m['new_bal']:>13,.0f}")

    print(f"\n  Final: Old=${old_results[-1][2]:,.0f}  New=${new_results[-1][2]:,.0f}")
