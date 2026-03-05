"""
Stress Test — Robustness checks on the optimized strategy.

Tests:
  1. Remove top 10% winning days → recalculate P&L
  2. Simulate with 0.2% slippage (4x worse)
  3. Calculate Kelly criterion
  4. Monte Carlo: random day removal
  5. Rolling drawdown analysis

Uses the optimizer's simulate_day_fast() for speed.

Usage: python stress_test.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

# Reuse optimizer's precompute and simulation
from optimize import load_all_picks as _load_picks, simulate_day_fast, SLIPPAGE_PCT

DATA_DIRS = ["stored_data", "stored_data_oos"]

# Optimized params from Phase 2
BEST_PARAMS = {
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
    "partial_sell_frac_2": 0.35,
    "partial_sell_pct_2": 25.0,
    "scale_in_trigger_pct": 14.0,
    "cash_account": True,
    "cash_wait": False,
    "vol_cap_pct": 5.0,
    "eod_exit_minutes": 30,
    "reset_stops_on_partial": False,
    "scale_in_frac": 0.50,
}


def load_all_picks():
    """Load precomputed picks from all data dirs."""
    return _load_picks(DATA_DIRS)


def run_backtest_daily(daily_picks, params, slippage_override=None):
    """Run backtest, return list of (date, day_pnl) tuples."""
    import optimize
    original_slippage = optimize.SLIPPAGE_PCT
    if slippage_override is not None:
        optimize.SLIPPAGE_PCT = slippage_override

    results = []
    for date_str in sorted(daily_picks.keys()):
        picks = daily_picks[date_str]
        day_pnl, _ = simulate_day_fast(picks, params)
        results.append((date_str, day_pnl))

    if slippage_override is not None:
        optimize.SLIPPAGE_PCT = original_slippage
    return results


def kelly_criterion(daily_pnls):
    """Calculate Kelly fraction from daily P&L array."""
    wins = [p for p in daily_pnls if p > 0]
    losses = [abs(p) for p in daily_pnls if p < 0]

    if not wins or not losses:
        return 0.0, 0.0, 0.0

    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)
    p_win = len(wins) / len(daily_pnls)
    p_loss = 1 - p_win

    # b = avg_win / avg_loss (win/loss ratio)
    b = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # Kelly: f* = (bp - q) / b = p - q/b
    kelly = (b * p_win - p_loss) / b if b > 0 else 0.0

    return kelly, p_win, b


def max_drawdown(equity_curve):
    """Calculate max drawdown from equity curve."""
    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        dd_pct = dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    return max_dd, max_dd_pct


def rolling_equity(daily_results, starting_cash=100_000):
    """Build compounding equity curve from daily P&L results."""
    equity = [starting_cash]
    for _, pnl in daily_results:
        equity.append(equity[-1] + pnl)
    return equity


def run_stress_tests(daily_picks, params=None):
    """Run all stress tests and return structured results dict.

    Can be called from test_full.py to embed results in the summary chart.
    Uses optimizer's simulate_day_fast() with flat $10K daily cash.

    Args:
        daily_picks: dict of {date_str: picks_list} from load_all_picks()
        params: override BEST_PARAMS if provided

    Returns:
        dict with keys: rows (list of dicts), kelly, mc_profitable, all_pass,
        max_streak, baseline_sharpe
    """
    params = params or BEST_PARAMS

    # Test 1: Baseline
    baseline = run_backtest_daily(daily_picks, params)
    baseline_pnls = [p for _, p in baseline]
    total_pnl = sum(baseline_pnls)
    std = np.std(baseline_pnls)
    sharpe = (np.mean(baseline_pnls) / std) * np.sqrt(252) if std > 0 else 0

    # Test 2: Remove top 10%
    sorted_by_pnl = sorted(enumerate(baseline_pnls), key=lambda x: x[1], reverse=True)
    n_remove = max(1, int(len(baseline_pnls) * 0.10))
    removed_indices = set(i for i, _ in sorted_by_pnl[:n_remove])
    remaining_pnls = [p for i, p in enumerate(baseline_pnls) if i not in removed_indices]
    remaining_total = sum(remaining_pnls)
    rem_std = np.std(remaining_pnls) if len(remaining_pnls) > 1 else 1.0
    rem_sharpe = (np.mean(remaining_pnls) / rem_std) * np.sqrt(252) if rem_std > 0 else 0

    # Test 3: 0.2% slippage
    high_slip = run_backtest_daily(daily_picks, params, slippage_override=0.20)
    high_slip_pnls = [p for _, p in high_slip]
    high_slip_total = sum(high_slip_pnls)
    hs_std = np.std(high_slip_pnls)
    hs_sharpe = (np.mean(high_slip_pnls) / hs_std) * np.sqrt(252) if hs_std > 0 else 0

    # Test 3b: 0.5% slippage (extreme)
    extreme_slip = run_backtest_daily(daily_picks, params, slippage_override=0.50)
    extreme_pnls = [p for _, p in extreme_slip]
    extreme_total = sum(extreme_pnls)
    es_std = np.std(extreme_pnls)
    es_sharpe = (np.mean(extreme_pnls) / es_std) * np.sqrt(252) if es_std > 0 else 0

    # Test 4: Kelly
    kelly, win_rate, wl_ratio = kelly_criterion(baseline_pnls)

    # Test 5: Monte Carlo
    np.random.seed(42)
    n_sims = 1000
    n_remove_mc = max(1, int(len(baseline_pnls) * 0.10))
    mc_totals = []
    for _ in range(n_sims):
        indices = np.random.choice(len(baseline_pnls), size=len(baseline_pnls) - n_remove_mc, replace=False)
        mc_totals.append(sum(baseline_pnls[i] for i in indices))
    mc_totals = np.array(mc_totals)
    pct_profitable = (mc_totals > 0).mean() * 100

    # Test 7: Consecutive losses
    streak = 0
    max_streak = 0
    for pnl in baseline_pnls:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Build summary rows
    rows = [
        {"test": "Baseline (0.05% slip)", "pnl": total_pnl, "sharpe": sharpe,
         "verdict": "PASS"},
        {"test": "Remove top 10% days", "pnl": remaining_total, "sharpe": rem_sharpe,
         "verdict": "PASS" if remaining_total > 0 else "FAIL"},
        {"test": "0.2% slippage", "pnl": high_slip_total, "sharpe": hs_sharpe,
         "verdict": "PASS" if high_slip_total > 0 else "FAIL"},
        {"test": "0.5% slippage (extreme)", "pnl": extreme_total, "sharpe": es_sharpe,
         "verdict": "PASS" if extreme_total > 0 else "FAIL"},
        {"test": "Kelly > 0", "pnl": None, "sharpe": kelly * 100,
         "verdict": "PASS" if kelly > 0 else "FAIL"},
        {"test": "Monte Carlo (>80% prof.)", "pnl": None, "sharpe": pct_profitable,
         "verdict": "PASS" if pct_profitable > 80 else ("WEAK" if pct_profitable > 50 else "FAIL")},
    ]

    all_pass = all([remaining_total > 0, high_slip_total > 0, kelly > 0, pct_profitable > 80])

    return {
        "rows": rows,
        "kelly": kelly * 100,
        "mc_profitable": pct_profitable,
        "all_pass": all_pass,
        "max_streak": max_streak,
        "baseline_sharpe": sharpe,
    }


def main():
    print("=" * 70)
    print("  STRATEGY STRESS TEST")
    print("=" * 70)

    # Load data
    print("\nLoading precomputed picks...")
    daily_picks = load_all_picks()
    dates = sorted(daily_picks.keys())
    print(f"  {len(dates)} trading days: {dates[0]} to {dates[-1]}")

    # === TEST 1: Baseline ===
    print(f"\n{'='*70}")
    print("  TEST 1: BASELINE (optimized params, 0.05% slippage)")
    print(f"{'='*70}")

    baseline = run_backtest_daily(daily_picks, BEST_PARAMS)
    baseline_pnls = [p for _, p in baseline]
    total_pnl = sum(baseline_pnls)
    equity = rolling_equity(baseline)
    dd, dd_pct = max_drawdown(equity)

    green_days = sum(1 for p in baseline_pnls if p > 0)
    red_days = sum(1 for p in baseline_pnls if p < 0)
    flat_days = sum(1 for p in baseline_pnls if p == 0)

    print(f"  Total P&L:       ${total_pnl:+,.2f}")
    print(f"  Days:            {green_days} green / {red_days} red / {flat_days} flat")
    print(f"  Avg daily P&L:   ${np.mean(baseline_pnls):+,.2f}")
    print(f"  Std daily P&L:   ${np.std(baseline_pnls):,.2f}")
    sharpe = (np.mean(baseline_pnls) / np.std(baseline_pnls)) * np.sqrt(252) if np.std(baseline_pnls) > 0 else 0
    print(f"  Sharpe (ann.):   {sharpe:.2f}")
    print(f"  Max Drawdown:    ${dd:,.2f} ({dd_pct*100:.1f}%)")
    print(f"  Best Day:        ${max(baseline_pnls):+,.2f}")
    print(f"  Worst Day:       ${min(baseline_pnls):+,.2f}")

    # === TEST 2: Remove top 10% winning days ===
    print(f"\n{'='*70}")
    print("  TEST 2: REMOVE TOP 10% WINNING DAYS")
    print(f"{'='*70}")

    sorted_by_pnl = sorted(enumerate(baseline_pnls), key=lambda x: x[1], reverse=True)
    n_remove = max(1, int(len(baseline_pnls) * 0.10))
    removed_indices = set(i for i, _ in sorted_by_pnl[:n_remove])
    removed_days = [(baseline[i][0], baseline[i][1]) for i in sorted(removed_indices)]

    print(f"  Removing {n_remove} best days:")
    for date, pnl in removed_days:
        print(f"    {date}: ${pnl:+,.2f}")

    remaining_pnls = [p for i, p in enumerate(baseline_pnls) if i not in removed_indices]
    remaining_total = sum(remaining_pnls)
    remaining_equity = rolling_equity(
        [(baseline[i][0], baseline[i][1]) for i in range(len(baseline)) if i not in removed_indices]
    )
    remaining_dd, remaining_dd_pct = max_drawdown(remaining_equity)

    print(f"\n  Without top 10% days:")
    print(f"  Total P&L:       ${remaining_total:+,.2f}")
    print(f"  vs baseline:     ${remaining_total - total_pnl:+,.2f} ({(remaining_total/total_pnl - 1)*100:+.1f}%)" if total_pnl != 0 else "")
    print(f"  Avg daily P&L:   ${np.mean(remaining_pnls):+,.2f}")
    rem_sharpe = (np.mean(remaining_pnls) / np.std(remaining_pnls)) * np.sqrt(252) if np.std(remaining_pnls) > 0 else 0
    print(f"  Sharpe (ann.):   {rem_sharpe:.2f}")
    print(f"  Max Drawdown:    ${remaining_dd:,.2f} ({remaining_dd_pct*100:.1f}%)")
    green_rem = sum(1 for p in remaining_pnls if p > 0)
    red_rem = sum(1 for p in remaining_pnls if p < 0)
    print(f"  Days:            {green_rem} green / {red_rem} red")

    # === TEST 3: 0.2% slippage ===
    print(f"\n{'='*70}")
    print("  TEST 3: HIGH SLIPPAGE (0.20% vs baseline 0.05%)")
    print(f"{'='*70}")

    high_slip = run_backtest_daily(daily_picks, BEST_PARAMS, slippage_override=0.20)
    high_slip_pnls = [p for _, p in high_slip]
    high_slip_total = sum(high_slip_pnls)
    high_slip_equity = rolling_equity(high_slip)
    hs_dd, hs_dd_pct = max_drawdown(high_slip_equity)

    print(f"  Total P&L:       ${high_slip_total:+,.2f}")
    print(f"  vs baseline:     ${high_slip_total - total_pnl:+,.2f} ({(high_slip_total/total_pnl - 1)*100:+.1f}%)" if total_pnl != 0 else "")
    hs_sharpe = (np.mean(high_slip_pnls) / np.std(high_slip_pnls)) * np.sqrt(252) if np.std(high_slip_pnls) > 0 else 0
    print(f"  Sharpe (ann.):   {hs_sharpe:.2f}")
    print(f"  Max Drawdown:    ${hs_dd:,.2f} ({hs_dd_pct*100:.1f}%)")
    green_hs = sum(1 for p in high_slip_pnls if p > 0)
    red_hs = sum(1 for p in high_slip_pnls if p < 0)
    print(f"  Days:            {green_hs} green / {red_hs} red")

    # Also test 0.5% slippage (extreme)
    print(f"\n  --- Extreme: 0.50% slippage ---")
    extreme_slip = run_backtest_daily(daily_picks, BEST_PARAMS, slippage_override=0.50)
    extreme_pnls = [p for _, p in extreme_slip]
    extreme_total = sum(extreme_pnls)
    print(f"  Total P&L:       ${extreme_total:+,.2f}")
    print(f"  vs baseline:     ${extreme_total - total_pnl:+,.2f}")
    es_sharpe = (np.mean(extreme_pnls) / np.std(extreme_pnls)) * np.sqrt(252) if np.std(extreme_pnls) > 0 else 0
    print(f"  Sharpe (ann.):   {es_sharpe:.2f}")

    # === TEST 4: Kelly Criterion ===
    print(f"\n{'='*70}")
    print("  TEST 4: KELLY CRITERION")
    print(f"{'='*70}")

    kelly, win_rate, wl_ratio = kelly_criterion(baseline_pnls)
    print(f"  Win Rate (daily): {win_rate*100:.1f}%")
    print(f"  Win/Loss Ratio:   {wl_ratio:.2f}")
    print(f"  Kelly Fraction:   {kelly*100:.1f}%")
    print(f"  Half-Kelly:       {kelly*50:.1f}%")
    print(f"  Current sizing:   50% per trade")

    if kelly > 0:
        if kelly * 100 < 50:
            print(f"\n  WARNING: Kelly says bet {kelly*100:.1f}% -- you're sizing at 50% (OVER-LEVERAGED)")
            print(f"  Consider reducing TRADE_PCT to {kelly*50:.1f}% (half-Kelly)")
        else:
            print(f"\n  Kelly supports current 50% sizing ({kelly*100:.1f}% optimal)")
    else:
        print(f"\n  WARNING: Kelly is negative -- strategy doesn't have positive expectancy on daily basis")

    # === TEST 5: Monte Carlo (random day removal) ===
    print(f"\n{'='*70}")
    print("  TEST 5: MONTE CARLO -- RANDOM DAY REMOVAL (1000 simulations)")
    print(f"{'='*70}")

    np.random.seed(42)
    n_sims = 1000
    n_remove_mc = max(1, int(len(baseline_pnls) * 0.10))
    mc_totals = []
    for _ in range(n_sims):
        indices = np.random.choice(len(baseline_pnls), size=len(baseline_pnls) - n_remove_mc, replace=False)
        mc_pnls = [baseline_pnls[i] for i in indices]
        mc_totals.append(sum(mc_pnls))

    mc_totals = np.array(mc_totals)
    print(f"  Removed {n_remove_mc} random days per simulation")
    print(f"  P&L Distribution (1000 sims):")
    print(f"    Mean:    ${np.mean(mc_totals):+,.2f}")
    print(f"    Median:  ${np.median(mc_totals):+,.2f}")
    print(f"    Std:     ${np.std(mc_totals):,.2f}")
    print(f"    5th pct: ${np.percentile(mc_totals, 5):+,.2f}")
    print(f"    25th:    ${np.percentile(mc_totals, 25):+,.2f}")
    print(f"    75th:    ${np.percentile(mc_totals, 75):+,.2f}")
    print(f"    95th:    ${np.percentile(mc_totals, 95):+,.2f}")
    pct_profitable = (mc_totals > 0).mean() * 100
    print(f"    %% Profitable: {pct_profitable:.1f}%%")

    # === TEST 6: Monthly breakdown ===
    print(f"\n{'='*70}")
    print("  TEST 6: MONTHLY BREAKDOWN")
    print(f"{'='*70}")

    monthly = defaultdict(list)
    for date_str, pnl in baseline:
        month = date_str[:7]
        monthly[month].append(pnl)

    print(f"\n  {'Month':<10} {'Days':>5} {'Green':>6} {'Red':>5} {'Total P&L':>12} {'Avg/Day':>10} {'Sharpe':>7}")
    print(f"  {'-'*10} {'-'*5} {'-'*6} {'-'*5} {'-'*12} {'-'*10} {'-'*7}")

    for month in sorted(monthly.keys()):
        pnls = monthly[month]
        green = sum(1 for p in pnls if p > 0)
        red = sum(1 for p in pnls if p < 0)
        total = sum(pnls)
        avg = np.mean(pnls)
        std = np.std(pnls) if len(pnls) > 1 else 1.0
        sh = (avg / std) * np.sqrt(252) if std > 0 else 0
        print(f"  {month:<10} {len(pnls):>5} {green:>6} {red:>5} ${total:>+11,.2f} ${avg:>+9,.2f} {sh:>7.2f}")

    # === TEST 7: Consecutive losses ===
    print(f"\n{'='*70}")
    print("  TEST 7: CONSECUTIVE LOSS STREAKS")
    print(f"{'='*70}")

    streak = 0
    max_streak = 0
    streak_pnl = 0
    max_streak_pnl = 0
    for pnl in baseline_pnls:
        if pnl < 0:
            streak += 1
            streak_pnl += pnl
            if streak > max_streak:
                max_streak = streak
                max_streak_pnl = streak_pnl
        else:
            streak = 0
            streak_pnl = 0

    print(f"  Max consecutive losing days: {max_streak}")
    print(f"  P&L during worst streak:     ${max_streak_pnl:+,.2f}")

    # Count streaks
    streaks = []
    current = 0
    for pnl in baseline_pnls:
        if pnl < 0:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    if streaks:
        print(f"  Avg losing streak: {np.mean(streaks):.1f} days")
        print(f"  Streaks >= 3 days: {sum(1 for s in streaks if s >= 3)}")
        print(f"  Streaks >= 5 days: {sum(1 for s in streaks if s >= 5)}")

    # === TEST 8: Cash Account vs Margin (compounding) ===
    print(f"\n{'='*70}")
    print("  TEST 8: CASH ACCOUNT vs MARGIN (compounding from $10K)")
    print(f"{'='*70}")

    MARGIN_THRESHOLD = 25_000
    rolling_cash = 10_000
    crossed_date = None
    pre_margin_pnls = []
    post_margin_pnls = []

    for date_str, pnl in baseline:
        if rolling_cash < MARGIN_THRESHOLD:
            pre_margin_pnls.append(pnl)
        else:
            if crossed_date is None:
                crossed_date = date_str
            post_margin_pnls.append(pnl)
        rolling_cash += pnl

    if crossed_date:
        print(f"\n  Margin unlocked:  {crossed_date}  (balance crossed ${MARGIN_THRESHOLD:,})")
    else:
        print(f"\n  Margin NOT reached -- balance never crossed ${MARGIN_THRESHOLD:,}")
        print(f"  Final balance: ${rolling_cash:,.2f}")

    def _period_stats(label, pnls):
        if not pnls:
            print(f"\n  {label}: no trading days")
            return
        arr = np.array(pnls)
        total = arr.sum()
        green = (arr > 0).sum()
        red = (arr < 0).sum()
        flat = (arr == 0).sum()
        mean = arr.mean()
        std = arr.std() if len(arr) > 1 else 1.0
        sh = (mean / std) * np.sqrt(252) if std > 0 else 0
        print(f"\n  {label}")
        print(f"  {'-'*45}")
        print(f"  Days:          {len(pnls)}  ({green} green / {red} red / {flat} flat)")
        print(f"  Total P&L:     ${total:+,.2f}")
        print(f"  Avg/Day:       ${mean:+,.2f}")
        print(f"  Sharpe (ann.): {sh:.2f}")
        print(f"  Best Day:      ${arr.max():+,.2f}")
        print(f"  Worst Day:     ${arr.min():+,.2f}")
        print(f"  Win Rate:      {green/(green+red)*100:.1f}%" if (green + red) > 0 else "")

    _period_stats("CASH ACCOUNT (T+1 settlement, < $25K)", pre_margin_pnls)
    _period_stats("MARGIN ACCOUNT (instant recycling, >= $25K)", post_margin_pnls)

    if pre_margin_pnls and post_margin_pnls:
        pre_avg = np.mean(pre_margin_pnls)
        post_avg = np.mean(post_margin_pnls)
        print(f"\n  Margin avg P&L is {post_avg/pre_avg:.1f}x the cash account avg" if pre_avg > 0 else "")

    # === SUMMARY ===
    print(f"\n{'='*70}")
    print("  STRESS TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Test':<30} {'P&L':>12} {'Sharpe':>8} {'Verdict':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*12}")
    print(f"  {'Baseline (0.05% slip)':<30} ${total_pnl:>+11,.2f} {sharpe:>8.2f} {'PASS':>12}")

    rem_verdict = "PASS" if remaining_total > 0 else "FAIL"
    print(f"  {'Remove top 10% days':<30} ${remaining_total:>+11,.2f} {rem_sharpe:>8.2f} {rem_verdict:>12}")

    hs_verdict = "PASS" if high_slip_total > 0 else "FAIL"
    print(f"  {'0.2% slippage':<30} ${high_slip_total:>+11,.2f} {hs_sharpe:>8.2f} {hs_verdict:>12}")

    es_verdict = "PASS" if extreme_total > 0 else "FAIL"
    print(f"  {'0.5% slippage (extreme)':<30} ${extreme_total:>+11,.2f} {es_sharpe:>8.2f} {es_verdict:>12}")

    kelly_verdict = "PASS" if kelly > 0 else "FAIL"
    print(f"  {'Kelly > 0':<30} {'':>12} {kelly*100:>7.1f}% {kelly_verdict:>12}")

    mc_verdict = "PASS" if pct_profitable > 80 else "WEAK" if pct_profitable > 50 else "FAIL"
    print(f"  {'Monte Carlo (>80% profitable)':<30} {'':>12} {pct_profitable:>7.1f}% {mc_verdict:>12}")

    all_pass = all([remaining_total > 0, high_slip_total > 0, kelly > 0, pct_profitable > 80])
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
