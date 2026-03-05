"""
Regime-Aware Stress Test — Compare baseline vs regime-filtered performance.

Tests regime filters individually and combined:
  1. Baseline (no filters)
  2. VIX filter only (skip VIX > 30)
  3. SPY trend filter only (skip when SPY < 20-day SMA)
  4. Adaptive sizing only (50% size when 5-day WR < 45%)
  5. All filters combined
  6. Monthly comparison: filtered vs unfiltered

Usage: python stress_test_regime.py
"""
import sys
import numpy as np
from collections import defaultdict

from optimize import load_all_picks as _load_picks, simulate_day_fast, SLIPPAGE_PCT
from regime_filters import RegimeFilter, run_regime_backtest

DATA_DIRS = ["stored_data", "stored_data_oos"]

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


def compute_stats(daily_pnls):
    """Compute summary stats from daily P&L list."""
    arr = np.array(daily_pnls)
    total = arr.sum()
    green = (arr > 0).sum()
    red = (arr < 0).sum()
    flat = (arr == 0).sum()
    mean = arr.mean()
    std = arr.std() if len(arr) > 1 else 1.0
    sharpe = (mean / std) * np.sqrt(252) if std > 0 else 0

    # Max drawdown (peak-to-trough)
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    drawdowns = cum - peak
    max_dd = drawdowns.min() if len(drawdowns) > 0 else 0

    return {
        "total": total,
        "green": int(green),
        "red": int(red),
        "flat": int(flat),
        "mean": mean,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "best_day": arr.max() if len(arr) > 0 else 0,
        "worst_day": arr.min() if len(arr) > 0 else 0,
    }


def print_stats(label, stats, n_days, n_skipped=0):
    """Print stats block."""
    print(f"  Total P&L:       ${stats['total']:+,.2f}")
    print(f"  Days traded:     {n_days - n_skipped}/{n_days} (skipped {n_skipped})")
    print(f"  Green/Red/Flat:  {stats['green']}/{stats['red']}/{stats['flat']}")
    print(f"  Avg daily P&L:   ${stats['mean']:+,.2f}")
    print(f"  Sharpe (ann.):   {stats['sharpe']:.2f}")
    print(f"  Max Drawdown:    ${stats['max_dd']:+,.2f}")
    print(f"  Best Day:        ${stats['best_day']:+,.2f}")
    print(f"  Worst Day:       ${stats['worst_day']:+,.2f}")


def main():
    print("=" * 70)
    print("  REGIME-AWARE STRESS TEST")
    print("=" * 70)

    # Load data
    print("\nLoading precomputed picks...")
    daily_picks = _load_picks(DATA_DIRS)
    dates = sorted(daily_picks.keys())
    n_days = len(dates)
    print(f"  {n_days} trading days: {dates[0]} to {dates[-1]}")

    # === BASELINE (no filters) ===
    print(f"\n{'='*70}")
    print("  TEST 1: BASELINE (no regime filters)")
    print(f"{'='*70}")

    baseline_results = []
    for d in dates:
        pnl, _ = simulate_day_fast(daily_picks[d], BEST_PARAMS)
        baseline_results.append((d, pnl))
    baseline_pnls = [p for _, p in baseline_results]
    baseline_stats = compute_stats(baseline_pnls)
    print_stats("Baseline", baseline_stats, n_days)

    # === DOWNLOAD REGIME DATA ===
    print(f"\n{'='*70}")
    print("  DOWNLOADING REGIME DATA (VIX, SPY)")
    print(f"{'='*70}")

    # Full regime filter for data download
    rf_full = RegimeFilter(
        vix_threshold=30.0,
        spy_ma_period=20,
        adaptive_lookback=5,
        adaptive_wr_threshold=0.45,
        adaptive_size_mult=0.50,
    )
    rf_full.load_data(dates[0], dates[-1])

    # === TEST 2: VIX FILTER ONLY ===
    print(f"\n{'='*70}")
    print("  TEST 2: VIX FILTER ONLY (skip when VIX > 30)")
    print(f"{'='*70}")

    rf_vix = RegimeFilter(
        vix_threshold=30.0,
        enable_vix=True, enable_spy_trend=False, enable_adaptive=False,
    )
    rf_vix.vix_data = rf_full.vix_data
    rf_vix._loaded = True

    vix_results = run_regime_backtest(daily_picks, BEST_PARAMS, rf_vix)
    vix_pnls = [p for _, p, _ in vix_results]
    vix_skipped = sum(1 for _, _, info in vix_results if info.get("skipped"))
    vix_stats = compute_stats(vix_pnls)
    print_stats("VIX Only", vix_stats, n_days, vix_skipped)

    # Show which days were blocked
    if vix_skipped > 0:
        print(f"\n  VIX-blocked days:")
        for d, _, info in vix_results:
            if info.get("skipped"):
                vix_val = info.get("vix", "?")
                print(f"    {d}: VIX={vix_val:.1f}" if isinstance(vix_val, float) else f"    {d}")

    # Also test different VIX thresholds
    print(f"\n  --- VIX Threshold Sensitivity ---")
    print(f"  {'Threshold':>10} {'Skipped':>8} {'Total P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*8}")
    for vix_thresh in [20, 25, 30, 35, 40]:
        rf_t = RegimeFilter(
            vix_threshold=vix_thresh,
            enable_vix=True, enable_spy_trend=False, enable_adaptive=False,
        )
        rf_t.vix_data = rf_full.vix_data
        rf_t._loaded = True
        t_results = run_regime_backtest(daily_picks, BEST_PARAMS, rf_t)
        t_pnls = [p for _, p, _ in t_results]
        t_skip = sum(1 for _, _, info in t_results if info.get("skipped"))
        t_stats = compute_stats(t_pnls)
        print(f"  VIX > {vix_thresh:<4} {t_skip:>8} ${t_stats['total']:>+11,.2f} {t_stats['sharpe']:>8.2f}")

    # === TEST 3: SPY TREND FILTER ONLY ===
    print(f"\n{'='*70}")
    print("  TEST 3: SPY TREND FILTER ONLY (skip when SPY < 20-day SMA)")
    print(f"{'='*70}")

    rf_spy = RegimeFilter(
        spy_ma_period=20,
        enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
    )
    rf_spy.spy_data = rf_full.spy_data
    rf_spy.spy_sma = rf_full.spy_sma
    rf_spy._loaded = True

    spy_results = run_regime_backtest(daily_picks, BEST_PARAMS, rf_spy)
    spy_pnls = [p for _, p, _ in spy_results]
    spy_skipped = sum(1 for _, _, info in spy_results if info.get("skipped"))
    spy_stats = compute_stats(spy_pnls)
    print_stats("SPY Trend", spy_stats, n_days, spy_skipped)

    # SPY MA period sensitivity
    print(f"\n  --- SPY MA Period Sensitivity ---")
    print(f"  {'MA Period':>10} {'Skipped':>8} {'Total P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*8}")
    for ma_period in [10, 20, 50, 100]:
        rf_m = RegimeFilter(
            spy_ma_period=ma_period,
            enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
        )
        rf_m.spy_data = rf_full.spy_data
        rf_m.spy_sma = rf_m.spy_data["Close"].rolling(ma_period).mean()
        rf_m._loaded = True
        m_results = run_regime_backtest(daily_picks, BEST_PARAMS, rf_m)
        m_pnls = [p for _, p, _ in m_results]
        m_skip = sum(1 for _, _, info in m_results if info.get("skipped"))
        m_stats = compute_stats(m_pnls)
        print(f"  SMA({ma_period:<3}) {m_skip:>8} ${m_stats['total']:>+11,.2f} {m_stats['sharpe']:>8.2f}")

    # === TEST 4: ADAPTIVE SIZING ONLY ===
    print(f"\n{'='*70}")
    print("  TEST 4: ADAPTIVE SIZING ONLY (50% size when 5-day WR < 45%)")
    print(f"{'='*70}")

    rf_adapt = RegimeFilter(
        adaptive_lookback=5,
        adaptive_wr_threshold=0.45,
        adaptive_size_mult=0.50,
        enable_vix=False, enable_spy_trend=False, enable_adaptive=True,
    )
    rf_adapt._loaded = True

    adapt_results = run_regime_backtest(daily_picks, BEST_PARAMS, rf_adapt)
    adapt_pnls = [p for _, p, _ in adapt_results]
    adapt_reduced = sum(1 for _, _, info in adapt_results if info.get("size_reduced"))
    adapt_stats = compute_stats(adapt_pnls)
    print_stats("Adaptive", adapt_stats, n_days)
    print(f"  Days with reduced size: {adapt_reduced}/{n_days}")

    # Adaptive parameter sensitivity
    print(f"\n  --- Adaptive Sizing Sensitivity ---")
    print(f"  {'Lookback':>8} {'WR Thresh':>10} {'Size Mult':>10} {'Reduced':>8} {'P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*12} {'-'*8}")
    for lb, wrt, sm in [(3, 0.40, 0.50), (5, 0.45, 0.50), (5, 0.50, 0.50),
                          (5, 0.45, 0.25), (10, 0.45, 0.50)]:
        rf_a = RegimeFilter(
            adaptive_lookback=lb, adaptive_wr_threshold=wrt, adaptive_size_mult=sm,
            enable_vix=False, enable_spy_trend=False, enable_adaptive=True,
        )
        rf_a._loaded = True
        a_results = run_regime_backtest(daily_picks, BEST_PARAMS, rf_a)
        a_pnls = [p for _, p, _ in a_results]
        a_red = sum(1 for _, _, info in a_results if info.get("size_reduced"))
        a_stats = compute_stats(a_pnls)
        print(f"  {lb:>8} {wrt:>10.2f} {sm:>10.2f} {a_red:>8} ${a_stats['total']:>+11,.2f} {a_stats['sharpe']:>8.2f}")

    # === TEST 5: ALL FILTERS COMBINED ===
    print(f"\n{'='*70}")
    print("  TEST 5: ALL FILTERS COMBINED")
    print(f"  VIX > 30 skip | SPY < SMA(20) skip | 5-day WR < 45% -> 50% size")
    print(f"{'='*70}")

    combined_results = run_regime_backtest(daily_picks, BEST_PARAMS, rf_full)
    combined_pnls = [p for _, p, _ in combined_results]
    combined_skipped = sum(1 for _, _, info in combined_results if info.get("skipped"))
    combined_reduced = sum(1 for _, _, info in combined_results
                           if not info.get("skipped") and info.get("size_reduced"))
    combined_stats = compute_stats(combined_pnls)
    print_stats("Combined", combined_stats, n_days, combined_skipped)
    print(f"  Days with reduced size: {combined_reduced}")

    # === TEST 6: MONTHLY COMPARISON ===
    print(f"\n{'='*70}")
    print("  TEST 6: MONTHLY COMPARISON — BASELINE vs FILTERED")
    print(f"{'='*70}")

    # Build monthly buckets
    base_monthly = defaultdict(list)
    filt_monthly = defaultdict(list)
    for (d, pnl), (_, fpnl, _) in zip(baseline_results, combined_results):
        month = d[:7]
        base_monthly[month].append(pnl)
        filt_monthly[month].append(fpnl)

    print(f"\n  {'Month':<10} {'---BASELINE---':>25} {'---FILTERED---':>25} {'Delta':>10}")
    print(f"  {'':10} {'P&L':>12} {'Sharpe':>7} {'Days':>5}  {'P&L':>12} {'Sharpe':>7} {'Days':>5}")
    print(f"  {'-'*85}")

    for month in sorted(base_monthly.keys()):
        bp = base_monthly[month]
        fp = filt_monthly[month]
        b_total = sum(bp)
        f_total = sum(fp)
        b_std = np.std(bp) if len(bp) > 1 else 1.0
        f_std = np.std(fp) if len(fp) > 1 else 1.0
        b_sharpe = (np.mean(bp) / b_std) * np.sqrt(252) if b_std > 0 else 0
        f_sharpe = (np.mean(fp) / f_std) * np.sqrt(252) if f_std > 0 else 0
        b_green = sum(1 for p in bp if p > 0)
        f_green = sum(1 for p in fp if p > 0)
        delta = f_total - b_total
        delta_color = "+" if delta >= 0 else ""
        print(f"  {month:<10} ${b_total:>+10,.2f} {b_sharpe:>7.2f} {b_green:>3}/{len(bp):<2}"
              f"  ${f_total:>+10,.2f} {f_sharpe:>7.2f} {f_green:>3}/{len(fp):<2}"
              f"  ${delta:>+9,.2f}")

    # === SUMMARY TABLE ===
    print(f"\n{'='*70}")
    print("  REGIME FILTER SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Filter':<30} {'P&L':>12} {'Sharpe':>8} {'Skip':>6} {'Delta':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*6} {'-'*12}")

    rows = [
        ("Baseline (none)", baseline_stats, 0),
        ("VIX > 30", vix_stats, vix_skipped),
        ("SPY < SMA(20)", spy_stats, spy_skipped),
        ("Adaptive sizing", adapt_stats, 0),
        ("ALL COMBINED", combined_stats, combined_skipped),
    ]
    for label, stats, skipped in rows:
        delta = stats["total"] - baseline_stats["total"]
        print(f"  {label:<30} ${stats['total']:>+11,.2f} {stats['sharpe']:>8.2f} "
              f"{skipped:>6} ${delta:>+11,.2f}")

    # Verdict
    print(f"\n  Best filter:  ", end="")
    best_row = max(rows, key=lambda x: x[1]["sharpe"])
    print(f"{best_row[0]} (Sharpe {best_row[1]['sharpe']:.2f})")

    # Did combined improve the worst month?
    base_worst_month = min(base_monthly.items(), key=lambda x: sum(x[1]))
    filt_worst_pnl = sum(filt_monthly[base_worst_month[0]])
    base_worst_pnl = sum(base_worst_month[1])
    print(f"  Worst month:   {base_worst_month[0]} "
          f"(Baseline: ${base_worst_pnl:+,.2f} -> Filtered: ${filt_worst_pnl:+,.2f})")

    if combined_stats["sharpe"] > baseline_stats["sharpe"]:
        print(f"\n  VERDICT: Regime filters IMPROVE risk-adjusted returns")
    else:
        print(f"\n  VERDICT: Regime filters do NOT improve risk-adjusted returns")
        print(f"  (Baseline Sharpe {baseline_stats['sharpe']:.2f} vs "
              f"Combined {combined_stats['sharpe']:.2f})")

    print("=" * 70)


if __name__ == "__main__":
    main()
