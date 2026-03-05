"""
Fast Optuna regime filter optimization — only 3 params.
Fixes all strategy params to current best, varies only:
  1. regime_ticker: SPY vs IWM
  2. regime_sma_period: 10-50
  3. vix_threshold: 0 (disabled) or 15-35
"""
import os
import sys
import time
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from optimize import load_all_picks, simulate_day_fast, DAILY_CASH, SLIPPAGE_PCT
from regime_filters import RegimeFilter, _download_and_cache
import pandas as pd

N_TRIALS = 200
DEFAULT_DATA_DIRS = ["stored_data_combined"]

# Fixed strategy params (current Optuna Phase 3 best)
FIXED_PARAMS = {
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


def build_regime(ticker, sma_period, vix_threshold, start_date, end_date):
    """Build a RegimeFilter with given params."""
    use_vix = vix_threshold > 0
    rf = RegimeFilter(
        spy_ma_period=sma_period,
        vix_threshold=vix_threshold if use_vix else 99,
        enable_vix=use_vix,
        enable_spy_trend=True,
        enable_adaptive=False,
    )

    # Override SPY with IWM if requested
    if ticker == "IWM":
        rf.spy_data = _download_and_cache("IWM", start_date, end_date, "iwm_daily.csv")
        if not rf.spy_data.empty:
            if isinstance(rf.spy_data.columns, pd.MultiIndex):
                rf.spy_data.columns = rf.spy_data.columns.get_level_values(0)
            rf.spy_sma = rf.spy_data["Close"].rolling(sma_period).mean()
    else:
        rf.load_data(start_date, end_date)

    # Load VIX separately if needed
    if use_vix and rf.vix_data is None:
        rf.vix_data = _download_and_cache("^VIX", start_date, end_date, "vix_daily.csv")

    rf._loaded = True
    return rf


def run_backtest_with_regime(daily_picks, params, regime):
    """Run backtest with regime filter. Returns total P&L."""
    total_pnl = 0.0
    daily_pnls = []

    for date_str in sorted(daily_picks.keys()):
        if regime is not None:
            should_trade, _, _ = regime.check(date_str)
            if not should_trade:
                continue
        picks = daily_picks[date_str]
        day_pnl, _ = simulate_day_fast(picks, params)
        total_pnl += day_pnl
        daily_pnls.append(day_pnl)

    daily_arr = np.array(daily_pnls)
    mean_d = np.mean(daily_arr) if len(daily_arr) > 0 else 0
    std_d = np.std(daily_arr) if len(daily_arr) > 1 else 1.0
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0
    return total_pnl, sharpe, daily_pnls


if __name__ == "__main__":
    data_dirs = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_DATA_DIRS

    print(f"Regime Filter Optimizer ({N_TRIALS} trials)")
    print(f"Data directories: {data_dirs}\n")

    daily_picks = load_all_picks(data_dirs)
    all_dates = sorted(daily_picks.keys())
    print(f"Loaded {len(all_dates)} trading days")
    print(f"Date range: {all_dates[0]} to {all_dates[-1]}")

    start_date, end_date = all_dates[0], all_dates[-1]

    # Pre-load all ticker data so trials don't re-download
    print("\nPre-loading market data...")
    spy_data = _download_and_cache("SPY", start_date, end_date, "spy_daily.csv")
    iwm_data = _download_and_cache("IWM", start_date, end_date, "iwm_daily.csv")
    vix_data = _download_and_cache("^VIX", start_date, end_date, "vix_daily.csv")
    print(f"  SPY: {len(spy_data)} days | IWM: {len(iwm_data)} days | VIX: {len(vix_data)} days")

    # Baseline: current SPY SMA(40), no VIX
    print("\nBenchmarking baseline (SPY SMA(40), no VIX)...")
    baseline_regime = build_regime("SPY", 40, 0, start_date, end_date)
    t0 = time.time()
    baseline_pnl, baseline_sharpe, baseline_daily = run_backtest_with_regime(
        daily_picks, FIXED_PARAMS, baseline_regime
    )
    t1 = time.time()
    tradeable_base = sum(1 for d in all_dates if baseline_regime.check(d)[0])
    print(f"  Baseline: ${baseline_pnl:+,.2f} | Sharpe: {baseline_sharpe:.2f} | "
          f"{tradeable_base} trade days | {t1-t0:.1f}s/trial")
    print(f"  Estimated: {(t1-t0) * N_TRIALS / 60:.0f} min for {N_TRIALS} trials\n")

    # Cache regime objects to avoid rebuilding each trial
    regime_cache = {}

    def get_regime(ticker, sma, vix_thresh):
        key = (ticker, sma, vix_thresh)
        if key not in regime_cache:
            regime_cache[key] = build_regime(ticker, sma, vix_thresh, start_date, end_date)
        return regime_cache[key]

    completed = [0]
    t_start = time.time()

    def objective(trial):
        ticker = trial.suggest_categorical("regime_ticker", ["SPY", "IWM"])
        sma_period = trial.suggest_int("regime_sma_period", 10, 50, step=5)
        vix_threshold = trial.suggest_categorical("vix_threshold", [0, 15, 20, 25, 30, 35])

        regime = get_regime(ticker, sma_period, vix_threshold)
        pnl, sharpe, _ = run_backtest_with_regime(daily_picks, FIXED_PARAMS, regime)

        completed[0] += 1
        if completed[0] % 10 == 0 or completed[0] == N_TRIALS:
            elapsed = time.time() - t_start
            eta = (elapsed / completed[0]) * (N_TRIALS - completed[0]) / 60
            print(f"  Trial {completed[0]}/{N_TRIALS} | Best: ${trial.study.best_value:+,.2f} | ETA: {eta:.1f} min",
                  end="", flush=True)
        return pnl

    print(f"{'='*60}")
    print(f"  OPTIMIZING REGIME FILTERS ({N_TRIALS} trials, 3 params)")
    print(f"  Params: regime_ticker, regime_sma_period, vix_threshold")
    print(f"{'='*60}")

    study = optuna.create_study(direction="maximize", study_name="regime_opt")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    elapsed = time.time() - t_start
    best = study.best_params
    best_pnl = study.best_value

    print(f"\n\n  Optimization complete in {elapsed/60:.1f} minutes.\n")
    print(f"{'='*60}")
    print(f"  REGIME FILTER RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline (SPY SMA40):  ${baseline_pnl:+,.2f} | Sharpe: {baseline_sharpe:.2f}")

    # Get best regime stats
    best_regime = get_regime(best["regime_ticker"], best["regime_sma_period"], best["vix_threshold"])
    best_pnl2, best_sharpe, best_daily = run_backtest_with_regime(daily_picks, FIXED_PARAMS, best_regime)
    tradeable_best = sum(1 for d in all_dates if best_regime.check(d)[0])
    skipped_best = len(all_dates) - tradeable_best

    print(f"  Optimized:             ${best_pnl2:+,.2f} | Sharpe: {best_sharpe:.2f}")
    print(f"  Improvement:           ${best_pnl2 - baseline_pnl:+,.2f}")
    print(f"\n  Best params:")
    print(f"    regime_ticker:     {best['regime_ticker']}")
    print(f"    regime_sma_period: {best['regime_sma_period']}")
    print(f"    vix_threshold:     {best['vix_threshold']} {'(disabled)' if best['vix_threshold'] == 0 else ''}")
    print(f"    Trade days:        {tradeable_best}/{len(all_dates)} ({skipped_best} skipped)")

    green = sum(1 for d in best_daily if d > 0)
    red = sum(1 for d in best_daily if d <= 0)
    cum = np.cumsum(best_daily)
    peak = np.maximum.accumulate(cum)
    max_dd = min(cum - peak) if len(cum) > 0 else 0
    print(f"\n  Stats:")
    print(f"    Green/Red days:    {green}/{red}")
    print(f"    Best day:          ${max(best_daily):+,.2f}")
    print(f"    Worst day:         ${min(best_daily):+,.2f}")
    print(f"    Max drawdown:      ${max_dd:+,.2f}")
    print(f"    Avg P&L/day:       ${np.mean(best_daily):+,.2f}")

    # Top 10
    print(f"\n  TOP 10 REGIME COMBINATIONS:")
    print(f"  {'Rank':<6} {'P&L':>14} {'Ticker':<6} {'SMA':<6} {'VIX':<6}")
    print(f"  {'-'*42}")
    top10 = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:10]
    for i, t in enumerate(top10):
        p = t.params
        print(f"  {i+1:<6} ${t.value:>+13,.2f} {p['regime_ticker']:<6} {p['regime_sma_period']:<6} {p['vix_threshold']:<6}")

    print(f"\n  COPY-PASTE FOR test_full.py:")
    print(f"  {'-'*40}")
    if best["regime_ticker"] == "IWM":
        print(f"  # Change regime filter to IWM")
        print(f"  REGIME_TICKER = 'IWM'")
    print(f"  SPY_SMA_PERIOD = {best['regime_sma_period']}")
    if best["vix_threshold"] > 0:
        print(f"  VIX_FILTER = True")
        print(f"  VIX_THRESHOLD = {best['vix_threshold']}")
    else:
        print(f"  # VIX filter: disabled (not beneficial)")
    print(f"{'='*60}")
