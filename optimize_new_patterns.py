"""
Optuna Optimizer for VWAP Reclaim (V) + Volume Spike Reversal (R)
=================================================================
Sweeps all parameters for both patterns simultaneously.
Objective: maximize total P&L % (sum of per-trade % returns).

Usage:
  python optimize_new_patterns.py              # 200 trials
  python optimize_new_patterns.py --trials 500
"""

import sys
import time
import numpy as np
import optuna

from test_full import load_all_picks, SLIPPAGE_PCT
import analyze_new_patterns as anp

optuna.logging.set_verbosity(optuna.logging.WARNING)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_builtin_print = print
def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _builtin_print(*args, **kwargs)

N_TRIALS = 200
DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2025-01-01", "2026-02-28")


def precompute_entries(daily_picks, all_dates):
    """Pre-scan all stocks once to get candle data for fast re-simulation.
    Returns list of (date, ticker, gap_pct, candles) for gap >= 10%.
    """
    stocks = []
    for d in all_dates:
        picks = daily_picks.get(d, [])
        for pick in picks:
            if pick["gap_pct"] < 10.0:  # Wide net, filter later
                continue
            candles = pick["market_hour_candles"]
            if len(candles) < 60:
                continue
            # Pre-compute VWAP and volume arrays once
            vwap = anp.compute_vwap(candles)
            vols = candles["Volume"].values.astype(float)
            highs = candles["High"].values.astype(float)
            closes = candles["Close"].values.astype(float)
            opens = candles["Open"].values.astype(float)
            lows = candles["Low"].values.astype(float)
            stocks.append({
                "date": d,
                "ticker": pick["ticker"],
                "gap_pct": pick["gap_pct"],
                "vwap": vwap,
                "vols": vols,
                "highs": highs,
                "lows": lows,
                "closes": closes,
                "opens": opens,
                "n": len(candles),
            })
    return stocks


def scan_v_fast(stock, params):
    """Fast VWAP reclaim scan using pre-computed arrays."""
    n = stock["n"]
    vwap = stock["vwap"]
    closes = stock["closes"]
    opens = stock["opens"]

    if stock["gap_pct"] < params["v_min_gap"]:
        return None
    if n < params["v_scan_end"]:
        return None

    was_below = False
    for i in range(params["v_scan_start"], min(params["v_scan_end"], n)):
        v = vwap[i]
        if np.isnan(v) or v <= 0:
            continue
        pct = (closes[i] - v) / v * 100
        if pct <= -params["v_below_thresh"]:
            was_below = True
        if was_below and closes[i] > v and opens[i] < v:
            return i, closes[i]
    return None


def scan_r_fast(stock, params):
    """Fast volume spike reversal scan using pre-computed arrays."""
    n = stock["n"]
    vols = stock["vols"]
    highs = stock["highs"]
    closes = stock["closes"]
    opens = stock["opens"]

    if stock["gap_pct"] < params["r_min_gap"]:
        return None
    start = params["r_scan_start"]
    if n < start + 20:
        return None

    for i in range(start, min(params["r_scan_end"], n)):
        c = closes[i]
        o = opens[i]
        if c <= o:
            continue
        body = (c - o) / o * 100 if o > 0 else 0
        if body < params["r_min_body"]:
            continue
        high_so_far = float(np.max(highs[:i]))
        if high_so_far <= 0:
            continue
        pullback = (high_so_far - c) / high_so_far * 100
        if pullback < params["r_pullback"]:
            continue
        if i < 20:
            continue
        avg_vol = np.mean(vols[i - 20:i])
        if avg_vol <= 0:
            continue
        if vols[i] / avg_vol < params["r_vol_mult"]:
            continue
        return i, c
    return None


def sim_fast(stock, entry_idx, entry_price, target_pct, stop_pct, time_limit):
    """Fast trade simulation from pre-computed arrays."""
    slip_entry = entry_price * (1 + SLIPPAGE_PCT / 100)
    highs = stock["highs"]
    lows = stock["lows"]
    closes = stock["closes"]
    n = stock["n"]

    for j in range(entry_idx + 1, min(entry_idx + time_limit + 1, n)):
        stop_p = slip_entry * (1 - stop_pct / 100)
        if lows[j] <= stop_p:
            exit_p = stop_p * (1 - SLIPPAGE_PCT / 100)
            return (exit_p / slip_entry - 1) * 100

        tgt_p = slip_entry * (1 + target_pct / 100)
        if highs[j] >= tgt_p:
            exit_p = tgt_p * (1 - SLIPPAGE_PCT / 100)
            return (exit_p / slip_entry - 1) * 100

    last = min(entry_idx + time_limit, n - 1)
    exit_p = closes[last] * (1 - SLIPPAGE_PCT / 100)
    return (exit_p / slip_entry - 1) * 100


def run_trial(stocks, params):
    """Run both patterns on all stocks, return metrics."""
    v_pnls = []
    r_pnls = []

    for s in stocks:
        # Pattern V
        v_res = scan_v_fast(s, params)
        if v_res:
            idx, price = v_res
            pnl = sim_fast(s, idx, price,
                           params["v_target"], params["v_stop"], params["v_time"])
            v_pnls.append(pnl)

        # Pattern R
        r_res = scan_r_fast(s, params)
        if r_res:
            idx, price = r_res
            pnl = sim_fast(s, idx, price,
                           params["r_target"], params["r_stop"], params["r_time"])
            r_pnls.append(pnl)

    v_n = len(v_pnls)
    r_n = len(r_pnls)
    v_wr = sum(1 for p in v_pnls if p > 0) / max(v_n, 1) * 100
    r_wr = sum(1 for p in r_pnls if p > 0) / max(r_n, 1) * 100
    v_total = sum(v_pnls)
    r_total = sum(r_pnls)
    v_pf = (abs(sum(p for p in v_pnls if p > 0) /
                sum(p for p in v_pnls if p <= 0))
            if any(p <= 0 for p in v_pnls) else 0)
    r_pf = (abs(sum(p for p in r_pnls if p > 0) /
                sum(p for p in r_pnls if p <= 0))
            if any(p <= 0 for p in r_pnls) else 0)

    return {
        "v_n": v_n, "v_wr": v_wr, "v_total": v_total, "v_pf": v_pf,
        "r_n": r_n, "r_wr": r_wr, "r_total": r_total, "r_pf": r_pf,
        "combined_pnl": v_total + r_total,
        "combined_trades": v_n + r_n,
    }


def objective(trial, stocks):
    """Optuna objective: maximize combined P&L %."""
    params = {
        # V: VWAP Reclaim
        "v_min_gap": trial.suggest_int("v_min_gap", 15, 50, step=5),
        "v_scan_start": trial.suggest_int("v_scan_start", 30, 120, step=15),
        "v_scan_end": trial.suggest_int("v_scan_end", 90, 240, step=30),
        "v_below_thresh": trial.suggest_float("v_below_thresh", 0.5, 3.0, step=0.5),
        "v_target": trial.suggest_float("v_target", 2.0, 10.0, step=1.0),
        "v_stop": trial.suggest_float("v_stop", 1.0, 5.0, step=0.5),
        "v_time": trial.suggest_int("v_time", 15, 60, step=15),

        # R: Volume Spike Reversal
        "r_min_gap": trial.suggest_int("r_min_gap", 15, 50, step=5),
        "r_scan_start": trial.suggest_int("r_scan_start", 30, 120, step=15),
        "r_scan_end": trial.suggest_int("r_scan_end", 120, 360, step=30),
        "r_pullback": trial.suggest_float("r_pullback", 2.0, 8.0, step=1.0),
        "r_vol_mult": trial.suggest_float("r_vol_mult", 1.0, 3.0, step=0.5),
        "r_min_body": trial.suggest_float("r_min_body", 0.3, 2.0, step=0.5),
        "r_target": trial.suggest_float("r_target", 2.0, 10.0, step=1.0),
        "r_stop": trial.suggest_float("r_stop", 1.0, 5.0, step=0.5),
        "r_time": trial.suggest_int("r_time", 15, 60, step=15),
    }

    # Scan start must be < scan end
    if params["v_scan_start"] >= params["v_scan_end"]:
        return -9999
    if params["r_scan_start"] >= params["r_scan_end"]:
        return -9999

    result = run_trial(stocks, params)

    # Must have some trades
    if result["combined_trades"] < 50:
        return -9999

    trial.set_user_attr("v_n", result["v_n"])
    trial.set_user_attr("v_wr", result["v_wr"])
    trial.set_user_attr("v_pf", result["v_pf"])
    trial.set_user_attr("v_total", result["v_total"])
    trial.set_user_attr("r_n", result["r_n"])
    trial.set_user_attr("r_wr", result["r_wr"])
    trial.set_user_attr("r_pf", result["r_pf"])
    trial.set_user_attr("r_total", result["r_total"])

    return result["combined_pnl"]


if __name__ == "__main__":
    n_trials = N_TRIALS
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--trials" and i + 2 <= len(sys.argv[1:]):
            n_trials = int(sys.argv[i + 2])

    _print(f"Optuna: VWAP Reclaim (V) + Volume Spike Reversal (R)")
    _print(f"{'='*70}")
    _print(f"  Trials:  {n_trials}")
    _print(f"  Data:    {DATA_DIRS}")
    _print(f"  Dates:   {DATE_RANGE[0]} to {DATE_RANGE[1]}")
    _print(f"{'='*70}\n")

    _print("Loading data...")
    t0 = time.time()
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
    _print(f"  {len(all_dates)} trading days in {time.time()-t0:.1f}s")

    _print("Pre-computing arrays...")
    stocks = precompute_entries(daily_picks, all_dates)
    _print(f"  {len(stocks)} stock-days with gap >= 10%\n")

    # Baseline
    _print("Running baseline...")
    baseline_params = {
        "v_min_gap": 15, "v_scan_start": 60, "v_scan_end": 150,
        "v_below_thresh": 0.5, "v_target": 4.0, "v_stop": 2.0, "v_time": 45,
        "r_min_gap": 15, "r_scan_start": 45, "r_scan_end": 300,
        "r_pullback": 3.0, "r_vol_mult": 1.5, "r_min_body": 0.5,
        "r_target": 4.0, "r_stop": 3.0, "r_time": 30,
    }
    base = run_trial(stocks, baseline_params)
    _print(f"  V: {base['v_n']} trades, {base['v_wr']:.0f}% WR, PF {base['v_pf']:.2f}, "
          f"P&L {base['v_total']:+.0f}%")
    _print(f"  R: {base['r_n']} trades, {base['r_wr']:.0f}% WR, PF {base['r_pf']:.2f}, "
          f"P&L {base['r_total']:+.0f}%")
    _print(f"  Combined: {base['combined_pnl']:+.0f}%\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="vr_patterns",
        storage="sqlite:///optuna_vr_patterns.db",
        load_if_exists=True,
    )

    best_so_far = [-99999]
    start_time = time.time()

    def callback(study, trial):
        elapsed = time.time() - start_time
        n = trial.number + 1
        val = trial.value if trial.value else -9999
        best = study.best_value
        if best > best_so_far[0]:
            best_so_far[0] = best
            bp = study.best_params
            bt = study.best_trial
            _print(f"  [{n:>3}/{n_trials}] NEW BEST {best:+.0f}%  "
                  f"V:{bt.user_attrs.get('v_n',0)}tr/{bt.user_attrs.get('v_wr',0):.0f}%WR/PF{bt.user_attrs.get('v_pf',0):.2f}  "
                  f"R:{bt.user_attrs.get('r_n',0)}tr/{bt.user_attrs.get('r_wr',0):.0f}%WR/PF{bt.user_attrs.get('r_pf',0):.2f}  "
                  f"[{elapsed:.0f}s]")
        elif n % 25 == 0:
            _print(f"  [{n:>3}/{n_trials}] trial {val:+.0f}%  best {best:+.0f}%  [{elapsed:.0f}s]")

    _print(f"Running {n_trials} trials...")
    _print("-" * 70)
    study.optimize(
        lambda trial: objective(trial, stocks),
        n_trials=n_trials,
        callbacks=[callback],
    )

    best = study.best_trial
    bp = best.params
    _print(f"\n{'='*70}")
    _print(f"  OPTIMIZATION COMPLETE")
    _print(f"{'='*70}")
    _print(f"  Best Combined P&L: {best.value:+.0f}%")
    _print(f"  vs Baseline:       {base['combined_pnl']:+.0f}%")

    _print(f"\n  Pattern V (VWAP Reclaim):")
    _print(f"    Trades: {best.user_attrs['v_n']}  WR: {best.user_attrs['v_wr']:.1f}%  "
          f"PF: {best.user_attrs['v_pf']:.2f}  P&L: {best.user_attrs['v_total']:+.0f}%")
    _print(f"    min_gap={bp['v_min_gap']}%  scan={bp['v_scan_start']}-{bp['v_scan_end']}  "
          f"below_thresh={bp['v_below_thresh']}%")
    _print(f"    target=+{bp['v_target']}%  stop=-{bp['v_stop']}%  time={bp['v_time']}m")

    _print(f"\n  Pattern R (Volume Spike Reversal):")
    _print(f"    Trades: {best.user_attrs['r_n']}  WR: {best.user_attrs['r_wr']:.1f}%  "
          f"PF: {best.user_attrs['r_pf']:.2f}  P&L: {best.user_attrs['r_total']:+.0f}%")
    _print(f"    min_gap={bp['r_min_gap']}%  scan={bp['r_scan_start']}-{bp['r_scan_end']}  "
          f"pullback={bp['r_pullback']}%  vol_mult={bp['r_vol_mult']}x  body={bp['r_min_body']}%")
    _print(f"    target=+{bp['r_target']}%  stop=-{bp['r_stop']}%  time={bp['r_time']}m")

    _print(f"\n{'='*70}")

    # Top 10
    _print(f"\n  TOP 10 TRIALS")
    _print(f"  {'#':>3} {'P&L':>7} {'V_n':>4} {'V_WR':>5} {'V_PF':>5} "
          f"{'R_n':>4} {'R_WR':>5} {'R_PF':>5}")
    _print(f"  {'-'*50}")
    sorted_t = sorted(study.trials, key=lambda t: t.value if t.value else -9999, reverse=True)
    for i, t in enumerate(sorted_t[:10]):
        if t.value is None or t.value <= -9999:
            continue
        _print(f"  {i+1:>3} {t.value:>+6.0f}% "
              f"{t.user_attrs.get('v_n',0):>4} {t.user_attrs.get('v_wr',0):>4.0f}% {t.user_attrs.get('v_pf',0):>4.2f} "
              f"{t.user_attrs.get('r_n',0):>4} {t.user_attrs.get('r_wr',0):>4.0f}% {t.user_attrs.get('r_pf',0):>4.2f}")
