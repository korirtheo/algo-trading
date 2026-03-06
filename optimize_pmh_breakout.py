"""
Optuna Optimizer: PM High Breakout + Pullback + Bounce (Pattern P)
==================================================================
The original test_full.py pattern, simplified and optimized:

  1. Stock gaps up >= X%
  2. Price breaks above premarket high (N closes above PM high in M candles)
  3. Price pulls back near PM high (within pullback_pct%)
  4. Price bounces: close > PM high again -> ENTRY
  5. Target: +Y%, Stop: -Z%, Time: T minutes

This tests whether the PM high breakout + pullback is profitable
as a standalone pattern with various parameter combinations.

Usage:
  python optimize_pmh_breakout.py
  python optimize_pmh_breakout.py --trials 300
"""

import sys
import time
import numpy as np
import optuna

from test_full import load_all_picks, SLIPPAGE_PCT

optuna.logging.set_verbosity(optuna.logging.WARNING)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_builtin_print = print
def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _builtin_print(*args, **kwargs)

N_TRIALS = 300
DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2025-01-01", "2026-02-28")


def precompute_stocks(daily_picks, all_dates):
    """Pre-compute arrays for all gap-up stocks."""
    stocks = []
    for d in all_dates:
        picks = daily_picks.get(d, [])
        for pick in picks:
            if pick["gap_pct"] < 5.0:
                continue
            candles = pick["market_hour_candles"]
            if len(candles) < 30:
                continue
            stocks.append({
                "date": d,
                "ticker": pick["ticker"],
                "gap_pct": pick["gap_pct"],
                "pm_high": pick["premarket_high"],
                "market_open": pick["market_open"],
                "highs": candles["High"].values.astype(float),
                "lows": candles["Low"].values.astype(float),
                "closes": candles["Close"].values.astype(float),
                "opens": candles["Open"].values.astype(float),
                "volumes": candles["Volume"].values.astype(float),
                "n": len(candles),
            })
    return stocks


def scan_and_trade(stock, params):
    """
    Scan for PM high breakout + pullback + bounce, then simulate trade.
    Returns (pnl_pct, entry_candle, reason) or None if no entry.
    """
    if stock["gap_pct"] < params["min_gap"]:
        return None

    pm_high = stock["pm_high"]
    closes = stock["closes"]
    lows = stock["lows"]
    highs = stock["highs"]
    n = stock["n"]

    confirm_above = params["confirm_above"]
    confirm_window = params["confirm_window"]
    pullback_pct = params["pullback_pct"]
    pullback_timeout = params["pullback_timeout"]
    max_entry_candle = params["max_entry_candle"]

    # Phase 1: Breakout confirmation
    # Need confirm_above closes above PM high within a rolling confirm_window
    breakout_candle = None
    recent = []

    for i in range(min(n, max_entry_candle)):
        recent.append(closes[i] > pm_high)
        if len(recent) > confirm_window:
            recent = recent[-confirm_window:]
        if sum(recent) >= confirm_above:
            breakout_candle = i
            break

    if breakout_candle is None:
        return None

    # Phase 2: Pullback detection
    # Price must come back within pullback_pct% of PM high
    pullback_zone = pm_high * (1 + pullback_pct / 100)
    pullback_detected = False
    candles_since_confirm = 0

    for i in range(breakout_candle + 1, min(n, max_entry_candle)):
        candles_since_confirm += 1

        if not pullback_detected:
            if lows[i] <= pullback_zone:
                pullback_detected = True
                # Immediate bounce: if this candle also closes above PM high
                if closes[i] > pm_high:
                    entry_idx = i
                    entry_price = closes[i]
                    break
            elif candles_since_confirm >= pullback_timeout:
                # Timeout: enter anyway if close > PM high
                if closes[i] > pm_high:
                    entry_idx = i
                    entry_price = closes[i]
                    break
                # Give up
                return None
        else:
            # Phase 3: Bounce - close above PM high
            if closes[i] > pm_high:
                entry_idx = i
                entry_price = closes[i]
                break
    else:
        return None

    # Simulate trade
    slip_entry = entry_price * (1 + SLIPPAGE_PCT / 100)
    target_pct = params["target_pct"]
    stop_pct = params["stop_pct"]
    time_limit = params["time_limit"]

    for j in range(entry_idx + 1, min(entry_idx + time_limit + 1, n)):
        # Stop first
        stop_p = slip_entry * (1 - stop_pct / 100)
        if lows[j] <= stop_p:
            exit_p = stop_p * (1 - SLIPPAGE_PCT / 100)
            pnl = (exit_p / slip_entry - 1) * 100
            return (pnl, entry_idx, "STOP")

        # Target
        tgt_p = slip_entry * (1 + target_pct / 100)
        if highs[j] >= tgt_p:
            exit_p = tgt_p * (1 - SLIPPAGE_PCT / 100)
            pnl = (exit_p / slip_entry - 1) * 100
            return (pnl, entry_idx, "TARGET")

    # Time stop
    last = min(entry_idx + time_limit, n - 1)
    exit_p = closes[last] * (1 - SLIPPAGE_PCT / 100)
    pnl = (exit_p / slip_entry - 1) * 100
    return (pnl, entry_idx, "TIME_STOP")


def run_trial(stocks, params):
    """Run pattern on all stocks, return metrics."""
    pnls = []
    entries = []
    reasons = {"TARGET": 0, "STOP": 0, "TIME_STOP": 0}

    for s in stocks:
        res = scan_and_trade(s, params)
        if res:
            pnl, entry_idx, reason = res
            pnls.append(pnl)
            entries.append(entry_idx)
            reasons[reason] = reasons.get(reason, 0) + 1

    n = len(pnls)
    if n == 0:
        return {"n": 0, "wr": 0, "pf": 0, "total_pnl": -9999,
                "avg_entry": 0, "reasons": reasons}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n * 100
    win_sum = sum(wins)
    loss_sum = abs(sum(losses))
    pf = win_sum / loss_sum if loss_sum > 0 else 0
    total = sum(pnls)
    avg_entry = np.mean(entries)

    return {
        "n": n, "wr": wr, "pf": pf, "total_pnl": total,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "avg_entry": avg_entry,
        "reasons": reasons,
    }


def objective(trial, stocks):
    """Optuna objective: maximize total P&L %."""
    params = {
        "min_gap": trial.suggest_int("min_gap", 10, 40, step=5),
        "confirm_above": trial.suggest_int("confirm_above", 1, 4),
        "confirm_window": trial.suggest_int("confirm_window", 2, 8),
        "pullback_pct": trial.suggest_float("pullback_pct", 1.0, 6.0, step=0.5),
        "pullback_timeout": trial.suggest_int("pullback_timeout", 5, 30, step=5),
        "max_entry_candle": trial.suggest_int("max_entry_candle", 15, 120, step=15),
        "target_pct": trial.suggest_float("target_pct", 3.0, 15.0, step=1.0),
        "stop_pct": trial.suggest_float("stop_pct", 2.0, 10.0, step=1.0),
        "time_limit": trial.suggest_int("time_limit", 10, 60, step=10),
    }

    # confirm_above must be <= confirm_window
    if params["confirm_above"] > params["confirm_window"]:
        return -9999

    result = run_trial(stocks, params)

    if result["n"] < 50:
        return -9999

    trial.set_user_attr("n", result["n"])
    trial.set_user_attr("wr", result["wr"])
    trial.set_user_attr("pf", result["pf"])
    trial.set_user_attr("avg_entry", result["avg_entry"])
    trial.set_user_attr("reasons", str(result["reasons"]))

    return result["total_pnl"]


if __name__ == "__main__":
    n_trials = N_TRIALS
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--trials" and i + 2 <= len(sys.argv[1:]):
            n_trials = int(sys.argv[i + 2])

    _print(f"Optuna: PM High Breakout + Pullback + Bounce")
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
    stocks = precompute_stocks(daily_picks, all_dates)
    _print(f"  {len(stocks)} stock-days with gap >= 5%\n")

    # Baseline: test_full.py defaults
    _print("Running baseline (test_full defaults)...")
    baseline_params = {
        "min_gap": 15, "confirm_above": 2, "confirm_window": 4,
        "pullback_pct": 4.0, "pullback_timeout": 24,
        "max_entry_candle": 60,
        "target_pct": 8.0, "stop_pct": 5.0, "time_limit": 30,
    }
    base = run_trial(stocks, baseline_params)
    _print(f"  {base['n']} trades, {base['wr']:.1f}% WR, PF {base['pf']:.2f}, "
          f"P&L {base['total_pnl']:+.0f}%")
    _print(f"  Avg entry candle: {base['avg_entry']:.0f}")
    _print(f"  Reasons: {base['reasons']}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="pmh_breakout",
        storage="sqlite:///optuna_pmh_breakout.db",
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
                  f"({bt.user_attrs.get('n',0)} tr, {bt.user_attrs.get('wr',0):.0f}% WR, "
                  f"PF {bt.user_attrs.get('pf',0):.2f})  "
                  f"gap>={bp.get('min_gap',0)}% tgt+{bp.get('target_pct',0):.0f}% "
                  f"stp-{bp.get('stop_pct',0):.0f}% {bp.get('time_limit',0)}m "
                  f"pb{bp.get('pullback_pct',0):.0f}% c{bp.get('max_entry_candle',0)}  "
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
    _print(f"  Best P&L:    {best.value:+.0f}%")
    _print(f"  vs Baseline: {base['total_pnl']:+.0f}%")
    _print(f"\n  Trades: {best.user_attrs['n']}  WR: {best.user_attrs['wr']:.1f}%  "
          f"PF: {best.user_attrs['pf']:.2f}")
    _print(f"  Avg Entry Candle: {best.user_attrs['avg_entry']:.0f}")
    _print(f"  Exit Reasons: {best.user_attrs['reasons']}")

    _print(f"\n  OPTIMAL PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  min_gap:          {bp['min_gap']}%")
    _print(f"  confirm_above:    {bp['confirm_above']} closes > PM high")
    _print(f"  confirm_window:   {bp['confirm_window']} candles")
    _print(f"  pullback_pct:     {bp['pullback_pct']}% from PM high")
    _print(f"  pullback_timeout: {bp['pullback_timeout']} candles")
    _print(f"  max_entry_candle: {bp['max_entry_candle']} (~{bp['max_entry_candle']}m after open)")
    _print(f"  target_pct:       +{bp['target_pct']}%")
    _print(f"  stop_pct:         -{bp['stop_pct']}%")
    _print(f"  time_limit:       {bp['time_limit']}m")
    _print(f"{'='*70}")

    # Top 10
    _print(f"\n  TOP 10 TRIALS")
    _print(f"  {'#':>3} {'P&L':>7} {'N':>5} {'WR':>5} {'PF':>5} "
          f"{'Gap':>4} {'Tgt':>4} {'Stp':>4} {'Tm':>3} {'PB':>4} {'MaxC':>4}")
    _print(f"  {'-'*60}")
    sorted_t = sorted(study.trials, key=lambda t: t.value if t.value else -9999, reverse=True)
    for i, t in enumerate(sorted_t[:10]):
        if t.value is None or t.value <= -9999:
            continue
        p = t.params
        _print(f"  {i+1:>3} {t.value:>+6.0f}% "
              f"{t.user_attrs.get('n',0):>5} {t.user_attrs.get('wr',0):>4.0f}% "
              f"{t.user_attrs.get('pf',0):>4.2f} "
              f"{p.get('min_gap',0):>3}% +{p.get('target_pct',0):>2.0f}% "
              f"-{p.get('stop_pct',0):>2.0f}% {p.get('time_limit',0):>2}m "
              f"{p.get('pullback_pct',0):>3.0f}% c{p.get('max_entry_candle',0):>3}")
