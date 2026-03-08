"""
Optuna Optimizer: Multi-Day Runner Strategy
=============================================
Day 1: massive gap-up (>=40%). Day 2: pullback then bounce — buy continuation.
Requires cross-day data linking (finds Day 2 intraday data for Day 1 gappers).

Usage:
  python optimize_multiday.py
  python optimize_multiday.py --trials 500
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import optuna

from test_full import load_all_picks, SLIPPAGE_PCT

optuna.logging.set_verbosity(optuna.logging.WARNING)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_builtin_print = print
def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _builtin_print(*args, **kwargs)

N_TRIALS = 500
DATA_DIRS = [
    "stored_data_jan_feb_2024", "stored_data_jan_mar_2024",
    "stored_data_apr_jun_2024", "stored_data_jul_sep_2024",
    "stored_data_oct_dec_2024", "stored_data_combined",
]
DATE_RANGE = ("2024-01-01", "2026-02-28")


def _load_intraday_for_date(ticker, target_date, data_dirs):
    """Try to load intraday CSV for a ticker, filter to target_date market hours."""
    for d in data_dirs:
        csv_path = os.path.join(d, "intraday", f"{ticker}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if df.empty:
                continue
            # Filter to target date
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            day_data = df[df.index.date == pd.Timestamp(target_date).date()]
            if len(day_data) < 20:
                continue
            # Filter to market hours (9:30-16:00 ET approx)
            day_data = day_data.between_time("09:30", "15:58")
            if len(day_data) < 20:
                continue
            return day_data
        except Exception:
            continue
    return None


def precompute_multiday_pairs(daily_picks, all_dates, data_dirs):
    """
    Find Day 1 gap-up stocks and load their Day 2 intraday data.
    Returns list of {day1_date, ticker, day1_gap, day1_close, day2 arrays}.
    """
    pairs = []
    _print("  Scanning for multi-day pairs...")

    for idx in range(len(all_dates) - 1):
        d1 = all_dates[idx]
        d2 = all_dates[idx + 1]

        picks = daily_picks.get(d1, [])
        for pick in picks:
            if pick["gap_pct"] < 30:  # pre-filter: only big gaps worth checking
                continue

            ticker = pick["ticker"]
            candles = pick["market_hour_candles"]
            if len(candles) < 10:
                continue

            day1_close = candles["Close"].values[-1]
            day1_high = candles["High"].values.max()

            # Try to load Day 2 intraday
            d2_data = _load_intraday_for_date(ticker, d2, data_dirs)
            if d2_data is None:
                continue

            d2_highs = d2_data["High"].values.astype(float) if "High" in d2_data.columns else None
            d2_lows = d2_data["Low"].values.astype(float) if "Low" in d2_data.columns else None
            d2_closes = d2_data["Close"].values.astype(float) if "Close" in d2_data.columns else None
            d2_opens = d2_data["Open"].values.astype(float) if "Open" in d2_data.columns else None
            d2_volumes = d2_data["Volume"].values.astype(float) if "Volume" in d2_data.columns else None

            if d2_highs is None or d2_lows is None or d2_closes is None or d2_opens is None:
                continue

            pairs.append({
                "day1_date": d1,
                "day2_date": d2,
                "ticker": ticker,
                "day1_gap": pick["gap_pct"],
                "day1_close": day1_close,
                "day1_high": day1_high,
                "d2_opens": d2_opens,
                "d2_highs": d2_highs,
                "d2_lows": d2_lows,
                "d2_closes": d2_closes,
                "d2_volumes": d2_volumes,
                "d2_n": len(d2_data),
            })

    _print(f"  Found {len(pairs)} multi-day pairs (gap >= 30%)")
    return pairs


def scan_and_trade_multiday(pair, params):
    """
    Multi-Day Runner:
    1. Day 1 gap >= threshold
    2. Day 2 pullback from open
    3. Day 2 bounce above reference price
    4. Exit: trailing stop + time stop
    """
    if pair["day1_gap"] < params["day1_min_gap"]:
        return []

    d2_opens = pair["d2_opens"]
    d2_highs = pair["d2_highs"]
    d2_lows = pair["d2_lows"]
    d2_closes = pair["d2_closes"]
    n = pair["d2_n"]

    d2_pullback_pct = params["d2_pullback_pct"]
    pullback_window = params["pullback_window"]
    bounce_ref = params["bounce_ref"]
    max_entry_candle = params["max_entry_candle"]

    if n < 20:
        return []

    d2_open = d2_opens[0]
    d1_close = pair["day1_close"]

    # Reference price for bounce
    ref_price = d2_open if bounce_ref == "d2_open" else d1_close

    # Phase 1: Pullback from Day 2 open
    pullback_level = d2_open * (1 - d2_pullback_pct / 100)
    pullback_seen = False
    entry_idx = None
    entry_price = None

    search_end = min(n, max_entry_candle)
    for i in range(min(pullback_window, search_end)):
        if d2_lows[i] <= pullback_level:
            pullback_seen = True
            break

    if not pullback_seen:
        return []

    # Phase 2: Bounce above reference
    for i in range(1, search_end):
        if d2_closes[i] > ref_price:
            entry_idx = i
            entry_price = d2_closes[i]
            break

    if entry_idx is None:
        return []

    # --- TRADE SIMULATION ---
    slip_entry = entry_price * (1 + SLIPPAGE_PCT / 100)

    target1_pct = params["target1_pct"]
    stop_pct = params["stop_pct"]
    time_limit = params["time_limit"]
    trail_pct = params["trail_pct"]
    trail_activate_pct = params["trail_activate_pct"]

    position = 1.0
    avg_entry = slip_entry
    total_pnl = 0.0
    highest_since_entry = slip_entry
    trailing_active = False

    for j in range(entry_idx + 1, min(entry_idx + time_limit + 1, n)):
        if position <= 0:
            break

        cur_high = d2_highs[j]
        cur_low = d2_lows[j]
        cur_close = d2_closes[j]

        if cur_high > highest_since_entry:
            highest_since_entry = cur_high

        # Stop loss (trailing or fixed)
        if trailing_active:
            trail_stop = highest_since_entry * (1 - trail_pct / 100)
            if cur_low <= trail_stop:
                exit_p = trail_stop * (1 - SLIPPAGE_PCT / 100)
                total_pnl += (exit_p / avg_entry - 1) * 100 * position
                position = 0
                break
        else:
            stop_price = avg_entry * (1 - stop_pct / 100)
            if cur_low <= stop_price:
                exit_p = stop_price * (1 - SLIPPAGE_PCT / 100)
                total_pnl += (exit_p / avg_entry - 1) * 100 * position
                position = 0
                break

        # Activate trailing stop
        if not trailing_active:
            unrealized = (cur_high / avg_entry - 1) * 100
            if unrealized >= trail_activate_pct:
                trailing_active = True
                highest_since_entry = cur_high

        # Target
        tgt = avg_entry * (1 + target1_pct / 100)
        if cur_high >= tgt:
            exit_p = tgt * (1 - SLIPPAGE_PCT / 100)
            total_pnl += (exit_p / avg_entry - 1) * 100 * position
            position = 0
            break

    # Time stop
    if position > 0.01:
        last = min(entry_idx + time_limit, n - 1)
        exit_p = d2_closes[last] * (1 - SLIPPAGE_PCT / 100)
        total_pnl += (exit_p / avg_entry - 1) * 100 * position

    return [(total_pnl, entry_idx)]


def run_trial(pairs, params):
    """Run pattern on all pairs, return metrics."""
    pnls = []
    entries = []

    for p in pairs:
        results = scan_and_trade_multiday(p, params)
        for pnl, entry_idx in results:
            pnls.append(pnl)
            entries.append(entry_idx)

    n = len(pnls)
    if n == 0:
        return {"n": 0, "wr": 0, "pf": 0, "total_pnl": -9999, "avg_entry": 0}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n * 100
    win_sum = sum(wins)
    loss_sum = abs(sum(losses))
    pf = win_sum / loss_sum if loss_sum > 0 else 0
    total = sum(pnls)

    return {
        "n": n, "wr": wr, "pf": pf, "total_pnl": total,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "avg_entry": np.mean(entries),
    }


def objective(trial, pairs):
    """Optuna objective: maximize risk-adjusted return."""

    day1_min_gap = trial.suggest_int("day1_min_gap", 40, 100, step=10)
    d2_pullback_pct = trial.suggest_int("d2_pullback_pct", 3, 10)
    pullback_window = trial.suggest_int("pullback_window", 5, 30, step=5)
    bounce_ref = trial.suggest_categorical("bounce_ref", ["d2_open", "d1_close"])
    max_entry_candle = trial.suggest_int("max_entry_candle", 15, 60, step=5)

    target1_pct = trial.suggest_float("target1_pct", 5.0, 20.0, step=2.0)
    stop_pct = trial.suggest_float("stop_pct", 5.0, 15.0, step=1.0)
    time_limit = trial.suggest_int("time_limit", 30, 90, step=10)

    trail_pct = trial.suggest_float("trail_pct", 3.0, 8.0, step=1.0)
    trail_activate_pct = trial.suggest_float("trail_activate_pct", 3.0, 8.0, step=1.0)

    params = {
        "day1_min_gap": day1_min_gap,
        "d2_pullback_pct": d2_pullback_pct,
        "pullback_window": pullback_window,
        "bounce_ref": bounce_ref,
        "max_entry_candle": max_entry_candle,
        "target1_pct": target1_pct,
        "stop_pct": stop_pct,
        "time_limit": time_limit,
        "trail_pct": trail_pct,
        "trail_activate_pct": trail_activate_pct,
    }

    result = run_trial(pairs, params)

    # Lower min trades threshold for rare pattern
    if result["n"] < 10:
        return -9999

    trial.set_user_attr("n", result["n"])
    trial.set_user_attr("wr", result["wr"])
    trial.set_user_attr("pf", result["pf"])
    trial.set_user_attr("avg_entry", result["avg_entry"])

    if result["pf"] < 0.5:
        return -9999
    return result["total_pnl"] * min(result["pf"], 3.0)


if __name__ == "__main__":
    n_trials = N_TRIALS
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--trials" and i + 2 <= len(sys.argv[1:]):
            n_trials = int(sys.argv[i + 2])

    _print(f"Optuna: Multi-Day Runner Strategy")
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

    _print("Pre-computing multi-day pairs...")
    pairs = precompute_multiday_pairs(daily_picks, all_dates, DATA_DIRS)
    _print(f"  {len(pairs)} pairs total\n")

    if len(pairs) < 5:
        _print("ERROR: Not enough multi-day pairs found. Need at least 5.")
        sys.exit(1)

    # Baseline
    _print("Running baseline...")
    baseline_params = {
        "day1_min_gap": 50, "d2_pullback_pct": 5, "pullback_window": 15,
        "bounce_ref": "d2_open", "max_entry_candle": 30,
        "target1_pct": 10.0, "stop_pct": 8.0, "time_limit": 60,
        "trail_pct": 5.0, "trail_activate_pct": 5.0,
    }
    base = run_trial(pairs, baseline_params)
    _print(f"  {base['n']} trades, {base['wr']:.1f}% WR, PF {base['pf']:.2f}, "
          f"P&L {base['total_pnl']:+.0f}%")
    _print(f"  Avg entry candle: {base['avg_entry']:.0f}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="multiday_2024_2026",
        storage="sqlite:///optuna_multiday.db",
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
            _print(f"  [{n:>3}/{n_trials}] NEW BEST  "
                  f"({bt.user_attrs.get('n',0)} tr, {bt.user_attrs.get('wr',0):.0f}% WR, "
                  f"PF {bt.user_attrs.get('pf',0):.2f})  "
                  f"d1gap>={bp.get('day1_min_gap',0)}% pb={bp.get('d2_pullback_pct',0)}% "
                  f"ref={bp.get('bounce_ref','?')} "
                  f"tgt+{bp.get('target1_pct',0):.0f}% stp-{bp.get('stop_pct',0):.0f}%  "
                  f"[{elapsed:.0f}s]")
        elif n % 25 == 0:
            _print(f"  [{n:>3}/{n_trials}] trial={val:+.0f}  best={best:+.0f}  [{elapsed:.0f}s]")

    _print(f"Running {n_trials} trials...")
    _print("-" * 70)
    study.optimize(
        lambda trial: objective(trial, pairs),
        n_trials=n_trials,
        callbacks=[callback],
    )

    best = study.best_trial
    bp = best.params
    _print(f"\n{'='*70}")
    _print(f"  OPTIMIZATION COMPLETE — Multi-Day Runner")
    _print(f"{'='*70}")
    _print(f"  Best Objective: {best.value:+.0f}")
    _print(f"  vs Baseline:    PnL {base['total_pnl']:+.0f}%")
    _print(f"\n  Trades: {best.user_attrs['n']}  WR: {best.user_attrs['wr']:.1f}%  "
          f"PF: {best.user_attrs['pf']:.2f}")
    _print(f"  Avg Entry Candle: {best.user_attrs['avg_entry']:.0f}")

    _print(f"\n  ENTRY PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  day1_min_gap:     {bp['day1_min_gap']}%")
    _print(f"  d2_pullback_pct:  {bp['d2_pullback_pct']}%")
    _print(f"  pullback_window:  {bp['pullback_window']} candles")
    _print(f"  bounce_ref:       {bp['bounce_ref']}")
    _print(f"  max_entry_candle: {bp['max_entry_candle']}")

    _print(f"\n  EXIT PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  target1_pct:      +{bp['target1_pct']}%")
    _print(f"  stop_pct:         -{bp['stop_pct']}%")
    _print(f"  time_limit:       {bp['time_limit']} candles")
    _print(f"  trail_pct:        {bp['trail_pct']}%")
    _print(f"  trail_activate:   +{bp['trail_activate_pct']}%")

    _print(f"\n{'='*70}")

    # Top 10 trials
    _print(f"\n  TOP 10 TRIALS")
    _print(f"  {'#':>3} {'Obj':>8} {'N':>5} {'WR':>5} {'PF':>5} {'D1Gap':>5} {'PB':>3} {'Ref':>7} {'Tgt':>4} {'Stp':>4}")
    _print(f"  {'-'*60}")
    sorted_t = sorted(study.trials, key=lambda t: t.value if t.value else -9999, reverse=True)
    for i, t in enumerate(sorted_t[:10]):
        if t.value is None or t.value <= -9999:
            continue
        p = t.params
        _print(f"  {i+1:>3} {t.value:>+7.0f} "
              f"{t.user_attrs.get('n',0):>5} {t.user_attrs.get('wr',0):>4.0f}% "
              f"{t.user_attrs.get('pf',0):>4.2f} "
              f"{p.get('day1_min_gap',0):>4}% "
              f"{p.get('d2_pullback_pct',0):>2}% "
              f"{p.get('bounce_ref','?'):>7} "
              f"+{p.get('target1_pct',0):>2.0f}% "
              f"-{p.get('stop_pct',0):>2.0f}%")
