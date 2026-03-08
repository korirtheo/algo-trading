"""
Optuna Optimizer: Midday Range Break Strategy
===============================================
Morning spike → 11am-1pm consolidation (tight range, low volume) →
afternoon breakout above consolidation high.

Usage:
  python optimize_midday_break.py
  python optimize_midday_break.py --trials 500
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

N_TRIALS = 500
DATA_DIRS = [
    "stored_data_jan_feb_2024", "stored_data_jan_mar_2024",
    "stored_data_apr_jun_2024", "stored_data_jul_sep_2024",
    "stored_data_oct_dec_2024", "stored_data_combined",
]
DATE_RANGE = ("2024-01-01", "2026-02-28")


def precompute_stocks(daily_picks, all_dates):
    """Pre-compute arrays for all gap-up stocks."""
    stocks = []
    for d in all_dates:
        picks = daily_picks.get(d, [])
        for pick in picks:
            if pick["gap_pct"] < 5.0:
                continue
            candles = pick["market_hour_candles"]
            if len(candles) < 90:  # need enough candles for midday pattern
                continue
            stocks.append({
                "date": d,
                "ticker": pick["ticker"],
                "gap_pct": pick["gap_pct"],
                "pm_high": pick["premarket_high"],
                "highs": candles["High"].values.astype(float),
                "lows": candles["Low"].values.astype(float),
                "closes": candles["Close"].values.astype(float),
                "opens": candles["Open"].values.astype(float),
                "volumes": candles["Volume"].values.astype(float),
                "n": len(candles),
            })
    return stocks


def scan_and_trade_midday_break(stock, params):
    """
    Midday Range Break:
    1. Verify morning spike in first N candles
    2. Define consolidation zone (tight range, low volume)
    3. Enter on breakout above consolidation high
    4. Exit: trailing stop + partial sell + time stop
    """
    if stock["gap_pct"] < params["min_gap"]:
        return []

    opens = stock["opens"]
    highs = stock["highs"]
    lows = stock["lows"]
    closes = stock["closes"]
    volumes = stock["volumes"]
    n = stock["n"]

    morning_spike_pct = params["morning_spike_pct"]
    morning_candles = params["morning_candles"]
    range_start = params["range_start_candle"]
    consol_len = params["consolidation_len"]
    max_range_pct = params["max_range_pct"]
    vol_ratio = params["vol_ratio"]
    max_entry_candle = params["max_entry_candle"]

    # Need enough candles for consolidation + breakout
    consol_end = range_start + consol_len
    if n < consol_end + 5:
        return []

    # Phase 1: Morning spike check
    morning_end = min(morning_candles, n)
    morning_high = np.max(highs[:morning_end])
    open_price = opens[0]

    spike_pct = (morning_high / open_price - 1) * 100
    if spike_pct < morning_spike_pct:
        return []

    # Phase 2: Consolidation zone
    consol_highs = highs[range_start:consol_end]
    consol_lows = lows[range_start:consol_end]
    consol_high = np.max(consol_highs)
    consol_low = np.min(consol_lows)

    # Range check: (high - low) / high * 100 must be tight
    range_pct = (consol_high - consol_low) / consol_high * 100
    if range_pct > max_range_pct:
        return []

    # Volume contraction check
    morning_avg_vol = np.mean(volumes[:min(30, n)])
    consol_avg_vol = np.mean(volumes[range_start:consol_end])
    if morning_avg_vol > 0 and consol_avg_vol / morning_avg_vol > vol_ratio:
        return []

    # Phase 3: Breakout after consolidation
    entry_idx = None
    entry_price = None

    for i in range(consol_end, min(n, max_entry_candle)):
        if closes[i] > consol_high:
            entry_idx = i
            entry_price = closes[i]
            break

    if entry_idx is None:
        return []

    # --- TRADE SIMULATION ---
    slip_entry = entry_price * (1 + SLIPPAGE_PCT / 100)

    target1_pct = params["target1_pct"]
    stop_pct = params["stop_pct"]
    time_limit = params["time_limit"]
    partial_sell_pct = params["partial_sell_pct"]
    trail_pct = params["trail_pct"]
    trail_activate_pct = params["trail_activate_pct"]

    position = 1.0
    avg_entry = slip_entry
    total_pnl = 0.0
    highest_since_entry = slip_entry
    trailing_active = False
    partial_taken = False

    for j in range(entry_idx + 1, min(entry_idx + time_limit + 1, n)):
        if position <= 0:
            break

        cur_high = highs[j]
        cur_low = lows[j]
        cur_close = closes[j]

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

        # Partial sell at target1
        if not partial_taken and partial_sell_pct > 0:
            tgt1 = avg_entry * (1 + target1_pct / 100)
            if cur_high >= tgt1:
                sell_frac = partial_sell_pct / 100
                exit_p = tgt1 * (1 - SLIPPAGE_PCT / 100)
                total_pnl += (exit_p / avg_entry - 1) * 100 * sell_frac
                position -= sell_frac * position
                partial_taken = True
                if position <= 0.01:
                    position = 0
                    break

        # Full target
        tgt = avg_entry * (1 + target1_pct / 100)
        if cur_high >= tgt:
            exit_p = tgt * (1 - SLIPPAGE_PCT / 100)
            total_pnl += (exit_p / avg_entry - 1) * 100 * position
            position = 0
            break

    # Time stop
    if position > 0.01:
        last = min(entry_idx + time_limit, n - 1)
        exit_p = closes[last] * (1 - SLIPPAGE_PCT / 100)
        total_pnl += (exit_p / avg_entry - 1) * 100 * position

    return [(total_pnl, entry_idx)]


def run_trial(stocks, params):
    """Run pattern on all stocks, return metrics."""
    pnls = []
    entries = []

    for s in stocks:
        results = scan_and_trade_midday_break(s, params)
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


def objective(trial, stocks):
    """Optuna objective: maximize risk-adjusted return."""

    min_gap = trial.suggest_int("min_gap", 8, 25, step=2)
    morning_spike_pct = trial.suggest_int("morning_spike_pct", 3, 15)
    morning_candles = trial.suggest_int("morning_candles", 15, 45, step=5)
    range_start_candle = trial.suggest_int("range_start_candle", 45, 75, step=5)
    consolidation_len = trial.suggest_int("consolidation_len", 15, 45, step=5)
    max_range_pct = trial.suggest_int("max_range_pct", 3, 10)
    vol_ratio = trial.suggest_float("vol_ratio", 0.3, 0.8, step=0.1)
    max_entry_candle = trial.suggest_int("max_entry_candle", 90, 150, step=10)

    target1_pct = trial.suggest_float("target1_pct", 3.0, 12.0, step=1.0)
    stop_pct = trial.suggest_float("stop_pct", 3.0, 10.0, step=1.0)
    time_limit = trial.suggest_int("time_limit", 20, 60, step=5)

    partial_sell_pct = trial.suggest_float("partial_sell_pct", 0.0, 50.0, step=25.0)

    trail_pct = trial.suggest_float("trail_pct", 2.0, 6.0, step=1.0)
    trail_activate_pct = trial.suggest_float("trail_activate_pct", 2.0, 6.0, step=1.0)

    params = {
        "min_gap": min_gap,
        "morning_spike_pct": morning_spike_pct,
        "morning_candles": morning_candles,
        "range_start_candle": range_start_candle,
        "consolidation_len": consolidation_len,
        "max_range_pct": max_range_pct,
        "vol_ratio": vol_ratio,
        "max_entry_candle": max_entry_candle,
        "target1_pct": target1_pct,
        "stop_pct": stop_pct,
        "time_limit": time_limit,
        "partial_sell_pct": partial_sell_pct,
        "trail_pct": trail_pct,
        "trail_activate_pct": trail_activate_pct,
    }

    result = run_trial(stocks, params)

    if result["n"] < 30:
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

    _print(f"Optuna: Midday Range Break Strategy")
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
    _print(f"  {len(stocks)} stock-days with gap >= 5% and >= 90 candles\n")

    # Baseline
    _print("Running baseline...")
    baseline_params = {
        "min_gap": 15, "morning_spike_pct": 5, "morning_candles": 30,
        "range_start_candle": 45, "consolidation_len": 30,
        "max_range_pct": 6, "vol_ratio": 0.6, "max_entry_candle": 120,
        "target1_pct": 6.0, "stop_pct": 5.0, "time_limit": 30,
        "partial_sell_pct": 0, "trail_pct": 3.0, "trail_activate_pct": 3.0,
    }
    base = run_trial(stocks, baseline_params)
    _print(f"  {base['n']} trades, {base['wr']:.1f}% WR, PF {base['pf']:.2f}, "
          f"P&L {base['total_pnl']:+.0f}%")
    _print(f"  Avg entry candle: {base['avg_entry']:.0f}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="midday_break_2024_2026",
        storage="sqlite:///optuna_midday_break.db",
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
                  f"gap>={bp.get('min_gap',0)}% range<={bp.get('max_range_pct',0)}% "
                  f"vol<={bp.get('vol_ratio',0):.1f}x "
                  f"tgt+{bp.get('target1_pct',0):.0f}% stp-{bp.get('stop_pct',0):.0f}%  "
                  f"[{elapsed:.0f}s]")
        elif n % 25 == 0:
            _print(f"  [{n:>3}/{n_trials}] trial={val:+.0f}  best={best:+.0f}  [{elapsed:.0f}s]")

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
    _print(f"  OPTIMIZATION COMPLETE — Midday Range Break")
    _print(f"{'='*70}")
    _print(f"  Best Objective: {best.value:+.0f}")
    _print(f"  vs Baseline:    PnL {base['total_pnl']:+.0f}%")
    _print(f"\n  Trades: {best.user_attrs['n']}  WR: {best.user_attrs['wr']:.1f}%  "
          f"PF: {best.user_attrs['pf']:.2f}")
    _print(f"  Avg Entry Candle: {best.user_attrs['avg_entry']:.0f}")

    _print(f"\n  ENTRY PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  min_gap:            {bp['min_gap']}%")
    _print(f"  morning_spike_pct:  {bp['morning_spike_pct']}%")
    _print(f"  morning_candles:    {bp['morning_candles']}")
    _print(f"  range_start_candle: {bp['range_start_candle']}")
    _print(f"  consolidation_len:  {bp['consolidation_len']}")
    _print(f"  max_range_pct:      {bp['max_range_pct']}%")
    _print(f"  vol_ratio:          {bp['vol_ratio']:.1f}x")
    _print(f"  max_entry_candle:   {bp['max_entry_candle']}")

    _print(f"\n  EXIT PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  target1_pct:        +{bp['target1_pct']}%")
    _print(f"  stop_pct:           -{bp['stop_pct']}%")
    _print(f"  time_limit:         {bp['time_limit']} candles")
    _print(f"  trail_pct:          {bp['trail_pct']}%")
    _print(f"  trail_activate:     +{bp['trail_activate_pct']}%")
    if bp.get('partial_sell_pct', 0) > 0:
        _print(f"  partial_sell:       {bp['partial_sell_pct']}% at +{bp['target1_pct']}%")

    _print(f"\n{'='*70}")

    # Top 10 trials
    _print(f"\n  TOP 10 TRIALS")
    _print(f"  {'#':>3} {'Obj':>8} {'N':>5} {'WR':>5} {'PF':>5} {'Gap':>4} {'Range':>5} {'Vol':>4} {'Tgt':>4} {'Stp':>4}")
    _print(f"  {'-'*65}")
    sorted_t = sorted(study.trials, key=lambda t: t.value if t.value else -9999, reverse=True)
    for i, t in enumerate(sorted_t[:10]):
        if t.value is None or t.value <= -9999:
            continue
        p = t.params
        _print(f"  {i+1:>3} {t.value:>+7.0f} "
              f"{t.user_attrs.get('n',0):>5} {t.user_attrs.get('wr',0):>4.0f}% "
              f"{t.user_attrs.get('pf',0):>4.2f} "
              f"{p.get('min_gap',0):>3}% "
              f"{p.get('max_range_pct',0):>4}% "
              f"{p.get('vol_ratio',0):>3.1f} "
              f"+{p.get('target1_pct',0):>2.0f}% "
              f"-{p.get('stop_pct',0):>2.0f}%")
