"""
Optuna Optimizer: VWAP Reclaim Strategy
========================================
Gap-up stock sells off below VWAP, then reclaims with volume spike.
Buy the reclaim candle.

Usage:
  python optimize_vwap_reclaim.py
  python optimize_vwap_reclaim.py --trials 500
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


def compute_vwap(highs, lows, closes, volumes):
    """Compute cumulative VWAP from OHLCV arrays."""
    typical = (highs + lows + closes) / 3.0
    cum_tp_vol = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    cum_vol[cum_vol == 0] = 1e-9
    return cum_tp_vol / cum_vol


def precompute_stocks(daily_picks, all_dates):
    """Pre-compute arrays for all gap-up stocks, including VWAP."""
    stocks = []
    for d in all_dates:
        picks = daily_picks.get(d, [])
        for pick in picks:
            if pick["gap_pct"] < 5.0:
                continue
            candles = pick["market_hour_candles"]
            if len(candles) < 30:
                continue
            highs = candles["High"].values.astype(float)
            lows = candles["Low"].values.astype(float)
            closes = candles["Close"].values.astype(float)
            opens = candles["Open"].values.astype(float)
            volumes = candles["Volume"].values.astype(float)
            stocks.append({
                "date": d,
                "ticker": pick["ticker"],
                "gap_pct": pick["gap_pct"],
                "pm_high": pick["premarket_high"],
                "highs": highs,
                "lows": lows,
                "closes": closes,
                "opens": opens,
                "volumes": volumes,
                "vwap": compute_vwap(highs, lows, closes, volumes),
                "n": len(candles),
            })
    return stocks


def scan_and_trade_vwap_reclaim(stock, params):
    """
    VWAP Reclaim:
    1. Track consecutive closes below VWAP
    2. Check minimum depth below VWAP
    3. Enter on VWAP reclaim candle with volume spike
    4. Exit: trailing stop + partial sell + time stop
    """
    if stock["gap_pct"] < params["min_gap"]:
        return []

    closes = stock["closes"]
    highs = stock["highs"]
    lows = stock["lows"]
    volumes = stock["volumes"]
    vwap = stock["vwap"]
    n = stock["n"]

    min_below_candles = params["min_below_candles"]
    min_below_pct = params["min_below_pct"]
    vol_spike_ratio = params["vol_spike_ratio"]
    max_entry_candle = params["max_entry_candle"]

    if n < 30:
        return []

    # Track consecutive closes below VWAP, then look for reclaim
    below_count = 0
    max_depth_below = 0.0  # max % below VWAP
    entry_idx = None
    entry_price = None

    for i in range(2, min(n, max_entry_candle)):
        if closes[i] < vwap[i]:
            below_count += 1
            depth = (vwap[i] - closes[i]) / vwap[i] * 100
            if depth > max_depth_below:
                max_depth_below = depth
        else:
            # Close above VWAP — check if this is a reclaim
            if below_count >= min_below_candles and max_depth_below >= min_below_pct:
                # Volume spike check: current volume vs avg of last 10 candles
                lookback = max(1, i - 10)
                avg_vol = np.mean(volumes[lookback:i])
                if avg_vol > 0 and volumes[i] / avg_vol >= vol_spike_ratio:
                    entry_idx = i
                    entry_price = closes[i]
                    break
            # Reset counter (was above VWAP, not a reclaim candidate)
            below_count = 0
            max_depth_below = 0.0

    if entry_idx is None:
        return []

    # --- TRADE SIMULATION ---
    slip_entry = entry_price * (1 + SLIPPAGE_PCT / 100)

    target1_pct = params["target1_pct"]
    target2_pct = params["target2_pct"]
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
        if partial_taken:
            tgt = avg_entry * (1 + target2_pct / 100)
        else:
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
        results = scan_and_trade_vwap_reclaim(s, params)
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
    min_below_candles = trial.suggest_int("min_below_candles", 2, 15)
    min_below_pct = trial.suggest_int("min_below_pct", 0, 5)
    vol_spike_ratio = trial.suggest_float("vol_spike_ratio", 1.0, 4.0, step=0.5)
    max_entry_candle = trial.suggest_int("max_entry_candle", 30, 120, step=10)

    target1_pct = trial.suggest_float("target1_pct", 3.0, 12.0, step=1.0)
    stop_pct = trial.suggest_float("stop_pct", 3.0, 10.0, step=1.0)
    time_limit = trial.suggest_int("time_limit", 15, 60, step=5)

    partial_sell_pct = trial.suggest_float("partial_sell_pct", 0.0, 75.0, step=25.0)
    target2_pct = target1_pct
    if partial_sell_pct > 0:
        target2_pct = trial.suggest_float("target2_pct", target1_pct + 2, 20.0, step=2.0)

    trail_pct = trial.suggest_float("trail_pct", 2.0, 6.0, step=1.0)
    trail_activate_pct = trial.suggest_float("trail_activate_pct", 2.0, 6.0, step=1.0)

    params = {
        "min_gap": min_gap,
        "min_below_candles": min_below_candles,
        "min_below_pct": min_below_pct,
        "vol_spike_ratio": vol_spike_ratio,
        "max_entry_candle": max_entry_candle,
        "target1_pct": target1_pct,
        "target2_pct": target2_pct,
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

    _print(f"Optuna: VWAP Reclaim Strategy")
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

    _print("Pre-computing arrays + VWAP...")
    stocks = precompute_stocks(daily_picks, all_dates)
    _print(f"  {len(stocks)} stock-days with gap >= 5%\n")

    # Baseline
    _print("Running baseline...")
    baseline_params = {
        "min_gap": 15, "min_below_candles": 5, "min_below_pct": 1,
        "vol_spike_ratio": 1.5, "max_entry_candle": 60,
        "target1_pct": 6.0, "target2_pct": 6.0,
        "stop_pct": 5.0, "time_limit": 30,
        "partial_sell_pct": 0, "trail_pct": 3.0, "trail_activate_pct": 3.0,
    }
    base = run_trial(stocks, baseline_params)
    _print(f"  {base['n']} trades, {base['wr']:.1f}% WR, PF {base['pf']:.2f}, "
          f"P&L {base['total_pnl']:+.0f}%")
    _print(f"  Avg entry candle: {base['avg_entry']:.0f}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="vwap_reclaim_2024_2026",
        storage="sqlite:///optuna_vwap_reclaim.db",
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
                  f"gap>={bp.get('min_gap',0)}% below>={bp.get('min_below_candles',0)} "
                  f"vol>={bp.get('vol_spike_ratio',0):.1f}x "
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
    _print(f"  OPTIMIZATION COMPLETE — VWAP Reclaim")
    _print(f"{'='*70}")
    _print(f"  Best Objective: {best.value:+.0f}")
    _print(f"  vs Baseline:    PnL {base['total_pnl']:+.0f}%")
    _print(f"\n  Trades: {best.user_attrs['n']}  WR: {best.user_attrs['wr']:.1f}%  "
          f"PF: {best.user_attrs['pf']:.2f}")
    _print(f"  Avg Entry Candle: {best.user_attrs['avg_entry']:.0f}")

    _print(f"\n  ENTRY PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  min_gap:            {bp['min_gap']}%")
    _print(f"  min_below_candles:  {bp['min_below_candles']}")
    _print(f"  min_below_pct:      {bp['min_below_pct']}%")
    _print(f"  vol_spike_ratio:    {bp['vol_spike_ratio']:.1f}x")
    _print(f"  max_entry_candle:   {bp['max_entry_candle']}")

    _print(f"\n  EXIT PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  target1_pct:        +{bp['target1_pct']}%")
    if 'target2_pct' in bp:
        _print(f"  target2_pct:        +{bp['target2_pct']}% (runner)")
    _print(f"  stop_pct:           -{bp['stop_pct']}%")
    _print(f"  time_limit:         {bp['time_limit']} candles")
    _print(f"  trail_pct:          {bp['trail_pct']}%")
    _print(f"  trail_activate:     +{bp['trail_activate_pct']}%")
    if bp.get('partial_sell_pct', 0) > 0:
        _print(f"  partial_sell:       {bp['partial_sell_pct']}% at +{bp['target1_pct']}%")

    _print(f"\n{'='*70}")

    # Top 10 trials
    _print(f"\n  TOP 10 TRIALS")
    _print(f"  {'#':>3} {'Obj':>8} {'N':>5} {'WR':>5} {'PF':>5} {'Gap':>4} {'Below':>5} {'VolX':>4} {'Tgt':>4} {'Stp':>4}")
    _print(f"  {'-'*60}")
    sorted_t = sorted(study.trials, key=lambda t: t.value if t.value else -9999, reverse=True)
    for i, t in enumerate(sorted_t[:10]):
        if t.value is None or t.value <= -9999:
            continue
        p = t.params
        _print(f"  {i+1:>3} {t.value:>+7.0f} "
              f"{t.user_attrs.get('n',0):>5} {t.user_attrs.get('wr',0):>4.0f}% "
              f"{t.user_attrs.get('pf',0):>4.2f} "
              f"{p.get('min_gap',0):>3}% "
              f"{p.get('min_below_candles',0):>4} "
              f"{p.get('vol_spike_ratio',0):>3.1f} "
              f"+{p.get('target1_pct',0):>2.0f}% "
              f"-{p.get('stop_pct',0):>2.0f}%")
