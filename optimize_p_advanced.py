"""
Optuna Optimizer: Pattern P Advanced (Trailing Stop, Partials, Scale-In)
========================================================================
Builds on the basic PM High Breakout + Pullback + Bounce pattern (P)
and tests advanced exit/position-management features:

  1. Trailing stop loss (ATR-based or fixed %)
  2. Partial selling (sell X% at first target, hold rest for bigger move)
  3. Scale-in (add to position on 2nd bounce confirmation)

Optimizes using 2025-2026 data (stored_data_combined).

Usage:
  python optimize_p_advanced.py
  python optimize_p_advanced.py --trials 500
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
DATA_DIRS = ["stored_data_jan_feb_2024", "stored_data_jan_mar_2024", "stored_data_apr_jun_2024", "stored_data_jul_sep_2024", "stored_data_oct_dec_2024", "stored_data_combined"]
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
            if len(candles) < 30:
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


def compute_atr(highs, lows, closes, period=14):
    """Compute ATR array (simple rolling)."""
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    atr = np.zeros(n)
    atr[:period] = np.nan
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def scan_and_trade_advanced(stock, params):
    """
    PM high breakout + pullback + bounce with advanced trade management:
    - Trailing stop (fixed % or ATR-based)
    - Partial selling at first target
    - Optional scale-in on 2nd bounce

    Returns list of (pnl_pct, entry_candle, cycles) or empty list.
    """
    if stock["gap_pct"] < params["min_gap"]:
        return []

    pm_high = stock["pm_high"]
    closes = stock["closes"]
    lows = stock["lows"]
    highs = stock["highs"]
    opens = stock["opens"]
    n = stock["n"]

    confirm_above = params["confirm_above"]
    confirm_window = params["confirm_window"]
    pullback_pct = params["pullback_pct"]
    pullback_timeout = params["pullback_timeout"]
    max_entry_candle = params["max_entry_candle"]

    # Phase 1: Breakout confirmation
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
        return []

    # Phase 2: Pullback detection + Phase 3: Bounce
    pullback_zone = pm_high * (1 + pullback_pct / 100)
    pullback_detected = False
    candles_since_confirm = 0
    entry_idx = None
    entry_price = None

    for i in range(breakout_candle + 1, min(n, max_entry_candle)):
        candles_since_confirm += 1

        if not pullback_detected:
            if lows[i] <= pullback_zone:
                pullback_detected = True
                if closes[i] > pm_high:
                    entry_idx = i
                    entry_price = closes[i]
                    break
            elif candles_since_confirm >= pullback_timeout:
                if closes[i] > pm_high:
                    entry_idx = i
                    entry_price = closes[i]
                    break
                return []
        else:
            if closes[i] > pm_high:
                entry_idx = i
                entry_price = closes[i]
                break
    else:
        return []

    if entry_idx is None:
        return []

    # --- TRADE SIMULATION WITH ADVANCED MANAGEMENT ---
    slip_entry = entry_price * (1 + SLIPPAGE_PCT / 100)

    # Parameters
    target1_pct = params["target1_pct"]       # First partial target
    target2_pct = params["target2_pct"]       # Full exit target (runner)
    stop_pct = params["stop_pct"]             # Initial stop %
    time_limit = params["time_limit"]
    partial_sell_pct = params["partial_sell_pct"]  # % of position to sell at target1 (0 = no partials)

    # Trailing stop
    use_trailing = params["use_trailing"]     # 0=no, 1=fixed%, 2=ATR
    trail_pct = params["trail_pct"]           # Fixed trailing %
    trail_atr_mult = params["trail_atr_mult"] # ATR multiplier for trailing
    trail_activate_pct = params["trail_activate_pct"]  # Only start trailing after this % profit

    # Scale-in
    use_scale_in = params["use_scale_in"]     # bool
    scale_in_pullback_pct = params["scale_in_pullback_pct"]  # pullback from entry to add
    scale_in_bounce_above = params["scale_in_bounce_above"]  # must bounce above this % of entry
    scale_in_size_pct = params["scale_in_size_pct"]  # size of scale-in as % of original

    # ATR for trailing
    atr = None
    if use_trailing == 2:
        atr = compute_atr(highs, lows, closes, period=14)

    # Position tracking
    position = 1.0  # 1.0 = full position
    avg_entry = slip_entry
    total_cost = 1.0  # normalized
    total_pnl = 0.0

    # State
    highest_since_entry = slip_entry
    trailing_active = False
    partial_taken = False
    scaled_in = False
    scale_in_pullback_seen = False

    for j in range(entry_idx + 1, min(entry_idx + time_limit + 1, n)):
        if position <= 0:
            break

        cur_high = highs[j]
        cur_low = lows[j]
        cur_close = closes[j]

        # Track highest price since entry
        if cur_high > highest_since_entry:
            highest_since_entry = cur_high

        # --- Scale-in logic ---
        if use_scale_in and not scaled_in and not partial_taken:
            scale_in_level = avg_entry * (1 - scale_in_pullback_pct / 100)
            if not scale_in_pullback_seen:
                if cur_low <= scale_in_level:
                    scale_in_pullback_seen = True
            else:
                bounce_level = avg_entry * (1 + scale_in_bounce_above / 100)
                if cur_close >= bounce_level:
                    # Scale in
                    add_size = scale_in_size_pct / 100  # fraction of original
                    add_price = cur_close * (1 + SLIPPAGE_PCT / 100)
                    new_total = total_cost + add_size
                    avg_entry = (avg_entry * total_cost + add_price * add_size) / new_total
                    total_cost = new_total
                    position += add_size
                    scaled_in = True

        # --- STOP LOSS (initial or trailing) ---
        if use_trailing > 0 and trailing_active:
            if use_trailing == 1:
                # Fixed % trailing
                trail_stop = highest_since_entry * (1 - trail_pct / 100)
            else:
                # ATR trailing
                cur_atr = atr[j] if atr is not None and j < len(atr) and not np.isnan(atr[j]) else 0
                trail_stop = highest_since_entry - trail_atr_mult * cur_atr

            if cur_low <= trail_stop:
                exit_p = trail_stop * (1 - SLIPPAGE_PCT / 100)
                pnl_pct = (exit_p / avg_entry - 1) * 100 * position
                total_pnl += pnl_pct
                position = 0
                break
        else:
            # Fixed stop
            stop_price = avg_entry * (1 - stop_pct / 100)
            if cur_low <= stop_price:
                exit_p = stop_price * (1 - SLIPPAGE_PCT / 100)
                pnl_pct = (exit_p / avg_entry - 1) * 100 * position
                total_pnl += pnl_pct
                position = 0
                break

        # --- Activate trailing stop ---
        if use_trailing > 0 and not trailing_active:
            unrealized = (cur_high / avg_entry - 1) * 100
            if unrealized >= trail_activate_pct:
                trailing_active = True
                highest_since_entry = cur_high  # reset to current

        # --- PARTIAL SELL at target1 ---
        if not partial_taken and partial_sell_pct > 0:
            tgt1 = avg_entry * (1 + target1_pct / 100)
            if cur_high >= tgt1:
                sell_frac = partial_sell_pct / 100
                exit_p = tgt1 * (1 - SLIPPAGE_PCT / 100)
                pnl_pct = (exit_p / avg_entry - 1) * 100 * sell_frac * (total_cost)
                total_pnl += pnl_pct
                position -= sell_frac * position
                partial_taken = True
                if position <= 0.01:
                    position = 0
                    break
                continue

        # --- FULL TARGET (target2 for runner, or target1 if no partials) ---
        if partial_taken:
            tgt = avg_entry * (1 + target2_pct / 100)
        else:
            tgt = avg_entry * (1 + target1_pct / 100)

        if cur_high >= tgt:
            exit_p = tgt * (1 - SLIPPAGE_PCT / 100)
            pnl_pct = (exit_p / avg_entry - 1) * 100 * position
            total_pnl += pnl_pct
            position = 0
            break

    # Time stop: exit remaining position
    if position > 0.01:
        last = min(entry_idx + time_limit, n - 1)
        exit_p = closes[last] * (1 - SLIPPAGE_PCT / 100)
        pnl_pct = (exit_p / avg_entry - 1) * 100 * position
        total_pnl += pnl_pct

    return [(total_pnl, entry_idx)]


def run_trial(stocks, params):
    """Run pattern on all stocks, return metrics."""
    pnls = []
    entries = []

    for s in stocks:
        results = scan_and_trade_advanced(s, params)
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
    """Optuna objective: maximize risk-adjusted return (total_pnl * sqrt(pf))."""

    # --- Entry parameters (narrowed around known-good values) ---
    min_gap = trial.suggest_int("min_gap", 10, 30, step=5)
    confirm_above = trial.suggest_int("confirm_above", 2, 4)
    confirm_window = trial.suggest_int("confirm_window", 3, 7)
    pullback_pct = trial.suggest_float("pullback_pct", 3.0, 8.0, step=0.5)
    pullback_timeout = trial.suggest_int("pullback_timeout", 10, 30, step=5)
    max_entry_candle = trial.suggest_int("max_entry_candle", 45, 120, step=15)

    if confirm_above > confirm_window:
        return -9999

    # --- Exit strategy mode ---
    # 0 = simple (target+stop), 1 = trailing only, 2 = partials only, 3 = trailing+partials
    exit_mode = trial.suggest_int("exit_mode", 0, 3)

    # --- Core exit params ---
    target1_pct = trial.suggest_float("target1_pct", 3.0, 15.0, step=1.0)
    stop_pct = trial.suggest_float("stop_pct", 3.0, 12.0, step=1.0)
    time_limit = trial.suggest_int("time_limit", 20, 90, step=10)

    # --- Trailing stop ---
    use_trailing = 0
    trail_pct = 5.0
    trail_atr_mult = 2.0
    trail_activate_pct = 3.0

    if exit_mode in (1, 3):
        trail_type = trial.suggest_int("trail_type", 1, 2)  # 1=fixed%, 2=ATR
        use_trailing = trail_type
        if trail_type == 1:
            trail_pct = trial.suggest_float("trail_pct", 2.0, 8.0, step=1.0)
        else:
            trail_atr_mult = trial.suggest_float("trail_atr_mult", 1.0, 4.0, step=0.5)
        trail_activate_pct = trial.suggest_float("trail_activate_pct", 2.0, 8.0, step=1.0)

    # --- Partial selling ---
    partial_sell_pct = 0.0
    target2_pct = target1_pct

    if exit_mode in (2, 3):
        partial_sell_pct = trial.suggest_float("partial_sell_pct", 25.0, 75.0, step=25.0)
        target2_pct = trial.suggest_float("target2_pct", target1_pct + 2, 25.0, step=2.0)

    # --- Scale-in ---
    use_scale_in = trial.suggest_categorical("use_scale_in", [True, False])
    scale_in_pullback_pct = 2.0
    scale_in_bounce_above = 1.0
    scale_in_size_pct = 50.0

    if use_scale_in:
        scale_in_pullback_pct = trial.suggest_float("scale_in_pullback_pct", 1.0, 5.0, step=0.5)
        scale_in_bounce_above = trial.suggest_float("scale_in_bounce_above", 0.5, 3.0, step=0.5)
        scale_in_size_pct = trial.suggest_float("scale_in_size_pct", 25.0, 75.0, step=25.0)

    params = {
        "min_gap": min_gap,
        "confirm_above": confirm_above,
        "confirm_window": confirm_window,
        "pullback_pct": pullback_pct,
        "pullback_timeout": pullback_timeout,
        "max_entry_candle": max_entry_candle,
        "target1_pct": target1_pct,
        "target2_pct": target2_pct,
        "stop_pct": stop_pct,
        "time_limit": time_limit,
        "partial_sell_pct": partial_sell_pct,
        "use_trailing": use_trailing,
        "trail_pct": trail_pct,
        "trail_atr_mult": trail_atr_mult,
        "trail_activate_pct": trail_activate_pct,
        "use_scale_in": use_scale_in,
        "scale_in_pullback_pct": scale_in_pullback_pct,
        "scale_in_bounce_above": scale_in_bounce_above,
        "scale_in_size_pct": scale_in_size_pct,
    }

    result = run_trial(stocks, params)

    if result["n"] < 30:
        return -9999

    trial.set_user_attr("n", result["n"])
    trial.set_user_attr("wr", result["wr"])
    trial.set_user_attr("pf", result["pf"])
    trial.set_user_attr("avg_entry", result["avg_entry"])
    trial.set_user_attr("exit_mode", exit_mode)
    trial.set_user_attr("use_scale_in", use_scale_in)

    # Objective: total P&L weighted by profit factor (reward consistency)
    if result["pf"] < 0.5:
        return -9999
    return result["total_pnl"] * min(result["pf"], 3.0)


if __name__ == "__main__":
    n_trials = N_TRIALS
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--trials" and i + 2 <= len(sys.argv[1:]):
            n_trials = int(sys.argv[i + 2])

    _print(f"Optuna: Pattern P Advanced (Trailing + Partials + Scale-In)")
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

    # Baseline: current P params (simple target+stop)
    _print("Running baseline (current P config)...")
    baseline_params = {
        "min_gap": 15, "confirm_above": 3, "confirm_window": 5,
        "pullback_pct": 5.5, "pullback_timeout": 20,
        "max_entry_candle": 90,
        "target1_pct": 12.0, "target2_pct": 12.0,
        "stop_pct": 10.0, "time_limit": 50,
        "partial_sell_pct": 0, "use_trailing": 0,
        "trail_pct": 5.0, "trail_atr_mult": 2.0, "trail_activate_pct": 3.0,
        "use_scale_in": False,
        "scale_in_pullback_pct": 2.0, "scale_in_bounce_above": 1.0,
        "scale_in_size_pct": 50.0,
    }
    base = run_trial(stocks, baseline_params)
    _print(f"  {base['n']} trades, {base['wr']:.1f}% WR, PF {base['pf']:.2f}, "
          f"P&L {base['total_pnl']:+.0f}%")
    _print(f"  Avg entry candle: {base['avg_entry']:.0f}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="p_advanced_2024_2026",
        storage="sqlite:///optuna_p_advanced_2024_2026.db",
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
            em = bt.user_attrs.get('exit_mode', 0)
            em_names = {0: "simple", 1: "trail", 2: "partial", 3: "trail+partial"}
            si = "scale" if bt.user_attrs.get('use_scale_in') else "no-scale"
            _print(f"  [{n:>3}/{n_trials}] NEW BEST  "
                  f"({bt.user_attrs.get('n',0)} tr, {bt.user_attrs.get('wr',0):.0f}% WR, "
                  f"PF {bt.user_attrs.get('pf',0):.2f})  "
                  f"gap>={bp.get('min_gap',0)}% tgt+{bp.get('target1_pct',0):.0f}% "
                  f"stp-{bp.get('stop_pct',0):.0f}% {bp.get('time_limit',0)}m  "
                  f"[{em_names.get(em,'?')} {si}]  "
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
    _print(f"  OPTIMIZATION COMPLETE")
    _print(f"{'='*70}")
    _print(f"  Best Objective: {best.value:+.0f}")
    _print(f"  vs Baseline:    PnL {base['total_pnl']:+.0f}%")
    _print(f"\n  Trades: {best.user_attrs['n']}  WR: {best.user_attrs['wr']:.1f}%  "
          f"PF: {best.user_attrs['pf']:.2f}")
    _print(f"  Avg Entry Candle: {best.user_attrs['avg_entry']:.0f}")

    em = best.user_attrs.get('exit_mode', 0)
    em_names = {0: "Simple (target+stop)", 1: "Trailing Stop", 2: "Partial Sells", 3: "Trailing + Partials"}
    _print(f"\n  EXIT MODE: {em_names.get(em, '?')}")

    _print(f"\n  ENTRY PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  min_gap:          {bp['min_gap']}%")
    _print(f"  confirm_above:    {bp['confirm_above']} closes > PM high")
    _print(f"  confirm_window:   {bp['confirm_window']} candles")
    _print(f"  pullback_pct:     {bp['pullback_pct']}% from PM high")
    _print(f"  pullback_timeout: {bp['pullback_timeout']} candles")
    _print(f"  max_entry_candle: {bp['max_entry_candle']}")

    _print(f"\n  EXIT PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  target1_pct:      +{bp['target1_pct']}%")
    if 'target2_pct' in bp:
        _print(f"  target2_pct:      +{bp['target2_pct']}% (runner)")
    _print(f"  stop_pct:         -{bp['stop_pct']}%")
    _print(f"  time_limit:       {bp['time_limit']}m")

    if em in (1, 3):
        _print(f"\n  TRAILING STOP")
        _print(f"  {'-'*50}")
        tt = bp.get('trail_type', 0)
        if tt == 1:
            _print(f"  Type: Fixed {bp.get('trail_pct', 0)}%")
        else:
            _print(f"  Type: ATR x{bp.get('trail_atr_mult', 0)}")
        _print(f"  Activates at:   +{bp.get('trail_activate_pct', 0)}% profit")

    if em in (2, 3):
        _print(f"\n  PARTIAL SELLING")
        _print(f"  {'-'*50}")
        _print(f"  Sell {bp.get('partial_sell_pct', 0)}% at +{bp['target1_pct']}%")
        _print(f"  Hold rest for +{bp.get('target2_pct', bp['target1_pct'])}%")

    if bp.get('use_scale_in'):
        _print(f"\n  SCALE-IN")
        _print(f"  {'-'*50}")
        _print(f"  Pullback: -{bp.get('scale_in_pullback_pct', 0)}% from entry")
        _print(f"  Bounce above: +{bp.get('scale_in_bounce_above', 0)}%")
        _print(f"  Add size: {bp.get('scale_in_size_pct', 0)}% of original")

    _print(f"\n{'='*70}")

    # Top 10 trials
    _print(f"\n  TOP 10 TRIALS")
    _print(f"  {'#':>3} {'Obj':>8} {'N':>5} {'WR':>5} {'PF':>5} {'Mode':>12} {'Gap':>4} {'Tgt':>4} {'Stp':>4}")
    _print(f"  {'-'*65}")
    sorted_t = sorted(study.trials, key=lambda t: t.value if t.value else -9999, reverse=True)
    for i, t in enumerate(sorted_t[:10]):
        if t.value is None or t.value <= -9999:
            continue
        p = t.params
        em = t.user_attrs.get('exit_mode', 0)
        em_short = {0: "simple", 1: "trail", 2: "partial", 3: "trail+part"}.get(em, "?")
        si = "+scale" if t.user_attrs.get('use_scale_in') else ""
        _print(f"  {i+1:>3} {t.value:>+7.0f} "
              f"{t.user_attrs.get('n',0):>5} {t.user_attrs.get('wr',0):>4.0f}% "
              f"{t.user_attrs.get('pf',0):>4.2f} "
              f"{em_short+si:>12} "
              f"{p.get('min_gap',0):>3}% +{p.get('target1_pct',0):>2.0f}% "
              f"-{p.get('stop_pct',0):>2.0f}%")
