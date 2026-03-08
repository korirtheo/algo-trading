"""
Optuna Optimizer: Power Hour Breakout Strategy (W)
===================================================
Gap-up stock consolidates all day → 3:00 PM+ volume surge → breaks day high.

Filters (from quant review):
  1. Volume vs morning spike (not just consolidation avg) — kills fake breakouts
  2. Morning run filter (morning_high >= open * min_run%) — filters dead gappers
  3. VWAP position filter (price > VWAP at breakout) — shorts underwater
  4. HOD breaks limit (< max_hod_breaks) — avoids choppy algo names
  5. VWAP deviation for consolidation (max deviation from VWAP < threshold)

Usage:
  python optimize_power_hour.py
  python optimize_power_hour.py --trials 500
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


def _compute_vwap(highs, lows, closes, volumes):
    """Compute cumulative VWAP from numpy arrays."""
    typical = (highs + lows + closes) / 3.0
    cum_tp_vol = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)
    cum_vol[cum_vol == 0] = 1e-9
    return cum_tp_vol / cum_vol


def precompute_stocks(daily_picks, all_dates):
    """Pre-compute arrays for all gap-up stocks.
    Need >= 175 candles (3:50 PM) for power hour pattern."""
    stocks = []
    for d in all_dates:
        picks = daily_picks.get(d, [])
        for pick in picks:
            if pick["gap_pct"] < 5.0:
                continue
            candles = pick["market_hour_candles"]
            if len(candles) < 175:  # need candles through 3:50 PM
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
                "vwap": _compute_vwap(highs, lows, closes, volumes),
                "n": len(candles),
            })
    return stocks


def scan_and_trade_power_hour(stock, params):
    """
    Power Hour Breakout (W):
    1. Verify morning run (gap + intraday move)
    2. Define consolidation from ~candle 30 to entry window
    3. Check VWAP deviation consolidation
    4. Check HOD breaks limit
    5. Enter on breakout above consolidation high with volume surge
    6. Must be above VWAP
    7. Exit: trailing stop + target + hard time stop at 3:55 PM
    """
    if stock["gap_pct"] < params["min_gap"]:
        return []

    opens = stock["opens"]
    highs = stock["highs"]
    lows = stock["lows"]
    closes = stock["closes"]
    volumes = stock["volumes"]
    vwap = stock["vwap"]
    n = stock["n"]

    # Params
    min_gap = params["min_gap"]
    min_morning_run = params["min_morning_run"]
    earliest_candle = params["earliest_candle"]
    latest_candle = params["latest_candle"]
    consol_start = params["consol_start"]
    max_vwap_dev_pct = params["max_vwap_dev_pct"]
    max_range_pct = params["max_range_pct"]
    vol_surge_mult = params["vol_surge_mult"]
    vol_vs_morning_mult = params["vol_vs_morning_mult"]
    max_hod_breaks = params["max_hod_breaks"]
    require_above_vwap = params["require_above_vwap"]

    target_pct = params["target_pct"]
    stop_pct = params["stop_pct"]
    trail_pct = params["trail_pct"]
    trail_activate_pct = params["trail_activate_pct"]
    partial_sell_pct = params["partial_sell_pct"]
    partial_target_pct = params["partial_target_pct"]

    if n < latest_candle + 5:
        return []

    # ---- FILTER 1: Morning run ----
    # Check that stock actually ran in the morning (not just gapped)
    morning_end = min(30, n)  # first 30 candles = first hour
    morning_high = np.max(highs[:morning_end])
    open_price = opens[0]

    morning_run_pct = (morning_high / open_price - 1) * 100
    if morning_run_pct < min_morning_run:
        return []

    # Morning spike volume (max single candle volume in first 30 candles)
    morning_spike_vol = np.max(volumes[:morning_end])

    # ---- FILTER 2: Consolidation quality ----
    # From consol_start to earliest_candle, price should be range-bound
    consol_slice_end = min(earliest_candle, n)
    if consol_start >= consol_slice_end:
        return []

    consol_highs = highs[consol_start:consol_slice_end]
    consol_lows = lows[consol_start:consol_slice_end]
    consol_closes = closes[consol_start:consol_slice_end]
    consol_vols = volumes[consol_start:consol_slice_end]
    consol_vwap = vwap[consol_start:consol_slice_end]

    consol_high = np.max(consol_highs)
    consol_low = np.min(consol_lows)

    # Range check
    if consol_high <= 0:
        return []
    range_pct = (consol_high - consol_low) / consol_high * 100
    if range_pct > max_range_pct:
        return []

    # VWAP deviation check: max abs deviation from VWAP during consolidation
    if len(consol_vwap) > 0 and np.all(consol_vwap > 0):
        vwap_devs = np.abs(consol_closes - consol_vwap) / consol_vwap * 100
        max_dev = np.max(vwap_devs)
        if max_dev > max_vwap_dev_pct:
            return []

    # Consolidation avg volume (for surge comparison)
    consol_avg_vol = np.mean(consol_vols) if len(consol_vols) > 0 else 0

    # ---- FILTER 3: HOD breaks limit ----
    # Count how many times price broke above prior HOD before our window
    hod_breaks = 0
    running_hod = highs[0]
    for i in range(1, earliest_candle):
        if i >= n:
            break
        if highs[i] > running_hod:
            hod_breaks += 1
            running_hod = highs[i]
    if hod_breaks > max_hod_breaks:
        return []

    # ---- ENTRY SCAN: Power hour window ----
    # Look for breakout above consolidation high with volume surge
    # Use consolidation high (NOT morning spike high) as breakout level
    breakout_level = consol_high
    entry_idx = None
    entry_price = None

    for i in range(earliest_candle, min(latest_candle + 1, n)):
        # Breakout: close above consolidation high
        if closes[i] <= breakout_level:
            continue

        # Volume surge: current candle vs consolidation avg
        if consol_avg_vol > 0 and volumes[i] < consol_avg_vol * vol_surge_mult:
            continue

        # Volume vs morning spike (kills fake breakouts)
        if morning_spike_vol > 0 and volumes[i] < morning_spike_vol * vol_vs_morning_mult:
            continue

        # VWAP filter: must be above VWAP
        if require_above_vwap and i < len(vwap) and closes[i] < vwap[i]:
            continue

        entry_idx = i
        entry_price = closes[i]
        break

    if entry_idx is None:
        return []

    # ---- TRADE SIMULATION ----
    slip_entry = entry_price * (1 + SLIPPAGE_PCT / 100)

    # Hard exit at candle ~178 (3:56 PM) — only ~20 min of runway
    hard_exit_candle = 178
    max_trade_candles = hard_exit_candle - entry_idx

    position = 1.0
    avg_entry = slip_entry
    total_pnl = 0.0
    highest_since_entry = slip_entry
    trailing_active = False
    partial_taken = False

    for j in range(entry_idx + 1, min(entry_idx + max_trade_candles + 1, n)):
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

        # Partial sell at first target
        if not partial_taken and partial_sell_pct > 0:
            tgt1 = avg_entry * (1 + partial_target_pct / 100)
            if cur_high >= tgt1:
                sell_frac = partial_sell_pct / 100
                exit_p = tgt1 * (1 - SLIPPAGE_PCT / 100)
                total_pnl += (exit_p / avg_entry - 1) * 100 * sell_frac * position
                position *= (1 - sell_frac)
                partial_taken = True
                if position <= 0.01:
                    position = 0
                    break

        # Full target
        full_tgt = avg_entry * (1 + target_pct / 100)
        if cur_high >= full_tgt:
            exit_p = full_tgt * (1 - SLIPPAGE_PCT / 100)
            total_pnl += (exit_p / avg_entry - 1) * 100 * position
            position = 0
            break

    # Time stop — hard exit at 3:55 PM
    if position > 0.01:
        last = min(entry_idx + max_trade_candles, n - 1)
        exit_p = closes[last] * (1 - SLIPPAGE_PCT / 100)
        total_pnl += (exit_p / avg_entry - 1) * 100 * position

    return [(total_pnl, entry_idx)]


def run_trial(stocks, params):
    """Run pattern on all stocks, return metrics."""
    pnls = []
    entries = []

    for s in stocks:
        results = scan_and_trade_power_hour(s, params)
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

    # Entry filters
    min_gap = trial.suggest_int("min_gap", 5, 20, step=3)
    min_morning_run = trial.suggest_int("min_morning_run", 3, 15, step=2)

    # Consolidation — wide ranges, gap-up stocks are volatile
    consol_start = trial.suggest_int("consol_start", 20, 60, step=5)
    max_range_pct = trial.suggest_int("max_range_pct", 8, 25)
    max_vwap_dev_pct = trial.suggest_float("max_vwap_dev_pct", 3.0, 12.0, step=1.0)

    # Entry window (2-min candles: candle 150 = 3:00 PM, candle 180 = 4:00 PM)
    earliest_candle = trial.suggest_int("earliest_candle", 140, 165, step=5)
    latest_candle = trial.suggest_int("latest_candle", 170, 180, step=5)

    # Volume
    vol_surge_mult = trial.suggest_float("vol_surge_mult", 0.5, 3.0, step=0.5)
    vol_vs_morning_mult = trial.suggest_float("vol_vs_morning_mult", 0.1, 0.6, step=0.1)

    # Quality filters
    max_hod_breaks = trial.suggest_int("max_hod_breaks", 2, 8)
    require_above_vwap = trial.suggest_categorical("require_above_vwap", [True, False])

    # Exit params — tight for late-day
    target_pct = trial.suggest_float("target_pct", 2.0, 6.0, step=0.5)
    stop_pct = trial.suggest_float("stop_pct", 1.5, 5.0, step=0.5)
    trail_pct = trial.suggest_float("trail_pct", 1.0, 3.0, step=0.5)
    trail_activate_pct = trial.suggest_float("trail_activate_pct", 1.0, 4.0, step=0.5)

    # Partial sell
    partial_sell_pct = trial.suggest_float("partial_sell_pct", 0.0, 50.0, step=25.0)
    partial_target_pct = trial.suggest_float("partial_target_pct", 1.0, 4.0, step=0.5)

    params = {
        "min_gap": min_gap,
        "min_morning_run": min_morning_run,
        "consol_start": consol_start,
        "max_range_pct": max_range_pct,
        "max_vwap_dev_pct": max_vwap_dev_pct,
        "earliest_candle": earliest_candle,
        "latest_candle": latest_candle,
        "vol_surge_mult": vol_surge_mult,
        "vol_vs_morning_mult": vol_vs_morning_mult,
        "max_hod_breaks": max_hod_breaks,
        "require_above_vwap": require_above_vwap,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "trail_pct": trail_pct,
        "trail_activate_pct": trail_activate_pct,
        "partial_sell_pct": partial_sell_pct,
        "partial_target_pct": partial_target_pct,
    }

    result = run_trial(stocks, params)

    if result["n"] < 20:  # lower threshold — rare strategy
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

    _print(f"Optuna: Power Hour Breakout Strategy (W)")
    _print(f"{'='*70}")
    _print(f"  Trials:  {n_trials}")
    _print(f"  Data:    {DATA_DIRS}")
    _print(f"  Dates:   {DATE_RANGE[0]} to {DATE_RANGE[1]}")
    _print(f"  Filters: morning run, VWAP, HOD breaks, vol surge")
    _print(f"{'='*70}\n")

    _print("Loading data...")
    t0 = time.time()
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
    _print(f"  {len(all_dates)} trading days in {time.time()-t0:.1f}s")

    _print("Pre-computing arrays (need >= 175 candles)...")
    stocks = precompute_stocks(daily_picks, all_dates)
    _print(f"  {len(stocks)} stock-days with gap >= 5% and >= 175 candles\n")

    # Diagnostic: count where stocks fail each filter
    _print("Running diagnostics (loose filters)...")
    pass_gap, pass_run, pass_range, pass_vwap_dev = 0, 0, 0, 0
    pass_hod, pass_entry, pass_entry_no_vwap = 0, 0, 0
    for s in stocks:
        if s["gap_pct"] < 5:
            continue
        pass_gap += 1
        h, o, c, v, vw = s["highs"], s["opens"], s["closes"], s["volumes"], s["vwap"]
        nn = s["n"]
        mh = np.max(h[:min(30, nn)])
        if (mh / o[0] - 1) * 100 < 3:
            continue
        pass_run += 1
        ce, cs = min(150, nn), 30
        if ce <= cs:
            continue
        ch, cl = np.max(h[cs:ce]), np.min(s["lows"][cs:ce])
        rp = (ch - cl) / ch * 100 if ch > 0 else 999
        if rp > 20:
            continue
        pass_range += 1
        cc = c[cs:ce]
        cvw = vw[cs:ce]
        if len(cvw) > 0 and np.all(cvw > 0):
            md = np.max(np.abs(cc - cvw) / cvw * 100)
            if md > 10:
                continue
        pass_vwap_dev += 1
        # HOD breaks
        hb = 0
        rh = h[0]
        for ii in range(1, min(150, nn)):
            if h[ii] > rh:
                hb += 1
                rh = h[ii]
        if hb > 6:
            continue
        pass_hod += 1
        # Entry scan (consol high breakout, any volume)
        cav = np.mean(v[cs:ce]) if ce > cs else 0
        for ii in range(140, min(176, nn)):
            if c[ii] > ch:
                pass_entry_no_vwap += 1
                if ii < len(vw) and c[ii] >= vw[ii]:
                    pass_entry += 1
                break
    _print(f"  gap>=5%:         {pass_gap}")
    _print(f"  +run>=3%:        {pass_run}")
    _print(f"  +range<=20%:     {pass_range}")
    _print(f"  +vwap_dev<=10%:  {pass_vwap_dev}")
    _print(f"  +hod<=6:         {pass_hod}")
    _print(f"  +breakout found: {pass_entry_no_vwap} (no VWAP filter)")
    _print(f"  +above VWAP:     {pass_entry}")
    _print()

    # Baseline with moderate params
    _print("Running baseline...")
    baseline_params = {
        "min_gap": 5, "min_morning_run": 3,
        "consol_start": 30, "max_range_pct": 18,
        "max_vwap_dev_pct": 8.0,
        "earliest_candle": 150, "latest_candle": 175,
        "vol_surge_mult": 1.0, "vol_vs_morning_mult": 0.1,
        "max_hod_breaks": 6, "require_above_vwap": False,
        "target_pct": 4.0, "stop_pct": 3.0,
        "trail_pct": 2.0, "trail_activate_pct": 2.0,
        "partial_sell_pct": 0.0, "partial_target_pct": 2.0,
    }
    base = run_trial(stocks, baseline_params)
    _print(f"  {base['n']} trades, {base['wr']:.1f}% WR, PF {base['pf']:.2f}, "
          f"P&L {base['total_pnl']:+.1f}%")
    if base['n'] > 0:
        _print(f"  Avg entry candle: {base['avg_entry']:.0f} "
              f"(~{9*60 + 30 + base['avg_entry']*2:.0f} mins from 9:30 = "
              f"{int((9*60+30+base['avg_entry']*2)//60)}:{int((9*60+30+base['avg_entry']*2)%60):02d})")
    _print()

    study = optuna.create_study(
        direction="maximize",
        study_name="power_hour_2024_2026",
        storage="sqlite:///optuna_power_hour.db",
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
            _print(f"\n  *** NEW BEST (trial {n}) ***")
            _print(f"      Score: {best:+.0f} | {bt.user_attrs.get('n',0)} trades, "
                  f"{bt.user_attrs.get('wr',0):.0f}% WR, PF {bt.user_attrs.get('pf',0):.2f}")
            _print(f"      gap>={bp.get('min_gap',0)}% run>={bp.get('min_morning_run',0)}% "
                  f"range<={bp.get('max_range_pct',0)}% vwap_dev<={bp.get('max_vwap_dev_pct',0):.0f}%")
            _print(f"      window={bp.get('earliest_candle',0)}-{bp.get('latest_candle',0)} "
                  f"vol>={bp.get('vol_surge_mult',0):.1f}x "
                  f"hod<={bp.get('max_hod_breaks',0)} "
                  f"vwap={'Y' if bp.get('require_above_vwap') else 'N'}")
            _print(f"      tgt+{bp.get('target_pct',0):.1f}% stp-{bp.get('stop_pct',0):.1f}% "
                  f"trail{bp.get('trail_pct',0):.1f}%@+{bp.get('trail_activate_pct',0):.1f}%")
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
    _print(f"  OPTIMIZATION COMPLETE — Power Hour Breakout (W)")
    _print(f"{'='*70}")
    _print(f"  Best Objective:  {best.value:+.0f}")
    _print(f"  vs Baseline:     PnL {base['total_pnl']:+.1f}%")
    if best.value <= -9999:
        _print(f"\n  NO VALID TRIALS FOUND — all pruned or zero trades")
        _print(f"  Check diagnostic output above. Filters may be too strict.")
        _print(f"{'='*70}")
        sys.exit(0)
    _print(f"\n  Trades: {best.user_attrs['n']}  WR: {best.user_attrs['wr']:.1f}%  "
          f"PF: {best.user_attrs['pf']:.2f}")
    avg_e = best.user_attrs['avg_entry']
    entry_time_min = 9*60 + 30 + avg_e * 2
    _print(f"  Avg Entry Candle: {avg_e:.0f} (~{int(entry_time_min//60)}:{int(entry_time_min%60):02d})")

    _print(f"\n  ENTRY FILTERS")
    _print(f"  {'-'*50}")
    _print(f"  min_gap:            {bp['min_gap']}%")
    _print(f"  min_morning_run:    {bp['min_morning_run']}%")
    _print(f"  consol_start:       candle {bp['consol_start']}")
    _print(f"  max_range_pct:      {bp['max_range_pct']}%")
    _print(f"  max_vwap_dev_pct:   {bp['max_vwap_dev_pct']:.1f}%")
    _print(f"  earliest_candle:    {bp['earliest_candle']}")
    _print(f"  latest_candle:      {bp['latest_candle']}")
    _print(f"  vol_surge_mult:     {bp['vol_surge_mult']:.1f}x")
    _print(f"  vol_vs_morning:     {bp['vol_vs_morning_mult']:.1f}x")
    _print(f"  max_hod_breaks:     {bp['max_hod_breaks']}")
    _print(f"  require_above_vwap: {bp['require_above_vwap']}")

    _print(f"\n  EXIT PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  target_pct:         +{bp['target_pct']:.1f}%")
    _print(f"  stop_pct:           -{bp['stop_pct']:.1f}%")
    _print(f"  trail_pct:          {bp['trail_pct']:.1f}%")
    _print(f"  trail_activate:     +{bp['trail_activate_pct']:.1f}%")
    if bp.get('partial_sell_pct', 0) > 0:
        _print(f"  partial_sell:       {bp['partial_sell_pct']:.0f}% at +{bp['partial_target_pct']:.1f}%")

    _print(f"\n{'='*70}")

    # Top 10 trials
    _print(f"\n  TOP 10 TRIALS")
    _print(f"  {'#':>3} {'Obj':>8} {'N':>5} {'WR':>5} {'PF':>5} {'Gap':>4} "
          f"{'Run':>4} {'Range':>5} {'HOD':>4} {'VWAP':>5} {'Tgt':>4} {'Stp':>4}")
    _print(f"  {'-'*75}")
    sorted_t = sorted(study.trials, key=lambda t: t.value if t.value else -9999, reverse=True)
    for i, t in enumerate(sorted_t[:10]):
        if t.value is None or t.value <= -9999:
            continue
        p = t.params
        _print(f"  {i+1:>3} {t.value:>+7.0f} "
              f"{t.user_attrs.get('n',0):>5} {t.user_attrs.get('wr',0):>4.0f}% "
              f"{t.user_attrs.get('pf',0):>4.2f} "
              f"{p.get('min_gap',0):>3}% "
              f"{p.get('min_morning_run',0):>3}% "
              f"{p.get('max_range_pct',0):>4}% "
              f"{p.get('max_hod_breaks',0):>3} "
              f"{'Y' if p.get('require_above_vwap') else 'N':>4} "
              f"+{p.get('target_pct',0):>2.0f}% "
              f"-{p.get('stop_pct',0):>2.0f}%")
