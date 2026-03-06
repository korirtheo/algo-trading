"""
Optuna Optimizer for Combined Green Candle Strategy (H + G + A + F)
===================================================================
Optimizes all strategy parameters: gap thresholds, body filters,
target percentages, and time stops for each strategy.

Uses compounding (100% balance sizing, MAX_POSITIONS=1).
No SPY regime filter.

Usage:
  python optimize_combined.py              # 300 trials (default)
  python optimize_combined.py --trials 500 # custom trial count
"""
import os
import sys
import time
import numpy as np
import optuna

from test_full import (
    load_all_picks,
    SLIPPAGE_PCT,
    STARTING_CASH,
    MARGIN_THRESHOLD,
    VOL_CAP_PCT,
    ET_TZ,
)
import test_green_candle_combined as tgcc

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Force line-buffered stdout so output appears in real-time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

_builtin_print = print
def _print(*args, **kwargs):
    """Print with immediate flush."""
    kwargs.setdefault("flush", True)
    _builtin_print(*args, **kwargs)

N_TRIALS = 300
DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2025-01-01", "2026-02-28")


def run_backtest(picks_by_date, all_dates, params):
    """Run full backtest with given params, return metrics."""
    # Patch module-level config — H
    tgcc.H_MIN_GAP_PCT = params["h_min_gap"]
    tgcc.H_MIN_BODY_PCT = params["h_min_body"]
    tgcc.H_REQUIRE_VOL_CONFIRM = params["h_vol_confirm"]
    tgcc.H_TARGET_PCT = params["h_target"]
    tgcc.H_TIME_LIMIT_MINUTES = params["h_time"]

    # Patch module-level config — G
    tgcc.G_MIN_GAP_PCT = params["g_min_gap"]
    tgcc.G_MIN_BODY_PCT = params["g_min_body"]
    tgcc.G_TARGET_PCT = params["g_target"]
    tgcc.G_TIME_LIMIT_MINUTES = params["g_time"]

    # Patch module-level config — A
    tgcc.A_MIN_GAP_PCT = params["a_min_gap"]
    tgcc.A_MIN_BODY_PCT = params["a_min_body"]
    tgcc.A_TARGET_PCT = params["a_target"]
    tgcc.A_TIME_LIMIT_MINUTES = params["a_time"]

    # Patch module-level config — F
    tgcc.F_MIN_GAP_PCT = params["f_min_gap"]
    tgcc.F_MIN_BODY_PCT = params["f_min_body"]
    tgcc.F_TARGET_PCT = params["f_target"]
    tgcc.F_TIME_LIMIT_MINUTES = params["f_time"]

    cash = STARTING_CASH
    unsettled = 0.0
    total_trades = 0
    total_wins = 0
    h_trades = 0
    g_trades = 0
    a_trades = 0
    f_trades = 0
    daily_pnls = []
    peak = cash
    max_dd_pct = 0.0

    for d in all_dates:
        if unsettled > 0:
            cash += unsettled
            unsettled = 0.0

        picks = picks_by_date.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        states, new_cash, new_unsettled = tgcc.simulate_day_combined(
            picks, cash, cash_account
        )

        day_pnl = 0.0
        for st in states:
            if st["exit_reason"] is not None:
                total_trades += 1
                day_pnl += st["pnl"]
                if st["pnl"] > 0:
                    total_wins += 1
                s = st["strategy"]
                if s == "H":
                    h_trades += 1
                elif s == "G":
                    g_trades += 1
                elif s == "A":
                    a_trades += 1
                elif s == "F":
                    f_trades += 1

        daily_pnls.append(day_pnl)
        cash = new_cash
        unsettled = new_unsettled

        # Track drawdown
        equity = cash + unsettled
        if equity > peak:
            peak = equity
        dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    final_equity = cash + unsettled
    win_rate = total_wins / max(total_trades, 1) * 100
    sharpe = 0.0
    if daily_pnls and np.std(daily_pnls) > 0:
        sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252)

    return {
        "final_equity": final_equity,
        "return_pct": (final_equity / STARTING_CASH - 1) * 100,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "h_trades": h_trades,
        "g_trades": g_trades,
        "a_trades": a_trades,
        "f_trades": f_trades,
        "daily_pnls": daily_pnls,
    }


def objective(trial, picks_by_date, all_dates):
    """Optuna objective: maximize final equity."""
    params = {
        # Strategy H: High Conviction
        "h_min_gap": trial.suggest_int("h_min_gap", 20, 40, step=5),
        "h_min_body": trial.suggest_float("h_min_body", 3.0, 8.0, step=1.0),
        "h_vol_confirm": trial.suggest_categorical("h_vol_confirm", [True, False]),
        "h_target": trial.suggest_float("h_target", 8.0, 16.0, step=1.0),
        "h_time": trial.suggest_int("h_time", 10, 25, step=5),

        # Strategy G: Big Gap Runner
        "g_min_gap": trial.suggest_int("g_min_gap", 20, 40, step=5),
        "g_min_body": 0.0,  # Fixed: no body filter for G
        "g_target": trial.suggest_float("g_target", 5.0, 14.0, step=1.0),
        "g_time": trial.suggest_int("g_time", 10, 30, step=2),

        # Strategy A: Quick Scalp
        "a_min_gap": trial.suggest_int("a_min_gap", 10, 25, step=5),
        "a_min_body": trial.suggest_float("a_min_body", 0.0, 4.0, step=1.0),
        "a_target": trial.suggest_float("a_target", 2.0, 6.0, step=0.5),
        "a_time": trial.suggest_int("a_time", 4, 16, step=2),

        # Strategy F: Catch-All
        "f_min_gap": trial.suggest_int("f_min_gap", 5, 15, step=5),
        "f_min_body": 0.0,  # Fixed: no body filter for F
        "f_target": trial.suggest_float("f_target", 3.0, 12.0, step=1.0),
        "f_time": trial.suggest_int("f_time", 1, 8, step=1),
    }

    # Constraints: gap thresholds must cascade H >= G > A > F
    if params["h_min_gap"] < params["g_min_gap"]:
        return 0.0
    if params["g_min_gap"] <= params["a_min_gap"]:
        return 0.0
    if params["a_min_gap"] <= params["f_min_gap"]:
        return 0.0

    result = run_backtest(picks_by_date, all_dates, params)

    # Penalty: reject if too few trades (overfitting to lucky trades)
    if result["total_trades"] < 100:
        return 0.0

    # Penalty: reject if max drawdown > 50%
    if result["max_dd_pct"] > 50:
        return result["final_equity"] * 0.5

    # Store metrics for reporting
    trial.set_user_attr("return_pct", result["return_pct"])
    trial.set_user_attr("total_trades", result["total_trades"])
    trial.set_user_attr("win_rate", result["win_rate"])
    trial.set_user_attr("sharpe", result["sharpe"])
    trial.set_user_attr("max_dd_pct", result["max_dd_pct"])
    trial.set_user_attr("h_trades", result["h_trades"])
    trial.set_user_attr("g_trades", result["g_trades"])
    trial.set_user_attr("a_trades", result["a_trades"])
    trial.set_user_attr("f_trades", result["f_trades"])

    return result["final_equity"]


if __name__ == "__main__":
    # Parse args
    n_trials = N_TRIALS
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--trials" and i + 2 <= len(sys.argv[1:]):
            n_trials = int(sys.argv[i + 2])

    _print(f"Optuna Optimizer: Combined H+G+A+F Strategy")
    _print(f"{'='*60}")
    _print(f"  Trials:    {n_trials}")
    _print(f"  Start:     ${STARTING_CASH:,}")
    _print(f"  Sizing:    100% balance, MAX_POSITIONS=1")
    _print(f"  Data:      {DATA_DIRS}")
    _print(f"  Dates:     {DATE_RANGE[0]} to {DATE_RANGE[1]}")
    _print(f"{'='*60}\n")

    # Load data once
    _print("Loading data...")
    t0 = time.time()
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
    _print(f"  {len(all_dates)} trading days loaded in {time.time()-t0:.1f}s\n")

    # Run baseline (current config)
    _print("Running baseline (current config)...")
    baseline_params = {
        "h_min_gap": tgcc.H_MIN_GAP_PCT,
        "h_min_body": tgcc.H_MIN_BODY_PCT,
        "h_vol_confirm": tgcc.H_REQUIRE_VOL_CONFIRM,
        "h_target": tgcc.H_TARGET_PCT,
        "h_time": tgcc.H_TIME_LIMIT_MINUTES,
        "g_min_gap": tgcc.G_MIN_GAP_PCT,
        "g_min_body": tgcc.G_MIN_BODY_PCT,
        "g_target": tgcc.G_TARGET_PCT,
        "g_time": tgcc.G_TIME_LIMIT_MINUTES,
        "a_min_gap": tgcc.A_MIN_GAP_PCT,
        "a_min_body": tgcc.A_MIN_BODY_PCT,
        "a_target": tgcc.A_TARGET_PCT,
        "a_time": tgcc.A_TIME_LIMIT_MINUTES,
        "f_min_gap": tgcc.F_MIN_GAP_PCT,
        "f_min_body": tgcc.F_MIN_BODY_PCT,
        "f_target": tgcc.F_TARGET_PCT,
        "f_time": tgcc.F_TIME_LIMIT_MINUTES,
    }
    baseline = run_backtest(daily_picks, all_dates, baseline_params)
    _print(f"  Baseline: ${baseline['final_equity']:,.0f} ({baseline['return_pct']:+.1f}%)")
    _print(f"            {baseline['total_trades']} trades, {baseline['win_rate']:.1f}% WR, "
          f"Sharpe {baseline['sharpe']:.2f}, DD {baseline['max_dd_pct']:.1f}%")
    _print(f"            H:{baseline['h_trades']} G:{baseline['g_trades']} "
          f"A:{baseline['a_trades']} F:{baseline['f_trades']}\n")

    # Create study (fresh — delete old DB if schema changed)
    db_path = "optuna_gc_hgaf.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="gc_hgaf_optimize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    # Progress callback
    best_so_far = [0]
    start_time = time.time()

    def progress_callback(study, trial):
        elapsed = time.time() - start_time
        n = trial.number + 1
        val = trial.value if trial.value else 0
        best = study.best_value
        if best > best_so_far[0]:
            best_so_far[0] = best
            bp = study.best_params
            wr = study.best_trial.user_attrs.get("win_rate", 0)
            trades = study.best_trial.user_attrs.get("total_trades", 0)
            dd = study.best_trial.user_attrs.get("max_dd_pct", 0)
            _print(f"  [{n:>3}/{n_trials}] NEW BEST ${best:>12,.0f}  "
                  f"({trades} tr, {wr:.0f}% WR, DD {dd:.0f}%)  "
                  f"H:{bp.get('h_target',0):.0f}%/{bp.get('h_time',0)}m  "
                  f"G:{bp.get('g_target',0):.0f}%/{bp.get('g_time',0)}m  "
                  f"A:{bp.get('a_target',0):.1f}%/{bp.get('a_time',0)}m  "
                  f"F:{bp.get('f_target',0):.0f}%/{bp.get('f_time',0)}m  "
                  f"[{elapsed:.0f}s]")
        elif n % 25 == 0:
            _print(f"  [{n:>3}/{n_trials}] trial ${val:>12,.0f}  best ${best:>12,.0f}  [{elapsed:.0f}s]")

    _print(f"Running {n_trials} Optuna trials...")
    _print("-" * 90)
    study.optimize(
        lambda trial: objective(trial, daily_picks, all_dates),
        n_trials=n_trials,
        callbacks=[progress_callback],
    )

    # Results
    best = study.best_trial
    bp = best.params
    _print(f"\n{'='*90}")
    _print(f"  OPTIMIZATION COMPLETE")
    _print(f"{'='*90}")
    _print(f"  Best Final Equity: ${best.value:,.0f} ({best.user_attrs['return_pct']:+.1f}%)")
    _print(f"  vs Baseline:       ${baseline['final_equity']:,.0f} ({baseline['return_pct']:+.1f}%)")
    improvement = best.value - baseline['final_equity']
    _print(f"  Improvement:       ${improvement:+,.0f}")
    _print(f"\n  Trades: {best.user_attrs['total_trades']}  "
          f"WR: {best.user_attrs['win_rate']:.1f}%  "
          f"Sharpe: {best.user_attrs['sharpe']:.2f}  "
          f"Max DD: {best.user_attrs['max_dd_pct']:.1f}%")
    _print(f"  H:{best.user_attrs['h_trades']} G:{best.user_attrs['g_trades']} "
          f"A:{best.user_attrs['a_trades']} F:{best.user_attrs['f_trades']}")

    _print(f"\n  OPTIMAL PARAMETERS")
    _print(f"  {'-'*50}")
    _print(f"  Strategy H (High Conviction):")
    _print(f"    H_MIN_GAP_PCT       = {bp['h_min_gap']}")
    _print(f"    H_MIN_BODY_PCT      = {bp['h_min_body']}")
    _print(f"    H_REQUIRE_VOL_CONFIRM = {bp['h_vol_confirm']}")
    _print(f"    H_TARGET_PCT        = {bp['h_target']}")
    _print(f"    H_TIME_LIMIT        = {bp['h_time']} min")
    _print(f"  Strategy G (Big Gap Runner):")
    _print(f"    G_MIN_GAP_PCT = {bp['g_min_gap']}")
    _print(f"    G_TARGET_PCT  = {bp['g_target']}")
    _print(f"    G_TIME_LIMIT  = {bp['g_time']} min")
    _print(f"  Strategy A (Quick Scalp):")
    _print(f"    A_MIN_GAP_PCT  = {bp['a_min_gap']}")
    _print(f"    A_MIN_BODY_PCT = {bp['a_min_body']}")
    _print(f"    A_TARGET_PCT   = {bp['a_target']}")
    _print(f"    A_TIME_LIMIT   = {bp['a_time']} min")
    _print(f"  Strategy F (Catch-All):")
    _print(f"    F_MIN_GAP_PCT = {bp['f_min_gap']}")
    _print(f"    F_TARGET_PCT  = {bp['f_target']}")
    _print(f"    F_TIME_LIMIT  = {bp['f_time']} min")
    _print(f"{'='*90}")

    # Top 10 trials
    _print(f"\n  TOP 10 TRIALS")
    _print(f"  {'#':>3} {'Equity':>14} {'Return':>8} {'Tr':>4} {'WR':>5} {'DD':>5} "
          f"{'H_t':>4} {'H_m':>3} {'G_t':>4} {'G_m':>3} {'A_t':>4} {'A_m':>3} {'F_t':>4} {'F_m':>3}")
    _print(f"  {'-'*90}")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(sorted_trials[:10]):
        if t.value is None or t.value <= 0:
            continue
        p = t.params
        _print(f"  {i+1:>3} ${t.value:>13,.0f} {t.user_attrs.get('return_pct',0):>+7.0f}% "
              f"{t.user_attrs.get('total_trades',0):>4} {t.user_attrs.get('win_rate',0):>4.0f}% "
              f"{t.user_attrs.get('max_dd_pct',0):>4.0f}% "
              f"{p.get('h_target',0):>4.0f} {p.get('h_time',0):>3} "
              f"{p.get('g_target',0):>4.0f} {p.get('g_time',0):>3} "
              f"{p.get('a_target',0):>4.1f} {p.get('a_time',0):>3} "
              f"{p.get('f_target',0):>4.0f} {p.get('f_time',0):>3}")
