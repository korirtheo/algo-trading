"""
Optuna Optimizer for Strategy R2G: Red-to-Green Move
====================================================
Optimizes all R2G strategy parameters independently.
Each trial runs the full backtest with modified globals.

Usage:
  python optimize_r2g.py                  # 500 trials (default)
  python optimize_r2g.py --trials 5       # quick smoke test
  python optimize_r2g.py --trials 1000    # extended search
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

import test_red_to_green as r2g
from test_full import load_all_picks, SLIPPAGE_PCT, STARTING_CASH, MARGIN_THRESHOLD

DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2024-01-01", "2026-02-28")


def set_strategy_params(params):
    """Set all R2G globals on the r2g module."""
    r2g.R2G_MIN_GAP_PCT = float(params["r2g_min_gap_pct"])
    r2g.R2G_MIN_DIP_PCT = params["r2g_min_dip_pct"]
    r2g.R2G_MAX_DIP_PCT = params["r2g_max_dip_pct"]
    r2g.R2G_EARLIEST_CANDLE = params["r2g_earliest_candle"]
    r2g.R2G_LATEST_CANDLE = params["r2g_latest_candle"]
    r2g.R2G_VOL_SURGE_MULT = params["r2g_vol_surge_mult"]
    r2g.R2G_MIN_RECLAIM_BODY_PCT = params["r2g_min_reclaim_body_pct"]
    r2g.R2G_REQUIRE_ABOVE_VWAP = params["r2g_require_above_vwap"]
    r2g.R2G_REQUIRE_CLOSE_ABOVE_OPEN = params["r2g_require_close_above_open"]
    r2g.R2G_TARGET_PCT = params["r2g_target_pct"]
    r2g.R2G_STOP_PCT = params["r2g_stop_pct"]
    r2g.R2G_TRAIL_PCT = params["r2g_trail_pct"]
    r2g.R2G_TRAIL_ACTIVATE_PCT = params["r2g_trail_activate_pct"]
    r2g.R2G_PARTIAL_SELL_PCT = params["r2g_partial_sell_pct"]
    r2g.R2G_PARTIAL_TARGET_PCT = params["r2g_partial_target_pct"]
    r2g.R2G_TIME_LIMIT_MINUTES = params["r2g_time_limit_min"]


def suggest_all_params(trial):
    """Suggest all R2G strategy parameters."""
    params = {}

    # --- FILTERS ---
    params["r2g_min_gap_pct"] = trial.suggest_int("r2g_min_gap_pct", 5, 40, step=5)

    # --- DIP RANGE ---
    params["r2g_min_dip_pct"] = trial.suggest_float("r2g_min_dip_pct", 0.5, 5.0, step=0.5)
    params["r2g_max_dip_pct"] = trial.suggest_float("r2g_max_dip_pct", 5.0, 25.0, step=2.5)

    # --- ENTRY TIMING ---
    params["r2g_earliest_candle"] = trial.suggest_int("r2g_earliest_candle", 2, 10)
    params["r2g_latest_candle"] = trial.suggest_int("r2g_latest_candle", 20, 120, step=10)

    # --- ENTRY CONDITIONS ---
    params["r2g_vol_surge_mult"] = trial.suggest_float("r2g_vol_surge_mult", 0.5, 4.0, step=0.5)
    params["r2g_min_reclaim_body_pct"] = trial.suggest_float("r2g_min_reclaim_body_pct", 0.0, 3.0, step=0.5)
    params["r2g_require_above_vwap"] = trial.suggest_categorical("r2g_require_above_vwap", [True, False])
    params["r2g_require_close_above_open"] = trial.suggest_categorical("r2g_require_close_above_open", [True, False])

    # --- EXIT PARAMS ---
    params["r2g_target_pct"] = trial.suggest_float("r2g_target_pct", 3.0, 25.0, step=1.0)
    params["r2g_stop_pct"] = trial.suggest_float("r2g_stop_pct", 2.0, 12.0, step=1.0)
    params["r2g_trail_pct"] = trial.suggest_float("r2g_trail_pct", 1.0, 8.0, step=0.5)
    params["r2g_trail_activate_pct"] = trial.suggest_float("r2g_trail_activate_pct", 1.0, 10.0, step=1.0)
    params["r2g_partial_sell_pct"] = trial.suggest_float("r2g_partial_sell_pct", 0.0, 75.0, step=25.0)
    # Partial target must be <= main target
    target = params["r2g_target_pct"]
    params["r2g_partial_target_pct"] = trial.suggest_float("r2g_partial_target_pct", 2.0, max(3.0, target - 1), step=1.0)
    params["r2g_time_limit_min"] = trial.suggest_int("r2g_time_limit_min", 15, 120, step=15)

    return params


def run_backtest(daily_picks, all_dates):
    """Run the full R2G backtest and return stats."""
    cash = float(STARTING_CASH)
    unsettled = 0.0
    all_trades = []
    daily_pnls = []

    for d in all_dates:
        cash += unsettled
        unsettled = 0.0

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        states, cash, unsettled = r2g.simulate_day_r2g(picks, cash, cash_account)

        day_pnl = 0.0
        for st in states:
            if st["exit_reason"] is not None:
                all_trades.append({
                    "pnl": st["pnl"],
                    "position_cost": st["position_cost"],
                    "gap_pct": st["gap_pct"],
                    "lowest_below_open": st["lowest_below_open"],
                    "exit_reason": st["exit_reason"],
                })
                day_pnl += st["pnl"]

        daily_pnls.append(day_pnl)

    equity = cash + unsettled
    n = len(all_trades)
    if n == 0:
        return {"n": 0, "pf": 0, "total_pnl": -9999, "equity": equity}

    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in all_trades if t["pnl"] <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 99
    wr = len(wins) / n * 100 if n > 0 else 0

    # Sharpe
    trade_days = [p for p in daily_pnls if p != 0]
    sharpe = 0
    if trade_days and np.std(trade_days) > 0:
        sharpe = (np.mean(trade_days) / np.std(trade_days)) * np.sqrt(252)

    # Dip depth breakdown
    dip_stats = {}
    for bucket, min_d, max_d in [("1-3%", 1, 3), ("3-5%", 3, 5), ("5-8%", 5, 8), ("8%+", 8, 100)]:
        bt = [t for t in all_trades if min_d <= t["lowest_below_open"] < max_d]
        if bt:
            dip_stats[bucket] = {
                "n": len(bt),
                "pnl": sum(t["pnl"] for t in bt),
                "wr": sum(1 for t in bt if t["pnl"] > 0) / len(bt) * 100,
            }

    # Gap size breakdown
    gap_stats = {}
    for bucket, min_g, max_g in [("10-20%", 10, 20), ("20-40%", 20, 40), ("40-60%", 40, 60), ("60%+", 60, 999)]:
        bt = [t for t in all_trades if min_g <= t["gap_pct"] < max_g]
        if bt:
            gap_stats[bucket] = {
                "n": len(bt),
                "pnl": sum(t["pnl"] for t in bt),
                "wr": sum(1 for t in bt if t["pnl"] > 0) / len(bt) * 100,
            }

    return {
        "n": n, "pf": pf, "wr": wr,
        "total_pnl": total_pnl,
        "equity": equity,
        "sharpe": sharpe,
        "dip_stats": dip_stats,
        "gap_stats": gap_stats,
    }


def objective(trial, daily_picks, all_dates):
    """Objective: suggest params -> run backtest -> score."""
    params = suggest_all_params(trial)
    set_strategy_params(params)

    result = run_backtest(daily_picks, all_dates)

    n = result["n"]
    if n < 10:
        return -9999

    pf = result["pf"]
    if pf < 0.5:
        return -9999

    total_pnl = result["total_pnl"]
    wr = result["wr"]
    sharpe = result["sharpe"]

    # Store user attributes
    trial.set_user_attr("n", n)
    trial.set_user_attr("pf", round(pf, 3))
    trial.set_user_attr("wr", round(wr, 1))
    trial.set_user_attr("total_pnl", round(total_pnl, 2))
    trial.set_user_attr("equity", round(result["equity"], 2))
    trial.set_user_attr("sharpe", round(sharpe, 2))

    for bucket, stats in result.get("dip_stats", {}).items():
        trial.set_user_attr(f"dip_{bucket}_n", stats["n"])
        trial.set_user_attr(f"dip_{bucket}_pnl", round(stats["pnl"], 2))
        trial.set_user_attr(f"dip_{bucket}_wr", round(stats["wr"], 1))

    for bucket, stats in result.get("gap_stats", {}).items():
        trial.set_user_attr(f"gap_{bucket}_n", stats["n"])
        trial.set_user_attr(f"gap_{bucket}_pnl", round(stats["pnl"], 2))
        trial.set_user_attr(f"gap_{bucket}_wr", round(stats["wr"], 1))

    # Score: total_pnl * min(pf, 3) * sharpe_boost
    score = total_pnl * min(pf, 3.0)
    if sharpe > 0:
        score *= min(sharpe / 2, 2.0)

    return score


def make_callback(start_time):
    best_score = [float("-inf")]
    def callback(study, trial):
        elapsed = time.time() - start_time
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if trial.value is not None and trial.value > best_score[0]:
            best_score[0] = trial.value
            pnl = trial.user_attrs.get("total_pnl", 0)
            pf = trial.user_attrs.get("pf", 0)
            wr = trial.user_attrs.get("wr", 0)
            n_trades = trial.user_attrs.get("n", 0)
            equity = trial.user_attrs.get("equity", 0)
            sharpe = trial.user_attrs.get("sharpe", 0)
            print(f"\n  *** NEW BEST (trial {trial.number}) ***")
            print(f"      Score: {trial.value:,.0f} | PnL: ${pnl:,.0f} | PF: {pf:.2f} | "
                  f"WR: {wr:.1f}% | Sharpe: {sharpe:.2f} | Trades: {n_trades}")
            print(f"      Equity: ${equity:,.0f}")
            # Dip breakdown
            for bk in ["1-3%", "3-5%", "5-8%", "8%+"]:
                bn = trial.user_attrs.get(f"dip_{bk}_n", 0)
                if bn > 0:
                    bp = trial.user_attrs.get(f"dip_{bk}_pnl", 0)
                    bw = trial.user_attrs.get(f"dip_{bk}_wr", 0)
                    print(f"      Dip {bk}: {bn} trades, {bw:.0f}% WR, ${bp:+,.0f}")
            # Gap breakdown
            for bk in ["10-20%", "20-40%", "40-60%", "60%+"]:
                bn = trial.user_attrs.get(f"gap_{bk}_n", 0)
                if bn > 0:
                    bp = trial.user_attrs.get(f"gap_{bk}_pnl", 0)
                    bw = trial.user_attrs.get(f"gap_{bk}_wr", 0)
                    print(f"      Gap {bk}: {bn} trades, {bw:.0f}% WR, ${bp:+,.0f}")
            # Key params
            p = trial.params
            print(f"      gap>={p.get('r2g_min_gap_pct',0)}% "
                  f"dip={p.get('r2g_min_dip_pct',0):.1f}-{p.get('r2g_max_dip_pct',0):.1f}% "
                  f"vol>={p.get('r2g_vol_surge_mult',0):.1f}x "
                  f"c{p.get('r2g_earliest_candle',0)}-{p.get('r2g_latest_candle',0)}")
            print(f"      target=+{p.get('r2g_target_pct',0):.0f}% "
                  f"stop=-{p.get('r2g_stop_pct',0):.0f}% "
                  f"trail={p.get('r2g_trail_pct',0):.1f}%@+{p.get('r2g_trail_activate_pct',0):.0f}% "
                  f"time={p.get('r2g_time_limit_min',0)}m")
            print()

        if n_complete % 10 == 0:
            rate = n_complete / elapsed if elapsed > 0 else 0
            print(f"  Trial {n_complete} | {elapsed/60:.1f}m elapsed | {rate:.2f} trials/s | "
                  f"Best: {best_score[0]:,.0f}")
    return callback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()
    n_trials = args.trials

    print("=" * 70)
    print("Optuna Optimizer: Strategy R2G (Red-to-Green Move)")
    print(f"  Parameters: ~17 (filters, entry conditions, exits)")
    print(f"  Trials: {n_trials}")
    print(f"  Objective: total_pnl * min(pf, 3) * sharpe_boost")
    print(f"  Data: {DATA_DIRS}")
    print("=" * 70)

    print("\nLoading data...")
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")

    db_path = "optuna_r2g.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="r2g_2024_2026",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=25),
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"\n  Resuming: {n_existing} existing trials found")
        best = study.best_trial
        print(f"  Current best: score={best.value:,.0f}, "
              f"PnL=${best.user_attrs.get('total_pnl', 0):,.0f}, "
              f"PF={best.user_attrs.get('pf', 0):.2f}")

    print(f"\n  Starting optimization ({n_trials} trials)...\n")
    start_time = time.time()

    study.optimize(
        lambda trial: objective(trial, daily_picks, all_dates),
        n_trials=n_trials,
        callbacks=[make_callback(start_time)],
    )

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Optimization complete: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"{'='*70}")

    best = study.best_trial
    bp = best.params
    ua = best.user_attrs

    print(f"\n  Best trial: #{best.number}")
    print(f"  Score:      {best.value:,.0f}")
    print(f"  Equity:     ${ua.get('equity', 0):,.0f}")
    print(f"  Total PnL:  ${ua.get('total_pnl', 0):,.0f}")
    print(f"  PF:         {ua.get('pf', 0):.3f}")
    print(f"  WR:         {ua.get('wr', 0):.1f}%")
    print(f"  Sharpe:     {ua.get('sharpe', 0):.2f}")
    print(f"  Trades:     {ua.get('n', 0)}")

    print(f"\n  --- Dip Depth Breakdown ---")
    for bk in ["1-3%", "3-5%", "5-8%", "8%+"]:
        bn = ua.get(f"dip_{bk}_n", 0)
        if bn > 0:
            bp_pnl = ua.get(f"dip_{bk}_pnl", 0)
            bw = ua.get(f"dip_{bk}_wr", 0)
            print(f"    {bk}: {bn:>4} trades | WR {bw:>5.1f}% | PnL ${bp_pnl:>+12,.0f}")

    print(f"\n  --- Gap Size Breakdown ---")
    for bk in ["10-20%", "20-40%", "40-60%", "60%+"]:
        bn = ua.get(f"gap_{bk}_n", 0)
        if bn > 0:
            bp_pnl = ua.get(f"gap_{bk}_pnl", 0)
            bw = ua.get(f"gap_{bk}_wr", 0)
            print(f"    {bk}: {bn:>4} trades | WR {bw:>5.1f}% | PnL ${bp_pnl:>+12,.0f}")

    print(f"\n  --- Best Parameters ---")
    for k in sorted(bp):
        print(f"    {k}: {bp[k]}")

    print(f"\n  --- Copy-paste defaults ---")
    print(f"  R2G_MIN_GAP_PCT = {bp.get('r2g_min_gap_pct', 10.0)}")
    print(f"  R2G_MIN_DIP_PCT = {bp.get('r2g_min_dip_pct', 1.0)}")
    print(f"  R2G_MAX_DIP_PCT = {bp.get('r2g_max_dip_pct', 15.0)}")
    print(f"  R2G_EARLIEST_CANDLE = {bp.get('r2g_earliest_candle', 3)}")
    print(f"  R2G_LATEST_CANDLE = {bp.get('r2g_latest_candle', 60)}")
    print(f"  R2G_VOL_SURGE_MULT = {bp.get('r2g_vol_surge_mult', 1.5)}")
    print(f"  R2G_MIN_RECLAIM_BODY_PCT = {bp.get('r2g_min_reclaim_body_pct', 0.5)}")
    print(f"  R2G_REQUIRE_ABOVE_VWAP = {bp.get('r2g_require_above_vwap', False)}")
    print(f"  R2G_REQUIRE_CLOSE_ABOVE_OPEN = {bp.get('r2g_require_close_above_open', True)}")
    print(f"  R2G_TARGET_PCT = {bp.get('r2g_target_pct', 8.0)}")
    print(f"  R2G_STOP_PCT = {bp.get('r2g_stop_pct', 5.0)}")
    print(f"  R2G_TRAIL_PCT = {bp.get('r2g_trail_pct', 3.0)}")
    print(f"  R2G_TRAIL_ACTIVATE_PCT = {bp.get('r2g_trail_activate_pct', 4.0)}")
    print(f"  R2G_PARTIAL_SELL_PCT = {bp.get('r2g_partial_sell_pct', 0.0)}")
    print(f"  R2G_PARTIAL_TARGET_PCT = {bp.get('r2g_partial_target_pct', 5.0)}")
    print(f"  R2G_TIME_LIMIT_MINUTES = {bp.get('r2g_time_limit_min', 60)}")

    print(f"\n  DB saved to: {db_path}")
    print(f"  Study name: r2g_2024_2026")


if __name__ == "__main__":
    main()
