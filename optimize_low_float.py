"""
Optuna Optimizer for Strategy L: Low Float Squeeze
===================================================
Optimizes all L strategy parameters independently.
Each trial runs the full backtest with modified globals.

Usage:
  python optimize_low_float.py                  # 500 trials (default)
  python optimize_low_float.py --trials 5       # quick smoke test
  python optimize_low_float.py --trials 1000    # extended search
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

import test_low_float_squeeze as lfs
from test_full import load_all_picks, SLIPPAGE_PCT, STARTING_CASH, MARGIN_THRESHOLD

DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2024-01-01", "2026-02-28")


def set_strategy_params(params):
    """Set all L globals on the lfs module."""
    lfs.L_MAX_FLOAT = params["l_max_float"]
    lfs.L_MIN_GAP_PCT = float(params["l_min_gap_pct"])
    lfs.L_EARLIEST_CANDLE = params["l_earliest_candle"]
    lfs.L_LATEST_CANDLE = params["l_latest_candle"]
    lfs.L_HOD_BREAK_REQUIRED = params["l_hod_break_required"]
    lfs.L_VOL_SURGE_MULT = params["l_vol_surge_mult"]
    lfs.L_MIN_PRICE_ACCEL_PCT = params["l_min_price_accel_pct"]
    lfs.L_REQUIRE_ABOVE_VWAP = params["l_require_above_vwap"]
    # Float-tiered targets
    lfs.L_TIER1_FLOAT = params["l_tier1_float"]
    lfs.L_TIER2_FLOAT = params["l_tier2_float"]
    lfs.L_TIER1_TARGET1_PCT = params["l_tier1_target1_pct"]
    lfs.L_TIER1_TARGET2_PCT = params["l_tier1_target2_pct"]
    lfs.L_TIER2_TARGET1_PCT = params["l_tier2_target1_pct"]
    lfs.L_TIER2_TARGET2_PCT = params["l_tier2_target2_pct"]
    lfs.L_TIER3_TARGET1_PCT = params["l_tier3_target1_pct"]
    lfs.L_TIER3_TARGET2_PCT = params["l_tier3_target2_pct"]
    lfs.L_STOP_PCT = params["l_stop_pct"]
    lfs.L_PARTIAL_SELL_PCT = params["l_partial_sell_pct"]
    lfs.L_TRAIL_PCT = params["l_trail_pct"]
    lfs.L_TRAIL_ACTIVATE_PCT = params["l_trail_activate_pct"]
    lfs.L_TIME_LIMIT_MINUTES = params["l_time_limit_min"]


def suggest_all_params(trial):
    """Suggest all L strategy parameters."""
    params = {}

    # --- FILTERS ---
    params["l_max_float"] = trial.suggest_categorical(
        "l_max_float", [2_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000])
    params["l_min_gap_pct"] = trial.suggest_int("l_min_gap_pct", 10, 60, step=5)

    # --- ENTRY TIMING ---
    params["l_earliest_candle"] = trial.suggest_int("l_earliest_candle", 1, 10)
    params["l_latest_candle"] = trial.suggest_int("l_latest_candle", 30, 195, step=5)

    # --- ENTRY CONDITIONS ---
    params["l_hod_break_required"] = trial.suggest_categorical("l_hod_break_required", [True, False])
    params["l_vol_surge_mult"] = trial.suggest_float("l_vol_surge_mult", 1.0, 5.0, step=0.5)
    params["l_min_price_accel_pct"] = trial.suggest_float("l_min_price_accel_pct", 0.0, 3.0, step=0.5)
    params["l_require_above_vwap"] = trial.suggest_categorical("l_require_above_vwap", [True, False])

    # --- FLOAT TIER BOUNDARIES ---
    params["l_tier1_float"] = trial.suggest_categorical("l_tier1_float", [1_000_000, 2_000_000, 3_000_000])
    params["l_tier2_float"] = trial.suggest_categorical("l_tier2_float", [5_000_000, 8_000_000])

    # --- TIERED EXITS ---
    # Tier 1: ultra-low float (most aggressive targets)
    params["l_tier1_target1_pct"] = trial.suggest_float("l_tier1_target1_pct", 8.0, 30.0, step=2.0)
    t1_t1 = params["l_tier1_target1_pct"]
    params["l_tier1_target2_pct"] = trial.suggest_float("l_tier1_target2_pct", t1_t1 + 5, 60.0, step=5.0)

    # Tier 2: low float (medium targets)
    params["l_tier2_target1_pct"] = trial.suggest_float("l_tier2_target1_pct", 5.0, 25.0, step=2.0)
    t2_t1 = params["l_tier2_target1_pct"]
    params["l_tier2_target2_pct"] = trial.suggest_float("l_tier2_target2_pct", t2_t1 + 5, 50.0, step=5.0)

    # Tier 3: mid float (conservative targets)
    params["l_tier3_target1_pct"] = trial.suggest_float("l_tier3_target1_pct", 3.0, 20.0, step=2.0)
    t3_t1 = params["l_tier3_target1_pct"]
    params["l_tier3_target2_pct"] = trial.suggest_float("l_tier3_target2_pct", t3_t1 + 3, 40.0, step=5.0)

    # --- SHARED EXITS ---
    params["l_partial_sell_pct"] = trial.suggest_float("l_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["l_stop_pct"] = trial.suggest_float("l_stop_pct", 3.0, 15.0, step=1.0)
    params["l_trail_pct"] = trial.suggest_float("l_trail_pct", 1.0, 8.0, step=0.5)
    params["l_trail_activate_pct"] = trial.suggest_float("l_trail_activate_pct", 2.0, 10.0, step=1.0)
    params["l_time_limit_min"] = trial.suggest_int("l_time_limit_min", 20, 180, step=10)

    return params


def run_backtest(daily_picks, all_dates):
    """Run the full L backtest and return stats."""
    cash = float(STARTING_CASH)
    unsettled = 0.0
    all_trades = []
    daily_pnls = []

    for d in all_dates:
        cash += unsettled
        unsettled = 0.0

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        states, cash, unsettled = lfs.simulate_day_l(picks, cash, cash_account)

        day_pnl = 0.0
        for st in states:
            if st["exit_reason"] is not None:
                all_trades.append({
                    "pnl": st["pnl"],
                    "position_cost": st["position_cost"],
                    "float_shares": st["float_shares"],
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

    # Float breakdown
    float_stats = {}
    for bucket, max_f in [("<2M", 2e6), ("2-5M", 5e6), ("5-10M", 10e6), ("10M+", 1e12)]:
        min_f = {"<2M": 0, "2-5M": 2e6, "5-10M": 5e6, "10M+": 10e6}[bucket]
        bt = [t for t in all_trades if min_f <= t["float_shares"] < max_f]
        if bt:
            float_stats[bucket] = {
                "n": len(bt),
                "pnl": sum(t["pnl"] for t in bt),
                "wr": sum(1 for t in bt if t["pnl"] > 0) / len(bt) * 100,
            }

    return {
        "n": n, "pf": pf, "wr": wr,
        "total_pnl": total_pnl,
        "equity": equity,
        "sharpe": sharpe,
        "float_stats": float_stats,
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

    for bucket, stats in result.get("float_stats", {}).items():
        trial.set_user_attr(f"float_{bucket}_n", stats["n"])
        trial.set_user_attr(f"float_{bucket}_pnl", round(stats["pnl"], 2))
        trial.set_user_attr(f"float_{bucket}_wr", round(stats["wr"], 1))

    # Score: total_pnl * min(pf, 3) * min(sharpe, 3)
    # Reward profitable, consistent strategies
    score = total_pnl * min(pf, 3.0)
    if sharpe > 0:
        score *= min(sharpe / 2, 2.0)  # Boost for good sharpe but cap the multiplier

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
            # Float breakdown
            for bk in ["<2M", "2-5M", "5-10M", "10M+"]:
                bn = trial.user_attrs.get(f"float_{bk}_n", 0)
                if bn > 0:
                    bp = trial.user_attrs.get(f"float_{bk}_pnl", 0)
                    bw = trial.user_attrs.get(f"float_{bk}_wr", 0)
                    print(f"      Float {bk}: {bn} trades, {bw:.0f}% WR, ${bp:+,.0f}")
            # Key params
            p = trial.params
            print(f"      float<{p.get('l_max_float',0)/1e6:.0f}M gap>={p.get('l_min_gap_pct',0)}% "
                  f"vol>={p.get('l_vol_surge_mult',0):.1f}x "
                  f"c{p.get('l_earliest_candle',0)}-{p.get('l_latest_candle',0)}")
            print(f"      T1(<{p.get('l_tier1_float',2e6)/1e6:.0f}M): +{p.get('l_tier1_target1_pct',0):.0f}%/+{p.get('l_tier1_target2_pct',0):.0f}% | "
                  f"T2(<{p.get('l_tier2_float',5e6)/1e6:.0f}M): +{p.get('l_tier2_target1_pct',0):.0f}%/+{p.get('l_tier2_target2_pct',0):.0f}% | "
                  f"T3: +{p.get('l_tier3_target1_pct',0):.0f}%/+{p.get('l_tier3_target2_pct',0):.0f}%")
            print(f"      stop=-{p.get('l_stop_pct',0):.0f}% trail={p.get('l_trail_pct',0):.1f}%"
                  f"@+{p.get('l_trail_activate_pct',0):.0f}%")
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
    print("Optuna Optimizer: Strategy L (Low Float Squeeze) - TIERED TARGETS")
    print(f"  Parameters: ~20 (filters, entry, tiered exits)")
    print(f"  Trials: {n_trials}")
    print(f"  Objective: total_pnl * min(pf, 3) * sharpe_boost")
    print(f"  Float tiers: ultra-low / low / mid -> separate targets")
    print(f"  Data: {DATA_DIRS}")
    print("=" * 70)

    print("\nLoading data...")
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")
    print(f"  Float data: {len(lfs.FLOAT_DATA)} tickers")

    db_path = "optuna_low_float_tiered_v2.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="low_float_tiered_v2_2024_2026",
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

    print(f"\n  --- Float Breakdown ---")
    for bk in ["<2M", "2-5M", "5-10M", "10M+"]:
        bn = ua.get(f"float_{bk}_n", 0)
        if bn > 0:
            bp_pnl = ua.get(f"float_{bk}_pnl", 0)
            bw = ua.get(f"float_{bk}_wr", 0)
            print(f"    {bk}: {bn:>4} trades | WR {bw:>5.1f}% | PnL ${bp_pnl:>+12,.0f}")

    print(f"\n  --- Best Parameters ---")
    for k in sorted(bp):
        v = bp[k]
        if k == "l_max_float":
            print(f"    {k}: {v:,.0f} ({v/1e6:.0f}M)")
        else:
            print(f"    {k}: {v}")

    print(f"\n  --- Copy-paste defaults ---")
    print(f"  L_MAX_FLOAT = {bp.get('l_max_float', 10_000_000)}")
    print(f"  L_MIN_GAP_PCT = {bp.get('l_min_gap_pct', 20.0)}")
    print(f"  L_EARLIEST_CANDLE = {bp.get('l_earliest_candle', 3)}")
    print(f"  L_LATEST_CANDLE = {bp.get('l_latest_candle', 100)}")
    print(f"  L_HOD_BREAK_REQUIRED = {bp.get('l_hod_break_required', True)}")
    print(f"  L_VOL_SURGE_MULT = {bp.get('l_vol_surge_mult', 2.0)}")
    print(f"  L_MIN_PRICE_ACCEL_PCT = {bp.get('l_min_price_accel_pct', 1.0)}")
    print(f"  L_REQUIRE_ABOVE_VWAP = {bp.get('l_require_above_vwap', True)}")
    print(f"  L_TIER1_FLOAT = {bp.get('l_tier1_float', 2_000_000)}")
    print(f"  L_TIER2_FLOAT = {bp.get('l_tier2_float', 5_000_000)}")
    print(f"  L_TIER1_TARGET1_PCT = {bp.get('l_tier1_target1_pct', 20.0)}")
    print(f"  L_TIER1_TARGET2_PCT = {bp.get('l_tier1_target2_pct', 40.0)}")
    print(f"  L_TIER2_TARGET1_PCT = {bp.get('l_tier2_target1_pct', 15.0)}")
    print(f"  L_TIER2_TARGET2_PCT = {bp.get('l_tier2_target2_pct', 30.0)}")
    print(f"  L_TIER3_TARGET1_PCT = {bp.get('l_tier3_target1_pct', 10.0)}")
    print(f"  L_TIER3_TARGET2_PCT = {bp.get('l_tier3_target2_pct', 20.0)}")
    print(f"  L_STOP_PCT = {bp.get('l_stop_pct', 8.0)}")
    print(f"  L_PARTIAL_SELL_PCT = {bp.get('l_partial_sell_pct', 50.0)}")
    print(f"  L_TRAIL_PCT = {bp.get('l_trail_pct', 3.0)}")
    print(f"  L_TRAIL_ACTIVATE_PCT = {bp.get('l_trail_activate_pct', 5.0)}")
    print(f"  L_TIME_LIMIT_MINUTES = {bp.get('l_time_limit_min', 60)}")

    print(f"\n  DB saved to: {db_path}")
    print(f"  Study name: low_float_squeeze_2024_2026")


if __name__ == "__main__":
    main()
