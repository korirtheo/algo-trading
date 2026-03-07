"""
Combined Optuna Optimizer v7: Single Pool + Strategy Selection
===============================================================
Optimizes H, G, A, F, D, V, P, L with single $25K cash pool.
Optuna decides which strategies to enable (1-8) and their priority.
Multiple strategies can enter the same ticker; 5% vol cap limits exposure.
L has dedicated states (signal independence from H/G/A/F on same tickers).
M, R, W are disabled.

Usage:
  python optimize_combined.py                   # 500 trials (default)
  python optimize_combined.py --trials 5        # quick smoke test
  python optimize_combined.py --trials 1000     # extended search
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

import test_green_candle_combined as tgc
from test_full import load_all_picks, SLIPPAGE_PCT, STARTING_CASH, MARGIN_THRESHOLD

DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2024-01-01", "2026-02-28")

STRAT_KEYS = ["H", "G", "A", "F", "D", "V", "P", "L"]


# ---------------------------------------------------------------------------
# Set strategy params on tgc module globals
# ---------------------------------------------------------------------------
def set_strategy_params(params):
    """Set all strategy globals on the tgc module."""
    # Reset H/G/A/F min_gap to defaults (enable/disable may have set to 9999)
    tgc.H_MIN_GAP_PCT = 35.0
    tgc.G_MIN_GAP_PCT = 30.0
    tgc.A_MIN_GAP_PCT = 15.0
    tgc.F_MIN_GAP_PCT = 10.0

    # --- H (High Conviction) ---
    tgc.H_TARGET_PCT = params["h_target_pct"]
    tgc.H_TIME_LIMIT_MINUTES = params["h_time_limit_min"]
    tgc.H_STOP_PCT = params["h_stop_pct"]
    tgc.H_TRAIL_PCT = params["h_trail_pct"]
    tgc.H_TRAIL_ACTIVATE_PCT = params["h_trail_activate_pct"]

    # --- G (Big Gap Runner) ---
    tgc.G_TARGET_PCT = params["g_target_pct"]
    tgc.G_TIME_LIMIT_MINUTES = params["g_time_limit_min"]
    tgc.G_STOP_PCT = params["g_stop_pct"]
    tgc.G_TRAIL_PCT = params["g_trail_pct"]
    tgc.G_TRAIL_ACTIVATE_PCT = params["g_trail_activate_pct"]

    # --- A (Quick Scalp) ---
    tgc.A_TARGET_PCT = params["a_target_pct"]
    tgc.A_TIME_LIMIT_MINUTES = params["a_time_limit_min"]
    tgc.A_STOP_PCT = params["a_stop_pct"]
    tgc.A_TRAIL_PCT = params["a_trail_pct"]
    tgc.A_TRAIL_ACTIVATE_PCT = params["a_trail_activate_pct"]

    # --- F (Catch-All) ---
    tgc.F_TARGET_PCT = params["f_target_pct"]
    tgc.F_TIME_LIMIT_MINUTES = params["f_time_limit_min"]
    tgc.F_STOP_PCT = params["f_stop_pct"]
    tgc.F_TRAIL_PCT = params["f_trail_pct"]
    tgc.F_TRAIL_ACTIVATE_PCT = params["f_trail_activate_pct"]

    # --- D (Opening Dip Buy) ---
    tgc.D_MIN_GAP_PCT = float(params["d_min_gap"])
    tgc.D_MIN_SPIKE_PCT = float(params["d_min_spike_pct"])
    tgc.D_SPIKE_WINDOW = params["d_spike_window"]
    tgc.D_DIP_PCT = float(params["d_dip_pct"])
    tgc.D_ENTRY_MODE = params["d_entry_mode"]
    tgc.D_MAX_ENTRY_CANDLE = params["d_max_entry_candle"]
    tgc.D_TARGET1_PCT = params["d_target1_pct"]
    tgc.D_TARGET2_PCT = params["d_target2_pct"]
    tgc.D_STOP_PCT = params["d_stop_pct"]
    tgc.D_PARTIAL_SELL_PCT = params["d_partial_sell_pct"]
    tgc.D_TRAIL_PCT = params["d_trail_pct"]
    tgc.D_TRAIL_ACTIVATE_PCT = params["d_trail_activate_pct"]
    tgc.D_TIME_LIMIT_MINUTES = params["d_time_limit_min"]

    # --- V (VWAP Reclaim) ---
    tgc.V_MIN_GAP_PCT = float(params["v_min_gap"])
    tgc.V_MIN_BELOW_CANDLES = params["v_min_below_candles"]
    tgc.V_MIN_BELOW_PCT = float(params["v_min_below_pct"])
    tgc.V_VOL_SPIKE_RATIO = params["v_vol_spike_ratio"]
    tgc.V_MAX_ENTRY_CANDLE = params["v_max_entry_candle"]
    tgc.V_TARGET1_PCT = params["v_target1_pct"]
    tgc.V_TARGET2_PCT = params["v_target2_pct"]
    tgc.V_STOP_PCT = params["v_stop_pct"]
    tgc.V_PARTIAL_SELL_PCT = params["v_partial_sell_pct"]
    tgc.V_TRAIL_PCT = params["v_trail_pct"]
    tgc.V_TRAIL_ACTIVATE_PCT = params["v_trail_activate_pct"]
    tgc.V_TIME_LIMIT_MINUTES = params["v_time_limit_min"]

    # --- P (PM High Breakout) ---
    tgc.P_MIN_GAP_PCT = float(params["p_min_gap"])
    tgc.P_CONFIRM_ABOVE = params["p_confirm_above"]
    tgc.P_CONFIRM_WINDOW = params["p_confirm_window"]
    tgc.P_PULLBACK_PCT = params["p_pullback_pct"]
    tgc.P_PULLBACK_TIMEOUT = params["p_pullback_timeout"]
    tgc.P_MAX_ENTRY_CANDLE = params["p_max_entry_candle"]
    tgc.P_TARGET1_PCT = params["p_target1_pct"]
    tgc.P_TARGET2_PCT = params["p_target2_pct"]
    tgc.P_STOP_PCT = params["p_stop_pct"]
    tgc.P_PARTIAL_SELL_PCT = params["p_partial_sell_pct"]
    tgc.P_TRAIL_PCT = params["p_trail_pct"]
    tgc.P_TRAIL_ACTIVATE_PCT = params["p_trail_activate_pct"]
    tgc.P_TIME_LIMIT_MINUTES = params["p_time_limit_min"]

    # --- L (Low Float Squeeze) ---
    tgc.L_MIN_GAP_PCT = float(params["l_min_gap"])
    tgc.L_MAX_FLOAT = params["l_max_float"]
    tgc.L_EARLIEST_CANDLE = params["l_earliest_candle"]
    tgc.L_LATEST_CANDLE = params["l_latest_candle"]
    tgc.L_VOL_SURGE_MULT = params["l_vol_surge_mult"]
    tgc.L_MIN_PRICE_ACCEL_PCT = params["l_min_price_accel_pct"]
    tgc.L_TIER1_FLOAT = params["l_tier1_float"]
    tgc.L_TIER2_FLOAT = params["l_tier2_float"]
    tgc.L_TIER1_TARGET1_PCT = params["l_tier1_target1_pct"]
    tgc.L_TIER1_TARGET2_PCT = params["l_tier1_target2_pct"]
    tgc.L_TIER2_TARGET1_PCT = params["l_tier2_target1_pct"]
    tgc.L_TIER2_TARGET2_PCT = params["l_tier2_target2_pct"]
    tgc.L_TIER3_TARGET1_PCT = params["l_tier3_target1_pct"]
    tgc.L_TIER3_TARGET2_PCT = params["l_tier3_target2_pct"]
    tgc.L_STOP_PCT = params["l_stop_pct"]
    tgc.L_PARTIAL_SELL_PCT = params["l_partial_sell_pct"]
    tgc.L_TRAIL_PCT = params["l_trail_pct"]
    tgc.L_TRAIL_ACTIVATE_PCT = params["l_trail_activate_pct"]
    tgc.L_TIME_LIMIT_MINUTES = params["l_time_limit_min"]

    # --- M/R/W disabled ---
    tgc.M_MIN_GAP_PCT = 9999.0
    tgc.R_DAY1_MIN_GAP = 9999.0
    tgc.W_MIN_GAP_PCT = 9999.0

    # --- Strategy enable/disable (Optuna decides which strategies to include) ---
    # Disabled strategies get min_gap set to 9999 (effectively removes them)
    if not params.get("enable_h", True):
        tgc.H_MIN_GAP_PCT = 9999.0
    if not params.get("enable_g", True):
        tgc.G_MIN_GAP_PCT = 9999.0
    if not params.get("enable_a", True):
        tgc.A_MIN_GAP_PCT = 9999.0
    if not params.get("enable_f", True):
        tgc.F_MIN_GAP_PCT = 9999.0
    if not params.get("enable_d", True):
        tgc.D_MIN_GAP_PCT = 9999.0
    if not params.get("enable_v", True):
        tgc.V_MIN_GAP_PCT = 9999.0
    if not params.get("enable_p", True):
        tgc.P_MIN_GAP_PCT = 9999.0
    if not params.get("enable_l", True):
        tgc.L_MIN_GAP_PCT = 9999.0

    # --- Strategy priority (Optuna decides entry order) ---
    tgc.STRAT_PRIORITY = {
        "H": params["priority_h"],
        "G": params["priority_g"],
        "A": params["priority_a"],
        "F": params["priority_f"],
        "D": params["priority_d"],
        "V": params["priority_v"],
        "P": params["priority_p"],
        "L": params["priority_l"],
    }


# ---------------------------------------------------------------------------
# Suggest all params from an Optuna trial
# ---------------------------------------------------------------------------
def suggest_all_params(trial):
    """Suggest params for all 7 active strategies + priority ordering."""
    params = {}

    # === H: High Conviction (5 params) ===
    params["h_target_pct"] = trial.suggest_float("h_target_pct", 5.0, 25.0, step=1.0)
    params["h_time_limit_min"] = trial.suggest_int("h_time_limit_min", 3, 30, step=3)
    params["h_stop_pct"] = trial.suggest_float("h_stop_pct", 0.0, 12.0, step=2.0)
    params["h_trail_pct"] = trial.suggest_float("h_trail_pct", 0.0, 5.0, step=1.0)
    params["h_trail_activate_pct"] = trial.suggest_float("h_trail_activate_pct", 0.0, 8.0, step=2.0)

    # === G: Big Gap Runner (5 params) ===
    params["g_target_pct"] = trial.suggest_float("g_target_pct", 4.0, 20.0, step=1.0)
    params["g_time_limit_min"] = trial.suggest_int("g_time_limit_min", 3, 30, step=3)
    params["g_stop_pct"] = trial.suggest_float("g_stop_pct", 0.0, 12.0, step=2.0)
    params["g_trail_pct"] = trial.suggest_float("g_trail_pct", 0.0, 5.0, step=1.0)
    params["g_trail_activate_pct"] = trial.suggest_float("g_trail_activate_pct", 0.0, 8.0, step=2.0)

    # === A: Quick Scalp (5 params) ===
    params["a_target_pct"] = trial.suggest_float("a_target_pct", 2.0, 15.0, step=1.0)
    params["a_time_limit_min"] = trial.suggest_int("a_time_limit_min", 3, 30, step=3)
    params["a_stop_pct"] = trial.suggest_float("a_stop_pct", 0.0, 12.0, step=2.0)
    params["a_trail_pct"] = trial.suggest_float("a_trail_pct", 0.0, 5.0, step=1.0)
    params["a_trail_activate_pct"] = trial.suggest_float("a_trail_activate_pct", 0.0, 8.0, step=2.0)

    # === F: Catch-All (5 params) ===
    params["f_target_pct"] = trial.suggest_float("f_target_pct", 2.0, 15.0, step=1.0)
    params["f_time_limit_min"] = trial.suggest_int("f_time_limit_min", 3, 30, step=3)
    params["f_stop_pct"] = trial.suggest_float("f_stop_pct", 0.0, 12.0, step=2.0)
    params["f_trail_pct"] = trial.suggest_float("f_trail_pct", 0.0, 5.0, step=1.0)
    params["f_trail_activate_pct"] = trial.suggest_float("f_trail_activate_pct", 0.0, 8.0, step=2.0)

    # === D: Opening Dip Buy (13 params) ===
    params["d_min_gap"] = trial.suggest_int("d_min_gap", 8, 30, step=2)
    params["d_min_spike_pct"] = trial.suggest_int("d_min_spike_pct", 3, 15)
    params["d_spike_window"] = trial.suggest_int("d_spike_window", 5, 20, step=5)
    params["d_dip_pct"] = trial.suggest_int("d_dip_pct", 5, 15)
    params["d_entry_mode"] = trial.suggest_categorical("d_entry_mode", ["vwap", "5candle"])
    params["d_max_entry_candle"] = trial.suggest_int("d_max_entry_candle", 15, 60, step=5)
    params["d_target1_pct"] = trial.suggest_float("d_target1_pct", 3.0, 15.0, step=1.0)
    d_t1 = params["d_target1_pct"]
    params["d_stop_pct"] = trial.suggest_float("d_stop_pct", 3.0, 12.0, step=1.0)
    params["d_partial_sell_pct"] = trial.suggest_float("d_partial_sell_pct", 0.0, 75.0, step=25.0)
    if params["d_partial_sell_pct"] > 0:
        params["d_target2_pct"] = trial.suggest_float("d_target2_pct", d_t1 + 2, 25.0, step=2.0)
    else:
        params["d_target2_pct"] = d_t1
    params["d_trail_pct"] = trial.suggest_float("d_trail_pct", 1.0, 6.0, step=1.0)
    params["d_trail_activate_pct"] = trial.suggest_float("d_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["d_time_limit_min"] = trial.suggest_int("d_time_limit_min", 30, 120, step=10)

    # === V: VWAP Reclaim (12 params) ===
    params["v_min_gap"] = trial.suggest_int("v_min_gap", 8, 30, step=2)
    params["v_min_below_candles"] = trial.suggest_int("v_min_below_candles", 2, 15)
    params["v_min_below_pct"] = trial.suggest_int("v_min_below_pct", 0, 5)
    params["v_vol_spike_ratio"] = trial.suggest_float("v_vol_spike_ratio", 1.0, 4.0, step=0.5)
    params["v_max_entry_candle"] = trial.suggest_int("v_max_entry_candle", 30, 120, step=10)
    params["v_target1_pct"] = trial.suggest_float("v_target1_pct", 3.0, 15.0, step=1.0)
    v_t1 = params["v_target1_pct"]
    params["v_stop_pct"] = trial.suggest_float("v_stop_pct", 3.0, 12.0, step=1.0)
    params["v_partial_sell_pct"] = trial.suggest_float("v_partial_sell_pct", 0.0, 75.0, step=25.0)
    if params["v_partial_sell_pct"] > 0:
        params["v_target2_pct"] = trial.suggest_float("v_target2_pct", v_t1 + 2, 25.0, step=2.0)
    else:
        params["v_target2_pct"] = v_t1
    params["v_trail_pct"] = trial.suggest_float("v_trail_pct", 1.0, 6.0, step=1.0)
    params["v_trail_activate_pct"] = trial.suggest_float("v_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["v_time_limit_min"] = trial.suggest_int("v_time_limit_min", 30, 120, step=10)

    # === P: PM High Breakout (13 params) ===
    params["p_min_gap"] = trial.suggest_int("p_min_gap", 8, 25, step=2)
    params["p_confirm_above"] = trial.suggest_int("p_confirm_above", 2, 6)
    params["p_confirm_window"] = trial.suggest_int("p_confirm_window", 3, 8)
    params["p_pullback_pct"] = trial.suggest_float("p_pullback_pct", 3.0, 12.0, step=1.0)
    params["p_pullback_timeout"] = trial.suggest_int("p_pullback_timeout", 10, 40, step=5)
    params["p_max_entry_candle"] = trial.suggest_int("p_max_entry_candle", 45, 120, step=15)
    params["p_target1_pct"] = trial.suggest_float("p_target1_pct", 3.0, 20.0, step=1.0)
    p_t1 = params["p_target1_pct"]
    params["p_stop_pct"] = trial.suggest_float("p_stop_pct", 3.0, 15.0, step=1.0)
    params["p_partial_sell_pct"] = trial.suggest_float("p_partial_sell_pct", 0.0, 75.0, step=25.0)
    if params["p_partial_sell_pct"] > 0:
        params["p_target2_pct"] = trial.suggest_float("p_target2_pct", p_t1 + 2, 30.0, step=2.0)
    else:
        params["p_target2_pct"] = p_t1
    params["p_trail_pct"] = trial.suggest_float("p_trail_pct", 1.0, 8.0, step=1.0)
    params["p_trail_activate_pct"] = trial.suggest_float("p_trail_activate_pct", 1.0, 8.0, step=1.0)
    params["p_time_limit_min"] = trial.suggest_int("p_time_limit_min", 30, 180, step=10)

    # Guard: p_confirm_above must be <= p_confirm_window
    if params["p_confirm_above"] > params["p_confirm_window"]:
        raise optuna.TrialPruned()

    # === L: Low Float Squeeze (19 params) ===
    params["l_min_gap"] = trial.suggest_int("l_min_gap", 15, 50, step=5)
    params["l_max_float"] = trial.suggest_int("l_max_float", 5_000_000, 20_000_000, step=5_000_000)
    params["l_earliest_candle"] = trial.suggest_int("l_earliest_candle", 3, 15, step=3)
    params["l_latest_candle"] = trial.suggest_int("l_latest_candle", 60, 150, step=15)
    params["l_vol_surge_mult"] = trial.suggest_float("l_vol_surge_mult", 1.0, 3.0, step=0.5)
    params["l_min_price_accel_pct"] = trial.suggest_float("l_min_price_accel_pct", 0.5, 3.0, step=0.5)
    # Float tier boundaries
    params["l_tier1_float"] = trial.suggest_int("l_tier1_float", 500_000, 2_000_000, step=500_000)
    params["l_tier2_float"] = trial.suggest_int("l_tier2_float", 3_000_000, 7_000_000, step=1_000_000)
    # Tier 1 targets (ultra-low float: biggest movers)
    params["l_tier1_target1_pct"] = trial.suggest_float("l_tier1_target1_pct", 15.0, 50.0, step=5.0)
    params["l_tier1_target2_pct"] = trial.suggest_float("l_tier1_target2_pct", 30.0, 60.0, step=5.0)
    # Tier 2 targets (low float)
    params["l_tier2_target1_pct"] = trial.suggest_float("l_tier2_target1_pct", 8.0, 30.0, step=2.0)
    params["l_tier2_target2_pct"] = trial.suggest_float("l_tier2_target2_pct", 20.0, 50.0, step=5.0)
    # Tier 3 targets (mid float)
    params["l_tier3_target1_pct"] = trial.suggest_float("l_tier3_target1_pct", 5.0, 20.0, step=1.0)
    params["l_tier3_target2_pct"] = trial.suggest_float("l_tier3_target2_pct", 15.0, 40.0, step=5.0)
    # Exit params
    params["l_stop_pct"] = trial.suggest_float("l_stop_pct", 5.0, 20.0, step=1.0)
    params["l_partial_sell_pct"] = trial.suggest_float("l_partial_sell_pct", 0.0, 50.0, step=25.0)
    params["l_trail_pct"] = trial.suggest_float("l_trail_pct", 1.0, 6.0, step=1.0)
    params["l_trail_activate_pct"] = trial.suggest_float("l_trail_activate_pct", 1.0, 8.0, step=1.0)
    params["l_time_limit_min"] = trial.suggest_int("l_time_limit_min", 30, 120, step=10)

    # === Strategy Enable/Disable (8 params — Optuna decides which to include) ===
    for s in ["h", "g", "a", "f", "d", "v", "p", "l"]:
        params[f"enable_{s}"] = trial.suggest_categorical(f"enable_{s}", [True, False])

    # At least 1 strategy must be enabled
    enabled = [params[f"enable_{s}"] for s in ["h", "g", "a", "f", "d", "v", "p", "l"]]
    if not any(enabled):
        raise optuna.TrialPruned()

    # === Strategy Priority (8 params — Optuna decides entry order) ===
    for s in ["h", "g", "a", "f", "d", "v", "p", "l"]:
        params[f"priority_{s}"] = trial.suggest_int(f"priority_{s}", 0, 7)

    return params


# ---------------------------------------------------------------------------
# Run full combined backtest with current tgc globals
# ---------------------------------------------------------------------------
def run_combined_backtest(daily_picks, all_dates):
    """Run the full backtest with single cash pool and return per-strategy stats."""
    cash = float(STARTING_CASH)
    unsettled = 0.0
    all_trades = []

    for d in all_dates:
        cash += unsettled
        unsettled = 0.0

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        states, cash, unsettled, _ = tgc.simulate_day_combined(
            picks, cash, cash_account
        )

        for st in states:
            if st["exit_reason"] is not None:
                all_trades.append({
                    "strategy": st.get("strategy", "?"),
                    "pnl": st["pnl"],
                    "position_cost": st["position_cost"],
                })

    equity = cash + unsettled

    n = len(all_trades)
    if n == 0:
        return {"n": 0, "pf": 0, "total_pnl": -9999, "equity": equity, "strats": {}}

    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in all_trades if t["pnl"] <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 99
    wr = len(wins) / n * 100 if n > 0 else 0

    strats = {}
    for t in all_trades:
        s = t["strategy"]
        if s not in strats:
            strats[s] = {"n": 0, "wins": 0, "pnl": 0.0}
        strats[s]["n"] += 1
        strats[s]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            strats[s]["wins"] += 1

    return {
        "n": n, "pf": pf, "wr": wr,
        "total_pnl": total_pnl,
        "equity": equity,
        "strats": strats,
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial, daily_picks, all_dates):
    """Objective function: suggest params -> run full backtest -> score."""
    params = suggest_all_params(trial)
    set_strategy_params(params)

    result = run_combined_backtest(daily_picks, all_dates)

    n = result["n"]
    if n < 30:
        return -9999

    pf = result["pf"]
    if pf < 0.5:
        return -9999

    total_pnl = result["total_pnl"]

    trial.set_user_attr("n", n)
    trial.set_user_attr("pf", round(pf, 3))
    trial.set_user_attr("wr", round(result["wr"], 1))
    trial.set_user_attr("total_pnl", round(total_pnl, 2))
    trial.set_user_attr("equity", round(result["equity"], 2))

    # Priority order for display (only enabled strategies)
    enabled_strats = [s for s in STRAT_KEYS if params.get(f"enable_{s.lower()}", True)]
    prio = {s: params[f"priority_{s.lower()}"] for s in enabled_strats}
    prio_str = ">".join(s for s, _ in sorted(prio.items(), key=lambda x: x[1]))
    trial.set_user_attr("priority", prio_str)
    trial.set_user_attr("enabled", ",".join(enabled_strats))
    trial.set_user_attr("n_strategies", len(enabled_strats))

    for s, v in result["strats"].items():
        trial.set_user_attr(f"{s}_n", v["n"])
        trial.set_user_attr(f"{s}_pnl", round(v["pnl"], 2))
        wr_s = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
        trial.set_user_attr(f"{s}_wr", round(wr_s, 1))

    return total_pnl * min(pf, 3.0)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
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
            prio_str = trial.user_attrs.get("priority", "?")
            print(f"\n  *** NEW BEST (trial {trial.number}) ***")
            print(f"      Score: {trial.value:,.0f} | PnL: ${pnl:,.0f} | PF: {pf:.2f} | "
                  f"WR: {wr:.1f}% | Trades: {n_trades}")
            enabled_str = trial.user_attrs.get("enabled", "?")
            n_strats = trial.user_attrs.get("n_strategies", "?")
            print(f"      Equity: ${equity:,.0f} | Priority: {prio_str} | {n_strats} strategies: {enabled_str}")
            for strat in STRAT_KEYS:
                sn = trial.user_attrs.get(f"{strat}_n", 0)
                if sn > 0:
                    spnl = trial.user_attrs.get(f"{strat}_pnl", 0)
                    swr = trial.user_attrs.get(f"{strat}_wr", 0)
                    print(f"      {strat}: {sn} trades, {swr:.0f}% WR, ${spnl:+,.0f}")
            print()

        if n_complete % 10 == 0:
            rate = n_complete / elapsed if elapsed > 0 else 0
            print(f"  Trial {n_complete} | {elapsed/60:.1f}m elapsed | {rate:.2f} trials/s | "
                  f"Best: {best_score[0]:,.0f}")
    return callback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()
    n_trials = args.trials

    print("=" * 70)
    print("Combined Optuna Optimizer v7: Single Pool + Strategy Selection")
    print(f"  Candidates: H, G, A, F, D, V, P, L")
    print(f"  Optuna decides: which strategies to enable (1-8) + priority order")
    print(f"  Single pool: ${STARTING_CASH:,}")
    print(f"  Trials: {n_trials}")
    print(f"  Objective: total_pnl * min(pf, 3.0)")
    print(f"  Data: {DATA_DIRS}")
    print("=" * 70)

    print("\nLoading data...")
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")

    db_path = "optuna_combined_v7.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="combined_v7_single_pool_2024_2026",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=50),
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
    print(f"  Trades:     {ua.get('n', 0)}")
    print(f"  Priority:   {ua.get('priority', '?')}")
    print(f"  Enabled:    {ua.get('enabled', '?')} ({ua.get('n_strategies', '?')} strategies)")

    print(f"\n  --- Per-Strategy Breakdown ---")
    for strat in STRAT_KEYS:
        sn = ua.get(f"{strat}_n", 0)
        spnl = ua.get(f"{strat}_pnl", 0)
        swr = ua.get(f"{strat}_wr", 0)
        verdict = ""
        if sn == 0:
            verdict = "  (NO TRADES)"
        elif spnl < 0:
            verdict = "  ** NEGATIVE **"
        elif sn < 10:
            verdict = "  (negligible)"
        print(f"    {strat}: {sn:>4} trades | WR {swr:>5.1f}% | PnL ${spnl:>+12,.0f}{verdict}")

    print(f"\n  --- Best Parameters ---")
    # Priority
    prio = {s: bp.get(f"priority_{s.lower()}", 99) for s in STRAT_KEYS}
    prio_order = sorted(prio.items(), key=lambda x: x[1])
    print(f"\n  Priority order: {' > '.join(s for s, _ in prio_order)}")

    for prefix, label in [("h_", "H (High Conv)"), ("g_", "G (Gap Runner)"),
                           ("a_", "A (Quick Scalp)"), ("f_", "F (Catch-All)"),
                           ("d_", "D (Dip Buy)"), ("v_", "V (VWAP)"),
                           ("p_", "P (PM High)"), ("l_", "L (Low Float)")]:
        pkeys = sorted(k for k in bp if k.startswith(prefix) and not k.startswith("priority"))
        if pkeys:
            print(f"\n  Strategy {label}:")
            for k in pkeys:
                print(f"    {k}: {bp[k]}")

    # Strategy selection
    enabled_list = []
    for s in ["h", "g", "a", "f", "d", "v", "p", "l"]:
        if bp.get(f"enable_{s}", True):
            enabled_list.append(s.upper())
    print(f"\n  Enabled strategies: {', '.join(enabled_list)} ({len(enabled_list)} of 8)")

    print(f"\n  DB saved to: {db_path}")
    print(f"  Study name: combined_v7_single_pool_2024_2026")


if __name__ == "__main__":
    main()
