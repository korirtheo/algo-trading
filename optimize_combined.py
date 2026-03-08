"""
Combined Optuna Optimizer v8: 20 Strategies + Single Pool
==========================================================
Optimizes all 20 strategies (H,G,A,F,D,V,P,M,R,W,O,B,K,C,S,E,I,J,N,L)
with single $25K cash pool. Optuna decides which to enable (1-20) and priority.
5% vol cap limits exposure per ticker across all active states.

Usage:
  python optimize_combined.py                   # 2000 trials (default)
  python optimize_combined.py --trials 5        # quick smoke test
  python optimize_combined.py --trials 1000     # medium run
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

import io
if __name__ == "__main__" and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

import test_green_candle_combined as tgc
from test_full import load_all_picks, SLIPPAGE_PCT, STARTING_CASH, MARGIN_THRESHOLD

DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2024-01-01", "2026-02-28")

ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l"]
STRAT_KEYS = [s.upper() for s in ALL_STRATS]


# ---------------------------------------------------------------------------
# Set strategy params on tgc module globals
# ---------------------------------------------------------------------------
def set_strategy_params(params):
    """Set all strategy globals on the tgc module."""
    # Reset ALL min_gap to defaults (enable/disable may have set to 9999)
    tgc.H_MIN_GAP_PCT = 35.0
    tgc.G_MIN_GAP_PCT = 30.0
    tgc.A_MIN_GAP_PCT = 15.0
    tgc.F_MIN_GAP_PCT = 10.0
    tgc.D_MIN_GAP_PCT = 10.0
    tgc.V_MIN_GAP_PCT = 14.0
    tgc.P_MIN_GAP_PCT = 10.0
    tgc.M_MIN_GAP_PCT = 10.0
    tgc.R_DAY1_MIN_GAP = 40.0
    tgc.W_MIN_GAP_PCT = 10.0
    tgc.O_MIN_GAP_PCT = 10.0
    tgc.B_MIN_GAP_PCT = 15.0
    tgc.K_MIN_GAP_PCT = 10.0
    tgc.C_MIN_GAP_PCT = 10.0
    tgc.S_MIN_GAP_PCT = 10.0
    tgc.E_MIN_GAP_PCT = 15.0
    tgc.I_MIN_GAP_PCT = 10.0
    tgc.J_MIN_GAP_PCT = 10.0
    tgc.N_MIN_GAP_PCT = 10.0
    tgc.L_MIN_GAP_PCT = 30.0

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

    # --- M (Midday Range Break) ---
    tgc.M_MIN_GAP_PCT = float(params["m_min_gap"])
    tgc.M_MORNING_SPIKE_PCT = params["m_morning_spike_pct"]
    tgc.M_MORNING_CANDLES = params["m_morning_candles"]
    tgc.M_RANGE_START_CANDLE = params["m_range_start_candle"]
    tgc.M_CONSOLIDATION_LEN = params["m_consolidation_len"]
    tgc.M_MAX_RANGE_PCT = params["m_max_range_pct"]
    tgc.M_VOL_RATIO = params["m_vol_ratio"]
    tgc.M_MAX_ENTRY_CANDLE = params["m_max_entry_candle"]
    tgc.M_TARGET1_PCT = params["m_target1_pct"]
    tgc.M_STOP_PCT = params["m_stop_pct"]
    tgc.M_TIME_LIMIT_MINUTES = params["m_time_limit_min"]
    tgc.M_PARTIAL_SELL_PCT = params["m_partial_sell_pct"]
    tgc.M_TRAIL_PCT = params["m_trail_pct"]
    tgc.M_TRAIL_ACTIVATE_PCT = params["m_trail_activate_pct"]

    # --- R (Multi-Day Runner) ---
    tgc.R_DAY1_MIN_GAP = float(params["r_day1_min_gap"])
    tgc.R_D2_PULLBACK_PCT = params["r_d2_pullback_pct"]
    tgc.R_PULLBACK_WINDOW = params["r_pullback_window"]
    tgc.R_BOUNCE_REF = params["r_bounce_ref"]
    tgc.R_MAX_ENTRY_CANDLE = params["r_max_entry_candle"]
    tgc.R_TARGET1_PCT = params["r_target1_pct"]
    tgc.R_STOP_PCT = params["r_stop_pct"]
    tgc.R_TRAIL_PCT = params["r_trail_pct"]
    tgc.R_TRAIL_ACTIVATE_PCT = params["r_trail_activate_pct"]
    tgc.R_TIME_LIMIT_MINUTES = params["r_time_limit_min"]

    # --- W (Power Hour Breakout) ---
    tgc.W_MIN_GAP_PCT = float(params["w_min_gap"])
    tgc.W_MIN_MORNING_RUN = params["w_min_morning_run"]
    tgc.W_CONSOL_START = params["w_consol_start"]
    tgc.W_MAX_RANGE_PCT = params["w_max_range_pct"]
    tgc.W_MAX_VWAP_DEV_PCT = params["w_max_vwap_dev_pct"]
    tgc.W_EARLIEST_CANDLE = params["w_earliest_candle"]
    tgc.W_LATEST_CANDLE = params["w_latest_candle"]
    tgc.W_VOL_SURGE_MULT = params["w_vol_surge_mult"]
    tgc.W_VOL_VS_MORNING_MULT = params["w_vol_vs_morning_mult"]
    tgc.W_MAX_HOD_BREAKS = params["w_max_hod_breaks"]
    tgc.W_TARGET_PCT = params["w_target_pct"]
    tgc.W_STOP_PCT = params["w_stop_pct"]
    tgc.W_TRAIL_PCT = params["w_trail_pct"]
    tgc.W_TRAIL_ACTIVATE_PCT = params["w_trail_activate_pct"]

    # --- O (Opening Range Breakout) ---
    tgc.O_MIN_GAP_PCT = float(params["o_min_gap"])
    tgc.O_RANGE_CANDLES = params["o_range_candles"]
    tgc.O_BREAKOUT_VOL_MULT = params["o_breakout_vol_mult"]
    tgc.O_MAX_ENTRY_CANDLE = params["o_max_entry_candle"]
    tgc.O_TARGET1_PCT = params["o_target1_pct"]
    tgc.O_TARGET2_PCT = params["o_target2_pct"]
    tgc.O_STOP_PCT = params["o_stop_pct"]
    tgc.O_PARTIAL_SELL_PCT = params["o_partial_sell_pct"]
    tgc.O_TRAIL_PCT = params["o_trail_pct"]
    tgc.O_TRAIL_ACTIVATE_PCT = params["o_trail_activate_pct"]
    tgc.O_TIME_LIMIT_MINUTES = params["o_time_limit_min"]

    # --- B (Red-to-Green R2G) ---
    tgc.B_MIN_GAP_PCT = float(params["b_min_gap"])
    tgc.B_MAX_DIP_PCT = params["b_max_dip_pct"]
    tgc.B_MIN_RECLAIM_VOL_MULT = params["b_min_reclaim_vol_mult"]
    tgc.B_MAX_ENTRY_CANDLE = params["b_max_entry_candle"]
    tgc.B_TARGET1_PCT = params["b_target1_pct"]
    tgc.B_TARGET2_PCT = params["b_target2_pct"]
    tgc.B_STOP_PCT = params["b_stop_pct"]
    tgc.B_PARTIAL_SELL_PCT = params["b_partial_sell_pct"]
    tgc.B_TRAIL_PCT = params["b_trail_pct"]
    tgc.B_TRAIL_ACTIVATE_PCT = params["b_trail_activate_pct"]
    tgc.B_TIME_LIMIT_MINUTES = params["b_time_limit_min"]

    # --- K (First Pullback Buy) ---
    tgc.K_MIN_GAP_PCT = float(params["k_min_gap"])
    tgc.K_MIN_RUN_PCT = params["k_min_run_pct"]
    tgc.K_RUN_WINDOW = params["k_run_window"]
    tgc.K_PULLBACK_PCT = params["k_pullback_pct"]
    tgc.K_PULLBACK_VOL_RATIO = params["k_pullback_vol_ratio"]
    tgc.K_BOUNCE_VOL_MULT = params["k_bounce_vol_mult"]
    tgc.K_MAX_ENTRY_CANDLE = params["k_max_entry_candle"]
    tgc.K_TARGET1_PCT = params["k_target1_pct"]
    tgc.K_TARGET2_PCT = params["k_target2_pct"]
    tgc.K_STOP_PCT = params["k_stop_pct"]
    tgc.K_PARTIAL_SELL_PCT = params["k_partial_sell_pct"]
    tgc.K_TRAIL_PCT = params["k_trail_pct"]
    tgc.K_TRAIL_ACTIVATE_PCT = params["k_trail_activate_pct"]
    tgc.K_TIME_LIMIT_MINUTES = params["k_time_limit_min"]

    # --- C (Micro Flag / Base Pattern) ---
    tgc.C_MIN_GAP_PCT = float(params["c_min_gap"])
    tgc.C_MIN_SPIKE_PCT = params["c_min_spike_pct"]
    tgc.C_MIN_BASE_CANDLES = params["c_min_base_candles"]
    tgc.C_MAX_BASE_CANDLES = params["c_max_base_candles"]
    tgc.C_MAX_BASE_RANGE_PCT = params["c_max_base_range_pct"]
    tgc.C_BREAKOUT_VOL_MULT = params["c_breakout_vol_mult"]
    tgc.C_MAX_ENTRY_CANDLE = params["c_max_entry_candle"]
    tgc.C_TARGET1_PCT = params["c_target1_pct"]
    tgc.C_TARGET2_PCT = params["c_target2_pct"]
    tgc.C_STOP_PCT = params["c_stop_pct"]
    tgc.C_PARTIAL_SELL_PCT = params["c_partial_sell_pct"]
    tgc.C_TRAIL_PCT = params["c_trail_pct"]
    tgc.C_TRAIL_ACTIVATE_PCT = params["c_trail_activate_pct"]
    tgc.C_TIME_LIMIT_MINUTES = params["c_time_limit_min"]

    # --- S (Stuff-and-Break) ---
    tgc.S_MIN_GAP_PCT = float(params["s_min_gap"])
    tgc.S_MIN_HOD_TESTS = params["s_min_hod_tests"]
    tgc.S_HOD_TOLERANCE_PCT = params["s_hod_tolerance_pct"]
    tgc.S_REJECTION_PCT = params["s_rejection_pct"]
    tgc.S_BREAKOUT_VOL_MULT = params["s_breakout_vol_mult"]
    tgc.S_MAX_ENTRY_CANDLE = params["s_max_entry_candle"]
    tgc.S_TARGET1_PCT = params["s_target1_pct"]
    tgc.S_TARGET2_PCT = params["s_target2_pct"]
    tgc.S_STOP_PCT = params["s_stop_pct"]
    tgc.S_PARTIAL_SELL_PCT = params["s_partial_sell_pct"]
    tgc.S_TRAIL_PCT = params["s_trail_pct"]
    tgc.S_TRAIL_ACTIVATE_PCT = params["s_trail_activate_pct"]
    tgc.S_TIME_LIMIT_MINUTES = params["s_time_limit_min"]

    # --- E (Gap-and-Go RelVol) ---
    tgc.E_MIN_GAP_PCT = float(params["e_min_gap"])
    tgc.E_MIN_PM_VOL_MULT = params["e_min_pm_vol_mult"]
    tgc.E_MAX_ENTRY_CANDLE = params["e_max_entry_candle"]
    tgc.E_TARGET1_PCT = params["e_target1_pct"]
    tgc.E_TARGET2_PCT = params["e_target2_pct"]
    tgc.E_STOP_PCT = params["e_stop_pct"]
    tgc.E_PARTIAL_SELL_PCT = params["e_partial_sell_pct"]
    tgc.E_TRAIL_PCT = params["e_trail_pct"]
    tgc.E_TRAIL_ACTIVATE_PCT = params["e_trail_activate_pct"]
    tgc.E_TIME_LIMIT_MINUTES = params["e_time_limit_min"]

    # --- I (P1 Immediate PM High Breakout) ---
    tgc.I_MIN_GAP_PCT = float(params["i_min_gap"])
    tgc.I_MAX_ENTRY_CANDLE = params["i_max_entry_candle"]
    tgc.I_BREAKOUT_VOL_MULT = params["i_breakout_vol_mult"]
    tgc.I_TARGET1_PCT = params["i_target1_pct"]
    tgc.I_TARGET2_PCT = params["i_target2_pct"]
    tgc.I_STOP_PCT = params["i_stop_pct"]
    tgc.I_PARTIAL_SELL_PCT = params["i_partial_sell_pct"]
    tgc.I_TRAIL_PCT = params["i_trail_pct"]
    tgc.I_TRAIL_ACTIVATE_PCT = params["i_trail_activate_pct"]
    tgc.I_TIME_LIMIT_MINUTES = params["i_time_limit_min"]

    # --- J (P3 VWAP + PM High Breakout) ---
    tgc.J_MIN_GAP_PCT = float(params["j_min_gap"])
    tgc.J_MAX_ENTRY_CANDLE = params["j_max_entry_candle"]
    tgc.J_VWAP_PROXIMITY_PCT = params["j_vwap_proximity_pct"]
    tgc.J_TARGET1_PCT = params["j_target1_pct"]
    tgc.J_TARGET2_PCT = params["j_target2_pct"]
    tgc.J_STOP_PCT = params["j_stop_pct"]
    tgc.J_PARTIAL_SELL_PCT = params["j_partial_sell_pct"]
    tgc.J_TRAIL_PCT = params["j_trail_pct"]
    tgc.J_TRAIL_ACTIVATE_PCT = params["j_trail_activate_pct"]
    tgc.J_TIME_LIMIT_MINUTES = params["j_time_limit_min"]

    # --- N (P4 HOD Reclaim) ---
    tgc.N_MIN_GAP_PCT = float(params["n_min_gap"])
    tgc.N_MIN_HOD_AGE = params["n_min_hod_age"]
    tgc.N_PULLBACK_FROM_HOD_PCT = params["n_pullback_from_hod_pct"]
    tgc.N_MAX_ENTRY_CANDLE = params["n_max_entry_candle"]
    tgc.N_TARGET1_PCT = params["n_target1_pct"]
    tgc.N_TARGET2_PCT = params["n_target2_pct"]
    tgc.N_STOP_PCT = params["n_stop_pct"]
    tgc.N_PARTIAL_SELL_PCT = params["n_partial_sell_pct"]
    tgc.N_TRAIL_PCT = params["n_trail_pct"]
    tgc.N_TRAIL_ACTIVATE_PCT = params["n_trail_activate_pct"]
    tgc.N_TIME_LIMIT_MINUTES = params["n_time_limit_min"]

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

    # --- Strategy enable/disable (Optuna decides which to include) ---
    _gap_keys = {
        "h": "H_MIN_GAP_PCT", "g": "G_MIN_GAP_PCT", "a": "A_MIN_GAP_PCT", "f": "F_MIN_GAP_PCT",
        "d": "D_MIN_GAP_PCT", "v": "V_MIN_GAP_PCT", "p": "P_MIN_GAP_PCT",
        "m": "M_MIN_GAP_PCT", "w": "W_MIN_GAP_PCT",
        "o": "O_MIN_GAP_PCT", "b": "B_MIN_GAP_PCT", "k": "K_MIN_GAP_PCT",
        "c": "C_MIN_GAP_PCT", "s": "S_MIN_GAP_PCT", "e": "E_MIN_GAP_PCT",
        "i": "I_MIN_GAP_PCT", "j": "J_MIN_GAP_PCT", "n": "N_MIN_GAP_PCT",
        "l": "L_MIN_GAP_PCT",
    }
    for s in ALL_STRATS:
        if s == "r":
            if not params.get("enable_r", True):
                tgc.R_DAY1_MIN_GAP = 9999.0
        else:
            if not params.get(f"enable_{s}", True):
                setattr(tgc, _gap_keys[s], 9999.0)

    # --- Strategy priority ---
    tgc.STRAT_PRIORITY = {s.upper(): params[f"priority_{s}"] for s in ALL_STRATS}


# ---------------------------------------------------------------------------
# Suggest all params from an Optuna trial
# ---------------------------------------------------------------------------
def suggest_all_params(trial):
    """Suggest params for all 20 strategies + enable/disable + priority."""
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
        params["d_target2_pct"] = trial.suggest_float("d_target2_pct", d_t1 + 2, max(d_t1 + 4, 25.0), step=2.0)
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
        params["v_target2_pct"] = trial.suggest_float("v_target2_pct", v_t1 + 2, max(v_t1 + 4, 25.0), step=2.0)
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
        params["p_target2_pct"] = trial.suggest_float("p_target2_pct", p_t1 + 2, max(p_t1 + 4, 30.0), step=2.0)
    else:
        params["p_target2_pct"] = p_t1
    params["p_trail_pct"] = trial.suggest_float("p_trail_pct", 1.0, 8.0, step=1.0)
    params["p_trail_activate_pct"] = trial.suggest_float("p_trail_activate_pct", 1.0, 8.0, step=1.0)
    params["p_time_limit_min"] = trial.suggest_int("p_time_limit_min", 30, 180, step=10)
    if params["p_confirm_above"] > params["p_confirm_window"]:
        raise optuna.TrialPruned()

    # === M: Midday Range Break (14 params) ===
    params["m_min_gap"] = trial.suggest_int("m_min_gap", 8, 25, step=2)
    params["m_morning_spike_pct"] = trial.suggest_float("m_morning_spike_pct", 3.0, 15.0, step=1.0)
    params["m_morning_candles"] = trial.suggest_int("m_morning_candles", 20, 60, step=10)
    params["m_range_start_candle"] = trial.suggest_int("m_range_start_candle", 40, 80, step=10)
    params["m_consolidation_len"] = trial.suggest_int("m_consolidation_len", 20, 60, step=10)
    params["m_max_range_pct"] = trial.suggest_float("m_max_range_pct", 3.0, 12.0, step=1.0)
    params["m_vol_ratio"] = trial.suggest_float("m_vol_ratio", 0.2, 1.0, step=0.2)
    params["m_max_entry_candle"] = trial.suggest_int("m_max_entry_candle", 100, 180, step=10)
    params["m_target1_pct"] = trial.suggest_float("m_target1_pct", 3.0, 12.0, step=1.0)
    params["m_stop_pct"] = trial.suggest_float("m_stop_pct", 3.0, 15.0, step=1.0)
    params["m_partial_sell_pct"] = trial.suggest_float("m_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["m_trail_pct"] = trial.suggest_float("m_trail_pct", 1.0, 6.0, step=1.0)
    params["m_trail_activate_pct"] = trial.suggest_float("m_trail_activate_pct", 1.0, 8.0, step=1.0)
    params["m_time_limit_min"] = trial.suggest_int("m_time_limit_min", 30, 180, step=10)

    # === R: Multi-Day Runner (10 params) ===
    params["r_day1_min_gap"] = trial.suggest_int("r_day1_min_gap", 20, 80, step=10)
    params["r_d2_pullback_pct"] = trial.suggest_float("r_d2_pullback_pct", 3.0, 15.0, step=1.0)
    params["r_pullback_window"] = trial.suggest_int("r_pullback_window", 10, 50, step=5)
    params["r_bounce_ref"] = trial.suggest_categorical("r_bounce_ref", ["d1_close", "d2_open"])
    params["r_max_entry_candle"] = trial.suggest_int("r_max_entry_candle", 20, 80, step=10)
    params["r_target1_pct"] = trial.suggest_float("r_target1_pct", 3.0, 20.0, step=1.0)
    params["r_stop_pct"] = trial.suggest_float("r_stop_pct", 3.0, 15.0, step=1.0)
    params["r_trail_pct"] = trial.suggest_float("r_trail_pct", 2.0, 10.0, step=1.0)
    params["r_trail_activate_pct"] = trial.suggest_float("r_trail_activate_pct", 2.0, 10.0, step=1.0)
    params["r_time_limit_min"] = trial.suggest_int("r_time_limit_min", 30, 180, step=10)

    # === W: Power Hour Breakout (14 params) ===
    params["w_min_gap"] = trial.suggest_int("w_min_gap", 8, 25, step=2)
    params["w_min_morning_run"] = trial.suggest_float("w_min_morning_run", 2.0, 8.0, step=1.0)
    params["w_consol_start"] = trial.suggest_int("w_consol_start", 20, 60, step=5)
    params["w_max_range_pct"] = trial.suggest_float("w_max_range_pct", 5.0, 20.0, step=1.0)
    params["w_max_vwap_dev_pct"] = trial.suggest_float("w_max_vwap_dev_pct", 2.0, 10.0, step=1.0)
    params["w_earliest_candle"] = trial.suggest_int("w_earliest_candle", 150, 180, step=5)
    params["w_latest_candle"] = trial.suggest_int("w_latest_candle", 185, 195, step=5)
    params["w_vol_surge_mult"] = trial.suggest_float("w_vol_surge_mult", 1.0, 4.0, step=0.5)
    params["w_vol_vs_morning_mult"] = trial.suggest_float("w_vol_vs_morning_mult", 0.05, 0.3, step=0.05)
    params["w_max_hod_breaks"] = trial.suggest_int("w_max_hod_breaks", 1, 5)
    params["w_target_pct"] = trial.suggest_float("w_target_pct", 3.0, 15.0, step=1.0)
    params["w_stop_pct"] = trial.suggest_float("w_stop_pct", 1.0, 6.0, step=0.5)
    params["w_trail_pct"] = trial.suggest_float("w_trail_pct", 1.0, 5.0, step=0.5)
    params["w_trail_activate_pct"] = trial.suggest_float("w_trail_activate_pct", 1.0, 5.0, step=0.5)

    # === O: Opening Range Breakout (11 params) ===
    params["o_min_gap"] = trial.suggest_int("o_min_gap", 8, 25, step=2)
    params["o_range_candles"] = trial.suggest_int("o_range_candles", 3, 10)
    params["o_breakout_vol_mult"] = trial.suggest_float("o_breakout_vol_mult", 1.0, 3.0, step=0.5)
    params["o_max_entry_candle"] = trial.suggest_int("o_max_entry_candle", 15, 45, step=5)
    params["o_target1_pct"] = trial.suggest_float("o_target1_pct", 3.0, 15.0, step=1.0)
    o_t1 = params["o_target1_pct"]
    params["o_target2_pct"] = trial.suggest_float("o_target2_pct", o_t1, max(o_t1 + 2, 25.0), step=2.0)
    params["o_stop_pct"] = trial.suggest_float("o_stop_pct", 0.0, 8.0, step=2.0)  # 0=dynamic
    params["o_partial_sell_pct"] = trial.suggest_float("o_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["o_trail_pct"] = trial.suggest_float("o_trail_pct", 1.0, 5.0, step=1.0)
    params["o_trail_activate_pct"] = trial.suggest_float("o_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["o_time_limit_min"] = trial.suggest_int("o_time_limit_min", 20, 90, step=10)

    # === B: Red-to-Green R2G (11 params) ===
    params["b_min_gap"] = trial.suggest_int("b_min_gap", 10, 30, step=2)
    params["b_max_dip_pct"] = trial.suggest_float("b_max_dip_pct", 2.0, 10.0, step=1.0)
    params["b_min_reclaim_vol_mult"] = trial.suggest_float("b_min_reclaim_vol_mult", 1.0, 3.0, step=0.5)
    params["b_max_entry_candle"] = trial.suggest_int("b_max_entry_candle", 10, 30, step=5)
    params["b_target1_pct"] = trial.suggest_float("b_target1_pct", 3.0, 12.0, step=1.0)
    b_t1 = params["b_target1_pct"]
    params["b_target2_pct"] = trial.suggest_float("b_target2_pct", b_t1, max(b_t1 + 2, 20.0), step=2.0)
    params["b_stop_pct"] = trial.suggest_float("b_stop_pct", 2.0, 8.0, step=1.0)
    params["b_partial_sell_pct"] = trial.suggest_float("b_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["b_trail_pct"] = trial.suggest_float("b_trail_pct", 1.0, 5.0, step=1.0)
    params["b_trail_activate_pct"] = trial.suggest_float("b_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["b_time_limit_min"] = trial.suggest_int("b_time_limit_min", 10, 60, step=10)

    # === K: First Pullback Buy (14 params) ===
    params["k_min_gap"] = trial.suggest_int("k_min_gap", 8, 25, step=2)
    params["k_min_run_pct"] = trial.suggest_float("k_min_run_pct", 3.0, 10.0, step=1.0)
    params["k_run_window"] = trial.suggest_int("k_run_window", 8, 25, step=3)
    params["k_pullback_pct"] = trial.suggest_float("k_pullback_pct", 1.0, 6.0, step=1.0)
    params["k_pullback_vol_ratio"] = trial.suggest_float("k_pullback_vol_ratio", 0.2, 1.0, step=0.2)
    params["k_bounce_vol_mult"] = trial.suggest_float("k_bounce_vol_mult", 1.0, 3.0, step=0.5)
    params["k_max_entry_candle"] = trial.suggest_int("k_max_entry_candle", 25, 60, step=5)
    params["k_target1_pct"] = trial.suggest_float("k_target1_pct", 3.0, 15.0, step=1.0)
    k_t1 = params["k_target1_pct"]
    params["k_target2_pct"] = trial.suggest_float("k_target2_pct", k_t1, max(k_t1 + 2, 25.0), step=2.0)
    params["k_stop_pct"] = trial.suggest_float("k_stop_pct", 2.0, 10.0, step=1.0)
    params["k_partial_sell_pct"] = trial.suggest_float("k_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["k_trail_pct"] = trial.suggest_float("k_trail_pct", 1.0, 5.0, step=1.0)
    params["k_trail_activate_pct"] = trial.suggest_float("k_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["k_time_limit_min"] = trial.suggest_int("k_time_limit_min", 20, 90, step=10)

    # === C: Micro Flag / Base Pattern (14 params) ===
    params["c_min_gap"] = trial.suggest_int("c_min_gap", 8, 25, step=2)
    params["c_min_spike_pct"] = trial.suggest_float("c_min_spike_pct", 3.0, 10.0, step=1.0)
    params["c_min_base_candles"] = trial.suggest_int("c_min_base_candles", 2, 6)
    params["c_max_base_candles"] = trial.suggest_int("c_max_base_candles", 5, 12)
    params["c_max_base_range_pct"] = trial.suggest_float("c_max_base_range_pct", 1.0, 5.0, step=0.5)
    params["c_breakout_vol_mult"] = trial.suggest_float("c_breakout_vol_mult", 1.0, 3.0, step=0.5)
    params["c_max_entry_candle"] = trial.suggest_int("c_max_entry_candle", 30, 90, step=10)
    params["c_target1_pct"] = trial.suggest_float("c_target1_pct", 3.0, 15.0, step=1.0)
    c_t1 = params["c_target1_pct"]
    params["c_target2_pct"] = trial.suggest_float("c_target2_pct", c_t1, max(c_t1 + 2, 25.0), step=2.0)
    params["c_stop_pct"] = trial.suggest_float("c_stop_pct", 2.0, 8.0, step=1.0)
    params["c_partial_sell_pct"] = trial.suggest_float("c_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["c_trail_pct"] = trial.suggest_float("c_trail_pct", 1.0, 5.0, step=1.0)
    params["c_trail_activate_pct"] = trial.suggest_float("c_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["c_time_limit_min"] = trial.suggest_int("c_time_limit_min", 20, 90, step=10)
    if params["c_min_base_candles"] > params["c_max_base_candles"]:
        raise optuna.TrialPruned()

    # === S: Stuff-and-Break (13 params) ===
    params["s_min_gap"] = trial.suggest_int("s_min_gap", 8, 25, step=2)
    params["s_min_hod_tests"] = trial.suggest_int("s_min_hod_tests", 2, 5)
    params["s_hod_tolerance_pct"] = trial.suggest_float("s_hod_tolerance_pct", 0.2, 1.5, step=0.2)
    params["s_rejection_pct"] = trial.suggest_float("s_rejection_pct", 0.5, 3.0, step=0.5)
    params["s_breakout_vol_mult"] = trial.suggest_float("s_breakout_vol_mult", 1.0, 3.0, step=0.5)
    params["s_max_entry_candle"] = trial.suggest_int("s_max_entry_candle", 45, 120, step=15)
    params["s_target1_pct"] = trial.suggest_float("s_target1_pct", 3.0, 15.0, step=1.0)
    s_t1 = params["s_target1_pct"]
    params["s_target2_pct"] = trial.suggest_float("s_target2_pct", s_t1, max(s_t1 + 2, 25.0), step=2.0)
    params["s_stop_pct"] = trial.suggest_float("s_stop_pct", 2.0, 8.0, step=1.0)
    params["s_partial_sell_pct"] = trial.suggest_float("s_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["s_trail_pct"] = trial.suggest_float("s_trail_pct", 1.0, 5.0, step=1.0)
    params["s_trail_activate_pct"] = trial.suggest_float("s_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["s_time_limit_min"] = trial.suggest_int("s_time_limit_min", 30, 120, step=10)

    # === E: Gap-and-Go RelVol (10 params) ===
    params["e_min_gap"] = trial.suggest_int("e_min_gap", 10, 30, step=2)
    params["e_min_pm_vol_mult"] = trial.suggest_float("e_min_pm_vol_mult", 2.0, 10.0, step=1.0)
    params["e_max_entry_candle"] = trial.suggest_int("e_max_entry_candle", 2, 10)
    params["e_target1_pct"] = trial.suggest_float("e_target1_pct", 3.0, 12.0, step=1.0)
    e_t1 = params["e_target1_pct"]
    params["e_target2_pct"] = trial.suggest_float("e_target2_pct", e_t1, max(e_t1 + 2, 20.0), step=2.0)
    params["e_stop_pct"] = trial.suggest_float("e_stop_pct", 2.0, 8.0, step=1.0)
    params["e_partial_sell_pct"] = trial.suggest_float("e_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["e_trail_pct"] = trial.suggest_float("e_trail_pct", 1.0, 5.0, step=1.0)
    params["e_trail_activate_pct"] = trial.suggest_float("e_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["e_time_limit_min"] = trial.suggest_int("e_time_limit_min", 10, 40, step=5)

    # === I: P1 Immediate PM High Breakout (10 params) ===
    params["i_min_gap"] = trial.suggest_int("i_min_gap", 8, 25, step=2)
    params["i_max_entry_candle"] = trial.suggest_int("i_max_entry_candle", 10, 45, step=5)
    params["i_breakout_vol_mult"] = trial.suggest_float("i_breakout_vol_mult", 1.0, 3.0, step=0.5)
    params["i_target1_pct"] = trial.suggest_float("i_target1_pct", 3.0, 15.0, step=1.0)
    i_t1 = params["i_target1_pct"]
    params["i_target2_pct"] = trial.suggest_float("i_target2_pct", i_t1, max(i_t1 + 2, 25.0), step=2.0)
    params["i_stop_pct"] = trial.suggest_float("i_stop_pct", 2.0, 10.0, step=1.0)
    params["i_partial_sell_pct"] = trial.suggest_float("i_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["i_trail_pct"] = trial.suggest_float("i_trail_pct", 1.0, 5.0, step=1.0)
    params["i_trail_activate_pct"] = trial.suggest_float("i_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["i_time_limit_min"] = trial.suggest_int("i_time_limit_min", 20, 90, step=10)

    # === J: P3 VWAP + PM High Breakout (10 params) ===
    params["j_min_gap"] = trial.suggest_int("j_min_gap", 8, 25, step=2)
    params["j_max_entry_candle"] = trial.suggest_int("j_max_entry_candle", 30, 120, step=10)
    params["j_vwap_proximity_pct"] = trial.suggest_float("j_vwap_proximity_pct", 1.0, 5.0, step=0.5)
    params["j_target1_pct"] = trial.suggest_float("j_target1_pct", 3.0, 15.0, step=1.0)
    j_t1 = params["j_target1_pct"]
    params["j_target2_pct"] = trial.suggest_float("j_target2_pct", j_t1, max(j_t1 + 2, 25.0), step=2.0)
    params["j_stop_pct"] = trial.suggest_float("j_stop_pct", 2.0, 10.0, step=1.0)
    params["j_partial_sell_pct"] = trial.suggest_float("j_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["j_trail_pct"] = trial.suggest_float("j_trail_pct", 1.0, 5.0, step=1.0)
    params["j_trail_activate_pct"] = trial.suggest_float("j_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["j_time_limit_min"] = trial.suggest_int("j_time_limit_min", 30, 120, step=10)

    # === N: P4 HOD Reclaim (11 params) ===
    params["n_min_gap"] = trial.suggest_int("n_min_gap", 8, 25, step=2)
    params["n_min_hod_age"] = trial.suggest_int("n_min_hod_age", 5, 20, step=5)
    params["n_pullback_from_hod_pct"] = trial.suggest_float("n_pullback_from_hod_pct", 1.0, 6.0, step=1.0)
    params["n_max_entry_candle"] = trial.suggest_int("n_max_entry_candle", 60, 150, step=15)
    params["n_target1_pct"] = trial.suggest_float("n_target1_pct", 3.0, 15.0, step=1.0)
    n_t1 = params["n_target1_pct"]
    params["n_target2_pct"] = trial.suggest_float("n_target2_pct", n_t1, max(n_t1 + 2, 25.0), step=2.0)
    params["n_stop_pct"] = trial.suggest_float("n_stop_pct", 2.0, 10.0, step=1.0)
    params["n_partial_sell_pct"] = trial.suggest_float("n_partial_sell_pct", 0.0, 75.0, step=25.0)
    params["n_trail_pct"] = trial.suggest_float("n_trail_pct", 1.0, 5.0, step=1.0)
    params["n_trail_activate_pct"] = trial.suggest_float("n_trail_activate_pct", 1.0, 6.0, step=1.0)
    params["n_time_limit_min"] = trial.suggest_int("n_time_limit_min", 30, 120, step=10)

    # === L: Low Float Squeeze (19 params) ===
    params["l_min_gap"] = trial.suggest_int("l_min_gap", 15, 50, step=5)
    params["l_max_float"] = trial.suggest_int("l_max_float", 5_000_000, 20_000_000, step=5_000_000)
    params["l_earliest_candle"] = trial.suggest_int("l_earliest_candle", 3, 15, step=3)
    params["l_latest_candle"] = trial.suggest_int("l_latest_candle", 60, 150, step=15)
    params["l_vol_surge_mult"] = trial.suggest_float("l_vol_surge_mult", 1.0, 3.0, step=0.5)
    params["l_min_price_accel_pct"] = trial.suggest_float("l_min_price_accel_pct", 0.5, 3.0, step=0.5)
    params["l_tier1_float"] = trial.suggest_int("l_tier1_float", 500_000, 2_000_000, step=500_000)
    params["l_tier2_float"] = trial.suggest_int("l_tier2_float", 3_000_000, 7_000_000, step=1_000_000)
    params["l_tier1_target1_pct"] = trial.suggest_float("l_tier1_target1_pct", 15.0, 50.0, step=5.0)
    params["l_tier1_target2_pct"] = trial.suggest_float("l_tier1_target2_pct", 30.0, 60.0, step=5.0)
    params["l_tier2_target1_pct"] = trial.suggest_float("l_tier2_target1_pct", 8.0, 30.0, step=2.0)
    params["l_tier2_target2_pct"] = trial.suggest_float("l_tier2_target2_pct", 20.0, 50.0, step=5.0)
    params["l_tier3_target1_pct"] = trial.suggest_float("l_tier3_target1_pct", 5.0, 20.0, step=1.0)
    params["l_tier3_target2_pct"] = trial.suggest_float("l_tier3_target2_pct", 15.0, 40.0, step=5.0)
    params["l_stop_pct"] = trial.suggest_float("l_stop_pct", 5.0, 20.0, step=1.0)
    params["l_partial_sell_pct"] = trial.suggest_float("l_partial_sell_pct", 0.0, 50.0, step=25.0)
    params["l_trail_pct"] = trial.suggest_float("l_trail_pct", 1.0, 6.0, step=1.0)
    params["l_trail_activate_pct"] = trial.suggest_float("l_trail_activate_pct", 1.0, 8.0, step=1.0)
    params["l_time_limit_min"] = trial.suggest_int("l_time_limit_min", 30, 120, step=10)

    # === Strategy Enable/Disable (20 params) ===
    for s in ALL_STRATS:
        params[f"enable_{s}"] = trial.suggest_categorical(f"enable_{s}", [True, False])
    enabled = [params[f"enable_{s}"] for s in ALL_STRATS]
    if not any(enabled):
        raise optuna.TrialPruned()

    # === Strategy Priority (20 params) ===
    for s in ALL_STRATS:
        params[f"priority_{s}"] = trial.suggest_int(f"priority_{s}", 0, 19)

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
# Dump best params to JSON (for live backtest runs)
# ---------------------------------------------------------------------------
BEST_PARAMS_FILE = "optuna_best_params_v8.json"

def dump_best_params(trial, elapsed_min=None):
    """Save best trial's full params + summary to JSON file."""
    bp = dict(trial.params)
    ua = trial.user_attrs

    data = {
        "trial_number": trial.number,
        "score": trial.value,
        "total_pnl": ua.get("total_pnl", 0),
        "pf": ua.get("pf", 0),
        "wr": ua.get("wr", 0),
        "trades": ua.get("n", 0),
        "equity": ua.get("equity", 0),
        "enabled": ua.get("enabled", ""),
        "n_strategies": ua.get("n_strategies", 0),
        "priority": ua.get("priority", ""),
        "elapsed_minutes": round(elapsed_min, 1) if elapsed_min else None,
        "per_strategy": {},
        "params": bp,
    }

    for strat in STRAT_KEYS:
        sn = ua.get(f"{strat}_n", 0)
        if sn > 0:
            data["per_strategy"][strat] = {
                "trades": sn,
                "wr": ua.get(f"{strat}_wr", 0),
                "pnl": ua.get(f"{strat}_pnl", 0),
            }

    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"      -> Params saved to {BEST_PARAMS_FILE}")


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
            dump_best_params(trial, elapsed / 60)
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
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--dump-best", action="store_true",
                        help="Extract best params from existing DB and exit (no new trials)")
    args = parser.parse_args()
    n_trials = args.trials

    db_path = "optuna_combined_v8.db"

    # --dump-best: extract best params from existing DB without running trials
    if args.dump_best:
        if not os.path.exists(db_path):
            print(f"ERROR: {db_path} not found. Run optimizer first.")
            sys.exit(1)
        study = optuna.create_study(
            direction="maximize",
            study_name="combined_v8_20strats_2024_2026",
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
        )
        if len(study.trials) == 0:
            print("No completed trials in DB.")
            sys.exit(1)
        best = study.best_trial
        dump_best_params(best)
        print(f"\nBest trial {best.number}:")
        print(f"  Score: {best.value:,.0f}")
        ua = best.user_attrs
        print(f"  PnL: ${ua.get('total_pnl',0):,.0f} | PF: {ua.get('pf',0):.2f} | "
              f"WR: {ua.get('wr',0):.1f}% | Trades: {ua.get('n',0)}")
        print(f"  Enabled: {ua.get('enabled','?')} ({ua.get('n_strategies','?')} strategies)")
        print(f"  Priority: {ua.get('priority','?')}")
        print(f"\nFull params written to {BEST_PARAMS_FILE}")
        sys.exit(0)

    print("=" * 70)
    print("Combined Optuna Optimizer v8: 20 Strategies + Single Pool")
    print(f"  Candidates: {', '.join(STRAT_KEYS)}")
    print(f"  Optuna decides: which strategies to enable (1-20) + priority")
    print(f"  Single pool: ${STARTING_CASH:,}")
    print(f"  Trials: {n_trials}")
    print(f"  Objective: total_pnl * min(pf, 3.0)")
    print(f"  Data: {DATA_DIRS}")
    print("=" * 70)

    print("\nLoading data...")
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")
    study = optuna.create_study(
        direction="maximize",
        study_name="combined_v8_20strats_2024_2026",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=80),
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
    prio = {s: bp.get(f"priority_{s.lower()}", 99) for s in STRAT_KEYS}
    prio_order = sorted(prio.items(), key=lambda x: x[1])
    print(f"\n  Priority order: {' > '.join(s for s, _ in prio_order)}")

    strat_labels = {
        "h": "H (High Conv)", "g": "G (Gap Runner)", "a": "A (Quick Scalp)",
        "f": "F (Catch-All)", "d": "D (Dip Buy)", "v": "V (VWAP)",
        "p": "P (PM High)", "m": "M (Midday)", "r": "R (Multi-Day)",
        "w": "W (Power Hour)", "o": "O (ORB)", "b": "B (R2G)",
        "k": "K (Pullback)", "c": "C (Micro Flag)", "s": "S (Stuff-Break)",
        "e": "E (Gap-Go)", "i": "I (PM Immed)", "j": "J (VWAP PM)",
        "n": "N (HOD Reclaim)", "l": "L (Low Float)",
    }
    for prefix in ALL_STRATS:
        label = strat_labels.get(prefix, prefix.upper())
        pkeys = sorted(k for k in bp if k.startswith(f"{prefix}_") and not k.startswith("priority"))
        if pkeys:
            print(f"\n  Strategy {label}:")
            for k in pkeys:
                print(f"    {k}: {bp[k]}")

    enabled_list = [s.upper() for s in ALL_STRATS if bp.get(f"enable_{s}", True)]
    print(f"\n  Enabled strategies: {', '.join(enabled_list)} ({len(enabled_list)} of 20)")

    # Final dump of best params to JSON
    dump_best_params(best, total_time / 60)
    print(f"\n  DB saved to: {db_path}")
    print(f"  Params saved to: {BEST_PARAMS_FILE}")
    print(f"  Study name: combined_v8_20strats_2024_2026")


if __name__ == "__main__":
    main()
