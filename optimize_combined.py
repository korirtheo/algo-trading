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
import multiprocessing
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

DATA_DIRS = [
    # Full 2021-2026 training window (~1330 days across 5+ years, 4 distinct
    # regimes): SPAC/meme 2021, bear 2022, recovery 2023, momentum/AI 2024-25,
    # mixed 2026 YTD. Maximum regime diversity. NO held-out OOS in this run —
    # comparison against trial #6 on the same dataset is the validation.
    # Skipped (still broken — sparse premarket coverage):
    #   stored_data_jan_feb_2024          -> mostly IEX-feed bars (2-7% PM)
    "stored_data_2021",           # 2021-01 -> 2021-12        (252 days)
    "stored_data_2022",           # 2022-01 -> 2022-12        (251 days)
    "stored_data_2023",           # 2023-01 -> 2023-12        (250 days)
    "stored_data_jan_mar_2024",   # 2024-03                    (18 days)
    "stored_data_apr_jun_2024",   # 2024-04 -> 2024-06         (62 days)
    "stored_data_jul_sep_2024",   # 2024-07 -> 2024-09         (63 days)
    "stored_data_oct_dec_2024",   # 2024-10 -> 2024-12         (63 days)
    "stored_data_jan_mar_2025",   # 2025-01 -> 2025-03         (59 days)
    "stored_data_apr_jun_2025",   # 2025-04 -> 2025-06         (61 days)
    "stored_data_jul_2025",       # 2025-07                    (21 days)
    "stored_data_oos",            # 2025-08 -> 2025-12        (105 days)
    "stored_data",                # 2026-01 -> 2026-02         (38 days)
    "stored_data_mar_may_2026",   # 2026-03 -> 2026-05         (43 days)
]
DATE_RANGE = ("2021-01-01", "2026-05-31")

GL_TRAIL_DATA_DIRS = [
    "stored_data_2022", "stored_data_2023",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data", "stored_data_mar_may_2026",
    "stored_data_jun_2026",
]
GL_TRAIL_DATE_LO = "2024-01-01"
GL_TRAIL_DATE_HI = "2026-02-28"

ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
STRAT_KEYS = [s.upper() for s in ALL_STRATS]


# ---------------------------------------------------------------------------
# Set strategy params on tgc module globals
# ---------------------------------------------------------------------------
def set_strategy_params(params):
    """Set all strategy globals on the tgc module."""
    # Reset ALL min_gap to defaults (enable/disable may have set to 9999).
    # G and L gap thresholds become TUNABLE (defaults preserved if param missing).
    tgc.H_MIN_GAP_PCT = 35.0
    tgc.G_MIN_GAP_PCT = float(params.get("g_min_gap_pct", 30.0))
    # W11: 2nd-green + 2nd-new-high gates made tunable. Ablation 2026-06-22
    # showed `2nd_new_high=False` lifts geomean 22.8x -> 27.7x across 2024-26.
    tgc.G_REQUIRE_2ND_GREEN = bool(params.get("g_require_2nd_green", True))
    tgc.G_REQUIRE_2ND_NEW_HIGH = bool(params.get("g_require_2nd_new_high", True))
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
    # L_MIN_GAP_PCT: was hardcoded 30%, ignoring l_min_gap from suggest_all_params.
    # Now properly wired so the existing l_min_gap tunable actually controls entry.
    tgc.L_MIN_GAP_PCT = float(params.get("l_min_gap", 30.0))

    # --- H (High Conviction) ---
    tgc.H_TARGET_PCT = params["h_target_pct"]
    tgc.H_TIME_LIMIT_MINUTES = params["h_time_limit_min"]
    tgc.H_STOP_PCT = params["h_stop_pct"]
    tgc.H_TRAIL_PCT = params["h_trail_pct"]
    tgc.H_TRAIL_ACTIVATE_PCT = params["h_trail_activate_pct"]

    # --- G (Big Gap Runner) ---
    tgc.G_TARGET_PCT = params["g_target_pct"]
    tgc.G_TARGET2_PCT = params.get("g_target2_pct", 30.0)
    tgc.G_PARTIAL_SELL_PCT = params.get("g_partial_sell_pct", 0.0)
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

    # --- Strategy X: Range Reversion (always applies — uses defaults if absent) ---
    tgc.X_MIN_PULLBACK_PCT      = params.get("x_min_pullback_pct",      10.0)
    tgc.X_MIN_RECOVERY_PCT      = params.get("x_min_recovery_pct",       3.0)
    tgc.X_MIN_BARS_SINCE_PEAK   = params.get("x_min_bars_since_peak",    5)
    tgc.X_TARGET_PCT_OF_PEAK    = params.get("x_target_pct_of_peak",    92.0)
    tgc.X_STOP_PCT_BELOW_TROUGH = params.get("x_stop_pct_below_trough",  2.0)
    tgc.X_TRAIL_PCT             = params.get("x_trail_pct",              5.0)
    tgc.X_TRAIL_ACTIVATE_PCT    = params.get("x_trail_activate_pct",     5.0)
    tgc.X_TIME_LIMIT_MINUTES    = params.get("x_time_limit_min",        60)
    tgc.X_MIN_ENTRY_ROOM_PCT    = params.get("x_min_entry_room_pct",     4.0)
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = params.get("x_min_first_leg_gain_pct", 5.0)

    # --- Strategy enable/disable (Optuna decides which to include) ---
    # For X (no MIN_GAP_PCT), the kill-switch flips X_MIN_FIRST_LEG_GAIN_PCT to 9999
    # so no day can ever qualify as a valid first leg.
    _gap_keys = {
        "h": "H_MIN_GAP_PCT", "g": "G_MIN_GAP_PCT", "a": "A_MIN_GAP_PCT", "f": "F_MIN_GAP_PCT",
        "d": "D_MIN_GAP_PCT", "v": "V_MIN_GAP_PCT", "p": "P_MIN_GAP_PCT",
        "m": "M_MIN_GAP_PCT", "w": "W_MIN_GAP_PCT",
        "o": "O_MIN_GAP_PCT", "b": "B_MIN_GAP_PCT", "k": "K_MIN_GAP_PCT",
        "c": "C_MIN_GAP_PCT", "s": "S_MIN_GAP_PCT", "e": "E_MIN_GAP_PCT",
        "i": "I_MIN_GAP_PCT", "j": "J_MIN_GAP_PCT", "n": "N_MIN_GAP_PCT",
        "l": "L_MIN_GAP_PCT",
        "x": "X_MIN_FIRST_LEG_GAIN_PCT",
    }
    for s in ALL_STRATS:
        if s == "r":
            if not params.get("enable_r", True):
                tgc.R_DAY1_MIN_GAP = 9999.0
        else:
            if not params.get(f"enable_{s}", True):
                setattr(tgc, _gap_keys[s], 9999.0)

    # --- Strategy priority ---
    # Default priorities for any strategy missing from params (e.g., baseline
    # config files that pre-date a new strategy addition).
    _default_priority = {
        "h": 0, "g": 1, "a": 2, "f": 3, "d": 4, "v": 5, "p": 6,
        "m": 7, "r": 8, "w": 9, "o": 10, "b": 11, "k": 12,
        "c": 13, "s": 14, "e": 15, "i": 16, "j": 17, "n": 18,
        "l": 19, "x": 20,
    }
    tgc.STRAT_PRIORITY = {
        s.upper(): params.get(f"priority_{s}", _default_priority[s])
        for s in ALL_STRATS
    }

    # ---------------------------------------------------------------------
    # PHASE 1A adaptive controls (added 2026-06-17)
    # All default to 0 (no filter / use global) for backward compatibility:
    # configs that don't have these keys behave exactly as before.
    # ---------------------------------------------------------------------
    tgc.MIN_PRICE = float(params.get("min_price", 0.0))
    tgc.MAX_MODELED_SLIP_BP = float(params.get("max_modeled_slip_bp", 0.0))
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = float(params.get("max_cum_dvol_at_entry_m", 0.0))
    tgc.MIN_ATR_PCT = float(params.get("min_atr_pct", 0.0))

    # Per-strategy participation caps
    for s in ALL_STRATS:
        key = f"{s}_participation_cap"
        setattr(tgc, f"{s.upper()}_PARTICIPATION_CAP",
                float(params.get(key, 0.0)))

    # Weighted favorability signals
    tgc.W_NOT_WED        = float(params.get("w_not_wed", 0.0))
    tgc.W_NOT_JUN_AUG    = float(params.get("w_not_jun_aug", 0.0))
    tgc.W_DXY_UP         = float(params.get("w_dxy_up", 0.0))
    tgc.W_BTC_FLAT       = float(params.get("w_btc_flat", 0.0))
    tgc.W_VIX_MID        = float(params.get("w_vix_mid", 0.0))
    tgc.W_IWM_MID        = float(params.get("w_iwm_mid", 0.0))
    tgc.W_PM_DVOL_MID    = float(params.get("w_pm_dvol_mid", 0.0))
    tgc.MIN_FAVORABILITY_THRESHOLD = float(params.get("min_favorability_threshold", 0.0))
    tgc.BTC_FLAT_BANDWIDTH = float(params.get("btc_flat_bandwidth", 5.0))

    # PIT news filter (added 2026-06-18) — drops watchlist picks lacking
    # premarket news coverage. Defaults are "no filter" so configs without
    # these keys behave exactly as before.
    tgc.NEWS_FILTER_ENABLED = bool(params.get("enable_news_filter", False))
    tgc.NEWS_MIN_ARTICLES = int(params.get("min_news_articles", 0))
    tgc.NEWS_REQUIRE_CATALYST = bool(params.get("require_news_catalyst", False))


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

    # === G: Big Gap Runner (6 params) ===
    # g_min_gap_pct made TUNABLE (was hardcoded 30%). ChatGPT hypothesis: edge
    # may concentrate in larger gaps (e.g., 50%+). Range 15-80% covers both
    # below-default exploration and well above #254's effective threshold.
    params["g_min_gap_pct"] = trial.suggest_float("g_min_gap_pct", 15.0, 80.0, step=5.0)
    # W11: 2nd-candle structural gates made tunable.
    # W12: g_require_2nd_green hardcoded True via env var (ablation proved it load-bearing).
    import os as _os_g
    if _os_g.environ.get("HARDCODE_2ND_GREEN", "") == "1":
        params["g_require_2nd_green"] = True
    else:
        params["g_require_2nd_green"] = trial.suggest_categorical("g_require_2nd_green", [True, False])
    params["g_require_2nd_new_high"] = trial.suggest_categorical("g_require_2nd_new_high", [True, False])
    params["g_target_pct"] = trial.suggest_float("g_target_pct", 4.0, 20.0, step=1.0)
    params["g_time_limit_min"] = trial.suggest_int("g_time_limit_min", 3, 30, step=3)
    import os as _os_gs
    _g_stop_lo = float(_os_gs.environ.get("G_STOP_MIN", "0.0"))
    params["g_stop_pct"] = trial.suggest_float("g_stop_pct", _g_stop_lo, 12.0, step=2.0)
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
    # l_min_gap range widened from 15-50 to 15-80 (2026-06-20) so TPE can explore
    # the "edge concentrates in 50%+ gaps" hypothesis on the squeeze pattern too.
    import os as _os_l
    _l_lo = int(_os_l.environ.get("L_MIN_GAP_MIN", "15"))
    params["l_min_gap"] = trial.suggest_int("l_min_gap", _l_lo, 80, step=5)
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

    # === Strategy X: Range Reversion (second-leg pattern) ===
    params["x_min_pullback_pct"]      = trial.suggest_float("x_min_pullback_pct", 5.0, 30.0, step=2.5)
    params["x_min_recovery_pct"]      = trial.suggest_float("x_min_recovery_pct", 1.0, 12.0, step=1.0)
    params["x_min_bars_since_peak"]   = trial.suggest_int("x_min_bars_since_peak", 2, 20)
    params["x_target_pct_of_peak"]    = trial.suggest_float("x_target_pct_of_peak", 75.0, 100.0, step=2.5)
    params["x_stop_pct_below_trough"] = trial.suggest_float("x_stop_pct_below_trough", 0.5, 5.0, step=0.5)
    params["x_trail_pct"]             = trial.suggest_float("x_trail_pct", 2.0, 12.0, step=1.0)
    params["x_trail_activate_pct"]    = trial.suggest_float("x_trail_activate_pct", 1.0, 10.0, step=1.0)
    params["x_time_limit_min"]        = trial.suggest_int("x_time_limit_min", 30, 240, step=15)
    params["x_min_entry_room_pct"]    = trial.suggest_float("x_min_entry_room_pct", 2.0, 15.0, step=1.0)
    params["x_min_first_leg_gain_pct"] = trial.suggest_float("x_min_first_leg_gain_pct", 3.0, 20.0, step=1.0)

    # === Strategy Enable/Disable (21 params, including X) ===
    # USER OVERRIDE 2026-06-20: FORCE_ENABLE_STRATS env var locks the enable
    # bits to a fixed set (e.g., "g,l"). Used by W10+ specialist studies after
    # ablation revealed G is the dominant alpha source.
    import os as _os
    _force_strats_env = _os.environ.get("FORCE_ENABLE_STRATS", "").strip()
    if _force_strats_env:
        _force_allowed = {s.strip().lower() for s in _force_strats_env.split(",") if s.strip()}
        for s in ALL_STRATS:
            params[f"enable_{s}"] = (s in _force_allowed)
    else:
        for s in ALL_STRATS:
            params[f"enable_{s}"] = trial.suggest_categorical(f"enable_{s}", [True, False])
    enabled = [params[f"enable_{s}"] for s in ALL_STRATS]
    if not any(enabled):
        raise optuna.TrialPruned()

    # === Strategy Priority (21 params, including X) ===
    for s in ALL_STRATS:
        params[f"priority_{s}"] = trial.suggest_int(f"priority_{s}", 0, 20)

    # === PHASE 1A adaptive controls (added 2026-06-17, revised 2026-06-18) ===
    # Gated via PHASE_1A_ENABLED env var so legacy studies can run unchanged.
    # Each FILTER GROUP has a separate enable boolean so Optuna can choose
    # to disable groups entirely (matches #124 baseline behavior). Without
    # these gates, random startup combinations of filters produced 0 trades
    # because favorability threshold could exceed max possible score.
    import os as _os
    if _os.environ.get("PHASE_1A_ENABLED", "0") == "1":
        # 4 global filters, each with its own enable boolean.
        # When disabled, the suggested value is ignored (set to 0 = no filter).
        for fname, lo, hi, step in [
            ("min_price",                0.0,   3.0,   0.25),
            ("max_modeled_slip_bp",      20.0,  150.0, 10.0),
            ("max_cum_dvol_at_entry_m",  5.0,   100.0, 5.0),
            ("min_atr_pct",              0.5,   10.0,  0.5),
        ]:
            enabled = trial.suggest_categorical(f"enable_{fname}", [True, False])
            if enabled:
                params[fname] = trial.suggest_float(fname, lo, hi, step=step)
            else:
                params[fname] = 0.0

        # 20 per-strategy participation caps. Toggleable as a single group.
        per_strat_caps_enabled = trial.suggest_categorical(
            "enable_per_strategy_caps", [True, False])
        for s in ALL_STRATS:
            if per_strat_caps_enabled:
                params[f"{s}_participation_cap"] = trial.suggest_float(
                    f"{s}_participation_cap", 0.02, 0.20, step=0.02)
            else:
                params[f"{s}_participation_cap"] = 0.0  # use global MAX_2MIN_PARTICIPATION

        # Favorability gating. When disabled, threshold=0 means "never skip".
        fav_enabled = trial.suggest_categorical(
            "enable_favorability_gate", [True, False])
        if fav_enabled:
            # Threshold range tightened to 0.5-4.0 (max achievable score is
            # ~7 with all weights at +1, so 4 is a meaningful but reachable bar).
            params["w_not_wed"]      = trial.suggest_float("w_not_wed", -2.0, 4.0, step=0.5)
            params["w_not_jun_aug"]  = trial.suggest_float("w_not_jun_aug", -2.0, 4.0, step=0.5)
            params["w_dxy_up"]       = trial.suggest_float("w_dxy_up", -2.0, 4.0, step=0.5)
            params["w_btc_flat"]     = trial.suggest_float("w_btc_flat", -2.0, 4.0, step=0.5)
            params["w_vix_mid"]      = trial.suggest_float("w_vix_mid", -2.0, 4.0, step=0.5)
            params["w_iwm_mid"]      = trial.suggest_float("w_iwm_mid", -2.0, 4.0, step=0.5)
            params["w_pm_dvol_mid"]  = trial.suggest_float("w_pm_dvol_mid", -2.0, 4.0, step=0.5)
            params["min_favorability_threshold"] = trial.suggest_float(
                "min_favorability_threshold", 0.5, 4.0, step=0.5)
            params["btc_flat_bandwidth"] = trial.suggest_float(
                "btc_flat_bandwidth", 1.0, 15.0, step=1.0)
        else:
            params["w_not_wed"] = 0.0
            params["w_not_jun_aug"] = 0.0
            params["w_dxy_up"] = 0.0
            params["w_btc_flat"] = 0.0
            params["w_vix_mid"] = 0.0
            params["w_iwm_mid"] = 0.0
            params["w_pm_dvol_mid"] = 0.0
            params["min_favorability_threshold"] = 0.0
            params["btc_flat_bandwidth"] = 5.0

    # === PHASE NEWS: point-in-time news filter as a tunable (2026-06-18) ===
    # Gated by PHASE_NEWS_ENABLED env var so legacy studies are unaffected.
    if _os.environ.get("PHASE_NEWS_ENABLED", "0") == "1":
        params["enable_news_filter"] = trial.suggest_categorical(
            "enable_news_filter", [True, False])
        if params["enable_news_filter"]:
            params["min_news_articles"] = trial.suggest_int(
                "min_news_articles", 0, 5)
            params["require_news_catalyst"] = trial.suggest_categorical(
                "require_news_catalyst", [True, False])
        else:
            params["min_news_articles"] = 0
            params["require_news_catalyst"] = False

    return params


# ---------------------------------------------------------------------------
# Run full combined backtest with current tgc globals
# ---------------------------------------------------------------------------
def _build_param_snapshot():
    """Snapshot all current tgc strategy globals into a dict the simulator
    can use thread-safely. Called after set_strategy_params() has applied
    a trial's params to tgc; the snapshot is then passed explicitly to
    simulate_day_combined so it's immune to other threads stomping on
    tgc globals during this trial's backtest."""
    g = tgc.__dict__
    names = (
        # Helper-function consumers (_classify_candle2, _get_tiered_targets)
        'H_MIN_GAP_PCT', 'G_MIN_GAP_PCT', 'A_MIN_GAP_PCT', 'F_MIN_GAP_PCT',
        'L_TIER1_FLOAT', 'L_TIER2_FLOAT',
        'L_TIER1_TARGET1_PCT', 'L_TIER1_TARGET2_PCT',
        'L_TIER2_TARGET1_PCT', 'L_TIER2_TARGET2_PCT',
        'L_TIER3_TARGET1_PCT', 'L_TIER3_TARGET2_PCT',
        'A_STOP_PCT', 'A_TARGET_PCT', 'A_TIME_LIMIT_MINUTES', 'A_TRAIL_ACTIVATE_PCT', 'A_TRAIL_PCT',
        'B_MAX_DIP_PCT', 'B_MAX_ENTRY_CANDLE', 'B_MIN_GAP_PCT', 'B_MIN_RECLAIM_VOL_MULT',
        'B_PARTIAL_SELL_PCT', 'B_STOP_PCT', 'B_TARGET1_PCT', 'B_TARGET2_PCT', 'B_TIME_LIMIT_MINUTES',
        'B_TRAIL_ACTIVATE_PCT', 'B_TRAIL_PCT',
        'C_BREAKOUT_VOL_MULT', 'C_MAX_BASE_CANDLES', 'C_MAX_BASE_RANGE_PCT', 'C_MAX_ENTRY_CANDLE',
        'C_MIN_BASE_CANDLES', 'C_MIN_GAP_PCT', 'C_MIN_SPIKE_PCT', 'C_PARTIAL_SELL_PCT', 'C_STOP_PCT',
        'C_TARGET1_PCT', 'C_TARGET2_PCT', 'C_TIME_LIMIT_MINUTES', 'C_TRAIL_ACTIVATE_PCT', 'C_TRAIL_PCT',
        'D_DIP_PCT', 'D_ENTRY_MODE', 'D_MAX_ENTRY_CANDLE', 'D_MIN_GAP_PCT', 'D_MIN_SPIKE_PCT',
        'D_PARTIAL_SELL_PCT', 'D_SPIKE_WINDOW', 'D_STOP_PCT', 'D_TARGET1_PCT', 'D_TARGET2_PCT',
        'D_TIME_LIMIT_MINUTES', 'D_TRAIL_ACTIVATE_PCT', 'D_TRAIL_PCT',
        'EOD_EXIT_MINUTES',
        'E_MAX_ENTRY_CANDLE', 'E_MIN_GAP_PCT', 'E_MIN_PM_VOL_MULT', 'E_PARTIAL_SELL_PCT',
        'E_STOP_PCT', 'E_TARGET1_PCT', 'E_TARGET2_PCT', 'E_TIME_LIMIT_MINUTES',
        'E_TRAIL_ACTIVATE_PCT', 'E_TRAIL_PCT',
        'F_STOP_PCT', 'F_TARGET_PCT', 'F_TIME_LIMIT_MINUTES', 'F_TRAIL_ACTIVATE_PCT', 'F_TRAIL_PCT',
        'G_PARTIAL_SELL_PCT', 'G_STOP_PCT', 'G_TARGET_PCT', 'G_TARGET2_PCT',
        'G_TIME_LIMIT_MINUTES', 'G_TRAIL_ACTIVATE_PCT', 'G_TRAIL_PCT',
        'H_STOP_PCT', 'H_TARGET_PCT', 'H_TIME_LIMIT_MINUTES', 'H_TRAIL_ACTIVATE_PCT', 'H_TRAIL_PCT',
        'I_BREAKOUT_VOL_MULT', 'I_MAX_ENTRY_CANDLE', 'I_MIN_GAP_PCT', 'I_PARTIAL_SELL_PCT',
        'I_STOP_PCT', 'I_TARGET1_PCT', 'I_TARGET2_PCT', 'I_TIME_LIMIT_MINUTES',
        'I_TRAIL_ACTIVATE_PCT', 'I_TRAIL_PCT',
        'J_MAX_ENTRY_CANDLE', 'J_MIN_GAP_PCT', 'J_PARTIAL_SELL_PCT', 'J_STOP_PCT',
        'J_TARGET1_PCT', 'J_TARGET2_PCT', 'J_TIME_LIMIT_MINUTES', 'J_TRAIL_ACTIVATE_PCT',
        'J_TRAIL_PCT', 'J_VWAP_PROXIMITY_PCT',
        'K_BOUNCE_VOL_MULT', 'K_MAX_ENTRY_CANDLE', 'K_MIN_GAP_PCT', 'K_MIN_RUN_PCT',
        'K_PARTIAL_SELL_PCT', 'K_PULLBACK_PCT', 'K_PULLBACK_VOL_RATIO', 'K_RUN_WINDOW',
        'K_STOP_PCT', 'K_TARGET1_PCT', 'K_TARGET2_PCT', 'K_TIME_LIMIT_MINUTES',
        'K_TRAIL_ACTIVATE_PCT', 'K_TRAIL_PCT',
        'L_EARLIEST_CANDLE', 'L_HOD_BREAK_REQUIRED', 'L_LATEST_CANDLE', 'L_MAX_FLOAT',
        'L_MIN_GAP_PCT', 'L_MIN_PRICE_ACCEL_PCT', 'L_PARTIAL_SELL_PCT', 'L_REQUIRE_ABOVE_VWAP',
        'L_STOP_PCT', 'L_TIME_LIMIT_MINUTES', 'L_TRAIL_ACTIVATE_PCT', 'L_TRAIL_PCT',
        'L_VOL_SURGE_MULT',
        'M_CONSOLIDATION_LEN', 'M_MAX_ENTRY_CANDLE', 'M_MAX_RANGE_PCT', 'M_MIN_GAP_PCT',
        'M_MORNING_CANDLES', 'M_MORNING_SPIKE_PCT', 'M_PARTIAL_SELL_PCT', 'M_RANGE_START_CANDLE',
        'M_STOP_PCT', 'M_TARGET1_PCT', 'M_TIME_LIMIT_MINUTES', 'M_TRAIL_ACTIVATE_PCT',
        'M_TRAIL_PCT', 'M_VOL_RATIO',
        'N_MAX_ENTRY_CANDLE', 'N_MIN_GAP_PCT', 'N_MIN_HOD_AGE', 'N_PARTIAL_SELL_PCT',
        'N_PULLBACK_FROM_HOD_PCT', 'N_STOP_PCT', 'N_TARGET1_PCT', 'N_TARGET2_PCT',
        'N_TIME_LIMIT_MINUTES', 'N_TRAIL_ACTIVATE_PCT', 'N_TRAIL_PCT',
        'O_BREAKOUT_VOL_MULT', 'O_MAX_ENTRY_CANDLE', 'O_MIN_GAP_PCT', 'O_PARTIAL_SELL_PCT',
        'O_RANGE_CANDLES', 'O_STOP_PCT', 'O_TARGET1_PCT', 'O_TARGET2_PCT',
        'O_TIME_LIMIT_MINUTES', 'O_TRAIL_ACTIVATE_PCT', 'O_TRAIL_PCT',
        'P_CONFIRM_ABOVE', 'P_CONFIRM_WINDOW', 'P_MAX_ENTRY_CANDLE', 'P_MIN_GAP_PCT',
        'P_PARTIAL_SELL_PCT', 'P_PULLBACK_PCT', 'P_PULLBACK_TIMEOUT', 'P_STOP_PCT',
        'P_TARGET1_PCT', 'P_TARGET2_PCT', 'P_TIME_LIMIT_MINUTES', 'P_TRAIL_ACTIVATE_PCT',
        'P_TRAIL_PCT',
        'R_BOUNCE_REF', 'R_D2_PULLBACK_PCT', 'R_MAX_ENTRY_CANDLE', 'R_PULLBACK_WINDOW',
        'R_STOP_PCT', 'R_TARGET1_PCT', 'R_TIME_LIMIT_MINUTES', 'R_TRAIL_ACTIVATE_PCT',
        'R_TRAIL_PCT',
        'S_BREAKOUT_VOL_MULT', 'S_HOD_TOLERANCE_PCT', 'S_MAX_ENTRY_CANDLE', 'S_MIN_GAP_PCT',
        'S_MIN_HOD_TESTS', 'S_PARTIAL_SELL_PCT', 'S_REJECTION_PCT', 'S_STOP_PCT',
        'S_TARGET1_PCT', 'S_TARGET2_PCT', 'S_TIME_LIMIT_MINUTES', 'S_TRAIL_ACTIVATE_PCT',
        'S_TRAIL_PCT',
        'VOL_CAP_PCT',
        'V_MAX_ENTRY_CANDLE', 'V_MIN_BELOW_CANDLES', 'V_MIN_BELOW_PCT', 'V_MIN_GAP_PCT',
        'V_PARTIAL_SELL_PCT', 'V_STOP_PCT', 'V_TARGET1_PCT', 'V_TARGET2_PCT',
        'V_TIME_LIMIT_MINUTES', 'V_TRAIL_ACTIVATE_PCT', 'V_TRAIL_PCT', 'V_VOL_SPIKE_RATIO',
        'W_CONSOL_START', 'W_EARLIEST_CANDLE', 'W_LATEST_CANDLE', 'W_MAX_HOD_BREAKS',
        'W_MAX_RANGE_PCT', 'W_MAX_VWAP_DEV_PCT', 'W_MIN_GAP_PCT', 'W_MIN_MORNING_RUN',
        'W_REQUIRE_ABOVE_VWAP', 'W_STOP_PCT', 'W_TARGET_PCT', 'W_TRAIL_ACTIVATE_PCT',
        'W_TRAIL_PCT', 'W_VOL_SURGE_MULT', 'W_VOL_VS_MORNING_MULT',
        'X_MIN_PULLBACK_PCT', 'X_MIN_RECOVERY_PCT', 'X_MIN_BARS_SINCE_PEAK',
        'X_TARGET_PCT_OF_PEAK', 'X_STOP_PCT_BELOW_TROUGH', 'X_TRAIL_PCT',
        'X_TRAIL_ACTIVATE_PCT', 'X_TIME_LIMIT_MINUTES', 'X_MIN_ENTRY_ROOM_PCT',
        'X_MIN_FIRST_LEG_GAIN_PCT', 'X_FIRST_LEG_WINDOW_BARS',
        'X_ENTRY_REQUIRE_GREEN', 'X_MIN_VOL_VS_AVG', 'X_VOL_AVG_BARS',
        'X_MAX_ENTRY_HHMM',
    )
    return {n: g[n] for n in names}


def run_combined_backtest(daily_picks, all_dates, params_snapshot=None):
    """Run the full backtest with single cash pool and return per-strategy stats.

    When params_snapshot is provided, each simulate_day_combined call uses it
    explicitly — fully parallel-safe regardless of n_jobs."""
    cash = float(STARTING_CASH)
    unsettled = 0.0
    all_trades = []
    daily_equity = [float(STARTING_CASH)]

    for d in all_dates:
        cash += unsettled
        unsettled = 0.0
        equity_start = cash

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        # PHASE NEWS: drop picks lacking premarket news coverage (PIT — only
        # articles published before 9:30 ET on day d count). No-op when the
        # filter is disabled or thresholds are trivial.
        if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as _news_filter
            picks = _news_filter(
                picks, d,
                min_articles=tgc.NEWS_MIN_ARTICLES,
                require_catalyst=tgc.NEWS_REQUIRE_CATALYST,
            )

        # PHASE 1A: day-level favorability gate. If macro is loaded and the
        # weighted score is below MIN_FAVORABILITY_THRESHOLD, skip the day
        # (no trades, cash carries forward unchanged).
        if tgc.MIN_FAVORABILITY_THRESHOLD > 0 and tgc._MACRO_DATA:
            _score, _ = tgc.compute_day_favorability(d, picks)
            if _score is not None and _score < tgc.MIN_FAVORABILITY_THRESHOLD:
                # Day gated out — record equity unchanged
                daily_equity.append(cash + unsettled)
                continue

        states, cash, unsettled, _ = tgc.simulate_day_combined(
            picks, cash, cash_account, params=params_snapshot
        )

        for st in states:
            if st["exit_reason"] is not None:
                all_trades.append({
                    "strategy": st.get("strategy", "?"),
                    "pnl": st["pnl"],
                    "position_cost": st["position_cost"],
                })

        # Track end-of-day equity (cash + unsettled to capture in-progress settlement)
        daily_equity.append(cash + unsettled)

    equity = cash + unsettled

    n = len(all_trades)
    if n == 0:
        return {"n": 0, "pf": 0, "total_pnl": -9999, "equity": equity,
                "sharpe_pct": 0.0, "strats": {}}

    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in all_trades if t["pnl"] <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 99
    wr = len(wins) / n * 100 if n > 0 else 0

    # Compute Sharpe + Sortino on daily percentage returns (slippage-robust metrics).
    # Sortino is the W18 metric: penalizes only DOWNSIDE variance (losses), not
    # total variance. Better fit than Sharpe for asymmetric strategies where
    # the BIG winners (50%+ pumps) shouldn't be penalized as "variance".
    import numpy as _np
    eq = _np.asarray(daily_equity, dtype=float)
    prev = eq[:-1]
    curr = eq[1:]
    mask = prev > 0
    sortino_pct = 0.0
    if mask.sum() > 1:
        ret = _np.where(mask, (curr - prev) / _np.where(prev == 0, 1, prev), 0)
        std = float(ret.std())
        sharpe_pct = float(ret.mean() / std * (252 ** 0.5)) if std > 0 else 0.0
        # Downside deviation: std of returns BELOW the MAR (here MAR=0)
        downside_returns = ret[ret < 0]
        if len(downside_returns) > 1:
            d_std = float(downside_returns.std())
            sortino_pct = float(ret.mean() / d_std * (252 ** 0.5)) if d_std > 0 else 0.0
        elif ret.mean() > 0:
            # No losing days at all — Sortino is technically infinite. Cap at a large finite.
            sortino_pct = 99.0
    else:
        sharpe_pct = 0.0

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
        "sharpe_pct": sharpe_pct,
        "sortino_pct": sortino_pct,
        "strats": strats,
    }


# ---------------------------------------------------------------------------
# Cross-validation backtest — runs the simulator INDEPENDENTLY per year with
# fresh $25K starting cash. Eliminates compounding distortion that masks
# per-year regime fragility from TPE.
#
# Empirical W7 finding (2026-06-19): min_year_pf correlates 0.638 with
# forward 2026 PnL vs the legacy objective's 0.281. Compounded backtest
# hides 2023's true cost; per-year reset makes it visible to TPE.
# ---------------------------------------------------------------------------
def run_combined_backtest_cv(daily_picks, all_dates, params_snapshot=None):
    """Returns dict keyed by year:
       { '2022': {'n':N, 'pf':PF, 'pnl':PNL, 'equity':EQ}, ... }
    Each year backtested independently with fresh STARTING_CASH.
    """
    from collections import defaultdict
    by_year = defaultdict(list)
    for d in all_dates:
        by_year[d[:4]].append(d)

    per_year = {}
    for year in sorted(by_year):
        cash = float(STARTING_CASH)
        unsettled = 0.0
        trades = []
        for d in sorted(by_year[year]):
            cash += unsettled
            unsettled = 0.0

            picks = daily_picks.get(d, [])
            cash_account = cash < MARGIN_THRESHOLD

            if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
                from news_filter import filter_picks as _news_filter
                picks = _news_filter(
                    picks, d,
                    min_articles=tgc.NEWS_MIN_ARTICLES,
                    require_catalyst=tgc.NEWS_REQUIRE_CATALYST,
                )

            if tgc.MIN_FAVORABILITY_THRESHOLD > 0 and tgc._MACRO_DATA:
                _s, _ = tgc.compute_day_favorability(d, picks)
                if _s is not None and _s < tgc.MIN_FAVORABILITY_THRESHOLD:
                    continue

            if not picks:
                continue

            states, cash, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account, params=params_snapshot
            )
            for st in states:
                if st["exit_reason"] is not None:
                    trades.append({"pnl": st["pnl"],
                                   "position_cost": st["position_cost"]})

        equity = cash + unsettled
        if not trades:
            per_year[year] = {"n": 0, "pf": 0.0, "pnl": 0.0, "equity": equity}
            continue
        pnl = sum(t["pnl"] for t in trades)
        wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        losses = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        pf = wins / losses if losses > 1e-9 else 99.0
        per_year[year] = {"n": len(trades), "pf": pf, "pnl": pnl, "equity": equity}

    return per_year


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
_param_lock = __import__("threading").RLock()

# Module-level flags for slippage-aware + Sharpe-aware objective.
# Default OFF for backward compatibility with existing studies.
# Walk-forward harness or main() can set these to True before calling objective.
USE_SHARPE_OBJECTIVE = False        # True => score = total_pnl * min(sharpe_pct, 4.0)
                                     # False => legacy score = total_pnl * min(pf, 3.0)
ENABLE_2MIN_SLIPPAGE = False         # True => set tgc.USE_2MIN_SLIPPAGE before each trial (superseded by ENABLE_MULTIWINDOW_SLIPPAGE)
ENABLE_MULTIWINDOW_SLIPPAGE = True   # True => set tgc.USE_MULTIWINDOW_SLIPPAGE before each trial (most realistic model)
USE_CV_OBJECTIVE = False             # True => per-year-reset CV:
                                     # score = min(year_pnl * min(year_pf, 3.0) across all years)
                                     # Each year starts at fresh STARTING_CASH so per-year
                                     # performance is visible to TPE (not masked by compounding).


def objective_val_multi_sortino(trial, daily_picks, train_dates, val_windows_list):
    """Sortino-based multi-window validation objective.

    Score per window: val_pnl × min(val_sortino, 4.0)
    Aggregate: min(per_window_score) — punishes worst-window overfit
    Sortino penalizes ONLY downside variance (losses), not the big winners,
    which is the right shape for asymmetric pump strategies. Solves the
    W16/W17 trap where no-stop / loose-gap trials had high val_pnl but
    catastrophic DD that Sharpe didn't catch (and PF too leniently penalized).
    """
    assert tgc.USE_DYNAMIC_SLIPPAGE and tgc.USE_MULTIWINDOW_SLIPPAGE, \
        "Slippage parity violation: both USE_DYNAMIC_SLIPPAGE and USE_MULTIWINDOW_SLIPPAGE must be True"
    params = suggest_all_params(trial)
    with _param_lock:
        set_strategy_params(params)
        if ENABLE_MULTIWINDOW_SLIPPAGE:
            tgc.USE_DYNAMIC_SLIPPAGE = True
            tgc.USE_MULTIWINDOW_SLIPPAGE = True
        elif ENABLE_2MIN_SLIPPAGE:
            tgc.USE_DYNAMIC_SLIPPAGE = True
            tgc.USE_2MIN_SLIPPAGE = True
        snapshot = _build_param_snapshot()

    def _safe(x, default=-9.9e12):
        import math
        try: x = float(x)
        except (TypeError, ValueError): return default
        if math.isnan(x) or math.isinf(x): return default
        return max(-9.9e12, min(9.9e12, x))

    train_result = run_combined_backtest(daily_picks, train_dates, params_snapshot=snapshot)
    if train_result["n"] < 100:
        return -9999
    if train_result["pf"] < 0.5:
        return -9999

    per_win_scores = []
    for label, win_dates in val_windows_list:
        r = run_combined_backtest(daily_picks, win_dates, params_snapshot=snapshot)
        if r["n"] < 5:
            return -9999
        pnl_safe = _safe(r["total_pnl"], -1e12)
        sortino_safe = _safe(r.get("sortino_pct", 0), 0)
        # Cap Sortino at 4.0 (like PF) to prevent runaway scores when downside
        # std is tiny but non-zero on calm val windows.
        win_score = pnl_safe * min(max(sortino_safe, 0.0), 4.0)
        per_win_scores.append(win_score)

        trial.set_user_attr(f"{label}_n", r["n"])
        trial.set_user_attr(f"{label}_pnl", round(pnl_safe, 2))
        trial.set_user_attr(f"{label}_pf", round(_safe(r["pf"]), 3))
        trial.set_user_attr(f"{label}_wr", round(_safe(r["wr"]), 1))
        trial.set_user_attr(f"{label}_sortino", round(sortino_safe, 3))
        trial.set_user_attr(f"{label}_sharpe", round(_safe(r.get("sharpe_pct", 0)), 3))
        trial.set_user_attr(f"{label}_equity", round(_safe(r["equity"]), 2))
        trial.set_user_attr(f"{label}_score", round(_safe(win_score), 2))

    # Training diagnostics (not optimized for, just observed)
    trial.set_user_attr("train_n", train_result["n"])
    trial.set_user_attr("train_pnl", round(_safe(train_result["total_pnl"]), 2))
    trial.set_user_attr("train_pf", round(_safe(train_result["pf"]), 3))
    trial.set_user_attr("train_wr", round(_safe(train_result["wr"]), 1))
    trial.set_user_attr("train_sortino", round(_safe(train_result.get("sortino_pct", 0)), 3))
    trial.set_user_attr("train_equity", round(_safe(train_result["equity"]), 2))

    enabled_strats = [s for s in STRAT_KEYS if params.get(f"enable_{s.lower()}", True)]
    prio = {s: params[f"priority_{s.lower()}"] for s in enabled_strats}
    prio_str = ">".join(s for s, _ in sorted(prio.items(), key=lambda x: x[1]))
    trial.set_user_attr("priority", prio_str)
    trial.set_user_attr("enabled", ",".join(enabled_strats))

    min_score = min(per_win_scores)
    trial.set_user_attr("min_val_score", round(_safe(min_score), 2))
    trial.set_user_attr("total_pnl", round(_safe(min_score), 2))
    trial.set_user_attr("pf", round(_safe(train_result["pf"]), 3))
    trial.set_user_attr("wr", round(_safe(train_result["wr"]), 1))
    trial.set_user_attr("n", train_result["n"])
    trial.set_user_attr("equity", 0)
    return _safe(min_score, -9999)


def objective_val_multi(trial, daily_picks, train_dates, val_windows_list):
    """Multi-window walk-forward validation objective.

    val_windows_list = [(label, [date_strs]), ...]  — one or more val periods.
    Objective = min(per_window val_pnl * min(val_pf, 3))
    A trial must score POSITIVELY on EVERY window to pass — punishes
    regime-specific overfit (the W16 trap: 38-day window had no DD event,
    so no-stop config looked free).
    """
    assert tgc.USE_DYNAMIC_SLIPPAGE and tgc.USE_MULTIWINDOW_SLIPPAGE, \
        "Slippage parity violation: both USE_DYNAMIC_SLIPPAGE and USE_MULTIWINDOW_SLIPPAGE must be True"
    params = suggest_all_params(trial)
    with _param_lock:
        set_strategy_params(params)
        if ENABLE_MULTIWINDOW_SLIPPAGE:
            tgc.USE_DYNAMIC_SLIPPAGE = True
            tgc.USE_MULTIWINDOW_SLIPPAGE = True
        elif ENABLE_2MIN_SLIPPAGE:
            tgc.USE_DYNAMIC_SLIPPAGE = True
            tgc.USE_2MIN_SLIPPAGE = True
        snapshot = _build_param_snapshot()

    def _safe(x, default=-9.9e12):
        import math
        try: x = float(x)
        except (TypeError, ValueError): return default
        if math.isnan(x) or math.isinf(x): return default
        return max(-9.9e12, min(9.9e12, x))

    train_result = run_combined_backtest(daily_picks, train_dates, params_snapshot=snapshot)
    if train_result["n"] < 100:
        return -9999
    if train_result["pf"] < 0.5:
        return -9999

    per_win_scores = []
    for label, win_dates in val_windows_list:
        r = run_combined_backtest(daily_picks, win_dates, params_snapshot=snapshot)
        if r["n"] < 5:
            return -9999  # any window with insufficient activity fails the trial
        pnl_safe = _safe(r["total_pnl"], -1e12)
        pf_safe = _safe(r["pf"], 0)
        win_score = pnl_safe * min(pf_safe, 3.0)
        per_win_scores.append(win_score)

        trial.set_user_attr(f"{label}_n", r["n"])
        trial.set_user_attr(f"{label}_pnl", round(pnl_safe, 2))
        trial.set_user_attr(f"{label}_pf", round(pf_safe, 3))
        trial.set_user_attr(f"{label}_wr", round(_safe(r["wr"]), 1))
        trial.set_user_attr(f"{label}_equity", round(_safe(r["equity"]), 2))
        trial.set_user_attr(f"{label}_score", round(_safe(win_score), 2))

    # Diagnostics
    trial.set_user_attr("train_n", train_result["n"])
    trial.set_user_attr("train_pnl", round(_safe(train_result["total_pnl"]), 2))
    trial.set_user_attr("train_pf", round(_safe(train_result["pf"]), 3))
    trial.set_user_attr("train_wr", round(_safe(train_result["wr"]), 1))
    trial.set_user_attr("train_equity", round(_safe(train_result["equity"]), 2))

    enabled_strats = [s for s in STRAT_KEYS if params.get(f"enable_{s.lower()}", True)]
    prio = {s: params[f"priority_{s.lower()}"] for s in enabled_strats}
    prio_str = ">".join(s for s, _ in sorted(prio.items(), key=lambda x: x[1]))
    trial.set_user_attr("priority", prio_str)
    trial.set_user_attr("enabled", ",".join(enabled_strats))

    # min across windows is the objective: a single bad window tanks the trial
    min_score = min(per_win_scores)
    trial.set_user_attr("min_val_score", round(_safe(min_score), 2))
    trial.set_user_attr("n", sum(_safe(0) for _ in per_win_scores))  # not meaningful but expected by dump
    trial.set_user_attr("total_pnl", round(_safe(min_score), 2))
    trial.set_user_attr("pf", round(_safe(train_result["pf"]), 3))
    trial.set_user_attr("wr", round(_safe(train_result["wr"]), 1))
    trial.set_user_attr("equity", 0)
    return _safe(min_score, -9999)


def objective_val(trial, daily_picks, train_dates, val_dates):
    """Validation-aware objective.

    Runs training backtest (just for activity gate + diagnostics), then a
    SEPARATE backtest starting fresh from $25K on the validation window.
    The objective Optuna sees is the VALIDATION score (val_pnl * min(val_pf, 3)).

    This naturally selects against training-period overfit: a trial that
    compounds aggressively on training but tanks on val gets a low score.
    The true blind OOS window (everything after val-end) stays untouched
    during optimization — used only for final forward-test verification.
    """
    assert tgc.USE_DYNAMIC_SLIPPAGE and tgc.USE_MULTIWINDOW_SLIPPAGE, \
        "Slippage parity violation: both USE_DYNAMIC_SLIPPAGE and USE_MULTIWINDOW_SLIPPAGE must be True"
    params = suggest_all_params(trial)
    with _param_lock:
        set_strategy_params(params)
        if ENABLE_MULTIWINDOW_SLIPPAGE:
            tgc.USE_DYNAMIC_SLIPPAGE = True
            tgc.USE_MULTIWINDOW_SLIPPAGE = True
        elif ENABLE_2MIN_SLIPPAGE:
            tgc.USE_DYNAMIC_SLIPPAGE = True
            tgc.USE_2MIN_SLIPPAGE = True
        snapshot = _build_param_snapshot()

    def _safe(x, default=-9.9e12):
        import math
        try: x = float(x)
        except (TypeError, ValueError): return default
        if math.isnan(x) or math.isinf(x): return default
        return max(-9.9e12, min(9.9e12, x))

    train_result = run_combined_backtest(daily_picks, train_dates, params_snapshot=snapshot)
    if train_result["n"] < 100:
        return -9999
    if train_result["pf"] < 0.5:
        return -9999

    val_result = run_combined_backtest(daily_picks, val_dates, params_snapshot=snapshot)
    if val_result["n"] < 5:
        return -9999

    # Persist both train and val metrics
    for prefix, r in (("train_", train_result), ("val_", val_result)):
        trial.set_user_attr(f"{prefix}n", r["n"])
        trial.set_user_attr(f"{prefix}pnl", round(_safe(r["total_pnl"]), 2))
        trial.set_user_attr(f"{prefix}pf", round(_safe(r["pf"]), 3))
        trial.set_user_attr(f"{prefix}wr", round(_safe(r["wr"]), 1))
        trial.set_user_attr(f"{prefix}equity", round(_safe(r["equity"]), 2))
        trial.set_user_attr(f"{prefix}sharpe", round(_safe(r.get("sharpe_pct", 0)), 3))

    # Aliases so dump_best_params still works
    trial.set_user_attr("total_pnl", round(_safe(val_result["total_pnl"]), 2))
    trial.set_user_attr("pf", round(_safe(val_result["pf"]), 3))
    trial.set_user_attr("wr", round(_safe(val_result["wr"]), 1))
    trial.set_user_attr("n", val_result["n"])
    trial.set_user_attr("equity", round(_safe(val_result["equity"]), 2))

    enabled_strats = [s for s in STRAT_KEYS if params.get(f"enable_{s.lower()}", True)]
    prio = {s: params[f"priority_{s.lower()}"] for s in enabled_strats}
    prio_str = ">".join(s for s, _ in sorted(prio.items(), key=lambda x: x[1]))
    trial.set_user_attr("priority", prio_str)
    trial.set_user_attr("enabled", ",".join(enabled_strats))

    val_pnl_safe = _safe(val_result["total_pnl"], -1e12)
    val_pf_safe = _safe(val_result["pf"], 0)
    score = val_pnl_safe * min(val_pf_safe, 3.0)
    return _safe(score, -9999)


def objective(trial, daily_picks, all_dates):
    """Objective function: suggest params -> run full backtest -> score.

    The set_strategy_params() + snapshot construction runs under a lock so
    that with n_jobs > 1, threads don't corrupt each other's tgc globals
    during the brief mutation window. Once the snapshot dict is built, it's
    passed explicitly to the simulator — no shared state during the long
    backtest phase. Optuna's parallelism is thus correct AND fully utilized.
    """
    assert tgc.USE_DYNAMIC_SLIPPAGE and tgc.USE_MULTIWINDOW_SLIPPAGE, \
        "Slippage parity violation: both USE_DYNAMIC_SLIPPAGE and USE_MULTIWINDOW_SLIPPAGE must be True"
    params = suggest_all_params(trial)
    with _param_lock:
        set_strategy_params(params)
        if ENABLE_MULTIWINDOW_SLIPPAGE:
            tgc.USE_DYNAMIC_SLIPPAGE = True
            tgc.USE_MULTIWINDOW_SLIPPAGE = True
        elif ENABLE_2MIN_SLIPPAGE:
            tgc.USE_DYNAMIC_SLIPPAGE = True
            tgc.USE_2MIN_SLIPPAGE = True
        snapshot = _build_param_snapshot()

    # ----------------------------------------------------------------------
    # CV PATH — per-year-reset cross-validation.
    # Default: score = min(year_pnl * min(year_pf, 3.0)) across all years.
    #
    # Variant selection via env var W9_VARIANT:
    #   (unset)             -> default: min(year_pnl * min(year_pf, 3))
    #                          punishes WORST year. May find boring strategies.
    #   activity_weighted   -> default * (geomean(year_n) / 100)
    #                          rewards trials that actually fire across all years.
    #                          Fix when default produces avoidance-overfit (boring).
    #   hybrid_sum_minpf2   -> sum_pnl * min_pf^2
    #                          keeps total-PnL signal; weights consistency squared.
    #                          Fix when default rejects too many active trials.
    # ----------------------------------------------------------------------
    if USE_CV_OBJECTIVE:
        per_year = run_combined_backtest_cv(daily_picks, all_dates,
                                             params_snapshot=snapshot)
        years = sorted(per_year.keys())
        if not years:
            return -9999

        def _safe(x, default=-9.9e12):
            import math
            try: x = float(x)
            except (TypeError, ValueError): return default
            if math.isnan(x) or math.isinf(x): return default
            return max(-9.9e12, min(9.9e12, x))

        # Per-year minimum activity: trial must trade meaningfully in every
        # year, else it's overfit to a single regime by avoidance.
        import os as _os
        variant = _os.environ.get("W9_VARIANT", "default")
        # Tighter floor under hybrid (which doesn't directly reward activity)
        MIN_TRADES_PER_YEAR = 15 if variant == "hybrid_sum_minpf2" else 20
        if any(per_year[y]["n"] < MIN_TRADES_PER_YEAR for y in years):
            return -9999

        pnls = [per_year[y]["pnl"] for y in years]
        pfs = [per_year[y]["pf"] for y in years]
        ns = [per_year[y]["n"] for y in years]

        # Compute base score = min(year_pnl × min(year_pf, 3.0))
        per_year_scores = [
            per_year[y]["pnl"] * min(per_year[y]["pf"], 3.0)
            for y in years
        ]
        base_score = min(per_year_scores)

        if variant == "activity_weighted":
            # Multiply by geomean(year_n)/100 — rewards trade activity.
            # /100 keeps scale sane: 100 trades/year is the reference.
            import numpy as _np
            geo_n = float(_np.prod([max(n, 1) for n in ns]) ** (1.0 / len(ns)))
            activity_mult = geo_n / 100.0
            score = _safe(base_score * activity_mult, -9999)
        elif variant == "geomean_year_score":
            # ChatGPT-suggested: geomean of yearly scores instead of min.
            # Pure min() ignores strong years -> TPE just barely-clears the floor.
            # Geomean still punishes weak years (log-scale) but rewards big winners,
            # so a strategy with [10k, 4k, 500k, 600k] beats [5k, 5k, 5k, 5k].
            # Any year with <=0 score -> hard penalty (preserves positivity gate).
            import numpy as _np
            yearly_scores = per_year_scores  # list of pnl_y * min(pf_y, 3)
            if any(s <= 0 for s in yearly_scores):
                # Negative or zero year -> can't take geomean; preserve sign by
                # falling back to min penalty (-large but proportional to worst)
                score = _safe(min(yearly_scores), -9999)
            else:
                geomean = float(_np.prod(yearly_scores) ** (1.0 / len(yearly_scores)))
                score = _safe(geomean, -9999)
        elif variant == "hybrid_sum_minpf2":
            # sum_pnl × min_pf² — keeps total PnL signal, weights consistency hard.
            min_pf = min(pfs)
            sum_pnl = sum(pnls)
            score = _safe(sum_pnl * (min_pf ** 2), -9999)
        else:
            score = _safe(base_score, -9999)

        # Persist per-year breakdown as user_attrs for inspection
        for y in years:
            r = per_year[y]
            trial.set_user_attr(f"pnl_{y}", round(_safe(r["pnl"], 0), 0))
            trial.set_user_attr(f"pf_{y}",  round(_safe(r["pf"], 0), 3))
            trial.set_user_attr(f"n_{y}",   r["n"])
        trial.set_user_attr("min_pnl",        round(_safe(min(pnls), 0), 0))
        trial.set_user_attr("min_pf",         round(_safe(min(pfs), 0), 3))
        trial.set_user_attr("min_year_score", round(_safe(base_score, 0), 0))
        trial.set_user_attr("sum_pnl",        round(_safe(sum(pnls), 0), 0))
        trial.set_user_attr("mean_pnl",       round(_safe(sum(pnls)/len(pnls), 0), 0))
        trial.set_user_attr("total_pnl",      round(_safe(sum(pnls), 0), 0))
        trial.set_user_attr("n",              sum(ns))
        trial.set_user_attr("objective_variant", variant)
        return score

    result = run_combined_backtest(daily_picks, all_dates, params_snapshot=snapshot)

    n = result["n"]
    # Min trade count: prevents Optuna from finding "got lucky on 5 trades" configs.
    # Bumped from 30 to 100 after shape-filtered runs showed thin-trade winners.
    if n < 100:
        return -9999

    pf = result["pf"]
    if pf < 0.5:
        return -9999

    total_pnl = result["total_pnl"]
    sharpe_pct = result.get("sharpe_pct", 0.0)

    # Sanitize numeric values for SQLite/JSON storage.
    # Under dynamic slippage, bad-params trials can produce NaN/Inf equities
    # (e.g. negative bankroll after compounded extreme slippage). Optuna's
    # rdb storage rejects those, killing the whole study. Clamp to a finite
    # large-magnitude sentinel instead.
    def _safe(x, default=-9.9e12):
        import math
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if math.isnan(x) or math.isinf(x):
            return default
        # SQLite's max is ~1e308 but JSON serialization is happier with smaller
        return max(-9.9e12, min(9.9e12, x))

    pf_safe = _safe(pf, 0)
    total_pnl_safe = _safe(total_pnl, -1e12)
    sharpe_safe = _safe(sharpe_pct, 0)

    trial.set_user_attr("n", n)
    trial.set_user_attr("pf", round(pf_safe, 3))
    trial.set_user_attr("wr", round(_safe(result["wr"], 0), 1))
    trial.set_user_attr("total_pnl", round(total_pnl_safe, 2))
    trial.set_user_attr("equity", round(_safe(result["equity"], -1e12), 2))
    trial.set_user_attr("sharpe_pct", round(sharpe_safe, 3))

    enabled_strats = [s for s in STRAT_KEYS if params.get(f"enable_{s.lower()}", True)]
    prio = {s: params[f"priority_{s.lower()}"] for s in enabled_strats}
    prio_str = ">".join(s for s, _ in sorted(prio.items(), key=lambda x: x[1]))
    trial.set_user_attr("priority", prio_str)
    trial.set_user_attr("enabled", ",".join(enabled_strats))
    trial.set_user_attr("n_strategies", len(enabled_strats))

    for s, v in result["strats"].items():
        trial.set_user_attr(f"{s}_n", v["n"])
        trial.set_user_attr(f"{s}_pnl", round(_safe(v["pnl"], 0), 2))
        wr_s = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
        trial.set_user_attr(f"{s}_wr", round(_safe(wr_s, 0), 1))

    if USE_SHARPE_OBJECTIVE:
        # Sharpe on % returns, capped at 4.0. Rewards consistent compounders;
        # bad-Sharpe trials still get scored (just lower) so TPE can map terrain.
        score = total_pnl_safe * min(max(sharpe_safe, 0.0), 4.0)
    else:
        score = total_pnl_safe * min(pf_safe, 3.0)
    return _safe(score, -9999)


# ---------------------------------------------------------------------------
# Dump best params to JSON (for live backtest runs)
# ---------------------------------------------------------------------------
BEST_PARAMS_FILE = "results/params/optuna_best_params_v8.json"

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

    def _compute_pearsons(study):
        """Compute train<->val + val<->val Pearsons across completed trials.

        Returns dict with keys: 'train_val', 'val1_val2', 'val1_val3', 'val2_val3'
        Each value is float or None (if pair not available).

        train<->val: weak overfit detector — catches training-memorization but NOT
        regime-shift overfit (W16 had +0.86 here yet failed Mar-Jun blind OOS).
        val<->val (multi-window only): REAL regime-consistency detector. If trials'
        rankings disagree across val regimes, no robust basin exists. The min()
        objective punishes regime-divergent trials but won't manufacture a
        non-existent robust solution.
        """
        train_vals = []; val_main = []
        v1=[]; v2=[]; v3=[]
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE: continue
            ua = t.user_attrs
            tr = ua.get("train_pnl")
            vl = ua.get("val_pnl") or ua.get("val1_pnl")
            try:
                if tr is not None and vl is not None:
                    trf = float(tr); vlf = float(vl)
                    if trf == trf and vlf == vlf:
                        train_vals.append(trf); val_main.append(vlf)
                if ua.get("val1_pnl") is not None: v1.append(float(ua["val1_pnl"]))
                if ua.get("val2_pnl") is not None: v2.append(float(ua["val2_pnl"]))
                if ua.get("val3_pnl") is not None: v3.append(float(ua["val3_pnl"]))
            except (ValueError, TypeError): continue
        import numpy as _np
        def _p(a, b):
            n = min(len(a), len(b))
            if n < 10: return None
            try:
                r = float(_np.corrcoef(a[:n], b[:n])[0,1])
                return r if r == r else None
            except Exception: return None
        return {
            "train_val": _p(train_vals, val_main),
            "val1_val2": _p(v1, v2), "val1_val3": _p(v1, v3), "val2_val3": _p(v2, v3),
            "n_trials": len(train_vals),
        }

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
            pearson_str = ""
            if n_complete % 50 == 0:
                pearsons = _compute_pearsons(study)
                parts = []
                if pearsons["train_val"] is not None:
                    parts.append(f"tv={pearsons['train_val']:+.2f}")
                # For multi-window: weakest val<->val pair is the regime-overfit metric
                vv_pairs = [(k, v) for k, v in pearsons.items() if k.startswith("val") and v is not None]
                if vv_pairs:
                    weakest = min(vv_pairs, key=lambda x: x[1])
                    parts.append(f"weak({weakest[0]})={weakest[1]:+.2f}")
                if parts:
                    pearson_str = f" | Pearson[{' '.join(parts)}]"
                study.set_user_attr("rolling_pearsons", {k: round(v, 4) if v is not None else None for k, v in pearsons.items()})
                study.set_user_attr("rolling_pearson_at_trial", n_complete)
            print(f"  Trial {n_complete} | {elapsed/60:.1f}m elapsed | {rate:.2f} trials/s | "
                  f"Best: {best_score[0]:,.0f}{pearson_str}")
    return callback


# ===========================================================================
# G+L Trail study — self-contained functions (--study-type gl_trail)
# ===========================================================================

def configure_gl_trail_simulator():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False


def configure_gl_trail_for_optimization():
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    tgc.NEWS_FILTER_ENABLED = False


def run_gl_trail_backtest(dates, picks_by_date, snapshot):
    cash = float(25_000)
    all_trades = []

    for d in dates:
        picks = picks_by_date.get(d, [])
        if not picks:
            continue
        cash_account = cash < MARGIN_THRESHOLD
        try:
            states, cash, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account, params=snapshot
            )
        except Exception:
            continue
        effective_cash = cash + (unsettled if cash_account else 0)
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0)
                if pnl is not None:
                    all_trades.append({
                        "strategy": st.get("strategy"),
                        "pnl": pnl,
                        "position_cost": st.get("position_cost", 0),
                        "exit_reason": st.get("exit_reason"),
                    })
        cash = effective_cash

    n = len(all_trades)
    if n == 0:
        return {"n": 0, "total_pnl": -9999, "pf": 0.0, "equity": cash,
                "gross_win": 0, "gross_loss": 1e-9, "wr": 0.0}

    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in all_trades if t["pnl"] <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss
    wr = len(wins) / n * 100
    return {"n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr,
            "equity": cash, "gross_win": gross_win, "gross_loss": gross_loss}


def suggest_gl_trail_params(trial, base_params):
    """Suggest G+L params (30 suggest calls). Returns merged param dict."""
    p = dict(base_params)

    # ---- G strategy params ----
    p["g_min_gap_pct"] = trial.suggest_float("g_min_gap_pct", 5.0, 50.0, step=5.0)
    p["g_require_2nd_green"] = trial.suggest_categorical("g_require_2nd_green", [True, False])
    p["g_require_2nd_new_high"] = trial.suggest_categorical("g_require_2nd_new_high", [True, False])
    p["g_stop_pct"] = trial.suggest_float("g_stop_pct", 5.0, 30.0, step=1.0)
    p["g_time_limit_min"] = trial.suggest_int("g_time_limit_min", 6, 60, step=3)

    g_partial = trial.suggest_float("g_partial_sell_pct", 0.0, 75.0, step=25.0)
    p["g_partial_sell_pct"] = g_partial
    p["g_target_pct"] = trial.suggest_float("g_target_pct", 5.0, 40.0, step=5.0)

    # Trail params always suggested unconditionally — avoids train/eval mismatch
    # where "no trail" branch would set 0.0 in training but a different base value
    # could leak in during forward test (bug in v2 study).
    p["g_trail_activate_pct"] = trial.suggest_float("g_trail_activate_pct", 0.0, 40.0, step=5.0)
    p["g_trail_pct"] = trial.suggest_float("g_trail_pct", 0.0, 20.0, step=0.5)

    if g_partial > 0:
        p["g_target2_pct"] = trial.suggest_float("g_target2_pct", 20.0, 100.0, step=10.0)
    else:
        p["g_target2_pct"] = 999.0

    # ---- L strategy params ----
    p["l_earliest_candle"] = trial.suggest_int("l_earliest_candle", 3, 30, step=3)
    p["l_latest_candle"] = trial.suggest_int("l_latest_candle", 30, 180, step=15)
    p["l_max_float"] = trial.suggest_int("l_max_float", 5_000_000, 25_000_000, step=5_000_000)
    p["l_min_gap"] = trial.suggest_int("l_min_gap", 10, 80, step=5)
    p["l_min_price_accel_pct"] = trial.suggest_float("l_min_price_accel_pct", 0.5, 3.0, step=0.5)
    p["l_partial_sell_pct"] = trial.suggest_float("l_partial_sell_pct", 0.0, 75.0, step=25.0)
    p["l_stop_pct"] = trial.suggest_float("l_stop_pct", 10.0, 30.0, step=1.0)
    p["l_tier1_target1_pct"] = trial.suggest_float("l_tier1_target1_pct", 15.0, 50.0, step=5.0)
    p["l_tier1_target2_pct"] = trial.suggest_float("l_tier1_target2_pct", 20.0, 80.0, step=5.0)
    p["l_tier2_target1_pct"] = trial.suggest_float("l_tier2_target1_pct", 10.0, 30.0, step=2.0)
    p["l_tier2_target2_pct"] = trial.suggest_float("l_tier2_target2_pct", 20.0, 60.0, step=5.0)
    p["l_tier3_target1_pct"] = trial.suggest_float("l_tier3_target1_pct", 5.0, 25.0, step=2.0)
    p["l_tier3_target2_pct"] = trial.suggest_float("l_tier3_target2_pct", 10.0, 50.0, step=5.0)

    # Trail params always suggested unconditionally — same fix as G.
    # l_trail_activate_pct=9999 effectively disables the trail (never activates).
    p["l_trail_activate_pct"] = trial.suggest_float("l_trail_activate_pct", 1.0, 200.0, step=1.0)
    p["l_trail_pct"] = trial.suggest_float("l_trail_pct", 1.0, 15.0, step=1.0)

    # Force trailing off for all other strategies
    for prefix in "vhafdrwobkcsexijn":
        p[f"{prefix}_trail_pct"] = 0.0
        p[f"{prefix}_trail_activate_pct"] = 0.0

    # Only G and L enabled; disable everything else
    for s in "vhafdrwobkcsexijn":
        p[f"enable_{s}"] = False
    p["enable_g"] = True
    p["enable_l"] = True

    return p


def objective_gl_trail(trial, data):
    """G+L trail objective."""
    assert tgc.USE_DYNAMIC_SLIPPAGE and tgc.USE_MULTIWINDOW_SLIPPAGE, \
        "Slippage parity violation: both USE_DYNAMIC_SLIPPAGE and USE_MULTIWINDOW_SLIPPAGE must be True"

    dates = data["dates"]
    picks_by_date = data["picks_by_date"]

    p = suggest_gl_trail_params(trial, data["base_params"])

    with _param_lock:
        set_strategy_params(p)
        configure_gl_trail_for_optimization()
        snapshot = _build_param_snapshot()

    result = run_gl_trail_backtest(dates, picks_by_date, snapshot)

    if result["n"] < 100 or result["total_pnl"] <= 0 or result["pf"] < 0.5:
        return -9999

    import math as _math
    score = result["total_pnl"] * min(result["pf"], 3.0)
    if _math.isnan(score) or _math.isinf(score):
        return -9999
    score = max(-9.9e12, min(9.9e12, float(score)))

    trial.set_user_attr("total_pnl", round(result["total_pnl"], 2))
    trial.set_user_attr("pf", round(result["pf"], 3))
    trial.set_user_attr("wr", round(result["wr"], 1))
    trial.set_user_attr("n", result["n"])
    trial.set_user_attr("equity", round(result["equity"], 0))

    return score


def worker_loop_gl_trail(data, args):
    storage = optuna.storages.RDBStorage(
        url=args.db,
        engine_kwargs={"pool_size": 2, "max_overflow": 1, "pool_pre_ping": True, "pool_recycle": 300},
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=args.startup_trials),
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    print(f"\nStudy state: {completed} complete, {running} running, {failed} failed "
          f"(target: {args.trials})", flush=True)

    if completed >= args.trials:
        print(f"Target {args.trials} trials already complete — nothing to do.", flush=True)
        return

    t_start = time.time()
    my_count = 0
    last_report = 0

    while True:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if completed >= args.trials:
            break

        try:
            trial = study.ask()
        except Exception as e:
            print(f"  ask() error: {e} — retrying in 5s", flush=True)
            time.sleep(5)
            continue

        try:
            value = objective_gl_trail(trial, data)
            study.tell(trial, value)
            my_count += 1
        except KeyboardInterrupt:
            try:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            except Exception:
                pass
            print(f"\n[worker] KeyboardInterrupt after {my_count} trials", flush=True)
            break
        except Exception as e:
            import traceback as _tb
            print(f"  [worker] trial exception: {e}", flush=True)
            _tb.print_exc()
            try:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            except Exception:
                pass
            continue

        if my_count % 20 == 0 or my_count == last_report + 1:
            completed_now = len([t for t in study.trials
                                 if t.state == optuna.trial.TrialState.COMPLETE])
            elapsed = time.time() - t_start
            try:
                bt = study.best_trial
                print(f"  [{completed_now}/{args.trials}] {elapsed/60:.1f}m | "
                      f"best ${bt.value:,.0f} (#{bt.number}) | "
                      f"PnL=${bt.user_attrs.get('total_pnl',0):,.0f} "
                      f"PF={bt.user_attrs.get('pf',0):.2f} "
                      f"WR={bt.user_attrs.get('wr',0):.1f}%",
                      flush=True)
            except Exception:
                print(f"  [{completed_now}/{args.trials}] {elapsed/60:.1f}m | "
                      f"no completed trials yet", flush=True)
            last_report = my_count

    total_elapsed = time.time() - t_start
    print(f"\n[worker] Done: {my_count} trials in {total_elapsed/60:.1f}m", flush=True)


def dump_best_gl_trail(study, args):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials.", flush=True)
        return

    best = study.best_trial
    ua = best.user_attrs

    print(f"\n{'='*60}", flush=True)
    print(f"BEST TRIAL #{best.number} (G+L TRAIL SEARCH)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Score:        ${best.value:,.0f}", flush=True)
    print(f"  PnL:          ${ua.get('total_pnl',0):,.0f}", flush=True)
    print(f"  PF:           {ua.get('pf',0):.3f}", flush=True)
    print(f"  WR:           {ua.get('wr',0):.1f}%", flush=True)
    print(f"  n:            {ua.get('n')}", flush=True)
    print(f"  Equity:       ${ua.get('equity',0):,.0f}", flush=True)
    print(f"\n  Params:", flush=True)
    for k, v in sorted(best.params.items()):
        print(f"    {k}: {v}", flush=True)

    if args.params_out:
        out = {
            "label": f"G+L trail search (optuna #{best.number})",
            "study": args.study,
            "trial": best.number,
            "score": best.value,
            "total_pnl": ua.get("total_pnl"),
            "pf": ua.get("pf"),
            "wr": ua.get("wr"),
            "n": ua.get("n"),
            "params": dict(best.params),
        }
        with open(args.params_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved to {args.params_out}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global BEST_PARAMS_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--dump-best", action="store_true",
                        help="Extract best params from existing DB and exit (no new trials)")
    parser.add_argument("--slippage", type=float, default=None,
                        help="Override tgc.SLIPPAGE_PCT (e.g. 1.0 = 1%% per leg, legacy constant)")
    parser.add_argument("--vol-cap", type=float, default=None,
                        help="Override tgc.VOL_CAP_PCT (e.g. 1.0 = 1%% of cumulative dollar vol)")
    parser.add_argument("--dynamic-slip", action="store_true",
                        help="Enable liquidity-aware slippage (base + sqrt-participation impact)")
    parser.add_argument("--slip-impact-k", type=float, default=None,
                        help="Override SLIP_IMPACT_K (default 3.0)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Threads per process (default: cpu_count//2). Use 1 when "
                             "running multiple processes via FORCE_ENABLE_STRATS pattern.")
    parser.add_argument("--startup-trials", type=int, default=200,
                        help="TPESampler n_startup_trials (default: 200; use 4x dimensions)")
    parser.add_argument("--date-start", default=None,
                        help="Override DATE_RANGE start (e.g., 2023-03-01 for 3yr recent window)")
    parser.add_argument("--date-end", default=None,
                        help="Override DATE_RANGE end (e.g., 2025-12-31 to hold 2026 OOS)")
    parser.add_argument("--val-start", default=None,
                        help="Validation window start (e.g., 2026-01-01). When set, "
                             "objective becomes val_pnl*min(val_pf,3). Train period = "
                             "[date-start..date-end] excluding [val-start..val-end].")
    parser.add_argument("--val-end", default=None,
                        help="Validation window end (e.g., 2026-02-28)")
    parser.add_argument("--val-windows", default=None,
                        help="Multi-window walk-forward val. Comma-separated "
                             "'start:end,start:end,...' pairs. Objective = "
                             "min(per-window score). Each val window evaluated fresh "
                             "from $25K. Train = date-range MINUS any val window day.")
    parser.add_argument("--use-sortino", action="store_true",
                        help="Use Sortino-weighted objective (penalizes downside var only) "
                             "instead of PF-weighted. Recommended with multi-window val.")
    parser.add_argument("--g-stop-min", type=float, default=None,
                        help="Minimum g_stop_pct in search space (e.g. 5). Kills the "
                             "no-stop overfit attractor (W16 trap).")
    parser.add_argument("--no-2g-tunable", action="store_true",
                        help="Hardcode g_require_2nd_green=True instead of tuning it")
    parser.add_argument("--l-min-gap-min", type=int, default=None,
                        help="Override lower bound of l_min_gap search (default 15). "
                             "Use 45 to skip the overfit l_gap=30 basin.")
    parser.add_argument("--db", default="optuna_combined_v8.db",
                        help="SQLite path for the study (default: optuna_combined_v8.db)")
    parser.add_argument("--study", default="combined_v8_20strats_2024_2026",
                        help="Optuna study name (default: combined_v8_20strats_2024_2026)")
    parser.add_argument("--params-out", default=None,
                        help="Best-trial JSON output path (default: optuna_best_params_v8.json)")
    parser.add_argument("--regime-filter", default=None,
                        choices=["squeeze", "normal", "dead"],
                        help="Filter daily picks to ONLY days where the regime gate "
                             "classifies as the given regime. Used to train regime "
                             "specialists. Requires strategies/regime_gate.py.")
    parser.add_argument("--study-type", default="combined",
                        choices=["combined", "gl_trail"],
                        help="Study type: 'combined' (default, 20-strategy optimizer) or "
                             "'gl_trail' (G+L trail specialist with PostgreSQL backend).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for gl_trail mode (default: 1).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # GL_TRAIL dispatch — independent entrypoint, exits when done
    # ------------------------------------------------------------------
    if args.study_type == "gl_trail":
        configure_gl_trail_simulator()

        # Default db to PostgreSQL for gl_trail unless overridden
        gl_db = args.db if args.db != "optuna_combined_v8.db" \
            else "postgresql://postgres@127.0.0.1:5432/optuna_gl_trail"
        gl_study = args.study if args.study != "combined_v8_20strats_2024_2026" \
            else "gl_trail"
        gl_params_out = args.params_out or "config/trial_gl_trail_best.json"

        # Reconstruct a simple namespace for the worker/dump helpers
        import types as _types
        gl_args = _types.SimpleNamespace(
            db=gl_db,
            study=gl_study,
            trials=args.trials,
            startup_trials=args.startup_trials,
            params_out=gl_params_out,
            workers=args.workers,
        )

        print("Loading G+L trail training data...", flush=True)
        dirs = [d for d in GL_TRAIL_DATA_DIRS if os.path.exists(d)]
        all_dates_gl, picks_by_date_gl = load_all_picks(dirs)
        dates_gl = sorted([d for d in all_dates_gl
                           if GL_TRAIL_DATE_LO <= d <= GL_TRAIL_DATE_HI])
        print(f"  Training window: {GL_TRAIL_DATE_LO} to {GL_TRAIL_DATE_HI} "
              f"({len(dates_gl)} days)", flush=True)
        if len(dates_gl) == 0:
            print("ERROR: no trading days in range!", flush=True)
            sys.exit(1)

        base_path = "config/trial_w21b_511_deploy.json"
        with open(base_path) as _f:
            _base_data = json.load(_f)
        base_params_gl = dict(_base_data.get("params", {}))
        print(f"  Base config: {base_path} ({len(base_params_gl)} params)", flush=True)

        gl_data = {"dates": dates_gl, "picks_by_date": picks_by_date_gl,
                   "base_params": base_params_gl}

        if gl_args.workers > 1:
            # Launch each worker as an independent subprocess with --workers 1.
            # This avoids Windows multiprocessing spawn issues where child processes
            # re-enter main() and cascade into data loading + more spawns.
            import subprocess as _sp
            _worker_cmd = [
                sys.executable, __file__,
                "--study-type", "gl_trail",
                "--db", gl_args.db,
                "--study", gl_args.study,
                "--trials", str(gl_args.trials),
                "--startup-trials", str(gl_args.startup_trials),
                "--workers", "1",
                "--params-out", gl_args.params_out or "config/trial_gl_trail_best.json",
            ]
            procs = [_sp.Popen(_worker_cmd) for _ in range(gl_args.workers)]
            for _p in procs:
                _p.wait()
        else:
            worker_loop_gl_trail(gl_data, gl_args)

        _gl_storage = optuna.storages.RDBStorage(url=gl_args.db)
        _gl_study = optuna.load_study(study_name=gl_args.study, storage=_gl_storage)
        dump_best_gl_trail(_gl_study, gl_args)
        print("\nG+L Trail Search complete.", flush=True)
        return

    # ------------------------------------------------------------------
    # COMBINED mode (legacy default)
    # ------------------------------------------------------------------
    n_trials = args.trials

    db_path = args.db
    study_name = args.study
    if args.params_out:
        BEST_PARAMS_FILE = args.params_out

    # Apply slippage / vol-cap overrides to the simulator BEFORE any trial runs.
    # These rebind the module globals tgc imported from test_full; simulate_day_combined
    # looks them up by name on each call, so the override takes effect.
    if args.slippage is not None:
        tgc.SLIPPAGE_PCT = float(args.slippage)
    if args.vol_cap is not None:
        tgc.VOL_CAP_PCT = float(args.vol_cap)
    if args.dynamic_slip:
        tgc.USE_DYNAMIC_SLIPPAGE = True
    if args.slip_impact_k is not None:
        tgc.SLIP_IMPACT_K = float(args.slip_impact_k)

    # --dump-best: extract best params from existing DB without running trials
    if args.dump_best:
        if not os.path.exists(db_path):
            print(f"ERROR: {db_path} not found. Run optimizer first.")
            sys.exit(1)
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
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
    print(f"  Candidates:  {', '.join(STRAT_KEYS)}")
    print(f"  Optuna decides: which strategies to enable (1-20) + priority")
    print(f"  Single pool: ${STARTING_CASH:,}")
    print(f"  Trials:      {n_trials}")
    print(f"  Objective:   total_pnl * min(pf, 3.0)")
    print(f"  Data:        {DATA_DIRS}")
    if getattr(tgc, "USE_DYNAMIC_SLIPPAGE", False):
        print(f"  Slippage:    DYNAMIC (base={tgc.SLIP_BASE_SPREAD}+{tgc.SLIP_PRICE_COEFF}/price + K={tgc.SLIP_IMPACT_K}*sqrt(participation))")
    else:
        print(f"  Slippage:    {tgc.SLIPPAGE_PCT}% per leg (legacy constant)")
    print(f"  Vol cap:     {tgc.VOL_CAP_PCT}% of cumulative dollar vol")
    print(f"  DB:          {db_path}")
    print(f"  Study:       {study_name}")
    print(f"  Best JSON:   {BEST_PARAMS_FILE}")
    print("=" * 70)

    print("\nLoading data...")
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    _date_start = args.date_start if args.date_start else DATE_RANGE[0]
    _date_end = args.date_end if args.date_end else DATE_RANGE[1]
    all_dates = [d for d in all_dates if _date_start <= d <= _date_end]
    print(f"  Training date range: {_date_start} to {_date_end} ({len(all_dates)} days)")
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")

    # Optional regime filter — keeps only days where today's pre-market scan
    # matches the requested regime. Used to train per-regime specialists.
    if args.regime_filter:
        from strategies.regime_gate import classify_regime
        filtered = [d for d in all_dates
                    if classify_regime(daily_picks.get(d, [])) == args.regime_filter]
        print(f"  Regime filter '{args.regime_filter}': {len(filtered)} days "
              f"kept ({100*len(filtered)/max(1,len(all_dates)):.1f}%)")
        all_dates = filtered
        daily_picks = {d: daily_picks[d] for d in all_dates}
    # Postgres support: if --db starts with postgresql://, skip SQLite bootstrap.
    is_postgres = db_path.startswith("postgresql://") or db_path.startswith("postgres://")
    if is_postgres:
        rdb = optuna.storages.RDBStorage(url=db_path)
    else:
        # Pre-set SQLite WAL mode ONCE before Optuna's connection pool spawns
        # parallel threads. WAL is a per-database setting (not per-connection), so
        # setting it once is enough; subsequent connections inherit it.
        import sqlite3
        _bootstrap_conn = sqlite3.connect(db_path, timeout=60)
        try:
            _bootstrap_conn.execute("PRAGMA journal_mode=WAL")
            _bootstrap_conn.execute("PRAGMA synchronous=NORMAL")
            _bootstrap_conn.commit()
        finally:
            _bootstrap_conn.close()

        from sqlalchemy import event
        rdb = optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path}",
            engine_kwargs={"connect_args": {"timeout": 120}},
        )

        @event.listens_for(rdb.engine, "connect")
        def _set_busy_timeout(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA busy_timeout=120000")
            cur.close()

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=rdb,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=args.startup_trials),
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"\n  Resuming: {n_existing} existing trials found")
        # Only report current best if at least one trial actually COMPLETED.
        # On a fresh-but-stale DB (all RUNNING/FAIL/PRUNED) study.best_trial
        # raises "Record does not exist."
        try:
            best = study.best_trial
            print(f"  Current best: score={best.value:,.0f}, "
                  f"PnL=${best.user_attrs.get('total_pnl', 0):,.0f}, "
                  f"PF={best.user_attrs.get('pf', 0):.2f}")
        except (ValueError, Exception) as e:
            print(f"  (No complete trials yet — starting fresh)")

    print(f"\n  Starting optimization ({n_trials} trials)...\n")
    start_time = time.time()

    n_jobs = args.n_jobs if args.n_jobs is not None else max(1, multiprocessing.cpu_count() // 2)
    print(f"  Using n_jobs={n_jobs} (threads per process)\n")

    # Validation-aware dispatch
    if args.val_windows:
        # Multi-window walk-forward
        win_specs = [w.strip() for w in args.val_windows.split(",") if w.strip()]
        val_windows_list = []
        all_val_days = set()
        for i, spec in enumerate(win_specs):
            s, e = spec.split(":")
            win_days = sorted([d for d in all_dates if s <= d <= e])
            label = f"val{i+1}"
            val_windows_list.append((label, win_days))
            all_val_days.update(win_days)
            print(f"  {label}: {s} to {e} ({len(win_days)} days)")
        train_dates = sorted([d for d in all_dates if d not in all_val_days])
        print(f"  Training window:   {len(train_dates)} days (date-range minus all val days)")
        print(f"  Objective: min(val_score) across {len(val_windows_list)} windows — TPE punishes worst-window overfit\n")
        if not train_dates or not all(w[1] for w in val_windows_list):
            print("ERROR: empty train or val window."); sys.exit(1)
        if args.use_sortino:
            print(f"  Objective variant: Sortino-weighted (penalizes downside variance only)\n")
            objective_fn = lambda trial: objective_val_multi_sortino(trial, daily_picks, train_dates, val_windows_list)
        else:
            objective_fn = lambda trial: objective_val_multi(trial, daily_picks, train_dates, val_windows_list)
    elif args.val_start and args.val_end:
        val_dates = sorted([d for d in all_dates if args.val_start <= d <= args.val_end])
        train_dates = sorted([d for d in all_dates if d < args.val_start or d > args.val_end])
        print(f"  Validation window: {args.val_start} to {args.val_end} ({len(val_dates)} days)")
        print(f"  Training window:   {len(train_dates)} days (date-range minus val window)")
        print(f"  Objective: val_pnl * min(val_pf, 3) — TPE selects for generalization\n")
        if not val_dates or not train_dates:
            print(f"ERROR: empty train or val window."); sys.exit(1)
        objective_fn = lambda trial: objective_val(trial, daily_picks, train_dates, val_dates)
    else:
        objective_fn = lambda trial: objective(trial, daily_picks, all_dates)

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        n_jobs=n_jobs,
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
    print(f"\n  DB saved to:     {db_path}")
    print(f"  Params saved to: {BEST_PARAMS_FILE}")
    print(f"  Study name:      {study_name}")


if __name__ == "__main__":
    main()
