"""
Strategy param registry — single source of truth for all strategy parameters.

Phase 1 of the monolith refactor: the ~300 module-level globals in
test_green_candle_combined.py (config block, lines ~449-800) and the
~235 write-lines in optimize_combined.set_strategy_params() both described
the SAME parameters, and they had drifted (e.g. min-gap defaults were
overwritten by set_strategy_params({})).

This module defines one authoritative table per strategy letter:

    StrategyConfig:
        letter: single-char code (H, G, A, ...)
        min_gap_global: the global that gates enable/disable (set to 9999 to disable)
        priority: default STRAT_PRIORITY entry
        params: list of (global_name, param_key, default, kind)
            kind in {'num','bool','str'} — num uses float(), bool uses bool()

The apply_params() function reproduces set_strategy_params() EXACTLY (same
defaults, same float()/bool() coercions, same enable/disable 9999 gate,
same priority map, same participation caps and favorability weights), but
driven by this table instead of 235 hand-written lines.

Compatibility: apply_params() writes results onto the `target` module as
attributes with the exact historical names (e.g. tgc.G_STOP_PCT), so all
existing readers (simulate_day_combined globals() snapshot, tgc.__dict__
snapshot, dir(tgc), and ~160 external scripts) work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sentinel used to disable a strategy: its MIN_GAP_PCT is set to this value so
# `gap_pct >= MIN_GAP_PCT` is never satisfied by a real pick. Must exceed ANY
# possible gap_pct — reverse-split artifacts have produced gaps > 9999%
# (e.g. PAVS 14020%, OCG 24751%), which leaked past the old 9999.0 sentinel.
DISABLED_GAP = 9e9

# ---------------------------------------------------------------------------
# Param entries: (GLOBAL_NAME, lowercase param key, default, kind)
# kind: 'num' -> float(...)    'bool' -> bool(...)    'str' -> raw
# ---------------------------------------------------------------------------


@dataclass
class StrategyConfig:
    letter: str
    min_gap_global: str  # gate global; set to 9999.0 to disable this strategy
    priority: int  # default priority (lower = higher)
    params: list = field(default_factory=list)  # list of (global, key, default, kind)


STRATEGIES: dict[str, StrategyConfig] = {}


def _reg(letter, min_gap_global, priority, params):
    STRATEGIES[letter] = StrategyConfig(
        letter=letter, min_gap_global=min_gap_global, priority=priority, params=params
    )


# --- H: High Conviction (filtered G) -------------------------------------
# NOTE: H_MIN_GAP_PCT is HARDCODED (35.0) in the original set_strategy_params
# (not read from params) — replicated here as non-tunable.
_reg("H", "H_MIN_GAP_PCT", 0, [
    ("H_MIN_GAP_PCT", "h_min_gap_pct", 35.0, "fixed"),
    ("H_TARGET_PCT", "h_target_pct", 16.0, "num"),
    ("H_TIME_LIMIT_MINUTES", "h_time_limit_min", 15, "num"),
    ("H_STOP_PCT", "h_stop_pct", 0.0, "num"),
    ("H_TRAIL_PCT", "h_trail_pct", 0.0, "num"),
    ("H_TRAIL_ACTIVATE_PCT", "h_trail_activate_pct", 0.0, "num"),
])

# --- G: Big Gap Runner ---------------------------------------------------
_reg("G", "G_MIN_GAP_PCT", 1, [
    ("G_MIN_GAP_PCT", "g_min_gap_pct", 30.0, "num"),
    ("G_REQUIRE_2ND_GREEN", "g_require_2nd_green", True, "bool"),
    ("G_REQUIRE_2ND_NEW_HIGH", "g_require_2nd_new_high", True, "bool"),
    ("G_REQUIRE_3RD_GREEN", "g_require_3rd_green", False, "bool"),
    ("G_REQUIRE_4TH_GREEN", "g_require_4th_green", False, "bool"),
    # Bar-strength gates (G2 experiment): filter weak opens by candle-1/2 size
    # and candle-2 volume, and scale position by conviction.
    ("G_MIN_1ST_BODY_PCT", "g_min_1st_body_pct", 0.0, "num"),
    ("G_MIN_2ND_BODY_PCT", "g_min_2nd_body_pct", 0.0, "num"),
    ("G_MIN_2ND_VOL_MULT", "g_min_2nd_vol_mult", 0.0, "num"),
    ("G_CONVICTION_MULT", "g_conviction_mult", 0.0, "num"),
    ("G_TARGET_PCT", "g_target_pct", 11.0, "num"),
    ("G_TARGET2_PCT", "g_target2_pct", 30.0, "num"),
    ("G_PARTIAL_SELL_PCT", "g_partial_sell_pct", 0.0, "num"),
    ("G_TIME_LIMIT_MINUTES", "g_time_limit_min", 10, "num"),
    ("G_STOP_PCT", "g_stop_pct", 0.0, "num"),
    ("G_TRAIL_PCT", "g_trail_pct", 0.0, "num"),
    ("G_TRAIL_ACTIVATE_PCT", "g_trail_activate_pct", 0.0, "num"),
])

# --- A: Quick Scalp ------------------------------------------------------
# NOTE: A_MIN_GAP_PCT hardcoded 15.0 in original (not params-tunable).
_reg("A", "A_MIN_GAP_PCT", 2, [
    ("A_MIN_GAP_PCT", "a_min_gap_pct", 15.0, "fixed"),
    ("A_TARGET_PCT", "a_target_pct", 6.0, "num"),
    ("A_TIME_LIMIT_MINUTES", "a_time_limit_min", 12, "num"),
    ("A_STOP_PCT", "a_stop_pct", 0.0, "num"),
    ("A_TRAIL_PCT", "a_trail_pct", 0.0, "num"),
    ("A_TRAIL_ACTIVATE_PCT", "a_trail_activate_pct", 0.0, "num"),
])

# --- F: Catch-All --------------------------------------------------------
# NOTE: F_MIN_GAP_PCT hardcoded 10.0 in original (not params-tunable).
_reg("F", "F_MIN_GAP_PCT", 3, [
    ("F_MIN_GAP_PCT", "f_min_gap_pct", 10.0, "fixed"),
    ("F_TARGET_PCT", "f_target_pct", 8.0, "num"),
    ("F_TIME_LIMIT_MINUTES", "f_time_limit_min", 3, "num"),
    ("F_STOP_PCT", "f_stop_pct", 0.0, "num"),
    ("F_TRAIL_PCT", "f_trail_pct", 0.0, "num"),
    ("F_TRAIL_ACTIVATE_PCT", "f_trail_activate_pct", 0.0, "num"),
])

# --- D: Opening Dip Buy --------------------------------------------------
_reg("D", "D_MIN_GAP_PCT", 4, [
    ("D_MIN_GAP_PCT", "d_min_gap", 30.0, "num"),
    ("D_MIN_SPIKE_PCT", "d_min_spike_pct", 10.0, "num"),
    ("D_SPIKE_WINDOW", "d_spike_window", 20, "num"),
    ("D_DIP_PCT", "d_dip_pct", 6.0, "num"),
    ("D_ENTRY_MODE", "d_entry_mode", "5candle", "str"),
    ("D_MAX_ENTRY_CANDLE", "d_max_entry_candle", 45, "num"),
    ("D_TARGET1_PCT", "d_target1_pct", 12.0, "num"),
    ("D_TARGET2_PCT", "d_target2_pct", 12.0, "num"),
    ("D_STOP_PCT", "d_stop_pct", 9.0, "num"),
    ("D_PARTIAL_SELL_PCT", "d_partial_sell_pct", 0.0, "num"),
    ("D_TRAIL_PCT", "d_trail_pct", 2.0, "num"),
    ("D_TRAIL_ACTIVATE_PCT", "d_trail_activate_pct", 2.0, "num"),
    ("D_TIME_LIMIT_MINUTES", "d_time_limit_min", 70, "num"),
])

# --- V: VWAP Reclaim -----------------------------------------------------
_reg("V", "V_MIN_GAP_PCT", 5, [
    ("V_MIN_GAP_PCT", "v_min_gap", 14.0, "num"),
    ("V_MIN_BELOW_CANDLES", "v_min_below_candles", 7, "num"),
    ("V_MIN_BELOW_PCT", "v_min_below_pct", 1.0, "num"),
    ("V_VOL_SPIKE_RATIO", "v_vol_spike_ratio", 3.5, "num"),
    ("V_MAX_ENTRY_CANDLE", "v_max_entry_candle", 60, "num"),
    ("V_TARGET1_PCT", "v_target1_pct", 3.0, "num"),
    ("V_TARGET2_PCT", "v_target2_pct", 19.0, "num"),
    ("V_STOP_PCT", "v_stop_pct", 9.0, "num"),
    ("V_PARTIAL_SELL_PCT", "v_partial_sell_pct", 25.0, "num"),
    ("V_TRAIL_PCT", "v_trail_pct", 3.0, "num"),
    ("V_TRAIL_ACTIVATE_PCT", "v_trail_activate_pct", 5.0, "num"),
    ("V_TIME_LIMIT_MINUTES", "v_time_limit_min", 100, "num"),
])

# --- P: PM High Breakout + Pullback + Bounce -----------------------------
_reg("P", "P_MIN_GAP_PCT", 6, [
    ("P_MIN_GAP_PCT", "p_min_gap", 10.0, "num"),
    ("P_CONFIRM_ABOVE", "p_confirm_above", 3, "num"),
    ("P_CONFIRM_WINDOW", "p_confirm_window", 3, "num"),
    ("P_PULLBACK_PCT", "p_pullback_pct", 9.0, "num"),
    ("P_PULLBACK_TIMEOUT", "p_pullback_timeout", 10, "num"),
    ("P_MAX_ENTRY_CANDLE", "p_max_entry_candle", 105, "num"),
    ("P_TARGET1_PCT", "p_target1_pct", 15.0, "num"),
    ("P_TARGET2_PCT", "p_target2_pct", 15.0, "num"),
    ("P_STOP_PCT", "p_stop_pct", 12.0, "num"),
    ("P_PARTIAL_SELL_PCT", "p_partial_sell_pct", 0.0, "num"),
    ("P_TRAIL_PCT", "p_trail_pct", 2.0, "num"),
    ("P_TRAIL_ACTIVATE_PCT", "p_trail_activate_pct", 2.0, "num"),
    ("P_TIME_LIMIT_MINUTES", "p_time_limit_min", 180, "num"),
])

# --- M: Midday Range Break -----------------------------------------------
_reg("M", "M_MIN_GAP_PCT", 7, [
    ("M_MIN_GAP_PCT", "m_min_gap", 10.0, "num"),
    ("M_MORNING_SPIKE_PCT", "m_morning_spike_pct", 8.0, "num"),
    ("M_MORNING_CANDLES", "m_morning_candles", 40, "num"),
    ("M_RANGE_START_CANDLE", "m_range_start_candle", 55, "num"),
    ("M_CONSOLIDATION_LEN", "m_consolidation_len", 40, "num"),
    ("M_MAX_RANGE_PCT", "m_max_range_pct", 7.0, "num"),
    ("M_VOL_RATIO", "m_vol_ratio", 0.5, "num"),
    ("M_MAX_ENTRY_CANDLE", "m_max_entry_candle", 150, "num"),
    ("M_TARGET1_PCT", "m_target1_pct", 6.0, "num"),
    ("M_STOP_PCT", "m_stop_pct", 10.0, "num"),
    ("M_TIME_LIMIT_MINUTES", "m_time_limit_min", 120, "num"),
    ("M_PARTIAL_SELL_PCT", "m_partial_sell_pct", 50.0, "num"),
    ("M_TRAIL_PCT", "m_trail_pct", 4.0, "num"),
    ("M_TRAIL_ACTIVATE_PCT", "m_trail_activate_pct", 5.0, "num"),
])

# --- R: Multi-Day Runner -------------------------------------------------
_reg("R", "R_DAY1_MIN_GAP", 8, [
    ("R_DAY1_MIN_GAP", "r_day1_min_gap", 40.0, "num"),
    ("R_D2_PULLBACK_PCT", "r_d2_pullback_pct", 10.0, "num"),
    ("R_PULLBACK_WINDOW", "r_pullback_window", 30, "num"),
    ("R_BOUNCE_REF", "r_bounce_ref", "d2_open", "str"),
    ("R_MAX_ENTRY_CANDLE", "r_max_entry_candle", 55, "num"),
    ("R_TARGET1_PCT", "r_target1_pct", 9.0, "num"),
    ("R_STOP_PCT", "r_stop_pct", 9.0, "num"),
    ("R_TRAIL_PCT", "r_trail_pct", 6.0, "num"),
    ("R_TRAIL_ACTIVATE_PCT", "r_trail_activate_pct", 6.0, "num"),
    ("R_TIME_LIMIT_MINUTES", "r_time_limit_min", 100, "num"),
])

# --- W: Power Hour Breakout ---------------------------------------------
_reg("W", "W_MIN_GAP_PCT", 9, [
    ("W_MIN_GAP_PCT", "w_min_gap", 10.0, "num"),
    ("W_MIN_MORNING_RUN", "w_min_morning_run", 4.0, "num"),
    ("W_CONSOL_START", "w_consol_start", 35, "num"),
    ("W_MAX_RANGE_PCT", "w_max_range_pct", 10.0, "num"),
    ("W_MAX_VWAP_DEV_PCT", "w_max_vwap_dev_pct", 5.0, "num"),
    ("W_EARLIEST_CANDLE", "w_earliest_candle", 165, "num"),
    ("W_LATEST_CANDLE", "w_latest_candle", 190, "num"),
    ("W_VOL_SURGE_MULT", "w_vol_surge_mult", 2.0, "num"),
    ("W_VOL_VS_MORNING_MULT", "w_vol_vs_morning_mult", 0.1, "num"),
    ("W_MAX_HOD_BREAKS", "w_max_hod_breaks", 3, "num"),
    ("W_TARGET_PCT", "w_target_pct", 8.0, "num"),
    ("W_STOP_PCT", "w_stop_pct", 3.0, "num"),
    ("W_TRAIL_PCT", "w_trail_pct", 2.5, "num"),
    ("W_TRAIL_ACTIVATE_PCT", "w_trail_activate_pct", 2.0, "num"),
])

# --- O: Opening Range Breakout ------------------------------------------
_reg("O", "O_MIN_GAP_PCT", 10, [
    ("O_MIN_GAP_PCT", "o_min_gap", 10.0, "num"),
    ("O_RANGE_CANDLES", "o_range_candles", 5, "num"),
    ("O_BREAKOUT_VOL_MULT", "o_breakout_vol_mult", 1.5, "num"),
    ("O_MAX_ENTRY_CANDLE", "o_max_entry_candle", 30, "num"),
    ("O_TARGET1_PCT", "o_target1_pct", 8.0, "num"),
    ("O_TARGET2_PCT", "o_target2_pct", 15.0, "num"),
    ("O_STOP_PCT", "o_stop_pct", 0.0, "num"),
    ("O_PARTIAL_SELL_PCT", "o_partial_sell_pct", 50.0, "num"),
    ("O_TRAIL_PCT", "o_trail_pct", 2.0, "num"),
    ("O_TRAIL_ACTIVATE_PCT", "o_trail_activate_pct", 3.0, "num"),
    ("O_TIME_LIMIT_MINUTES", "o_time_limit_min", 60, "num"),
])

# --- B: Red-to-Green (R2G) ----------------------------------------------
_reg("B", "B_MIN_GAP_PCT", 11, [
    ("B_MIN_GAP_PCT", "b_min_gap", 15.0, "num"),
    ("B_MAX_DIP_PCT", "b_max_dip_pct", 5.0, "num"),
    ("B_MIN_RECLAIM_VOL_MULT", "b_min_reclaim_vol_mult", 1.5, "num"),
    ("B_MAX_ENTRY_CANDLE", "b_max_entry_candle", 20, "num"),
    ("B_TARGET1_PCT", "b_target1_pct", 6.0, "num"),
    ("B_TARGET2_PCT", "b_target2_pct", 12.0, "num"),
    ("B_STOP_PCT", "b_stop_pct", 4.0, "num"),
    ("B_PARTIAL_SELL_PCT", "b_partial_sell_pct", 50.0, "num"),
    ("B_TRAIL_PCT", "b_trail_pct", 2.0, "num"),
    ("B_TRAIL_ACTIVATE_PCT", "b_trail_activate_pct", 3.0, "num"),
    ("B_TIME_LIMIT_MINUTES", "b_time_limit_min", 30, "num"),
])

# --- K: First Pullback Buy ----------------------------------------------
_reg("K", "K_MIN_GAP_PCT", 12, [
    ("K_MIN_GAP_PCT", "k_min_gap", 10.0, "num"),
    ("K_MIN_RUN_PCT", "k_min_run_pct", 5.0, "num"),
    ("K_RUN_WINDOW", "k_run_window", 15, "num"),
    ("K_PULLBACK_PCT", "k_pullback_pct", 3.0, "num"),
    ("K_PULLBACK_VOL_RATIO", "k_pullback_vol_ratio", 0.5, "num"),
    ("K_BOUNCE_VOL_MULT", "k_bounce_vol_mult", 1.5, "num"),
    ("K_MAX_ENTRY_CANDLE", "k_max_entry_candle", 45, "num"),
    ("K_TARGET1_PCT", "k_target1_pct", 8.0, "num"),
    ("K_TARGET2_PCT", "k_target2_pct", 15.0, "num"),
    ("K_STOP_PCT", "k_stop_pct", 5.0, "num"),
    ("K_PARTIAL_SELL_PCT", "k_partial_sell_pct", 50.0, "num"),
    ("K_TRAIL_PCT", "k_trail_pct", 2.0, "num"),
    ("K_TRAIL_ACTIVATE_PCT", "k_trail_activate_pct", 3.0, "num"),
    ("K_TIME_LIMIT_MINUTES", "k_time_limit_min", 60, "num"),
])

# --- C: Micro Flag / Base Pattern ---------------------------------------
_reg("C", "C_MIN_GAP_PCT", 13, [
    ("C_MIN_GAP_PCT", "c_min_gap", 10.0, "num"),
    ("C_MIN_SPIKE_PCT", "c_min_spike_pct", 5.0, "num"),
    ("C_MIN_BASE_CANDLES", "c_min_base_candles", 3, "num"),
    ("C_MAX_BASE_CANDLES", "c_max_base_candles", 8, "num"),
    ("C_MAX_BASE_RANGE_PCT", "c_max_base_range_pct", 3.0, "num"),
    ("C_BREAKOUT_VOL_MULT", "c_breakout_vol_mult", 1.5, "num"),
    ("C_MAX_ENTRY_CANDLE", "c_max_entry_candle", 60, "num"),
    ("C_TARGET1_PCT", "c_target1_pct", 8.0, "num"),
    ("C_TARGET2_PCT", "c_target2_pct", 15.0, "num"),
    ("C_STOP_PCT", "c_stop_pct", 4.0, "num"),
    ("C_PARTIAL_SELL_PCT", "c_partial_sell_pct", 50.0, "num"),
    ("C_TRAIL_PCT", "c_trail_pct", 2.0, "num"),
    ("C_TRAIL_ACTIVATE_PCT", "c_trail_activate_pct", 3.0, "num"),
    ("C_TIME_LIMIT_MINUTES", "c_time_limit_min", 60, "num"),
])

# --- S: Stuff-and-Break -------------------------------------------------
_reg("S", "S_MIN_GAP_PCT", 14, [
    ("S_MIN_GAP_PCT", "s_min_gap", 10.0, "num"),
    ("S_MIN_HOD_TESTS", "s_min_hod_tests", 2, "num"),
    ("S_HOD_TOLERANCE_PCT", "s_hod_tolerance_pct", 0.5, "num"),
    ("S_REJECTION_PCT", "s_rejection_pct", 1.0, "num"),
    ("S_BREAKOUT_VOL_MULT", "s_breakout_vol_mult", 1.5, "num"),
    ("S_MAX_ENTRY_CANDLE", "s_max_entry_candle", 90, "num"),
    ("S_TARGET1_PCT", "s_target1_pct", 8.0, "num"),
    ("S_TARGET2_PCT", "s_target2_pct", 15.0, "num"),
    ("S_STOP_PCT", "s_stop_pct", 4.0, "num"),
    ("S_PARTIAL_SELL_PCT", "s_partial_sell_pct", 50.0, "num"),
    ("S_TRAIL_PCT", "s_trail_pct", 2.0, "num"),
    ("S_TRAIL_ACTIVATE_PCT", "s_trail_activate_pct", 3.0, "num"),
    ("S_TIME_LIMIT_MINUTES", "s_time_limit_min", 90, "num"),
])

# --- E: Gap-and-Go RelVol -----------------------------------------------
_reg("E", "E_MIN_GAP_PCT", 15, [
    ("E_MIN_GAP_PCT", "e_min_gap", 15.0, "num"),
    ("E_MIN_PM_VOL_MULT", "e_min_pm_vol_mult", 5.0, "num"),
    ("E_MAX_ENTRY_CANDLE", "e_max_entry_candle", 5, "num"),
    ("E_TARGET1_PCT", "e_target1_pct", 6.0, "num"),
    ("E_TARGET2_PCT", "e_target2_pct", 12.0, "num"),
    ("E_STOP_PCT", "e_stop_pct", 4.0, "num"),
    ("E_PARTIAL_SELL_PCT", "e_partial_sell_pct", 50.0, "num"),
    ("E_TRAIL_PCT", "e_trail_pct", 2.0, "num"),
    ("E_TRAIL_ACTIVATE_PCT", "e_trail_activate_pct", 3.0, "num"),
    ("E_TIME_LIMIT_MINUTES", "e_time_limit_min", 20, "num"),
])

# --- I: P1 Immediate PM High Breakout -----------------------------------
_reg("I", "I_MIN_GAP_PCT", 16, [
    ("I_MIN_GAP_PCT", "i_min_gap", 10.0, "num"),
    ("I_MAX_ENTRY_CANDLE", "i_max_entry_candle", 30, "num"),
    ("I_BREAKOUT_VOL_MULT", "i_breakout_vol_mult", 1.5, "num"),
    ("I_TARGET1_PCT", "i_target1_pct", 8.0, "num"),
    ("I_TARGET2_PCT", "i_target2_pct", 15.0, "num"),
    ("I_STOP_PCT", "i_stop_pct", 5.0, "num"),
    ("I_PARTIAL_SELL_PCT", "i_partial_sell_pct", 50.0, "num"),
    ("I_TRAIL_PCT", "i_trail_pct", 2.0, "num"),
    ("I_TRAIL_ACTIVATE_PCT", "i_trail_activate_pct", 3.0, "num"),
    ("I_TIME_LIMIT_MINUTES", "i_time_limit_min", 60, "num"),
])

# --- J: P3 VWAP + PM High Breakout --------------------------------------
_reg("J", "J_MIN_GAP_PCT", 17, [
    ("J_MIN_GAP_PCT", "j_min_gap", 10.0, "num"),
    ("J_MAX_ENTRY_CANDLE", "j_max_entry_candle", 90, "num"),
    ("J_VWAP_PROXIMITY_PCT", "j_vwap_proximity_pct", 2.0, "num"),
    ("J_TARGET1_PCT", "j_target1_pct", 8.0, "num"),
    ("J_TARGET2_PCT", "j_target2_pct", 15.0, "num"),
    ("J_STOP_PCT", "j_stop_pct", 5.0, "num"),
    ("J_PARTIAL_SELL_PCT", "j_partial_sell_pct", 50.0, "num"),
    ("J_TRAIL_PCT", "j_trail_pct", 2.0, "num"),
    ("J_TRAIL_ACTIVATE_PCT", "j_trail_activate_pct", 3.0, "num"),
    ("J_TIME_LIMIT_MINUTES", "j_time_limit_min", 90, "num"),
])

# --- N: P4 HOD Reclaim ---------------------------------------------------
_reg("N", "N_MIN_GAP_PCT", 18, [
    ("N_MIN_GAP_PCT", "n_min_gap", 10.0, "num"),
    ("N_MIN_HOD_AGE", "n_min_hod_age", 10, "num"),
    ("N_PULLBACK_FROM_HOD_PCT", "n_pullback_from_hod_pct", 3.0, "num"),
    ("N_MAX_ENTRY_CANDLE", "n_max_entry_candle", 120, "num"),
    ("N_TARGET1_PCT", "n_target1_pct", 8.0, "num"),
    ("N_TARGET2_PCT", "n_target2_pct", 15.0, "num"),
    ("N_STOP_PCT", "n_stop_pct", 5.0, "num"),
    ("N_PARTIAL_SELL_PCT", "n_partial_sell_pct", 50.0, "num"),
    ("N_TRAIL_PCT", "n_trail_pct", 2.0, "num"),
    ("N_TRAIL_ACTIVATE_PCT", "n_trail_activate_pct", 3.0, "num"),
    ("N_TIME_LIMIT_MINUTES", "n_time_limit_min", 90, "num"),
])

# --- L: Low Float Squeeze ------------------------------------------------
_reg("L", "L_MIN_GAP_PCT", 19, [
    ("L_MIN_GAP_PCT", "l_min_gap", 30.0, "num"),
    ("L_MAX_FLOAT", "l_max_float", 15000000, "num"),
    ("L_EARLIEST_CANDLE", "l_earliest_candle", 8, "num"),
    ("L_LATEST_CANDLE", "l_latest_candle", 115, "num"),
    ("L_VOL_SURGE_MULT", "l_vol_surge_mult", 1.5, "num"),
    ("L_MIN_PRICE_ACCEL_PCT", "l_min_price_accel_pct", 1.0, "num"),
    ("L_TIER1_FLOAT", "l_tier1_float", 1000000, "num"),
    ("L_TIER2_FLOAT", "l_tier2_float", 5000000, "num"),
    ("L_TIER1_TARGET1_PCT", "l_tier1_target1_pct", 30.0, "num"),
    ("L_TIER1_TARGET2_PCT", "l_tier1_target2_pct", 40.0, "num"),
    ("L_TIER2_TARGET1_PCT", "l_tier2_target1_pct", 15.0, "num"),
    ("L_TIER2_TARGET2_PCT", "l_tier2_target2_pct", 40.0, "num"),
    ("L_TIER3_TARGET1_PCT", "l_tier3_target1_pct", 9.0, "num"),
    ("L_TIER3_TARGET2_PCT", "l_tier3_target2_pct", 32.0, "num"),
    ("L_STOP_PCT", "l_stop_pct", 14.0, "num"),
    ("L_PARTIAL_SELL_PCT", "l_partial_sell_pct", 0.0, "num"),
    ("L_TRAIL_PCT", "l_trail_pct", 1.0, "num"),
    ("L_TRAIL_ACTIVATE_PCT", "l_trail_activate_pct", 2.0, "num"),
    ("L_TIME_LIMIT_MINUTES", "l_time_limit_min", 70, "num"),
])

# --- X: Range Reversion (always applies) --------------------------------
# Original set_strategy_params tunes 10 X params via params.get(); the other 5
# (X_FIRST_LEG_WINDOW_BARS, X_ENTRY_REQUIRE_GREEN, X_VOL_AVG_BARS,
# X_MIN_VOL_VS_AVG, X_MAX_ENTRY_HHMM) are config-block constants — marked
# 'fixed' so apply_params does not read them from params.
_reg("X", "X_MIN_FIRST_LEG_GAIN_PCT", 20, [
    ("X_MIN_FIRST_LEG_GAIN_PCT", "x_min_first_leg_gain_pct", 5.0, "num"),
    ("X_FIRST_LEG_WINDOW_BARS", "x_first_leg_window_bars", 15, "fixed"),
    ("X_MIN_PULLBACK_PCT", "x_min_pullback_pct", 10.0, "num"),
    ("X_MIN_RECOVERY_PCT", "x_min_recovery_pct", 3.0, "num"),
    ("X_MIN_BARS_SINCE_PEAK", "x_min_bars_since_peak", 5, "num"),
    ("X_ENTRY_REQUIRE_GREEN", "x_entry_require_green", True, "fixed"),
    ("X_TARGET_PCT_OF_PEAK", "x_target_pct_of_peak", 92.0, "num"),
    ("X_STOP_PCT_BELOW_TROUGH", "x_stop_pct_below_trough", 2.0, "num"),
    ("X_TRAIL_PCT", "x_trail_pct", 5.0, "num"),
    ("X_TRAIL_ACTIVATE_PCT", "x_trail_activate_pct", 5.0, "num"),
    ("X_TIME_LIMIT_MINUTES", "x_time_limit_min", 60, "num"),
    ("X_MIN_ENTRY_ROOM_PCT", "x_min_entry_room_pct", 4.0, "num"),
    ("X_MIN_VOL_VS_AVG", "x_min_vol_vs_avg", 1.0, "fixed"),
    ("X_VOL_AVG_BARS", "x_vol_avg_bars", 5, "fixed"),
    ("X_MAX_ENTRY_HHMM", "x_max_entry_hhmm", "14:30", "fixed"),
])


# ---------------------------------------------------------------------------
# apply_params: data-driven replacement for optimize_combined.set_strategy_params
# ---------------------------------------------------------------------------

# Default priority for strategies (matches _default_priority in optimize_combined)
DEFAULT_PRIORITY = {s.upper(): c.priority for s, c in STRATEGIES.items()}

# participation-cap globals set by set_strategy_params
PARTICIPATION_CAP_GLOBALS = [f"{l.upper()}_PARTICIPATION_CAP" for l in STRATEGIES]

# Phase-1A day-level filter globals
PHASE1A_FILTERS = [
    ("MIN_PRICE", "min_price", 0.0),
    ("MAX_MODELED_SLIP_BP", "max_modeled_slip_bp", 0.0),
    ("MAX_CUM_DVOL_AT_ENTRY_M", "max_cum_dvol_at_entry_m", 0.0),
    ("MIN_ATR_PCT", "min_atr_pct", 0.0),
]

# Weighted favorability signal globals
FAVORABILITY_WEIGHTS = [
    ("W_NOT_WED", "w_not_wed", 0.0),
    ("W_NOT_JUN_AUG", "w_not_jun_aug", 0.0),
    ("W_DXY_UP", "w_dxy_up", 0.0),
    ("W_BTC_FLAT", "w_btc_flat", 0.0),
    ("W_VIX_MID", "w_vix_mid", 0.0),
    ("W_IWM_MID", "w_iwm_mid", 0.0),
    ("W_PM_DVOL_MID", "w_pm_dvol_mid", 0.0),
    ("MIN_FAVORABILITY_THRESHOLD", "min_favorability_threshold", 0.0),
    ("BTC_FLAT_BANDWIDTH", "btc_flat_bandwidth", 5.0),
]


# ---------------------------------------------------------------------------
# Coercion policy — derived EXACTLY from the original set_strategy_params.
# The original applied float()/bool()/int() wrappers only to specific globals;
# everything else used raw params.get() (preserving the params dict's type).
# ---------------------------------------------------------------------------

# Globals the original wrapped in float(...)
FLOAT_COERCE = {
    "BTC_FLAT_BANDWIDTH", "B_MIN_GAP_PCT", "C_MIN_GAP_PCT", "D_DIP_PCT",
    "D_MIN_GAP_PCT", "D_MIN_SPIKE_PCT", "E_MIN_GAP_PCT", "G_MIN_GAP_PCT",
    "I_MIN_GAP_PCT", "J_MIN_GAP_PCT", "K_MIN_GAP_PCT", "L_MIN_GAP_PCT",
    "MAX_CUM_DVOL_AT_ENTRY_M", "MAX_MODELED_SLIP_BP", "MIN_ATR_PCT",
    "MIN_FAVORABILITY_THRESHOLD", "MIN_PRICE", "M_MIN_GAP_PCT",
    "N_MIN_GAP_PCT", "O_MIN_GAP_PCT", "P_MIN_GAP_PCT", "R_DAY1_MIN_GAP",
    "S_MIN_GAP_PCT", "V_MIN_BELOW_PCT", "V_MIN_GAP_PCT", "W_MIN_GAP_PCT",
    # phase-1A filters
    "MIN_PRICE", "MAX_MODELED_SLIP_BP", "MAX_CUM_DVOL_AT_ENTRY_M", "MIN_ATR_PCT",
}

# Globals the original wrapped in bool(...)
BOOL_COERCE = {
    "G_REQUIRE_2ND_GREEN", "G_REQUIRE_2ND_NEW_HIGH",
    "NEWS_FILTER_ENABLED", "NEWS_REQUIRE_CATALYST",
}

# Globals the original wrapped in int(...)
INT_COERCE = {"NEWS_MIN_ARTICLES"}


def _coerce(kind, raw, default):
    if kind == "fixed":
        # non-tunable constant — never read from params
        return default
    if kind == "num":
        # 'num' is only used for globals the original wrapped in float();
        # everything else is 'raw' (preserve params-dict type exactly).
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default
    if kind == "bool":
        return bool(raw)
    if kind == "int":
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default
    return raw


def apply_params(params, target):
    """Apply a params dict onto `target` (a module or object), exactly
    reproducing optimize_combined.set_strategy_params() semantics.

    `params` keys are the lowercase optimizer keys (g_target_pct, ...).
    Missing keys fall back to the registry default.
    """
    params = params or {}

    # Apply each strategy's params ('fixed' entries always use their default;
    # coercion follows the ORIGINAL set_strategy_params wrapper policy).
    for cfg in STRATEGIES.values():
        for global_name, key, default, kind in cfg.params:
            if kind == "fixed":
                val = default
            else:
                val = params.get(key, default)
                if global_name in FLOAT_COERCE:
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = default
                elif global_name in INT_COERCE:
                    try:
                        val = int(val)
                    except (TypeError, ValueError):
                        val = default
                elif global_name in BOOL_COERCE:
                    val = bool(val)
                # else: raw — preserve params-dict type exactly (original behavior)
            setattr(target, global_name, val)

    # Enable/disable gates: set min_gap (or X first-leg) to DISABLED_GAP when
    # disabled. Re-reset enabled strategies' min gaps afterward (matches
    # original reset of ALL min_gap thresholds at the top of set_strategy_params,
    # which only runs for the tunable gap globals).
    for s, cfg in STRATEGIES.items():
        low = s.lower()
        enabled = params.get(f"enable_{low}", False)
        if s == "R":
            if not enabled:
                setattr(target, "R_DAY1_MIN_GAP", DISABLED_GAP)
            else:
                setattr(target, "R_DAY1_MIN_GAP", float(params.get("r_day1_min_gap", 40.0)))
        else:
            if not enabled:
                setattr(target, cfg.min_gap_global, DISABLED_GAP)
            else:
                # re-apply the min gap from params (matches original reset)
                first = cfg.params[0]
                g, k, d, kind = first
                if kind == "fixed":
                    # H/A/F min gaps are hardcoded constants in the original —
                    # never read from params even when the strategy is enabled.
                    setattr(target, g, d)
                elif g in FLOAT_COERCE:
                    try:
                        setattr(target, g, float(params.get(k, d)))
                    except (TypeError, ValueError):
                        setattr(target, g, d)
                else:
                    setattr(target, g, params.get(k, d))

    # Strategy priority
    priority = {
        s.upper(): params.get(f"priority_{s.lower()}", DEFAULT_PRIORITY[s.upper()])
        for s in STRATEGIES
    }
    setattr(target, "STRAT_PRIORITY", priority)

    # Phase 1A day-level filters
    for global_name, key, default in PHASE1A_FILTERS:
        setattr(target, global_name, float(params.get(key, default)))

    # Per-strategy participation caps
    for s in STRATEGIES:
        key = f"{s.lower()}_participation_cap"
        setattr(target, f"{s.upper()}_PARTICIPATION_CAP", float(params.get(key, 0.0)))

    # Weighted favorability signals
    for global_name, key, default in FAVORABILITY_WEIGHTS:
        setattr(target, global_name, float(params.get(key, default)))

    # PIT news filter (matches set_strategy_params lines 446-448)
    setattr(target, "NEWS_FILTER_ENABLED", bool(params.get("enable_news_filter", False)))
    setattr(target, "NEWS_MIN_ARTICLES", int(params.get("min_news_articles", 0)))
    setattr(target, "NEWS_REQUIRE_CATALYST", bool(params.get("require_news_catalyst", False)))

    return target


def defaults_snapshot():
    """Return {GLOBAL_NAME: default} for every param in the registry."""
    out = {}
    for cfg in STRATEGIES.values():
        for global_name, _key, default, _kind in cfg.params:
            out[global_name] = default
    for _g, _k, d in PHASE1A_FILTERS:
        # defaults handled at apply time
        pass
    return out
