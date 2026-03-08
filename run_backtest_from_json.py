"""Run backtest using params from optuna JSON file.

Usage:
    python run_backtest_from_json.py                          # charts, stored_data_combined
    python run_backtest_from_json.py --no-charts              # no charts
    python run_backtest_from_json.py --json optuna_best.json  # custom json
    python run_backtest_from_json.py stored_data_2023          # different data dir
"""
import sys
import os
import io
import json

# Parse args
json_path = "optuna_best_params_v8.json"
no_charts = "--no-charts" in sys.argv
remaining = [a for a in sys.argv[1:] if a not in ("--no-charts",)]

data_dirs = []
i = 0
while i < len(remaining):
    if remaining[i] == "--json" and i + 1 < len(remaining):
        json_path = remaining[i + 1]
        i += 2
    else:
        data_dirs.append(remaining[i])
        i += 1
if not data_dirs:
    data_dirs = ["stored_data_combined"]

# Load JSON
with open(json_path) as f:
    data = json.load(f)
params = data["params"]

# Save fd before imports can break stdout
_backup_fd = os.dup(1)

# Import to get the param mapping
from optimize_combined import ALL_STRATS

# Fix stdout if broken
try:
    sys.stdout.write("")
except (ValueError, OSError):
    sys.stdout = io.TextIOWrapper(os.fdopen(_backup_fd, "wb"), encoding="utf-8",
                                  errors="replace", line_buffering=True)

print(f"=== Backtest with Optuna trial {data['trial_number']} params ===")
print(f"  Source: {json_path}")
print(f"  Optimizer score: {data['score']:,.0f} | PnL: ${data['total_pnl']:,.0f} | PF: {data['pf']}")
print(f"  Enabled: {data['enabled']} ({data['n_strategies']} strategies)")
print(f"  Priority: {data['priority']}")
print()

# Set sys.argv
sys.argv = ["test_green_candle_combined.py"] + data_dirs
if no_charts:
    sys.argv.append("--no-charts")

# Read source
with open("test_green_candle_combined.py", "r", encoding="utf-8") as f:
    source = f.read()

# Disable stdout re-wrapping
source = source.replace(
    "if hasattr(_sys.stdout, 'buffer'):\n    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer",
    "if False:  # disabled by runner\n    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer"
)

# Build direct variable override statements from the JSON params
# These will be injected right before __main__ to override the module-level defaults
overrides = []

# Map JSON param keys to module-level variable names
param_to_var = {
    "h_target_pct": "H_TARGET_PCT", "h_time_limit_min": "H_TIME_LIMIT_MINUTES",
    "h_stop_pct": "H_STOP_PCT", "h_trail_pct": "H_TRAIL_PCT", "h_trail_activate_pct": "H_TRAIL_ACTIVATE_PCT",
    "g_target_pct": "G_TARGET_PCT", "g_time_limit_min": "G_TIME_LIMIT_MINUTES",
    "g_stop_pct": "G_STOP_PCT", "g_trail_pct": "G_TRAIL_PCT", "g_trail_activate_pct": "G_TRAIL_ACTIVATE_PCT",
    "a_target_pct": "A_TARGET_PCT", "a_time_limit_min": "A_TIME_LIMIT_MINUTES",
    "a_stop_pct": "A_STOP_PCT", "a_trail_pct": "A_TRAIL_PCT", "a_trail_activate_pct": "A_TRAIL_ACTIVATE_PCT",
    "f_target_pct": "F_TARGET_PCT", "f_time_limit_min": "F_TIME_LIMIT_MINUTES",
    "f_stop_pct": "F_STOP_PCT", "f_trail_pct": "F_TRAIL_PCT", "f_trail_activate_pct": "F_TRAIL_ACTIVATE_PCT",
    "d_min_gap": "D_MIN_GAP_PCT", "d_min_spike_pct": "D_MIN_SPIKE_PCT", "d_spike_window": "D_SPIKE_WINDOW",
    "d_dip_pct": "D_DIP_PCT", "d_entry_mode": "D_ENTRY_MODE", "d_max_entry_candle": "D_MAX_ENTRY_CANDLE",
    "d_target1_pct": "D_TARGET1_PCT", "d_target2_pct": "D_TARGET2_PCT", "d_stop_pct": "D_STOP_PCT",
    "d_partial_sell_pct": "D_PARTIAL_SELL_PCT", "d_trail_pct": "D_TRAIL_PCT",
    "d_trail_activate_pct": "D_TRAIL_ACTIVATE_PCT", "d_time_limit_min": "D_TIME_LIMIT_MINUTES",
    "v_min_gap": "V_MIN_GAP_PCT", "v_min_below_candles": "V_MIN_BELOW_CANDLES",
    "v_min_below_pct": "V_MIN_BELOW_PCT", "v_vol_spike_ratio": "V_VOL_SPIKE_RATIO",
    "v_max_entry_candle": "V_MAX_ENTRY_CANDLE", "v_target1_pct": "V_TARGET1_PCT",
    "v_target2_pct": "V_TARGET2_PCT", "v_stop_pct": "V_STOP_PCT",
    "v_partial_sell_pct": "V_PARTIAL_SELL_PCT", "v_trail_pct": "V_TRAIL_PCT",
    "v_trail_activate_pct": "V_TRAIL_ACTIVATE_PCT", "v_time_limit_min": "V_TIME_LIMIT_MINUTES",
    "p_min_gap": "P_MIN_GAP_PCT", "p_confirm_above": "P_CONFIRM_ABOVE",
    "p_confirm_window": "P_CONFIRM_WINDOW", "p_pullback_pct": "P_PULLBACK_PCT",
    "p_pullback_timeout": "P_PULLBACK_TIMEOUT", "p_max_entry_candle": "P_MAX_ENTRY_CANDLE",
    "p_target1_pct": "P_TARGET1_PCT", "p_target2_pct": "P_TARGET2_PCT", "p_stop_pct": "P_STOP_PCT",
    "p_partial_sell_pct": "P_PARTIAL_SELL_PCT", "p_trail_pct": "P_TRAIL_PCT",
    "p_trail_activate_pct": "P_TRAIL_ACTIVATE_PCT", "p_time_limit_min": "P_TIME_LIMIT_MINUTES",
    "m_min_gap": "M_MIN_GAP_PCT", "m_morning_spike_pct": "M_MORNING_SPIKE_PCT",
    "m_morning_candles": "M_MORNING_CANDLES", "m_range_start_candle": "M_RANGE_START_CANDLE",
    "m_consolidation_len": "M_CONSOLIDATION_LEN", "m_max_range_pct": "M_MAX_RANGE_PCT",
    "m_vol_ratio": "M_VOL_RATIO", "m_max_entry_candle": "M_MAX_ENTRY_CANDLE",
    "m_target1_pct": "M_TARGET1_PCT", "m_stop_pct": "M_STOP_PCT",
    "m_time_limit_min": "M_TIME_LIMIT_MINUTES", "m_partial_sell_pct": "M_PARTIAL_SELL_PCT",
    "m_trail_pct": "M_TRAIL_PCT", "m_trail_activate_pct": "M_TRAIL_ACTIVATE_PCT",
    "r_day1_min_gap": "R_DAY1_MIN_GAP", "r_d2_pullback_pct": "R_D2_PULLBACK_PCT",
    "r_pullback_window": "R_PULLBACK_WINDOW", "r_bounce_ref": "R_BOUNCE_REF",
    "r_max_entry_candle": "R_MAX_ENTRY_CANDLE", "r_target1_pct": "R_TARGET1_PCT",
    "r_stop_pct": "R_STOP_PCT", "r_trail_pct": "R_TRAIL_PCT",
    "r_trail_activate_pct": "R_TRAIL_ACTIVATE_PCT", "r_time_limit_min": "R_TIME_LIMIT_MINUTES",
    "w_min_gap": "W_MIN_GAP_PCT", "w_min_morning_run": "W_MIN_MORNING_RUN",
    "w_consol_start": "W_CONSOL_START", "w_max_range_pct": "W_MAX_RANGE_PCT",
    "w_max_vwap_dev_pct": "W_MAX_VWAP_DEV_PCT", "w_earliest_candle": "W_EARLIEST_CANDLE",
    "w_latest_candle": "W_LATEST_CANDLE", "w_vol_surge_mult": "W_VOL_SURGE_MULT",
    "w_vol_vs_morning_mult": "W_VOL_VS_MORNING_MULT", "w_max_hod_breaks": "W_MAX_HOD_BREAKS",
    "w_target_pct": "W_TARGET_PCT", "w_stop_pct": "W_STOP_PCT",
    "w_trail_pct": "W_TRAIL_PCT", "w_trail_activate_pct": "W_TRAIL_ACTIVATE_PCT",
    "o_min_gap": "O_MIN_GAP_PCT", "o_range_candles": "O_RANGE_CANDLES",
    "o_breakout_vol_mult": "O_BREAKOUT_VOL_MULT", "o_max_entry_candle": "O_MAX_ENTRY_CANDLE",
    "o_target1_pct": "O_TARGET1_PCT", "o_target2_pct": "O_TARGET2_PCT", "o_stop_pct": "O_STOP_PCT",
    "o_partial_sell_pct": "O_PARTIAL_SELL_PCT", "o_trail_pct": "O_TRAIL_PCT",
    "o_trail_activate_pct": "O_TRAIL_ACTIVATE_PCT", "o_time_limit_min": "O_TIME_LIMIT_MINUTES",
    "b_min_gap": "B_MIN_GAP_PCT", "b_max_dip_pct": "B_MAX_DIP_PCT",
    "b_min_reclaim_vol_mult": "B_MIN_RECLAIM_VOL_MULT", "b_max_entry_candle": "B_MAX_ENTRY_CANDLE",
    "b_target1_pct": "B_TARGET1_PCT", "b_target2_pct": "B_TARGET2_PCT", "b_stop_pct": "B_STOP_PCT",
    "b_partial_sell_pct": "B_PARTIAL_SELL_PCT", "b_trail_pct": "B_TRAIL_PCT",
    "b_trail_activate_pct": "B_TRAIL_ACTIVATE_PCT", "b_time_limit_min": "B_TIME_LIMIT_MINUTES",
    "k_min_gap": "K_MIN_GAP_PCT", "k_min_run_pct": "K_MIN_RUN_PCT", "k_run_window": "K_RUN_WINDOW",
    "k_pullback_pct": "K_PULLBACK_PCT", "k_pullback_vol_ratio": "K_PULLBACK_VOL_RATIO",
    "k_bounce_vol_mult": "K_BOUNCE_VOL_MULT", "k_max_entry_candle": "K_MAX_ENTRY_CANDLE",
    "k_target1_pct": "K_TARGET1_PCT", "k_target2_pct": "K_TARGET2_PCT", "k_stop_pct": "K_STOP_PCT",
    "k_partial_sell_pct": "K_PARTIAL_SELL_PCT", "k_trail_pct": "K_TRAIL_PCT",
    "k_trail_activate_pct": "K_TRAIL_ACTIVATE_PCT", "k_time_limit_min": "K_TIME_LIMIT_MINUTES",
    "c_min_gap": "C_MIN_GAP_PCT", "c_min_spike_pct": "C_MIN_SPIKE_PCT",
    "c_min_base_candles": "C_MIN_BASE_CANDLES", "c_max_base_candles": "C_MAX_BASE_CANDLES",
    "c_max_base_range_pct": "C_MAX_BASE_RANGE_PCT", "c_breakout_vol_mult": "C_BREAKOUT_VOL_MULT",
    "c_max_entry_candle": "C_MAX_ENTRY_CANDLE", "c_target1_pct": "C_TARGET1_PCT",
    "c_target2_pct": "C_TARGET2_PCT", "c_stop_pct": "C_STOP_PCT",
    "c_partial_sell_pct": "C_PARTIAL_SELL_PCT", "c_trail_pct": "C_TRAIL_PCT",
    "c_trail_activate_pct": "C_TRAIL_ACTIVATE_PCT", "c_time_limit_min": "C_TIME_LIMIT_MINUTES",
    "s_min_gap": "S_MIN_GAP_PCT", "s_min_hod_tests": "S_MIN_HOD_TESTS",
    "s_hod_tolerance_pct": "S_HOD_TOLERANCE_PCT", "s_rejection_pct": "S_REJECTION_PCT",
    "s_breakout_vol_mult": "S_BREAKOUT_VOL_MULT", "s_max_entry_candle": "S_MAX_ENTRY_CANDLE",
    "s_target1_pct": "S_TARGET1_PCT", "s_target2_pct": "S_TARGET2_PCT", "s_stop_pct": "S_STOP_PCT",
    "s_partial_sell_pct": "S_PARTIAL_SELL_PCT", "s_trail_pct": "S_TRAIL_PCT",
    "s_trail_activate_pct": "S_TRAIL_ACTIVATE_PCT", "s_time_limit_min": "S_TIME_LIMIT_MINUTES",
    "e_min_gap": "E_MIN_GAP_PCT", "e_min_pm_vol_mult": "E_MIN_PM_VOL_MULT",
    "e_max_entry_candle": "E_MAX_ENTRY_CANDLE", "e_target1_pct": "E_TARGET1_PCT",
    "e_target2_pct": "E_TARGET2_PCT", "e_stop_pct": "E_STOP_PCT",
    "e_partial_sell_pct": "E_PARTIAL_SELL_PCT", "e_trail_pct": "E_TRAIL_PCT",
    "e_trail_activate_pct": "E_TRAIL_ACTIVATE_PCT", "e_time_limit_min": "E_TIME_LIMIT_MINUTES",
    "i_min_gap": "I_MIN_GAP_PCT", "i_max_entry_candle": "I_MAX_ENTRY_CANDLE",
    "i_breakout_vol_mult": "I_BREAKOUT_VOL_MULT", "i_target1_pct": "I_TARGET1_PCT",
    "i_target2_pct": "I_TARGET2_PCT", "i_stop_pct": "I_STOP_PCT",
    "i_partial_sell_pct": "I_PARTIAL_SELL_PCT", "i_trail_pct": "I_TRAIL_PCT",
    "i_trail_activate_pct": "I_TRAIL_ACTIVATE_PCT", "i_time_limit_min": "I_TIME_LIMIT_MINUTES",
    "j_min_gap": "J_MIN_GAP_PCT", "j_max_entry_candle": "J_MAX_ENTRY_CANDLE",
    "j_vwap_proximity_pct": "J_VWAP_PROXIMITY_PCT", "j_target1_pct": "J_TARGET1_PCT",
    "j_target2_pct": "J_TARGET2_PCT", "j_stop_pct": "J_STOP_PCT",
    "j_partial_sell_pct": "J_PARTIAL_SELL_PCT", "j_trail_pct": "J_TRAIL_PCT",
    "j_trail_activate_pct": "J_TRAIL_ACTIVATE_PCT", "j_time_limit_min": "J_TIME_LIMIT_MINUTES",
    "n_min_gap": "N_MIN_GAP_PCT", "n_min_hod_age": "N_MIN_HOD_AGE",
    "n_pullback_from_hod_pct": "N_PULLBACK_FROM_HOD_PCT", "n_max_entry_candle": "N_MAX_ENTRY_CANDLE",
    "n_target1_pct": "N_TARGET1_PCT", "n_target2_pct": "N_TARGET2_PCT", "n_stop_pct": "N_STOP_PCT",
    "n_partial_sell_pct": "N_PARTIAL_SELL_PCT", "n_trail_pct": "N_TRAIL_PCT",
    "n_trail_activate_pct": "N_TRAIL_ACTIVATE_PCT", "n_time_limit_min": "N_TIME_LIMIT_MINUTES",
    "l_min_gap": "L_MIN_GAP_PCT", "l_max_float": "L_MAX_FLOAT",
    "l_earliest_candle": "L_EARLIEST_CANDLE", "l_latest_candle": "L_LATEST_CANDLE",
    "l_vol_surge_mult": "L_VOL_SURGE_MULT", "l_min_price_accel_pct": "L_MIN_PRICE_ACCEL_PCT",
    "l_tier1_float": "L_TIER1_FLOAT", "l_tier2_float": "L_TIER2_FLOAT",
    "l_tier1_target1_pct": "L_TIER1_TARGET1_PCT", "l_tier1_target2_pct": "L_TIER1_TARGET2_PCT",
    "l_tier2_target1_pct": "L_TIER2_TARGET1_PCT", "l_tier2_target2_pct": "L_TIER2_TARGET2_PCT",
    "l_tier3_target1_pct": "L_TIER3_TARGET1_PCT", "l_tier3_target2_pct": "L_TIER3_TARGET2_PCT",
    "l_stop_pct": "L_STOP_PCT", "l_partial_sell_pct": "L_PARTIAL_SELL_PCT",
    "l_trail_pct": "L_TRAIL_PCT", "l_trail_activate_pct": "L_TRAIL_ACTIVATE_PCT",
    "l_time_limit_min": "L_TIME_LIMIT_MINUTES",
}

for json_key, var_name in param_to_var.items():
    if json_key in params:
        val = params[json_key]
        if isinstance(val, str):
            overrides.append(f'{var_name} = "{val}"')
        else:
            overrides.append(f"{var_name} = {val}")

# Strategy enable/disable — set min_gap to 9999 for disabled strategies
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
            overrides.append("R_DAY1_MIN_GAP = 9999.0")
    else:
        if not params.get(f"enable_{s}", True):
            overrides.append(f"{_gap_keys[s]} = 9999.0")

# Priority
priority_dict = {s.upper(): params[f"priority_{s}"] for s in ALL_STRATS}
overrides.append(f"STRAT_PRIORITY = {priority_dict}")

override_block = "\n# --- Optuna params injected by run_backtest_from_json.py ---\n"
override_block += "\n".join(overrides)
override_block += "\n# --- End injection ---\n\n"

# Inject overrides right before __main__
source = source.replace(
    'if __name__ == "__main__":',
    override_block + 'if __name__ == "__main__":'
)

exec(compile(source, "test_green_candle_combined.py", "exec"))
