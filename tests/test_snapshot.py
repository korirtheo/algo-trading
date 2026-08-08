"""
Unit tests for snapshot building, param mapping, and enable/disable logic.

Tests the data pipeline: suggest_params -> map_to_standard -> build_snapshot
ensuring all required keys are present and correct at each stage.

This catches the class of bugs where:
  - Snapshot is missing keys the simulator reads
  - Split params (G1/G2, L1/L2) aren't propagated to snapshot
  - Disabled strategies still have non-9999 gap values
  - Param mapping between split and standard names is wrong
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_green_candle_combined as tgc
from optimize_combined import (
    ALL_STRATS, set_strategy_params, _build_param_snapshot, _param_lock,
)
from optimize_gl_split import (
    _map_split_to_standard, _build_split_snapshot,
)
from tests.conftest import (
    make_split_g_params, make_split_l_params, make_hgaf_params,
)


# ═══════════════════════════════════════════════════════════════════════════
#  SNAPSHOT KEY COVERAGE
# ═══════════════════════════════════════════════════════════════════════════
class TestSnapshotKeyCoverage:
    """Verify that snapshots contain every key the simulator reads."""

    def test_base_snapshot_has_all_simulator_keys(self):
        """_build_param_snapshot() must contain every uppercase key used by
        simulate_day_combined when params=None (the fallback path)."""
        snapshot = _build_param_snapshot()
        # The simulator's params=None path snapshots these names:
        required = [
            "H_MIN_GAP_PCT", "G_MIN_GAP_PCT", "A_MIN_GAP_PCT", "F_MIN_GAP_PCT",
            "G_TARGET_PCT", "G_TARGET2_PCT", "G_PARTIAL_SELL_PCT",
            "G_TIME_LIMIT_MINUTES", "G_STOP_PCT", "G_TRAIL_PCT", "G_TRAIL_ACTIVATE_PCT",
            "L_STOP_PCT", "L_TRAIL_PCT", "L_TRAIL_ACTIVATE_PCT", "L_TIME_LIMIT_MINUTES",
            "L_PARTIAL_SELL_PCT", "L_TIER1_FLOAT", "L_TIER2_FLOAT",
            "L_TIER1_TARGET1_PCT", "L_TIER1_TARGET2_PCT",
            "L_TIER2_TARGET1_PCT", "L_TIER2_TARGET2_PCT",
            "L_TIER3_TARGET1_PCT", "L_TIER3_TARGET2_PCT",
            "L_MIN_GAP_PCT", "L_MAX_FLOAT", "L_VOL_SURGE_MULT",
            "L_MIN_PRICE_ACCEL_PCT", "L_EARLIEST_CANDLE", "L_LATEST_CANDLE",
            "D_STOP_PCT", "D_TRAIL_PCT", "D_TRAIL_ACTIVATE_PCT",
            "D_TARGET1_PCT", "D_TARGET2_PCT", "D_TIME_LIMIT_MINUTES",
            "D_PARTIAL_SELL_PCT",
            "V_STOP_PCT", "V_TRAIL_PCT", "V_TRAIL_ACTIVATE_PCT",
            "V_TARGET1_PCT", "V_TARGET2_PCT", "V_TIME_LIMIT_MINUTES",
            "V_PARTIAL_SELL_PCT",
            "M_STOP_PCT", "M_TRAIL_PCT", "M_TRAIL_ACTIVATE_PCT",
            "M_TARGET1_PCT", "M_TIME_LIMIT_MINUTES", "M_PARTIAL_SELL_PCT",
            "P_STOP_PCT", "P_TRAIL_PCT", "P_TRAIL_ACTIVATE_PCT",
            "P_TARGET1_PCT", "P_TARGET2_PCT", "P_TIME_LIMIT_MINUTES",
            "P_PARTIAL_SELL_PCT",
            "R_STOP_PCT", "R_TRAIL_PCT", "R_TRAIL_ACTIVATE_PCT",
            "R_TARGET1_PCT", "R_TIME_LIMIT_MINUTES",
            "W_STOP_PCT", "W_TRAIL_PCT", "W_TRAIL_ACTIVATE_PCT",
            "W_TARGET_PCT",
            "O_STOP_PCT", "O_TRAIL_PCT", "O_TRAIL_ACTIVATE_PCT",
            "O_TARGET1_PCT", "O_TARGET2_PCT", "O_TIME_LIMIT_MINUTES",
            "B_STOP_PCT", "B_TRAIL_PCT", "B_TRAIL_ACTIVATE_PCT",
            "B_TARGET1_PCT", "B_TARGET2_PCT", "B_TIME_LIMIT_MINUTES",
            "K_STOP_PCT", "K_TRAIL_PCT", "K_TRAIL_ACTIVATE_PCT",
            "K_TARGET1_PCT", "K_TARGET2_PCT", "K_TIME_LIMIT_MINUTES",
            "C_STOP_PCT", "C_TRAIL_PCT", "C_TRAIL_ACTIVATE_PCT",
            "C_TARGET1_PCT", "C_TARGET2_PCT", "C_TIME_LIMIT_MINUTES",
            "S_STOP_PCT", "S_TRAIL_PCT", "S_TRAIL_ACTIVATE_PCT",
            "S_TARGET1_PCT", "S_TARGET2_PCT", "S_TIME_LIMIT_MINUTES",
            "E_STOP_PCT", "E_TRAIL_PCT", "E_TRAIL_ACTIVATE_PCT",
            "E_TARGET1_PCT", "E_TARGET2_PCT", "E_TIME_LIMIT_MINUTES",
            "I_STOP_PCT", "I_TRAIL_PCT", "I_TRAIL_ACTIVATE_PCT",
            "I_TARGET1_PCT", "I_TARGET2_PCT", "I_TIME_LIMIT_MINUTES",
            "J_STOP_PCT", "J_TRAIL_PCT", "J_TRAIL_ACTIVATE_PCT",
            "J_TARGET1_PCT", "J_TARGET2_PCT", "J_TIME_LIMIT_MINUTES",
            "N_STOP_PCT", "N_TRAIL_PCT", "N_TRAIL_ACTIVATE_PCT",
            "N_TARGET1_PCT", "N_TARGET2_PCT", "N_TIME_LIMIT_MINUTES",
            "EOD_EXIT_MINUTES", "VOL_CAP_PCT",
        ]
        for key in required:
            assert key in snapshot, f"Snapshot missing required key: {key}"


# ═══════════════════════════════════════════════════════════════════════════
#  SPLIT SNAPSHOT: G1/G2/L1/L2 KEYS
# ═══════════════════════════════════════════════════════════════════════════
class TestSplitSnapshot:
    """Verify _build_split_snapshot includes all split keys needed by simulator."""

    def test_split_snapshot_has_base_g_keys(self, param_lock, slippage_on):
        """G1 exit logic reads g_stop_pct, g_trail_pct etc. from entry_params.
        These must be in the snapshot."""
        params = make_split_g_params(g1_stop=5.0, g1_trail=3.0, g1_trail_act=2.0,
                                      g1_target=10.0, g1_time=15)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        # Base G keys (used by G1 exit logic)
        assert "g_stop_pct" in snapshot, "Snapshot missing g_stop_pct"
        assert "g_trail_pct" in snapshot, "Snapshot missing g_trail_pct"
        assert "g_trail_activate_pct" in snapshot, "Snapshot missing g_trail_activate_pct"
        assert "g_target_pct" in snapshot, "Snapshot missing g_target_pct"
        assert "g_time_limit_min" in snapshot, "Snapshot missing g_time_limit_min"

        # G2 keys (used by G2 exit logic)
        assert "g2_stop_pct" in snapshot, "Snapshot missing g2_stop_pct"
        assert "g2_trail_pct" in snapshot, "Snapshot missing g2_trail_pct"
        assert "g2_trail_activate_pct" in snapshot, "Snapshot missing g2_trail_activate_pct"
        assert "g2_target_pct" in snapshot, "Snapshot missing g2_target_pct"
        assert "g2_time_limit_min" in snapshot, "Snapshot missing g2_time_limit_min"

    def test_split_snapshot_has_base_l_keys(self, param_lock, slippage_on):
        """L1 exit logic reads l_stop_pct, l_trail_pct etc. from entry_params.
        These must be in the snapshot."""
        params = make_split_l_params(l1_stop=10.0, l1_trail=3.0, l1_trail_act=2.0,
                                      l1_time=60,
                                      l2_stop=8.0, l2_trail=2.0, l2_trail_act=3.0,
                                      l2_time=45)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        # Base L keys (used by L1 exit logic)
        assert "l_stop_pct" in snapshot, "Snapshot missing l_stop_pct"
        assert "l_trail_pct" in snapshot, "Snapshot missing l_trail_pct"
        assert "l_trail_activate_pct" in snapshot, "Snapshot missing l_trail_activate_pct"
        assert "l_time_limit_min" in snapshot, "Snapshot missing l_time_limit_min"
        assert "l_partial_sell_pct" in snapshot, "Snapshot missing l_partial_sell_pct"

        # L2 keys (used by L2 exit logic)
        assert "l2_stop_pct" in snapshot, "Snapshot missing l2_stop_pct"
        assert "l2_trail_pct" in snapshot, "Snapshot missing l2_trail_pct"
        assert "l2_trail_activate_pct" in snapshot, "Snapshot missing l2_trail_activate_pct"
        assert "l2_time_limit_min" in snapshot, "Snapshot missing l2_time_limit_min"
        assert "l2_partial_sell_pct" in snapshot, "Snapshot missing l2_partial_sell_pct"

    def test_split_snapshot_g1_values_match_params(self, param_lock, slippage_on):
        """G1 exit values in snapshot must match the g1_* input params."""
        params = make_split_g_params(g1_stop=7.0, g1_trail=4.0, g1_trail_act=3.0,
                                      g1_target=15.0, g1_time=20)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        # Base G keys should have G1 values (since _map_split_to_standard maps g1->g)
        assert snapshot["g_stop_pct"] == 7.0
        assert snapshot["g_trail_pct"] == 4.0
        assert snapshot["g_trail_activate_pct"] == 3.0
        assert snapshot["g_target_pct"] == 15.0
        assert snapshot["g_time_limit_min"] == 20

    def test_split_snapshot_g2_values_match_params(self, param_lock, slippage_on):
        """G2 exit values in snapshot must match the g2_* input params."""
        params = make_split_g_params(g2_stop=2.0, g2_trail=1.0, g2_trail_act=5.0,
                                      g2_target=6.0, g2_time=8)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert snapshot["g2_stop_pct"] == 2.0
        assert snapshot["g2_trail_pct"] == 1.0
        assert snapshot["g2_trail_activate_pct"] == 5.0
        assert snapshot["g2_target_pct"] == 6.0
        assert snapshot["g2_time_limit_min"] == 8

    def test_split_snapshot_l1_values_match_params(self, param_lock, slippage_on):
        """L1 exit values in snapshot must match the l1_* input params."""
        params = make_split_l_params(l1_stop=12.0, l1_trail=4.0, l1_trail_act=3.0,
                                      l1_time=75)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert snapshot["l_stop_pct"] == 12.0
        assert snapshot["l_trail_pct"] == 4.0
        assert snapshot["l_trail_activate_pct"] == 3.0
        assert snapshot["l_time_limit_min"] == 75

    def test_split_snapshot_l2_values_match_params(self, param_lock, slippage_on):
        """L2 exit values in snapshot must match the l2_* input params."""
        params = make_split_l_params(l2_stop=6.0, l2_trail=2.0, l2_trail_act=4.0,
                                      l2_time=30)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert snapshot["l2_stop_pct"] == 6.0
        assert snapshot["l2_trail_pct"] == 2.0
        assert snapshot["l2_trail_activate_pct"] == 4.0
        assert snapshot["l2_time_limit_min"] == 30


# ═══════════════════════════════════════════════════════════════════════════
#  DISABLED STRATEGIES: GAP = 9999
# ═══════════════════════════════════════════════════════════════════════════
class TestDisabledStrategies:
    """Verify that disabled strategies have gap=9999 in the snapshot."""

    def test_disabled_strategies_gap_9999_g_only(self, param_lock, slippage_on):
        """When only G is enabled, all other strategies must have gap=9999."""
        params = make_split_g_params()
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        gap_keys = {
            "h": "H_MIN_GAP_PCT", "a": "A_MIN_GAP_PCT", "f": "F_MIN_GAP_PCT",
            "d": "D_MIN_GAP_PCT", "v": "V_MIN_GAP_PCT", "m": "M_MIN_GAP_PCT",
            "r": "R_DAY1_MIN_GAP", "w": "W_MIN_GAP_PCT", "o": "O_MIN_GAP_PCT",
            "b": "B_MIN_GAP_PCT", "k": "K_MIN_GAP_PCT", "c": "C_MIN_GAP_PCT",
            "s": "S_MIN_GAP_PCT", "e": "E_MIN_GAP_PCT", "i": "I_MIN_GAP_PCT",
            "j": "J_MIN_GAP_PCT", "n": "N_MIN_GAP_PCT",
        }

        for strat, gap_key in gap_keys.items():
            if std.get(f"enable_{strat}", False):
                continue  # skip enabled strategies
            val = snapshot.get(gap_key)
            assert val == 9999.0, (
                f"Strategy {strat} is disabled but {gap_key}={val} != 9999"
            )

    def test_disabled_strategies_gap_9999_l_only(self, param_lock, slippage_on):
        """When only L is enabled, all other strategies must have gap=9999."""
        params = make_split_l_params()
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        gap_keys = {
            "h": "H_MIN_GAP_PCT", "a": "A_MIN_GAP_PCT", "f": "F_MIN_GAP_PCT",
            "d": "D_MIN_GAP_PCT", "v": "V_MIN_GAP_PCT", "m": "M_MIN_GAP_PCT",
            "r": "R_DAY1_MIN_GAP", "w": "W_MIN_GAP_PCT", "o": "O_MIN_GAP_PCT",
            "b": "B_MIN_GAP_PCT", "k": "K_MIN_GAP_PCT", "c": "C_MIN_GAP_PCT",
            "s": "S_MIN_GAP_PCT", "e": "E_MIN_GAP_PCT", "i": "I_MIN_GAP_PCT",
            "j": "J_MIN_GAP_PCT", "n": "N_MIN_GAP_PCT",
        }

        for strat, gap_key in gap_keys.items():
            if std.get(f"enable_{strat}", False):
                continue
            val = snapshot.get(gap_key)
            assert val == 9999.0, (
                f"Strategy {strat} is disabled but {gap_key}={val} != 9999"
            )

    def test_disabled_strategies_gap_9999_g_and_l(self, param_lock, slippage_on):
        """When G+L are enabled, all others must have gap=9999."""
        params = make_split_g_params()
        params.update(make_split_l_params())
        params["enable_g"] = True
        params["enable_l"] = True

        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        gap_keys = {
            "h": "H_MIN_GAP_PCT", "a": "A_MIN_GAP_PCT", "f": "F_MIN_GAP_PCT",
            "d": "D_MIN_GAP_PCT", "v": "V_MIN_GAP_PCT", "m": "M_MIN_GAP_PCT",
            "r": "R_DAY1_MIN_GAP", "w": "W_MIN_GAP_PCT", "o": "O_MIN_GAP_PCT",
            "b": "B_MIN_GAP_PCT", "k": "K_MIN_GAP_PCT", "c": "C_MIN_GAP_PCT",
            "s": "S_MIN_GAP_PCT", "e": "E_MIN_GAP_PCT", "i": "I_MIN_GAP_PCT",
            "j": "J_MIN_GAP_PCT", "n": "N_MIN_GAP_PCT",
        }

        for strat, gap_key in gap_keys.items():
            if std.get(f"enable_{strat}", False):
                continue
            val = snapshot.get(gap_key)
            assert val == 9999.0, (
                f"Strategy {strat} is disabled but {gap_key}={val} != 9999"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  PARAM MAPPING: _map_split_to_standard
# ═══════════════════════════════════════════════════════════════════════════
class TestParamMapping:
    """Verify _map_split_to_standard correctly maps G1->G, L1->L for module globals."""

    def test_g1_maps_to_g_standard(self):
        """_map_split_to_standard must map g1_* values to g_* keys."""
        params = make_split_g_params(g1_stop=7.0, g1_trail=4.0, g1_trail_act=3.0,
                                      g1_target=15.0, g1_time=20)
        std = _map_split_to_standard(params)

        assert std["g_stop_pct"] == 7.0, f"g_stop_pct={std['g_stop_pct']}, expected 7.0"
        assert std["g_trail_pct"] == 4.0
        assert std["g_trail_activate_pct"] == 3.0
        assert std["g_target_pct"] == 15.0
        assert std["g_time_limit_min"] == 20

    def test_l1_maps_to_l_standard(self):
        """_map_split_to_standard must map l1_* values to l_* keys."""
        params = make_split_l_params(l1_stop=12.0, l1_trail=4.0, l1_trail_act=3.0,
                                      l1_time=75)
        std = _map_split_to_standard(params)

        assert std["l_stop_pct"] == 12.0
        assert std["l_trail_pct"] == 4.0
        assert std["l_trail_activate_pct"] == 3.0
        assert std["l_time_limit_min"] == 75

    def test_split_keys_preserved_in_output(self):
        """_map_split_to_standard must preserve the original g1_*/g2_*/l1_*/l2_* keys."""
        params = make_split_g_params()
        std = _map_split_to_standard(params)

        for key in ["g1_stop_pct", "g2_stop_pct", "g1_trail_pct", "g2_trail_pct"]:
            assert key in std, f"Split key {key} lost after mapping"

    def test_enable_flags_preserved_in_mapping(self):
        """Enable flags must pass through _map_split_to_standard."""
        params = make_split_g_params()
        std = _map_split_to_standard(params)

        assert std.get("enable_g") is True
        assert std.get("enable_l") is False
        # x is not a real strategy (no enable flag) — skip it
        for s in "vhafdrwobkcseijn":
            assert std.get(f"enable_{s}") is False, f"enable_{s} should be False"

    def test_g2_values_not_overwritten_by_g1_mapping(self):
        """Mapping g1->g must NOT overwrite g2 values."""
        params = make_split_g_params(g1_stop=7.0, g2_stop=2.0)
        std = _map_split_to_standard(params)

        assert std["g_stop_pct"] == 7.0  # G1 mapped to G
        assert std["g2_stop_pct"] == 2.0  # G2 preserved


# ═══════════════════════════════════════════════════════════════════════════
#  UPPERCASE SNAPSHOT MATCHES set_strategy_params
# ═══════════════════════════════════════════════════════════════════════════
class TestUppercaseSnapshot:
    """Verify _build_param_snapshot captures values set by set_strategy_params."""

    def test_g_params_appear_in_uppercase_snapshot(self, param_lock, slippage_on):
        """After set_strategy_params, _build_param_snapshot should have correct G values."""
        params = {
            "enable_g": True,
            "g_min_gap_pct": 35.0,
            "g_target_pct": 12.0,
            "g_stop_pct": 4.0,
            "g_trail_pct": 2.0,
            "g_trail_activate_pct": 3.0,
            "g_time_limit_min": 10,
        }
        set_strategy_params(params)
        snapshot = _build_param_snapshot()

        assert snapshot["G_MIN_GAP_PCT"] == 35.0
        assert snapshot["G_TARGET_PCT"] == 12.0
        assert snapshot["G_STOP_PCT"] == 4.0
        assert snapshot["G_TRAIL_PCT"] == 2.0
        assert snapshot["G_TRAIL_ACTIVATE_PCT"] == 3.0
        assert snapshot["G_TIME_LIMIT_MINUTES"] == 10

    def test_l_params_appear_in_uppercase_snapshot(self, param_lock, slippage_on):
        """After set_strategy_params, _build_param_snapshot should have correct L values."""
        params = {
            "enable_l": True,
            "l_min_gap": 25.0,
            "l_stop_pct": 8.0,
            "l_trail_pct": 3.0,
            "l_trail_activate_pct": 2.0,
            "l_time_limit_min": 60,
            "l_partial_sell_pct": 25.0,
            "l_tier1_target1_pct": 20.0,
            "l_tier1_target2_pct": 40.0,
            "l_tier2_target1_pct": 12.0,
            "l_tier2_target2_pct": 30.0,
            "l_tier3_target1_pct": 8.0,
            "l_tier3_target2_pct": 20.0,
        }
        set_strategy_params(params)
        snapshot = _build_param_snapshot()

        assert snapshot["L_MIN_GAP_PCT"] == 25.0
        assert snapshot["L_STOP_PCT"] == 8.0
        assert snapshot["L_TRAIL_PCT"] == 3.0
        assert snapshot["L_TRAIL_ACTIVATE_PCT"] == 2.0
        assert snapshot["L_TIME_LIMIT_MINUTES"] == 60
        assert snapshot["L_PARTIAL_SELL_PCT"] == 25.0
        assert snapshot["L_TIER1_TARGET1_PCT"] == 20.0
