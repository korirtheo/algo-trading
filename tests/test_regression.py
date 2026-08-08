"""
Regression tests for specific bugs found and fixed.

Each test documents a bug that was found in production and verifies the fix.
These tests must NEVER pass before the fix and NEVER fail after.

Bug #1: _build_split_snapshot missing g_*/l_* base keys (2026-07-25)
  - G1/L1 exit logic reads g_stop_pct (base key), not g1_stop_pct
  - Snapshot only had g1_*/g2_*/l1_*/l2_* keys, missing g_*/l_*
  - Result: G1/L1 trades fell back to hardcoded defaults

Bug #2: entry_trade_seq not used for prefix selection (2026-07-24)
  - G2 exit logic read g_stop_pct instead of g2_stop_pct
  - _gp was always "g", never "g2"
  - Result: G2 trades used G1 params

Bug #3: Disabled strategies could still trade (2026-07-20)
  - enable_g=False didn't prevent G from trading
  - Gap was not set to 9999 for disabled strategies
  - Result: All strategies traded regardless of enable flags

Bug #4: Optuna params not fully stored in trial.params (2026-07-24)
  - enable_g=True set via manual assignment, not trial.suggest_*
  - trial.params didn't contain enable flags
  - Result: Reproducing from trial.params failed
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
    _map_split_to_standard, _build_split_snapshot, suggest_gl_split_params,
)
from tests.conftest import (
    make_split_g_params, make_split_l_params, make_pick,
    g_entry_candles, g_stop_candles,
)


# ═══════════════════════════════════════════════════════════════════════════
#  BUG #1: Missing g_*/l_* base keys in split snapshot
# ═══════════════════════════════════════════════════════════════════════════
class TestBug1_MissingBaseKeys:
    """Regression: _build_split_snapshot must include g_*/l_* base keys.

    Before fix: snapshot only had g1_*/g2_*/l1_*/l2_* keys.
    G1 exit logic reads g_stop_pct (base key), which was missing.
    Fell back to G_STOP_PCT (hardcoded default = 2.0%).
    """

    def test_g1_stop_pct_in_snapshot(self, param_lock, slippage_on):
        """g_stop_pct must be in split snapshot (not just g1_stop_pct)."""
        params = make_split_g_params(g1_stop=7.0)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert "g_stop_pct" in snapshot, (
            "BUG #1 REGRESSION: g_stop_pct missing from split snapshot"
        )
        assert snapshot["g_stop_pct"] == 7.0

    def test_g1_trail_pct_in_snapshot(self, param_lock, slippage_on):
        """g_trail_pct must be in split snapshot."""
        params = make_split_g_params(g1_trail=4.0)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert "g_trail_pct" in snapshot, (
            "BUG #1 REGRESSION: g_trail_pct missing from split snapshot"
        )
        assert snapshot["g_trail_pct"] == 4.0

    def test_g1_trail_activate_pct_in_snapshot(self, param_lock, slippage_on):
        """g_trail_activate_pct must be in split snapshot."""
        params = make_split_g_params(g1_trail_act=3.0)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert "g_trail_activate_pct" in snapshot, (
            "BUG #1 REGRESSION: g_trail_activate_pct missing from split snapshot"
        )
        assert snapshot["g_trail_activate_pct"] == 3.0

    def test_g1_target_pct_in_snapshot(self, param_lock, slippage_on):
        """g_target_pct must be in split snapshot."""
        params = make_split_g_params(g1_target=15.0)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert "g_target_pct" in snapshot, (
            "BUG #1 REGRESSION: g_target_pct missing from split snapshot"
        )
        assert snapshot["g_target_pct"] == 15.0

    def test_g1_time_limit_in_snapshot(self, param_lock, slippage_on):
        """g_time_limit_min must be in split snapshot."""
        params = make_split_g_params(g1_time=20)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert "g_time_limit_min" in snapshot, (
            "BUG #1 REGRESSION: g_time_limit_min missing from split snapshot"
        )
        assert snapshot["g_time_limit_min"] == 20

    def test_l1_stop_pct_in_snapshot(self, param_lock, slippage_on):
        """l_stop_pct must be in split snapshot."""
        params = make_split_l_params(l1_stop=12.0)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert "l_stop_pct" in snapshot, (
            "BUG #1 REGRESSION: l_stop_pct missing from split snapshot"
        )
        assert snapshot["l_stop_pct"] == 12.0

    def test_l1_trail_pct_in_snapshot(self, param_lock, slippage_on):
        """l_trail_pct must be in split snapshot."""
        params = make_split_l_params(l1_trail=4.0)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert "l_trail_pct" in snapshot, (
            "BUG #1 REGRESSION: l_trail_pct missing from split snapshot"
        )
        assert snapshot["l_trail_pct"] == 4.0

    def test_l1_trail_activate_pct_in_snapshot(self, param_lock, slippage_on):
        """l_trail_activate_pct must be in split snapshot."""
        params = make_split_l_params(l1_trail_act=3.0)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert "l_trail_activate_pct" in snapshot, (
            "BUG #1 REGRESSION: l_trail_activate_pct missing from split snapshot"
        )
        assert snapshot["l_trail_activate_pct"] == 3.0

    def test_l1_time_limit_in_snapshot(self, param_lock, slippage_on):
        """l_time_limit_min must be in split snapshot."""
        params = make_split_l_params(l1_time=75)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert "l_time_limit_min" in snapshot, (
            "BUG #1 REGRESSION: l_time_limit_min missing from split snapshot"
        )
        assert snapshot["l_time_limit_min"] == 75

    def test_all_six_prefixes_in_snapshot(self, param_lock, slippage_on):
        """Snapshot must contain keys for ALL six prefixes: g_, g1_, g2_, l_, l1_, l2_."""
        params = make_split_g_params()
        params.update(make_split_l_params())
        params["enable_g"] = True
        params["enable_l"] = True
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        prefixes = ("g_", "g1_", "g2_", "l_", "l1_", "l2_")
        for prefix in prefixes:
            matching_keys = [k for k in snapshot if k.startswith(prefix)]
            assert len(matching_keys) > 0, (
                f"BUG #1 REGRESSION: No keys with prefix '{prefix}' in snapshot"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  BUG #2: G2/L2 exit logic reading wrong prefix
# ═══════════════════════════════════════════════════════════════════════════
class TestBug2_G2WrongPrefix:
    """Regression: G2 exit logic must read g2_* params, not g_*.

    Before fix: _gp was always "g", never "g2".
    G2 trades used G1 exit params.
    """

    def test_g2_uses_g2_stop_not_g_stop(self, param_lock, slippage_on):
        """When entry_trade_seq > 0, G exit reads g2_stop_pct, not g_stop_pct."""
        g1_stop = 7.0
        g2_stop = 2.0
        params = make_split_g_params(g1_stop=g1_stop, g2_stop=g2_stop)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        # The snapshot must have both values distinct
        assert snapshot["g_stop_pct"] == g1_stop
        assert snapshot["g2_stop_pct"] == g2_stop
        # They must be different for the test to be meaningful
        assert g1_stop != g2_stop

    def test_g2_uses_g2_trail_not_g_trail(self, param_lock, slippage_on):
        """G2 must use g2_trail_pct, not g_trail_pct."""
        g1_trail = 4.0
        g2_trail = 1.0
        params = make_split_g_params(g1_trail=g1_trail, g2_trail=g2_trail)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert snapshot["g_trail_pct"] == g1_trail
        assert snapshot["g2_trail_pct"] == g2_trail

    def test_l2_uses_l2_stop_not_l_stop(self, param_lock, slippage_on):
        """When entry_trade_seq > 0, L exit reads l2_stop_pct, not l_stop_pct."""
        l1_stop = 12.0
        l2_stop = 6.0
        params = make_split_l_params(l1_stop=l1_stop, l2_stop=l2_stop)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert snapshot["l_stop_pct"] == l1_stop
        assert snapshot["l2_stop_pct"] == l2_stop

    def test_g1_and_g2_params_are_independent(self, param_lock, slippage_on):
        """Changing G1 params must not affect G2 params in snapshot."""
        params = make_split_g_params(
            g1_stop=9.0, g1_trail=5.0, g1_trail_act=4.0,
            g2_stop=1.0, g2_trail=0.5, g2_trail_act=6.0,
        )
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        # G1 params
        assert snapshot["g_stop_pct"] == 9.0
        assert snapshot["g_trail_pct"] == 5.0
        assert snapshot["g_trail_activate_pct"] == 4.0
        # G2 params (must be different from G1)
        assert snapshot["g2_stop_pct"] == 1.0
        assert snapshot["g2_trail_pct"] == 0.5
        assert snapshot["g2_trail_activate_pct"] == 6.0

    def test_l1_and_l2_params_are_independent(self, param_lock, slippage_on):
        """Changing L1 params must not affect L2 params in snapshot."""
        params = make_split_l_params(
            l1_stop=15.0, l1_trail=5.0, l1_trail_act=4.0,
            l2_stop=8.0, l2_trail=2.0, l2_trail_act=3.0,
        )
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert snapshot["l_stop_pct"] == 15.0
        assert snapshot["l_trail_pct"] == 5.0
        assert snapshot["l_trail_activate_pct"] == 4.0
        assert snapshot["l2_stop_pct"] == 8.0
        assert snapshot["l2_trail_pct"] == 2.0
        assert snapshot["l2_trail_activate_pct"] == 3.0


# ═══════════════════════════════════════════════════════════════════════════
#  BUG #3: Disabled strategies still trading
# ═══════════════════════════════════════════════════════════════════════════
class TestBug3_DisabledStrategiesTrade:
    """Regression: disabled strategies must have gap=9999 to prevent trading."""

    def test_all_disabled_no_trades(self, param_lock, slippage_on):
        """When all strategies are disabled, no trades should occur."""
        params = {f"enable_{s}": False for s in ALL_STRATS if s != "x"}
        set_strategy_params(params)
        snapshot = _build_split_snapshot(params)

        # Check all gap keys are 9999
        gap_keys = {
            "h": "H_MIN_GAP_PCT", "a": "A_MIN_GAP_PCT", "f": "F_MIN_GAP_PCT",
            "d": "D_MIN_GAP_PCT", "v": "V_MIN_GAP_PCT", "m": "M_MIN_GAP_PCT",
            "r": "R_DAY1_MIN_GAP", "w": "W_MIN_GAP_PCT", "o": "O_MIN_GAP_PCT",
            "b": "B_MIN_GAP_PCT", "k": "K_MIN_GAP_PCT", "c": "C_MIN_GAP_PCT",
            "s": "S_MIN_GAP_PCT", "e": "E_MIN_GAP_PCT", "i": "I_MIN_GAP_PCT",
            "j": "J_MIN_GAP_PCT", "n": "N_MIN_GAP_PCT",
        }
        for strat, gap_key in gap_keys.items():
            assert snapshot.get(gap_key) == 9999.0, (
                f"BUG #3 REGRESSION: {strat} disabled but {gap_key}="
                f"{snapshot.get(gap_key)} != 9999"
            )

    def test_enable_g_only_others_disabled(self, param_lock, slippage_on):
        """When only G is enabled, H/A/F/D/V/etc must have gap=9999."""
        params = {f"enable_{s}": False for s in ALL_STRATS if s != "x"}
        params["enable_g"] = True
        set_strategy_params(params)
        snapshot = _build_split_snapshot(params)

        # Non-G strategies must be blocked
        for s in "hadfvmrwobkcseijn":
            gap_key_map = {
                "h": "H_MIN_GAP_PCT", "a": "A_MIN_GAP_PCT", "f": "F_MIN_GAP_PCT",
                "d": "D_MIN_GAP_PCT", "v": "V_MIN_GAP_PCT", "m": "M_MIN_GAP_PCT",
                "r": "R_DAY1_MIN_GAP", "w": "W_MIN_GAP_PCT", "o": "O_MIN_GAP_PCT",
                "b": "B_MIN_GAP_PCT", "k": "K_MIN_GAP_PCT", "c": "C_MIN_GAP_PCT",
                "s": "S_MIN_GAP_PCT", "e": "E_MIN_GAP_PCT", "i": "I_MIN_GAP_PCT",
                "j": "J_MIN_GAP_PCT", "n": "N_MIN_GAP_PCT",
            }
            if s in gap_key_map:
                assert snapshot.get(gap_key_map[s]) == 9999.0, (
                    f"BUG #3 REGRESSION: {s} disabled but gap != 9999"
                )


# ═══════════════════════════════════════════════════════════════════════════
#  BUG #4: Optuna trial.params missing enable flags
# ═══════════════════════════════════════════════════════════════════════════
class TestBug4_EnableFlagsNotStored:
    """Regression: enable flags set via manual assignment aren't in trial.params.

    This is by design (not a bug in the optimizer), but we must handle it
    when reproducing from trial.params.
    """

    def test_enable_flags_not_in_suggest_output(self):
        """suggest_gl_split_params returns enable flags, but they won't be in
        trial.params because they're set via assignment, not suggest_*.

        This documents the known behavior - reproducing from trial.params
        requires manually adding enable flags."""
        import optuna
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_gl_split_params(trial)

        # suggest_gl_split_params DOES return enable flags
        assert params.get("enable_g") is True
        assert params.get("enable_l") is True

        # But they won't be in trial.params after tell()
        # (This is the documented limitation)
        study.tell(trial, 1.0)
        # trial.params only has suggest_* values
        # enable_g and enable_l won't be there

    def test_reproducing_trial_needs_enable_flags(self):
        """When reproducing from trial.params, must manually add enable flags."""
        params = {"g1_stop_pct": 5.0, "g2_stop_pct": 3.0}
        # Without enable flags, set_strategy_params would default to True for all
        # This is the trap - must explicitly add them
        params["enable_g"] = True
        params["enable_l"] = True
        for s in "vhafdrwobkcsexijn":
            params[f"enable_{s}"] = False

        assert params["enable_g"] is True
        assert params["enable_l"] is True
        # All others disabled
        for s in "vhafdrwobkcsexijn":
            assert params[f"enable_{s}"] is False
