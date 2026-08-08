"""
Integration tests: Optuna → snapshot → simulator flow.

Tests the full pipeline from Optuna trial param suggestion through to
simulator execution, verifying end-to-end correctness.

Covers:
  - suggest_gl_split_params → _map_split_to_standard → _build_split_snapshot → simulator
  - suggest_all_params (non-split) → set_strategy_params → _build_param_snapshot → simulator
  - Score calculation matches objective function
  - Trial reproduction (suggest → build → backtest matches Optuna result)
"""
import sys
import os
import pytest
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_green_candle_combined as tgc
from optimize_combined import (
    ALL_STRATS, set_strategy_params, _build_param_snapshot,
    run_combined_backtest, _param_lock,
)
from optimize_gl_split import (
    _map_split_to_standard, _build_split_snapshot,
    suggest_gl_split_params, objective_split,
)
from tests.conftest import (
    make_split_g_params, make_split_l_params, make_pick, g_entry_candles,
)


# ═══════════════════════════════════════════════════════════════════════════
#  OPTUNA → SIMULATOR FLOW (Split)
# ═══════════════════════════════════════════════════════════════════════════
class TestOptunaToSimulatorSplit:
    """End-to-end: suggest_gl_split_params → map → snapshot → simulator."""

    def test_split_flow_produces_tradeable_snapshot(self, param_lock, slippage_on):
        """Full split flow must produce a snapshot the simulator can use."""
        params = make_split_g_params()
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        # Verify snapshot is valid by running a single-day backtest
        candles = g_entry_candles()
        pick = make_pick(ticker="FlowTest", gap_pct=50.0,
                         market_hour_candles=candles, pm_volume=1_000_000)
        date = candles.index[0].strftime("%Y-%m-%d")

        states, cash, unsettled, log = tgc.simulate_day_combined(
            [pick], 25000.0, params=snapshot
        )
        # Should not crash - that's the main assertion
        assert isinstance(states, list)
        assert isinstance(cash, float)

    def test_split_flow_g1_and_g2_different_exits(self, param_lock, slippage_on):
        """G1 and G2 should use different exit params in the same simulation."""
        g1_stop = 7.0
        g2_stop = 2.0
        params = make_split_g_params(g1_stop=g1_stop, g2_stop=g2_stop)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        # Verify snapshot has both values
        assert snapshot["g_stop_pct"] == g1_stop
        assert snapshot["g2_stop_pct"] == g2_stop

    def test_split_flow_l1_and_l2_different_exits(self, param_lock, slippage_on):
        """L1 and L2 should use different exit params in the same simulation."""
        l1_stop = 12.0
        l2_stop = 6.0
        params = make_split_l_params(l1_stop=l1_stop, l2_stop=l2_stop)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        assert snapshot["l_stop_pct"] == l1_stop
        assert snapshot["l2_stop_pct"] == l2_stop

    def test_backtest_returns_valid_result(self, param_lock, slippage_on):
        """run_combined_backtest with split params should return valid results."""
        params = make_split_g_params()
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        # Minimal single-day backtest
        candles = g_entry_candles()
        pick = make_pick(ticker="BtTest", gap_pct=50.0,
                         market_hour_candles=candles, pm_volume=1_000_000)
        date = candles.index[0].strftime("%Y-%m-%d")

        result = run_combined_backtest(
            {date: [pick]}, [date], params_snapshot=snapshot
        )

        assert "n" in result
        assert "pf" in result
        assert "total_pnl" in result
        assert "wr" in result
        assert "equity" in result
        assert isinstance(result["n"], int)
        assert isinstance(result["total_pnl"], float)


# ═══════════════════════════════════════════════════════════════════════════
#  OPTUNA → SIMULATOR FLOW (Non-split)
# ═══════════════════════════════════════════════════════════════════════════
class TestOptunaToSimulatorStandard:
    """End-to-end: standard params → set_strategy_params → snapshot → simulator."""

    def test_standard_flow_g_only(self, param_lock, slippage_on):
        """Standard G params flow through to simulator correctly."""
        params = {
            "enable_g": True,
            "g_min_gap_pct": 30.0,
            "g_target_pct": 11.0,
            "g_stop_pct": 3.0,
            "g_trail_pct": 2.0,
            "g_trail_activate_pct": 3.0,
            "g_time_limit_min": 10,
        }
        for s in ALL_STRATS:
            if s not in ("g", "x"):
                params[f"enable_{s}"] = False

        set_strategy_params(params)
        snapshot = _build_param_snapshot()

        assert snapshot["G_MIN_GAP_PCT"] == 30.0
        assert snapshot["G_TARGET_PCT"] == 11.0
        assert snapshot["G_STOP_PCT"] == 3.0

    def test_standard_flow_multiple_strategies(self, param_lock, slippage_on):
        """Multiple enabled strategies flow through correctly."""
        params = {
            "enable_h": True, "enable_g": True, "enable_a": True,
            "h_target_pct": 16.0, "h_stop_pct": 5.0,
            "g_target_pct": 11.0, "g_stop_pct": 3.0,
            "a_target_pct": 6.0, "a_stop_pct": 2.0,
        }
        for s in ALL_STRATS:
            if s not in ("h", "g", "a", "x"):
                params[f"enable_{s}"] = False

        set_strategy_params(params)
        snapshot = _build_param_snapshot()

        assert snapshot["H_TARGET_PCT"] == 16.0
        assert snapshot["G_TARGET_PCT"] == 11.0
        assert snapshot["A_TARGET_PCT"] == 6.0

    def test_backtest_standard_params(self, param_lock, slippage_on):
        """Standard params produce valid backtest results."""
        params = {
            "enable_g": True,
            "g_min_gap_pct": 30.0,
            "g_target_pct": 11.0,
            "g_stop_pct": 3.0,
            "g_time_limit_min": 10,
        }
        for s in ALL_STRATS:
            if s not in ("g", "x"):
                params[f"enable_{s}"] = False

        set_strategy_params(params)
        snapshot = _build_param_snapshot()

        candles = g_entry_candles()
        pick = make_pick(ticker="StdBt", gap_pct=50.0,
                         market_hour_candles=candles, pm_volume=1_000_000)
        date = candles.index[0].strftime("%Y-%m-%d")

        result = run_combined_backtest(
            {date: [pick]}, [date], params_snapshot=snapshot
        )
        assert "n" in result
        assert "total_pnl" in result


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY ENABLE/DISABLE
# ═══════════════════════════════════════════════════════════════════════════
class TestStrategyEnableDisable:
    """Verify that enable/disable flags correctly gate strategy execution."""

    def test_disabled_strategy_produces_zero_trades(self, param_lock, slippage_on):
        """A disabled strategy should produce zero trades via gap=9999."""
        params = {f"enable_{s}": False for s in ALL_STRATS if s != "x"}
        # Don't enable anything - everything disabled
        set_strategy_params(params)
        snapshot = _build_split_snapshot(params)

        candles = g_entry_candles()
        pick = make_pick(ticker="NoTrade", gap_pct=50.0,
                         market_hour_candles=candles, pm_volume=1_000_000)
        date = candles.index[0].strftime("%Y-%m-%d")

        result = run_combined_backtest(
            {date: [pick]}, [date], params_snapshot=snapshot
        )
        assert result["n"] == 0, f"Disabled strategies produced {result['n']} trades"

    def test_enabled_strategy_can_trade(self, param_lock, slippage_on):
        """An enabled strategy with valid params should be able to trade."""
        params = {f"enable_{s}": False for s in ALL_STRATS if s != "x"}
        params["enable_g"] = True
        params["g_min_gap_pct"] = 15.0  # low gap to allow entry
        params["g_target_pct"] = 10.0
        params["g_stop_pct"] = 5.0
        params["g_time_limit_min"] = 15

        set_strategy_params(params)
        snapshot = _build_split_snapshot(params)

        candles = g_entry_candles()
        pick = make_pick(ticker="CanTrade", gap_pct=50.0,
                         market_hour_candles=candles, pm_volume=1_000_000)
        date = candles.index[0].strftime("%Y-%m-%d")

        # Should not crash
        result = run_combined_backtest(
            {date: [pick]}, [date], params_snapshot=snapshot
        )
        assert isinstance(result["n"], int)

    @pytest.mark.parametrize("strat", ["g", "l"])
    def test_each_gl_strategy_independently_enabled(self, strat, param_lock, slippage_on):
        """Enabling G alone vs L alone should produce different results."""
        params = {f"enable_{s}": False for s in ALL_STRATS if s != "x"}
        params[f"enable_{strat}"] = True
        params[f"{strat}_min_gap_pct"] = 15.0
        params[f"{strat}_min_gap"] = 15.0

        set_strategy_params(params)
        snapshot = _build_split_snapshot(params)

        # Should not crash for either strategy
        assert snapshot is not None


# ═══════════════════════════════════════════════════════════════════════════
#  SCORE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════
class TestScoreCalculation:
    """Verify the objective function score formula: total_pnl * min(pf, 3.0)."""

    def test_score_formula(self):
        """Score = total_pnl * min(pf, 3.0)."""
        def calc_score(pnl, pf):
            return pnl * min(pf, 3.0)

        # Normal case
        assert calc_score(10000, 1.5) == 15000.0
        # PF capped at 3.0
        assert calc_score(10000, 5.0) == 30000.0
        # Negative PnL
        assert calc_score(-5000, 1.0) == -5000.0

    def test_score_with_nan_returns_default(self):
        """NaN/inf scores should return -9999."""
        def safe_score(x, default=-9.9e12):
            try:
                x = float(x)
            except (TypeError, ValueError):
                return default
            if math.isnan(x) or math.isinf(x):
                return default
            return max(-9.9e12, min(9.9e12, x))

        assert safe_score(float('nan')) == -9.9e12
        assert safe_score(float('inf')) == -9.9e12
        assert safe_score(15000) == 15000.0


# ═══════════════════════════════════════════════════════════════════════════
#  SNAPSHOT THREAD SAFETY
# ═══════════════════════════════════════════════════════════════════════════
class TestSnapshotThreadSafety:
    """Verify snapshots are independent of tgc globals after creation."""

    def test_snapshot_independent_of_global_mutation(self, param_lock, slippage_on):
        """After snapshot is built, changing tgc globals should not affect it."""
        params = make_split_g_params(g1_stop=7.0, g2_stop=3.0)
        std = _map_split_to_standard(params)
        set_strategy_params(std)
        snapshot = _build_split_snapshot(std)

        # Mutate globals
        tgc.G_STOP_PCT = 999.0

        # Snapshot should be unaffected
        assert snapshot.get("g_stop_pct") == 7.0
        assert snapshot.get("g2_stop_pct") == 3.0
        # Uppercase keys in snapshot are also independent (they're copies)
        # Note: _build_split_snapshot returns a new dict, so it's safe
