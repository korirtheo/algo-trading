"""
Run 3 isolated A/B tests to measure each change independently.
Uses test_full.py's simulation but suppresses chart generation.
"""
import sys
import os
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

import test_full as tf
from regime_filters import RegimeFilter

DATA_DIRS = tf.DEFAULT_DATA_DIRS

def run_backtest_with_config(label, **overrides):
    """Run full backtest with config overrides, return summary dict."""
    # Save originals
    originals = {}
    for key, val in overrides.items():
        originals[key] = getattr(tf, key)
        setattr(tf, key, val)

    try:
        ALL_DATES, daily_picks = tf.load_all_picks(DATA_DIRS)

        regime = RegimeFilter(
            spy_ma_period=tf.SPY_SMA_PERIOD,
            enable_vix=False,
            enable_spy_trend=True,
            enable_adaptive=False,
        )
        regime.load_data(ALL_DATES[0], ALL_DATES[-1])

        rolling_cash = tf.STARTING_CASH
        all_results = []
        regime_skipped = 0

        for date_str in ALL_DATES:
            should_trade, _, _ = regime.check(date_str)
            if not should_trade:
                regime_skipped += 1
                all_results.append({
                    "date": date_str, "day_pnl": 0.0, "ending_cash": rolling_cash,
                    "starting_cash": rolling_cash, "states": [], "regime_skip": True,
                })
                continue

            picks = daily_picks.get(date_str, [])
            if not picks:
                all_results.append({
                    "date": date_str, "day_pnl": 0.0, "ending_cash": rolling_cash,
                    "starting_cash": rolling_cash, "states": [], "regime_skip": False,
                })
                continue

            starting = rolling_cash
            is_cash_account = rolling_cash < tf.MARGIN_THRESHOLD
            states, ending_cash = tf.simulate_day(picks, rolling_cash, cash_account=is_cash_account)
            day_pnl = sum(s["pnl"] for s in states if s["entry_price"] is not None)
            rolling_cash = ending_cash

            all_results.append({
                "date": date_str, "day_pnl": day_pnl, "ending_cash": rolling_cash,
                "starting_cash": starting, "states": states, "regime_skip": False,
            })

        # Compute stats
        daily_pnls = [r["day_pnl"] for r in all_results]
        total_pnl = sum(daily_pnls)
        final_cash = all_results[-1]["ending_cash"]

        total_trades = 0
        total_winners = 0
        total_losers = 0
        for r in all_results:
            for s in r["states"]:
                if s["entry_price"] is not None:
                    total_trades += 1
                    if s["pnl"] > 0:
                        total_winners += 1
                    else:
                        total_losers += 1

        win_rate = (total_winners / total_trades * 100) if total_trades > 0 else 0
        green_days = sum(1 for p in daily_pnls if p > 0)
        red_days = sum(1 for p in daily_pnls if p <= 0 and p != 0)

        # Non-zero daily pnls for Sharpe
        active_pnls = [p for p in daily_pnls if p != 0]
        std = np.std(active_pnls) if active_pnls else 1.0
        sharpe = (np.mean(active_pnls) / std) * np.sqrt(252) if std > 0 else 0

        # Max drawdown
        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = (cum - peak).min() if len(cum) > 0 else 0

        # Jan 2026 stats
        jan_pnl = sum(r["day_pnl"] for r in all_results if r["date"].startswith("2026-01"))
        jan_trades = sum(
            1 for r in all_results if r["date"].startswith("2026-01")
            for s in r["states"] if s["entry_price"] is not None
        )
        jan_winners = sum(
            1 for r in all_results if r["date"].startswith("2026-01")
            for s in r["states"] if s["entry_price"] is not None and s["pnl"] > 0
        )
        jan_losers = jan_trades - jan_winners

        # Scale-in count
        scale_ins = sum(
            1 for r in all_results for s in r["states"]
            if s["entry_price"] is not None and s.get("scaled_in")
        )

        # Breakeven stop exits
        be_exits = sum(
            1 for r in all_results for s in r["states"]
            if s.get("breakeven_active") and s.get("exit_reason") == "STOP_LOSS"
        )

        # Late entry skips
        late_skips = sum(
            1 for r in all_results for s in r["states"]
            if s.get("exit_reason") == "LATE_ENTRY"
        )

        return {
            "label": label,
            "final_cash": final_cash,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "winners": total_winners,
            "losers": total_losers,
            "win_rate": win_rate,
            "green_days": green_days,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "jan_pnl": jan_pnl,
            "jan_trades": jan_trades,
            "jan_winners": jan_winners,
            "jan_losers": jan_losers,
            "scale_ins": scale_ins,
            "be_exits": be_exits,
            "late_skips": late_skips,
            "regime_skipped": regime_skipped,
        }
    finally:
        # Restore originals
        for key, val in originals.items():
            setattr(tf, key, val)


def print_result(r):
    print(f"\n  {'='*50}")
    print(f"  {r['label']}")
    print(f"  {'='*50}")
    print(f"  Final Cash:    ${r['final_cash']:>15,.2f}")
    print(f"  Total Trades:  {r['total_trades']:>6}  ({r['winners']}W / {r['losers']}L)")
    print(f"  Win Rate:      {r['win_rate']:>6.1f}%")
    print(f"  Green Days:    {r['green_days']}")
    print(f"  Sharpe:        {r['sharpe']:>6.2f}")
    print(f"  Max Drawdown:  ${r['max_dd']:>+15,.2f}")
    print(f"  Scale-ins:     {r['scale_ins']}")
    if r['be_exits'] > 0:
        print(f"  BE Stop Exits: {r['be_exits']}")
    if r['late_skips'] > 0:
        print(f"  Late Skips:    {r['late_skips']}")
    print(f"  --- JAN 2026 ---")
    print(f"  Jan P&L:       ${r['jan_pnl']:>+15,.2f}")
    print(f"  Jan Trades:    {r['jan_trades']}  ({r['jan_winners']}W / {r['jan_losers']}L)")


if __name__ == "__main__":
    print("=" * 60)
    print("  ISOLATED A/B TESTS (each change tested alone)")
    print("=" * 60)

    results = []

    # BASELINE (no scale-in fix, no entry cutoff)
    print("\n[1/5] Running BASELINE...")
    r = run_backtest_with_config(
        "BASELINE (no fixes)",
        BREAKEVEN_STOP_PCT=0.0,
        ENTRY_CUTOFF_MINUTES=0,
        SCALE_IN_TRIGGER_PCT=19.0,
        SCALE_IN_GATE_PARTIAL=True,
    )
    results.append(r)
    print_result(r)

    # TEST A: Scale-in 14% + entry cutoff 2hr
    print("\n[2/5] Running TEST A: Scale-in 14% + cutoff 2hr...")
    r = run_backtest_with_config(
        "A: SI 14% + Cutoff 2hr",
        BREAKEVEN_STOP_PCT=0.0,
        ENTRY_CUTOFF_MINUTES=120,
        SCALE_IN_TRIGGER_PCT=14.0,
        SCALE_IN_GATE_PARTIAL=False,
    )
    results.append(r)
    print_result(r)

    # TEST B: Scale-in 13% + entry cutoff 2hr
    print("\n[3/5] Running TEST B: Scale-in 13% + cutoff 2hr...")
    r = run_backtest_with_config(
        "B: SI 13% + Cutoff 2hr",
        BREAKEVEN_STOP_PCT=0.0,
        ENTRY_CUTOFF_MINUTES=120,
        SCALE_IN_TRIGGER_PCT=13.0,
        SCALE_IN_GATE_PARTIAL=False,
    )
    results.append(r)
    print_result(r)

    # TEST C: Scale-in 14% only (no cutoff)
    print("\n[4/5] Running TEST C: Scale-in 14% only...")
    r = run_backtest_with_config(
        "C: SI 14% only",
        BREAKEVEN_STOP_PCT=0.0,
        ENTRY_CUTOFF_MINUTES=0,
        SCALE_IN_TRIGGER_PCT=14.0,
        SCALE_IN_GATE_PARTIAL=False,
    )
    results.append(r)
    print_result(r)

    # TEST D: Scale-in 13% only (no cutoff)
    print("\n[5/5] Running TEST D: Scale-in 13% only...")
    r = run_backtest_with_config(
        "D: SI 13% only",
        BREAKEVEN_STOP_PCT=0.0,
        ENTRY_CUTOFF_MINUTES=0,
        SCALE_IN_TRIGGER_PCT=13.0,
        SCALE_IN_GATE_PARTIAL=False,
    )
    results.append(r)
    print_result(r)

    # COMPARISON TABLE
    print(f"\n\n{'='*100}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Test':<35} {'Final Cash':>15} {'Win%':>6} {'Sharpe':>7} {'MaxDD':>15} {'Jan P&L':>15} {'Jan W/L':>8}")
    print(f"  {'-'*35} {'-'*15} {'-'*6} {'-'*7} {'-'*15} {'-'*15} {'-'*8}")

    baseline = results[0]
    for r in results:
        delta = r['final_cash'] - baseline['final_cash']
        delta_str = f" ({'+' if delta >= 0 else ''}{delta/1e6:.1f}M)" if r != baseline else ""
        print(
            f"  {r['label']:<35} "
            f"${r['final_cash']/1e6:>10.1f}M "
            f"{r['win_rate']:>5.1f}% "
            f"{r['sharpe']:>7.2f} "
            f"${r['max_dd']/1e6:>+10.1f}M "
            f"${r['jan_pnl']/1e6:>+10.1f}M "
            f"{r['jan_winners']:>3}W/{r['jan_losers']}L"
            f"{delta_str}"
        )

    print(f"{'='*100}")
