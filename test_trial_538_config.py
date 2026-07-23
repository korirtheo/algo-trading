"""
Test BOTH L and G strategies with EXACT Trial #538 deployed parameters
to validate the $1.8M figure from earlier testing.

Key differences from defaults:
- G_MIN_GAP_PCT: 30% → 10% (much more trades)
- G_TARGET_PCT: 11% → 15%
- G_PARTIAL_SELL_PCT: 0% → 25% (partial sells)
- G_TARGET2_PCT: 30% → 60% (runners go much further)
- G_TIME_LIMIT_MINUTES: 10 → 12
- G_STOP_PCT: 0% → 26% (hard stops)
- L_MIN_GAP: 30% → 25%
- L_MAX_FLOAT: 15M → 25M (much more trades)
- L_TIME_LIMIT_MINUTES: 70 → 110
"""
import json
import sys
import os

sys.path.insert(0, ".")

import test_green_candle_combined as tgc
from test_full import load_all_picks


def run_trial_538_test():
    print("=" * 80)
    print("TRIAL #538 EXACT CONFIG BACKTEST")
    print("Both L and G strategies with deployed parameters")
    print("=" * 80)

    # Store original values
    originals = {
        "G_MIN_GAP_PCT": tgc.G_MIN_GAP_PCT,
        "G_TARGET_PCT": tgc.G_TARGET_PCT,
        "G_TARGET2_PCT": tgc.G_TARGET2_PCT,
        "G_PARTIAL_SELL_PCT": tgc.G_PARTIAL_SELL_PCT,
        "G_TIME_LIMIT_MINUTES": tgc.G_TIME_LIMIT_MINUTES,
        "G_STOP_PCT": tgc.G_STOP_PCT,
        "G_TRAIL_PCT": tgc.G_TRAIL_PCT,
        "G_TRAIL_ACTIVATE_PCT": tgc.G_TRAIL_ACTIVATE_PCT,
        "L_MIN_GAP_PCT": tgc.L_MIN_GAP_PCT,
        "L_MAX_FLOAT": tgc.L_MAX_FLOAT,
        "L_TIME_LIMIT_MINUTES": tgc.L_TIME_LIMIT_MINUTES,
        "L_TRAIL_PCT": tgc.L_TRAIL_PCT,
        "L_TRAIL_ACTIVATE_PCT": tgc.L_TRAIL_ACTIVATE_PCT,
        "L_STOP_PCT": tgc.L_STOP_PCT,
        "L_PARTIAL_SELL_PCT": tgc.L_PARTIAL_SELL_PCT,
    }

    # === APPLY EXACT TRIAL #538 PARAMETERS ===
    print("\nApplying Trial #538 parameters:")

    # Strategy G
    tgc.G_MIN_GAP_PCT = 10.0
    tgc.G_TARGET_PCT = 15.0
    tgc.G_TARGET2_PCT = 60.0
    tgc.G_PARTIAL_SELL_PCT = 25.0
    tgc.G_TIME_LIMIT_MINUTES = 12
    tgc.G_STOP_PCT = 26.0
    tgc.G_TRAIL_PCT = 0.5
    tgc.G_TRAIL_ACTIVATE_PCT = 0.0

    print(f"  G: gap>={tgc.G_MIN_GAP_PCT}%, target={tgc.G_TARGET_PCT}%, "
          f"partial={tgc.G_PARTIAL_SELL_PCT}%, target2={tgc.G_TARGET2_PCT}%, "
          f"stop={tgc.G_STOP_PCT}%, time={tgc.G_TIME_LIMIT_MINUTES}m, "
          f"trail={tgc.G_TRAIL_PCT}%")

    # Strategy L
    tgc.L_MIN_GAP_PCT = 25.0
    tgc.L_MAX_FLOAT = 25_000_000
    tgc.L_TIME_LIMIT_MINUTES = 110
    tgc.L_TRAIL_PCT = 0.0
    tgc.L_TRAIL_ACTIVATE_PCT = 1.0
    tgc.L_STOP_PCT = 18.0
    tgc.L_PARTIAL_SELL_PCT = 75.0

    # L tiered targets
    tgc.L_TIER1_FLOAT = 1_500_000
    tgc.L_TIER1_TARGET1_PCT = 40.0
    tgc.L_TIER1_TARGET2_PCT = 30.0
    tgc.L_TIER2_FLOAT = 6_000_000
    tgc.L_TIER2_TARGET1_PCT = 30.0
    tgc.L_TIER2_TARGET2_PCT = 55.0
    tgc.L_TIER3_TARGET1_PCT = 25.0
    tgc.L_TIER3_TARGET2_PCT = 40.0

    print(f"  L: gap>={tgc.L_MIN_GAP_PCT}%, float<={tgc.L_MAX_FLOAT/1e6:.0f}M, "
          f"time={tgc.L_TIME_LIMIT_MINUTES}m, trail={tgc.L_TRAIL_PCT}%")

    # Load data
    data_dirs = ["stored_data_mar_may_2026", "stored_data_2026"]
    data_dir = None
    for d in data_dirs:
        if os.path.exists(d):
            data_dir = d
            break

    if not data_dir:
        print("ERROR: No data directory found")
        return

    print(f"\nLoading data from: {data_dir}")
    all_dates, daily_picks = load_all_picks([data_dir])
    print(f"Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")

    # Run backtest WITH COMPOUNDING (rolling cash forward like live system)
    STARTING_CASH = 25000
    current_cash = STARTING_CASH
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0
    daily_results = []

    for i, date in enumerate(all_dates):
        picks = daily_picks.get(date, [])
        if not picks:
            continue

        states, new_cash, unsettled, log = tgc.simulate_day_combined(
            picks=picks,
            cash=current_cash,
            cash_account=False,
            is_live=False,
            params=None
        )

        day_pnl = new_cash - current_cash
        current_cash = new_cash  # Compound: roll forward

        day_trades = 0
        day_wins = 0
        day_losses = 0
        g_trades = 0
        l_trades = 0

        for st in states:
            if st.get("exit_time") and st.get("pnl") is not None:
                day_trades += 1
                if st["pnl"] > 0:
                    day_wins += 1
                else:
                    day_losses += 1
                if st.get("strategy") == "G":
                    g_trades += 1
                elif st.get("strategy") == "L":
                    l_trades += 1

        total_trades += day_trades
        winning_trades += day_wins
        losing_trades += day_losses
        total_pnl += day_pnl
        daily_results.append({"date": date, "trades": day_trades, "pnl": day_pnl, "equity": current_cash})

        if day_trades > 0:
            print(f"  {date}: {day_trades} trades (G:{g_trades} L:{l_trades}) "
                  f"W:{day_wins} L:{day_losses} P&L: ${day_pnl:,.2f} Equity: ${current_cash:,.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("TRIAL #538 RESULTS SUMMARY (COMPOUNDED)")
    print("=" * 80)
    print(f"Starting cash: ${STARTING_CASH:,.2f}")
    print(f"Final equity: ${current_cash:,.2f}")
    print(f"Days tested: {len(all_dates)}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {winning_trades / total_trades * 100:.1f}%")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Return: {(current_cash / STARTING_CASH - 1) * 100:.1f}%")
    print(f"Avg P&L per trade: ${total_pnl / total_trades if total_trades > 0 else 0:,.2f}")

    # Restore original values
    for key, val in originals.items():
        setattr(tgc, key, val)

    print(f"\n✓ Original parameters restored")
    return total_pnl, total_trades, winning_trades


if __name__ == "__main__":
    run_trial_538_test()
