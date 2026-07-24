"""
Test BOTH L (Low Float Squeeze) and G (Big Gap Runner) strategies
with G_TRAIL_PCT = 0.5%
Run backtest on 2026 OOS data from March onwards
"""
import json
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, ".")

import test_green_candle_combined as tgc
from test_full import load_all_picks

def run_g_trail_test():
    print("=" * 80)
    print("STRATEGY G TRAIL TEST: G_TRAIL_PCT = 0.5%")
    print("=" * 80)

    # Store original values for restoration
    original_g_trail = tgc.G_TRAIL_PCT

    # Configure BOTH L and G strategies
    # Strategy L: Keep existing trail settings (1.0% with 2.0% activation)
    # Strategy G: New configuration with 0.5% trail (as requested)
    print(f"\nStrategy Configuration:")
    print(f"  Strategy L TRAIL: {tgc.L_TRAIL_PCT}% (activates at {tgc.L_TRAIL_ACTIVATE_PCT}% - unchanged)")
    print(f"  Strategy G TRAIL: {tgc.G_TRAIL_PCT}% (NEW: increased from {original_g_trail}% to 0.5%)")
    print(f"  Strategy G TRAIL ACTIVATION: 0.0% (immediate)")
    print(f"  Other G settings unchanged:")
    print(f"    - G_TARGET_PCT: {tgc.G_TARGET_PCT}%")
    print(f"    - G_TIME_LIMIT_MINUTES: {tgc.G_TIME_LIMIT_MINUTES} min")
    print(f"    - G_STOP_PCT: {tgc.G_STOP_PCT}%")
    print(f"    - G_PARTIAL_SELL_PCT: {tgc.G_PARTIAL_SELL_PCT}%")
    print(f"    - G_TARGET2_PCT: {tgc.G_TARGET2_PCT}%")

    # Set G strategy to 0.5% trail as requested (increase from 0% to provide protection)
    tgc.G_TRAIL_PCT = 0.5  # As requested: G_TRAIL_PCT = 0.5%
    tgc.G_TRAIL_ACTIVATE_PCT = 0.0  # Start trailing immediately (0% activation)

    print(f"\nNew Strategy G Trail Configuration (as requested):")
    print(f"  G_TRAIL_PCT: {tgc.G_TRAIL_PCT}% (as requested)")
    print(f"  G_TRAIL_ACTIVATE_PCT: {tgc.G_TRAIL_ACTIVATE_PCT}% (immediate)")

    # Load 2026 data from March onwards
    print(f"\nLoading 2026 OOS data (March onwards)...")

    # Try different possible directories
    data_dirs = [
        "stored_data_mar_may_2026",
        "stored_data_2026",
        "stored_data_2026_gap_fill"
    ]

    data_dir = None
    for d in data_dirs:
        import os
        if os.path.exists(d):
            data_dir = d
            print(f"Using data directory: {d}")
            break

    if not data_dir:
        print("No 2026 data directory found. Available directories:")
        import os
        for item in os.listdir("."):
            if "stored_data" in item and os.path.isdir(item):
                print(f"  {item}")
        return

    all_dates, daily_picks = load_all_picks([data_dir])

    # Filter to March - December 2026 (approximately 60 days)
    print(f"\nOriginal data range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")

    # Check what month we're looking at
    start_date = all_dates[0]
    if start_date >= "2026-03-01":
        print(f"Data starts in March 2026 - using full dataset")
        filtered_dates = all_dates
    elif start_date >= "2026-01-01":
        print(f"Data starts in Jan-Feb 2026 - filtering from March 2026")
        filtered_dates = [d for d in all_dates if d >= "2026-03-01"]
    else:
        print(f"Data doesn't start in 2026 - checking...")
        filtered_dates = all_dates

    print(f"Filtered dates: {filtered_dates[0]} to {filtered_dates[-1]} ({len(filtered_dates)} days)")

    # Run backtest similar to the main combined strategy
    print("\n" + "=" * 80)
    print("RUNNING BACKTEST WITH G_TRAIL_PCT = 0.5%")
    print("=" * 80)

    # Base simulation function from test_green_candle_combined.py
    def run_simulation(date_str, picks):
        # Use the existing simulate_day_combined function
        states, cash, unsettled, log = tgc.simulate_day_combined(
            picks=picks,
            cash=100000,  # Use $100K base cash like in memory
            cash_account=False,
            is_live=False,
            params=None  # Use module globals
        )
        return states, cash, unsettled, log

    # Run simulation for each date
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0

    for i, date in enumerate(filtered_dates[:60]):  # Test first 60 days
        picks = daily_picks.get(date, [])
        if not picks:
            continue

        print(f"\nDate {date} ({i+1}/60): {len(picks)} picks")

        states, cash, unsettled, log = run_simulation(date, picks)

        # Count trades and P&L for the day
        day_trades = 0
        day_wins = 0
        day_losses = 0
        day_pnl = 0

        for st in states:
            if st.get("exit_time") and st.get("pnl") is not None:
                day_trades += 1
                day_pnl += st["pnl"]
                if st["pnl"] > 0:
                    day_wins += 1
                else:
                    day_losses += 1

        total_trades += day_trades
        winning_trades += day_wins
        losing_trades += day_losses
        total_pnl += day_pnl

        print(f"  Trades: {day_trades} | Wins: {day_wins} | Losses: {day_losses} | P&L: ${day_pnl:,.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total dates tested: {len(filtered_dates[:60])}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {winning_trades / total_trades * 100:.1f}%")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Avg P&L per trade: ${total_pnl / total_trades if total_trades > 0 else 0:,.2f}")

    # Exit strategy analysis
    print(f"\nStrategy Configuration Analysis:")
    print(f"  Strategy L TRAIL: {tgc.L_TRAIL_PCT}% (activates at {tgc.L_TRAIL_ACTIVATE_PCT}% - unchanged)")
    print(f"  Strategy G TRAIL: {tgc.G_TRAIL_PCT}% (NEW: increased from {original_g_trail}% to 0.5%)")
    print(f"  New G settings:")
    print(f"    - Time limit: {tgc.G_TIME_LIMIT_MINUTES} minutes")
    print(f"    - Profit target: {tgc.G_TARGET_PCT}% (sell all)")
    print(f"    - Trail stop: {tgc.G_TRAIL_PCT}% (NEW: provides downside protection)")

    # Restore original values
    tgc.G_TRAIL_PCT = original_g_trail
    tgc.G_TRAIL_ACTIVATE_PCT = original_g_trail_activate

    print(f"\n✓ Test complete. Strategy G has been tested with G_TRAIL_PCT = 0.5%")
    print(f"✓ Strategy L unchanged with L_TRAIL_PCT = {tgc.L_TRAIL_PCT}% and L_TRAIL_ACTIVATE_PCT = {tgc.L_TRAIL_ACTIVATE_PCT}%")

if __name__ == "__main__":
    run_g_trail_test()