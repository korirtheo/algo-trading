"""
Comprehensive re-entry comparison: 2024-2026 data.

Tests re-entry impact across full 3-year dataset with deployed config.
"""
import argparse
import json
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks


def run_backtest(data_dirs, params, start_cash, label, enable_reentry=True):
    """Run backtest across multiple data directories."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}\n")

    # Set re-entry flag
    tgc.ENABLE_REENTRY = enable_reentry
    print(f"Re-entry: {'ENABLED' if enable_reentry else 'DISABLED'}\n")

    all_trades = []
    cash = start_cash
    days_traded = 0

    for data_dir in data_dirs:
        if not Path(data_dir).exists():
            print(f"Skipping {data_dir} (not found)")
            continue

        print(f"\nProcessing {data_dir}...")
        all_dates, all_picks = load_all_picks([data_dir])

        for date_str in all_dates:
            picks = all_picks[date_str]
            if not picks:
                continue

            # Run backtest
            states, final_cash, unsettled, log = tgc.simulate_day_combined(
                picks=picks,
                cash=cash,
                cash_account=False,
                is_live=False
            )

            # Extract completed trades
            trades_today = []
            for st in states:
                if st.get("exit_time") and st.get("pnl") is not None:
                    trades_today.append({
                        "date": date_str,
                        "ticker": st["ticker"],
                        "strategy": st["strategy"],
                        "entry_time": str(st["entry_time"]),
                        "exit_time": str(st["exit_time"]),
                        "entry_price": st.get("entry_price", 0),
                        "exit_price": st.get("exit_price", 0),
                        "shares": st.get("shares", 0),
                        "pnl": st["pnl"],
                        "pnl_pct": st.get("pnl_pct", 0),
                    })

            all_trades.extend(trades_today)
            cash = final_cash
            days_traded += 1

            if days_traded % 50 == 0:
                print(f"  {days_traded} days | Cash: ${cash:,.0f}")

    print(f"\nCompleted: {days_traded} trading days")
    print(f"Final cash: ${cash:,.2f}")
    print(f"Total PnL: ${cash - start_cash:,.2f}")
    print(f"Multiplier: {cash / start_cash:.2f}x")

    return pd.DataFrame(all_trades), cash


def split_trades(trades_df):
    """Split trades into first-entries and re-entries."""
    trades_sorted = trades_df.sort_values(['date', 'entry_time']).copy()

    first_trades = []
    reentries = []
    traded_today = defaultdict(set)

    for _, trade in trades_sorted.iterrows():
        date = trade['date']
        key = (trade['ticker'], trade['strategy'])

        if key in traded_today[date]:
            reentries.append(trade)
        else:
            traded_today[date].add(key)
            first_trades.append(trade)

    return pd.DataFrame(first_trades), pd.DataFrame(reentries)


def compare_results(baseline_df, first_df, reentry_df, baseline_cash, start_cash):
    """Compare performance with and without re-entry."""
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}\n")

    baseline_pnl = baseline_df['pnl'].sum()
    first_pnl = first_df['pnl'].sum()
    reentry_pnl = reentry_df['pnl'].sum() if len(reentry_df) > 0 else 0

    baseline_wins = (baseline_df['pnl'] > 0).sum()
    baseline_losses = (baseline_df['pnl'] < 0).sum()
    first_wins = (first_df['pnl'] > 0).sum()
    first_losses = (first_df['pnl'] < 0).sum()

    print(f"WITH RE-ENTRY (baseline):")
    print(f"  Trades: {len(baseline_df)}")
    print(f"  Wins: {baseline_wins} | Losses: {baseline_losses}")
    print(f"  Win rate: {baseline_wins / len(baseline_df) * 100:.1f}%")
    print(f"  Total PnL: ${baseline_pnl:,.2f}")
    print(f"  Final cash: ${baseline_cash:,.2f}")
    print(f"  Multiplier: {baseline_cash / start_cash:.2f}x\n")

    no_reentry_final = start_cash + first_pnl

    print(f"WITHOUT RE-ENTRY (blocked):")
    print(f"  Trades: {len(first_df)}")
    print(f"  Wins: {first_wins} | Losses: {first_losses}")
    print(f"  Win rate: {first_wins / len(first_df) * 100:.1f}%")
    print(f"  Total PnL: ${first_pnl:,.2f}")
    print(f"  Estimated final: ${no_reentry_final:,.2f}")
    print(f"  Estimated multiplier: {no_reentry_final / start_cash:.2f}x\n")

    pnl_diff = baseline_pnl - first_pnl
    print(f"RE-ENTRY IMPACT:")
    print(f"  Blocked trades: {len(reentry_df)}")
    print(f"  PnL from blocked: ${reentry_pnl:,.2f}")
    print(f"  Net impact: {'+' if pnl_diff > 0 else ''}${pnl_diff:,.2f}")

    if pnl_diff > 0:
        print(f"  Result: Re-entry HELPS (+${abs(pnl_diff):,.2f})")
    else:
        print(f"  Result: Re-entry HURTS (${abs(pnl_diff):,.2f})")
    print()

    if len(reentry_df) > 0:
        reentry_wins = (reentry_df['pnl'] > 0).sum()
        reentry_losses = (reentry_df['pnl'] < 0).sum()
        print(f"RE-ENTRIES BREAKDOWN:")
        print(f"  Winners: {reentry_wins} (${reentry_df[reentry_df['pnl'] > 0]['pnl'].sum():,.2f})")
        print(f"  Losers: {reentry_losses} (${reentry_df[reentry_df['pnl'] < 0]['pnl'].sum():,.2f})")
        print(f"  Win rate: {reentry_wins / len(reentry_df) * 100:.1f}%")
        print(f"  Avg PnL: ${reentry_pnl / len(reentry_df):.2f}\n")

        # By strategy
        print("Re-entry PnL by strategy:")
        for strat in sorted(reentry_df['strategy'].unique()):
            strat_trades = reentry_df[reentry_df['strategy'] == strat]
            wins = (strat_trades['pnl'] > 0).sum()
            losses = (strat_trades['pnl'] < 0).sum()
            pnl = strat_trades['pnl'].sum()
            print(f"  {strat}: {len(strat_trades)} trades ({wins}W/{losses}L), ${pnl:,.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cash", type=float, default=25000, help="Starting cash")
    parser.add_argument("--params", default="config/trial_g511_l626_v3_overlay.json", help="Parameters JSON")
    args = parser.parse_args()

    print("="*60)
    print("BACKTEST: Re-Entry Analysis 2024-2026")
    print("="*60)
    print(f"\nStarting cash: ${args.cash:,.0f}")
    print(f"Params: {args.params}")

    # Data directories for 2024-2026 (only those with intraday data)
    data_dirs = [
        "stored_data_jan_feb_2024",
        "stored_data_jan_mar_2024",
        "stored_data_apr_jun_2024",
        "stored_data_jul_sep_2024",
        "stored_data_oct_dec_2024",
        "stored_data_jan_mar_2025",
        "stored_data_apr_jun_2025",
        "stored_data_jul_2025",
        "stored_data",  # 2026 Jan-Jun
        "stored_data_jul_2026",
    ]

    existing_dirs = [d for d in data_dirs if Path(d).exists()]
    print(f"\nData directories found: {existing_dirs}\n")

    # Load params
    with open(args.params) as f:
        params = json.load(f)
    if isinstance(params, dict) and "params" in params:
        params = params["params"]
    set_strategy_params(params)

    # Run backtest WITH re-entry
    print("\n" + "="*60)
    print("RUNNING: WITH RE-ENTRY")
    print("="*60)
    baseline_df, cash_with_reentry = run_backtest(
        existing_dirs, params, args.cash,
        "BASELINE: With Re-Entry",
        enable_reentry=True
    )

    # Run backtest WITHOUT re-entry
    print("\n" + "="*60)
    print("RUNNING: WITHOUT RE-ENTRY")
    print("="*60)
    no_reentry_df, cash_no_reentry = run_backtest(
        existing_dirs, params, args.cash,
        "COMPARISON: Without Re-Entry",
        enable_reentry=False
    )

    # Split baseline into first trades and re-entries for analysis
    first_df, reentry_df = split_trades(baseline_df)

    # Analysis
    print("\n" + "="*60)
    print("DIRECT COMPARISON")
    print("="*60)
    print(f"\nWITH RE-ENTRY:")
    print(f"  Trades: {len(baseline_df)}")
    print(f"  Final cash: ${cash_with_reentry:,.2f}")
    print(f"  Multiplier: {cash_with_reentry / args.cash:.2f}x")
    print(f"\nWITHOUT RE-ENTRY:")
    print(f"  Trades: {len(no_reentry_df)}")
    print(f"  Final cash: ${cash_no_reentry:,.2f}")
    print(f"  Multiplier: {cash_no_reentry / args.cash:.2f}x")
    print(f"\nDIFFERENCE:")
    diff = cash_with_reentry - cash_no_reentry
    print(f"  Cash delta: {'+' if diff > 0 else ''}${diff:,.2f}")
    print(f"  Trade delta: {'+' if len(baseline_df) > len(no_reentry_df) else ''}{len(baseline_df) - len(no_reentry_df)}")
    if diff > 0:
        print(f"  Result: Re-entry HELPS (+${abs(diff):,.2f}, +{(diff/cash_no_reentry)*100:.1f}%)")
    else:
        print(f"  Result: Re-entry HURTS (${abs(diff):,.2f}, {(diff/cash_no_reentry)*100:.1f}%)")

    compare_results(baseline_df, first_df, reentry_df, cash_with_reentry, args.cash)

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    baseline_df.to_csv(f"backtest_reentry_{timestamp}_all.csv", index=False)
    first_df.to_csv(f"backtest_reentry_{timestamp}_first.csv", index=False)
    if len(reentry_df) > 0:
        reentry_df.to_csv(f"backtest_reentry_{timestamp}_reentries.csv", index=False)

    print(f"\nSaved results to backtest_reentry_{timestamp}_*.csv")
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
