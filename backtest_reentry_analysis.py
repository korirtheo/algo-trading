"""
Backtest 2026 data with re-entry analysis.

Runs TWO backtests:
1. BASELINE: Current behavior (allows re-entry)
2. NO_REENTRY: Filters out re-entries to see impact

Usage:
  python backtest_reentry_analysis.py
  python backtest_reentry_analysis.py --data stored_data_2026_jan_jul
"""
import argparse
import json
import os
from collections import defaultdict
import pandas as pd

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks


def load_params(path):
    """Load parameters from JSON and set on tgc module."""
    with open(path) as f:
        params = json.load(f)
    if isinstance(params, dict) and "params" in params:
        params = params["params"]

    # Use the optimizer's set_strategy_params to apply to module globals
    from optimize_combined import set_strategy_params
    set_strategy_params(params)

    return params


def extract_trades_from_states(states, date_str):
    """Extract completed trades from backtest states."""
    trades = []
    for st in states:
        if st.get("exit_time") and st.get("pnl") is not None:
            trades.append({
                "date": date_str,
                "ticker": st["ticker"],
                "strategy": st["strategy"],
                "entry_time": str(st["entry_time"]),
                "exit_time": str(st["exit_time"]),
                "entry_price": st["entry_price"],
                "exit_price": st.get("exit_price", 0),
                "shares": st["shares"],
                "pnl": st["pnl"],
                "pnl_pct": st.get("pnl_pct", 0),
            })
    return trades


def run_baseline_backtest(data_dir, params, start_cash):
    """Run backtest with current behavior (allows re-entry)."""
    print(f"\n{'='*60}")
    print("BASELINE: Current behavior (allows re-entry)")
    print(f"{'='*60}\n")

    all_dates, all_picks = load_all_picks([data_dir])
    all_trades = []
    cash = start_cash

    for date_str in all_dates:
        picks = all_picks[date_str]
        if not picks:
            continue

        print(f"{date_str}...", end=' ')

        # Run backtest (params already set on tgc module globals via set_strategy_params)
        states, final_cash, unsettled, log = tgc.simulate_day_combined(
            picks=picks,
            cash=cash,
            cash_account=False,
            is_live=False
        )

        # Extract trades
        trades = extract_trades_from_states(states, date_str)
        all_trades.extend(trades)

        # Update cash for next day
        cash = final_cash

        print(f"{len(trades)} trades, cash=${final_cash:,.0f}")

    print(f"\nFinal cash: ${cash:,.2f}")
    print(f"Total PnL: ${cash - start_cash:,.2f}")
    print(f"Multiplier: {cash / start_cash:.2f}x\n")

    return pd.DataFrame(all_trades), cash


def filter_reentries(trades_df):
    """Split trades into first-entries and re-entries."""
    trades_sorted = trades_df.sort_values(['date', 'entry_time']).copy()

    first_trades = []
    reentries = []
    traded_today = defaultdict(set)  # date -> set of (ticker, strategy)

    for _, trade in trades_sorted.iterrows():
        date = trade['date']
        key = (trade['ticker'], trade['strategy'])

        if key in traded_today[date]:
            reentries.append(trade)
        else:
            traded_today[date].add(key)
            first_trades.append(trade)

    return pd.DataFrame(first_trades), pd.DataFrame(reentries)


def analyze_reentries(all_trades_df, first_trades_df, reentries_df):
    """Analyze re-entry patterns and impact."""
    print(f"\n{'='*60}")
    print("RE-ENTRY ANALYSIS")
    print(f"{'='*60}\n")

    # Overall stats
    all_trades_df['combo'] = (
        all_trades_df['date'] + '_' +
        all_trades_df['ticker'] + '_' +
        all_trades_df['strategy']
    )

    reentry_counts = all_trades_df.groupby('combo').size()
    reentries_detected = reentry_counts[reentry_counts > 1]

    print(f"Total days: {all_trades_df['date'].nunique()}")
    print(f"Total trades: {len(all_trades_df)}")
    print(f"First trades: {len(first_trades_df)}")
    print(f"Re-entries: {len(reentries_df)}")
    print(f"Re-entry rate: {len(reentries_df) / len(all_trades_df) * 100:.1f}%\n")

    if len(reentries_df) > 0:
        # Top re-entry cases
        print("Top 10 re-entry cases:")
        for combo, count in reentries_detected.sort_values(ascending=False).head(10).items():
            date, ticker, strat = combo.rsplit('_', 2)
            print(f"  {date} {ticker} ({strat}): {count} trades")
        print()

        # By strategy
        reentry_by_strat = reentries_df.groupby('strategy').size().sort_values(ascending=False)
        print("Re-entries by strategy:")
        for strat, count in reentry_by_strat.items():
            print(f"  {strat}: {count} trades")
        print()

        # Show example sequences
        print("Example sequences (top 3 most re-entered):")
        for i, combo in enumerate(reentries_detected.head(3).index, 1):
            date, ticker, strat = combo.rsplit('_', 2)
            seq = all_trades_df[all_trades_df['combo'] == combo].sort_values('entry_time')
            print(f"\n{i}. {date} - {ticker} ({strat}) - {len(seq)} trades:")
            for j, (_, t) in enumerate(seq.iterrows(), 1):
                entry_p = t['entry_price'] if pd.notna(t['entry_price']) else 0
                exit_p = t['exit_price'] if pd.notna(t['exit_price']) else 0
                print(f"   #{j}: ${entry_p:.2f} → ${exit_p:.2f} | "
                      f"PnL: ${t['pnl']:.2f} ({t['pnl_pct']:.1f}%)")


def compare_results(baseline_df, first_df, reentry_df, baseline_final_cash, start_cash):
    """Compare baseline vs no-reentry performance."""
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

    print(f"BASELINE (allows re-entry):")
    print(f"  Trades: {len(baseline_df)}")
    print(f"  Wins: {baseline_wins} | Losses: {baseline_losses}")
    print(f"  Win rate: {baseline_wins / len(baseline_df) * 100:.1f}%")
    print(f"  Total PnL: ${baseline_pnl:,.2f}")
    print(f"  Avg PnL/trade: ${baseline_pnl / len(baseline_df):.2f}")
    print(f"  Final cash: ${baseline_final_cash:,.2f}")
    print(f"  Multiplier: {baseline_final_cash / start_cash:.2f}x\n")

    # Estimate no-reentry final cash (rough approximation)
    no_reentry_final = start_cash + first_pnl

    print(f"NO RE-ENTRY (one per ticker/strategy/day):")
    print(f"  Trades: {len(first_df)}")
    print(f"  Wins: {first_wins} | Losses: {first_losses}")
    print(f"  Win rate: {first_wins / len(first_df) * 100:.1f}%")
    print(f"  Total PnL: ${first_pnl:,.2f}")
    print(f"  Avg PnL/trade: ${first_pnl / len(first_df):.2f}")
    print(f"  Estimated final: ${no_reentry_final:,.2f}")
    print(f"  Estimated multiplier: {no_reentry_final / start_cash:.2f}x\n")

    pnl_diff = baseline_pnl - first_pnl
    print(f"DIFFERENCE:")
    print(f"  Blocked trades: {len(reentry_df)}")
    print(f"  PnL from blocked: ${reentry_pnl:,.2f}")
    print(f"  Net impact: {'+' if pnl_diff > 0 else ''}${pnl_diff:,.2f}")
    print(f"  {'✓ Re-entry HELPS (+' if pnl_diff > 0 else '✗ Re-entry HURTS ('}"
          f"${abs(pnl_diff):,.2f})\n")

    if len(reentry_df) > 0:
        reentry_wins = (reentry_df['pnl'] > 0).sum()
        reentry_losses = (reentry_df['pnl'] < 0).sum()
        print(f"BLOCKED RE-ENTRIES BREAKDOWN:")
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
    parser.add_argument("--data", default="stored_data_2026_jan_jul",
                        help="Data directory")
    parser.add_argument("--cash", type=float, default=25000,
                        help="Starting cash")
    parser.add_argument("--params", default="live/trial_g511_l626_v3_overlay.json",
                        help="Parameters JSON")
    args = parser.parse_args()

    print("="*60)
    print("BACKTEST: Re-Entry Impact Analysis")
    print("="*60)
    print(f"\nData: {args.data}")
    print(f"Starting cash: ${args.cash:,.0f}")
    print(f"Params: {args.params}\n")

    # Load params
    params = load_params(args.params)

    # Run baseline backtest
    baseline_df, final_cash = run_baseline_backtest(args.data, params, args.cash)

    # Split into first trades and re-entries
    first_df, reentry_df = filter_reentries(baseline_df)

    # Analysis
    analyze_reentries(baseline_df, first_df, reentry_df)
    compare_results(baseline_df, first_df, reentry_df, final_cash, args.cash)

    # Save results
    baseline_df.to_csv("backtest_baseline_all_trades.csv", index=False)
    first_df.to_csv("backtest_first_trades_only.csv", index=False)
    if len(reentry_df) > 0:
        reentry_df.to_csv("backtest_reentries.csv", index=False)

    print(f"\n✓ Saved results to CSV files")
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
