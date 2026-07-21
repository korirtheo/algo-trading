"""
Backtest Jan-Jul 2026: Compare allowing re-entry vs one-trade-per-ticker-per-day.

This script analyzes LIVE trading data from 2026 to understand:
1. How often re-entry actually happens in practice
2. Which strategies/tickers re-enter most
3. PnL impact of re-entries (were they winners or losers?)
4. What the results would be if we blocked re-entry

Since we have live trade logs, we'll analyze actual behavior rather than
running full backtests (which would require downloading months of bar data).
"""
import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from collections import defaultdict


def load_live_trades():
    """Load all 2026 live trading data from AWS logs."""
    import subprocess

    print("Fetching live trade logs from AWS...")

    # Get list of trade log files
    result = subprocess.run(
        ['ssh', '-i', 'trading-key-v2.pem', '-p', '2222', '-o', 'StrictHostKeyChecking=no',
         'ubuntu@54.172.65.25', 'ls ~/algo-trading/logs/2026-*_trades.json'],
        capture_output=True, text=True
    )

    files = result.stdout.strip().split('\n')
    print(f"Found {len(files)} trading days")

    all_trades = []

    for file_path in files:
        date_str = file_path.split('/')[-1].replace('_trades.json', '')

        # Fetch the file
        result = subprocess.run(
            ['ssh', '-i', 'trading-key-v2.pem', '-p', '2222', '-o', 'StrictHostKeyChecking=no',
             'ubuntu@54.172.65.25', f'cat {file_path}'],
            capture_output=True, text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            try:
                trades = json.loads(result.stdout)
                for trade in trades:
                    trade['date'] = date_str
                all_trades.extend(trades)
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse {date_str}: {e}")

    print(f"Loaded {len(all_trades)} total trade records\n")
    return pd.DataFrame(all_trades)


def deduplicate_trades(df):
    """Remove duplicate trades caused by the fill event bug.

    Duplicates have:
    - Same entry_time, entry_price, ticker, strategy
    - Different shares (cumulative fill events)
    - Keep the one with highest shares (final fill)
    """
    print("Deduplicating trades (removing fill event bug artifacts)...")

    before_count = len(df)

    # Handle missing shares - fill with 0
    df['shares'] = pd.to_numeric(df['shares'], errors='coerce').fillna(0)

    # Group by trade identifier
    df['trade_id'] = (
        df['date'] + '_' +
        df['ticker'] + '_' +
        df['strategy'] + '_' +
        pd.to_datetime(df['entry_time'], errors='coerce').dt.strftime('%H:%M:%S')
    )

    # Remove rows with invalid entry_time (couldn't parse)
    df = df[df['trade_id'].notna()].copy()

    # Within each trade_id group, keep the row with max shares (final fill)
    # If shares are all 0 or equal, keep first one
    idx_to_keep = []
    for trade_id, group in df.groupby('trade_id'):
        if group['shares'].max() > 0:
            idx_to_keep.append(group['shares'].idxmax())
        else:
            idx_to_keep.append(group.index[0])

    df_deduped = df.loc[idx_to_keep].copy()

    after_count = len(df_deduped)
    removed = before_count - after_count

    print(f"  Before: {before_count} records")
    print(f"  After: {after_count} records")
    print(f"  Removed: {removed} duplicates ({removed/before_count*100:.1f}%)\n")

    return df_deduped.drop(columns=['trade_id'])


def analyze_reentries(trades_df):
    """Analyze re-entry patterns in live trading data."""
    print(f"\n{'='*60}")
    print("RE-ENTRY ANALYSIS (Live Trading)")
    print(f"{'='*60}\n")

    # Count trades per ticker per day per strategy
    trades_df['date_ticker_strategy'] = (
        trades_df['date'] + '_' +
        trades_df['ticker'] + '_' +
        trades_df['strategy']
    )

    reentry_counts = trades_df.groupby('date_ticker_strategy').size()
    reentries = reentry_counts[reentry_counts > 1]

    print(f"Total trading days: {trades_df['date'].nunique()}")
    print(f"Total trades: {len(trades_df)}")
    print(f"Unique date-ticker-strategy combos: {trades_df['date_ticker_strategy'].nunique()}")
    print(f"Combos with re-entry: {len(reentries)}")
    print(f"Re-entry rate: {len(reentries) / trades_df['date_ticker_strategy'].nunique() * 100:.2f}%\n")

    if len(reentries) > 0:
        print("Top 10 re-entry cases:")
        top_reentries = reentries.sort_values(ascending=False).head(10)
        for combo, count in top_reentries.items():
            date, ticker, strategy = combo.rsplit('_', 2)
            print(f"  {date} - {ticker} ({strategy}): {count} trades")
        print()

        # Which strategies re-enter most?
        reentry_by_strategy = defaultdict(int)
        reentry_trades_by_strategy = defaultdict(int)
        for combo, count in reentries.items():
            strategy = combo.split('_')[-1]
            reentry_by_strategy[strategy] += 1  # number of unique combos
            reentry_trades_by_strategy[strategy] += (count - 1)  # extra trades beyond first

        print("Re-entries by strategy:")
        for strategy in sorted(reentry_by_strategy.keys()):
            print(f"  {strategy}: {reentry_by_strategy[strategy]} cases ({reentry_trades_by_strategy[strategy]} extra trades)")
        print()

        # Show examples of actual re-entry trades
        print("\nExample re-entry sequences (first 5 cases with most trades):")
        for i, combo in enumerate(top_reentries.head(5).index):
            date, ticker, strategy = combo.rsplit('_', 2)
            trades = trades_df[trades_df['date_ticker_strategy'] == combo].sort_values('entry_time')
            print(f"\n{i+1}. {date} - {ticker} ({strategy}) - {len(trades)} trades:")
            for j, (_, trade) in enumerate(trades.iterrows(), 1):
                entry_time = pd.to_datetime(trade['entry_time']).strftime('%H:%M:%S')
                exit_time = pd.to_datetime(trade['exit_time']).strftime('%H:%M:%S') if pd.notna(trade.get('exit_time')) else 'N/A'
                print(f"   Trade {j}: Entry {entry_time} @ ${trade['entry_price']:.2f} → "
                      f"Exit {exit_time} @ ${trade['exit_price']:.2f} | "
                      f"PnL: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")


def simulate_no_reentry(trades_df):
    """Simulate what results would be if we blocked re-entry."""
    print(f"\n{'='*60}")
    print("SIMULATING NO RE-ENTRY POLICY")
    print(f"{'='*60}\n")

    # Sort by entry time to process chronologically
    trades_sorted = trades_df.sort_values(['date', 'entry_time']).copy()

    # Track first trades and re-entries
    first_trades = []
    reentries = []
    traded_today = defaultdict(set)  # date -> set of (ticker, strategy)

    for _, trade in trades_sorted.iterrows():
        date = trade['date']
        ticker = trade['ticker']
        strategy = trade['strategy']
        key = (ticker, strategy)

        if key in traded_today[date]:
            # This is a re-entry
            reentries.append(trade)
        else:
            # First trade for this ticker/strategy today
            traded_today[date].add(key)
            first_trades.append(trade)

    first_trades_df = pd.DataFrame(first_trades)
    reentries_df = pd.DataFrame(reentries)

    print(f"First trades (would execute): {len(first_trades_df)}")
    print(f"Re-entries (would skip): {len(reentries_df)}\n")

    return first_trades_df, reentries_df


def compare_results(all_trades_df, first_trades_df, reentries_df):
    """Compare baseline (all trades) vs no-reentry (first only) performance."""
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}\n")

    all_pnl = all_trades_df['pnl'].sum()
    first_pnl = first_trades_df['pnl'].sum()
    reentry_pnl = reentries_df['pnl'].sum()

    all_wins = (all_trades_df['pnl'] > 0).sum()
    all_losses = (all_trades_df['pnl'] < 0).sum()
    first_wins = (first_trades_df['pnl'] > 0).sum()
    first_losses = (first_trades_df['pnl'] < 0).sum()

    print(f"CURRENT (allows re-entry):")
    print(f"  Total trades: {len(all_trades_df)}")
    print(f"  Wins: {all_wins} | Losses: {all_losses}")
    print(f"  Win rate: {all_wins / len(all_trades_df) * 100:.2f}%")
    print(f"  Total PnL: ${all_pnl:,.2f}")
    print(f"  Avg PnL per trade: ${all_pnl / len(all_trades_df):.2f}")
    print()

    print(f"NO_REENTRY (one per ticker per strategy per day):")
    print(f"  Total trades: {len(first_trades_df)}")
    print(f"  Wins: {first_wins} | Losses: {first_losses}")
    print(f"  Win rate: {first_wins / len(first_trades_df) * 100:.2f}%")
    print(f"  Total PnL: ${first_pnl:,.2f}")
    print(f"  Avg PnL per trade: ${first_pnl / len(first_trades_df):.2f}")
    print()

    pnl_diff = all_pnl - first_pnl
    print(f"DIFFERENCE:")
    print(f"  Trades prevented: {len(reentries_df)}")
    print(f"  PnL from prevented re-entries: ${reentry_pnl:,.2f}")
    print(f"  Net impact: {'+' if pnl_diff > 0 else ''}${pnl_diff:,.2f}")
    print(f"  {'✓ Re-entry HELPS (+' if pnl_diff > 0 else '✗ Re-entry HURTS ('}"
          f"${abs(pnl_diff):,.2f})")
    print()

    # Analyze prevented re-entries
    if len(reentries_df) > 0:
        reentry_wins = (reentries_df['pnl'] > 0).sum()
        reentry_losses = (reentries_df['pnl'] < 0).sum()
        print(f"RE-ENTRIES BREAKDOWN:")
        print(f"  Winners: {reentry_wins} (${reentries_df[reentries_df['pnl'] > 0]['pnl'].sum():,.2f})")
        print(f"  Losers: {reentry_losses} (${reentries_df[reentries_df['pnl'] < 0]['pnl'].sum():,.2f})")
        print(f"  Win rate: {reentry_wins / len(reentries_df) * 100:.2f}%")
        print(f"  Avg PnL: ${reentry_pnl / len(reentries_df):.2f}")
        print()

        # Which strategies' re-entries were most valuable?
        print("Re-entry PnL by strategy:")
        for strategy in sorted(reentries_df['strategy'].unique()):
            strat_trades = reentries_df[reentries_df['strategy'] == strategy]
            wins = (strat_trades['pnl'] > 0).sum()
            losses = (strat_trades['pnl'] < 0).sum()
            pnl = strat_trades['pnl'].sum()
            print(f"  {strategy}: {len(strat_trades)} trades ({wins}W/{losses}L), ${pnl:.2f} total")


def main():
    print("=" * 60)
    print("RE-ENTRY IMPACT ANALYSIS: 2026 Live Trading Data")
    print("=" * 60)
    print("\nThis analyzes ACTUAL live trading behavior from 2026.")
    print("We'll see:")
    print("  1. How often re-entry actually happens")
    print("  2. Which strategies re-enter most")
    print("  3. Whether re-entries are profitable")
    print("  4. Impact if we blocked re-entry\n")

    # Load live trades
    trades_df = load_live_trades()

    # Deduplicate (remove fill event bug artifacts)
    trades_clean = deduplicate_trades(trades_df)

    # Analyze re-entry patterns
    analyze_reentries(trades_clean)

    # Simulate no re-entry policy
    first_trades_df, reentries_df = simulate_no_reentry(trades_clean)

    # Compare results
    compare_results(trades_clean, first_trades_df, reentries_df)

    # Save results
    trades_clean.to_csv("live_trades_2026_deduped.csv", index=False)
    first_trades_df.to_csv("simulated_no_reentry_2026.csv", index=False)
    reentries_df.to_csv("prevented_reentries_2026.csv", index=False)
    print(f"\n✓ Saved results to CSV files")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
