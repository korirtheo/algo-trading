"""
Simple re-entry analysis: Run backtest on recent dates and analyze trades.
"""
import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Import backtest engine
import test_green_candle_combined as tgc
from live.scanner import get_watchlist_date


def load_params():
    """Load deployed parameters."""
    params_path = Path(__file__).parent / "live" / "trial_g511_l626_v3_overlay.json"
    with open(params_path) as f:
        return json.load(f)


def run_backtest_with_trades(dates, params):
    """Run backtest and extract individual trades."""
    all_trades = []

    for date_str in dates:
        print(f"Backtesting {date_str}...", end=' ')

        try:
            picks = get_watchlist_date(date_str)
            if not picks:
                print("No picks")
                continue

            # Run backtest
            states, final_cash, unsettled, log = tgc.simulate_day_combined(
                picks=picks,
                cash=10000,
                cash_account=False,
                is_live=False,
                params=params
            )

            # Extract trades from states
            trades_today = []
            for st in states:
                if st.get("exit_time") and st.get("pnl") is not None:
                    trades_today.append({
                        "date": date_str,
                        "ticker": st["ticker"],
                        "strategy": st["strategy"],
                        "entry_time": str(st["entry_time"]),
                        "exit_time": str(st["exit_time"]),
                        "entry_price": st["entry_price"],
                        "exit_price": st.get("exit_price"),
                        "shares": st["shares"],
                        "pnl": st["pnl"],
                        "pnl_pct": st.get("pnl_pct", 0),
                    })

            all_trades.extend(trades_today)
            print(f"{len(trades_today)} trades")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return pd.DataFrame(all_trades)


def analyze_reentry(trades_df):
    """Analyze re-entry patterns."""
    print(f"\n{'='*60}")
    print("RE-ENTRY ANALYSIS")
    print(f"{'='*60}\n")

    # Count trades per ticker per day per strategy
    trades_df['combo'] = (
        trades_df['date'] + '_' +
        trades_df['ticker'] + '_' +
        trades_df['strategy']
    )

    reentry_counts = trades_df.groupby('combo').size()
    reentries = reentry_counts[reentry_counts > 1]

    print(f"Total days: {trades_df['date'].nunique()}")
    print(f"Total trades: {len(trades_df)}")
    print(f"Unique combos: {trades_df['combo'].nunique()}")
    print(f"Combos with re-entry: {len(reentries)}")
    if len(reentries) > 0:
        print(f"Re-entry rate: {len(reentries) / trades_df['combo'].nunique() * 100:.1f}%\n")

        # Top re-entry cases
        print("Top 10 re-entry cases:")
        for combo, count in reentries.sort_values(ascending=False).head(10).items():
            date, ticker, strat = combo.rsplit('_', 2)
            print(f"  {date} {ticker} ({strat}): {count} trades")
        print()

        # By strategy
        reentry_by_strat = defaultdict(int)
        for combo in reentries.index:
            strat = combo.split('_')[-1]
            reentry_by_strat[strat] += 1

        print("Re-entries by strategy:")
        for strat, count in sorted(reentry_by_strat.items(), key=lambda x: -x[1]):
            print(f"  {strat}: {count} cases")
        print()

        # Show examples
        print("Example sequences (top 3):")
        for i, combo in enumerate(reentries.head(3).index, 1):
            date, ticker, strat = combo.rsplit('_', 2)
            seq = trades_df[trades_df['combo'] == combo].sort_values('entry_time')
            print(f"\n{i}. {date} - {ticker} ({strat}) - {len(seq)} trades:")
            for j, (_, t) in enumerate(seq.iterrows(), 1):
                print(f"   #{j}: ${t['entry_price']:.2f} → ${t['exit_price']:.2f} | "
                      f"PnL: ${t['pnl']:.2f} ({t['pnl_pct']:.1f}%)")
    else:
        print("No re-entries found!\n")


def simulate_no_reentry(trades_df):
    """Calculate impact of blocking re-entry."""
    print(f"\n{'='*60}")
    print("NO RE-ENTRY SIMULATION")
    print(f"{'='*60}\n")

    trades_sorted = trades_df.sort_values(['date', 'entry_time']).copy()

    first_trades = []
    reentries = []
    traded = defaultdict(set)  # date -> set of (ticker, strategy)

    for _, trade in trades_sorted.iterrows():
        key = (trade['ticker'], trade['strategy'])
        if key in traded[trade['date']]:
            reentries.append(trade)
        else:
            traded[trade['date']].add(key)
            first_trades.append(trade)

    first_df = pd.DataFrame(first_trades)
    reentry_df = pd.DataFrame(reentries)

    all_pnl = trades_df['pnl'].sum()
    first_pnl = first_df['pnl'].sum()
    reentry_pnl = reentry_df['pnl'].sum() if len(reentry_df) > 0 else 0

    print(f"CURRENT (allows re-entry):")
    print(f"  Trades: {len(trades_df)}")
    print(f"  Total PnL: ${all_pnl:,.2f}")
    print(f"  Avg PnL/trade: ${all_pnl / len(trades_df):.2f}\n")

    print(f"NO RE-ENTRY:")
    print(f"  Trades: {len(first_df)}")
    print(f"  Total PnL: ${first_pnl:,.2f}")
    print(f"  Avg PnL/trade: ${first_pnl / len(first_df):.2f}\n")

    print(f"DIFFERENCE:")
    print(f"  Blocked trades: {len(reentry_df)}")
    print(f"  PnL from blocked: ${reentry_pnl:,.2f}")
    print(f"  Net impact: ${all_pnl - first_pnl:,.2f}")
    print(f"  {'✓ Re-entry HELPS' if reentry_pnl > 0 else '✗ Re-entry HURTS'}\n")

    if len(reentry_df) > 0:
        wins = (reentry_df['pnl'] > 0).sum()
        losses = (reentry_df['pnl'] < 0).sum()
        print(f"Blocked re-entries:")
        print(f"  Winners: {wins} (${reentry_df[reentry_df['pnl'] > 0]['pnl'].sum():.2f})")
        print(f"  Losers: {losses} (${reentry_df[reentry_df['pnl'] < 0]['pnl'].sum():.2f})")
        print(f"  Win rate: {wins / len(reentry_df) * 100:.1f}%")


def main():
    print("="*60)
    print("RE-ENTRY ANALYSIS: Backtest on Recent Dates")
    print("="*60 + "\n")

    params = load_params()

    # Test dates - last 2 weeks of data we have
    dates = [
        "2026-07-01", "2026-07-02", "2026-07-07", "2026-07-08",
        "2026-07-09", "2026-07-10", "2026-07-14", "2026-07-15"
    ]

    print(f"Running backtest on {len(dates)} dates...\n")
    trades_df = run_backtest_with_trades(dates, params)

    if len(trades_df) == 0:
        print("No trades found!")
        return

    # Save
    trades_df.to_csv("backtest_trades_recent.csv", index=False)
    print(f"\n✓ Saved {len(trades_df)} trades to backtest_trades_recent.csv")

    # Analyze
    analyze_reentry(trades_df)
    simulate_no_reentry(trades_df)

    print(f"\n{'='*60}")
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
