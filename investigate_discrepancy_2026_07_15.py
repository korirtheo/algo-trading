"""Detailed investigation: backtest vs live discrepancy on 2026-07-15."""
import re
import pandas as pd
from pathlib import Path
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import json

ET = ZoneInfo("America/New_York")
DATA_DIR = Path("stored_data_jul_2026")
LIVE_LOG = Path("logs/2026-07-15/live_log_2026-07-15")

# Tickers that traded in backtest
BACKTEST_TRADES = ['VIVS', 'ELVA', 'KUST', 'ERNA', 'KOPN', 'GEVO', 'SOBR', 'TGHL']


def load_backtest_bars(ticker):
    """Load and aggregate to 2-min bars (same as backtest)."""
    path = DATA_DIR / f"{ticker}_2026-07-15.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.time >= dt_time(9, 30)]
    df = df[df['timestamp'].dt.time < dt_time(16, 0)]

    if df.empty:
        return df

    # Resample to 2-min aligned to 9:30
    df = df.set_index('timestamp')
    df = df.resample('2min', origin=datetime(2026, 7, 15, 9, 30, tzinfo=ET)).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }).dropna()

    return df.reset_index()


def extract_live_bars(ticker, log_path):
    """Extract live 2-min bars from log file."""
    bars = []

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Look for "EMIT 2min: TICKER c=price count=N"
            if f'EMIT 2min: {ticker}' in line:
                # Parse timestamp from line start: "HH:MM:SS ET"
                time_match = re.match(r'(\d{2}:\d{2}:\d{2})', line)
                if not time_match:
                    continue

                time_str = time_match.group(1)

                # Parse close price and count
                close_match = re.search(r'c=([\d.]+)', line)
                count_match = re.search(r'count=(\d+)', line)

                if close_match:
                    bars.append({
                        'time': time_str,
                        'close': float(close_match.group(1)),
                        'count': int(count_match.group(1)) if count_match else None,
                    })

    return pd.DataFrame(bars)


def extract_live_signals(ticker, log_path):
    """Extract entry signals from live log."""
    signals = []

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Look for entry signals
            if ticker in line and 'entry signal' in line.lower():
                signals.append(line.strip())
            # Also check for BUY orders
            if ticker in line and 'BUY' in line and 'entry' in line.lower():
                signals.append(line.strip())

    return signals


def check_position_conflicts(log_path):
    """Check what positions were active and when."""
    positions = []

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Look for position entries
            if 'BUY' in line and 'submitted' in line.lower():
                time_match = re.match(r'(\d{2}:\d{2}:\d{2})', line)
                ticker_match = re.search(r'BUY (\w+)', line)
                if time_match and ticker_match:
                    positions.append({
                        'time': time_match.group(1),
                        'action': 'BUY',
                        'ticker': ticker_match.group(1),
                    })

            # Look for exits
            if 'SELL' in line and ('filled' in line.lower() or 'submitted' in line.lower()):
                time_match = re.match(r'(\d{2}:\d{2}:\d{2})', line)
                ticker_match = re.search(r'SELL (\w+)', line)
                if time_match and ticker_match:
                    positions.append({
                        'time': time_match.group(1),
                        'action': 'SELL',
                        'ticker': ticker_match.group(1),
                    })

    return pd.DataFrame(positions)


def main():
    print("=" * 80)
    print("DETAILED INVESTIGATION: Backtest vs Live Discrepancy")
    print("=" * 80)

    # Check position activity
    print("\n[1] POSITION ACTIVITY")
    print("-" * 80)
    positions = check_position_conflicts(LIVE_LOG)
    if not positions.empty:
        for _, pos in positions.iterrows():
            print(f"  {pos['time']} - {pos['action']:4s} {pos['ticker']}")
    else:
        print("  No BUY/SELL activity found (checking differently...)")

    print("\n[2] TICKER-BY-TICKER ANALYSIS")
    print("-" * 80)

    for ticker in BACKTEST_TRADES:
        print(f"\n{'=' * 80}")
        print(f"[{ticker}]")
        print('=' * 80)

        # Load backtest bars
        bt_bars = load_backtest_bars(ticker)
        print(f"Backtest bars: {len(bt_bars)}")
        if len(bt_bars) > 0:
            print(f"  First bar: {bt_bars.iloc[0]['timestamp']} close=${bt_bars.iloc[0]['Close']:.2f}")
            print(f"  Last bar:  {bt_bars.iloc[-1]['timestamp']} close=${bt_bars.iloc[-1]['Close']:.2f}")

        # Extract live bars
        live_bars = extract_live_bars(ticker, LIVE_LOG)
        print(f"\nLive 2-min bars: {len(live_bars)}")
        if len(live_bars) > 0:
            for i, bar in live_bars.iterrows():
                print(f"  {bar['time']} - close=${bar['close']:.2f} (count={bar['count']})")

        # Check for signals
        signals = extract_live_signals(ticker, LIVE_LOG)
        print(f"\nLive signals: {len(signals)}")
        for sig in signals:
            print(f"  {sig[:150]}")

        # Compare bar counts
        if len(bt_bars) > 0 and len(live_bars) > 0:
            print(f"\n** Bar count: Backtest={len(bt_bars)}, Live={len(live_bars)} (diff={len(bt_bars)-len(live_bars)})")
        elif len(bt_bars) > 0 and len(live_bars) == 0:
            print(f"\n** ISSUE: Backtest had {len(bt_bars)} bars, Live had ZERO bars!")

        # Summary
        if ticker == 'SOBR':
            print("\n>>> SOBR was NOT in live watchlist (scanner miss)")
        elif len(live_bars) == 0:
            print("\n>>> Live received NO bars - streaming issue?")
        elif len(signals) == 0:
            print("\n>>> Live received bars but generated NO signals")
        else:
            print("\n>>> Live had signals - check position conflicts")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Check above for:")
    print("  1. Tickers with NO live bars → streaming/subscription issue")
    print("  2. Tickers with bars but NO signals → strategy logic difference")
    print("  3. Tickers with signals but no trades → position slot conflict")


if __name__ == "__main__":
    main()
