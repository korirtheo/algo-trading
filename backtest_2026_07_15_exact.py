"""Exact backtest for 2026-07-15 using the SAME engine code as live."""
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

# Use the SAME engine as live
from live.engine_combined import CombinedEngine, load_trial_params
import test_green_candle_combined as tgc

ET = ZoneInfo("America/New_York")
DATA_DIR = Path("stored_data_jul_2026")

# Watchlist from live (at 9:42 AM restart)
WATCHLIST = ['VIVS', 'ELVA', 'KUST', 'ERNA', 'VTAK', 'TGHL', 'TRT', 'IZM',
             'GNTA', 'VEEE', 'KOPN', 'NVVE', 'GCTK', 'NTHI', 'QNC', 'SHMD',
             'YJ', 'MTEX', 'GEVO', 'BCDA']


def load_and_prepare_data(ticker):
    """Load 1-min bars and convert to proper timezone."""
    path = DATA_DIR / f"{ticker}_2026-07-15.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['timestamp_et'] = df['timestamp'].dt.tz_convert(ET)
    df['time_et'] = df['timestamp_et'].dt.time

    # MARKET HOURS ONLY: 9:30 - 16:00 ET (same as live)
    df = df[(df['time_et'] >= dt_time(9, 30)) & (df['time_et'] < dt_time(16, 0))]

    if df.empty:
        return df

    # Aggregate to 2-min bars (same as live)
    df = df.set_index('timestamp_et')
    df = df.resample('2min', origin=datetime(2026, 7, 15, 9, 30, tzinfo=ET)).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }).dropna()

    return df.reset_index()


class MockExecutor:
    """Mock executor for backtest."""
    def __init__(self, initial_cash=25000):
        self.cash = initial_cash
        self.positions = {}

    def get_buying_power(self):
        return self.cash

    def get_account(self):
        class Acct:
            def __init__(self, cash):
                self.cash = cash
                self.buying_power = cash
        return Acct(self.cash)

    def get_positions(self):
        return []

    def buy(self, ticker, **kwargs):
        pass

    def sell(self, ticker, **kwargs):
        pass


def main():
    print("=" * 80)
    print("EXACT BACKTEST: 2026-07-15 (Using live engine code)")
    print("=" * 80)

    # Load params (same as live)
    params = load_trial_params()

    # Create engine
    executor = MockExecutor()
    engine = CombinedEngine(executor)

    # Initialize with scanner picks
    candidates = []
    for ticker in WATCHLIST:
        candidates.append({
            'ticker': ticker,
            'gap_pct': 10.0,  # dummy
            'pm_volume': 100000,
            'premarket_high': 1.0,
            'prev_close': 0.9,
            'float_shares': 1_000_000,
        })

    engine.initialize_watchlist(candidates)

    # Load bar data
    print(f"\nLoading bar data for {len(WATCHLIST)} tickers...")
    all_bars = {}
    for ticker in WATCHLIST:
        df = load_and_prepare_data(ticker)
        if not df.empty:
            all_bars[ticker] = df
            print(f"  {ticker}: {len(df)} bars ({df.iloc[0]['timestamp_et'].strftime('%H:%M')}-{df.iloc[-1]['timestamp_et'].strftime('%H:%M')})")

    # Simulate bar-by-bar (same as live)
    print("\nSimulating bar-by-bar...")

    # Collect all timestamps
    all_timestamps = set()
    for ticker, df in all_bars.items():
        all_timestamps.update(df['timestamp_et'].tolist())

    all_timestamps = sorted(all_timestamps)
    print(f"Total unique timestamps: {len(all_timestamps)}")

    # Process each timestamp
    for ts in all_timestamps:
        # Feed bars at this timestamp to engine
        for ticker, df in all_bars.items():
            bar_row = df[df['timestamp_et'] == ts]
            if not bar_row.empty:
                bar = bar_row.iloc[0]
                bar_dict = {
                    'Open': bar['Open'],
                    'High': bar['High'],
                    'Low': bar['Low'],
                    'Close': bar['Close'],
                    'Volume': int(bar['Volume']),
                    'timestamp': ts,
                }
                # Call the same on_bar method as live
                engine.on_bar(ticker, bar_dict)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    summary = engine.get_summary()
    print(f"Trades: {summary['trades']}")
    print(f"Wins: {summary['wins']}")
    print(f"Losses: {summary['losses']}")
    print(f"PnL: ${summary['daily_pnl']:+,.2f}")

    if summary['trade_details']:
        print("\nTrade Details:")
        for t in summary['trade_details']:
            print(f"  {t['ticker']} ({t.get('strategy','?')}): "
                  f"${t['entry_price']:.2f} -> ${t['exit_price']:.2f} = "
                  f"${t['pnl']:+,.2f} ({t['reason']})")

    print("\n" + "=" * 80)
    print("COMPARISON WITH LIVE")
    print("=" * 80)
    print(f"Live: 1 trade (TGHL) PnL: -$330.59")
    print(f"Backtest: {summary['trades']} trades, PnL: ${summary['daily_pnl']:+,.2f}")


if __name__ == "__main__":
    main()
