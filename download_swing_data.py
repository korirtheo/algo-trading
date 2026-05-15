"""
Download daily bars from Alpaca for swing trading analysis.
Fetches all active US stocks with sufficient volume, saves as parquet.

Usage:
  python download_swing_data.py                        # 2025 YTD, vol > 100K
  python download_swing_data.py --start 2024-01-01     # custom start
  python download_swing_data.py --min-volume 500000    # higher volume filter
  python download_swing_data.py --symbols AAPL TSLA    # specific tickers only
"""
import os
import sys
import argparse
import time
from datetime import datetime

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = "stored_data_swing"


def get_active_tickers():
    """Get active, tradable US stock tickers from Alpaca."""
    trading = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
    assets = trading.get_all_assets()
    tickers = [
        a.symbol for a in assets
        if a.status == "active"
        and a.tradable
        and a.exchange in ("NASDAQ", "NYSE", "AMEX", "ARCA", "BATS")
        and "." not in a.symbol
        and len(a.symbol) <= 5
    ]
    return sorted(set(tickers))


def download_bars(client, symbols, start, end, batch_size=200):
    """Download daily bars in batches."""
    all_frames = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches}: "
              f"{batch[0]}..{batch[-1]} ({len(batch)} symbols)", end="", flush=True)

        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
                adjustment="split",
                feed="sip",
            )
            bars = client.get_stock_bars(req)
            if not bars.df.empty:
                df = bars.df.reset_index()
                all_frames.append(df)
                print(f" -> {len(df)} bars, {df['symbol'].nunique()} symbols")
            else:
                print(" -> no data")
        except Exception as e:
            print(f" -> ERROR: {str(e)[:80]}")

        if i + batch_size < len(symbols):
            time.sleep(0.5)

    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download daily bars for swing trading")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Specific symbols")
    parser.add_argument("--min-volume", type=int, default=100_000,
                        help="Min avg daily volume to keep (default: 100K)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    end_str = args.end or datetime.now().strftime("%Y-%m-%d")

    print(f"Daily bars download: {args.start} to {end_str}")
    print(f"Min avg volume: {args.min_volume:,}")

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        print("\nFetching active tickers from Alpaca...")
        symbols = get_active_tickers()
    print(f"Symbols to download: {len(symbols)}")

    # Download
    print(f"\nDownloading...")
    df = download_bars(client, symbols, args.start, end_str)

    if df.empty:
        print("No data downloaded!")
        return

    # Clean columns
    df.columns = [c.lower() for c in df.columns]
    print(f"\nRaw: {len(df)} bars, {df['symbol'].nunique()} symbols, "
          f"{df['timestamp'].dt.date.nunique()} days")

    # Filter by minimum average volume
    avg_vol = df.groupby("symbol")["volume"].mean()
    liquid = avg_vol[avg_vol >= args.min_volume].index
    df = df[df["symbol"].isin(liquid)].copy()
    print(f"After volume filter (>={args.min_volume:,}): "
          f"{df['symbol'].nunique()} symbols, {len(df)} bars")

    # Save
    out_path = os.path.join(OUT_DIR, f"daily_{args.start}_{end_str}.pkl")
    df.to_pickle(out_path)
    print(f"\nSaved to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    # Stats
    print(f"\nDate range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Symbols: {df['symbol'].nunique()}")
    print(f"Trading days: {df['timestamp'].dt.date.nunique()}")
    print(f"Sample: {list(df['symbol'].unique()[:10])}")


if __name__ == "__main__":
    main()
