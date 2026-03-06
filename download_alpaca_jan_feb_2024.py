"""
Download 2-minute intraday bars for Jan-Feb 2024 gap-up stocks from Alpaca.
Alpaca free tier has no historical data time limit (unlike Polygon's 2-year window).

Usage:
    # Set env vars first:
    export ALPACA_API_KEY="your-key"
    export ALPACA_API_SECRET="your-secret"

    python download_alpaca_jan_feb_2024.py

    # Or pass keys directly:
    python download_alpaca_jan_feb_2024.py --key YOUR_KEY --secret YOUR_SECRET
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

DATA_DIR = "stored_data_jan_feb_2024"
GAINERS_CSV = os.path.join(DATA_DIR, "daily_top_gainers.csv")
INTRADAY_DIR = os.path.join(DATA_DIR, "intraday")

# Alpaca free tier: 200 requests/minute
RATE_LIMIT_DELAY = 0.35  # seconds between requests (~170/min, safe margin)


def main():
    parser = argparse.ArgumentParser(description="Download Jan-Feb 2024 intraday from Alpaca")
    parser.add_argument("--key", default=os.environ.get("ALPACA_API_KEY"), help="Alpaca API key")
    parser.add_argument("--secret", default=os.environ.get("ALPACA_API_SECRET"), help="Alpaca API secret")
    parser.add_argument("--resume", action="store_true", help="Skip tickers that already have intraday files")
    args = parser.parse_args()

    if not args.key or not args.secret:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET env vars, or use --key and --secret")
        sys.exit(1)

    client = StockHistoricalDataClient(args.key, args.secret)

    # Load tickers from daily_top_gainers.csv
    df = pd.read_csv(GAINERS_CSV)
    tickers = sorted(df["ticker"].unique())
    print(f"Loaded {len(tickers)} unique tickers from {GAINERS_CSV}")

    # Get date range
    dates = pd.to_datetime(df["date"])
    start_date = dates.min().strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")
    print(f"Date range: {start_date} to {end_date}")

    os.makedirs(INTRADAY_DIR, exist_ok=True)

    # Check existing files if resuming
    existing = set()
    if args.resume:
        existing = {f.replace(".csv", "") for f in os.listdir(INTRADAY_DIR) if f.endswith(".csv")}
        print(f"Resume mode: {len(existing)} tickers already downloaded, skipping them")

    downloaded = 0
    skipped = 0
    errors = 0

    for i, ticker in enumerate(tickers):
        if ticker in existing:
            skipped += 1
            continue

        try:
            # Alpaca wants start/end as datetime, and returns data in UTC
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=datetime(2024, 1, 1),
                end=datetime(2024, 3, 1),  # exclusive end, covers all of Feb
                adjustment="raw",
            )
            # Get 1-minute bars, then resample to 2-minute
            bars = client.get_stock_bars(request)

            if bars.df.empty:
                print(f"  [{i+1}/{len(tickers)}] {ticker}: no data")
                errors += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue

            # bars.df has multi-index (symbol, timestamp)
            bar_df = bars.df.reset_index()
            if "symbol" in bar_df.columns:
                bar_df = bar_df[bar_df["symbol"] == ticker].copy()
                bar_df = bar_df.drop(columns=["symbol"])

            bar_df = bar_df.rename(columns={"timestamp": "Datetime"})
            bar_df = bar_df.set_index("Datetime")

            # Resample 1-min to 2-min bars
            bar_2m = bar_df.resample("2min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trade_count": "sum",
                "vwap": "last",
            }).dropna(subset=["open"])

            # Format to match existing CSV format: Datetime,Open,High,Low,Close,Volume
            out = bar_2m[["open", "high", "low", "close", "volume"]].copy()
            out.columns = ["Open", "High", "Low", "Close", "Volume"]
            out.index.name = "Datetime"

            out_path = os.path.join(INTRADAY_DIR, f"{ticker}.csv")
            out.to_csv(out_path)
            downloaded += 1

            n_bars = len(out)
            print(f"  [{i+1}/{len(tickers)}] {ticker}: {n_bars} bars saved")

        except Exception as e:
            err_str = str(e)
            # Truncate long error messages
            if len(err_str) > 100:
                err_str = err_str[:100] + "..."
            print(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR - {err_str}")
            errors += 1

        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone! Downloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")
    print(f"Files in {INTRADAY_DIR}/: {len(os.listdir(INTRADAY_DIR))}")


if __name__ == "__main__":
    main()
