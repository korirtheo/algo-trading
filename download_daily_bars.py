"""
Download daily bars from Alpaca for tickers that have intraday data but missing daily CSVs.
Saves per-ticker daily CSVs to {data_dir}/daily/ for pickle building.

Usage:
    python download_daily_bars.py stored_data_2023
    python download_daily_bars.py stored_data_2022
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

ALPACA_KEY = "PK34OGXUBAOCLG7E6KYTI6QMZ3"
ALPACA_SECRET = "DBn7mXAKdTBAR9XZnkhnu1CykZDYNZEVzkBojKDtoYbJ"

RATE_LIMIT_DELAY = 0.35
BATCH_SIZE = 500

# Date ranges per data directory
DATE_RANGES = {
    "stored_data_2023": ("2022-11-01", "2024-01-01"),
    "stored_data_2022": ("2021-11-01", "2023-01-01"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Data directory (e.g. stored_data_2023)")
    args = parser.parse_args()

    data_dir = args.data_dir
    intraday_dir = os.path.join(data_dir, "intraday")
    daily_dir = os.path.join(data_dir, "daily")
    os.makedirs(daily_dir, exist_ok=True)

    if not os.path.isdir(intraday_dir):
        print(f"ERROR: {intraday_dir} not found")
        sys.exit(1)

    # Get tickers from intraday that don't have daily CSVs yet
    all_tickers = sorted(f.replace(".csv", "") for f in os.listdir(intraday_dir) if f.endswith(".csv"))
    existing = set(f.replace(".csv", "") for f in os.listdir(daily_dir) if f.endswith(".csv"))
    needed = [t for t in all_tickers if t not in existing]

    print(f"Tickers with intraday data: {len(all_tickers)}")
    print(f"Already have daily CSVs: {len(existing)}")
    print(f"Need to download: {len(needed)}")

    if not needed:
        print("Nothing to download!")
        return

    # Date range
    base = os.path.basename(os.path.abspath(data_dir))
    if base in DATE_RANGES:
        start_str, end_str = DATE_RANGES[base]
    else:
        print(f"WARNING: Unknown data_dir '{base}', using 2022-11-01 to 2024-01-01")
        start_str, end_str = "2022-11-01", "2024-01-01"

    start_date = pd.Timestamp(start_str)
    end_date = pd.Timestamp(end_str)
    print(f"Date range: {start_str} to {end_str}")

    client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    total_saved = 0
    total_errors = 0

    for batch_start in range(0, len(needed), BATCH_SIZE):
        batch = needed[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(needed) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nBatch {batch_num}/{total_batches}: {len(batch)} tickers...", end="", flush=True)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                adjustment="raw",
            )
            bars = client.get_stock_bars(request)

            if bars.df.empty:
                print(f" no data returned")
                total_errors += len(batch)
                time.sleep(RATE_LIMIT_DELAY * 2)
                continue

            df = bars.df.reset_index()
            batch_saved = 0

            for ticker in df["symbol"].unique():
                tdf = df[df["symbol"] == ticker].copy()
                tdf = tdf.set_index("timestamp").sort_index()
                tdf = tdf.drop(columns=["symbol"], errors="ignore")

                out = tdf[["open", "high", "low", "close", "volume"]].copy()
                out.columns = ["Open", "High", "Low", "Close", "Volume"]
                out.index.name = "Date"

                out_path = os.path.join(daily_dir, f"{ticker}.csv")
                out.to_csv(out_path)
                batch_saved += 1

            not_found = len(batch) - batch_saved
            print(f" saved {batch_saved}, missing {not_found}")
            total_saved += batch_saved
            total_errors += not_found

        except Exception as e:
            print(f" ERROR: {str(e)[:100]}")
            total_errors += len(batch)

        time.sleep(RATE_LIMIT_DELAY * 2)

    print(f"\nDone! Saved: {total_saved}, Errors/Missing: {total_errors}")
    print(f"Files in {daily_dir}/: {len(os.listdir(daily_dir))}")


if __name__ == "__main__":
    main()
