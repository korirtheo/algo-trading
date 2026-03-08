"""
Re-download 2023 intraday data with SIP feed (includes premarket bars).
The original download used default IEX feed which only has regular hours.

Usage:
    python redownload_2023_sip.py
    python redownload_2023_sip.py --resume   # skip already-fixed files
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

API_KEY = "PK34OGXUBAOCLG7E6KYTI6QMZ3"
API_SECRET = "DBn7mXAKdTBAR9XZnkhnu1CykZDYNZEVzkBojKDtoYbJ"
RATE_LIMIT_DELAY = 0.35

DATA_DIR = "stored_data_2023"
INTRADAY_DIR = os.path.join(DATA_DIR, "intraday")
INTRADAY_1MIN_DIR = os.path.join(DATA_DIR, "intraday_1min")

# Date range for 2023 data (with buffer for premarket on first/last day)
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 1, 1)


def has_premarket(filepath):
    """Check if a file already has premarket data."""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if len(df) == 0:
            return False
        if df.index.tz is not None:
            et = df.index.tz_convert("America/New_York")
        else:
            et = df.index.tz_localize("UTC").tz_convert("America/New_York")
        # Check if any bars before 9:30 AM
        early = et[(et.hour < 9) | ((et.hour == 9) & (et.minute < 30))]
        return len(early) > 0
    except Exception:
        return False


def main():
    resume = "--resume" in sys.argv

    # Get per-ticker dates from gainers
    gdf = pd.read_csv(os.path.join(DATA_DIR, "daily_top_gainers.csv"))
    ticker_date_map = gdf.groupby("ticker")["date"].apply(set).to_dict()

    # Get tickers to process
    tickers = [f.replace(".csv", "") for f in os.listdir(INTRADAY_DIR) if f.endswith(".csv")]
    tickers.sort()
    print(f"Tickers to re-download: {len(tickers)}")

    os.makedirs(INTRADAY_1MIN_DIR, exist_ok=True)
    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    skipped = 0
    downloaded = 0
    errors = 0

    for i, ticker in enumerate(tickers):
        filepath = os.path.join(INTRADAY_DIR, f"{ticker}.csv")

        if resume and has_premarket(filepath):
            skipped += 1
            continue

        # Get the dates this ticker needs
        needed_dates = ticker_date_map.get(ticker, set())
        if not needed_dates:
            continue

        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=START_DATE,
                end=END_DATE,
                adjustment="raw",
                feed="sip",
            )
            bars = client.get_stock_bars(req)

            if bars.df.empty:
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(tickers)}] {ticker}: no data", flush=True)
                errors += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue

            bar_df = bars.df.reset_index()
            if "symbol" in bar_df.columns:
                bar_df = bar_df[bar_df["symbol"] == ticker].copy()
                bar_df = bar_df.drop(columns=["symbol"])

            bar_df = bar_df.rename(columns={"timestamp": "Datetime"})
            bar_df = bar_df.set_index("Datetime")

            # Rename columns to match expected format
            bar_df = bar_df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            # Keep only OHLCV
            bar_df = bar_df[["Open", "High", "Low", "Close", "Volume"]]

            # Filter to only needed dates (per-ticker)
            if bar_df.index.tz is not None:
                dates = bar_df.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
            else:
                dates = bar_df.index.tz_localize("UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
            mask = dates.isin(needed_dates)
            bar_df = bar_df[mask]

            if len(bar_df) == 0:
                errors += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue

            # Save 1-min data
            bar_df.to_csv(os.path.join(INTRADAY_1MIN_DIR, f"{ticker}.csv"))

            # Resample 1-min to 2-min and save to intraday/
            bar_2m = bar_df.resample("2min").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna(subset=["Open"])

            bar_2m.to_csv(filepath)
            downloaded += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(tickers):
                print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(bar_df)} 1min / {len(bar_2m)} 2min bars", flush=True)

        except Exception as e:
            errors += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR {e}", flush=True)

        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone! Downloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")
    print(f"Files in {INTRADAY_DIR}: {len(os.listdir(INTRADAY_DIR))}")


if __name__ == "__main__":
    main()
