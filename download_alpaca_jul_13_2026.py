"""
Comprehensive download for 2026-07-13 - all symbols with trading activity.
"""
import os, sys, time
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "stored_data_jul_2026"
INTRADAY_DIR = os.path.join(DATA_DIR, "intraday")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
TARGET_DATE = datetime(2026, 7, 13)
BATCH_SIZE = 100

os.makedirs(INTRADAY_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)

print(f"Downloading all data for {TARGET_DATE.strftime('%Y-%m-%d')}...")

data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# Get all active US stocks
print("Fetching asset list...")
all_assets = tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
tickers = [a.symbol for a in all_assets if a.tradable and a.fractionable]
print(f"  {len(tickers)} tradable symbols")

# Previous trading day for prev_close
prev_date = TARGET_DATE - timedelta(days=1)
while prev_date.weekday() > 4:  # skip weekends
    prev_date -= timedelta(days=1)

# Download in batches
saved_count = 0
total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_start in range(0, len(tickers), BATCH_SIZE):
    batch_tickers = tickers[batch_start:batch_start+BATCH_SIZE]
    batch_num = batch_start // BATCH_SIZE + 1
    print(f"Batch {batch_num}/{total_batches} ({len(batch_tickers)} symbols)...", flush=True)

    # Get prev day close
    try:
        prev_req = StockBarsRequest(
            symbol_or_symbols=batch_tickers,
            timeframe=TimeFrame.Day,
            start=prev_date,
            end=prev_date + timedelta(days=1),
            limit=1
        )
        prev_batch = data_client.get_stock_bars(prev_req)
    except Exception as e:
        print(f"  ERROR fetching prev day: {e}")
        time.sleep(1)
        continue

    # Get target date 1-min bars
    try:
        target_req = StockBarsRequest(
            symbol_or_symbols=batch_tickers,
            timeframe=TimeFrame.Minute,
            start=TARGET_DATE,
            end=TARGET_DATE + timedelta(days=1),
            limit=1000
        )
        target_batch = data_client.get_stock_bars(target_req)
    except Exception as e:
        print(f"  ERROR fetching intraday: {e}")
        time.sleep(1)
        continue

    # Save CSVs for tickers with data
    for ticker in batch_tickers:
        try:
            if ticker not in target_batch:
                continue
            df = target_batch[ticker].df
            if len(df) == 0:
                continue

            # Save intraday
            df.to_csv(os.path.join(INTRADAY_DIR, f"{ticker}.csv"))

            # Save daily (if exists)
            if ticker in prev_batch and len(prev_batch[ticker].df) > 0:
                prev_batch[ticker].df.to_csv(os.path.join(DAILY_DIR, f"{ticker}.csv"))

            saved_count += 1
        except Exception:
            continue

    time.sleep(0.2)  # Rate limiting

print(f"\nDone. Saved {saved_count} tickers with intraday bars to {DATA_DIR}/")
