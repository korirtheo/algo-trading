"""Download 2026-07-15 watchlist data for backtest comparison."""
import os
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED

ET = ZoneInfo("America/New_York")

# 7/15/2026 watchlist from live log
WATCHLIST = ['VIVS', 'ELVA', 'KUST', 'ERNA', 'VTAK', 'TGHL', 'TRT', 'IZM',
             'GNTA', 'VEEE', 'KOPN', 'NVVE', 'GCTK', 'NTHI', 'QNC', 'SHMD',
             'YJ', 'MTEX', 'GEVO', 'BCDA', 'SOBR']  # Added SOBR to check

OUTPUT_DIR = "stored_data_jul_2026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

target_date = datetime(2026, 7, 15, tzinfo=ET)
start = target_date.replace(hour=4, minute=0)
end = target_date.replace(hour=20, minute=0)

print(f"Downloading 2026-07-15 watchlist data...")
print(f"Tickers: {len(WATCHLIST)}")

for ticker in WATCHLIST:
    print(f"\n[{ticker}]", flush=True)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            adjustment="raw",
            feed=ALPACA_FEED,
        )
        bars = client.get_stock_bars(req)
        if bars.df.empty:
            print(f"  No data")
            continue

        df = bars.df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
        })

        out_path = os.path.join(OUTPUT_DIR, f"{ticker}_2026-07-15.csv")
        df.to_csv(out_path, index=False)
        print(f"  {len(df)} bars saved")

    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nDownload complete. Saved to {OUTPUT_DIR}/")
