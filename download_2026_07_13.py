"""
Quick download for 2026-07-13 only.
"""
import os, sys, re, time
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

DATA_DIR     = "stored_data_jul_2026"
GAINERS_CSV  = os.path.join(DATA_DIR, "daily_top_gainers.csv")
INTRADAY_DIR = os.path.join(DATA_DIR, "intraday")
DAILY_DIR    = os.path.join(DATA_DIR, "daily")

MIN_GAP_PCT       = 2.0
MAX_PRICE         = 50.0
TOP_N             = 20
RATE_LIMIT_DELAY  = 0.35

TARGET_DATE = datetime(2026, 7, 13)

def _is_warrant_or_unit(ticker):
    RESERVED_WIN = {"con","prn","aux","nul", "com1","com2","com3","com4","com5",
                    "com6","com7","com8","com9", "lpt1","lpt2","lpt3","lpt4","lpt5",
                    "lpt6","lpt7","lpt8","lpt9"}
    if ticker.lower() in RESERVED_WIN: return True
    if ".WS" in ticker or ".RT" in ticker: return True
    if re.match(r"^[A-Z]{3,}W$", ticker): return True
    if ticker.endswith("WW"): return True
    if re.match(r"^[A-Z]{3,}U$", ticker): return True
    if re.match(r"^[A-Z]{3,}R$", ticker): return True
    return False

print(f"Downloading {TARGET_DATE.strftime('%Y-%m-%d')}...")

os.makedirs(INTRADAY_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)

data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# Get all active US stocks
print("Fetching asset list...")
all_assets = tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
tickers = [a.symbol for a in all_assets if a.tradable]
print(f"  {len(tickers)} tradable symbols")

# Batch the requests (Alpaca has a 13k symbol limit)
prev_closes = {}
target_bars_all = {}

BATCH_SIZE = 100
print(f"Fetching data in batches of {BATCH_SIZE}...")

for batch_start in range(0, len(tickers), BATCH_SIZE):
    batch_tickers = tickers[batch_start:batch_start+BATCH_SIZE]
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Batch {batch_num}/{total_batches} ({len(batch_tickers)} symbols)")

    # Get previous day's close for gap calculation
    prev_date = TARGET_DATE - timedelta(days=1)
    while prev_date.weekday() > 4:  # skip weekends
        prev_date -= timedelta(days=1)

    prev_req = StockBarsRequest(symbol_or_symbols=batch_tickers, timeframe=TimeFrame.Day,
                                start=prev_date, end=prev_date + timedelta(days=1),
                                limit=1)
    prev_batch = data_client.get_stock_bars(prev_req)
    for ticker in batch_tickers:
        if ticker in prev_batch:
            prev_closes[ticker] = float(prev_batch[ticker].df.iloc[-1]['close']) if len(prev_batch[ticker].df) > 0 else None

    # Get target date 1-min bars
    target_req = StockBarsRequest(symbol_or_symbols=batch_tickers, timeframe=TimeFrame.Minute,
                                  start=TARGET_DATE, end=TARGET_DATE + timedelta(days=1),
                                  limit=1000)
    target_batch = data_client.get_stock_bars(target_req)
    target_bars_all.update(target_batch)

    time.sleep(1)  # Rate limiting

target_bars = target_bars_all
print(f"Got bars for {len(target_bars)} symbols")

# Identify gap-ups
gainers = []
for ticker in target_bars.keys():
    if ticker in prev_closes and prev_closes[ticker] and len(target_bars[ticker].df) > 0:
        open_price = float(target_bars[ticker].df.iloc[0]['open'])
        prev_close = prev_closes[ticker]
        gap_pct = (open_price / prev_close - 1) * 100
        if gap_pct >= MIN_GAP_PCT and open_price <= MAX_PRICE:
            gainers.append({
                "ticker": ticker,
                "prev_close": prev_close,
                "open": open_price,
                "gap_pct": gap_pct,
            })

gainers.sort(key=lambda x: x["gap_pct"], reverse=True)
top_gainers = gainers[:TOP_N]
print(f"  {len(gainers)} gap-ups >= {MIN_GAP_PCT}% found")
print(f"  Top {len(top_gainers)}:")
for g in top_gainers:
    print(f"    {g['ticker']:>6}  gap={g['gap_pct']:>+6.2f}%  prev_close=${g['prev_close']:.2f} -> open=${g['open']:.2f}")

# Save daily bars for top gainers
for g in top_gainers:
    ticker = g['ticker']
    if ticker in target_bars:
        df = target_bars[ticker].df.copy()
        df.to_csv(os.path.join(DAILY_DIR, f"{ticker}.csv"))
        print(f"  Saved {len(df)} daily bars for {ticker}")

# Save intraday 1-min bars for top gainers
print(f"\nSaving intraday 1-min bars...")
for g in top_gainers:
    ticker = g['ticker']
    if ticker in target_bars:
        df = target_bars[ticker].df.copy()
        if len(df) > 0:
            df.to_csv(os.path.join(INTRADAY_DIR, f"{ticker}.csv"))
            print(f"  {ticker}: {len(df)} bars")

print(f"\nDone. Data saved to {DATA_DIR}/")
