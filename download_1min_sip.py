"""Download 1-min Alpaca SIP historical bars for the replay universe.

Universe: 3,449 known tickers from gainers_index_2024_2026.json (the top-20
gappers per day, 2024-01-02 -> 2026-08-07). We only fetch tickers we already
know appeared — no full-market scan.

Output (mirrors existing stored_data_*/ layout):
  stored_data_1min/intraday/<TICKER>.csv   # 1-min bars, UTC tz-aware, Datetime index
  stored_data_1min/daily/<TICKER>.csv      # daily bars, UTC tz-aware, Datetime index
  stored_data_1min/daily_top_gainers.csv   # merged (date, ticker, gap_pct)

Each ticker = 2 API calls (1-min + daily) covering its full appearance range.
Resumable: skips tickers whose intraday file already exists and has data.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

ET = ZoneInfo("America/New_York")
OUT_DIR = "stored_data_1min"
INTRADAY_DIR = os.path.join(OUT_DIR, "intraday")
DAILY_DIR = os.path.join(OUT_DIR, "daily")
GAINERS_CSV = os.path.join(OUT_DIR, "daily_top_gainers.csv")
RATE_LIMIT_DELAY = 0.6

WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

with open("gainers_index_2024_2026.json") as f:
    INDEX = json.load(f)
ranges = INDEX["ticker_ranges"]

os.makedirs(INTRADAY_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)


def _is_warrant_or_unit(t):
    import re
    if t in WIN_RESERVED or ".WS" in t or ".RT" in t:
        return True
    if re.match(r"^[A-Z]{3,}W$", t) or t.endswith("WW") or re.match(r"^[A-Z]{3,}U$", t) or re.match(r"^[A-Z]{3,}R$", t):
        return True
    return False


def fetch_minute(ticker, start, end):
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        adjustment="raw",
        feed="sip",
    )
    resp = client.get_stock_bars(req)
    if resp.df is None or resp.df.empty:
        return None
    df = resp.df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == ticker].drop(columns=["symbol"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out.index.name = "Datetime"
    return out


def fetch_daily(ticker, start, end):
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="raw",
        feed="sip",
    )
    resp = client.get_stock_bars(req)
    if resp.df is None or resp.df.empty:
        return None
    df = resp.df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == ticker].drop(columns=["symbol"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out.index.name = "Datetime"
    return out


def already_have(ticker):
    p = os.path.join(INTRADAY_DIR, f"{ticker}.csv")
    if not os.path.exists(p):
        return False
    try:
        return os.path.getsize(p) > 60
    except OSError:
        return False


def main():
    skip = 0
    done = 0
    errors = []
    tickers = sorted(ranges.keys())

    for i, ticker in enumerate(tickers):
        if ticker in WIN_RESERVED or _is_warrant_or_unit(ticker):
            skip += 1
            continue
        if already_have(ticker):
            skip += 1
            continue

        d0, d1 = ranges[ticker]
        start = datetime.strptime(d0, "%Y-%m-%d").replace(hour=4, tzinfo=ET)
        end = datetime.strptime(d1, "%Y-%m-%d").replace(hour=0, tzinfo=ET) + timedelta(days=1)

        try:
            mdf = fetch_minute(ticker, start, end)
            time.sleep(RATE_LIMIT_DELAY)
            ddf = fetch_daily(ticker, start, end)
        except Exception as e:
            errors.append((ticker, str(e)[:100]))
            print(f"[{i+1}/{len(tickers)}] {ticker}: ERR {str(e)[:100]}", flush=True)
            time.sleep(RATE_LIMIT_DELAY * 2)
            continue

        if mdf is not None and len(mdf) > 0:
            mdf.to_csv(os.path.join(INTRADAY_DIR, f"{ticker}.csv"))
        if ddf is not None and len(ddf) > 0:
            ddf.to_csv(os.path.join(DAILY_DIR, f"{ticker}.csv"))
        done += 1
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{len(tickers)}] {ticker}: {len(mdf) if mdf is not None else 0} 1min, "
                  f"{len(ddf) if ddf is not None else 0} daily (done={done} skip={skip} err={len(errors)})", flush=True)

    # Write merged gainers csv
    rows = []
    for d, tk in INDEX["index"].items():
        for t, gap in tk.items():
            rows.append({"date": d, "ticker": t, "gap_pct": gap})
    with open(GAINERS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "ticker", "gap_pct"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nDONE: downloaded={done} skipped={skip} errors={len(errors)}")
    for t, e in errors[:20]:
        print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
