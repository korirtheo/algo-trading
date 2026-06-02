"""
Download Mar 15 - May 16 2026 top-20-gap-up data from Alpaca.

Step 1: Pull daily bars for all tradeable US equities in the range
Step 2: Score each trading day's open-vs-prev-close gap, keep top 20
Step 3: Download 2-min intraday for the unique candidate tickers
Step 4: Resample 1-min -> 2-min, write intraday/<TICKER>.csv + daily/<TICKER>.csv
        + daily_top_gainers.csv into stored_data_mar_may_2026/

Mirrors download_alpaca_2023.py — same layout, same gap definition, so
test_full.load_picks_for_dir() and test_green_candle_combined.simulate_day_combined()
can consume it without changes.

Usage:
  python download_alpaca_mar_may_2026.py
  python download_alpaca_mar_may_2026.py --resume     # skip already-downloaded tickers
  python download_alpaca_mar_may_2026.py --skip-daily # reuse existing daily_top_gainers.csv
"""

import os
import sys
import re
import time
import argparse
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

DATA_DIR     = "stored_data_mar_may_2026"
GAINERS_CSV  = os.path.join(DATA_DIR, "daily_top_gainers.csv")
INTRADAY_DIR = os.path.join(DATA_DIR, "intraday")
DAILY_DIR    = os.path.join(DATA_DIR, "daily")

MIN_GAP_PCT       = 2.0
MAX_PRICE         = 50.0
TOP_N             = 20
RATE_LIMIT_DELAY  = 0.35   # ~170 req/min (Alpaca free tier: 200/min)

START_DATE = datetime(2026, 3, 15)
END_DATE   = datetime(2026, 5, 17)  # exclusive — include May 16


def _is_warrant_or_unit(ticker):
    if ".WS" in ticker or ".RT" in ticker:
        return True
    if re.match(r"^[A-Z]{3,}W$", ticker):
        return True
    if ticker.endswith("WW"):
        return True
    if re.match(r"^[A-Z]{3,}U$", ticker):
        return True
    if re.match(r"^[A-Z]{3,}R$", ticker):
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip tickers already in intraday/")
    parser.add_argument("--skip-daily", action="store_true", help="Reuse existing daily_top_gainers.csv")
    args = parser.parse_args()

    os.makedirs(INTRADAY_DIR, exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)

    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # ===== Step 1: Daily scan -> top-20 per day =====
    if not args.skip_daily or not os.path.exists(GAINERS_CSV):
        print(f"Step 1: Getting US equity ticker universe...")
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        assets = trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        all_tickers = [
            a.symbol for a in assets
            if a.tradable
            and not _is_warrant_or_unit(a.symbol)
            and "." not in a.symbol
            and len(a.symbol) <= 5
        ]
        print(f"  {len(all_tickers)} tradeable US equities")

        print(f"\nStep 2: Downloading daily bars {START_DATE.date()} -> {END_DATE.date()}")
        all_daily = {}
        batch_size = 500
        for batch_start in range(0, len(all_tickers), batch_size):
            batch = all_tickers[batch_start:batch_start + batch_size]
            batch_end = min(batch_start + batch_size, len(all_tickers))
            print(f"  Batch {batch_start // batch_size + 1}: tickers {batch_start + 1}-{batch_end}...",
                  end="", flush=True)

            try:
                req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=START_DATE - timedelta(days=5),  # need prev close before start
                    end=END_DATE,
                    adjustment="raw",
                )
                bars = data_client.get_stock_bars(req)
                if not bars.df.empty:
                    df = bars.df.reset_index()
                    for ticker in df["symbol"].unique():
                        tdf = df[df["symbol"] == ticker].copy()
                        tdf = tdf.set_index("timestamp").sort_index()
                        all_daily[ticker] = tdf
                print(f" got {len([t for t in batch if t in all_daily])} tickers")
            except Exception as e:
                print(f" ERROR: {str(e)[:80]}")

            time.sleep(RATE_LIMIT_DELAY * 2)

        print(f"  Total tickers with daily data: {len(all_daily)}")

        # Save raw daily/<TICKER>.csv (needed by test_full.py for prev_close lookups)
        print(f"\nStep 2b: Writing daily/<ticker>.csv...")
        for ticker, tdf in all_daily.items():
            out_path = os.path.join(DAILY_DIR, f"{ticker}.csv")
            tdf.to_csv(out_path)

        # Step 3: Compute gaps, pick top 20 per day
        print(f"\nStep 3: Computing gap-ups, top {TOP_N} per day...")
        gainers_rows = []
        trading_days = pd.bdate_range(start=START_DATE, end=END_DATE - timedelta(days=1))

        for day in trading_days:
            day_str = day.strftime("%Y-%m-%d")
            day_candidates = []

            for ticker, tdf in all_daily.items():
                day_mask = tdf.index.date == day.date()
                if not day_mask.any():
                    continue
                day_row = tdf[day_mask].iloc[0]

                prev_mask = tdf.index.date < day.date()
                if not prev_mask.any():
                    continue
                prev_close = tdf[prev_mask].iloc[-1]["close"]

                if prev_close <= 0 or prev_close > MAX_PRICE:
                    continue

                gap_pct = (day_row["open"] / prev_close - 1) * 100
                if gap_pct >= MIN_GAP_PCT:
                    day_candidates.append({
                        "date": day_str,
                        "ticker": ticker,
                        "open": day_row["open"],
                        "high": day_row["high"],
                        "low": day_row["low"],
                        "close": day_row["close"],
                        "volume": day_row["volume"],
                        "prev_close": prev_close,
                        "gap_pct": round(gap_pct, 2),
                    })

            day_candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
            top = day_candidates[:TOP_N]
            gainers_rows.extend(top)

            if top:
                print(f"  {day_str}: {len(day_candidates)} gap-ups, top {len(top)} saved "
                      f"(best: {top[0]['ticker']} +{top[0]['gap_pct']:.0f}%)", flush=True)
            else:
                print(f"  {day_str}: no gap-ups", flush=True)

        gainers_df = pd.DataFrame(gainers_rows)
        gainers_df.to_csv(GAINERS_CSV, index=False)
        print(f"\n  Saved {len(gainers_df)} rows to {GAINERS_CSV}")
        print(f"  Unique tickers: {gainers_df['ticker'].nunique()}")
        if len(gainers_df) > 0:
            print(f"  Date range: {gainers_df['date'].min()} to {gainers_df['date'].max()}")
    else:
        print(f"Step 1-3: Using existing {GAINERS_CSV}")
        gainers_df = pd.read_csv(GAINERS_CSV)

    # ===== Step 4: Download 2-min intraday for unique candidates =====
    tickers = sorted(gainers_df["ticker"].unique())
    print(f"\nStep 4: Downloading 2-min intraday for {len(tickers)} unique tickers...")

    existing = set()
    if args.resume:
        existing = {f.replace(".csv", "") for f in os.listdir(INTRADAY_DIR) if f.endswith(".csv")}
        print(f"  Resume mode: {len(existing)} already downloaded")

    downloaded = 0
    skipped = 0
    errors = 0

    for i, ticker in enumerate(tickers):
        if ticker in existing:
            skipped += 1
            continue

        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=START_DATE,
                end=END_DATE,
                adjustment="raw",
            )
            bars = data_client.get_stock_bars(req)

            if bars.df.empty:
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

            # Resample 1-min -> 2-min, aligned to clock minutes
            bar_2m = bar_df.resample("2min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "trade_count": "sum",
                "vwap": "last",
            }).dropna(subset=["open"])

            out = bar_2m[["open", "high", "low", "close", "volume"]].copy()
            out.columns = ["Open", "High", "Low", "Close", "Volume"]
            out.index.name = "Datetime"

            out_path = os.path.join(INTRADAY_DIR, f"{ticker}.csv")
            out.to_csv(out_path)
            downloaded += 1

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(out)} bars saved", flush=True)

        except Exception as e:
            err_str = str(e)[:100]
            print(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR - {err_str}", flush=True)
            errors += 1

        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")
    print(f"Files in {INTRADAY_DIR}/: {len(os.listdir(INTRADAY_DIR))}")


if __name__ == "__main__":
    main()
