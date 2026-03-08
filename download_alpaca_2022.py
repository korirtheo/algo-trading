"""
Download 2022 gap-up data from Alpaca (daily scan + 2-min intraday).

Usage:
    python download_alpaca_2022.py --key YOUR_KEY --secret YOUR_SECRET
    python download_alpaca_2022.py --resume  # skip already-downloaded tickers
"""

import os
import sys
import re
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "stored_data_2022"
GAINERS_CSV = os.path.join(DATA_DIR, "daily_top_gainers.csv")
INTRADAY_DIR = os.path.join(DATA_DIR, "intraday")
INTRADAY_1MIN_DIR = os.path.join(DATA_DIR, "intraday_1min")
DAILY_DIR = os.path.join(DATA_DIR, "daily")

MIN_GAP_PCT = 2.0
MAX_PRICE = 50.0
TOP_N = 20
RATE_LIMIT_DELAY = 0.35

START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 1, 1)


def _is_warrant_or_unit(ticker):
    if '.WS' in ticker or '.RT' in ticker:
        return True
    if re.match(r'^[A-Z]{3,}W$', ticker):
        return True
    if ticker.endswith('WW'):
        return True
    if re.match(r'^[A-Z]{3,}U$', ticker):
        return True
    if re.match(r'^[A-Z]{3,}R$', ticker):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Download 2022 data from Alpaca")
    parser.add_argument("--key", default=os.environ.get("ALPACA_API_KEY"), help="Alpaca API key")
    parser.add_argument("--secret", default=os.environ.get("ALPACA_API_SECRET"), help="Alpaca API secret")
    parser.add_argument("--resume", action="store_true", help="Skip already-downloaded tickers")
    parser.add_argument("--skip-daily", action="store_true", help="Skip daily scan, use existing CSV")
    args = parser.parse_args()

    if not args.key or not args.secret:
        print("ERROR: Set ALPACA_API_KEY/ALPACA_API_SECRET or use --key/--secret")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INTRADAY_DIR, exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)

    data_client = StockHistoricalDataClient(args.key, args.secret)

    if not args.skip_daily or not os.path.exists(GAINERS_CSV):
        print("Step 1: Getting US equity ticker list...")
        trading_client = TradingClient(args.key, args.secret)
        assets = trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        all_tickers = [a.symbol for a in assets
                       if a.tradable and not _is_warrant_or_unit(a.symbol)
                       and '.' not in a.symbol and len(a.symbol) <= 5]
        print(f"  {len(all_tickers)} tradeable US equities")

        print(f"\nStep 2: Downloading daily bars for 2022...")
        all_daily = {}
        batch_size = 500
        for batch_start in range(0, len(all_tickers), batch_size):
            batch = all_tickers[batch_start:batch_start + batch_size]
            batch_end = min(batch_start + batch_size, len(all_tickers))
            print(f"  Batch {batch_start//batch_size + 1}: tickers {batch_start+1}-{batch_end}...", end="", flush=True)

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=START_DATE - timedelta(days=5),
                    end=END_DATE,
                    adjustment="raw",
                )
                bars = data_client.get_stock_bars(request)
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

        print(f"\nStep 3: Computing gap-ups...")
        gainers_rows = []
        trading_days = pd.bdate_range(start="2022-01-01", end="2022-12-31")

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

        gainers_df = pd.DataFrame(gainers_rows)
        gainers_df.to_csv(GAINERS_CSV, index=False)
        print(f"\n  Saved {len(gainers_df)} rows to {GAINERS_CSV}")
        print(f"  Unique tickers: {gainers_df['ticker'].nunique()}")
        print(f"  Date range: {gainers_df['date'].min()} to {gainers_df['date'].max()}")
    else:
        print(f"Step 1-3: Using existing {GAINERS_CSV}")
        gainers_df = pd.read_csv(GAINERS_CSV)

    gainers_df = gainers_df.dropna(subset=["ticker"])
    tickers = sorted(gainers_df["ticker"].unique())
    ticker_date_map = gainers_df.groupby("ticker")["date"].apply(set).to_dict()
    print(f"\nStep 4: Downloading 1-min intraday (SIP) for {len(tickers)} tickers...")

    os.makedirs(INTRADAY_1MIN_DIR, exist_ok=True)

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
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=START_DATE,
                end=END_DATE,
                adjustment="raw",
                feed="sip",
            )
            bars = data_client.get_stock_bars(request)

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
            bar_df = bar_df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            bar_df = bar_df[["Open", "High", "Low", "Close", "Volume"]]

            # Filter to only needed dates (per-ticker)
            needed_dates = ticker_date_map.get(ticker, set())
            if needed_dates:
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

            # Resample to 2-min and save
            bar_2m = bar_df.resample("2min").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna(subset=["Open"])

            bar_2m.to_csv(os.path.join(INTRADAY_DIR, f"{ticker}.csv"))
            downloaded += 1

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(bar_df)} 1min / {len(bar_2m)} 2min bars", flush=True)

        except Exception as e:
            err_str = str(e)[:100]
            print(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR - {err_str}", flush=True)
            errors += 1

        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone! Downloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")
    print(f"Files in {INTRADAY_DIR}/: {len(os.listdir(INTRADAY_DIR))}")


if __name__ == "__main__":
    main()
