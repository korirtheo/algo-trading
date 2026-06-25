"""Download top-20 gap-up data for the 3 missing date ranges in 2026.

Existing coverage:
  stored_data:               2026-01-05 -> 2026-02-27
  stored_data_mar_may_2026:  2026-03-17 -> 2026-05-15
  stored_data_jun_2026:      2026-05-21 -> 2026-06-16

Gaps to fill (~21 trading days, "month of missing data"):
  2026-03-02 -> 2026-03-16  (Feb 28 was Saturday)
  2026-05-18 -> 2026-05-20
  2026-06-17 -> 2026-06-24  (today)

Output dir: stored_data_2026_gap_fill/  (intraday/, daily/, daily_top_gainers.csv)
load_all_picks merges with first-dir-wins, so safe to add to DATA_DIRS.

Usage:
  python scripts/download/download_alpaca_2026_gaps.py
  python scripts/download/download_alpaca_2026_gaps.py --resume
"""
import os
import sys
import re
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "stored_data_2026_gap_fill"
GAINERS_CSV = os.path.join(DATA_DIR, "daily_top_gainers.csv")
INTRADAY_DIR = os.path.join(DATA_DIR, "intraday")
DAILY_DIR = os.path.join(DATA_DIR, "daily")

MIN_GAP_PCT = 2.0
MAX_PRICE = 50.0
TOP_N = 20
RATE_LIMIT_DELAY = 0.35

GAP_RANGES = [
    (datetime(2026, 3, 2),  datetime(2026, 3, 17)),   # exclusive end -> includes Mar 16
    (datetime(2026, 5, 18), datetime(2026, 5, 21)),   # includes May 18, 19, 20
    # (datetime(2026, 6, 17), datetime(2026, 6, 25)),   # BLOCKED by Alpaca SIP free-tier 15-day delay
]

# Windows reserved filenames — cannot be written to disk on Windows
WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def _is_warrant_or_unit(ticker):
    if ticker in WIN_RESERVED:
        return True
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


def fetch_daily_for_range(data_client, all_tickers, start, end):
    all_daily = {}
    batch_size = 500
    for batch_start in range(0, len(all_tickers), batch_size):
        batch = all_tickers[batch_start:batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(all_tickers))
        print(f"    Batch {batch_start // batch_size + 1}: tickers {batch_start + 1}-{batch_end}...", end="", flush=True)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start - timedelta(days=5),
                end=end,
                adjustment="raw",
            )
            bars = data_client.get_stock_bars(req)
            if not bars.df.empty:
                df = bars.df.reset_index()
                for ticker in df["symbol"].unique():
                    tdf = df[df["symbol"] == ticker].copy().set_index("timestamp").sort_index()
                    if ticker in all_daily:
                        all_daily[ticker] = pd.concat([all_daily[ticker], tdf]).sort_index()
                        all_daily[ticker] = all_daily[ticker][~all_daily[ticker].index.duplicated(keep='last')]
                    else:
                        all_daily[ticker] = tdf
            print(f" got {sum(1 for t in batch if t in all_daily)} tickers")
        except Exception as e:
            print(f" ERROR: {str(e)[:80]}")
        time.sleep(RATE_LIMIT_DELAY * 2)
    return all_daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip already-downloaded tickers")
    parser.add_argument("--skip-daily", action="store_true", help="Reuse existing daily_top_gainers.csv")
    args = parser.parse_args()

    os.makedirs(INTRADAY_DIR, exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)

    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # ===== Step 1-3: Daily scan for each gap range =====
    if not args.skip_daily or not os.path.exists(GAINERS_CSV):
        print("Step 1: Getting US equity ticker universe...")
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        assets = trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        all_tickers = [
            a.symbol for a in assets
            if a.tradable and not _is_warrant_or_unit(a.symbol)
            and "." not in a.symbol and len(a.symbol) <= 5
        ]
        print(f"  {len(all_tickers)} tradeable US equities")

        merged_daily = {}
        for start, end in GAP_RANGES:
            print(f"\nStep 2: Daily bars {start.date()} -> {(end - timedelta(days=1)).date()}")
            range_daily = fetch_daily_for_range(data_client, all_tickers, start, end)
            for t, tdf in range_daily.items():
                if t in merged_daily:
                    merged_daily[t] = pd.concat([merged_daily[t], tdf]).sort_index()
                    merged_daily[t] = merged_daily[t][~merged_daily[t].index.duplicated(keep='last')]
                else:
                    merged_daily[t] = tdf
        print(f"\n  Total tickers with daily data: {len(merged_daily)}")

        print("\nStep 2b: Writing daily/<ticker>.csv...")
        for ticker, tdf in merged_daily.items():
            if ticker in WIN_RESERVED:
                continue
            tdf.to_csv(os.path.join(DAILY_DIR, f"{ticker}.csv"))

        # Step 3: Compute gap-ups for each range, take top-20 per day
        print(f"\nStep 3: Computing gap-ups, top {TOP_N} per day...")
        gainers_rows = []
        all_target_days = []
        for start, end in GAP_RANGES:
            all_target_days.extend(pd.bdate_range(start=start, end=end - timedelta(days=1)))

        for day in all_target_days:
            day_str = day.strftime("%Y-%m-%d")
            day_candidates = []
            for ticker, tdf in merged_daily.items():
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
                        "date": day_str, "ticker": ticker,
                        "open": day_row["open"], "high": day_row["high"],
                        "low": day_row["low"], "close": day_row["close"],
                        "volume": day_row["volume"], "prev_close": prev_close,
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
    else:
        print(f"Step 1-3: Using existing {GAINERS_CSV}")
        gainers_df = pd.read_csv(GAINERS_CSV)

    # ===== Step 4: 2-min intraday for each gap range =====
    print(f"\nStep 4: Downloading 2-min intraday...")
    existing = set()
    if args.resume:
        existing = {f.replace(".csv", "") for f in os.listdir(INTRADAY_DIR) if f.endswith(".csv")}
        print(f"  Resume mode: {len(existing)} already downloaded")

    # Build per-ticker date ranges
    ticker_dates = gainers_df.groupby("ticker")["date"].apply(list).to_dict()
    tickers = sorted(ticker_dates.keys())
    print(f"  {len(tickers)} unique tickers to download")

    downloaded, skipped, errors = 0, 0, 0
    for i, ticker in enumerate(tickers):
        if ticker in existing:
            skipped += 1
            continue
        date_strs = sorted(ticker_dates[ticker])
        ticker_start = datetime.strptime(date_strs[0], "%Y-%m-%d")
        ticker_end = datetime.strptime(date_strs[-1], "%Y-%m-%d") + timedelta(days=1)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=ticker_start,
                end=ticker_end,
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
                bar_df = bar_df[bar_df["symbol"] == ticker].copy().drop(columns=["symbol"])
            bar_df = bar_df.rename(columns={"timestamp": "Datetime"}).set_index("Datetime")
            bar_2m = bar_df.resample("2min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
                "trade_count": "sum", "vwap": "last",
            }).dropna(subset=["open"])
            out = bar_2m[["open", "high", "low", "close", "volume"]].copy()
            out.columns = ["Open", "High", "Low", "Close", "Volume"]
            out.index.name = "Datetime"
            out.to_csv(os.path.join(INTRADAY_DIR, f"{ticker}.csv"))
            downloaded += 1
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(out)} bars saved", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR - {str(e)[:100]}", flush=True)
            errors += 1
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")
    print(f"Files in {INTRADAY_DIR}/: {len(os.listdir(INTRADAY_DIR))}")
    print(f"\nNext: add '{DATA_DIR}' to DATA_DIRS in test scripts to use the new data.")


if __name__ == "__main__":
    main()
