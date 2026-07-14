"""
Download Jun 17 - Jul 10 2026 top-20-gap-up data from Alpaca.

Layout matches download_alpaca_mar_may_2026.py so test_full.load_picks_for_dir()
can consume it.

Usage:
  python download_alpaca_jun_jul_2026.py
  python download_alpaca_jun_jul_2026.py --resume
"""
import os, sys, re, time, argparse
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

START_DATE = datetime(2026, 6, 17)
END_DATE   = datetime(2026, 7, 11)  # exclusive

RESERVED_WIN = {"con","prn","aux","nul",
                "com1","com2","com3","com4","com5",
                "com6","com7","com8","com9",
                "lpt1","lpt2","lpt3","lpt4","lpt5",
                "lpt6","lpt7","lpt8","lpt9"}


def _is_warrant_or_unit(ticker):
    if ticker.lower() in RESERVED_WIN: return True
    if ".WS" in ticker or ".RT" in ticker: return True
    if re.match(r"^[A-Z]{3,}W$", ticker): return True
    if ticker.endswith("WW"): return True
    if re.match(r"^[A-Z]{3,}U$", ticker): return True
    if re.match(r"^[A-Z]{3,}R$", ticker): return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-daily", action="store_true")
    args = parser.parse_args()

    os.makedirs(INTRADAY_DIR, exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # ── Step 1-2: Daily bars ──────────────────────────────────────────
    if args.skip_daily and os.path.exists(GAINERS_CSV):
        print(f"Using existing {GAINERS_CSV}")
        gainers_df = pd.read_csv(GAINERS_CSV)
    else:
        if args.resume:
            existing_daily = {f.replace(".csv","") for f in os.listdir(DAILY_DIR) if f.endswith(".csv")}
            if existing_daily:
                print(f"Resume: reloading {len(existing_daily)} daily files from {DAILY_DIR}/")
                all_daily = {}
                for ticker in sorted(existing_daily):
                    try:
                        tdf = pd.read_csv(os.path.join(DAILY_DIR, f"{ticker}.csv"),
                                          index_col=0, parse_dates=True)
                        all_daily[ticker] = tdf
                    except Exception:
                        pass
                print(f"  Reloaded {len(all_daily)} tickers")
                # skip to step 3
                gainers_df = _compute_gainers(all_daily)
                gainers_df.to_csv(GAINERS_CSV, index=False)
                print(f"  Saved {len(gainers_df)} gainers rows to {GAINERS_CSV}")
                _download_intraday(data_client, gainers_df, args)
                return

        print(f"Step 1: Getting US equity ticker universe...")
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        assets = trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        all_tickers = sorted(set(
            a.symbol for a in assets if a.tradable
            and not _is_warrant_or_unit(a.symbol)
            and "." not in a.symbol and len(a.symbol) <= 5
        ))
        print(f"  {len(all_tickers)} tradeable US equities")

        print(f"\nStep 2: Downloading daily bars {START_DATE.date()} -> {END_DATE.date()}")
        all_daily = {}
        batch_size = 500
        for batch_start in range(0, len(all_tickers), batch_size):
            batch = all_tickers[batch_start:batch_start + batch_size]
            batch_end = min(batch_start + batch_size, len(all_tickers))
            for attempt in range(3):
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame.Day,
                        start=START_DATE - timedelta(days=5),
                        end=END_DATE,
                        adjustment="raw",
                    )
                    bars = data_client.get_stock_bars(req)
                    if not bars.df.empty:
                        df = bars.df.reset_index()
                        for ticker in df["symbol"].unique():
                            tdf = df[df["symbol"] == ticker].copy().set_index("timestamp").sort_index()
                            all_daily[ticker] = tdf
                    n_ok = sum(1 for t in batch if t in all_daily)
                    print(f"  Batch {batch_start//batch_size+1}: {batch_start+1}-{batch_end} -> {n_ok}")
                    break
                except Exception as e:
                    print(f"  Batch {batch_start//batch_size+1} attempt {attempt+1}: {str(e)[:60]}")
                    time.sleep(RATE_LIMIT_DELAY * 5 * (attempt + 1))
        print(f"  Total: {len(all_daily)} tickers")

        print(f"\nWriting daily/<ticker>.csv...")
        written = 0
        for ticker, tdf in all_daily.items():
            # Standardise to Title Case columns + only Open/High/Low/Close/Volume
            # so test_full._process_one_day can read ddf["Close"] correctly.
            for rename_col, title_col in [("open","Open"),("high","High"),("low","Low"),("close","Close"),("volume","Volume")]:
                if rename_col in tdf.columns and title_col not in tdf.columns:
                    tdf[title_col] = tdf[rename_col]
            keep = [c for c in ["Open","High","Low","Close","Volume"] if c in tdf.columns]
            out = tdf[keep].copy()
            try:
                out.to_csv(os.path.join(DAILY_DIR, f"{ticker}.csv"))
                written += 1
            except OSError as e:
                print(f"  SKIP {ticker}: {e}")
        print(f"  Written: {written}/{len(all_daily)}")

        gainers_df = _compute_gainers(all_daily)
        gainers_df.to_csv(GAINERS_CSV, index=False)
        print(f"\n  Saved {len(gainers_df)} gainers rows to {GAINERS_CSV}")
        print(f"  Date range: {gainers_df['date'].min()} to {gainers_df['date'].max()}")

    _download_intraday(data_client, gainers_df, args)


def _compute_gainers(all_daily):
    print(f"Computing top-{TOP_N} gap-ups per day...")
    gainers_rows = []
    # Use US market calendar - exclude Jul 3 (Independence Day observed)
    trading_days = pd.bdate_range(start=START_DATE, end=END_DATE - timedelta(days=1))
    trading_days = [d for d in trading_days if not (d.month == 7 and d.day == 3)]

    for day in trading_days:
        day_str = day.strftime("%Y-%m-%d")
        candidates = []
        for ticker, tdf in all_daily.items():
            day_mask = tdf.index.date == day.date()
            if not day_mask.any(): continue
            day_row = tdf[day_mask].iloc[0]
            prev_mask = tdf.index.date < day.date()
            if not prev_mask.any(): continue
            prev_close = tdf[prev_mask].iloc[-1]["close"]
            if prev_close <= 0 or prev_close > MAX_PRICE: continue
            gap_pct = (day_row["open"] / prev_close - 1) * 100
            if gap_pct >= MIN_GAP_PCT:
                candidates.append({
                    "date": day_str, "ticker": ticker,
                    "open": day_row["open"], "high": day_row["high"],
                    "low": day_row["low"], "close": day_row["close"],
                    "volume": day_row["volume"],
                    "prev_close": prev_close, "gap_pct": round(gap_pct, 2),
                })
        candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
        top = candidates[:TOP_N]
        gainers_rows.extend(top)
        if top:
            print(f"  {day_str}: {len(candidates)} gap-ups, best: {top[0]['ticker']} +{top[0]['gap_pct']:.0f}%")
        else:
            print(f"  {day_str}: no gap-ups")

    return pd.DataFrame(gainers_rows)


def _download_intraday(data_client, gainers_df, args):
    tickers = sorted(gainers_df["ticker"].unique())
    print(f"\nDownloading 2-min intraday for {len(tickers)} unique tickers...")

    existing = set()
    if args.resume:
        existing = {f.replace(".csv","") for f in os.listdir(INTRADAY_DIR) if f.endswith(".csv")}
        print(f"  Resume: {len(existing)} already downloaded")

    dl = sk = errs = 0
    for i, ticker in enumerate(tickers):
        if ticker in existing:
            sk += 1; continue
        for attempt in range(3):
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
                    print(f"  [{i+1}/{len(tickers)}] {ticker}: no data")
                    errs += 1
                    time.sleep(RATE_LIMIT_DELAY)
                    break
                bar_df = bars.df.reset_index()
                if "symbol" in bar_df.columns:
                    bar_df = bar_df[bar_df["symbol"] == ticker].copy().drop(columns=["symbol"])
                bar_df = bar_df.rename(columns={"timestamp": "Datetime"}).set_index("Datetime")
                bar_2m = bar_df.resample("2min").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                    "trade_count": "sum", "vwap": "last",
                }).dropna(subset=["open"])
                out = bar_2m[["open","high","low","close","volume"]].copy()
                out.columns = ["Open","High","Low","Close","Volume"]
                out.index.name = "Datetime"
                out.to_csv(os.path.join(INTRADAY_DIR, f"{ticker}.csv"))
                dl += 1
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(out)} bars")
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(RATE_LIMIT_DELAY * 3 * (attempt + 1))
                    continue
                print(f"  [{i+1}/{len(tickers)}] {ticker}: ERROR {str(e)[:80]}")
                errs += 1
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone. Downloaded: {dl}, Skipped: {sk}, Errors: {errs}")
    print(f"Files in {INTRADAY_DIR}/: {len(os.listdir(INTRADAY_DIR))}")


if __name__ == "__main__":
    main()
