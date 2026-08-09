"""Parallel 1-min Alpaca SIP downloader for the replay universe.

Resumable: skips tickers whose intraday file already exists. Use 8 worker
threads (well under Alpaca free-tier 200 req/min), each ticker = 2 requests
(1-min + daily). Retries transient connection errors.

Usage:
  python download_1min_sip_parallel.py                                 # 2024-2026
  python download_1min_sip_parallel.py --index gainers_index_2022_2023.json --out-dir stored_data_1min_2022_2023
"""
import argparse
import csv
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

parser = argparse.ArgumentParser()
parser.add_argument("--index", default="gainers_index_2024_2026.json")
parser.add_argument("--out-dir", default="stored_data_1min")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

ET = ZoneInfo("America/New_York")
OUT_DIR = args.out_dir
INTRADAY_DIR = os.path.join(OUT_DIR, "intraday")
DAILY_DIR = os.path.join(OUT_DIR, "daily")
GAINERS_CSV = os.path.join(OUT_DIR, "daily_top_gainers.csv")
N_WORKERS = args.workers
MIN_GAP_BETWEEN_REQS = 0.15  # seconds per request -> ~8 req/s across 8 workers

WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}

with open(args.index) as f:
    INDEX = json.load(f)
ranges = INDEX["ticker_ranges"]

os.makedirs(INTRADAY_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)

_lock = threading.Lock()
done = 0
errs = 0
err_list = []


def _is_warrant_or_unit(t):
    if t in WIN_RESERVED or ".WS" in t or ".RT" in t:
        return True
    if re.match(r"^[A-Z]{3,}W$", t) or t.endswith("WW") or re.match(r"^[A-Z]{3,}U$", t) or re.match(r"^[A-Z]{3,}R$", t):
        return True
    return False


def _new_client():
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)


def _fetch(client, ticker, timeframe, start, end):
    req = StockBarsRequest(
        symbol_or_symbols=ticker, timeframe=timeframe,
        start=start, end=end, adjustment="raw", feed="sip",
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


def _fetch_with_retry(client, ticker, timeframe, start, end, tries=3):
    for a in range(tries):
        try:
            return _fetch(client, ticker, timeframe, start, end)
        except Exception as e:
            if a < tries - 1:
                time.sleep(1.5 * (a + 1))
            else:
                raise


def _already_have(ticker):
    p = os.path.join(INTRADAY_DIR, f"{ticker}.csv")
    if not os.path.exists(p):
        return False
    try:
        return os.path.getsize(p) > 60
    except OSError:
        return False


def _work(ticker):
    global done, errs
    if ticker in WIN_RESERVED or _is_warrant_or_unit(ticker):
        return
    if _already_have(ticker):
        with _lock:
            done += 1
        return

    d0, d1 = ranges[ticker]
    start = datetime.strptime(d0, "%Y-%m-%d").replace(hour=4, tzinfo=ET)
    end = datetime.strptime(d1, "%Y-%m-%d").replace(hour=0, tzinfo=ET) + timedelta(days=1)

    client = _new_client()
    try:
        mdf = _fetch_with_retry(client, ticker, TimeFrame.Minute, start, end)
        time.sleep(MIN_GAP_BETWEEN_REQS)
        ddf = _fetch_with_retry(client, ticker, TimeFrame.Day, start, end)
        if mdf is not None and len(mdf) > 0:
            mdf.to_csv(os.path.join(INTRADAY_DIR, f"{ticker}.csv"))
        if ddf is not None and len(ddf) > 0:
            ddf.to_csv(os.path.join(DAILY_DIR, f"{ticker}.csv"))
        with _lock:
            done += 1
    except Exception as e:
        with _lock:
            errs += 1
            err_list.append((ticker, str(e)[:100]))
        print(f"ERR {ticker}: {str(e)[:100]}", flush=True)


def _worker(tickers, idx):
    client = _new_client()
    for ticker in tickers:
        _work(ticker)


def main():
    import threading as th

    tickers = sorted(ranges.keys())
    # skip already-done up front for speed
    todo = [t for t in tickers if not _already_have(t) and not _is_warrant_or_unit(t)]
    print(f"Total {len(tickers)} tickers, {len(todo)} to download (rest already present)", flush=True)

    chunks = [todo[i::N_WORKERS] for i in range(N_WORKERS)]
    threads = [th.Thread(target=_worker, args=(chunks[i], i), daemon=True) for i in range(N_WORKERS)]
    t0 = time.time()
    for t in threads:
        t.start()

    # progress monitor
    last = 0
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        files = sum(1 for _ in os.listdir(INTRADAY_DIR) if _.endswith(".csv"))
        el = time.time() - t0
        rate = files / el
        remain = len(todo) - files
        print(f"[{files}/{len(todo)}] err={errs} rate={rate:.2f}/s ETA={remain/rate/60:.1f}min", flush=True)
        if files == last and el > 120:
            print("  WARNING: no new files in 2min — checking threads...", flush=True)
            time.sleep(60)
        last = files
        if files >= len(todo):
            break

    for t in threads:
        t.join(timeout=5)

    # Write merged gainers csv
    rows = []
    for d, tk in INDEX["index"].items():
        for t, gap in tk.items():
            rows.append({"date": d, "ticker": t, "gap_pct": gap})
    with open(GAINERS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "ticker", "gap_pct"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nDONE: downloaded={done} errors={errs} in {time.time()-t0:.0f}s")
    for t, e in err_list[:30]:
        print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
