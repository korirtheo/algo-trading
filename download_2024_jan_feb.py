"""
Download Jan-Feb 2024 stock data to fill the dataset gap.
============================================================
The existing stored_data_jan_mar_2024/ only has data starting March 5, 2024.
This script downloads January 2 - February 29, 2024 into stored_data_jan_feb_2024/.

Strategy:
  Step 1: Use yfinance to get daily bars for ALL small-cap tickers (free, unlimited history)
  Step 2: Build grouped_daily.json from the daily data
  Step 3: Find gap-up candidates (>2% gap from previous close, <$50)
  Step 4: Save daily_top_gainers.csv
  Step 5: Download 2-min intraday bars for gap-up candidates using Polygon or Alpaca
  Step 6: Save individual daily CSVs for gap-up tickers

IMPORTANT: Polygon.io free plan only covers 2 years of history. As of March 2026,
Jan-Feb 2024 is outside the free window. You need EITHER:
  a) Paid Polygon key ($29/mo Starter, 5-year history)
  b) Free Alpaca paper trading key (unlimited historical bars)
     Sign up at: https://app.alpaca.markets/signup (no credit card needed)

Usage:
  python download_2024_jan_feb.py                                      # daily only
  python download_2024_jan_feb.py --polygon POLYGON_KEY                # + Polygon intraday
  python download_2024_jan_feb.py --alpaca KEY_ID SECRET_KEY           # + Alpaca intraday
  python download_2024_jan_feb.py --alpaca KEY_ID SECRET_KEY --skip-daily  # resume intraday

Estimated time:
  Step 1 (yfinance daily):     ~15-20 minutes (2700 tickers in batches)
  Step 5 (Polygon intraday):   ~6-10 hours (400-800 ticker-days, 12.5s/call)
  Step 5 (Alpaca intraday):    ~30-60 minutes (parallel, no strict rate limit)
"""
import os
import sys
import re
import time
import json
import datetime
import pandas as pd
import numpy as np
import requests
from zoneinfo import ZoneInfo

# --- CONFIG ---
OUT_DIR = "stored_data_jan_feb_2024"
TARGET_START = "2024-01-02"  # Jan 1 is a holiday
TARGET_END = "2024-02-29"
DAILY_LOOKBACK_START = "2023-12-01"  # need prev close before Jan 2024
MIN_GAP_PCT = 2.0
TOP_N = 20
MAX_PRICE = 50.0
POLYGON_RATE_LIMIT = 5
POLYGON_CALL_INTERVAL = 60.0 / POLYGON_RATE_LIMIT + 0.5

ET_TZ = ZoneInfo("America/New_York")
POLYGON_BASE_URL = "https://api.polygon.io"
ALPACA_BASE_URL = "https://data.alpaca.markets"

# Ticker list path
TICKER_LIST = os.path.join("stored_data", "small_cap_tickers.txt")


def _is_warrant_or_unit(ticker):
    """Filter out warrants, rights, units."""
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


def load_ticker_list():
    """Load small-cap ticker list."""
    if os.path.exists(TICKER_LIST):
        with open(TICKER_LIST) as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(tickers)} tickers from {TICKER_LIST}")
        return tickers
    else:
        print(f"  WARNING: {TICKER_LIST} not found")
        return None


# ─── STEP 1: YFINANCE DAILY DATA ────────────────────────────────────────────

def download_daily_yfinance(tickers):
    """Download daily OHLCV for all tickers using yfinance."""
    import yfinance as yf

    print(f"\nStep 1: Downloading daily data via yfinance...")
    print(f"  Period: {DAILY_LOOKBACK_START} to {TARGET_END}")
    print(f"  Tickers: {len(tickers)}")

    os.makedirs(os.path.join(OUT_DIR, "daily"), exist_ok=True)

    batch_size = 100
    all_daily = {}
    failed = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        sys.stdout.write(
            f"\r  Batch [{batch_num}/{total_batches}] "
            f"({i + len(batch)}/{len(tickers)} tickers)..."
        )
        sys.stdout.flush()

        try:
            data = yf.download(
                batch,
                start=DAILY_LOOKBACK_START,
                end=(pd.Timestamp(TARGET_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )

            if data is None or data.empty:
                failed.extend(batch)
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = data.copy()
                    else:
                        df = data[ticker].copy()

                    df = df.dropna(subset=["Open", "Close"])
                    if len(df) == 0:
                        continue

                    df = df[["Open", "High", "Low", "Close", "Volume"]]

                    if df.index.tz is None:
                        df.index = pd.to_datetime(df.index)
                        df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")

                    df.index.name = "Date"
                    all_daily[ticker] = df

                    csv_path = os.path.join(OUT_DIR, "daily", f"{ticker}.csv")
                    df.to_csv(csv_path)

                except Exception:
                    continue

        except Exception as e:
            print(f"\n    Batch error: {e}")
            failed.extend(batch)
            continue

    print(f"\r  Downloaded daily data for {len(all_daily)} tickers. "
          f"Failed: {len(failed)}.          ")
    return all_daily


def build_grouped_daily(all_daily):
    """Build grouped_daily.json from individual daily DataFrames."""
    print(f"\nStep 2: Building grouped_daily.json...")

    # Include lookback dates for gap calculation (need prev close from Dec 2023)
    # This ensures grouped_daily.json has December data for proper gap calc on Jan 2
    all_dates = pd.bdate_range(start=DAILY_LOOKBACK_START, end=TARGET_END)

    grouped = {}
    for date in all_dates:
        date_str = date.strftime("%Y-%m-%d")
        stocks = []
        for ticker, df in all_daily.items():
            if df.index.tz is not None:
                df_dates = df.index.tz_convert(ET_TZ).strftime("%Y-%m-%d")
            else:
                df_dates = df.index.strftime("%Y-%m-%d")

            mask = df_dates == date_str
            if not mask.any():
                continue

            row = df[mask].iloc[0]
            stocks.append({
                "ticker": ticker,
                "open": float(row["Open"]),
                "close": float(row["Close"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "volume": int(row["Volume"]),
            })

        if stocks:
            grouped[date_str] = stocks

    target_days = len([d for d in grouped if d >= TARGET_START])
    print(f"  Built grouped_daily.json: {len(grouped)} total days "
          f"({target_days} in target range)")
    print(f"  Avg stocks per day: {np.mean([len(v) for v in grouped.values()]):.0f}")
    return grouped


# ─── GAP CANDIDATES ─────────────────────────────────────────────────────────

def find_gap_candidates(daily_data):
    """Find gap-up candidates from daily data."""
    print(f"\nStep 3: Finding gap-up candidates (>{MIN_GAP_PCT}% gap, <${MAX_PRICE})...")

    sorted_dates = sorted(daily_data.keys())
    prev_closes = {}
    gap_candidates = {}

    for date_str in sorted_dates:
        stocks = daily_data[date_str]
        day_candidates = []

        for s in stocks:
            ticker = s["ticker"]
            if _is_warrant_or_unit(ticker):
                continue
            if s["open"] is None or s["open"] <= 0:
                continue
            if s["open"] > MAX_PRICE:
                continue

            if ticker in prev_closes:
                pc = prev_closes[ticker]
                if pc > 0:
                    gap_pct = (s["open"] - pc) / pc * 100
                    if gap_pct >= MIN_GAP_PCT:
                        day_candidates.append({
                            "ticker": ticker,
                            "gap_pct": gap_pct,
                            "open": s["open"],
                            "prev_close": pc,
                        })

            if s["close"] is not None and s["close"] > 0:
                prev_closes[ticker] = s["close"]

        day_candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
        gap_candidates[date_str] = day_candidates[:TOP_N]

    # Only include dates in target range
    target_candidates = {
        d: v for d, v in gap_candidates.items()
        if TARGET_START <= d <= TARGET_END
    }

    total = sum(len(v) for v in target_candidates.values())
    days = sum(1 for v in target_candidates.values() if len(v) > 0)
    print(f"  Found {total} gap-up candidates across {days} days")
    return target_candidates


def save_gainers_csv(gap_candidates):
    """Save daily_top_gainers.csv."""
    rows = []
    for date_str, candidates in sorted(gap_candidates.items()):
        for c in candidates:
            rows.append({
                "date": date_str,
                "ticker": c["ticker"],
                "gap_pct": c["gap_pct"],
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "daily_top_gainers.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved {len(df)} entries to {csv_path}")


# ─── POLYGON INTRADAY ───────────────────────────────────────────────────────

def call_polygon_api(url, api_key, params=None):
    """Make Polygon API call with rate limiting and retries."""
    if params is None:
        params = {}
    params["apiKey"] = api_key

    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"\n    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            elif resp.status_code == 403:
                return None
            else:
                print(f"\n    HTTP {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"\n    Request error: {e}")
            time.sleep(5)
    return None


def download_polygon_intraday(api_key, gap_candidates):
    """Download 2-min intraday bars from Polygon."""
    total = sum(len(v) for v in gap_candidates.values())
    os.makedirs(os.path.join(OUT_DIR, "intraday"), exist_ok=True)

    print(f"\nStep 5: Downloading 2-min intraday bars from Polygon...")
    print(f"  Testing API access for Jan 2024...")

    # Test access
    test_url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/AAPL/range/2/minute/2024-01-15/2024-01-15"
    test_result = call_polygon_api(test_url, api_key)
    if test_result is None:
        print(f"  ERROR: Polygon API key cannot access Jan 2024 intraday data.")
        print(f"  The free plan only covers 2 years of history.")
        print(f"  Options:")
        print(f"    1. Paid Polygon plan: https://polygon.io/pricing ($29/mo)")
        print(f"    2. Free Alpaca account: https://app.alpaca.markets/signup")
        print(f"       Then re-run: python download_2024_jan_feb.py --alpaca KEY SECRET --skip-daily")
        return 0

    print(f"  API access confirmed!")

    # Pre-scan existing files
    idir = os.path.join(OUT_DIR, "intraday")
    existing_dates_by_ticker = {}
    for date_str in sorted(gap_candidates.keys()):
        for cand in gap_candidates[date_str]:
            ticker = cand["ticker"]
            if ticker in existing_dates_by_ticker:
                continue
            csv_path = os.path.join(idir, f"{ticker}.csv")
            dates_set = set()
            if os.path.exists(csv_path):
                try:
                    idf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    if len(idf) > 0:
                        if idf.index.tz is not None:
                            et = idf.index.tz_convert(ET_TZ)
                        else:
                            et = idf.index.tz_localize("UTC").tz_convert(ET_TZ)
                        dates_set = set(et.strftime("%Y-%m-%d"))
                except Exception:
                    pass
            existing_dates_by_ticker[ticker] = dates_set

    to_download = []
    skipped = 0
    for date_str in sorted(gap_candidates.keys()):
        for cand in gap_candidates[date_str]:
            ticker = cand["ticker"]
            if date_str in existing_dates_by_ticker.get(ticker, set()):
                skipped += 1
            else:
                to_download.append((date_str, cand))

    print(f"  Total: {total}, Existing: {skipped}, To download: {len(to_download)}")
    est_time = len(to_download) * POLYGON_CALL_INTERVAL / 60
    print(f"  Estimated time: {est_time:.0f} minutes\n")

    completed = 0
    failed = 0

    for date_str, cand in to_download:
        ticker = cand["ticker"]
        completed += 1
        sys.stdout.write(
            f"\r  [{completed}/{len(to_download)}] {date_str} {ticker} "
            f"(gap {cand['gap_pct']:.1f}%)...     "
        )
        sys.stdout.flush()

        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/2/minute/{date_str}/{date_str}"
        data = call_polygon_api(url, api_key, {
            "adjusted": "true", "sort": "asc", "limit": "50000",
        })

        if data is None or data.get("resultsCount", 0) == 0:
            failed += 1
            time.sleep(POLYGON_CALL_INTERVAL)
            continue

        rows = []
        for bar in data["results"]:
            ts = pd.Timestamp(bar["t"], unit="ms", tz="UTC")
            rows.append({
                "Datetime": ts,
                "Open": bar["o"],
                "High": bar["h"],
                "Low": bar["l"],
                "Close": bar["c"],
                "Volume": bar.get("v", 0),
            })

        df = pd.DataFrame(rows)
        df.set_index("Datetime", inplace=True)

        csv_path = os.path.join(OUT_DIR, "intraday", f"{ticker}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
        df.to_csv(csv_path)

        time.sleep(POLYGON_CALL_INTERVAL)

    print(f"\r  Downloaded {completed - failed}/{len(to_download)} ticker-days. "
          f"Skipped {skipped}. Failed: {failed}.          ")
    return completed - failed


# ─── ALPACA INTRADAY ─────────────────────────────────────────────────────────

def download_alpaca_intraday(alpaca_key_id, alpaca_secret, gap_candidates):
    """Download 2-min intraday bars from Alpaca (free paper trading account)."""
    total = sum(len(v) for v in gap_candidates.values())
    os.makedirs(os.path.join(OUT_DIR, "intraday"), exist_ok=True)

    headers = {
        "APCA-API-KEY-ID": alpaca_key_id,
        "APCA-API-SECRET-KEY": alpaca_secret,
    }

    print(f"\nStep 5: Downloading 2-min intraday bars from Alpaca...")
    print(f"  Testing API access...")

    # Test access
    test_url = f"{ALPACA_BASE_URL}/v2/stocks/AAPL/bars"
    test_params = {
        "start": "2024-01-15T09:30:00Z",
        "end": "2024-01-15T16:00:00Z",
        "timeframe": "2Min",
        "limit": "5",
    }
    try:
        r = requests.get(test_url, headers=headers, params=test_params, timeout=15)
        if r.status_code != 200:
            print(f"  ERROR: Alpaca API returned {r.status_code}: {r.text[:200]}")
            print(f"  Check your API key and secret.")
            return 0
        test_data = r.json()
        if not test_data.get("bars"):
            print(f"  WARNING: No test data returned. Key may lack data permissions.")
        else:
            print(f"  API access confirmed! ({len(test_data['bars'])} test bars)")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0

    # Pre-scan existing files
    idir = os.path.join(OUT_DIR, "intraday")
    existing_dates_by_ticker = {}
    for date_str in sorted(gap_candidates.keys()):
        for cand in gap_candidates[date_str]:
            ticker = cand["ticker"]
            if ticker in existing_dates_by_ticker:
                continue
            csv_path = os.path.join(idir, f"{ticker}.csv")
            dates_set = set()
            if os.path.exists(csv_path):
                try:
                    idf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    if len(idf) > 0:
                        if idf.index.tz is not None:
                            et = idf.index.tz_convert(ET_TZ)
                        else:
                            et = idf.index.tz_localize("UTC").tz_convert(ET_TZ)
                        dates_set = set(et.strftime("%Y-%m-%d"))
                except Exception:
                    pass
            existing_dates_by_ticker[ticker] = dates_set

    to_download = []
    skipped = 0
    for date_str in sorted(gap_candidates.keys()):
        for cand in gap_candidates[date_str]:
            ticker = cand["ticker"]
            if date_str in existing_dates_by_ticker.get(ticker, set()):
                skipped += 1
            else:
                to_download.append((date_str, cand))

    print(f"  Total: {total}, Existing: {skipped}, To download: {len(to_download)}")

    completed = 0
    failed = 0
    rate_delay = 0.25  # Alpaca free = 200 req/min, so ~0.3s between calls is safe

    for date_str, cand in to_download:
        ticker = cand["ticker"]
        completed += 1
        sys.stdout.write(
            f"\r  [{completed}/{len(to_download)}] {date_str} {ticker} "
            f"(gap {cand['gap_pct']:.1f}%)...     "
        )
        sys.stdout.flush()

        # Alpaca: get bars for the full extended day (4:00 AM - 8:00 PM ET)
        url = f"{ALPACA_BASE_URL}/v2/stocks/{ticker}/bars"
        params = {
            "start": f"{date_str}T08:00:00Z",  # 4:00 AM ET = 08:00 UTC (winter)
            "end": f"{date_str}T21:00:00Z",     # 4:00 PM ET = 21:00 UTC
            "timeframe": "2Min",
            "limit": "10000",
            "adjustment": "all",
            "feed": "sip",  # Use SIP (all exchanges) for best coverage
        }

        try:
            all_bars = []
            page_token = None

            while True:
                if page_token:
                    params["page_token"] = page_token
                elif "page_token" in params:
                    del params["page_token"]

                r = requests.get(url, headers=headers, params=params, timeout=30)

                if r.status_code == 429:
                    time.sleep(5)
                    continue
                elif r.status_code == 422:
                    # Ticker might not exist in Alpaca
                    break
                elif r.status_code != 200:
                    break

                data = r.json()
                bars = data.get("bars", [])
                if not bars:
                    break

                all_bars.extend(bars)
                page_token = data.get("next_page_token")
                if not page_token:
                    break
                time.sleep(rate_delay)

            if not all_bars:
                failed += 1
                time.sleep(rate_delay)
                continue

            rows = []
            for bar in all_bars:
                ts = pd.Timestamp(bar["t"])
                if ts.tz is None:
                    ts = ts.tz_localize("UTC")
                rows.append({
                    "Datetime": ts,
                    "Open": bar["o"],
                    "High": bar["h"],
                    "Low": bar["l"],
                    "Close": bar["c"],
                    "Volume": bar.get("v", 0),
                })

            df = pd.DataFrame(rows)
            df.set_index("Datetime", inplace=True)

            # Save in same format as Polygon data
            csv_path = os.path.join(OUT_DIR, "intraday", f"{ticker}.csv")
            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)
            df.to_csv(csv_path)

        except Exception as e:
            print(f"\n    Error for {ticker}: {e}")
            failed += 1

        time.sleep(rate_delay)

    print(f"\r  Downloaded {completed - failed}/{len(to_download)} ticker-days. "
          f"Skipped {skipped}. Failed: {failed}.          ")
    return completed - failed


# ─── MAIN ────────────────────────────────────────────────────────────────────
def parse_args():
    """Parse command-line arguments."""
    args = {
        "polygon_key": None,
        "alpaca_key_id": None,
        "alpaca_secret": None,
        "skip_daily": "--skip-daily" in sys.argv,
    }

    argv = [a for a in sys.argv[1:] if a != "--skip-daily"]
    i = 0
    while i < len(argv):
        if argv[i] == "--polygon" and i + 1 < len(argv):
            args["polygon_key"] = argv[i + 1]
            i += 2
        elif argv[i] == "--alpaca" and i + 2 < len(argv):
            args["alpaca_key_id"] = argv[i + 1]
            args["alpaca_secret"] = argv[i + 2]
            i += 3
        else:
            # Legacy: positional argument = polygon key
            args["polygon_key"] = argv[i]
            i += 1

    return args


if __name__ == "__main__":
    args = parse_args()

    has_intraday_source = args["polygon_key"] or args["alpaca_key_id"]

    print(f"{'='*70}")
    print(f"  JAN-FEB 2024 DATA DOWNLOADER")
    print(f"  Period: {TARGET_START} to {TARGET_END}")
    print(f"  Output: {OUT_DIR}/")
    if args["polygon_key"]:
        k = args["polygon_key"]
        print(f"  Intraday source: Polygon ({k[:8]}...{k[-4:]})")
    elif args["alpaca_key_id"]:
        k = args["alpaca_key_id"]
        print(f"  Intraday source: Alpaca ({k[:8]}...)")
    else:
        print(f"  Intraday source: NONE (daily data only)")
    if args["skip_daily"]:
        print(f"  Skipping daily download (using existing data)")
    print(f"{'='*70}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "daily"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "intraday"), exist_ok=True)

    # --- STEP 1-2: Get daily data ---
    daily_data = None

    if not args["skip_daily"]:
        # Try Polygon grouped daily first (if key provided and has access)
        if args["polygon_key"]:
            print(f"\n  Testing Polygon grouped daily access...")
            test_url = f"{POLYGON_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/2024-01-15"
            test = call_polygon_api(test_url, args["polygon_key"])
            if test and test.get("resultsCount", 0) > 0:
                print(f"  Polygon has access! Using Polygon for daily data.")
                # Download using Polygon grouped daily (more complete than yfinance)
                trading_days = pd.bdate_range(start=DAILY_LOOKBACK_START, end=TARGET_END)
                daily_data = {}
                for idx, date in enumerate(trading_days):
                    ds = date.strftime("%Y-%m-%d")
                    sys.stdout.write(f"\r  [{idx+1}/{len(trading_days)}] {ds}...")
                    sys.stdout.flush()
                    url = f"{POLYGON_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{ds}"
                    d = call_polygon_api(url, args["polygon_key"], {"adjusted": "true"})
                    if d and d.get("resultsCount", 0) > 0:
                        daily_data[ds] = [
                            {"ticker": b["T"], "open": b["o"], "close": b["c"],
                             "high": b["h"], "low": b["l"], "volume": b.get("v", 0)}
                            for b in d.get("results", [])
                        ]
                    time.sleep(POLYGON_CALL_INTERVAL)
                print(f"\r  Downloaded {len(daily_data)} days from Polygon.          ")
            else:
                print(f"  Polygon cannot access Jan 2024 grouped daily. Using yfinance.")

        # Fall back to yfinance
        if daily_data is None:
            tickers = load_ticker_list()
            if tickers is None:
                print("ERROR: No ticker list available. Cannot proceed.")
                sys.exit(1)

            tickers = [t for t in tickers if not _is_warrant_or_unit(t)]
            print(f"  After filtering: {len(tickers)} tickers")

            all_daily = download_daily_yfinance(tickers)
            if not all_daily:
                print("\nERROR: No daily data downloaded.")
                sys.exit(1)

            daily_data = build_grouped_daily(all_daily)
    else:
        # Load existing grouped_daily.json
        json_path = os.path.join(OUT_DIR, "grouped_daily.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                daily_data = json.load(f)
            print(f"\n  Loaded existing grouped_daily.json ({len(daily_data)} days)")
        else:
            print(f"ERROR: {json_path} not found. Run without --skip-daily first.")
            sys.exit(1)

    if not daily_data:
        print("\nERROR: No daily data available.")
        sys.exit(1)

    # Save/update grouped_daily.json
    json_path = os.path.join(OUT_DIR, "grouped_daily.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        existing.update(daily_data)
        daily_data = existing
    with open(json_path, "w") as f:
        json.dump(daily_data, f)
    print(f"  Saved grouped_daily.json ({len(daily_data)} total days)")

    # --- STEP 3: Find gap candidates ---
    gap_candidates = find_gap_candidates(daily_data)

    if not any(gap_candidates.values()):
        print("\nERROR: No gap-up candidates found.")
        sys.exit(1)

    # --- STEP 4: Save gainers CSV ---
    save_gainers_csv(gap_candidates)

    # --- STEP 5: Download intraday ---
    intraday_downloaded = 0
    if args["polygon_key"]:
        intraday_downloaded = download_polygon_intraday(args["polygon_key"], gap_candidates)
    elif args["alpaca_key_id"]:
        intraday_downloaded = download_alpaca_intraday(
            args["alpaca_key_id"], args["alpaca_secret"], gap_candidates
        )

    # --- Summary ---
    intraday_files = len([f for f in os.listdir(os.path.join(OUT_DIR, "intraday"))
                          if f.endswith(".csv")])
    daily_files = len([f for f in os.listdir(os.path.join(OUT_DIR, "daily"))
                       if f.endswith(".csv")])

    print(f"\n{'='*70}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"  Period:          {TARGET_START} to {TARGET_END}")
    print(f"  Trading days:    {len([d for d in daily_data if TARGET_START <= d <= TARGET_END])}")
    print(f"  Daily CSVs:      {daily_files}")
    print(f"  Intraday CSVs:   {intraday_files}")
    print(f"  Output dir:      {OUT_DIR}/")
    print(f"{'='*70}")

    if intraday_files == 0:
        print(f"\n  *** NO INTRADAY DATA ***")
        print(f"  The backtest requires 2-min intraday bars for trade simulation.")
        print(f"  Daily data (grouped_daily.json, daily CSVs) is ready.")
        print(f"")
        print(f"  To get intraday data, choose one of:")
        print(f"")
        print(f"  Option A - Alpaca (FREE, recommended):")
        print(f"    1. Sign up at https://app.alpaca.markets/signup (no credit card)")
        print(f"    2. Get API keys from the dashboard")
        print(f"    3. Run: python download_2024_jan_feb.py --alpaca YOUR_KEY YOUR_SECRET --skip-daily")
        print(f"")
        print(f"  Option B - Polygon (PAID, $29/mo):")
        print(f"    1. Upgrade at https://polygon.io/pricing (Stocks Starter)")
        print(f"    2. Run: python download_2024_jan_feb.py --polygon YOUR_KEY --skip-daily")
    else:
        print(f"\n  Data is ready! Next steps:")
        print(f"  1. Generate picks:")
        print(f"     python -c \"from test_full import load_picks_for_dir; load_picks_for_dir('{OUT_DIR}')\"")
        print(f"  2. Run backtest:")
        print(f"     python test_full.py {OUT_DIR}")
