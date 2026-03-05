"""
Download out-of-sample 2-minute intraday data from Polygon.io
=============================================================
Smart approach:
  Step 1: Use "grouped daily" endpoint to get all stocks' OHLCV for each day (1 API call/day)
  Step 2: Find gap-up candidates (>2% gap from previous close)
  Step 3: Download 2-min bars (with premarket) only for gap-up candidates (~10-15/day)

This minimizes API calls vs downloading all 2,700 tickers.

Usage:
  python download_polygon.py YOUR_API_KEY [START_DATE] [END_DATE]

Examples:
  python download_polygon.py KEY123                    # defaults to Oct-Nov 2025
  python download_polygon.py KEY123 2025-08-01 2025-09-30  # Aug-Sep 2025

Free plan: 5 API calls/minute, 2 years of history.
"""
import os
import sys
import re
import time
import json
import datetime
import pandas as pd
import requests
from zoneinfo import ZoneInfo


def _is_warrant_or_unit(ticker):
    """Filter out warrants (.WS, W suffix), rights (R suffix), units (U suffix)."""
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

# --- CONFIG ---
OOS_DIR = os.environ.get("POLYGON_OUT_DIR", "stored_data_oos")
MIN_GAP_PCT = 2.0
TOP_N = 20  # download wider pool; backtest filters to 10 by premarket volume
MAX_PRICE = 50.0  # skip stocks above $50 (not small caps)
RATE_LIMIT = 5  # calls per minute (free plan)
CALL_INTERVAL = 60.0 / RATE_LIMIT + 0.5  # seconds between calls (with buffer)

# Target period (overridable via CLI args)
TARGET_START = sys.argv[2] if len(sys.argv) > 3 else "2025-10-01"
TARGET_END = sys.argv[3] if len(sys.argv) > 3 else "2025-11-28"

ET_TZ = ZoneInfo("America/New_York")
BASE_URL = "https://api.polygon.io"


def get_trading_days(api_key, start_date, end_date):
    """Get list of trading days by checking market status."""
    # Generate all business days, we'll filter non-trading days later
    dates = pd.bdate_range(start=start_date, end=end_date)
    return [d.strftime("%Y-%m-%d") for d in dates]


def call_api(url, api_key, params=None):
    """Make API call with rate limiting and retries."""
    if params is None:
        params = {}
    params["apiKey"] = api_key

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limited — wait and retry
                wait = 15 * (attempt + 1)
                print(f"\n    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            elif resp.status_code == 403:
                print(f"\n    ERROR 403: API key may be invalid or plan doesn't support this endpoint")
                return None
            else:
                print(f"\n    HTTP {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"\n    Request error: {e}")
            time.sleep(5)

    return None


def download_grouped_daily(api_key, trading_days):
    """Step 1: Download grouped daily bars (all stocks in 1 call per day).
    Returns dict: date -> list of {ticker, open, close, high, low, volume}"""
    print(f"\nStep 1: Downloading grouped daily bars for {len(trading_days)} days...")
    print(f"  (1 API call per day, {CALL_INTERVAL:.0f}s between calls)\n")

    daily_data = {}
    skipped_dates = []

    for i, date_str in enumerate(trading_days):
        sys.stdout.write(f"\r  [{i+1}/{len(trading_days)}] {date_str}...")
        sys.stdout.flush()

        url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        data = call_api(url, api_key, {"adjusted": "true"})

        if data is None or data.get("resultsCount", 0) == 0:
            skipped_dates.append(date_str)
            time.sleep(CALL_INTERVAL)
            continue

        stocks = []
        for bar in data.get("results", []):
            stocks.append({
                "ticker": bar["T"],
                "open": bar["o"],
                "close": bar["c"],
                "high": bar["h"],
                "low": bar["l"],
                "volume": bar.get("v", 0),
            })

        daily_data[date_str] = stocks
        time.sleep(CALL_INTERVAL)

    print(f"\r  Downloaded {len(daily_data)} trading days. Skipped {len(skipped_dates)} non-trading days.")
    return daily_data


def find_gap_candidates(daily_data):
    """Step 2: Find gap-up candidates by comparing open to previous close."""
    print(f"\nStep 2: Finding gap-up candidates (>{MIN_GAP_PCT}% gap, <${MAX_PRICE})...")

    sorted_dates = sorted(daily_data.keys())
    # Build prev_close lookup: ticker -> close on each day
    prev_closes = {}  # ticker -> most recent close

    gap_candidates = {}  # date -> list of {ticker, gap_pct, ...}

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

            # Check gap vs previous close
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

            # Update prev close for next day
            if s["close"] is not None and s["close"] > 0:
                prev_closes[ticker] = s["close"]

        # Sort by gap % and take top N
        day_candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
        gap_candidates[date_str] = day_candidates[:TOP_N]

    total_downloads = sum(len(v) for v in gap_candidates.values())
    days_with_gaps = sum(1 for v in gap_candidates.values() if len(v) > 0)
    print(f"  Found {total_downloads} gap-up candidates across {days_with_gaps} days")
    est_time = total_downloads * CALL_INTERVAL / 60
    print(f"  Estimated download time for intraday data: {est_time:.0f} minutes")

    return gap_candidates


def download_intraday(api_key, gap_candidates):
    """Step 3: Download 2-min intraday bars for each gap-up candidate."""
    total = sum(len(v) for v in gap_candidates.values())

    os.makedirs(os.path.join(OOS_DIR, "intraday"), exist_ok=True)
    os.makedirs(os.path.join(OOS_DIR, "daily"), exist_ok=True)

    # Pre-scan existing intraday files to skip already-downloaded ticker-dates
    idir = os.path.join(OOS_DIR, "intraday")
    existing_dates_by_ticker = {}  # ticker -> set of dates already in file
    for date_str in sorted(gap_candidates.keys()):
        for cand in gap_candidates[date_str]:
            ticker = cand["ticker"]
            if ticker in existing_dates_by_ticker:
                continue  # already scanned this ticker
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

    # Count how many we can skip
    to_download = []
    skipped = 0
    for date_str in sorted(gap_candidates.keys()):
        for cand in gap_candidates[date_str]:
            ticker = cand["ticker"]
            if date_str in existing_dates_by_ticker.get(ticker, set()):
                skipped += 1
            else:
                to_download.append((date_str, cand))

    print(f"\nStep 3: Downloading 2-min intraday bars...")
    print(f"  Total candidates: {total}, Already have: {skipped}, To download: {len(to_download)}")
    print(f"  (1 API call per ticker-day, {CALL_INTERVAL:.0f}s between calls)\n")

    completed = 0
    failed = 0

    for date_str, cand in to_download:
            ticker = cand["ticker"]
            completed += 1
            sys.stdout.write(
                f"\r  [{completed}/{len(to_download)}] {date_str} {ticker} "
                f"(gap {cand['gap_pct']:.1f}%)..."
            )
            sys.stdout.flush()

            # Download 2-min bars for this date (extended hours: 4AM-8PM ET)
            url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/2/minute/{date_str}/{date_str}"
            data = call_api(url, api_key, {
                "adjusted": "true",
                "sort": "asc",
                "limit": "50000",
            })

            if data is None or data.get("resultsCount", 0) == 0:
                failed += 1
                time.sleep(CALL_INTERVAL)
                continue

            # Convert to DataFrame matching yfinance format
            bars = data["results"]
            rows = []
            for bar in bars:
                # Polygon timestamp is in milliseconds UTC
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

            # Append to existing CSV (ticker may have data from multiple days)
            csv_path = os.path.join(OOS_DIR, "intraday", f"{ticker}.csv")
            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)
            df.to_csv(csv_path)

            time.sleep(CALL_INTERVAL)

    print(f"\r  Downloaded {completed - failed}/{len(to_download)} ticker-days. "
          f"Skipped {skipped} existing. Failed: {failed}.")
    return completed - failed


def download_daily_bars(api_key, gap_candidates):
    """Download daily bars for gap-up tickers (needed for prev_close in backtest)."""
    # Collect unique tickers
    all_tickers = set()
    for candidates in gap_candidates.values():
        for c in candidates:
            all_tickers.add(c["ticker"])

    os.makedirs(os.path.join(OOS_DIR, "daily"), exist_ok=True)

    # Get a wider date range for daily data (need prev close before target start)
    daily_start = (pd.Timestamp(TARGET_START) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    daily_end = TARGET_END

    # Skip tickers that already have daily data covering the target period
    ddir = os.path.join(OOS_DIR, "daily")
    need_download = []
    already_have = 0
    for ticker in sorted(all_tickers):
        csv_path = os.path.join(ddir, f"{ticker}.csv")
        if os.path.exists(csv_path):
            try:
                ddf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if len(ddf) > 0:
                    first = str(ddf.index[0])[:10]
                    last = str(ddf.index[-1])[:10]
                    # Already covers the range we need
                    if first <= daily_start and last >= TARGET_END:
                        already_have += 1
                        continue
            except Exception:
                pass
        need_download.append(ticker)

    print(f"\nStep 4: Downloading daily bars for {len(all_tickers)} unique tickers...")
    print(f"  Already have: {already_have}, To download: {len(need_download)}")

    completed = 0
    for ticker in need_download:
        completed += 1
        sys.stdout.write(f"\r  [{completed}/{len(need_download)}] {ticker}...")
        sys.stdout.flush()

        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{daily_start}/{daily_end}"
        data = call_api(url, api_key, {
            "adjusted": "true",
            "sort": "asc",
            "limit": "5000",
        })

        if data is None or data.get("resultsCount", 0) == 0:
            time.sleep(CALL_INTERVAL)
            continue

        rows = []
        for bar in data["results"]:
            ts = pd.Timestamp(bar["t"], unit="ms", tz="UTC")
            rows.append({
                "Date": ts,
                "Open": bar["o"],
                "High": bar["h"],
                "Low": bar["l"],
                "Close": bar["c"],
                "Volume": bar.get("v", 0),
            })

        df = pd.DataFrame(rows)
        df.set_index("Date", inplace=True)

        # Merge with existing daily data (don't overwrite)
        csv_path = os.path.join(OOS_DIR, "daily", f"{ticker}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
        df.to_csv(csv_path)

        time.sleep(CALL_INTERVAL)

    print(f"\r  Downloaded daily bars for {completed} tickers. "
          f"Skipped {already_have} existing.")


def save_gainers_csv(gap_candidates):
    """Append to daily_top_gainers.csv (merges with existing data)."""
    rows = []
    for date_str, candidates in sorted(gap_candidates.items()):
        for c in candidates:
            rows.append({
                "date": date_str,
                "ticker": c["ticker"],
                "gap_pct": c["gap_pct"],
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OOS_DIR, "daily_top_gainers.csv")

    # Append to existing CSV if it exists
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df])
        df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
        df = df.sort_values(["date", "gap_pct"], ascending=[True, False])

    df.to_csv(csv_path, index=False)
    print(f"\n  Saved {len(df)} total entries to {csv_path} ({len(rows)} new)")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_polygon.py API_KEY [START_DATE] [END_DATE]")
        print("  e.g. python download_polygon.py KEY123 2025-08-01 2025-09-30")
        print("\nGet your free API key at: https://polygon.io/dashboard/signup")
        sys.exit(1)

    api_key = sys.argv[1]
    print(f"{'='*70}")
    print(f"  POLYGON.IO OUT-OF-SAMPLE DATA DOWNLOADER")
    print(f"  Period: {TARGET_START} to {TARGET_END}")
    print(f"  Output: {OOS_DIR}/")
    print(f"{'='*70}")

    # Create output directory
    os.makedirs(OOS_DIR, exist_ok=True)

    # Step 1: Get all trading days
    trading_days = get_trading_days(api_key, TARGET_START, TARGET_END)
    print(f"\nTarget: {len(trading_days)} business days from {TARGET_START} to {TARGET_END}")

    # Step 2: Download grouped daily bars
    daily_data = download_grouped_daily(api_key, trading_days)

    if not daily_data:
        print("\nERROR: No daily data downloaded. Check your API key.")
        sys.exit(1)

    # Merge with existing grouped daily data
    json_path = os.path.join(OOS_DIR, "grouped_daily.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing = json.load(f)
        existing.update(daily_data)
        daily_data = existing
    with open(json_path, "w") as f:
        json.dump(daily_data, f)
    print(f"  Saved raw daily data to {json_path} ({len(daily_data)} total days)")

    # Step 3: Find gap-up candidates
    gap_candidates = find_gap_candidates(daily_data)

    if not any(gap_candidates.values()):
        print("\nERROR: No gap-up candidates found.")
        sys.exit(1)

    # Save gainers CSV
    save_gainers_csv(gap_candidates)

    # Step 4: Download 2-min intraday bars for candidates
    download_intraday(api_key, gap_candidates)

    # Step 5: Download daily bars for prev_close
    download_daily_bars(api_key, gap_candidates)

    # Summary
    total_intraday = len([f for f in os.listdir(os.path.join(OOS_DIR, "intraday"))
                         if f.endswith(".csv")])
    total_daily = len([f for f in os.listdir(os.path.join(OOS_DIR, "daily"))
                      if f.endswith(".csv")])

    print(f"\n{'='*70}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  Period:          {TARGET_START} to {TARGET_END}")
    print(f"  Trading days:    {len(daily_data)}")
    print(f"  Intraday CSVs:   {total_intraday}")
    print(f"  Daily CSVs:      {total_daily}")
    print(f"  Output dir:      {OOS_DIR}/")
    print(f"{'='*70}")
    print(f"\n  Next steps:")
    print(f"  1. Update test_full.py DATA_DIR to '{OOS_DIR}'")
    print(f"  2. Delete the pickle cache: del stored_data\\precomputed_picks.pkl")
    print(f"  3. Run: python test_full.py")
