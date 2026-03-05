"""
Resume download for Jan-Mar 2025 stock data.
Skips Step 1 (grouped daily) since we already have grouped_daily.json.
Resumes intraday and daily bar downloads, skipping already-downloaded data.

Usage:
  python resume_download.py
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

# --- CONFIG ---
API_KEY = "XKAw9xOfkdhbNT9iKpZit_npwf010c8q"
OOS_DIR = "stored_data_jan_mar_2025"
TARGET_START = "2025-01-01"
TARGET_END = "2025-03-31"
MIN_GAP_PCT = 2.0
TOP_N = 20
MAX_PRICE = 50.0
RATE_LIMIT = 5
CALL_INTERVAL = 60.0 / RATE_LIMIT + 0.5  # ~12.5s between calls

ET_TZ = ZoneInfo("America/New_York")
BASE_URL = "https://api.polygon.io"


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


def call_api(url, params=None):
    """Make API call with rate limiting and retries."""
    if params is None:
        params = {}
    params["apiKey"] = API_KEY

    for attempt in range(10):
        try:
            resp = requests.get(url, params=params, timeout=45)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"\n    Rate limited, waiting {wait}s... (attempt {attempt+1}/10)")
                time.sleep(wait)
                continue
            elif resp.status_code == 403:
                print(f"\n    ERROR 403: API key may be invalid or plan doesn't support this endpoint")
                return None
            else:
                print(f"\n    HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(10)
                continue
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"\n    Request error (attempt {attempt+1}/10): {e}")
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)

    return None


def find_gap_candidates(daily_data):
    """Find gap-up candidates by comparing open to previous close."""
    print(f"\nFinding gap-up candidates (>{MIN_GAP_PCT}% gap, <${MAX_PRICE})...")

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

    total_downloads = sum(len(v) for v in gap_candidates.values())
    days_with_gaps = sum(1 for v in gap_candidates.values() if len(v) > 0)
    print(f"  Found {total_downloads} gap-up candidates across {days_with_gaps} days")

    return gap_candidates


def download_intraday(gap_candidates):
    """Download 2-min intraday bars, skipping already-downloaded ticker-dates."""
    total = sum(len(v) for v in gap_candidates.values())

    os.makedirs(os.path.join(OOS_DIR, "intraday"), exist_ok=True)
    os.makedirs(os.path.join(OOS_DIR, "daily"), exist_ok=True)

    # Pre-scan existing intraday files to skip already-downloaded ticker-dates
    idir = os.path.join(OOS_DIR, "intraday")
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

    # Count skippable vs to-download
    to_download = []
    skipped = 0
    for date_str in sorted(gap_candidates.keys()):
        for cand in gap_candidates[date_str]:
            ticker = cand["ticker"]
            if date_str in existing_dates_by_ticker.get(ticker, set()):
                skipped += 1
            else:
                to_download.append((date_str, cand))

    print(f"\nStep 1: Downloading 2-min intraday bars...")
    print(f"  Total candidates: {total}, Already have: {skipped}, To download: {len(to_download)}")
    est_time = len(to_download) * CALL_INTERVAL / 60
    print(f"  Estimated time: {est_time:.0f} minutes\n")

    if len(to_download) == 0:
        print("  Nothing to download - all intraday data already exists!")
        return 0

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

        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/2/minute/{date_str}/{date_str}"
        data = call_api(url, {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
        })

        if data is None or data.get("resultsCount", 0) == 0:
            failed += 1
            time.sleep(CALL_INTERVAL)
            continue

        bars = data["results"]
        rows = []
        for bar in bars:
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

        csv_path = os.path.join(OOS_DIR, "intraday", f"{ticker}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
        df.to_csv(csv_path)

        time.sleep(CALL_INTERVAL)

    print(f"\r  Downloaded {completed - failed}/{len(to_download)} ticker-days. "
          f"Skipped {skipped} existing. Failed: {failed}.          ")
    return completed - failed


def download_daily_bars(gap_candidates):
    """Download daily bars for gap-up tickers (needed for prev_close in backtest)."""
    all_tickers = set()
    for candidates in gap_candidates.values():
        for c in candidates:
            all_tickers.add(c["ticker"])

    os.makedirs(os.path.join(OOS_DIR, "daily"), exist_ok=True)

    daily_start = (pd.Timestamp(TARGET_START) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    daily_end = TARGET_END

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
                    if first <= daily_start and last >= TARGET_END:
                        already_have += 1
                        continue
            except Exception:
                pass
        need_download.append(ticker)

    print(f"\nStep 2: Downloading daily bars for {len(all_tickers)} unique tickers...")
    print(f"  Already have: {already_have}, To download: {len(need_download)}")
    est_time = len(need_download) * CALL_INTERVAL / 60
    print(f"  Estimated time: {est_time:.0f} minutes\n")

    if len(need_download) == 0:
        print("  Nothing to download - all daily data already exists!")
        return

    completed = 0
    failed = 0
    for ticker in need_download:
        completed += 1
        sys.stdout.write(f"\r  [{completed}/{len(need_download)}] {ticker}...     ")
        sys.stdout.flush()

        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{daily_start}/{daily_end}"
        data = call_api(url, {
            "adjusted": "true",
            "sort": "asc",
            "limit": "5000",
        })

        if data is None or data.get("resultsCount", 0) == 0:
            failed += 1
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

        csv_path = os.path.join(OOS_DIR, "daily", f"{ticker}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
        df.to_csv(csv_path)

        time.sleep(CALL_INTERVAL)

    print(f"\r  Downloaded daily bars for {completed - failed}/{len(need_download)} tickers. "
          f"Skipped {already_have} existing. Failed: {failed}.          ")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def run_download():
    """Main download logic — called in a retry loop."""
    print(f"{'='*70}")
    print(f"  RESUME DOWNLOAD: Jan-Mar 2025 Stock Data")
    print(f"  Period: {TARGET_START} to {TARGET_END}")
    print(f"  Output: {OOS_DIR}/")
    print(f"{'='*70}")

    # Load existing grouped daily JSON (skip re-downloading ~60 API calls)
    json_path = os.path.join(OOS_DIR, "grouped_daily.json")
    if not os.path.exists(json_path):
        print(f"\nERROR: {json_path} not found. Run the full download_polygon.py first.")
        sys.exit(1)

    print(f"\nLoading existing grouped_daily.json...")
    with open(json_path, "r") as f:
        daily_data = json.load(f)
    print(f"  Loaded {len(daily_data)} trading days of data")

    # Re-derive gap candidates from the existing data
    gap_candidates = find_gap_candidates(daily_data)

    if not any(gap_candidates.values()):
        print("\nERROR: No gap-up candidates found in the data.")
        sys.exit(1)

    # Show current state
    intraday_count = len([f for f in os.listdir(os.path.join(OOS_DIR, "intraday"))
                          if f.endswith(".csv")]) if os.path.exists(os.path.join(OOS_DIR, "intraday")) else 0
    daily_count = len([f for f in os.listdir(os.path.join(OOS_DIR, "daily"))
                       if f.endswith(".csv")]) if os.path.exists(os.path.join(OOS_DIR, "daily")) else 0
    print(f"\nCurrent state:")
    print(f"  Intraday CSVs: {intraday_count}")
    print(f"  Daily CSVs:    {daily_count}")

    # Download remaining intraday bars
    download_intraday(gap_candidates)

    # Download remaining daily bars
    download_daily_bars(gap_candidates)

    # Summary
    intraday_count = len([f for f in os.listdir(os.path.join(OOS_DIR, "intraday"))
                          if f.endswith(".csv")])
    daily_count = len([f for f in os.listdir(os.path.join(OOS_DIR, "daily"))
                       if f.endswith(".csv")])

    print(f"\n{'='*70}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  Period:          {TARGET_START} to {TARGET_END}")
    print(f"  Trading days:    {len(daily_data)}")
    print(f"  Intraday CSVs:   {intraday_count}")
    print(f"  Daily CSVs:      {daily_count}")
    print(f"  Output dir:      {OOS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    MAX_RESTARTS = 20
    for restart in range(MAX_RESTARTS):
        try:
            run_download()
            break  # completed successfully
        except Exception as e:
            print(f"\n\n  !!! CRASH (attempt {restart+1}/{MAX_RESTARTS}): {e}")
            print(f"  Restarting in 30s (all progress saved)...\n")
            time.sleep(30)
    else:
        print(f"\n  Failed after {MAX_RESTARTS} restarts.")
