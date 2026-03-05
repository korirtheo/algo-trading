"""
Patch missing Aug-Sep OOS data:
  1. Download Sep intraday (completely missing)
  2. Download/fix daily bars for Aug-Sep tickers (need prev close back to July)

Only downloads what's actually missing — won't re-download existing data.

Usage: python patch_aug_sep.py API_KEY
"""
import os
import sys
import time
import pandas as pd
import requests
from zoneinfo import ZoneInfo

OOS_DIR = "stored_data_oos"
RATE_LIMIT = 5
CALL_INTERVAL = 60.0 / RATE_LIMIT + 0.5
BASE_URL = "https://api.polygon.io"
ET_TZ = ZoneInfo("America/New_York")


def call_api(url, api_key, params=None):
    if params is None:
        params = {}
    params["apiKey"] = api_key
    for attempt in range(3):
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
                print(f"\n    ERROR 403: check API key")
                return None
            else:
                print(f"\n    HTTP {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"\n    Error: {e}")
            time.sleep(5)
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_aug_sep.py API_KEY")
        sys.exit(1)

    api_key = sys.argv[1]
    idir = os.path.join(OOS_DIR, "intraday")
    ddir = os.path.join(OOS_DIR, "daily")

    # Load the gainers CSV to know which tickers need data
    gdf = pd.read_csv(os.path.join(OOS_DIR, "daily_top_gainers.csv"))

    # --- STEP 1: Find missing Sep intraday data ---
    print("=" * 70)
    print("  STEP 1: Download missing September intraday data")
    print("=" * 70)

    sep_entries = gdf[gdf["date"].str.startswith("2025-09")]
    aug_entries = gdf[gdf["date"].str.startswith("2025-08")]

    # Check which Sep ticker-dates are missing
    missing_sep = []
    for _, row in sep_entries.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        csv_path = os.path.join(idir, f"{ticker}.csv")
        if os.path.exists(csv_path):
            try:
                idf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if idf.index.tz is not None:
                    et = idf.index.tz_convert(ET_TZ)
                else:
                    et = idf.index.tz_localize("UTC").tz_convert(ET_TZ)
                if (et.strftime("%Y-%m-%d") == date).any():
                    continue  # already has this date
            except Exception:
                pass
        missing_sep.append((date, ticker))

    # Also check missing Aug intraday
    missing_aug = []
    for _, row in aug_entries.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        csv_path = os.path.join(idir, f"{ticker}.csv")
        if os.path.exists(csv_path):
            try:
                idf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if idf.index.tz is not None:
                    et = idf.index.tz_convert(ET_TZ)
                else:
                    et = idf.index.tz_localize("UTC").tz_convert(ET_TZ)
                if (et.strftime("%Y-%m-%d") == date).any():
                    continue
            except Exception:
                pass
        missing_aug.append((date, ticker))

    all_missing = missing_aug + missing_sep
    print(f"  Missing Aug intraday: {len(missing_aug)} ticker-dates")
    print(f"  Missing Sep intraday: {len(missing_sep)} ticker-dates")
    print(f"  Total to download: {len(all_missing)} ticker-dates")
    est = len(all_missing) * CALL_INTERVAL / 60
    print(f"  Estimated time: {est:.0f} minutes\n")

    completed = 0
    failed = 0
    for date_str, ticker in sorted(all_missing):
        completed += 1
        sys.stdout.write(f"\r  [{completed}/{len(all_missing)}] {date_str} {ticker}...")
        sys.stdout.flush()

        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/2/minute/{date_str}/{date_str}"
        data = call_api(url, api_key, {"adjusted": "true", "sort": "asc", "limit": "50000"})

        if data is None or data.get("resultsCount", 0) == 0:
            failed += 1
            time.sleep(CALL_INTERVAL)
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

        df = pd.DataFrame(rows).set_index("Datetime")

        # Append to existing CSV
        csv_path = os.path.join(idir, f"{ticker}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
        df.to_csv(csv_path)
        time.sleep(CALL_INTERVAL)

    print(f"\r  Downloaded {completed - failed}/{len(all_missing)} ticker-dates. Failed: {failed}.")

    # --- STEP 2: Fix daily bars (need prev close going back to July) ---
    print(f"\n{'='*70}")
    print("  STEP 2: Download/fix daily bars for Aug-Sep tickers")
    print("=" * 70)

    # Collect all unique tickers that need daily data
    aug_sep_tickers = set(gdf[gdf["date"] < "2025-10-01"]["ticker"].unique())
    # Also include Oct-Nov tickers to preserve their data
    oct_nov_tickers = set(gdf[gdf["date"] >= "2025-10-01"]["ticker"].unique())
    all_tickers = aug_sep_tickers | oct_nov_tickers

    # We need daily data from July 1 to Nov 28 for complete coverage
    daily_start = "2025-07-01"
    daily_end = "2025-11-28"

    # Only re-download tickers whose daily data doesn't go back far enough
    need_fix = []
    for ticker in sorted(all_tickers):
        csv_path = os.path.join(ddir, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            need_fix.append(ticker)
            continue
        try:
            ddf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            first_date = str(ddf.index[0])[:10]
            # If daily data starts after Aug 1, we need to go back further
            if first_date > "2025-08-01":
                need_fix.append(ticker)
        except Exception:
            need_fix.append(ticker)

    print(f"  Tickers needing daily fix: {len(need_fix)}/{len(all_tickers)}")
    est = len(need_fix) * CALL_INTERVAL / 60
    print(f"  Estimated time: {est:.0f} minutes\n")

    completed = 0
    for ticker in need_fix:
        completed += 1
        sys.stdout.write(f"\r  [{completed}/{len(need_fix)}] {ticker}...")
        sys.stdout.flush()

        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{daily_start}/{daily_end}"
        data = call_api(url, api_key, {"adjusted": "true", "sort": "asc", "limit": "5000"})

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

        new_df = pd.DataFrame(rows).set_index("Date")

        # MERGE with existing daily data (don't overwrite)
        csv_path = os.path.join(ddir, f"{ticker}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            new_df = pd.concat([existing, new_df])
            new_df = new_df[~new_df.index.duplicated(keep="last")]
            new_df.sort_index(inplace=True)
        new_df.to_csv(csv_path)
        time.sleep(CALL_INTERVAL)

    print(f"\r  Fixed daily bars for {completed} tickers.                    ")

    # --- SUMMARY ---
    print(f"\n{'='*70}")
    print("  PATCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Next steps:")
    print(f"  1. Delete stale pickle caches:")
    print(f"     del stored_data_oos\\fulltest_picks_v2.pkl")
    print(f"     del stored_data_oos\\optimize_picks_novol.pkl")
    print(f"     del stored_data_oos\\analyze_picks_novol.pkl")
    print(f"     del stored_data_oos\\pattern_picks.pkl")
    print(f"  2. Re-run: python test_full.py")


if __name__ == "__main__":
    main()
