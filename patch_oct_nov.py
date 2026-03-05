"""
Patch Oct-Nov OOS data: re-select top 20 without warrants, download missing tickers.
(Backtest scripts filter to top 10 by premarket volume >= 1M)
"""
import os
import sys
import re
import time
import json
import pandas as pd
import requests
from zoneinfo import ZoneInfo

API_KEY = sys.argv[1] if len(sys.argv) > 1 else None
if not API_KEY:
    print("Usage: python patch_oct_nov.py API_KEY")
    sys.exit(1)

OOS_DIR = "stored_data_oos"
MIN_GAP_PCT = 2.0
MAX_PRICE = 50.0
TOP_N = 20
RATE_LIMIT = 5
CALL_INTERVAL = 60.0 / RATE_LIMIT + 0.5
BASE_URL = "https://api.polygon.io"


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


def call_api(url, params=None):
    if params is None:
        params = {}
    params["apiKey"] = API_KEY
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
            else:
                print(f"\n    HTTP {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"\n    Request error: {e}")
            time.sleep(5)
    return None


# --- Step 1: Load existing grouped daily and recompute Oct-Nov top 10 ---
print("=" * 70)
print("  PATCH: Re-select Oct-Nov top 10 without warrants")
print("=" * 70)

json_path = os.path.join(OOS_DIR, "grouped_daily.json")
with open(json_path) as f:
    daily_data = json.load(f)

all_dates = sorted(daily_data.keys())
prev_closes = {}
new_candidates = {}

for date_str in all_dates:
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
    if "2025-10-01" <= date_str <= "2025-11-28":
        new_candidates[date_str] = day_candidates[:TOP_N]

total = sum(len(v) for v in new_candidates.values())
print(f"\n  Oct-Nov dates: {len(new_candidates)}")
print(f"  Total candidates (warrant-filtered): {total}")

# --- Step 2: Find missing intraday ticker-days ---
intraday_dir = os.path.join(OOS_DIR, "intraday")
existing_intraday = set(
    f.replace(".csv", "") for f in os.listdir(intraday_dir) if f.endswith(".csv")
)

missing_ticker_days = []
for date_str, candidates in sorted(new_candidates.items()):
    for c in candidates:
        if c["ticker"] not in existing_intraday:
            missing_ticker_days.append((date_str, c["ticker"], c["gap_pct"]))

# Unique tickers needing daily bars
all_new_tickers = set()
for candidates in new_candidates.values():
    for c in candidates:
        all_new_tickers.add(c["ticker"])

daily_dir = os.path.join(OOS_DIR, "daily")
existing_daily = set(
    f.replace(".csv", "") for f in os.listdir(daily_dir) if f.endswith(".csv")
)
missing_daily = sorted(all_new_tickers - existing_daily)

print(f"  Missing intraday ticker-days: {len(missing_ticker_days)}")
print(f"  Missing daily tickers: {len(missing_daily)}")
est = (len(missing_ticker_days) + len(missing_daily)) * CALL_INTERVAL / 60
print(f"  Estimated time: {est:.0f} minutes")

# --- Step 3: Download missing intraday data ---
if missing_ticker_days:
    print(f"\nStep 3: Downloading {len(missing_ticker_days)} missing intraday ticker-days...")
    completed = 0
    failed = 0
    for date_str, ticker, gap_pct in missing_ticker_days:
        completed += 1
        sys.stdout.write(f"\r  [{completed}/{len(missing_ticker_days)}] {date_str} {ticker} (gap {gap_pct:.1f}%)...")
        sys.stdout.flush()

        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/2/minute/{date_str}/{date_str}"
        data = call_api(url, {"adjusted": "true", "sort": "asc", "limit": "50000"})

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
        csv_path = os.path.join(intraday_dir, f"{ticker}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
        df.to_csv(csv_path)
        time.sleep(CALL_INTERVAL)

    print(f"\r  Downloaded {completed - failed}/{len(missing_ticker_days)} ticker-days. Failed: {failed}.")

# --- Step 4: Download missing daily bars ---
if missing_daily:
    print(f"\nStep 4: Downloading daily bars for {len(missing_daily)} tickers...")
    daily_start = "2025-09-01"
    daily_end = "2025-11-28"
    completed = 0
    for ticker in missing_daily:
        completed += 1
        sys.stdout.write(f"\r  [{completed}/{len(missing_daily)}] {ticker}...")
        sys.stdout.flush()

        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{daily_start}/{daily_end}"
        data = call_api(url, {"adjusted": "true", "sort": "asc", "limit": "5000"})

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

        df = pd.DataFrame(rows).set_index("Date")
        df.to_csv(os.path.join(daily_dir, f"{ticker}.csv"))
        time.sleep(CALL_INTERVAL)

    print(f"\r  Downloaded daily bars for {completed} tickers.")

# --- Step 5: Update daily_top_gainers.csv for Oct-Nov ---
print("\nStep 5: Updating daily_top_gainers.csv...")
rows = []
for date_str, candidates in sorted(new_candidates.items()):
    for c in candidates:
        rows.append({"date": date_str, "ticker": c["ticker"], "gap_pct": c["gap_pct"]})

df_new = pd.DataFrame(rows)
csv_path = os.path.join(OOS_DIR, "daily_top_gainers.csv")

if os.path.exists(csv_path):
    existing = pd.read_csv(csv_path)
    # Remove old Oct-Nov entries, keep Aug-Sep and Dec entries
    existing = existing[~((existing["date"] >= "2025-10-01") & (existing["date"] <= "2025-11-28"))]
    df_new = pd.concat([existing, df_new])
    df_new = df_new.sort_values(["date", "gap_pct"], ascending=[True, False])

df_new.to_csv(csv_path, index=False)
print(f"  Saved {len(df_new)} total entries ({len(rows)} Oct-Nov entries)")

# --- Step 6: Delete pickle cache ---
pkl_path = os.path.join(OOS_DIR, "precomputed_picks.pkl")
if os.path.exists(pkl_path):
    os.remove(pkl_path)
    print(f"\n  Deleted {pkl_path}")

print(f"\n{'='*70}")
print("  PATCH COMPLETE - Oct-Nov top 10 now warrant-free")
print(f"{'='*70}")
