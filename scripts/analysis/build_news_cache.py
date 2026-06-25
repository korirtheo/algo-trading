"""Pre-cache Alpaca News for all 2022-2026 picks tickers.

Stores per-(ticker, date) news with timestamps so the simulator can
apply a point-in-time news filter without hitting the API mid-trial.

Output: results/news_cache/master.json
        {
          "TICKER|YYYY-MM-DD": [{"ts": "<iso>", "headline": "...", "source": "..."}],
          ...
        }

The cache is incremental — resume-safe and skips already-fetched keys.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from test_full import load_all_picks

DATA_DIRS = [
    "stored_data_2021", "stored_data_2022", "stored_data_2023",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026",
]

CACHE_DIR = "results/news_cache"
CACHE_PATH = os.path.join(CACHE_DIR, "master.json")
PROGRESS_PATH = os.path.join(CACHE_DIR, "progress.txt")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load existing cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Resuming with {len(cache)} cached entries")
    else:
        print("Fresh cache")

    # Load picks
    present = [d for d in DATA_DIRS if os.path.exists(d)]
    print(f"Loading picks from {len(present)} dirs...")
    all_dates, picks_by_date = load_all_picks(present)
    print(f"Total days: {len(all_dates)}")

    # Build (ticker, date) work list — filter to 2022+ (need train + validation)
    targets = []
    for d in all_dates:
        if int(d[:4]) < 2022:  # skip 2021 unless we want it later
            continue
        for p in picks_by_date.get(d, []):
            t = p.get("ticker")
            if not t: continue
            key = f"{t}|{d}"
            if key not in cache:
                targets.append((t, d))
    print(f"Need to fetch {len(targets)} (ticker, date) keys")

    client = NewsClient(
        "PKIPXFIETM7H4BAGQ64FQV3IWJ",
        "25RY682kuN9EBcFr6SFxgbSpFjdZkn613PLKPy1TdYzG"
    )

    t_start = time.time()
    fail_count = 0
    for i, (ticker, d) in enumerate(targets):
        key = f"{ticker}|{d}"
        start = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(days=2)
        end = start + timedelta(days=4)
        try:
            n = client.get_news(NewsRequest(symbols=ticker, start=start, end=end, limit=30))
            arts = n.data.get("news", [])
            cache[key] = [
                {"ts": h.created_at.isoformat(), "headline": h.headline, "source": h.source}
                for h in arts
            ]
        except Exception as e:
            fail_count += 1
            cache[key] = []
            if fail_count > 100:
                print(f"\n[FATAL] too many failures, stopping")
                break

        if i % 50 == 0 and i > 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            remaining = (len(targets) - i) / rate / 60
            print(f"  [{i}/{len(targets)}] {ticker} {d}  rate={rate:.1f}/s  remaining ~{remaining:.0f} min", flush=True)

        # Flush cache every 200 entries
        if i % 200 == 0 and i > 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f)

        time.sleep(0.04)  # ~25/sec, conservative

    # Final flush
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"\nDone. Cache size: {len(cache)} entries, {fail_count} failures")
    print(f"Wrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
