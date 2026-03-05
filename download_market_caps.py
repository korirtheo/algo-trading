"""
Download market cap data using Yahoo Finance for all tickers in our picks.
yfinance provides market cap for free (no API key needed).
Saves to market_cap_data.json.
"""
import os
import sys
import json
import time
import yfinance as yf
from optimize import load_all_picks


CACHE_FILE = "market_cap_data.json"
BATCH_SIZE = 50  # yfinance handles batches well


def get_needed_tickers():
    """Collect all unique tickers from all datasets."""
    all_dirs = []
    for d in ["stored_data", "stored_data_oos", "stored_data_jan_mar_2025", "stored_data_apr_jun_2025"]:
        if os.path.exists(d):
            all_dirs.append(d)

    if not all_dirs:
        print("No data directories found!")
        return set()

    picks = load_all_picks(all_dirs)
    tickers = set()
    for date, plist in picks.items():
        for p in plist:
            tickers.add(p["ticker"])
    return tickers


def download_market_caps():
    """Download market caps using Yahoo Finance."""
    needed = get_needed_tickers()
    print(f"Need market cap for {len(needed)} tickers")

    # Load existing cache (only keep entries with actual market_cap values)
    existing = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            raw = json.load(f)
        # Only keep entries that have a real market_cap value
        for k, v in raw.items():
            if v.get("market_cap") is not None:
                existing[k] = v
        print(f"Loaded {len(existing)} with valid market_cap from cache")

    # Check what's missing
    missing = needed - set(existing.keys())
    if not missing:
        print("All tickers already cached!")
        return existing

    print(f"Missing: {len(missing)} tickers")
    missing_list = sorted(missing)

    # Process in batches
    fetched = {}
    for i in range(0, len(missing_list), BATCH_SIZE):
        batch = missing_list[i:i + BATCH_SIZE]
        batch_str = " ".join(batch)
        done = i + len(batch)
        sys.stdout.write(f"\r  [{done}/{len(missing_list)}] Fetching batch...")
        sys.stdout.flush()

        try:
            tickers_obj = yf.Tickers(batch_str)
            for ticker in batch:
                try:
                    info = tickers_obj.tickers[ticker].info
                    mc = info.get("marketCap")
                    fetched[ticker] = {
                        "market_cap": mc,
                        "name": info.get("shortName", info.get("longName", "")),
                        "type": info.get("quoteType", ""),
                    }
                except Exception:
                    fetched[ticker] = {"market_cap": None, "name": "", "type": ""}
        except Exception as e:
            print(f"\n  Batch error: {e}")
            # Fall back to one-by-one for this batch
            for ticker in batch:
                try:
                    t = yf.Ticker(ticker)
                    info = t.info
                    mc = info.get("marketCap")
                    fetched[ticker] = {
                        "market_cap": mc,
                        "name": info.get("shortName", info.get("longName", "")),
                        "type": info.get("quoteType", ""),
                    }
                except Exception:
                    fetched[ticker] = {"market_cap": None, "name": "", "type": ""}

        # Save progress every 200 tickers
        if done % 200 == 0 or done == len(missing_list):
            merged = {**existing, **fetched}
            with open(CACHE_FILE, "w") as f:
                json.dump(merged, f, indent=2)

        time.sleep(0.5)  # be polite

    # Final merge and save
    existing.update(fetched)
    with open(CACHE_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    found = sum(1 for t in needed if t in existing and existing[t].get("market_cap") is not None)
    print(f"\n\nSaved {len(existing)} tickers to {CACHE_FILE}")
    print(f"Market cap available for {found}/{len(needed)} needed tickers")

    # Summary
    import numpy as np
    caps = [existing[t]["market_cap"] for t in needed if t in existing and existing[t].get("market_cap") is not None]
    if caps:
        caps = np.array(caps)
        print(f"\nMarket Cap Distribution of our picks:")
        brackets = [(0, 100e6), (100e6, 500e6), (500e6, 1e9), (1e9, 5e9), (5e9, 50e9), (50e9, float('inf'))]
        labels = ["Nano (<$100M)", "Micro ($100M-$500M)", "Small ($500M-$1B)", "Mid-Small ($1B-$5B)", "Mid ($5B-$50B)", "Large (>$50B)"]
        for (lo, hi), label in zip(brackets, labels):
            count = ((caps >= lo) & (caps < hi)).sum()
            pct = count / len(caps) * 100
            print(f"  {label:<25} {count:>5} ({pct:>5.1f}%)")

        under_5b = (caps < 5e9).sum()
        print(f"\n  Under $5B: {under_5b}/{len(caps)} ({under_5b/len(caps)*100:.1f}%)")
        over_5b = (caps >= 5e9).sum()
        print(f"  Over $5B:  {over_5b}/{len(caps)} ({over_5b/len(caps)*100:.1f}%)")

    return existing


if __name__ == "__main__":
    download_market_caps()
