"""Download float/shares outstanding data for all gap-up tickers via yfinance."""
import os
import sys
import time
import json
import yfinance as yf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

TICKER_FILE = "float_tickers.txt"
OUTPUT_FILE = "float_data.json"

def main():
    with open(TICKER_FILE) as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"Downloading float data for {len(tickers)} tickers...")

    # Load existing progress
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        print(f"  Resuming: {len(existing)} already downloaded")

    results = dict(existing)
    errors = 0
    start = time.time()

    for i, ticker in enumerate(tickers):
        if ticker in results:
            continue

        try:
            info = yf.Ticker(ticker).info
            float_shares = info.get("floatShares")
            outstanding = info.get("sharesOutstanding")
            market_cap = info.get("marketCap")

            results[ticker] = {
                "floatShares": float_shares,
                "sharesOutstanding": outstanding,
                "marketCap": market_cap,
            }

            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - start
                done = i + 1 - len(existing)
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(tickers) - i - 1) / rate / 60 if rate > 0 else 0
                have_float = sum(1 for v in results.values() if v.get("floatShares"))
                print(f"  [{i+1}/{len(tickers)}] {ticker}: float={float_shares or 'N/A'} | "
                      f"{have_float} with float | {remaining:.0f}m remaining", flush=True)

        except Exception as e:
            results[ticker] = {"floatShares": None, "sharesOutstanding": None, "marketCap": None}
            errors += 1
            if errors <= 5:
                print(f"  [{i+1}] {ticker}: ERROR - {str(e)[:60]}", flush=True)

        # Save every 100 tickers
        if (i + 1) % 100 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=1)

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=1)

    elapsed = time.time() - start
    have_float = sum(1 for v in results.values() if v.get("floatShares"))
    print(f"\nDone! {len(results)} tickers in {elapsed/60:.1f} minutes")
    print(f"  With float data: {have_float}")
    print(f"  Missing float:   {len(results) - have_float}")
    print(f"  Errors:          {errors}")
    print(f"  Saved to:        {OUTPUT_FILE}")

    # Quick stats
    floats = [v["floatShares"] for v in results.values() if v.get("floatShares")]
    if floats:
        import numpy as np
        arr = np.array(floats)
        print(f"\n  Float distribution:")
        print(f"    < 10M:    {sum(1 for x in arr if x < 10e6)}")
        print(f"    10M-50M:  {sum(1 for x in arr if 10e6 <= x < 50e6)}")
        print(f"    50M-200M: {sum(1 for x in arr if 50e6 <= x < 200e6)}")
        print(f"    > 200M:   {sum(1 for x in arr if x >= 200e6)}")


if __name__ == "__main__":
    main()
