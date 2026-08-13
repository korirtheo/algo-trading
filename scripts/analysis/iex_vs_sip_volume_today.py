"""Compare bot's IEX-feed-derived cum $vol vs Polygon full-market for today.

For each watchlist ticker:
  1. Pull Polygon 1-min bars from 9:30 ET to current time
  2. Compute cumulative $-volume at 5-min checkpoints
  3. Test whether a fixed multiplier (40x) maps IEX -> SIP accurately

Output:
  - Per-ticker volume curve
  - Per-ticker ratio (Polygon / bot's IEX if we have it, else just Polygon for reference)
  - Cross-ticker variance assessment
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import requests
import datetime as dt
import time
import json
from collections import defaultdict

from config.settings import POLYGON_API_KEY as API_KEY

# Today's watchlist (from live bot log 09:30 ET)
WATCHLIST = [
    "ADTX", "CAST", "CDT", "LNKS", "WKSP", "BYAH", "LPA", "APWC",
    "WOK", "GPUS", "BFLY", "CRVO", "CTNT", "QTEX", "WPRT", "ATPC",
    "SPRO", "BIRD", "RUM", "AVD",
]

# Today's known data point: APWC at 09:33:34 ET had bot cum_$vol = $2,894
KNOWN_BOT_VALUES = {
    "APWC": {"ts": "2026-06-18 09:33:34", "cum_dvol": 2894.0},
}

TODAY = "2026-06-18"
CHECKPOINTS_ET = ["09:33", "09:35", "09:40", "09:45", "10:00", "10:30", "11:00"]


def fetch_polygon_minute(ticker, date):
    """Fetch 1-min bars for ticker on date from Polygon (premarket+market)."""
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
           f"{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 429:
            time.sleep(13)  # free tier rate limit
            r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])
    except Exception as e:
        return []


def cum_dollar_vol_up_to(bars, ts_et):
    """Compute cum $-vol from market open (9:30 ET) up to ts_et."""
    target_h, target_m = map(int, ts_et.split(":"))
    target_unix_ms = (
        dt.datetime(2026, 6, 18, target_h, target_m, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        .timestamp() * 1000
    )
    market_open_ms = (
        dt.datetime(2026, 6, 18, 9, 30, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        .timestamp() * 1000
    )
    cum = 0.0
    for b in bars:
        t = b["t"]
        if t < market_open_ms or t >= target_unix_ms:
            continue
        cum += b["c"] * b["v"]
    return cum


def main():
    print(f"Comparing Polygon full-market vs bot IEX for {len(WATCHLIST)} watchlist tickers on {TODAY}\n")

    results = {}
    for i, ticker in enumerate(WATCHLIST):
        print(f"[{i+1}/{len(WATCHLIST)}] Fetching {ticker}...", end=" ")
        bars = fetch_polygon_minute(ticker, TODAY)
        if not bars:
            print("no data")
            continue
        results[ticker] = {}
        for cp in CHECKPOINTS_ET:
            results[ticker][cp] = cum_dollar_vol_up_to(bars, cp)
        print(f"  {len(bars)} bars")
        time.sleep(13)  # free tier rate limit: 5 calls/min

    print(f"\n{'='*100}")
    print(f"FULL-MARKET CUMULATIVE $-VOLUME (from Polygon) for each watchlist ticker")
    print(f"{'='*100}")
    header = f"  {'ticker':<8}"
    for cp in CHECKPOINTS_ET:
        header += f"  {cp:>10}"
    print(header)
    for ticker, cps in results.items():
        row = f"  {ticker:<8}"
        for cp in CHECKPOINTS_ET:
            v = cps.get(cp, 0)
            if v > 1e6:
                row += f"  {f'${v/1e6:.2f}M':>10}"
            elif v > 1e3:
                row += f"  {f'${v/1e3:.1f}K':>10}"
            else:
                row += f"  {f'${v:.0f}':>10}"
        print(row)

    # Known comparison: APWC at 09:33
    print(f"\n{'='*100}")
    print(f"APWC RATIO: bot's IEX vs Polygon full-market at known timestamp")
    print(f"{'='*100}")
    bot_apwc_0933 = KNOWN_BOT_VALUES["APWC"]["cum_dvol"]
    poly_apwc_0933 = results.get("APWC", {}).get("09:33", 0)
    if poly_apwc_0933 > 0:
        ratio = poly_apwc_0933 / bot_apwc_0933
        print(f"  Bot's view (IEX):       ${bot_apwc_0933:,.0f}")
        print(f"  Polygon full-market:    ${poly_apwc_0933:,.0f}")
        print(f"  Multiplier needed:      {ratio:.1f}x")
        if 20 <= ratio <= 60:
            print(f"  -> Within 'IEX share' range (typical 30-50x); Option C plausible")
        else:
            print(f"  -> Out of typical IEX range; might need per-ticker tuning")

    # Summary of all volumes
    if results:
        print(f"\n{'='*100}")
        print(f"VOLUME RANGE AT 09:35 (5 min after open)")
        print(f"{'='*100}")
        vols_935 = [(t, r.get("09:35", 0)) for t, r in results.items()]
        vols_935.sort(key=lambda x: -x[1])
        for ticker, v in vols_935:
            print(f"  {ticker:<8}  ${v:>14,.0f}")
        med = sorted(v for _, v in vols_935)[len(vols_935) // 2]
        print(f"\n  Median across watchlist: ${med:,.0f}")

    # Save to JSON
    out_path = "results/iex_vs_sip_comparison.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw data saved to {out_path}")


if __name__ == "__main__":
    import os
    main()
