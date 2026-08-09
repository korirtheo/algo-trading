"""Build a unified index of (date -> top-20 tickers) for a date range
from all existing stored_data*/daily_top_gainers.csv files.

Merges across directories (some overlap, e.g. stored_data_combined), dedupes,
and keeps the top-20 by gap_pct per day.

Usage:
  python build_gainers_index.py                          # 2024-01-01 -> now  (default)
  python build_gainers_index.py --start 2022-01-01 --end 2023-12-31 --out gainers_index_2022_2023.json
"""
import csv
import json
import glob
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--start", default="2024-01-01")
parser.add_argument("--end", default="2099-12-31")
parser.add_argument("--out", default="gainers_index_2024_2026.json")
parser.add_argument("--live-watchlists", default="replay_watchlists.json")
args = parser.parse_args()

START = args.start
END = args.end
OUT = args.out
LIVE_WATCHLISTS = args.live_watchlists

by_date = defaultdict(dict)  # date -> {ticker: gap_pct}

for path in sorted(glob.glob("stored_data*/daily_top_gainers.csv")):
    try:
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            cols = r.fieldnames
            for row in r:
                d = row.get("date", "")
                if not d or d < START or d > END:
                    continue
                t = row.get("ticker", "")
                if not t:
                    continue
                if "gap_pct" in cols:
                    gap = row["gap_pct"]
                else:
                    gap = row.get("gap", "")
                try:
                    gap = float(gap) if gap not in ("", None) else 0.0
                except ValueError:
                    gap = 0.0
                # keep the larger gap_pct if a ticker appears in multiple files for same date
                if t not in by_date[d] or gap > by_date[d][t]:
                    by_date[d][t] = gap
    except Exception as e:
        print(f"  ERR {path}: {e}")

# Add live watchlists
if os.path.exists(LIVE_WATCHLISTS):
    with open(LIVE_WATCHLISTS) as f:
        live = json.load(f)
    for d, picks in live.items():
        if d < START:
            continue
        for p in picks:
            t = p["ticker"]
            gap = p.get("gap_pct", 0.0)
            if t not in by_date[d] or gap > by_date[d][t]:
                by_date[d][t] = gap
    print(f"  Merged live watchlists from {LIVE_WATCHLISTS}")

# Keep top-20 per day
index = {}
all_tickers = set()
for d in sorted(by_date):
    top = sorted(by_date[d].items(), key=lambda kv: -kv[1])[:20]
    index[d] = {t: round(g, 2) for t, g in top}
    all_tickers.update(t for t, _ in top)

# Per-ticker date range
ticker_ranges = {}
for d, tk in index.items():
    for t in tk:
        rng = ticker_ranges.setdefault(t, [d, d])
        rng[0] = min(rng[0], d)
        rng[1] = max(rng[1], d)

# actual appearance counts
from collections import Counter as _C
appear = _C()
for d, tk in index.items():
    for t in tk:
        appear[t] += 1

out = {"days": len(index), "tickers": len(all_tickers), "index": index, "ticker_ranges": ticker_ranges, "appearances": dict(appear)}
with open(OUT, "w") as f:
    json.dump(out, f)

print(f"Days: {len(index)} | unique tickers: {len(all_tickers)}")
print(f"Range: {min(index)} -> {max(index)}")
# distribution of appearance counts
cnt = _C(appear.values())
print(f"Tickers by #appearances: {dict(sorted(cnt.items()))}")
print(f"\nSample day: {list(index.items())[0]}")
