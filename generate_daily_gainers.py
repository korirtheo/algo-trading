"""
Generate daily top 10 premarket gainers CSV from stored data.
Processes one ticker at a time to avoid loading everything into RAM.
Gap % = (market_open - prev_close) / prev_close
"""
import os
import sys
import pandas as pd
from zoneinfo import ZoneInfo

DATA_DIR = "stored_data"
MIN_GAP_PCT = 2.0
TOP_N = 10
ET_TZ = ZoneInfo("America/New_York")

intraday_dir = os.path.join(DATA_DIR, "intraday")
daily_dir = os.path.join(DATA_DIR, "daily")

tickers = [f.replace(".csv", "") for f in os.listdir(intraday_dir) if f.endswith(".csv")]
print(f"Processing {len(tickers)} tickers...")

# Build a dict: date -> list of candidates
# Process one ticker at a time to save memory
date_candidates = {}
for i, t in enumerate(tickers):
    if (i + 1) % 100 == 0:
        sys.stdout.write(f"\r  {i+1}/{len(tickers)} tickers processed...")
        sys.stdout.flush()

    ipath = os.path.join(intraday_dir, f"{t}.csv")
    dpath = os.path.join(daily_dir, f"{t}.csv")

    try:
        idf = pd.read_csv(ipath, index_col=0, parse_dates=True)
    except Exception:
        continue
    if len(idf) == 0:
        continue

    # Convert to ET
    if idf.index.tz is not None:
        et_index = idf.index.tz_convert(ET_TZ)
    else:
        et_index = idf.index.tz_localize("UTC").tz_convert(ET_TZ)

    # Load daily data for prev close
    ddf = None
    if os.path.exists(dpath):
        try:
            ddf = pd.read_csv(dpath, index_col=0, parse_dates=True)
        except Exception:
            pass

    # Get unique ET dates
    et_dates = et_index.strftime("%Y-%m-%d").unique()

    for date_str in et_dates:
        day_mask = et_index.strftime("%Y-%m-%d") == date_str
        et_day = et_index[day_mask]
        day_candles = idf[day_mask]

        # First hour (9:30-10:30 AM ET)
        fh_mask = (
            ((et_day.hour == 9) & (et_day.minute >= 30))
            | ((et_day.hour == 10) & (et_day.minute <= 30))
        )
        first_hour = day_candles[fh_mask]
        if len(first_hour) == 0:
            continue

        market_open = float(first_hour.iloc[0]["Open"])

        # Premarket high
        pm_mask = (et_day.hour < 9) | ((et_day.hour == 9) & (et_day.minute < 30))
        premarket = day_candles[pm_mask]
        pm_high = float(premarket["High"].max()) if len(premarket) > 0 else market_open

        # Previous close
        prev_close = None
        if ddf is not None and len(ddf) > 0:
            date_naive = pd.Timestamp(date_str)
            ddf_dates = ddf.index.tz_localize(None) if ddf.index.tz else ddf.index
            prev_mask = ddf_dates < date_naive
            if prev_mask.any():
                prev_close = float(ddf.loc[ddf.index[prev_mask][-1], "Close"])

        if prev_close is None or prev_close <= 0:
            continue

        gap_pct = (market_open - prev_close) / prev_close * 100
        if gap_pct < MIN_GAP_PCT:
            continue

        if date_str not in date_candidates:
            date_candidates[date_str] = []
        date_candidates[date_str].append({
            "ticker": t,
            "gap_pct": gap_pct,
            "market_open": market_open,
            "premarket_high": pm_high,
            "prev_close": prev_close,
        })

print(f"\r  {len(tickers)}/{len(tickers)} tickers processed.   ")

# Build final CSV: top 10 per day
rows = []
for date_str in sorted(date_candidates.keys()):
    candidates = date_candidates[date_str]
    candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
    for rank, c in enumerate(candidates[:TOP_N], 1):
        rows.append({
            "date": date_str,
            "rank": rank,
            "ticker": c["ticker"],
            "gap_pct": round(c["gap_pct"], 2),
            "market_open": round(c["market_open"], 4),
            "premarket_high": round(c["premarket_high"], 4),
            "prev_close": round(c["prev_close"], 4),
        })

df = pd.DataFrame(rows)
path = os.path.join(DATA_DIR, "daily_top_gainers.csv")
df.to_csv(path, index=False)

dates = sorted(df["date"].unique())
print(f"\nSaved {len(df)} rows across {len(dates)} trading days to {path}")
print(f"Date range: {dates[0]} to {dates[-1]}")

# Show a few days
for d in dates[:2] + dates[-1:]:
    print(f"\n  {d}:")
    day = df[df["date"] == d]
    for _, r in day.iterrows():
        print(f"    #{int(r['rank'])} {r['ticker']:6s}  gap={r['gap_pct']:6.1f}%  open=${r['market_open']:.4f}  pm_high=${r['premarket_high']:.4f}  prev_cl=${r['prev_close']:.4f}")
