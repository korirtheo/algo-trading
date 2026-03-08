"""
Pre-filter intraday data to match the format used by stored_data_combined:
- Resample 1-min bars to 2-min bars
- Keep only tickers that appear in daily_top_gainers.csv
- Save to intraday/ directory (overwrites originals)

This reduces data from ~1.5GB to ~150-200MB, making pickle builds 10x faster.

Usage:
    python prefilter_data.py stored_data_2023
    python prefilter_data.py stored_data_2022
"""

import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


def process_ticker(args):
    ticker, intraday_dir, ticker_dates = args
    ipath = os.path.join(intraday_dir, f"{ticker}.csv")
    try:
        df = pd.read_csv(ipath, index_col=0, parse_dates=True)
    except Exception:
        return ticker, 0, 0, "read_error"

    orig_rows = len(df)

    # Check if already 2-min (look at time deltas)
    if len(df) > 1:
        dt = df.index.to_series().diff().median()
        is_1min = dt is not None and dt.total_seconds() < 90
    else:
        is_1min = False

    # Filter to only dates this ticker is a top gainer (per-ticker filtering)
    if df.index.tz is not None:
        dates = df.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
    else:
        try:
            dates = df.index.tz_localize("UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
        except Exception:
            dates = df.index.strftime("%Y-%m-%d")

    mask = dates.isin(ticker_dates)
    df = df[mask]

    if len(df) == 0:
        # No data for gap-up days — remove file
        os.remove(ipath)
        return ticker, orig_rows, 0, "removed"

    # Resample to 2-min if 1-min
    if is_1min:
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if 'open' in cl:
                col_map[c] = 'first'
            elif 'high' in cl:
                col_map[c] = 'max'
            elif 'low' in cl:
                col_map[c] = 'min'
            elif 'close' in cl:
                col_map[c] = 'last'
            elif 'vol' in cl:
                col_map[c] = 'sum'
            else:
                col_map[c] = 'last'

        df = df.resample("2min").agg(col_map).dropna(subset=[df.columns[0]])

    new_rows = len(df)
    df.to_csv(ipath)
    return ticker, orig_rows, new_rows, "ok"


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "stored_data_2023"
    intraday_dir = os.path.join(data_dir, "intraday")
    gainers_csv = os.path.join(data_dir, "daily_top_gainers.csv")

    if not os.path.exists(gainers_csv):
        print(f"ERROR: {gainers_csv} not found")
        sys.exit(1)

    # Get per-ticker dates from gainers (only keep data for days each ticker is a gainer)
    gdf = pd.read_csv(gainers_csv)
    ticker_date_map = gdf.groupby("ticker")["date"].apply(set).to_dict()
    gainer_tickers = set(ticker_date_map.keys())
    total_dates = len(gdf["date"].unique())
    avg_days = sum(len(v) for v in ticker_date_map.values()) / max(len(ticker_date_map), 1)
    print(f"Gainers CSV: {total_dates} dates, {len(gainer_tickers)} unique tickers, avg {avg_days:.1f} gainer days/ticker")

    # Get all intraday files
    all_files = [f.replace(".csv", "") for f in os.listdir(intraday_dir) if f.endswith(".csv")]
    print(f"Intraday files: {len(all_files)}")

    # Remove tickers not in gainers at all
    to_remove = [t for t in all_files if t not in gainer_tickers]
    print(f"Tickers not in gainers (removing): {len(to_remove)}")
    for t in to_remove:
        os.remove(os.path.join(intraday_dir, f"{t}.csv"))

    # Filter remaining tickers
    to_process = [t for t in all_files if t in gainer_tickers]
    print(f"Tickers to filter: {len(to_process)}")

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    args_list = [(t, intraday_dir, ticker_date_map[t]) for t in to_process]

    completed = 0
    total_orig = 0
    total_new = 0
    removed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_ticker, a): a[0] for a in args_list}
        for future in as_completed(futures):
            ticker, orig, new, status = future.result()
            total_orig += orig
            total_new += new
            if status == "removed":
                removed += 1
            completed += 1
            if completed % 50 == 0 or completed == len(to_process):
                print(f"  [{completed}/{len(to_process)}] processed...", flush=True)

    remaining = len([f for f in os.listdir(intraday_dir) if f.endswith(".csv")])
    print(f"\nDone!")
    print(f"  Original rows: {total_orig:,}")
    print(f"  Filtered rows: {total_new:,}")
    print(f"  Reduction: {(1 - total_new/max(total_orig,1))*100:.1f}%")
    print(f"  Removed (no gap-up days): {removed + len(to_remove)}")
    print(f"  Remaining files: {remaining}")


if __name__ == "__main__":
    main()
