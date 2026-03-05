"""
Validate ALL data completeness across 7 months (Aug 2025 - Feb 2026).
Checks both OOS (stored_data_oos) and in-sample (stored_data) directories.

Usage: python validate_data.py
"""
import os
import sys
import pandas as pd
from collections import defaultdict
from zoneinfo import ZoneInfo

OOS_DIR = "stored_data_oos"
IS_DIR = "stored_data"
ET_TZ = ZoneInfo("America/New_York")


def check_directory(data_dir, label, expected_months):
    """Validate a single data directory. Returns (ok, missing_months, missing_intraday_count)."""
    idir = os.path.join(data_dir, "intraday")
    ddir = os.path.join(data_dir, "daily")
    gainers_path = os.path.join(data_dir, "daily_top_gainers.csv")

    print(f"\n  {'='*60}")
    print(f"  {label}: {data_dir}/")
    print(f"  {'='*60}")

    # --- 1. Check gainers CSV ---
    if not os.path.exists(gainers_path):
        print(f"  ERROR: {gainers_path} not found!")
        return False, expected_months, 0

    gdf = pd.read_csv(gainers_path)
    gdf["month"] = gdf["date"].str[:7]
    print(f"\n  Gainers CSV: {len(gdf)} entries across {gdf['date'].nunique()} dates")

    # Monthly breakdown
    print(f"\n  {'Month':<10} {'Dates':>6} {'Entries':>8} {'Tickers':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8}")
    months = sorted(gdf["month"].unique())
    for m in months:
        mdf = gdf[gdf["month"] == m]
        print(f"  {m:<10} {mdf['date'].nunique():>6} {len(mdf):>8} {mdf['ticker'].nunique():>8}")

    # --- 2. Check expected months ---
    missing_months = [m for m in expected_months if m not in months]
    if missing_months:
        print(f"\n  WARNING: Missing months: {', '.join(missing_months)}")
    else:
        print(f"\n  OK: All expected months present ({', '.join(expected_months)})")

    # --- 3. Check intraday completeness ---
    print(f"\n  --- Intraday Data Check ---")
    missing_intraday = []
    ok_intraday = 0

    if not os.path.exists(idir):
        print(f"  ERROR: {idir} directory not found!")
        return False, missing_months, len(gdf)

    for _, row in gdf.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        csv_path = os.path.join(idir, f"{ticker}.csv")

        if not os.path.exists(csv_path):
            missing_intraday.append((date, ticker, "no file"))
            continue

        try:
            idf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if len(idf) == 0:
                missing_intraday.append((date, ticker, "empty file"))
                continue

            if idf.index.tz is not None:
                et = idf.index.tz_convert(ET_TZ)
            else:
                et = idf.index.tz_localize("UTC").tz_convert(ET_TZ)

            if (et.strftime("%Y-%m-%d") == date).any():
                ok_intraday += 1
            else:
                missing_intraday.append((date, ticker, "date not in file"))
        except Exception as e:
            missing_intraday.append((date, ticker, f"read error: {e}"))

    print(f"  OK: {ok_intraday}/{len(gdf)} ticker-dates have intraday data")
    if missing_intraday:
        by_month = defaultdict(int)
        for d, t, r in missing_intraday:
            by_month[d[:7]] += 1
        print(f"  Missing: {len(missing_intraday)} ticker-dates")
        for m in sorted(by_month):
            print(f"    {m}: {by_month[m]} missing")
        if len(missing_intraday) <= 30:
            for d, t, r in sorted(missing_intraday)[:30]:
                print(f"      {d} {t}: {r}")

    # --- 4. Check daily completeness ---
    print(f"\n  --- Daily Data Check ---")
    missing_daily = []
    no_prev_close = []
    ok_daily = 0

    if not os.path.exists(ddir):
        print(f"  ERROR: {ddir} directory not found!")
    else:
        unique_tickers = gdf["ticker"].unique()
        for ticker in unique_tickers:
            csv_path = os.path.join(ddir, f"{ticker}.csv")
            if not os.path.exists(csv_path):
                missing_daily.append(ticker)
                continue

            try:
                ddf = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if len(ddf) == 0:
                    missing_daily.append(ticker)
                    continue

                first_date = str(ddf.index[0])[:10]
                last_date = str(ddf.index[-1])[:10]
                # Check if daily data has enough history for prev_close
                ticker_dates = gdf[gdf["ticker"] == ticker]["date"].tolist()
                earliest = min(ticker_dates)
                # Need at least 1 day before earliest gap date
                if first_date >= earliest:
                    no_prev_close.append((ticker, first_date, earliest))
                ok_daily += 1
            except Exception:
                missing_daily.append(ticker)

        print(f"  OK: {ok_daily}/{len(unique_tickers)} tickers have daily data")
        if missing_daily:
            print(f"  Missing daily files: {len(missing_daily)}")
            if len(missing_daily) <= 20:
                for t in sorted(missing_daily)[:20]:
                    print(f"    {t}")
        if no_prev_close:
            print(f"  WARNING: {len(no_prev_close)} tickers may lack prev_close")
            for t, first, earliest in sorted(no_prev_close)[:10]:
                print(f"    {t}: daily starts {first}, first gap date {earliest}")

    # --- 5. File counts ---
    print(f"\n  --- File Counts ---")
    if os.path.exists(idir):
        intraday_files = [f for f in os.listdir(idir) if f.endswith(".csv")]
        print(f"  Intraday CSV files: {len(intraday_files)}")
    if os.path.exists(ddir):
        daily_files = [f for f in os.listdir(ddir) if f.endswith(".csv")]
        print(f"  Daily CSV files: {len(daily_files)}")

    return len(missing_months) == 0 and len(missing_intraday) <= 20, missing_months, len(missing_intraday)


def validate():
    print("=" * 70)
    print("  COMPREHENSIVE DATA VALIDATION REPORT (7 MONTHS)")
    print("  Aug 2025 - Feb 2026")
    print("=" * 70)

    results = []

    # --- OOS data (Aug-Dec 2025) ---
    oos_expected = ["2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
    oos_ok, oos_missing_m, oos_missing_i = check_directory(
        OOS_DIR, "OUT-OF-SAMPLE", oos_expected
    )
    results.append(("OOS (Aug-Dec 2025)", oos_ok, oos_missing_m, oos_missing_i))

    # --- In-sample data (Jan-Feb 2026) ---
    is_expected = ["2026-01", "2026-02"]
    is_ok, is_missing_m, is_missing_i = check_directory(
        IS_DIR, "IN-SAMPLE", is_expected
    )
    results.append(("IS (Jan-Feb 2026)", is_ok, is_missing_m, is_missing_i))

    # --- Check pickle caches (stale or present) ---
    print(f"\n  {'='*60}")
    print(f"  PICKLE CACHE STATUS")
    print(f"  {'='*60}")
    cache_files = [
        "fulltest_picks_v2.pkl",
        "optimize_picks_novol.pkl",
        "analyze_picks_novol.pkl",
        "pattern_picks.pkl",
    ]
    stale_caches = []
    for d in [OOS_DIR, IS_DIR]:
        for cf in cache_files:
            path = os.path.join(d, cf)
            if os.path.exists(path):
                stale_caches.append(path)
                print(f"  STALE: {path}")
    if not stale_caches:
        print(f"  OK: No stale pickle caches found")
    else:
        print(f"\n  WARNING: {len(stale_caches)} stale caches should be deleted before running backtest")

    # --- FINAL SUMMARY ---
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    all_ok = True
    for label, ok, missing_m, missing_i in results:
        status = "PASS" if ok else "FAIL"
        details = []
        if missing_m:
            details.append(f"missing months: {', '.join(missing_m)}")
        if missing_i > 20:
            details.append(f"{missing_i} missing intraday")
        detail_str = f" ({'; '.join(details)})" if details else ""
        print(f"  {status}: {label}{detail_str}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\n  ALL VALIDATIONS PASSED - Ready for optimization pipeline")
    else:
        print(f"\n  SOME ISSUES FOUND - Review above and fix before proceeding")

    if stale_caches:
        print(f"  NOTE: Delete {len(stale_caches)} stale caches before running backtest")

    print("=" * 70)
    return all_ok


if __name__ == "__main__":
    validate()
