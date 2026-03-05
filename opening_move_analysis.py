"""
Opening Move Pattern Analysis
==============================
Analyze which gap-up stocks rise 5-10% from open within the first 5-15 minutes.
Find common characteristics that predict these moves vs faders.

Focuses on Oct 2025 - Feb 2026 data.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from collections import defaultdict

ET_TZ = ZoneInfo("America/New_York")

# Load picks from stored_data_oos + stored_data (covers Aug-Feb)
def load_picks(data_dirs):
    merged = {}
    for d in data_dirs:
        # Try multiple cache names
        for cache_name in ["fulltest_picks_gap2_vol250k.pkl", "fulltest_picks_gap2_vol0k.pkl"]:
            pkl = os.path.join(d, cache_name)
            if os.path.exists(pkl):
                print(f"  Loading {pkl}...")
                with open(pkl, "rb") as f:
                    picks = pickle.load(f)
                for date_str, day_picks in picks.items():
                    if date_str not in merged:
                        merged[date_str] = day_picks
                    else:
                        existing = {p["ticker"] for p in merged[date_str]}
                        for p in day_picks:
                            if p["ticker"] not in existing:
                                merged[date_str].append(p)
                break
    return merged


def _load_premarket_candles(ticker, date_str, data_dirs):
    """Load raw premarket candles (before 9:30 ET) from intraday CSV."""
    for d in data_dirs:
        ipath = os.path.join(d, "intraday", f"{ticker}.csv")
        if not os.path.exists(ipath):
            continue
        try:
            idf = pd.read_csv(ipath, index_col=0, parse_dates=True)
            if idf.index.tz is not None:
                et_idx = idf.index.tz_convert(ET_TZ)
            else:
                et_idx = idf.index.tz_localize("UTC").tz_convert(ET_TZ)
            idf_et = idf.copy()
            idf_et.index = et_idx

            day_mask = idf_et.index.strftime("%Y-%m-%d") == date_str
            day_candles = idf_et[day_mask]

            pm_mask = (day_candles.index.hour < 9) | (
                (day_candles.index.hour == 9) & (day_candles.index.minute < 30)
            )
            return day_candles[pm_mask]
        except Exception:
            continue
    return None


def analyze_opening_moves(daily_picks, data_dirs, start_date="2025-10-01", end_date="2026-02-28"):
    """Analyze first 15 minutes of each stock to find 5-10% movers."""

    all_stocks = []

    dates = sorted(d for d in daily_picks.keys() if start_date <= d <= end_date)
    print(f"\nAnalyzing {len(dates)} trading days: {dates[0]} to {dates[-1]}")

    for di, date_str in enumerate(dates):
        picks = daily_picks[date_str]
        sys.stdout.write(f"\r  [{di+1}/{len(dates)}] {date_str}...")
        sys.stdout.flush()
        for pick in picks:
            ticker = pick["ticker"]
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) == 0:
                continue

            market_open = pick["market_open"]
            premarket_high = pick["premarket_high"]
            prev_close = pick["prev_close"]
            gap_pct = pick["gap_pct"]
            pm_volume = pick.get("pm_volume", 0)

            if market_open <= 0 or prev_close <= 0:
                continue

            # Convert to ET for time-based analysis
            if mh.index.tz is not None:
                et_idx = mh.index.tz_convert(ET_TZ)
            else:
                et_idx = mh.index.tz_localize("UTC").tz_convert(ET_TZ)

            mh_et = mh.copy()
            mh_et.index = et_idx

            # --- Load premarket candles for last-30-min analysis ---
            pm_candles = _load_premarket_candles(ticker, date_str, data_dirs)
            # Premarket analysis windows
            pm_last30_momentum = 0.0
            pm_last30_vol = 0
            pm_last30_range_pct = 0.0
            pm_last30_trend = "flat"  # up, down, flat
            pm_last30_close_vs_high = 0.0
            # Extended premarket: 7:00-9:30
            pm_7am_momentum = 0.0
            pm_7am_vol = 0
            pm_7am_trend = "flat"
            pm_7am_high = 0.0
            pm_7am_low = 0.0
            pm_7am_close_vs_high = 0.0

            if pm_candles is not None and len(pm_candles) > 0:
                # Last 30 min = 9:00-9:30 ET
                last30 = pm_candles[
                    (pm_candles.index.hour == 9) & (pm_candles.index.minute < 30)
                ]
                if len(last30) >= 2:
                    l30_open = float(last30.iloc[0]["Open"])
                    l30_close = float(last30.iloc[-1]["Close"])
                    l30_high = float(last30["High"].max())
                    l30_low = float(last30["Low"].min())
                    pm_last30_vol = int(last30["Volume"].sum())
                    if l30_open > 0:
                        pm_last30_momentum = (l30_close / l30_open - 1) * 100
                        pm_last30_range_pct = (l30_high - l30_low) / l30_open * 100
                    if premarket_high > 0:
                        pm_last30_close_vs_high = (l30_close / premarket_high - 1) * 100
                    if pm_last30_momentum > 1:
                        pm_last30_trend = "up"
                    elif pm_last30_momentum < -1:
                        pm_last30_trend = "down"

                # 7:00 AM - 9:30 AM window
                from_7am = pm_candles[pm_candles.index.hour >= 7]
                if len(from_7am) >= 2:
                    f7_open = float(from_7am.iloc[0]["Open"])
                    f7_close = float(from_7am.iloc[-1]["Close"])
                    pm_7am_high = float(from_7am["High"].max())
                    pm_7am_low = float(from_7am["Low"].min())
                    pm_7am_vol = int(from_7am["Volume"].sum())
                    if f7_open > 0:
                        pm_7am_momentum = (f7_close / f7_open - 1) * 100
                    if pm_7am_high > 0:
                        pm_7am_close_vs_high = (f7_close / pm_7am_high - 1) * 100
                    if pm_7am_momentum > 1:
                        pm_7am_trend = "up"
                    elif pm_7am_momentum < -1:
                        pm_7am_trend = "down"

            # First candle = market open (9:30)
            open_price = float(mh_et.iloc[0]["Open"])
            if open_price <= 0:
                continue

            # Compute max move in first N minutes
            def max_move_in_window(minutes):
                cutoff_hour = 9
                cutoff_min = 30 + minutes
                if cutoff_min >= 60:
                    cutoff_hour += cutoff_min // 60
                    cutoff_min = cutoff_min % 60

                window = mh_et[mh_et.index.hour * 60 + mh_et.index.minute <= cutoff_hour * 60 + cutoff_min]
                if len(window) == 0:
                    return 0.0, 0.0, 0
                max_high = float(window["High"].max())
                min_low = float(window["Low"].min())
                vol = int(window["Volume"].sum())
                max_up_pct = (max_high / open_price - 1) * 100
                max_down_pct = (min_low / open_price - 1) * 100
                return max_up_pct, max_down_pct, vol

            max_5min_up, max_5min_down, vol_5min = max_move_in_window(6)   # 3 candles
            max_10min_up, max_10min_down, vol_10min = max_move_in_window(10)  # 5 candles
            max_15min_up, max_15min_down, vol_15min = max_move_in_window(16)  # 8 candles
            max_30min_up, max_30min_down, vol_30min = max_move_in_window(30)

            # First candle stats
            first_candle_high = float(mh_et.iloc[0]["High"])
            first_candle_low = float(mh_et.iloc[0]["Low"])
            first_candle_close = float(mh_et.iloc[0]["Close"])
            first_candle_vol = int(mh_et.iloc[0]["Volume"])
            first_candle_range_pct = (first_candle_high - first_candle_low) / open_price * 100
            first_candle_body_pct = (first_candle_close - open_price) / open_price * 100

            # Open vs premarket high
            open_vs_pm_high_pct = (open_price / premarket_high - 1) * 100 if premarket_high > 0 else 0

            # Price level
            price_bucket = (
                "<$2" if open_price < 2 else
                "$2-5" if open_price < 5 else
                "$5-10" if open_price < 10 else
                "$10-20" if open_price < 20 else
                "$20+"
            )

            # Full day stats
            day_high = float(mh_et["High"].max())
            day_low = float(mh_et["Low"].min())
            day_close = float(mh_et.iloc[-1]["Close"])
            day_max_up = (day_high / open_price - 1) * 100
            day_max_down = (day_low / open_price - 1) * 100
            day_return = (day_close / open_price - 1) * 100
            day_volume = int(mh_et["Volume"].sum())

            # PM volume dollar value
            pm_dollar_vol = pm_volume * premarket_high if pm_volume > 0 else 0

            all_stocks.append({
                "date": date_str,
                "ticker": ticker,
                "open_price": open_price,
                "prev_close": prev_close,
                "premarket_high": premarket_high,
                "gap_pct": gap_pct,
                "pm_volume": pm_volume,
                "pm_dollar_vol": pm_dollar_vol,
                "price_bucket": price_bucket,
                "open_vs_pm_high_pct": open_vs_pm_high_pct,
                # Premarket last 30 min
                "pm_last30_momentum": pm_last30_momentum,
                "pm_last30_vol": pm_last30_vol,
                "pm_last30_range_pct": pm_last30_range_pct,
                "pm_last30_trend": pm_last30_trend,
                "pm_last30_close_vs_high": pm_last30_close_vs_high,
                # PM 7am-9:30 window
                "pm_7am_momentum": pm_7am_momentum,
                "pm_7am_vol": pm_7am_vol,
                "pm_7am_trend": pm_7am_trend,
                "pm_7am_high": pm_7am_high,
                "pm_7am_close_vs_high": pm_7am_close_vs_high,
                # First candle
                "first_candle_range_pct": first_candle_range_pct,
                "first_candle_body_pct": first_candle_body_pct,
                "first_candle_vol": first_candle_vol,
                # Max moves by time window
                "max_5min_up": max_5min_up,
                "max_10min_up": max_10min_up,
                "max_15min_up": max_15min_up,
                "max_30min_up": max_30min_up,
                "max_5min_down": max_5min_down,
                "max_10min_down": max_10min_down,
                "max_15min_down": max_15min_down,
                "max_30min_down": max_30min_down,
                "vol_5min": vol_5min,
                "vol_10min": vol_10min,
                "vol_15min": vol_15min,
                # Full day
                "day_max_up": day_max_up,
                "day_max_down": day_max_down,
                "day_return": day_return,
                "day_volume": day_volume,
            })

    print(f"\r  Done.                                ")
    df = pd.DataFrame(all_stocks)
    print(f"  Total stocks analyzed: {len(df)}")
    return df


def print_analysis(df):
    """Print comprehensive pattern analysis."""

    print(f"\n{'='*80}")
    print(f"  OPENING MOVE PATTERN ANALYSIS")
    print(f"  {df['date'].min()} to {df['date'].max()} | {df['date'].nunique()} days | {len(df)} stocks")
    print(f"{'='*80}")

    # --- 1. How many stocks hit +5-10% from open? ---
    print(f"\n--- HOW OFTEN DO STOCKS MOVE +5-10% FROM OPEN? ---\n")

    for window, col in [("5 min", "max_5min_up"), ("10 min", "max_10min_up"),
                         ("15 min", "max_15min_up"), ("30 min", "max_30min_up"),
                         ("Full day", "day_max_up")]:
        hit_5 = (df[col] >= 5).sum()
        hit_10 = (df[col] >= 10).sum()
        hit_15 = (df[col] >= 15).sum()
        hit_20 = (df[col] >= 20).sum()
        print(f"  {window:>8}: {hit_5:>4} ({hit_5/len(df)*100:5.1f}%) hit +5%  |  "
              f"{hit_10:>4} ({hit_10/len(df)*100:5.1f}%) hit +10%  |  "
              f"{hit_15:>4} ({hit_15/len(df)*100:5.1f}%) hit +15%  |  "
              f"{hit_20:>4} ({hit_20/len(df)*100:5.1f}%) hit +20%")

    # --- 2. What predicts a +5% move in 15 min? ---
    print(f"\n--- PREDICTORS: +5% IN FIRST 15 MIN vs REST ---\n")

    movers = df[df["max_15min_up"] >= 5]
    faders = df[df["max_15min_up"] < 5]
    print(f"  Movers (+5% in 15min): {len(movers)} ({len(movers)/len(df)*100:.1f}%)")
    print(f"  Faders (< +5%):        {len(faders)} ({len(faders)/len(df)*100:.1f}%)")

    features = [
        ("gap_pct", "Gap %"),
        ("pm_volume", "PM Volume"),
        ("pm_dollar_vol", "PM $ Volume"),
        ("open_price", "Open Price"),
        ("open_vs_pm_high_pct", "Open vs PM High %"),
        ("first_candle_range_pct", "1st Candle Range %"),
        ("first_candle_body_pct", "1st Candle Body %"),
        ("first_candle_vol", "1st Candle Volume"),
    ]

    print(f"\n  {'Feature':<22} {'Movers Avg':>14} {'Faders Avg':>14} {'Movers Med':>14} {'Faders Med':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    for col, label in features:
        m_avg = movers[col].mean()
        f_avg = faders[col].mean()
        m_med = movers[col].median()
        f_med = faders[col].median()
        if col in ("pm_volume", "pm_dollar_vol", "first_candle_vol"):
            print(f"  {label:<22} {m_avg:>14,.0f} {f_avg:>14,.0f} {m_med:>14,.0f} {f_med:>14,.0f}")
        else:
            print(f"  {label:<22} {m_avg:>14.1f} {f_avg:>14.1f} {m_med:>14.1f} {f_med:>14.1f}")

    # --- 3. Gap % buckets ---
    print(f"\n--- +5% IN 15MIN BY GAP SIZE ---\n")
    gap_bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 200)]
    print(f"  {'Gap Range':<14} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Hit +10%':>9} {'Rate':>8} {'Avg Max Up':>11}")
    print(f"  {'-'*14} {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*11}")
    for lo, hi in gap_bins:
        bucket = df[(df["gap_pct"] >= lo) & (df["gap_pct"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        h10 = (bucket["max_15min_up"] >= 10).sum()
        print(f"  {lo:>3}-{hi:>3}%     {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{h10:>9} {h10/len(bucket)*100:>7.1f}% {bucket['max_15min_up'].mean():>10.1f}%")

    # --- 4. Price buckets ---
    print(f"\n--- +5% IN 15MIN BY PRICE LEVEL ---\n")
    print(f"  {'Price':<10} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Hit +10%':>9} {'Rate':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
    for bucket_name in ["<$2", "$2-5", "$5-10", "$10-20", "$20+"]:
        bucket = df[df["price_bucket"] == bucket_name]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        h10 = (bucket["max_15min_up"] >= 10).sum()
        print(f"  {bucket_name:<10} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{h10:>9} {h10/len(bucket)*100:>7.1f}%")

    # --- 5. PM Volume buckets ---
    print(f"\n--- +5% IN 15MIN BY PREMARKET VOLUME ---\n")
    vol_bins = [(0, 50_000), (50_000, 250_000), (250_000, 1_000_000),
                (1_000_000, 5_000_000), (5_000_000, float("inf"))]
    vol_labels = ["<50K", "50K-250K", "250K-1M", "1M-5M", "5M+"]
    print(f"  {'PM Volume':<12} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Hit +10%':>9} {'Rate':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
    for (lo, hi), label in zip(vol_bins, vol_labels):
        bucket = df[(df["pm_volume"] >= lo) & (df["pm_volume"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        h10 = (bucket["max_15min_up"] >= 10).sum()
        print(f"  {label:<12} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{h10:>9} {h10/len(bucket)*100:>7.1f}%")

    # --- 6. Open vs PM High (gap above PM high = strong?) ---
    print(f"\n--- +5% IN 15MIN BY OPEN vs PM HIGH ---\n")
    opm_bins = [(-100, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 100)]
    opm_labels = ["<-5% below", "-5 to -2%", "-2 to 0%", "0 to +2%", "+2 to +5%", ">+5% above"]
    print(f"  {'Open vs PMH':<14} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Avg Max Up':>11}")
    print(f"  {'-'*14} {'-'*6} {'-'*8} {'-'*8} {'-'*11}")
    for (lo, hi), label in zip(opm_bins, opm_labels):
        bucket = df[(df["open_vs_pm_high_pct"] >= lo) & (df["open_vs_pm_high_pct"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        print(f"  {label:<14} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{bucket['max_15min_up'].mean():>10.1f}%")

    # --- 7. First candle body (green vs red open) ---
    print(f"\n--- +5% IN 15MIN BY FIRST CANDLE (9:30-9:32) ---\n")
    green_open = df[df["first_candle_body_pct"] > 0]
    red_open = df[df["first_candle_body_pct"] <= 0]
    for label, subset in [("Green 1st candle", green_open), ("Red 1st candle", red_open)]:
        if len(subset) == 0:
            continue
        h5 = (subset["max_15min_up"] >= 5).sum()
        h10 = (subset["max_15min_up"] >= 10).sum()
        print(f"  {label:<20} n={len(subset):>5}  |  +5%: {h5} ({h5/len(subset)*100:.1f}%)  |  "
              f"+10%: {h10} ({h10/len(subset)*100:.1f}%)  |  "
              f"Avg max: {subset['max_15min_up'].mean():.1f}%")

    # Strong green (>2% first candle) vs weak
    strong_green = df[df["first_candle_body_pct"] > 2]
    print(f"  {'Strong green (>2%)':20} n={len(strong_green):>5}  |  +5%: "
          f"{(strong_green['max_15min_up'] >= 5).sum()} "
          f"({(strong_green['max_15min_up'] >= 5).sum()/max(len(strong_green),1)*100:.1f}%)  |  "
          f"Avg max: {strong_green['max_15min_up'].mean():.1f}%")

    # --- 8. Premarket last 30 min momentum ---
    print(f"\n--- +5% IN 15MIN BY PREMARKET LAST 30 MIN (9:00-9:30) ---\n")
    pm_bins = [(-100, -3), (-3, -1), (-1, 0), (0, 1), (1, 3), (3, 100)]
    pm_labels = ["<-3% (selling)", "-3 to -1%", "-1 to 0%", "0 to +1%", "+1 to +3%", ">+3% (surging)"]
    print(f"  {'PM30 Momentum':<18} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Avg 15m Up':>11}")
    print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*8} {'-'*11}")
    for (lo, hi), label in zip(pm_bins, pm_labels):
        bucket = df[(df["pm_last30_momentum"] >= lo) & (df["pm_last30_momentum"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        print(f"  {label:<18} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{bucket['max_15min_up'].mean():>10.1f}%")

    # --- 8b. PM last 30 min trend (up/down/flat) ---
    print(f"\n  By PM30 trend direction:")
    for trend in ["up", "flat", "down"]:
        subset = df[df["pm_last30_trend"] == trend]
        if len(subset) == 0:
            continue
        h5 = (subset["max_15min_up"] >= 5).sum()
        h10 = (subset["max_15min_up"] >= 10).sum()
        print(f"    {trend:>5}: n={len(subset):>5}  |  +5%: {h5} ({h5/len(subset)*100:.1f}%)  |  "
              f"+10%: {h10} ({h10/len(subset)*100:.1f}%)  |  "
              f"Avg max: {subset['max_15min_up'].mean():.1f}%")

    # --- 8c. PM close vs PM high (fading into open vs holding) ---
    print(f"\n--- +5% IN 15MIN BY PM CLOSE vs PM HIGH (9:29 vs day high) ---\n")
    cvh_bins = [(-100, -10), (-10, -5), (-5, -2), (-2, 0), (0, 100)]
    cvh_labels = [">10% below PMH", "5-10% below", "2-5% below", "0-2% below (near high)", "At/above PMH"]
    print(f"  {'PM Close vs High':<22} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Avg 15m Up':>11}")
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*11}")
    for (lo, hi), label in zip(cvh_bins, cvh_labels):
        bucket = df[(df["pm_last30_close_vs_high"] >= lo) & (df["pm_last30_close_vs_high"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        print(f"  {label:<22} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{bucket['max_15min_up'].mean():>10.1f}%")

    # --- 8d. PM last 30 min volume ---
    print(f"\n--- +5% IN 15MIN BY PM LAST 30 MIN VOLUME ---\n")
    pvol_bins = [(0, 10_000), (10_000, 50_000), (50_000, 200_000),
                 (200_000, 1_000_000), (1_000_000, float("inf"))]
    pvol_labels = ["<10K", "10K-50K", "50K-200K", "200K-1M", "1M+"]
    print(f"  {'PM30 Volume':<12} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Avg 15m Up':>11}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*11}")
    for (lo, hi), label in zip(pvol_bins, pvol_labels):
        bucket = df[(df["pm_last30_vol"] >= lo) & (df["pm_last30_vol"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        print(f"  {label:<12} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{bucket['max_15min_up'].mean():>10.1f}%")

    # --- 8e. PM 7:00 AM - 9:30 AM momentum ---
    print(f"\n--- +5% IN 15MIN BY PREMARKET 7AM-9:30AM MOMENTUM ---\n")
    pm7_bins = [(-100, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 15), (15, 100)]
    pm7_labels = ["<-5% (dumping)", "-5 to -2%", "-2 to 0%", "0 to +2%", "+2 to +5%", "+5 to +15%", ">+15% (parabolic)"]
    print(f"  {'PM 7am Momentum':<20} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Avg 15m Up':>11}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*11}")
    for (lo, hi), label in zip(pm7_bins, pm7_labels):
        bucket = df[(df["pm_7am_momentum"] >= lo) & (df["pm_7am_momentum"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        print(f"  {label:<20} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{bucket['max_15min_up'].mean():>10.1f}%")

    # --- 8f. PM 7am trend direction ---
    print(f"\n  By PM 7am-9:30 trend:")
    for trend in ["up", "flat", "down"]:
        subset = df[df["pm_7am_trend"] == trend]
        if len(subset) == 0:
            continue
        h5 = (subset["max_15min_up"] >= 5).sum()
        h10 = (subset["max_15min_up"] >= 10).sum()
        print(f"    {trend:>5}: n={len(subset):>5}  |  +5%: {h5} ({h5/len(subset)*100:.1f}%)  |  "
              f"+10%: {h10} ({h10/len(subset)*100:.1f}%)  |  "
              f"Avg max: {subset['max_15min_up'].mean():.1f}%")

    # --- 8g. PM 7am close vs 7am high ---
    print(f"\n--- +5% IN 15MIN BY PM 7AM CLOSE vs 7AM HIGH ---\n")
    cvh7_bins = [(-100, -10), (-10, -5), (-5, -2), (-2, 0), (0, 100)]
    cvh7_labels = [">10% off 7am high", "5-10% off", "2-5% off", "0-2% off (near high)", "At 7am high"]
    print(f"  {'7am Close vs High':<22} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Avg 15m Up':>11}")
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*11}")
    for (lo, hi), label in zip(cvh7_bins, cvh7_labels):
        bucket = df[(df["pm_7am_close_vs_high"] >= lo) & (df["pm_7am_close_vs_high"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        print(f"  {label:<22} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{bucket['max_15min_up'].mean():>10.1f}%")

    # --- 8h. PM 7am volume ---
    print(f"\n--- +5% IN 15MIN BY PM 7AM-9:30 VOLUME ---\n")
    pvol7_bins = [(0, 50_000), (50_000, 200_000), (200_000, 500_000),
                  (500_000, 2_000_000), (2_000_000, float("inf"))]
    pvol7_labels = ["<50K", "50K-200K", "200K-500K", "500K-2M", "2M+"]
    print(f"  {'PM7am Volume':<14} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Avg 15m Up':>11}")
    print(f"  {'-'*14} {'-'*6} {'-'*8} {'-'*8} {'-'*11}")
    for (lo, hi), label in zip(pvol7_bins, pvol7_labels):
        bucket = df[(df["pm_7am_vol"] >= lo) & (df["pm_7am_vol"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        print(f"  {label:<14} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{bucket['max_15min_up'].mean():>10.1f}%")

    # --- 9. First candle range (volatility at open) ---
    print(f"\n--- +5% IN 15MIN BY FIRST CANDLE RANGE ---\n")
    range_bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
    range_labels = ["<2%", "2-5%", "5-10%", "10-20%", ">20%"]
    print(f"  {'1st Range':<10} {'Total':>6} {'Hit +5%':>8} {'Rate':>8} {'Avg 15m Up':>11}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*11}")
    for (lo, hi), label in zip(range_bins, range_labels):
        bucket = df[(df["first_candle_range_pct"] >= lo) & (df["first_candle_range_pct"] < hi)]
        if len(bucket) == 0:
            continue
        h5 = (bucket["max_15min_up"] >= 5).sum()
        print(f"  {label:<10} {len(bucket):>6} {h5:>8} {h5/len(bucket)*100:>7.1f}% "
              f"{bucket['max_15min_up'].mean():>10.1f}%")

    # --- 9. Combined filter: what's the best predictor? ---
    print(f"\n--- BEST COMBINED FILTERS ---\n")

    filters = [
        ("Gap>20% + PM Vol>250K",
         (df["gap_pct"] >= 20) & (df["pm_volume"] >= 250_000)),
        ("Gap>20% + PM Vol>250K + Green 1st",
         (df["gap_pct"] >= 20) & (df["pm_volume"] >= 250_000) & (df["first_candle_body_pct"] > 0)),
        ("Gap>30% + PM Vol>100K",
         (df["gap_pct"] >= 30) & (df["pm_volume"] >= 100_000)),
        ("Gap>10% + Strong green 1st (>2%)",
         (df["gap_pct"] >= 10) & (df["first_candle_body_pct"] > 2)),
        ("Gap>20% + Strong green 1st (>2%)",
         (df["gap_pct"] >= 20) & (df["first_candle_body_pct"] > 2)),
        ("Gap>10% + Open > PM High",
         (df["gap_pct"] >= 10) & (df["open_vs_pm_high_pct"] > 0)),
        ("Gap>20% + Open > PM High",
         (df["gap_pct"] >= 20) & (df["open_vs_pm_high_pct"] > 0)),
        ("Gap>10% + 1st range>5% + green",
         (df["gap_pct"] >= 10) & (df["first_candle_range_pct"] > 5) & (df["first_candle_body_pct"] > 0)),
        ("Gap>10% + PM$Vol>$500K",
         (df["gap_pct"] >= 10) & (df["pm_dollar_vol"] >= 500_000)),
        ("Gap>20% + PM$Vol>$1M",
         (df["gap_pct"] >= 20) & (df["pm_dollar_vol"] >= 1_000_000)),
        ("Price $2-10 + Gap>20% + Vol>250K",
         (df["open_price"] >= 2) & (df["open_price"] < 10) & (df["gap_pct"] >= 20) & (df["pm_volume"] >= 250_000)),
        # Premarket momentum filters
        ("Gap>10% + PM30 surging (>+3%)",
         (df["gap_pct"] >= 10) & (df["pm_last30_momentum"] > 3)),
        ("Gap>10% + PM30 up + Green 1st",
         (df["gap_pct"] >= 10) & (df["pm_last30_trend"] == "up") & (df["first_candle_body_pct"] > 0)),
        ("Gap>20% + PM near high (<2%)",
         (df["gap_pct"] >= 20) & (df["pm_last30_close_vs_high"] > -2)),
        ("Gap>10% + PM near high + PM30 up",
         (df["gap_pct"] >= 10) & (df["pm_last30_close_vs_high"] > -2) & (df["pm_last30_trend"] == "up")),
        ("Gap>10% + PM30vol>200K + PM up",
         (df["gap_pct"] >= 10) & (df["pm_last30_vol"] >= 200_000) & (df["pm_last30_trend"] == "up")),
        ("Gap>20% + PM near high + Green 1st",
         (df["gap_pct"] >= 20) & (df["pm_last30_close_vs_high"] > -2) & (df["first_candle_body_pct"] > 0)),
        # 7am-based filters
        ("Gap>10% + PM7am up + near high",
         (df["gap_pct"] >= 10) & (df["pm_7am_trend"] == "up") & (df["pm_7am_close_vs_high"] > -2)),
        ("Gap>10% + PM7am parabolic (>+5%)",
         (df["gap_pct"] >= 10) & (df["pm_7am_momentum"] > 5)),
        ("Gap>20% + PM7am up + Vol>200K",
         (df["gap_pct"] >= 20) & (df["pm_7am_trend"] == "up") & (df["pm_7am_vol"] >= 200_000)),
        ("Gap>10% + PM7am up + PM30 up",
         (df["gap_pct"] >= 10) & (df["pm_7am_trend"] == "up") & (df["pm_last30_trend"] == "up")),
        ("BEST: Gap>10% + PM7am up + near high + Green1st",
         (df["gap_pct"] >= 10) & (df["pm_7am_trend"] == "up") & (df["pm_7am_close_vs_high"] > -2) & (df["first_candle_body_pct"] > 0)),
    ]

    print(f"  {'Filter':<40} {'N':>5} {'Hit+5%':>7} {'Rate':>7} {'Hit+10%':>8} {'Rate':>7} {'Avg Up':>8}")
    print(f"  {'-'*40} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*8}")
    for label, mask in filters:
        subset = df[mask]
        if len(subset) == 0:
            continue
        h5 = (subset["max_15min_up"] >= 5).sum()
        h10 = (subset["max_15min_up"] >= 10).sum()
        print(f"  {label:<40} {len(subset):>5} {h5:>7} {h5/len(subset)*100:>6.1f}% "
              f"{h10:>8} {h10/len(subset)*100:>6.1f}% {subset['max_15min_up'].mean():>7.1f}%")

    # --- 10. What happens AFTER a stock hits +5% in 15 min? ---
    print(f"\n--- AFTER HITTING +5% IN 15 MIN: DOES IT KEEP GOING? ---\n")
    movers = df[df["max_15min_up"] >= 5]
    if len(movers) > 0:
        print(f"  Of {len(movers)} stocks that hit +5% in 15 min:")
        h10_day = (movers["day_max_up"] >= 10).sum()
        h15_day = (movers["day_max_up"] >= 15).sum()
        h20_day = (movers["day_max_up"] >= 20).sum()
        print(f"    Hit +10% during day: {h10_day} ({h10_day/len(movers)*100:.1f}%)")
        print(f"    Hit +15% during day: {h15_day} ({h15_day/len(movers)*100:.1f}%)")
        print(f"    Hit +20% during day: {h20_day} ({h20_day/len(movers)*100:.1f}%)")
        print(f"    Avg day max up:      {movers['day_max_up'].mean():.1f}%")
        print(f"    Avg day return:      {movers['day_return'].mean():.1f}%")
        print(f"    Closed green:        {(movers['day_return'] > 0).sum()} "
              f"({(movers['day_return'] > 0).sum()/len(movers)*100:.1f}%)")

    # --- 11. Top 20 biggest 15-min movers ---
    print(f"\n--- TOP 20 BIGGEST 15-MIN MOVERS ---\n")
    top20 = df.nlargest(20, "max_15min_up")
    print(f"  {'Date':<12} {'Ticker':<8} {'Open':>7} {'Gap%':>6} {'PMVol':>10} "
          f"{'15m Up':>7} {'Day Up':>7} {'Day Ret':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*7} {'-'*7} {'-'*8}")
    for _, row in top20.iterrows():
        print(f"  {row['date']:<12} {row['ticker']:<8} ${row['open_price']:>5.2f} "
              f"{row['gap_pct']:>5.0f}% {row['pm_volume']:>10,.0f} "
              f"{row['max_15min_up']:>6.1f}% {row['day_max_up']:>6.1f}% "
              f"{row['day_return']:>+7.1f}%")


if __name__ == "__main__":
    data_dirs = ["stored_data_oos", "stored_data"]
    if len(sys.argv) > 1:
        data_dirs = sys.argv[1:]

    print("Loading picks...")
    daily_picks = load_picks(data_dirs)

    df = analyze_opening_moves(daily_picks, data_dirs, start_date="2025-10-01", end_date="2026-02-28")
    print_analysis(df)
