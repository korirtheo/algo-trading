"""
Post-First-Candle Pattern Analysis
====================================
We know: Gap>10% + first candle green >2% → buy at first candle close.
Question: What predicts which stocks go +5% FROM ENTRY (not from open)?

Analyzes features of the first candle AND early price action to find
which stocks continue vs fade after entry.
"""

import os
import sys
import pickle
import io
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ET_TZ = ZoneInfo("America/New_York")

# --- Config ---
MIN_GAP_PCT = 10.0
GREEN_CANDLE_MIN_BODY_PCT = 2.0
TARGET_FROM_ENTRY = 5.0  # +5% from entry price
TIME_LIMIT_MIN = 15      # within 15 min of entry

DATE_START = "2025-10-01"
DATE_END = "2026-02-28"


def load_picks(data_dirs):
    merged = {}
    for d in data_dirs:
        pkl = os.path.join(d, "fulltest_picks_gap2_vol250k.pkl")
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
    return merged


def _load_premarket_candles(ticker, date_str, data_dirs):
    """Load premarket candles from 7:00 AM to 9:30 AM."""
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

            # Premarket: 7:00 - 9:30
            pm_mask = (day_candles.index.hour >= 7) & (
                (day_candles.index.hour < 9) |
                ((day_candles.index.hour == 9) & (day_candles.index.minute < 30))
            )
            return day_candles[pm_mask]
        except Exception:
            continue
    return None


def analyze(data_dirs):
    print("Loading data...")
    picks_by_date = load_picks(data_dirs)
    dates = sorted(d for d in picks_by_date.keys() if DATE_START <= d <= DATE_END)
    print(f"  {len(dates)} trading days: {dates[0]} to {dates[-1]}\n")

    records = []

    for date_str in dates:
        for pick in picks_by_date.get(date_str, []):
            if pick["gap_pct"] < MIN_GAP_PCT:
                continue
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) == 0:
                continue

            timestamps = sorted(mh.index)
            if len(timestamps) < 2:
                continue

            # First candle
            fc = mh.loc[timestamps[0]]
            fc_open = float(fc["Open"])
            fc_close = float(fc["Close"])
            fc_high = float(fc["High"])
            fc_low = float(fc["Low"])
            fc_vol = float(fc["Volume"]) if "Volume" in fc.index else 0

            if fc_open <= 0:
                continue
            body_pct = (fc_close - fc_open) / fc_open * 100
            if body_pct < GREEN_CANDLE_MIN_BODY_PCT:
                continue

            entry_price = fc_close * 1.001  # slippage
            target_price = entry_price * (1 + TARGET_FROM_ENTRY / 100)

            # === FIRST CANDLE FEATURES ===
            fc_body_pct = body_pct
            fc_range_pct = (fc_high - fc_low) / fc_open * 100
            fc_upper_wick_pct = (fc_high - fc_close) / fc_open * 100
            fc_lower_wick_pct = (fc_open - fc_low) / fc_open * 100
            fc_close_vs_high = (fc_close - fc_low) / max(fc_high - fc_low, 0.001)  # 0=closed at low, 1=closed at high

            # === SECOND CANDLE FEATURES ===
            sc = mh.loc[timestamps[1]]
            sc_open = float(sc["Open"])
            sc_close = float(sc["Close"])
            sc_high = float(sc["High"])
            sc_low = float(sc["Low"])
            sc_vol = float(sc["Volume"]) if "Volume" in sc.index else 0

            sc_body_pct = (sc_close - sc_open) / sc_open * 100 if sc_open > 0 else 0
            sc_green = sc_close > sc_open
            sc_vs_fc = (sc_close - fc_close) / fc_close * 100  # 2nd candle continuation
            sc_made_new_high = sc_high > fc_high

            # === PRICE/VOLUME FEATURES ===
            price = fc_close
            gap = pick["gap_pct"]
            pm_vol = pick.get("pm_volume", 0)

            # === PREMARKET FEATURES ===
            pm_candles = _load_premarket_candles(pick["ticker"], date_str, data_dirs)
            pm_last30_trend = 0
            pm_last30_vol = 0
            pm_close_vs_high = 0
            pm_momentum_7am = 0
            if pm_candles is not None and len(pm_candles) > 0:
                # Last 30 min of PM (9:00-9:30)
                last30 = pm_candles[pm_candles.index.hour == 9]
                if len(last30) >= 2:
                    pm_last30_open = float(last30.iloc[0]["Open"])
                    pm_last30_close = float(last30.iloc[-1]["Close"])
                    pm_last30_high = last30["High"].max()
                    if pm_last30_open > 0:
                        pm_last30_trend = (pm_last30_close - pm_last30_open) / pm_last30_open * 100
                        pm_close_vs_high = (pm_last30_close - pm_last30_open) / max(pm_last30_high - pm_last30_open, 0.001)
                    pm_last30_vol = last30["Volume"].sum() if "Volume" in last30.columns else 0

                # Full PM momentum (7am to 9:30)
                if len(pm_candles) >= 2:
                    pm_first = float(pm_candles.iloc[0]["Open"])
                    pm_last = float(pm_candles.iloc[-1]["Close"])
                    if pm_first > 0:
                        pm_momentum_7am = (pm_last - pm_first) / pm_first * 100

            # === OUTCOME: does it hit +5% from entry within 15 min? ===
            hit_target = False
            max_gain_15min = 0.0
            max_dd_15min = 0.0
            max_gain_full_day = 0.0
            time_to_target = None

            for i, ts in enumerate(timestamps[1:], 1):
                candle = mh.loc[ts]
                c_high = float(candle["High"])
                c_low = float(candle["Low"])

                gain = (c_high - entry_price) / entry_price * 100
                dd = (entry_price - c_low) / entry_price * 100
                max_gain_full_day = max(max_gain_full_day, gain)

                minutes_elapsed = i * 2  # 2-min candles
                if minutes_elapsed <= TIME_LIMIT_MIN:
                    max_gain_15min = max(max_gain_15min, gain)
                    max_dd_15min = max(max_dd_15min, dd)

                if not hit_target and c_high >= target_price:
                    hit_target = True
                    time_to_target = minutes_elapsed

            records.append({
                "ticker": pick["ticker"],
                "date": date_str,
                "gap_pct": gap,
                "price": price,
                "pm_volume": pm_vol,
                # First candle
                "fc_body_pct": fc_body_pct,
                "fc_range_pct": fc_range_pct,
                "fc_upper_wick_pct": fc_upper_wick_pct,
                "fc_lower_wick_pct": fc_lower_wick_pct,
                "fc_close_vs_high": fc_close_vs_high,
                "fc_volume": fc_vol,
                # Second candle
                "sc_body_pct": sc_body_pct,
                "sc_green": sc_green,
                "sc_vs_fc": sc_vs_fc,
                "sc_made_new_high": sc_made_new_high,
                # Premarket
                "pm_last30_trend": pm_last30_trend,
                "pm_close_vs_high": pm_close_vs_high,
                "pm_momentum_7am": pm_momentum_7am,
                # Outcome
                "hit_target": hit_target,
                "max_gain_15min": max_gain_15min,
                "max_dd_15min": max_dd_15min,
                "max_gain_full_day": max_gain_full_day,
                "time_to_target": time_to_target,
            })

    df = pd.DataFrame(records)
    total = len(df)
    winners = df[df["hit_target"]]
    losers = df[~df["hit_target"]]
    base_wr = len(winners) / total * 100

    print("=" * 70)
    print(f"  POST-ENTRY PATTERN ANALYSIS: What predicts +{TARGET_FROM_ENTRY}% from entry?")
    print(f"  Sample: {total} stocks (Gap>{MIN_GAP_PCT}%, 1st candle green >{GREEN_CANDLE_MIN_BODY_PCT}%)")
    print(f"  Base hit rate: {len(winners)}/{total} ({base_wr:.1f}%)")
    print("=" * 70)

    # ── 1. FIRST CANDLE BODY SIZE ──
    print(f"\n{'─'*70}")
    print("  1. FIRST CANDLE BODY SIZE (% from open)")
    print(f"{'─'*70}")
    for lo, hi, label in [(2, 3, "2-3%"), (3, 5, "3-5%"), (5, 8, "5-8%"),
                           (8, 12, "8-12%"), (12, 20, "12-20%"), (20, 999, ">20%")]:
        sub = df[(df["fc_body_pct"] >= lo) & (df["fc_body_pct"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["hit_target"].mean() * 100
        avg_gain = sub["max_gain_15min"].mean()
        print(f"  Body {label:>6}: n={len(sub):>4}, hit +5%: {wr:>5.1f}%, "
              f"avg max gain 15m: {avg_gain:>5.1f}%")

    # ── 2. FIRST CANDLE UPPER WICK ──
    print(f"\n{'─'*70}")
    print("  2. FIRST CANDLE UPPER WICK (rejection signal)")
    print(f"{'─'*70}")
    for lo, hi, label in [(0, 0.5, "<0.5%"), (0.5, 1, "0.5-1%"), (1, 2, "1-2%"),
                           (2, 4, "2-4%"), (4, 999, ">4%")]:
        sub = df[(df["fc_upper_wick_pct"] >= lo) & (df["fc_upper_wick_pct"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["hit_target"].mean() * 100
        print(f"  Wick {label:>6}: n={len(sub):>4}, hit +5%: {wr:>5.1f}%")

    # ── 3. FIRST CANDLE CLOSE VS HIGH (strength indicator) ──
    print(f"\n{'─'*70}")
    print("  3. FIRST CANDLE CLOSE VS RANGE (0=low, 1=high)")
    print(f"{'─'*70}")
    for lo, hi, label in [(0, 0.5, "<0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 0.85, "0.7-0.85"),
                           (0.85, 0.95, "0.85-0.95"), (0.95, 1.01, "0.95-1.0")]:
        sub = df[(df["fc_close_vs_high"] >= lo) & (df["fc_close_vs_high"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["hit_target"].mean() * 100
        print(f"  Close pos {label:>9}: n={len(sub):>4}, hit +5%: {wr:>5.1f}%")

    # ── 4. SECOND CANDLE DIRECTION ──
    print(f"\n{'─'*70}")
    print("  4. SECOND CANDLE (continuation signal)")
    print(f"{'─'*70}")
    green2 = df[df["sc_green"]]
    red2 = df[~df["sc_green"]]
    print(f"  2nd candle GREEN: n={len(green2):>4}, hit +5%: {green2['hit_target'].mean()*100:>5.1f}%")
    print(f"  2nd candle RED:   n={len(red2):>4}, hit +5%: {red2['hit_target'].mean()*100:>5.1f}%")

    # 2nd candle made new high
    new_hi = df[df["sc_made_new_high"]]
    no_hi = df[~df["sc_made_new_high"]]
    print(f"\n  2nd candle NEW HIGH: n={len(new_hi):>4}, hit +5%: {new_hi['hit_target'].mean()*100:>5.1f}%")
    print(f"  2nd candle NO high:  n={len(no_hi):>4}, hit +5%: {no_hi['hit_target'].mean()*100:>5.1f}%")

    # 2nd candle continuation %
    print(f"\n  2nd candle change from 1st close:")
    for lo, hi, label in [(-999, -2, "<-2%"), (-2, 0, "-2 to 0%"), (0, 2, "0-2%"),
                           (2, 5, "+2-5%"), (5, 999, ">+5%")]:
        sub = df[(df["sc_vs_fc"] >= lo) & (df["sc_vs_fc"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["hit_target"].mean() * 100
        print(f"    {label:>10}: n={len(sub):>4}, hit +5%: {wr:>5.1f}%")

    # ── 5. GAP SIZE ──
    print(f"\n{'─'*70}")
    print("  5. GAP SIZE")
    print(f"{'─'*70}")
    for lo, hi, label in [(10, 15, "10-15%"), (15, 20, "15-20%"), (20, 30, "20-30%"),
                           (30, 50, "30-50%"), (50, 100, "50-100%"), (100, 999, ">100%")]:
        sub = df[(df["gap_pct"] >= lo) & (df["gap_pct"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["hit_target"].mean() * 100
        avg_gain = sub["max_gain_15min"].mean()
        print(f"  Gap {label:>8}: n={len(sub):>4}, hit +5%: {wr:>5.1f}%, avg max gain 15m: {avg_gain:>5.1f}%")

    # ── 6. PRICE LEVEL ──
    print(f"\n{'─'*70}")
    print("  6. STOCK PRICE")
    print(f"{'─'*70}")
    for lo, hi, label in [(0, 3, "<$3"), (3, 5, "$3-5"), (5, 10, "$5-10"),
                           (10, 20, "$10-20"), (20, 50, "$20-50"), (50, 999, ">$50")]:
        sub = df[(df["price"] >= lo) & (df["price"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["hit_target"].mean() * 100
        print(f"  Price {label:>6}: n={len(sub):>4}, hit +5%: {wr:>5.1f}%")

    # ── 7. PREMARKET LAST 30 MIN TREND ──
    print(f"\n{'─'*70}")
    print("  7. PREMARKET LAST 30 MIN (9:00-9:30) TREND")
    print(f"{'─'*70}")
    for lo, hi, label in [(-999, -3, "Down >3%"), (-3, -1, "Down 1-3%"),
                           (-1, 1, "Flat"), (1, 3, "Up 1-3%"), (3, 999, "Up >3%")]:
        sub = df[(df["pm_last30_trend"] >= lo) & (df["pm_last30_trend"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["hit_target"].mean() * 100
        print(f"  PM30 {label:>12}: n={len(sub):>4}, hit +5%: {wr:>5.1f}%")

    # ── 8. PM FULL MOMENTUM (7am-9:30) ──
    print(f"\n{'─'*70}")
    print("  8. PREMARKET FULL MOMENTUM (7AM-9:30)")
    print(f"{'─'*70}")
    for lo, hi, label in [(-999, -5, "Down >5%"), (-5, -2, "Down 2-5%"),
                           (-2, 0, "Down 0-2%"), (0, 2, "Up 0-2%"),
                           (2, 5, "Up 2-5%"), (5, 999, "Up >5%")]:
        sub = df[(df["pm_momentum_7am"] >= lo) & (df["pm_momentum_7am"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["hit_target"].mean() * 100
        print(f"  PM7am {label:>12}: n={len(sub):>4}, hit +5%: {wr:>5.1f}%")

    # ── 9. COMBINED FILTERS (find the best combos) ──
    print(f"\n{'─'*70}")
    print("  9. COMBINED FILTERS — FINDING THE BEST EDGE")
    print(f"{'─'*70}")

    combos = []

    # Test many combinations
    for body_min in [2, 3, 5, 8]:
        for wick_max in [1, 2, 4, 999]:
            for sc_green_req in [True, False]:
                for sc_new_hi_req in [True, False]:
                    for gap_min in [10, 20, 30]:
                        mask = (
                            (df["fc_body_pct"] >= body_min) &
                            (df["fc_upper_wick_pct"] <= wick_max) &
                            (df["gap_pct"] >= gap_min)
                        )
                        if sc_green_req:
                            mask = mask & df["sc_green"]
                        if sc_new_hi_req:
                            mask = mask & df["sc_made_new_high"]

                        sub = df[mask]
                        if len(sub) < 15:  # need statistical significance
                            continue
                        wr = sub["hit_target"].mean() * 100
                        avg_gain = sub["max_gain_15min"].mean()
                        avg_dd = sub["max_dd_15min"].mean()
                        combos.append({
                            "filter": (f"body>={body_min}% wick<={wick_max}% gap>={gap_min}%"
                                      f"{' +2ndGreen' if sc_green_req else ''}"
                                      f"{' +2ndNewHi' if sc_new_hi_req else ''}"),
                            "n": len(sub),
                            "wr": wr,
                            "avg_gain_15m": avg_gain,
                            "avg_dd_15m": avg_dd,
                            "edge": wr - base_wr,
                        })

    # Sort by win rate (descending), min n=15
    combos.sort(key=lambda x: -x["wr"])
    print(f"\n  Top 20 filters by hit rate (base: {base_wr:.1f}%):\n")
    print(f"  {'Filter':<60} {'n':>4} {'WR%':>6} {'Edge':>6} {'AvgGain':>8} {'AvgDD':>6}")
    print(f"  {'─'*60} {'─'*4} {'─'*6} {'─'*6} {'─'*8} {'─'*6}")
    for c in combos[:20]:
        print(f"  {c['filter']:<60} {c['n']:>4} {c['wr']:>5.1f}% {c['edge']:>+5.1f}% "
              f"{c['avg_gain_15m']:>7.1f}% {c['avg_dd_15m']:>5.1f}%")

    # ── 10. SECOND CANDLE WAIT STRATEGY ──
    print(f"\n{'─'*70}")
    print("  10. WAIT FOR 2nd CANDLE CONFIRMATION (enter at 2nd candle close)")
    print(f"{'─'*70}")
    print("  If we wait for 2nd candle green + new high, then enter at 2nd close:")

    confirmed = df[df["sc_green"] & df["sc_made_new_high"]].copy()
    if len(confirmed) > 0:
        # Recalculate outcome from 2nd candle close entry
        # We already have sc_vs_fc which tells us 2nd close vs 1st close
        # New entry would be ~sc_vs_fc% higher, so effective target from new entry
        # is reduced
        print(f"    Sample: {len(confirmed)} stocks")
        print(f"    Hit +5% from 1st close: {confirmed['hit_target'].mean()*100:.1f}%")
        print(f"    Avg 2nd candle gain: {confirmed['sc_vs_fc'].mean():.1f}%")
        print(f"    Avg max gain 15m: {confirmed['max_gain_15min'].mean():.1f}%")
        print(f"    Avg max DD 15m: {confirmed['max_dd_15min'].mean():.1f}%")

    # For these, what % hit +5% from 2nd candle close (remaining upside)?
    # Approximate: max_gain_full_day - sc_vs_fc > 5%
    if len(confirmed) > 0:
        remaining_gain = confirmed["max_gain_full_day"] - confirmed["sc_vs_fc"]
        pct_5_from_2nd = (remaining_gain >= 5.0).mean() * 100
        pct_3_from_2nd = (remaining_gain >= 3.0).mean() * 100
        print(f"    Hit +5% from 2nd close (est): {pct_5_from_2nd:.1f}%")
        print(f"    Hit +3% from 2nd close (est): {pct_3_from_2nd:.1f}%")

    # ── 11. BODY + LOW WICK (BEST COMBO DEEP DIVE) ──
    print(f"\n{'─'*70}")
    print("  11. DEEP DIVE: Body size + close position combos")
    print(f"{'─'*70}")
    for body_min in [2, 3, 5, 8]:
        for close_pos_min in [0.5, 0.7, 0.85, 0.95]:
            sub = df[(df["fc_body_pct"] >= body_min) & (df["fc_close_vs_high"] >= close_pos_min)]
            if len(sub) < 10:
                continue
            wr = sub["hit_target"].mean() * 100
            edge = wr - base_wr
            if edge > 3:  # Only show meaningful edges
                print(f"  Body>={body_min}% + ClosePos>={close_pos_min}: "
                      f"n={len(sub):>4}, WR={wr:.1f}% (edge: {edge:+.1f}%)")

    # ── 12. FIRST CANDLE VOLUME ──
    print(f"\n{'─'*70}")
    print("  12. FIRST CANDLE VOLUME")
    print(f"{'─'*70}")
    vol_valid = df[df["fc_volume"] > 0]
    if len(vol_valid) > 0:
        for pctile, label in [(25, "Bottom 25%"), (50, "25-50%"), (75, "50-75%"), (100, "Top 25%")]:
            lo_pct = pctile - 25
            lo_val = np.percentile(vol_valid["fc_volume"], lo_pct)
            hi_val = np.percentile(vol_valid["fc_volume"], pctile)
            sub = vol_valid[(vol_valid["fc_volume"] >= lo_val) & (vol_valid["fc_volume"] <= hi_val)]
            if len(sub) == 0:
                continue
            wr = sub["hit_target"].mean() * 100
            print(f"  Vol {label:>12} ({lo_val/1000:.0f}K-{hi_val/1000:.0f}K): "
                  f"n={len(sub):>4}, hit +5%: {wr:.1f}%")

    # ── SUMMARY ──
    print(f"\n{'='*70}")
    print("  SUMMARY: KEY FINDINGS")
    print(f"{'='*70}")
    print(f"\n  Base rate: {base_wr:.1f}% hit +5% from entry within full day")
    print(f"  (within 15 min: {(df['max_gain_15min'] >= 5.0).mean()*100:.1f}%)")

    if combos:
        best = combos[0]
        print(f"\n  Best filter: {best['filter']}")
        print(f"    n={best['n']}, WR={best['wr']:.1f}%, edge=+{best['edge']:.1f}%")
        print(f"    Avg max gain in 15 min: {best['avg_gain_15m']:.1f}%")
        print(f"    Avg max drawdown in 15 min: {best['avg_dd_15m']:.1f}%")

    print()


if __name__ == "__main__":
    data_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["stored_data_oos", "stored_data"]
    analyze(data_dirs)
