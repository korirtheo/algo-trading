"""
Stop Loss Analysis for First Green Candle Strategy
====================================================
For each stock matching Gap>10% + first candle green >2%:
  - Track the MAX DRAWDOWN from entry before hitting +X% target
  - This tells us the minimum stop width needed to capture those winners
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from collections import defaultdict

import io, sys as _sys
_sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")

ET_TZ = ZoneInfo("America/New_York")

# --- Config ---
MIN_GAP_PCT = 10.0
GREEN_CANDLE_MIN_BODY_PCT = 2.0
TARGET_THRESHOLDS = [3.0, 5.0, 7.0, 10.0]  # Multiple targets to analyze

# Date filter: Oct 2025 - Feb 2026
DATE_START = "2025-10-01"
DATE_END = "2026-02-28"


def load_picks(data_dirs):
    merged = {}
    for d in data_dirs:
        for cache_name in ["fulltest_picks_gap2_vol250k.pkl"]:
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


def analyze_drawdowns(data_dirs):
    print("Loading data...")
    picks_by_date = load_picks(data_dirs)

    # Filter to date range
    dates = sorted(d for d in picks_by_date.keys() if DATE_START <= d <= DATE_END)
    print(f"  {len(dates)} trading days: {dates[0]} to {dates[-1]}")

    # Collect per-stock drawdown data
    records = []  # Each: {ticker, date, gap, body_pct, entry_price, open_price,
    #                       max_dd_from_entry, max_dd_from_open, hit_targets, time_to_target}

    total_picks = 0
    qualifying = 0

    for date_str in dates:
        day_picks = picks_by_date.get(date_str, [])
        for pick in day_picks:
            if pick["gap_pct"] < MIN_GAP_PCT:
                continue
            total_picks += 1

            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) == 0:
                continue

            timestamps = sorted(mh.index)
            if not timestamps:
                continue

            # First candle
            first_ts = timestamps[0]
            fc = mh.loc[first_ts]
            fc_open = float(fc["Open"])
            fc_close = float(fc["Close"])
            fc_high = float(fc["High"])
            fc_low = float(fc["Low"])

            if fc_open <= 0:
                continue

            body_pct = (fc_close - fc_open) / fc_open * 100
            if body_pct < GREEN_CANDLE_MIN_BODY_PCT:
                continue

            qualifying += 1

            # Entry = first candle close (with 0.1% slippage)
            entry_price = fc_close * 1.001
            open_price = fc_open

            # Track drawdown and target hits through remaining candles
            max_dd_from_entry = 0.0  # Max % drop from entry before recovery
            max_dd_from_open = 0.0
            min_price_seen = entry_price
            max_price_from_open = open_price

            hit_targets = {}  # target% -> {hit: bool, dd_before: float, time_candles: int}
            targets_hit = set()

            # Also track drawdown specifically BEFORE each target is hit
            dd_before_target = {t: 0.0 for t in TARGET_THRESHOLDS}

            for i, ts in enumerate(timestamps[1:], 1):  # Skip first candle (already in position)
                candle = mh.loc[ts]
                c_high = float(candle["High"])
                c_low = float(candle["Low"])
                c_close = float(candle["Close"])

                # Update drawdown from entry
                if c_low < min_price_seen:
                    min_price_seen = c_low
                    dd_pct = (entry_price - c_low) / entry_price * 100
                    max_dd_from_entry = max(max_dd_from_entry, dd_pct)

                # Check target hits (using high - intracandle)
                for target in TARGET_THRESHOLDS:
                    if target in targets_hit:
                        continue
                    target_price_from_entry = entry_price * (1 + target / 100)
                    target_price_from_open = open_price * (1 + target / 100)

                    if c_high >= target_price_from_entry:
                        targets_hit.add(target)
                        hit_targets[target] = {
                            "hit": True,
                            "dd_before": max_dd_from_entry,  # Max DD before this target was hit
                            "candles_to_hit": i,
                            "minutes_to_hit": i * 2,  # 2-min candles
                        }
                        # Record the drawdown that occurred before this target
                        dd_before_target[target] = max_dd_from_entry

                # Update dd_before for unhit targets
                for target in TARGET_THRESHOLDS:
                    if target not in targets_hit:
                        dd_before_target[target] = max_dd_from_entry

            # For targets not hit, record as not hit
            for target in TARGET_THRESHOLDS:
                if target not in hit_targets:
                    hit_targets[target] = {
                        "hit": False,
                        "dd_before": max_dd_from_entry,
                        "candles_to_hit": None,
                        "minutes_to_hit": None,
                    }

            records.append({
                "ticker": pick["ticker"],
                "date": date_str,
                "gap_pct": pick["gap_pct"],
                "body_pct": body_pct,
                "entry_price": entry_price,
                "open_price": open_price,
                "max_dd_from_entry": max_dd_from_entry,
                "hit_targets": hit_targets,
            })

    print(f"\n  Total gap>10% picks: {total_picks}")
    print(f"  Qualifying (green 1st candle >2%): {qualifying}")
    print(f"  Records analyzed: {len(records)}")

    # --- ANALYSIS ---
    print("\n" + "=" * 70)
    print("  STOP LOSS ANALYSIS: How wide must the stop be?")
    print("=" * 70)

    # For each target, show the distribution of max drawdown BEFORE hitting that target
    for target in TARGET_THRESHOLDS:
        print(f"\n{'─' * 70}")
        print(f"  TARGET: +{target}% from entry")
        print(f"{'─' * 70}")

        winners = [r for r in records if r["hit_targets"][target]["hit"]]
        losers = [r for r in records if not r["hit_targets"][target]["hit"]]
        hit_rate = len(winners) / len(records) * 100 if records else 0

        print(f"  Hit rate: {len(winners)}/{len(records)} ({hit_rate:.1f}%)")

        if not winners:
            print("  No winners to analyze.")
            continue

        # Drawdown distribution for WINNERS (before they hit target)
        winner_dds = [r["hit_targets"][target]["dd_before"] for r in winners]
        winner_times = [r["hit_targets"][target]["minutes_to_hit"] for r in winners]

        print(f"\n  Drawdown BEFORE hitting +{target}% (winners only):")
        print(f"    Mean:   {np.mean(winner_dds):.2f}%")
        print(f"    Median: {np.median(winner_dds):.2f}%")
        print(f"    P75:    {np.percentile(winner_dds, 75):.2f}%")
        print(f"    P90:    {np.percentile(winner_dds, 90):.2f}%")
        print(f"    P95:    {np.percentile(winner_dds, 95):.2f}%")
        print(f"    Max:    {np.max(winner_dds):.2f}%")

        print(f"\n  Time to hit +{target}% (winners only):")
        print(f"    Mean:   {np.mean(winner_times):.0f} min")
        print(f"    Median: {np.median(winner_times):.0f} min")
        print(f"    P90:    {np.percentile(winner_times, 90):.0f} min")

        # KEY TABLE: For each stop width, how many winners would we keep?
        print(f"\n  Stop Width vs Winners Captured (of {len(winners)} winners):")
        print(f"  {'Stop %':>8}  {'Kept':>6}  {'Lost':>6}  {'Capture%':>10}  {'Stopped Losers':>15}  {'Est WR':>8}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*15}  {'─'*8}")

        for stop in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
            # Winners kept = those whose DD before target < stop
            kept = sum(1 for dd in winner_dds if dd < stop)
            lost = len(winners) - kept

            # Losers stopped = those who never hit target AND DD exceeded stop
            loser_dds = [r["max_dd_from_entry"] for r in losers]
            stopped_losers = sum(1 for dd in loser_dds if dd >= stop)
            # Losers who survive stop = will exit EOD (partial loss or recovery)
            surviving_losers = len(losers) - stopped_losers

            total_trades = kept + (len(losers))  # All losers still enter
            est_wr = kept / total_trades * 100 if total_trades > 0 else 0

            # Expected value per $1000 position:
            # Winners: +target% * kept
            # Stopped losers: -stop% * stopped_losers
            # Surviving losers (EOD): estimate avg loss from their max DD (rough)
            avg_loser_eod_loss = np.mean([min(dd, stop) for dd in loser_dds]) if loser_dds else 0

            ev_win = kept * (target / 100) * 1000
            ev_stop_loss = stopped_losers * (stop / 100) * 1000
            ev_surviving = surviving_losers * (avg_loser_eod_loss / 100) * 1000
            net_ev = ev_win - ev_stop_loss - ev_surviving
            ev_per_trade = net_ev / total_trades if total_trades > 0 else 0

            print(f"  {stop:>7.1f}%  {kept:>6}  {lost:>6}  {kept/len(winners)*100:>9.1f}%  {stopped_losers:>15}  {est_wr:>7.1f}%")

        # Best stop: maximize (capture_rate - false_stop_rate)
        print(f"\n  Optimal stop analysis (maximizing expected value):")
        best_stop = None
        best_ev = -9999
        for stop_x10 in range(10, 201, 5):  # 1.0% to 20.0% in 0.5% steps
            stop = stop_x10 / 10
            kept = sum(1 for dd in winner_dds if dd < stop)
            loser_dds = [r["max_dd_from_entry"] for r in losers]
            stopped_losers = sum(1 for dd in loser_dds if dd >= stop)
            surviving_losers = len(losers) - stopped_losers

            total_trades = kept + len(losers)
            ev_win = kept * target
            ev_stop = stopped_losers * stop
            # Rough: surviving losers lose half the stop on average
            ev_surviving = surviving_losers * (stop * 0.5)
            net = ev_win - ev_stop - ev_surviving
            ev_per = net / total_trades if total_trades > 0 else 0

            if ev_per > best_ev:
                best_ev = ev_per
                best_stop = stop

        if best_stop:
            # Recalculate details for best stop
            kept = sum(1 for dd in winner_dds if dd < best_stop)
            loser_dds = [r["max_dd_from_entry"] for r in losers]
            stopped = sum(1 for dd in loser_dds if dd >= best_stop)
            capture = kept / len(winners) * 100
            print(f"    Best stop: {best_stop:.1f}%")
            print(f"    Captures: {kept}/{len(winners)} winners ({capture:.1f}%)")
            print(f"    Stops: {stopped}/{len(losers)} losers")
            print(f"    Est EV/trade: {best_ev:.2f}% per position")

    # --- COMBINED RECOMMENDATION ---
    print(f"\n{'=' * 70}")
    print("  RECOMMENDATION")
    print(f"{'=' * 70}")

    # Focus on +5% from entry (the original analysis target minus the 2% entry gap)
    target_focus = 3.0
    winners_3 = [r for r in records if r["hit_targets"][target_focus]["hit"]]
    if winners_3:
        dds = [r["hit_targets"][target_focus]["dd_before"] for r in winners_3]
        p90 = np.percentile(dds, 90)
        p95 = np.percentile(dds, 95)
        print(f"\n  For +{target_focus}% target:")
        print(f"    90% of winners dip less than {p90:.1f}% before hitting target")
        print(f"    95% of winners dip less than {p95:.1f}% before hitting target")
        print(f"    → Suggested stop: {p90:.1f}% (captures 90% of winners)")
        print(f"    → Conservative:   {p95:.1f}% (captures 95% of winners)")

    target_focus = 5.0
    winners_5 = [r for r in records if r["hit_targets"][target_focus]["hit"]]
    if winners_5:
        dds = [r["hit_targets"][target_focus]["dd_before"] for r in winners_5]
        p90 = np.percentile(dds, 90)
        p95 = np.percentile(dds, 95)
        print(f"\n  For +{target_focus}% target:")
        print(f"    90% of winners dip less than {p90:.1f}% before hitting target")
        print(f"    95% of winners dip less than {p95:.1f}% before hitting target")
        print(f"    → Suggested stop: {p90:.1f}% (captures 90% of winners)")
        print(f"    → Conservative:   {p95:.1f}% (captures 95% of winners)")

    # --- SCATTER: DD vs gap size ---
    print(f"\n{'─' * 70}")
    print("  DRAWDOWN BY GAP SIZE (for +5% target winners)")
    print(f"{'─' * 70}")
    if winners_5:
        for gap_lo, gap_hi in [(10, 15), (15, 20), (20, 30), (30, 50), (50, 999)]:
            subset = [r for r in winners_5 if gap_lo <= r["gap_pct"] < gap_hi]
            if not subset:
                continue
            dds = [r["hit_targets"][5.0]["dd_before"] for r in subset]
            print(f"  Gap {gap_lo}-{gap_hi}%: n={len(subset):>3}, "
                  f"median DD={np.median(dds):.1f}%, "
                  f"P90 DD={np.percentile(dds, 90):.1f}%, "
                  f"max DD={np.max(dds):.1f}%")

    print()


if __name__ == "__main__":
    data_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["stored_data_oos", "stored_data"]
    analyze_drawdowns(data_dirs)
