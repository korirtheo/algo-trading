"""
Analyze time-to-target for Strategy A and Strategy B green-candle trades.
Tracks how long winners take to hit targets, and what happens to losers over time.
"""
import io, sys as _sys; _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from test_full import load_all_picks, ET_TZ

# ── Strategy configs (matching test_green_candle_combined.py) ──
A_MIN_GAP_PCT = 20.0
A_MIN_BODY_PCT = 2.0
A_REQUIRE_2ND_GREEN = True
A_REQUIRE_2ND_NEW_HIGH = True
A_TARGET_PCT = 3.0
A_TIME_LIMIT_MINUTES = 5

B_MIN_GAP_PCT = 10.0
B_MIN_BODY_PCT = 8.0
B_REQUIRE_2ND_GREEN = True
B_REQUIRE_2ND_NEW_HIGH = False
B_TARGET_PCT = 7.0
B_TIME_LIMIT_MINUTES = 15

SLIPPAGE_PCT = 0.15

# Time marks to report on (minutes after entry)
TIME_MARKS = [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 60]


def classify_stock(gap_pct, body_pct, second_green, second_new_high):
    """B priority, then A, else None."""
    qualifies_b = (
        gap_pct >= B_MIN_GAP_PCT
        and body_pct >= B_MIN_BODY_PCT
        and (not B_REQUIRE_2ND_GREEN or second_green)
        and (not B_REQUIRE_2ND_NEW_HIGH or second_new_high)
    )
    qualifies_a = (
        gap_pct >= A_MIN_GAP_PCT
        and body_pct >= A_MIN_BODY_PCT
        and (not A_REQUIRE_2ND_GREEN or second_green)
        and (not A_REQUIRE_2ND_NEW_HIGH or second_new_high)
    )
    if qualifies_b:
        return "B"
    if qualifies_a:
        return "A"
    return None


def analyze_picks(all_dates, daily_picks):
    """For every qualifying stock, track candle-by-candle P&L after entry."""
    records = []  # one per qualifying trade

    for date_str in all_dates:
        picks = daily_picks.get(date_str, [])
        for pick in picks:
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) < 2:
                continue

            gap_pct = pick["gap_pct"]

            # Convert index to ET
            if mh.index.tz is not None:
                et_index = mh.index.tz_convert(ET_TZ)
            else:
                et_index = mh.index.tz_localize("UTC").tz_convert(ET_TZ)

            # ── Candle 1 check ──
            c1 = mh.iloc[0]
            c1_open = float(c1["Open"])
            c1_close = float(c1["Close"])
            c1_high = float(c1["High"])
            if c1_open <= 0:
                continue
            body_pct = (c1_close / c1_open - 1) * 100

            # Must pass at least one strategy's min body
            min_body = min(A_MIN_BODY_PCT, B_MIN_BODY_PCT)
            if body_pct < min_body:
                continue

            # ── Candle 2 check ──
            c2 = mh.iloc[1]
            c2_open = float(c2["Open"])
            c2_close = float(c2["Close"])
            c2_high = float(c2["High"])
            second_green = c2_close > c2_open
            second_new_high = c2_high > c1_high

            strategy = classify_stock(gap_pct, body_pct, second_green, second_new_high)
            if strategy is None:
                continue

            # ── Entry: at close of candle 2 + slippage ──
            signal_price = c2_close
            if signal_price <= 0:
                continue
            entry_price = signal_price * (1 + SLIPPAGE_PCT / 100)
            entry_ts = et_index[1]

            target_pct = A_TARGET_PCT if strategy == "A" else B_TARGET_PCT
            time_limit = A_TIME_LIMIT_MINUTES if strategy == "A" else B_TIME_LIMIT_MINUTES
            target_price = entry_price * (1 + target_pct / 100)

            # ── Track every candle after entry ──
            # Candles from index 2 onward (after the entry candle)
            post_entry = mh.iloc[2:]
            post_et = et_index[2:]

            if len(post_entry) == 0:
                continue

            candle_pnls = []       # (minutes_after_entry, pnl_pct, close_price)
            hit_target = False
            target_minutes = None
            max_pnl = -999.0
            max_pnl_minutes = 0

            for j in range(len(post_entry)):
                candle = post_entry.iloc[j]
                ts_et = post_et[j]
                c_high = float(candle["High"])
                c_close = float(candle["Close"])

                # Minutes since entry (candles are 2-min)
                minutes_elapsed = (ts_et.hour * 60 + ts_et.minute) - (entry_ts.hour * 60 + entry_ts.minute)
                if minutes_elapsed < 0:
                    continue

                pnl_pct = (c_close / entry_price - 1) * 100

                candle_pnls.append((minutes_elapsed, pnl_pct, c_close, c_high))

                if pnl_pct > max_pnl:
                    max_pnl = pnl_pct
                    max_pnl_minutes = minutes_elapsed

                # Check target hit on HIGH (could hit intracandle)
                if not hit_target and c_high >= target_price:
                    hit_target = True
                    target_minutes = minutes_elapsed

            if len(candle_pnls) == 0:
                continue

            records.append({
                "date": date_str,
                "ticker": pick["ticker"],
                "strategy": strategy,
                "gap_pct": gap_pct,
                "body_pct": body_pct,
                "entry_price": entry_price,
                "target_pct": target_pct,
                "time_limit": time_limit,
                "hit_target": hit_target,
                "target_minutes": target_minutes,
                "candle_pnls": candle_pnls,    # list of (minutes, pnl%, close, high)
                "max_pnl": max_pnl,
                "max_pnl_minutes": max_pnl_minutes,
            })

    return records


def get_pnl_at_time(candle_pnls, target_minutes):
    """Get the P&L at or just after a specific minute mark.
    Returns the pnl% of the first candle at or after target_minutes,
    or the last available candle if trade ended before that time."""
    for minutes, pnl, close, high in candle_pnls:
        if minutes >= target_minutes:
            return pnl
    # Trade ended before this time mark - return last known
    if candle_pnls:
        return candle_pnls[-1][1]
    return None


def check_target_hit_by_time(candle_pnls, target_pct, target_minutes):
    """Check if target was hit by high at or before target_minutes."""
    for minutes, pnl, close, high in candle_pnls:
        if minutes > target_minutes:
            return False
        # pnl is based on close; we need to check high vs entry
        # We stored high in the tuple, so compute high-based return
        # Actually we need entry_price... let's use the ratio approach
        # Since pnl = (close/entry - 1)*100, and we have high,
        # high_pnl = (high/entry - 1)*100 = pnl + (high - close)/entry * 100
        # Simpler: just check if high >= entry * (1 + target_pct/100)
        # But we don't have entry here. Use a flag-based approach instead.
        pass
    return False


def report_strategy(records, strategy_name):
    """Print full analysis for one strategy."""
    strat = [r for r in records if r["strategy"] == strategy_name]
    if not strat:
        print(f"\n  No trades found for Strategy {strategy_name}")
        return

    target_pct = A_TARGET_PCT if strategy_name == "A" else B_TARGET_PCT
    time_limit = A_TIME_LIMIT_MINUTES if strategy_name == "A" else B_TIME_LIMIT_MINUTES

    winners = [r for r in strat if r["hit_target"]]
    losers = [r for r in strat if not r["hit_target"]]

    # Also define "time-stop losers": those that never hit target within the time limit
    time_stop_losers = [r for r in strat if not r["hit_target"]]

    print(f"\n{'='*80}")
    print(f"  STRATEGY {strategy_name}: {'Quick Scalp' if strategy_name == 'A' else 'Big Body'}")
    print(f"  Gap >= {A_MIN_GAP_PCT if strategy_name == 'A' else B_MIN_GAP_PCT}% | "
          f"Body >= {A_MIN_BODY_PCT if strategy_name == 'A' else B_MIN_BODY_PCT}% | "
          f"Target: +{target_pct}% | Time Stop: {time_limit} min")
    print(f"{'='*80}")
    print(f"\n  Total qualifying trades: {len(strat)}")
    print(f"  Winners (hit +{target_pct}%):   {len(winners)} ({len(winners)/len(strat)*100:.1f}%)")
    print(f"  Losers (never hit target): {len(losers)} ({len(losers)/len(strat)*100:.1f}%)")

    # ── 1. Time-to-target distribution (winners only) ──
    if winners:
        ttts = [r["target_minutes"] for r in winners]
        ttts_arr = np.array(ttts)
        print(f"\n  {'─'*60}")
        print(f"  TIME TO TARGET (winners only, N={len(winners)})")
        print(f"  {'─'*60}")
        print(f"    Mean:   {np.mean(ttts_arr):.1f} min")
        print(f"    Median: {np.median(ttts_arr):.1f} min")
        print(f"    P25:    {np.percentile(ttts_arr, 25):.1f} min")
        print(f"    P75:    {np.percentile(ttts_arr, 75):.1f} min")
        print(f"    P90:    {np.percentile(ttts_arr, 90):.1f} min")
        print(f"    Min:    {np.min(ttts_arr):.0f} min")
        print(f"    Max:    {np.max(ttts_arr):.0f} min")

        # Histogram buckets
        bins = [0, 2, 4, 6, 8, 10, 15, 20, 30, 60, 120, 9999]
        labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-20", "20-30", "30-60", "60-120", "120+"]
        hist, _ = np.histogram(ttts_arr, bins=bins)
        print(f"\n    Distribution:")
        for label, count in zip(labels, hist):
            if count > 0:
                bar = "#" * int(count / max(hist) * 30) if max(hist) > 0 else ""
                print(f"      {label:>8} min: {count:>4} ({count/len(winners)*100:>5.1f}%) {bar}")

    # ── 2. P&L by hold time (ALL positions) ──
    print(f"\n  {'─'*60}")
    print(f"  P&L BY HOLD TIME (all {len(strat)} positions)")
    print(f"  {'─'*60}")
    print(f"  {'Time':>6} | {'Avg P&L%':>9} | {'Med P&L%':>9} | {'Hit Tgt%':>9} | {'Profitable%':>12} | {'N':>4}")
    print(f"  {'─'*6}-+-{'─'*9}-+-{'─'*9}-+-{'─'*9}-+-{'─'*12}-+-{'─'*4}")

    for t in TIME_MARKS:
        pnls_at_t = []
        targets_hit_by_t = 0
        profitable_at_t = 0

        for r in strat:
            pnl = get_pnl_at_time(r["candle_pnls"], t)
            if pnl is not None:
                # If target was already hit, use target P&L
                if r["hit_target"] and r["target_minutes"] is not None and r["target_minutes"] <= t:
                    pnl_use = r["target_pct"] - SLIPPAGE_PCT  # target minus exit slippage
                    targets_hit_by_t += 1
                    profitable_at_t += 1
                else:
                    pnl_use = pnl
                    if pnl > 0:
                        profitable_at_t += 1
                pnls_at_t.append(pnl_use)

        if pnls_at_t:
            avg_pnl = np.mean(pnls_at_t)
            med_pnl = np.median(pnls_at_t)
            pct_target = targets_hit_by_t / len(pnls_at_t) * 100
            pct_profitable = profitable_at_t / len(pnls_at_t) * 100
            print(f"  {t:>4}m | {avg_pnl:>+8.2f}% | {med_pnl:>+8.2f}% | {pct_target:>8.1f}% | {pct_profitable:>11.1f}% | {len(pnls_at_t):>4}")

    # ── 3. Loser analysis ──
    if losers:
        print(f"\n  {'─'*60}")
        print(f"  LOSER ANALYSIS (N={len(losers)}, never hit +{target_pct}%)")
        print(f"  {'─'*60}")
        print(f"\n  P&L at each time mark for LOSERS only:")
        print(f"  {'Time':>6} | {'Avg P&L%':>9} | {'Med P&L%':>9} | {'Profitable%':>12} | {'N':>4}")
        print(f"  {'─'*6}-+-{'─'*9}-+-{'─'*9}-+-{'─'*12}-+-{'─'*4}")

        best_avg_pnl = -999
        best_time = 0

        for t in TIME_MARKS:
            loser_pnls = []
            profitable = 0
            for r in losers:
                pnl = get_pnl_at_time(r["candle_pnls"], t)
                if pnl is not None:
                    loser_pnls.append(pnl)
                    if pnl > 0:
                        profitable += 1
            if loser_pnls:
                avg = np.mean(loser_pnls)
                med = np.median(loser_pnls)
                pct_prof = profitable / len(loser_pnls) * 100
                print(f"  {t:>4}m | {avg:>+8.2f}% | {med:>+8.2f}% | {pct_prof:>11.1f}% | {len(loser_pnls):>4}")
                if avg > best_avg_pnl:
                    best_avg_pnl = avg
                    best_time = t

        # When do losers peak?
        peak_times = [r["max_pnl_minutes"] for r in losers]
        peak_pnls = [r["max_pnl"] for r in losers]
        print(f"\n  Loser Peak P&L Statistics:")
        print(f"    Avg time of max P&L:  {np.mean(peak_times):.1f} min")
        print(f"    Median time of max:   {np.median(peak_times):.1f} min")
        print(f"    Avg max P&L reached:  {np.mean(peak_pnls):+.2f}%")
        print(f"    Median max P&L:       {np.median(peak_pnls):+.2f}%")

        # What % of losers were ever profitable?
        ever_profitable = sum(1 for r in losers if r["max_pnl"] > 0)
        print(f"    Losers ever profitable: {ever_profitable}/{len(losers)} ({ever_profitable/len(losers)*100:.1f}%)")

        # Optimal exit time for losers
        print(f"\n  Optimal exit for losers (time that minimizes avg loss):")
        print(f"    Best avg P&L at: {best_time} min ({best_avg_pnl:+.2f}%)")

        # Also check fine-grained (every 2 minutes up to 60)
        fine_best_pnl = -999
        fine_best_t = 0
        for t in range(2, 62, 2):
            loser_pnls = []
            for r in losers:
                pnl = get_pnl_at_time(r["candle_pnls"], t)
                if pnl is not None:
                    loser_pnls.append(pnl)
            if loser_pnls:
                avg = np.mean(loser_pnls)
                if avg > fine_best_pnl:
                    fine_best_pnl = avg
                    fine_best_t = t

        print(f"    Fine-grained optimal: {fine_best_t} min ({fine_best_pnl:+.2f}%)")

    # ── 4. Max drawdown from entry for all trades ──
    print(f"\n  {'─'*60}")
    print(f"  MAX DRAWDOWN FROM ENTRY")
    print(f"  {'─'*60}")
    all_max_dd = []
    for r in strat:
        min_pnl = min(pnl for _, pnl, _, _ in r["candle_pnls"])
        all_max_dd.append(min_pnl)
    print(f"    Avg worst P&L:    {np.mean(all_max_dd):+.2f}%")
    print(f"    Median worst P&L: {np.median(all_max_dd):+.2f}%")
    print(f"    P10 worst:        {np.percentile(all_max_dd, 10):+.2f}%")
    print(f"    Worst ever:       {np.min(all_max_dd):+.2f}%")


def main():
    data_dirs = ["stored_data_oos", "stored_data"]
    date_start = "2025-10-01"
    date_end = "2026-02-28"

    print("=" * 80)
    print("  TIME-TO-TARGET ANALYSIS: Strategy A & B")
    print("=" * 80)
    print(f"  Date range: {date_start} to {date_end}")
    print(f"  Data dirs:  {data_dirs}")
    print(f"  Slippage:   {SLIPPAGE_PCT}%")
    print(f"  Strategy A: Gap>={A_MIN_GAP_PCT}%, Body>={A_MIN_BODY_PCT}%, 2nd green+new hi, Target +{A_TARGET_PCT}%")
    print(f"  Strategy B: Gap>={B_MIN_GAP_PCT}%, Body>={B_MIN_BODY_PCT}%, 2nd green, Target +{B_TARGET_PCT}%")
    print(f"  Priority:   B > A (if qualifies for both, use B)")
    print()

    print("Loading picks...")
    all_dates, daily_picks = load_all_picks(data_dirs)
    all_dates = [d for d in all_dates if date_start <= d <= date_end]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")

    print("\nAnalyzing candle-by-candle P&L after entry...")
    records = analyze_picks(all_dates, daily_picks)
    print(f"  Total qualifying trades: {len(records)}")

    a_count = sum(1 for r in records if r["strategy"] == "A")
    b_count = sum(1 for r in records if r["strategy"] == "B")
    print(f"    Strategy A: {a_count}")
    print(f"    Strategy B: {b_count}")

    # Report each strategy
    report_strategy(records, "A")
    report_strategy(records, "B")

    # ── RECOMMENDATION ──
    print(f"\n{'='*80}")
    print(f"  RECOMMENDATIONS")
    print(f"{'='*80}")

    for strat_name in ["A", "B"]:
        strat = [r for r in records if r["strategy"] == strat_name]
        if not strat:
            continue
        losers = [r for r in strat if not r["hit_target"]]
        target_pct = A_TARGET_PCT if strat_name == "A" else B_TARGET_PCT
        time_limit = A_TIME_LIMIT_MINUTES if strat_name == "A" else B_TIME_LIMIT_MINUTES

        if losers:
            # Find optimal exit for losers (every 2 min)
            best_pnl = -999
            best_t = time_limit
            for t in range(2, 62, 2):
                loser_pnls = []
                for r in losers:
                    pnl = get_pnl_at_time(r["candle_pnls"], t)
                    if pnl is not None:
                        loser_pnls.append(pnl)
                if loser_pnls:
                    avg = np.mean(loser_pnls)
                    if avg > best_pnl:
                        best_pnl = avg
                        best_t = t

            # How much $ saved by early exit vs current time stop?
            current_pnls = []
            optimal_pnls = []
            for r in losers:
                curr = get_pnl_at_time(r["candle_pnls"], time_limit)
                opt = get_pnl_at_time(r["candle_pnls"], best_t)
                if curr is not None:
                    current_pnls.append(curr)
                if opt is not None:
                    optimal_pnls.append(opt)

            curr_avg = np.mean(current_pnls) if current_pnls else 0
            opt_avg = np.mean(optimal_pnls) if optimal_pnls else 0
            improvement = opt_avg - curr_avg

            print(f"\n  Strategy {strat_name}:")
            print(f"    Current time stop: {time_limit} min (avg loser P&L: {curr_avg:+.2f}%)")
            print(f"    Optimal loser exit: {best_t} min (avg loser P&L: {best_pnl:+.2f}%)")
            print(f"    Improvement: {improvement:+.2f}% per loser ({len(losers)} losers)")
            if best_t < time_limit:
                print(f"    --> RECOMMENDATION: Reduce time stop from {time_limit}m to {best_t}m for losers")
            elif best_t > time_limit:
                print(f"    --> RECOMMENDATION: Consider extending time stop to {best_t}m")
            else:
                print(f"    --> Current time stop is near-optimal")
        else:
            print(f"\n  Strategy {strat_name}: No losers found -- all trades hit target!")

    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
