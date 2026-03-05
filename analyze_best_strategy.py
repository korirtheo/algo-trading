"""
Comprehensive Strategy Sweep
==============================
Tests all combinations of entry filters, targets, stops, and time limits
to find the best risk-adjusted strategy for gap-up stocks.

For each qualifying stock, simulates the trade and tracks P&L.
"""

import os
import sys
import io
import pickle
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from collections import defaultdict
from itertools import product

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ET_TZ = ZoneInfo("America/New_York")
DATE_START = "2025-10-01"
DATE_END = "2026-02-28"
SLIPPAGE_PCT = 0.1


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


def analyze(data_dirs):
    print("Loading data...")
    picks_by_date = load_picks(data_dirs)
    dates = sorted(d for d in picks_by_date.keys() if DATE_START <= d <= DATE_END)
    print(f"  {len(dates)} trading days: {dates[0]} to {dates[-1]}")

    # Pre-compute all stock data for fast simulation
    print("  Pre-computing stock data...")
    all_stocks = []

    for date_str in dates:
        for pick in picks_by_date.get(date_str, []):
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) == 0:
                continue
            timestamps = sorted(mh.index)
            if len(timestamps) < 3:
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

            fc_body_pct = (fc_close - fc_open) / fc_open * 100
            fc_range_pct = (fc_high - fc_low) / fc_open * 100
            fc_green = fc_close > fc_open
            fc_upper_wick = (fc_high - max(fc_open, fc_close)) / fc_open * 100

            # Second candle
            sc = mh.loc[timestamps[1]]
            sc_open = float(sc["Open"])
            sc_close = float(sc["Close"])
            sc_high = float(sc["High"])
            sc_green = sc_close > sc_open
            sc_new_high = sc_high > fc_high
            sc_body_pct = (sc_close - sc_open) / sc_open * 100 if sc_open > 0 else 0

            # Build price series from entry points
            # Entry option A: first candle close (9:32)
            entry_a = fc_close * (1 + SLIPPAGE_PCT / 100)
            # Entry option B: second candle close (9:34)
            entry_b = sc_close * (1 + SLIPPAGE_PCT / 100)

            # Price path: high and low at each 2-min candle after each entry
            path_from_a = []  # [(minutes, high, low, close), ...]
            path_from_b = []

            for i, ts in enumerate(timestamps):
                candle = mh.loc[ts]
                c_h = float(candle["High"])
                c_l = float(candle["Low"])
                c_c = float(candle["Close"])
                minutes = i * 2  # 2-min candles

                if i >= 1:  # After 1st candle = after entry A
                    path_from_a.append((minutes, c_h, c_l, c_c))
                if i >= 2:  # After 2nd candle = after entry B
                    path_from_b.append((minutes - 2, c_h, c_l, c_c))

            all_stocks.append({
                "date": date_str,
                "ticker": pick["ticker"],
                "gap_pct": pick["gap_pct"],
                "price": fc_close,
                "pm_volume": pick.get("pm_volume", 0),
                "fc_body_pct": fc_body_pct,
                "fc_range_pct": fc_range_pct,
                "fc_green": fc_green,
                "fc_upper_wick": fc_upper_wick,
                "fc_vol": fc_vol,
                "sc_green": sc_green,
                "sc_new_high": sc_new_high,
                "sc_body_pct": sc_body_pct,
                "entry_a": entry_a,
                "entry_b": entry_b,
                "path_from_a": path_from_a,
                "path_from_b": path_from_b,
            })

    print(f"  {len(all_stocks)} stocks ready for simulation\n")

    # --- SIMULATE FUNCTION ---
    def simulate_trade(stock, entry_price, path, target_pct, stop_pct, time_limit_min):
        """Simulate a single trade. Returns (pnl_pct, exit_reason, exit_minutes)."""
        target_price = entry_price * (1 + target_pct / 100)
        stop_price = entry_price * (1 - stop_pct / 100) if stop_pct > 0 else 0

        for minutes, c_high, c_low, c_close in path:
            # Check stop first (conservative)
            if stop_pct > 0 and c_low <= stop_price:
                pnl = -stop_pct - SLIPPAGE_PCT
                return pnl, "STOP", minutes

            # Check target
            if c_high >= target_price:
                pnl = target_pct - SLIPPAGE_PCT
                return pnl, "TARGET", minutes

            # Check time limit
            if time_limit_min > 0 and minutes >= time_limit_min:
                pnl = (c_close / entry_price - 1) * 100 - SLIPPAGE_PCT
                return pnl, "TIME", minutes

        # EOD exit at last candle
        if path:
            last_close = path[-1][3]
            pnl = (last_close / entry_price - 1) * 100 - SLIPPAGE_PCT
            return pnl, "EOD", path[-1][0]
        return 0, "NO_DATA", 0

    # --- SWEEP ---
    print("=" * 80)
    print("  STRATEGY SWEEP: Testing all combinations")
    print("=" * 80)

    # Entry filters to test
    entry_filters = {
        "ANY_GREEN_1st": lambda s: s["fc_green"] and s["fc_body_pct"] >= 0.5,
        "GREEN_1st>2%": lambda s: s["fc_green"] and s["fc_body_pct"] >= 2.0,
        "GREEN_1st>5%": lambda s: s["fc_green"] and s["fc_body_pct"] >= 5.0,
        "GREEN_1st>8%": lambda s: s["fc_green"] and s["fc_body_pct"] >= 8.0,
        "1st_range>5%": lambda s: s["fc_range_pct"] >= 5.0,
        "1st_range>8%": lambda s: s["fc_range_pct"] >= 8.0,
        "2nd_green": lambda s: s["fc_green"] and s["fc_body_pct"] >= 2.0 and s["sc_green"],
        "2nd_green_newhi": lambda s: s["fc_green"] and s["fc_body_pct"] >= 2.0 and s["sc_green"] and s["sc_new_high"],
        "big_body+2nd": lambda s: s["fc_body_pct"] >= 8.0 and s["sc_green"],
        "any_1st+gap>20": lambda s: s["fc_green"] and s["fc_body_pct"] >= 2.0 and s["gap_pct"] >= 20,
        "any_1st+gap>30": lambda s: s["fc_green"] and s["fc_body_pct"] >= 2.0 and s["gap_pct"] >= 30,
        "low_wick_1st": lambda s: s["fc_green"] and s["fc_body_pct"] >= 2.0 and s["fc_upper_wick"] < 1.0,
        "green1st+hi_vol": lambda s: s["fc_green"] and s["fc_body_pct"] >= 2.0 and s["fc_vol"] > 500000,
    }

    # Parameters to test
    gap_mins = [10, 20, 30]
    targets = [3, 5, 7, 10]
    stops = [0, 3, 5, 8, 10, 15]  # 0 = no price stop
    time_limits = [0, 5, 10, 15, 20, 30, 60]  # 0 = no time limit (EOD)

    # Entry points
    entry_points = {
        "1st_close": ("entry_a", "path_from_a"),
        "2nd_close": ("entry_b", "path_from_b"),
    }

    results = []
    total_combos = len(entry_filters) * len(gap_mins) * len(targets) * len(stops) * len(time_limits) * len(entry_points)
    print(f"  Testing {total_combos:,} combinations...\n")

    for filter_name, filter_fn in entry_filters.items():
        for gap_min in gap_mins:
            for entry_name, (entry_key, path_key) in entry_points.items():
                for target in targets:
                    for stop in stops:
                        for time_limit in time_limits:
                            # Filter stocks
                            qualified = [s for s in all_stocks
                                        if s["gap_pct"] >= gap_min and filter_fn(s)]

                            if len(qualified) < 20:
                                continue

                            # Simulate each trade
                            pnls = []
                            wins = 0
                            losses = 0
                            for s in qualified:
                                ep = s[entry_key]
                                path = s[path_key]
                                if not path or ep <= 0:
                                    continue
                                pnl, reason, _ = simulate_trade(
                                    s, ep, path, target, stop, time_limit
                                )
                                pnls.append(pnl)
                                if pnl > 0:
                                    wins += 1
                                else:
                                    losses += 1

                            if len(pnls) < 20:
                                continue

                            avg_pnl = np.mean(pnls)
                            total_pnl = np.sum(pnls)
                            win_pnls = [p for p in pnls if p > 0]
                            loss_pnls = [p for p in pnls if p <= 0]
                            avg_win = np.mean(win_pnls) if win_pnls else 0
                            avg_loss = abs(np.mean(loss_pnls)) if loss_pnls else 0
                            wr = wins / len(pnls) * 100
                            pf = (sum(win_pnls) / abs(sum(loss_pnls))
                                  if loss_pnls and sum(loss_pnls) != 0 else 99)
                            sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)
                                     if np.std(pnls) > 0 else 0)

                            results.append({
                                "filter": filter_name,
                                "gap": gap_min,
                                "entry": entry_name,
                                "target": target,
                                "stop": stop if stop > 0 else "none",
                                "time": time_limit if time_limit > 0 else "EOD",
                                "n": len(pnls),
                                "wr": wr,
                                "avg_pnl": avg_pnl,
                                "total_pnl": total_pnl,
                                "avg_win": avg_win,
                                "avg_loss": avg_loss,
                                "pf": pf,
                                "sharpe": sharpe,
                            })

    print(f"  {len(results):,} viable combinations found\n")

    # --- SORT BY MULTIPLE CRITERIA ---

    # 1. Best by Profit Factor (min n=30)
    pf_results = [r for r in results if r["n"] >= 30 and r["pf"] < 99]
    pf_results.sort(key=lambda x: -x["pf"])

    print("=" * 80)
    print("  TOP 25 BY PROFIT FACTOR (min 30 trades)")
    print("=" * 80)
    _print_table(pf_results[:25])

    # 2. Best by avg P&L per trade
    avg_results = [r for r in results if r["n"] >= 30]
    avg_results.sort(key=lambda x: -x["avg_pnl"])

    print("\n" + "=" * 80)
    print("  TOP 25 BY AVG P&L PER TRADE (min 30 trades)")
    print("=" * 80)
    _print_table(avg_results[:25])

    # 3. Best by total P&L (most money made)
    total_results = [r for r in results if r["n"] >= 30]
    total_results.sort(key=lambda x: -x["total_pnl"])

    print("\n" + "=" * 80)
    print("  TOP 25 BY TOTAL P&L (min 30 trades)")
    print("=" * 80)
    _print_table(total_results[:25])

    # 4. Best by Sharpe
    sharpe_results = [r for r in results if r["n"] >= 30]
    sharpe_results.sort(key=lambda x: -x["sharpe"])

    print("\n" + "=" * 80)
    print("  TOP 25 BY SHARPE RATIO (min 30 trades)")
    print("=" * 80)
    _print_table(sharpe_results[:25])

    # 5. Best "balanced" score: PF * sqrt(n) * avg_pnl (rewards consistency + size + edge)
    for r in results:
        r["score"] = r["pf"] * np.sqrt(r["n"]) * max(r["avg_pnl"], 0)

    score_results = [r for r in results if r["n"] >= 30 and r["avg_pnl"] > 0]
    score_results.sort(key=lambda x: -x["score"])

    print("\n" + "=" * 80)
    print("  TOP 25 BY BALANCED SCORE (PF * sqrt(n) * avg_pnl)")
    print("=" * 80)
    _print_table(score_results[:25])

    # --- DEEP DIVE on top 3 ---
    print("\n" + "=" * 80)
    print("  DEEP DIVE: TOP 3 STRATEGIES")
    print("=" * 80)

    for i, r in enumerate(score_results[:3], 1):
        print(f"\n  #{i}: {r['filter']} | gap>={r['gap']}% | entry={r['entry']} | "
              f"target=+{r['target']}% | stop={r['stop']}% | time={r['time']}min")
        print(f"      Trades: {r['n']} | WR: {r['wr']:.1f}% | "
              f"Avg P&L: {r['avg_pnl']:+.2f}% | PF: {r['pf']:.2f} | "
              f"Sharpe: {r['sharpe']:.2f}")
        print(f"      Avg Win: +{r['avg_win']:.2f}% | Avg Loss: -{r['avg_loss']:.2f}%")
        print(f"      Total P&L: {r['total_pnl']:+.1f}% across {r['n']} trades")

        # Estimate $ P&L with $25K, 50% sizing, 3 max positions
        est_per_trade = r['avg_pnl'] / 100 * 12500 * 0.5  # 50% of half
        # Rough: ~2 trades/day avg
        trades_per_day = r['n'] / 83  # 83 trading days
        est_daily = est_per_trade * trades_per_day
        print(f"      Est: ~{trades_per_day:.1f} trades/day, ~${est_per_trade:+.0f}/trade, "
              f"~${est_daily:+.0f}/day")


def _print_table(rows):
    if not rows:
        print("  (no results)")
        return
    hdr = (f"  {'Filter':<20} {'Gap':>4} {'Entry':<10} {'Tgt':>4} {'Stop':>5} "
           f"{'Time':>5} {'n':>5} {'WR%':>6} {'AvgPnL':>7} {'PF':>5} {'Sharpe':>7} "
           f"{'AvgWin':>7} {'AvgLoss':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        stop_str = f"{r['stop']}%" if r['stop'] != 'none' else 'none'
        time_str = f"{r['time']}m" if r['time'] != 'EOD' else 'EOD'
        print(f"  {r['filter']:<20} {r['gap']:>3}% {r['entry']:<10} "
              f"+{r['target']:>2}% {stop_str:>5} {time_str:>5} "
              f"{r['n']:>5} {r['wr']:>5.1f}% {r['avg_pnl']:>+6.2f}% "
              f"{r['pf']:>5.2f} {r['sharpe']:>+7.2f} "
              f"+{r['avg_win']:>5.2f}% -{r['avg_loss']:>5.2f}%")


if __name__ == "__main__":
    data_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["stored_data_oos", "stored_data"]
    analyze(data_dirs)
