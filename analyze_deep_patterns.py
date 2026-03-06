"""
Deep Pattern Analysis: Failed Gap Shorts + Big Body Filter + Combined Filters
==============================================================================
Follows up on analyze_new_patterns.py findings:
1. Failed gap short: gap>=25%, 1st red, 2nd red -> short at candle 2 close
2. Big body filter: does adding body>5% on 1st candle improve G+A+F?
3. Combined filters: big body + vol confirm + low PM vol

Usage:
  python analyze_deep_patterns.py
"""

import os
import sys
import numpy as np
from collections import defaultdict
from zoneinfo import ZoneInfo

from test_full import (
    load_all_picks,
    SLIPPAGE_PCT,
    ET_TZ,
)

DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2025-01-01", "2026-02-28")


def _simulate_trade(mh_candles, entry_idx, entry_price, target_pct, stop_pct, max_minutes, direction="long"):
    """Simulate a trade. direction='short' inverts P&L."""
    et_tz = ZoneInfo("America/New_York")
    if entry_idx >= len(mh_candles):
        return {"hit": False, "exit_pnl": 0, "reason": "NO_DATA", "max_up": 0, "max_down": 0, "minutes": 0}

    entry_time = mh_candles.index[entry_idx]
    try:
        entry_et = entry_time.astimezone(et_tz)
    except Exception:
        entry_et = entry_time

    max_up = 0.0
    max_down = 0.0

    for i in range(entry_idx, len(mh_candles)):
        ts = mh_candles.index[i]
        try:
            ts_et = ts.astimezone(et_tz)
        except Exception:
            ts_et = ts

        minutes_in = (ts_et.hour * 60 + ts_et.minute) - (entry_et.hour * 60 + entry_et.minute)
        minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)

        c_high = float(mh_candles.iloc[i]["High"])
        c_low = float(mh_candles.iloc[i]["Low"])
        c_close = float(mh_candles.iloc[i]["Close"])

        up_pct = (c_high / entry_price - 1) * 100
        down_pct = (c_low / entry_price - 1) * 100
        max_up = max(max_up, up_pct)
        max_down = min(max_down, down_pct)

        if direction == "short":
            # For shorts: stock going DOWN = profit
            # Stop: price goes UP by stop_pct
            if stop_pct > 0 and up_pct >= stop_pct:
                return {"hit": False, "exit_pnl": -stop_pct, "reason": "STOP", "max_up": max_up, "max_down": max_down, "minutes": minutes_in}
            # Target: price goes DOWN by target_pct
            if down_pct <= -target_pct:
                return {"hit": True, "exit_pnl": target_pct, "reason": "TARGET", "max_up": max_up, "max_down": max_down, "minutes": minutes_in}
        else:
            # Long
            if stop_pct > 0 and down_pct <= -stop_pct:
                return {"hit": False, "exit_pnl": -stop_pct, "reason": "STOP", "max_up": max_up, "max_down": max_down, "minutes": minutes_in}
            if up_pct >= target_pct:
                return {"hit": True, "exit_pnl": target_pct, "reason": "TARGET", "max_up": max_up, "max_down": max_down, "minutes": minutes_in}

        # Time stop
        if max_minutes > 0 and minutes_in >= max_minutes:
            raw_pnl = (c_close / entry_price - 1) * 100
            if direction == "short":
                raw_pnl = -raw_pnl
            return {"hit": raw_pnl > 0, "exit_pnl": raw_pnl, "reason": "TIME", "max_up": max_up, "max_down": max_down, "minutes": minutes_in}

        # EOD
        if minutes_to_close <= 15:
            raw_pnl = (c_close / entry_price - 1) * 100
            if direction == "short":
                raw_pnl = -raw_pnl
            return {"hit": raw_pnl > 0, "exit_pnl": raw_pnl, "reason": "EOD", "max_up": max_up, "max_down": max_down, "minutes": minutes_in}

    last_close = float(mh_candles.iloc[-1]["Close"])
    raw_pnl = (last_close / entry_price - 1) * 100
    if direction == "short":
        raw_pnl = -raw_pnl
    return {"hit": raw_pnl > 0, "exit_pnl": raw_pnl, "reason": "EOD", "max_up": max_up, "max_down": max_down, "minutes": 999}


def analyze_all(picks_by_date, all_dates):
    """Run all deep analyses."""

    # ============================================================
    # SECTION 1: Failed Gap Short sweep
    # ============================================================
    print(f"\n{'='*70}")
    print("  FAILED GAP SHORT: TARGET/STOP/TIME SWEEP")
    print(f"{'='*70}")
    print("  Entry: gap>=25%, 1st candle RED, 2nd candle RED -> short at c2 close\n")

    # Collect failed gap candidates
    short_candidates = []
    for d in all_dates:
        picks = picks_by_date.get(d, [])
        for pick in picks:
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) < 3:
                continue
            gap_pct = pick["gap_pct"]
            if gap_pct < 25:
                continue

            c1 = mh.iloc[0]
            c2 = mh.iloc[1]
            c1_green = float(c1["Close"]) > float(c1["Open"])
            c2_green = float(c2["Close"]) > float(c2["Open"])

            if not c1_green and not c2_green:  # Both red
                entry_price = float(c2["Close"])
                if entry_price > 0:
                    short_candidates.append({
                        "ticker": pick["ticker"],
                        "date": d,
                        "gap_pct": gap_pct,
                        "entry_price": entry_price,
                        "mh": mh,
                    })

    print(f"  Found {len(short_candidates)} failed-gap-short candidates (gap>=25%, 2 red candles)\n")

    # Sweep short targets/stops/times
    print(f"  {'Target':>6} {'Stop':>6} {'Time':>5} {'Trades':>6} {'WR':>6} {'Avg PnL':>8} {'PF':>6}")
    print(f"  {'-'*50}")
    best_short = None
    best_short_pf = 0

    for target_pct in [2.0, 3.0, 5.0, 7.0]:
        for stop_pct in [3.0, 5.0, 8.0]:
            for max_min in [5, 10, 15, 20]:
                wins = 0
                pnls = []
                for c in short_candidates:
                    result = _simulate_trade(c["mh"], 2, c["entry_price"], target_pct, stop_pct, max_min, "short")
                    pnls.append(result["exit_pnl"])
                    if result["hit"]:
                        wins += 1

                n = len(pnls)
                if n == 0:
                    continue
                wr = wins / n * 100
                avg_pnl = np.mean(pnls)
                win_sum = sum(p for p in pnls if p > 0)
                loss_sum = abs(sum(p for p in pnls if p <= 0))
                pf = win_sum / loss_sum if loss_sum > 0 else 999

                if pf > best_short_pf:
                    best_short_pf = pf
                    best_short = (target_pct, stop_pct, max_min, n, wr, avg_pnl, pf)

                if pf >= 1.0 or (target_pct == 3.0 and stop_pct == 5.0):
                    print(f"  {target_pct:>5.1f}% {stop_pct:>5.1f}% {max_min:>4}m {n:>6} {wr:>5.1f}% {avg_pnl:>+7.2f}% {pf:>5.2f}")

    if best_short:
        print(f"\n  BEST SHORT CONFIG: target={best_short[0]}%, stop={best_short[1]}%, "
              f"time={best_short[2]}m -> WR={best_short[4]:.1f}%, PF={best_short[6]:.2f}")

    # Also test gap>=30% and gap>=35%
    for gap_filter in [30, 35, 40]:
        filtered = [c for c in short_candidates if c["gap_pct"] >= gap_filter]
        if not filtered:
            continue
        print(f"\n  --- Gap >= {gap_filter}% ({len(filtered)} candidates) ---")
        for target_pct in [3.0, 5.0]:
            for stop_pct in [5.0, 8.0]:
                for max_min in [10, 15]:
                    wins = 0
                    pnls = []
                    for c in filtered:
                        result = _simulate_trade(c["mh"], 2, c["entry_price"], target_pct, stop_pct, max_min, "short")
                        pnls.append(result["exit_pnl"])
                        if result["hit"]:
                            wins += 1
                    n = len(pnls)
                    if n == 0:
                        continue
                    wr = wins / n * 100
                    avg_pnl = np.mean(pnls)
                    win_sum = sum(p for p in pnls if p > 0)
                    loss_sum = abs(sum(p for p in pnls if p <= 0))
                    pf = win_sum / loss_sum if loss_sum > 0 else 999
                    print(f"  {target_pct:>5.1f}% {stop_pct:>5.1f}% {max_min:>4}m {n:>6} {wr:>5.1f}% {avg_pnl:>+7.2f}% {pf:>5.2f}")

    # ============================================================
    # SECTION 2: Big Body Filter Enhancement
    # ============================================================
    print(f"\n{'='*70}")
    print("  BIG BODY FILTER: IMPACT ON G+A+F STRATEGIES")
    print(f"{'='*70}")
    print("  Does requiring body>X% on 1st candle improve win rate?\n")

    # Collect all green-green entries categorized by strategy
    strat_results = defaultdict(list)

    for d in all_dates:
        picks = picks_by_date.get(d, [])
        for pick in picks:
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) < 3:
                continue

            gap_pct = pick["gap_pct"]
            pm_vol = pick.get("pm_volume", 0)
            pm_high = pick["premarket_high"]
            pm_dollar_vol = pm_vol * pm_high if pm_high > 0 else 0

            c1 = mh.iloc[0]
            c2 = mh.iloc[1]
            c1_open, c1_close, c1_high = float(c1["Open"]), float(c1["Close"]), float(c1["High"])
            c1_vol = float(c1["Volume"])
            c2_open, c2_close, c2_high = float(c2["Open"]), float(c2["Close"]), float(c2["High"])
            c2_vol = float(c2["Volume"])

            if c1_open <= 0 or c2_close <= 0:
                continue

            c1_body_pct = (c1_close / c1_open - 1) * 100
            c1_green = c1_close > c1_open
            c2_green = c2_close > c2_open
            c2_new_high = c2_high > c1_high
            c1_upper_wick_pct = ((c1_high - max(c1_open, c1_close)) / c1_open * 100) if c1_open > 0 else 0
            vol_confirm = c2_vol > c1_vol

            entry_price = c2_close

            # Classify into G, A, F (same logic as test_green_candle_combined)
            if not c1_green:
                continue  # Only green first candles for this section

            strategy = None
            if gap_pct >= 25 and c2_green and c2_new_high:
                strategy = "G"
            elif gap_pct >= 15 and c2_green and c2_new_high:
                strategy = "A"
            elif gap_pct >= 10 and c2_green:
                strategy = "F"

            if not strategy:
                continue

            # Run trades with strategy-specific params
            targets = {"G": (8.0, 20), "A": (3.0, 10), "F": (10.0, 3)}
            target_pct, time_limit = targets[strategy]

            result = _simulate_trade(mh, 2, entry_price, target_pct, 0, time_limit, "long")

            strat_results[strategy].append({
                "ticker": pick["ticker"],
                "date": d,
                "gap_pct": gap_pct,
                "c1_body_pct": c1_body_pct,
                "c1_upper_wick_pct": c1_upper_wick_pct,
                "vol_confirm": vol_confirm,
                "pm_dollar_vol": pm_dollar_vol,
                "hit": result["hit"],
                "exit_pnl": result["exit_pnl"],
                "max_up": result["max_up"],
                "max_down": result["max_down"],
            })

    for strat in ["G", "A", "F"]:
        trades = strat_results[strat]
        if not trades:
            continue

        targets = {"G": (8.0, 20), "A": (3.0, 10), "F": (10.0, 3)}
        target_pct, time_limit = targets[strat]
        print(f"\n  Strategy {strat} (target={target_pct}%, time={time_limit}m)")
        print(f"  {'-'*60}")

        # Baseline (no filter)
        n = len(trades)
        wins = sum(1 for t in trades if t["hit"])
        pnls = [t["exit_pnl"] for t in trades]
        w_sum = sum(p for p in pnls if p > 0)
        l_sum = abs(sum(p for p in pnls if p <= 0))
        pf = w_sum / l_sum if l_sum > 0 else 999
        print(f"    Baseline (no filter):  {n:>4} trades, {wins/n*100:.1f}% WR, avg {np.mean(pnls):+.2f}%, PF {pf:.2f}")

        # Body filters
        for body_min in [2.0, 3.0, 5.0, 7.0]:
            filtered = [t for t in trades if t["c1_body_pct"] >= body_min]
            if not filtered:
                continue
            n2 = len(filtered)
            wins2 = sum(1 for t in filtered if t["hit"])
            pnls2 = [t["exit_pnl"] for t in filtered]
            w2 = sum(p for p in pnls2 if p > 0)
            l2 = abs(sum(p for p in pnls2 if p <= 0))
            pf2 = w2 / l2 if l2 > 0 else 999
            delta_wr = wins2/n2*100 - wins/n*100
            print(f"    Body >= {body_min:>4.1f}%:         {n2:>4} trades, {wins2/n2*100:.1f}% WR ({delta_wr:+.1f}), avg {np.mean(pnls2):+.2f}%, PF {pf2:.2f}")

        # Volume confirm filter
        vc_yes = [t for t in trades if t["vol_confirm"]]
        vc_no = [t for t in trades if not t["vol_confirm"]]
        if vc_yes:
            n_vc = len(vc_yes)
            wins_vc = sum(1 for t in vc_yes if t["hit"])
            pnls_vc = [t["exit_pnl"] for t in vc_yes]
            w_vc = sum(p for p in pnls_vc if p > 0)
            l_vc = abs(sum(p for p in pnls_vc if p <= 0))
            pf_vc = w_vc / l_vc if l_vc > 0 else 999
            print(f"    Vol Confirm YES:       {n_vc:>4} trades, {wins_vc/n_vc*100:.1f}% WR, avg {np.mean(pnls_vc):+.2f}%, PF {pf_vc:.2f}")

        if vc_no:
            n_vn = len(vc_no)
            wins_vn = sum(1 for t in vc_no if t["hit"])
            pnls_vn = [t["exit_pnl"] for t in vc_no]
            w_vn = sum(p for p in pnls_vn if p > 0)
            l_vn = abs(sum(p for p in pnls_vn if p <= 0))
            pf_vn = w_vn / l_vn if l_vn > 0 else 999
            print(f"    Vol Confirm NO:        {n_vn:>4} trades, {wins_vn/n_vn*100:.1f}% WR, avg {np.mean(pnls_vn):+.2f}%, PF {pf_vn:.2f}")

        # PM Dollar Volume filter
        low_pm = [t for t in trades if t["pm_dollar_vol"] < 5_000_000]
        high_pm = [t for t in trades if t["pm_dollar_vol"] >= 5_000_000]
        if low_pm:
            n_lp = len(low_pm)
            wins_lp = sum(1 for t in low_pm if t["hit"])
            pnls_lp = [t["exit_pnl"] for t in low_pm]
            w_lp = sum(p for p in pnls_lp if p > 0)
            l_lp = abs(sum(p for p in pnls_lp if p <= 0))
            pf_lp = w_lp / l_lp if l_lp > 0 else 999
            print(f"    Low PM $Vol (<$5M):    {n_lp:>4} trades, {wins_lp/n_lp*100:.1f}% WR, avg {np.mean(pnls_lp):+.2f}%, PF {pf_lp:.2f}")

        if high_pm:
            n_hp = len(high_pm)
            wins_hp = sum(1 for t in high_pm if t["hit"])
            pnls_hp = [t["exit_pnl"] for t in high_pm]
            w_hp = sum(p for p in pnls_hp if p > 0)
            l_hp = abs(sum(p for p in pnls_hp if p <= 0))
            pf_hp = w_hp / l_hp if l_hp > 0 else 999
            print(f"    High PM $Vol (>=$5M):  {n_hp:>4} trades, {wins_hp/n_hp*100:.1f}% WR, avg {np.mean(pnls_hp):+.2f}%, PF {pf_hp:.2f}")

        # Combined best filters: body>=5% + vol confirm
        combo = [t for t in trades if t["c1_body_pct"] >= 5.0 and t["vol_confirm"]]
        if combo:
            n_co = len(combo)
            wins_co = sum(1 for t in combo if t["hit"])
            pnls_co = [t["exit_pnl"] for t in combo]
            w_co = sum(p for p in pnls_co if p > 0)
            l_co = abs(sum(p for p in pnls_co if p <= 0))
            pf_co = w_co / l_co if l_co > 0 else 999
            print(f"    COMBO body>=5%+VolConf:{n_co:>4} trades, {wins_co/n_co*100:.1f}% WR, avg {np.mean(pnls_co):+.2f}%, PF {pf_co:.2f}")

    # ============================================================
    # SECTION 3: New strategy idea - "D" for Doji/reversal
    # ============================================================
    print(f"\n{'='*70}")
    print("  NEW STRATEGY IDEA: 'H' (High-Conviction)")
    print(f"{'='*70}")
    print("  Entry: gap>=25% + 1st body>5% + 2nd green + new high + vol confirm")
    print("  Basically G strategy but with body + vol filters\n")

    h_candidates = []
    for d in all_dates:
        picks = picks_by_date.get(d, [])
        for pick in picks:
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) < 3:
                continue

            gap_pct = pick["gap_pct"]
            if gap_pct < 25:
                continue

            c1 = mh.iloc[0]
            c2 = mh.iloc[1]
            c1_open, c1_close = float(c1["Open"]), float(c1["Close"])
            c1_vol = float(c1["Volume"])
            c2_open, c2_close, c2_high = float(c2["Open"]), float(c2["Close"]), float(c2["High"])
            c2_vol = float(c2["Volume"])
            c1_high = float(c1["High"])

            if c1_open <= 0:
                continue

            c1_body_pct = (c1_close / c1_open - 1) * 100
            c1_green = c1_close > c1_open
            c2_green = c2_close > c2_open
            c2_new_high = c2_high > c1_high

            if c1_green and c1_body_pct >= 5.0 and c2_green and c2_new_high and c2_vol > c1_vol:
                entry_price = c2_close
                if entry_price > 0:
                    h_candidates.append({
                        "ticker": pick["ticker"],
                        "date": d,
                        "gap_pct": gap_pct,
                        "entry_price": entry_price,
                        "mh": mh,
                        "c1_body_pct": c1_body_pct,
                    })

    print(f"  Found {len(h_candidates)} H-strategy candidates\n")

    print(f"  {'Target':>6} {'Time':>5} {'Trades':>6} {'WR':>6} {'Avg PnL':>8} {'PF':>6}")
    print(f"  {'-'*45}")

    for target_pct in [3.0, 5.0, 8.0, 10.0, 12.0, 15.0]:
        for max_min in [5, 10, 15, 20, 30]:
            wins = 0
            pnls = []
            for c in h_candidates:
                result = _simulate_trade(c["mh"], 2, c["entry_price"], target_pct, 0, max_min, "long")
                pnls.append(result["exit_pnl"])
                if result["hit"]:
                    wins += 1
            n = len(pnls)
            if n == 0:
                continue
            wr = wins / n * 100
            avg_pnl = np.mean(pnls)
            w_s = sum(p for p in pnls if p > 0)
            l_s = abs(sum(p for p in pnls if p <= 0))
            pf = w_s / l_s if l_s > 0 else 999
            if pf >= 1.2 or (target_pct in [8.0, 10.0] and max_min == 20):
                print(f"  {target_pct:>5.1f}% {max_min:>4}m {n:>6} {wr:>5.1f}% {avg_pnl:>+7.2f}% {pf:>5.2f}")

    # ============================================================
    # SECTION 4: Third candle entry instead of second
    # ============================================================
    print(f"\n{'='*70}")
    print("  CANDLE 3 ENTRY: Wait for 3rd green candle in a row")
    print(f"{'='*70}")
    print("  Entry: gap>=25% + 1st green + 2nd green + 3rd green -> buy c3 close\n")

    c3_candidates = []
    for d in all_dates:
        picks = picks_by_date.get(d, [])
        for pick in picks:
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) < 4:
                continue

            gap_pct = pick["gap_pct"]
            if gap_pct < 15:
                continue

            c1 = mh.iloc[0]
            c2 = mh.iloc[1]
            c3 = mh.iloc[2]

            c1_green = float(c1["Close"]) > float(c1["Open"])
            c2_green = float(c2["Close"]) > float(c2["Open"])
            c3_green = float(c3["Close"]) > float(c3["Open"])
            c3_new_high = float(c3["High"]) > max(float(c1["High"]), float(c2["High"]))

            entry_price = float(c3["Close"])
            if entry_price <= 0:
                continue

            if c1_green and c2_green and c3_green:
                c3_candidates.append({
                    "ticker": pick["ticker"],
                    "date": d,
                    "gap_pct": gap_pct,
                    "entry_price": entry_price,
                    "mh": mh,
                    "c3_new_high": c3_new_high,
                })

    all_c3 = c3_candidates
    c3_new_hi = [c for c in c3_candidates if c["c3_new_high"]]

    print(f"  3-green candles: {len(all_c3)} candidates")
    print(f"  3-green + new high on c3: {len(c3_new_hi)} candidates\n")

    for label, candidates in [("3 Green (all)", all_c3), ("3 Green + New Hi", c3_new_hi)]:
        if not candidates:
            continue
        print(f"  {label}:")
        print(f"    {'Target':>6} {'Time':>5} {'Trades':>6} {'WR':>6} {'Avg PnL':>8} {'PF':>6}")
        print(f"    {'-'*45}")
        for target_pct in [3.0, 5.0, 8.0, 10.0]:
            for max_min in [5, 10, 15, 20]:
                wins = 0
                pnls = []
                for c in candidates:
                    result = _simulate_trade(c["mh"], 3, c["entry_price"], target_pct, 0, max_min, "long")
                    pnls.append(result["exit_pnl"])
                    if result["hit"]:
                        wins += 1
                n = len(pnls)
                if n == 0:
                    continue
                wr = wins / n * 100
                avg_pnl = np.mean(pnls)
                w_s = sum(p for p in pnls if p > 0)
                l_s = abs(sum(p for p in pnls if p <= 0))
                pf = w_s / l_s if l_s > 0 else 999
                print(f"    {target_pct:>5.1f}% {max_min:>4}m {n:>6} {wr:>5.1f}% {avg_pnl:>+7.2f}% {pf:>5.2f}")
        print()

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("Loading data...")
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")

    analyze_all(daily_picks, all_dates)
