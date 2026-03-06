"""
Explore New Patterns Beyond the Opening Minutes
================================================
Tests two patterns on gap-up stocks using full intraday (9:30-16:00) data:

Pattern V (VWAP Reclaim):
  - Gap >= 15%
  - Stock drifts below VWAP by 10:30-12:00
  - Then closes a candle above VWAP -> entry
  - Target: +4%, Stop: -2%, Time: 45 min

Pattern R (Volume Spike Reversal):
  - Gap >= 15%
  - After 10:15, stock pulls back from intraday high
  - Green candle with volume > 150% of 20-bar MA -> entry
  - Target: +4%, Stop: -3%, Time: 30 min

Usage:
  python analyze_new_patterns.py
  python analyze_new_patterns.py stored_data_combined
"""

import os
import sys
import numpy as np
import pandas as pd

from test_full import load_all_picks, SLIPPAGE_PCT, ET_TZ

# Force line-buffered stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# -- CONFIG -------------------------------------------------------------------
MIN_GAP_PCT = 15.0

# Pattern V: VWAP Reclaim
V_SCAN_START_MIN = 60       # Start scanning at 10:30 (60 min after open)
V_SCAN_END_MIN = 150        # Stop scanning at 12:00 (150 min after open)
V_BELOW_VWAP_THRESH = 0.5   # Must be >= 0.5% below VWAP before reclaim
V_TARGET_PCT = 4.0
V_STOP_PCT = 2.0
V_TIME_LIMIT_MIN = 45

# Pattern R: Volume Spike Reversal
R_SCAN_START_MIN = 45       # Start scanning at 10:15 (45 min after open)
R_SCAN_END_MIN = 300        # Stop scanning at 14:30 (300 min after open)
R_PULLBACK_FROM_HIGH = 3.0  # Must have pulled back >= 3% from intraday high
R_VOL_MULT = 1.5            # Volume > 1.5x of 20-bar average
R_MIN_BODY_PCT = 0.5        # Green candle with body >= 0.5%
R_TARGET_PCT = 4.0
R_STOP_PCT = 3.0
R_TIME_LIMIT_MIN = 30


def compute_vwap(candles):
    """Compute cumulative VWAP from 1-min candles (as a Series)."""
    typical = (candles["High"].values + candles["Low"].values + candles["Close"].values) / 3
    vol = candles["Volume"].values.astype(float)
    cum_tp_vol = np.cumsum(typical * vol)
    cum_vol = np.cumsum(vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    return vwap


def scan_vwap_reclaim(candles, gap_pct):
    """Look for VWAP reclaim pattern. Returns entry dict or None."""
    n = len(candles)
    if n < V_SCAN_END_MIN:
        return None

    vwap = compute_vwap(candles)
    was_below = False

    for i in range(V_SCAN_START_MIN, min(V_SCAN_END_MIN, n)):
        v = vwap[i]
        if np.isnan(v) or v <= 0:
            continue

        close = float(candles.iloc[i]["Close"])
        opn = float(candles.iloc[i]["Open"])
        pct_from_vwap = (close - v) / v * 100

        # Mark if we've been meaningfully below VWAP
        if pct_from_vwap <= -V_BELOW_VWAP_THRESH:
            was_below = True

        # Reclaim: was below, now candle closes above VWAP and opened below
        if was_below and close > v and opn < v:
            return {
                "candle_idx": i,
                "entry_price": close,
                "vwap": v,
            }

    return None


def scan_vol_spike_reversal(candles, gap_pct):
    """Look for volume spike reversal. Returns entry dict or None."""
    n = len(candles)
    if n < R_SCAN_START_MIN + 20:
        return None

    vols = candles["Volume"].values.astype(float)
    highs = candles["High"].values.astype(float)

    for i in range(R_SCAN_START_MIN, min(R_SCAN_END_MIN, n)):
        close = float(candles.iloc[i]["Close"])
        opn = float(candles.iloc[i]["Open"])

        # Must be green
        if close <= opn:
            continue

        body_pct = (close - opn) / opn * 100 if opn > 0 else 0
        if body_pct < R_MIN_BODY_PCT:
            continue

        # Pullback from intraday high
        high_so_far = float(np.max(highs[:i]))
        if high_so_far <= 0:
            continue
        pullback = (high_so_far - close) / high_so_far * 100
        if pullback < R_PULLBACK_FROM_HIGH:
            continue

        # Volume spike
        if i < 20:
            continue
        avg_vol = np.mean(vols[i - 20:i])
        if avg_vol <= 0:
            continue
        vol_mult = vols[i] / avg_vol
        if vol_mult < R_VOL_MULT:
            continue

        return {
            "candle_idx": i,
            "entry_price": close,
            "pullback_pct": pullback,
            "vol_mult": vol_mult,
        }

    return None


def simulate_trade(candles, entry_idx, entry_price, target_pct, stop_pct, time_limit):
    """Simulate from entry with target/stop/time. Returns result dict."""
    slip_entry = entry_price * (1 + SLIPPAGE_PCT / 100)

    for j in range(entry_idx + 1, min(entry_idx + time_limit + 1, len(candles))):
        low = float(candles.iloc[j]["Low"])
        high = float(candles.iloc[j]["High"])

        # Stop first (conservative)
        stop_price = slip_entry * (1 - stop_pct / 100)
        if low <= stop_price:
            exit_p = stop_price * (1 - SLIPPAGE_PCT / 100)
            return {"pnl_pct": (exit_p / slip_entry - 1) * 100,
                    "reason": "STOP", "bars_held": j - entry_idx}

        # Target
        tgt_price = slip_entry * (1 + target_pct / 100)
        if high >= tgt_price:
            exit_p = tgt_price * (1 - SLIPPAGE_PCT / 100)
            return {"pnl_pct": (exit_p / slip_entry - 1) * 100,
                    "reason": "TARGET", "bars_held": j - entry_idx}

    # Time stop
    last = min(entry_idx + time_limit, len(candles) - 1)
    exit_p = float(candles.iloc[last]["Close"]) * (1 - SLIPPAGE_PCT / 100)
    return {"pnl_pct": (exit_p / slip_entry - 1) * 100,
            "reason": "TIME_STOP", "bars_held": last - entry_idx}


def analyze_all(data_dirs, date_range):
    """Scan both patterns across all dates."""
    print("Loading data...", flush=True)
    all_dates, daily_picks = load_all_picks(data_dirs)
    all_dates = [d for d in all_dates if date_range[0] <= d <= date_range[1]]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}\n", flush=True)

    v_results = []
    r_results = []

    for di, d in enumerate(all_dates):
        picks = daily_picks.get(d, [])
        for pick in picks:
            if pick["gap_pct"] < MIN_GAP_PCT:
                continue

            candles = pick["market_hour_candles"]
            if len(candles) < 60:
                continue

            ticker = pick["ticker"]
            gap = pick["gap_pct"]

            # Pattern V
            v_entry = scan_vwap_reclaim(candles, gap)
            if v_entry:
                res = simulate_trade(candles, v_entry["candle_idx"],
                                     v_entry["entry_price"],
                                     V_TARGET_PCT, V_STOP_PCT, V_TIME_LIMIT_MIN)
                v_results.append({
                    "date": d, "ticker": ticker, "gap_pct": gap,
                    "entry_candle": v_entry["candle_idx"],
                    "entry_price": v_entry["entry_price"],
                    **res,
                })

            # Pattern R
            r_entry = scan_vol_spike_reversal(candles, gap)
            if r_entry:
                res = simulate_trade(candles, r_entry["candle_idx"],
                                     r_entry["entry_price"],
                                     R_TARGET_PCT, R_STOP_PCT, R_TIME_LIMIT_MIN)
                r_results.append({
                    "date": d, "ticker": ticker, "gap_pct": gap,
                    "entry_candle": r_entry["candle_idx"],
                    "entry_price": r_entry["entry_price"],
                    "pullback_pct": r_entry.get("pullback_pct", 0),
                    "vol_mult": r_entry.get("vol_mult", 0),
                    **res,
                })

        if (di + 1) % 50 == 0:
            print(f"  Scanned {di+1}/{len(all_dates)} days  "
                  f"(V:{len(v_results)} R:{len(r_results)} signals so far)", flush=True)

    return v_results, r_results


def print_summary(results, name, target_pct, stop_pct, time_limit):
    """Print analysis summary for one pattern."""
    if not results:
        print(f"\n  {name}: 0 trades found\n")
        return

    n = len(results)
    wins = [r for r in results if r["pnl_pct"] > 0]
    losses = [r for r in results if r["pnl_pct"] <= 0]
    wr = len(wins) / n * 100
    avg_win = np.mean([r["pnl_pct"] for r in wins]) if wins else 0
    avg_loss = np.mean([r["pnl_pct"] for r in losses]) if losses else 0
    total_pnl = sum(r["pnl_pct"] for r in results)
    win_sum = sum(r["pnl_pct"] for r in wins)
    loss_sum = abs(sum(r["pnl_pct"] for r in losses))
    pf = win_sum / loss_sum if loss_sum > 0 else 0
    avg_bars = np.mean([r["bars_held"] for r in results])

    reasons = {}
    for r in results:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    # Time buckets
    buckets = {"10:30-11:00": (60, 90), "11:00-11:30": (90, 120),
               "11:30-12:00": (120, 150), "12:00-13:00": (150, 210),
               "13:00-14:30": (210, 300), "14:30+": (300, 999)}

    print(f"\n{'='*60}")
    print(f"  PATTERN {name}")
    print(f"  Target: +{target_pct}% | Stop: -{stop_pct}% | Time: {time_limit}m")
    print(f"{'='*60}")
    print(f"  Trades:        {n}")
    print(f"  Winners:       {len(wins)} ({wr:.1f}%)")
    print(f"  Losers:        {len(losses)}")
    print(f"  Avg Win:       {avg_win:+.2f}%")
    print(f"  Avg Loss:      {avg_loss:+.2f}%")
    print(f"  Total P&L:     {total_pnl:+.1f}% (sum of %)")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg Hold:      {avg_bars:.0f} bars")

    print(f"\n  Exit Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:>12}: {count:>4} ({count/n*100:.0f}%)")

    print(f"\n  Entry Time Distribution:")
    for label, (lo, hi) in buckets.items():
        trades = [r for r in results if lo <= r["entry_candle"] < hi]
        if trades:
            bwr = sum(1 for t in trades if t["pnl_pct"] > 0) / len(trades) * 100
            bpnl = sum(t["pnl_pct"] for t in trades)
            print(f"    {label:>12}: {len(trades):>4} trades, {bwr:.0f}% WR, {bpnl:+.1f}% P&L")

    # Gap buckets
    print(f"\n  By Gap Size:")
    for lo, hi, label in [(15, 25, "15-25%"), (25, 40, "25-40%"), (40, 100, "40-100%"), (100, 9999, "100%+")]:
        trades = [r for r in results if lo <= r["gap_pct"] < hi]
        if trades:
            bwr = sum(1 for t in trades if t["pnl_pct"] > 0) / len(trades) * 100
            bpnl = sum(t["pnl_pct"] for t in trades)
            bpf = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0) /
                       sum(t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0)) if any(t["pnl_pct"] <= 0 for t in trades) else 0
            print(f"    Gap {label:>7}: {len(trades):>4} trades, {bwr:.0f}% WR, PF {bpf:.2f}, {bpnl:+.1f}%")

    sorted_by_pnl = sorted(results, key=lambda x: x["pnl_pct"], reverse=True)
    print(f"\n  Top 5 Winners:")
    for r in sorted_by_pnl[:5]:
        print(f"    {r['date']} {r['ticker']:>6} gap {r['gap_pct']:.0f}%  "
              f"entry@c{r['entry_candle']}  {r['pnl_pct']:+.2f}%  {r['reason']}")
    print(f"\n  Top 5 Losers:")
    for r in sorted_by_pnl[-5:]:
        print(f"    {r['date']} {r['ticker']:>6} gap {r['gap_pct']:.0f}%  "
              f"entry@c{r['entry_candle']}  {r['pnl_pct']:+.2f}%  {r['reason']}")
    print()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    data_dirs = args if args else ["stored_data_combined"]
    date_range = ("2025-01-01", "2026-02-28")

    print(f"New Pattern Explorer: VWAP Reclaim + Volume Spike Reversal")
    print(f"{'='*60}")
    print(f"  Data:    {data_dirs}")
    print(f"  Dates:   {date_range[0]} to {date_range[1]}")
    print(f"  Min Gap: {MIN_GAP_PCT}%")
    print(f"{'='*60}\n")

    v_results, r_results = analyze_all(data_dirs, date_range)

    print_summary(v_results, "V (VWAP Reclaim)",
                  V_TARGET_PCT, V_STOP_PCT, V_TIME_LIMIT_MIN)
    print_summary(r_results, "R (Volume Spike Reversal)",
                  R_TARGET_PCT, R_STOP_PCT, R_TIME_LIMIT_MIN)

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    for name, res in [("V (VWAP)", v_results), ("R (VolSpike)", r_results)]:
        if res:
            n = len(res)
            wr = sum(1 for r in res if r["pnl_pct"] > 0) / n * 100
            ws = sum(r["pnl_pct"] for r in res if r["pnl_pct"] > 0)
            ls = abs(sum(r["pnl_pct"] for r in res if r["pnl_pct"] <= 0))
            pf = ws / ls if ls > 0 else 0
            avg = np.mean([r["pnl_pct"] for r in res])
            print(f"  {name:>15}: {n:>4} trades, {wr:.0f}% WR, PF {pf:.2f}, avg {avg:+.2f}%")
        else:
            print(f"  {name:>15}: 0 trades")
    print()
