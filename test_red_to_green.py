"""
Strategy R2G: Red-to-Green Move Backtest (standalone)
=====================================================
Gap-up stock opens, first candle goes RED (close < open), price dips below
the day open, then reverses and reclaims above open with volume -> momentum entry.

Most strategies discard stocks with a red first candle. But a red-to-green reversal
on a gapper is one of the strongest intraday signals: sellers are exhausted, and the
reclaim of open triggers buy stops + FOMO.

Entry: Price crosses above day open after dipping below it, with volume confirmation
Exit:  Target % + trailing stop + hard stop + time stop + EOD

Usage:
  python test_red_to_green.py
  python test_red_to_green.py --no-charts
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
from datetime import datetime

from test_full import (
    load_all_picks,
    _is_warrant_or_unit,
    SLIPPAGE_PCT,
    STARTING_CASH,
    TOP_N,
    MARGIN_THRESHOLD,
    VOL_CAP_PCT,
    ET_TZ,
)

import io, sys as _sys
if hasattr(_sys.stdout, 'buffer'):
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8",
                                    errors="replace", line_buffering=True)

# --- STRATEGY R2G CONFIG (defaults — will be optimized) ---
R2G_MIN_GAP_PCT = 10.0           # Minimum gap % to qualify
R2G_MIN_DIP_PCT = 1.0            # Min dip below open % (must actually go red)
R2G_MAX_DIP_PCT = 15.0           # Max dip — too deep = broken, not a reversal
R2G_EARLIEST_CANDLE = 3          # Don't enter on very first candles
R2G_LATEST_CANDLE = 60           # Latest candle for reclaim (no late entries)
R2G_VOL_SURGE_MULT = 1.5         # Reclaim candle vol >= Nx avg of prior candles
R2G_MIN_RECLAIM_BODY_PCT = 0.5   # Min green body % on reclaim candle
R2G_REQUIRE_ABOVE_VWAP = False   # Must be above VWAP at entry
R2G_REQUIRE_CLOSE_ABOVE_OPEN = True  # Reclaim candle must close above day open

# Exit params
R2G_TARGET_PCT = 8.0             # Profit target %
R2G_STOP_PCT = 5.0               # Hard stop %
R2G_TRAIL_PCT = 3.0              # Trailing stop %
R2G_TRAIL_ACTIVATE_PCT = 4.0     # Start trailing at this % gain
R2G_PARTIAL_SELL_PCT = 0.0       # Partial sell % at target1 (0 = disabled)
R2G_PARTIAL_TARGET_PCT = 5.0     # Partial target (if partial enabled)
R2G_TIME_LIMIT_MINUTES = 60      # Max time in trade

# --- SHARED CONFIG ---
EOD_EXIT_MINUTES = 15
FULL_BALANCE_SIZING = True


def _compute_vwap(mh):
    """Compute cumulative VWAP from market-hour candle DataFrame."""
    h = mh["High"].values.astype(float)
    l = mh["Low"].values.astype(float)
    c = mh["Close"].values.astype(float)
    v = mh["Volume"].values.astype(float)
    typical = (h + l + c) / 3.0
    cum_tp_vol = np.cumsum(typical * v)
    cum_vol = np.cumsum(v)
    cum_vol[cum_vol == 0] = 1e-9
    return cum_tp_vol / cum_vol


def simulate_day_r2g(picks, cash, cash_account=False):
    """Simulate R2G strategy for one day.
    Returns: (states, cash, unsettled)
    """
    unsettled = 0.0

    def _receive_proceeds(amount):
        nonlocal cash, unsettled
        if cash_account:
            unsettled += amount
        else:
            cash += amount

    # Build unified timestamp index
    all_timestamps = set()
    for pick in picks:
        mh = pick.get("market_hour_candles")
        if mh is not None and len(mh) > 0:
            all_timestamps.update(mh.index.tolist())
    all_timestamps = sorted(all_timestamps)

    if not all_timestamps:
        return [], cash, unsettled

    # Initialize states - all gap-up stocks are candidates
    states = []
    for pick in picks:
        ticker = pick["ticker"]
        if pick["gap_pct"] < R2G_MIN_GAP_PCT:
            continue

        mh = pick.get("market_hour_candles")
        if mh is None or len(mh) < 5:
            continue

        vwap = _compute_vwap(mh)
        states.append({
            "ticker": ticker,
            "gap_pct": pick["gap_pct"],
            "premarket_high": pick["premarket_high"],
            "pm_volume": pick.get("pm_volume", 0),
            "mh": mh,
            "vwap": vwap,
            # Tracking
            "candle_count": 0,
            "day_open": None,           # Day open price (first candle open)
            "went_red": False,          # Has price dipped below day open?
            "lowest_below_open": 0.0,   # Deepest dip below open (% terms)
            "strategy": "R2G",
            "signal": False,
            "signal_price": None,
            # Position
            "entry_price": None,
            "entry_time": None,
            "exit_price": None,
            "exit_time": None,
            "exit_reason": None,
            "shares": 0,
            "position_cost": 0.0,
            "pnl": 0.0,
            "vol_capped": False,
            "done": False,
            # R2G exit management
            "r2g_highest_since_entry": 0.0,
            "r2g_trailing_active": False,
            "r2g_partial_taken": False,
            "r2g_partial_proceeds": 0.0,
        })

    for ts in all_timestamps:
        entry_candidates = []

        for st in states:
            if st["done"]:
                continue
            if ts not in st["mh"].index:
                continue

            candle = st["mh"].loc[ts]
            c_open = float(candle["Open"])
            c_high = float(candle["High"])
            c_low = float(candle["Low"])
            c_close = float(candle["Close"])
            c_vol = float(candle["Volume"])

            # --- IN POSITION: check exits ---
            if st["entry_price"] is not None:
                try:
                    ts_et = ts.astimezone(ET_TZ)
                except Exception:
                    ts_et = ts
                minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)

                entry_et = st["entry_time"]
                try:
                    entry_et = entry_et.astimezone(ET_TZ)
                except Exception:
                    pass
                minutes_in_trade = (ts_et.hour * 60 + ts_et.minute) - (entry_et.hour * 60 + entry_et.minute)

                def _close_position(st, price, reason, ts_now):
                    sell_price = price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["shares"] * sell_price
                    partial_procs = st.get("r2g_partial_proceeds", 0)
                    st["pnl"] = partial_procs + proceeds - st["position_cost"]
                    st["exit_price"] = price
                    st["exit_time"] = ts_now
                    st["exit_reason"] = reason
                    st["entry_price"] = None
                    st["shares"] = 0
                    st["done"] = True
                    _receive_proceeds(proceeds)

                # EOD forced exit
                if minutes_to_close <= EOD_EXIT_MINUTES:
                    _close_position(st, c_close, "EOD_CLOSE", ts)
                    continue

                # Update highest since entry
                if c_high > st["r2g_highest_since_entry"]:
                    st["r2g_highest_since_entry"] = c_high

                # 1. Trailing stop (if active)
                if st["r2g_trailing_active"]:
                    trail_stop = st["r2g_highest_since_entry"] * (1 - R2G_TRAIL_PCT / 100)
                    if c_low <= trail_stop:
                        _close_position(st, trail_stop, "TRAIL", ts)
                        continue
                else:
                    # 2. Hard stop (before trail activates)
                    stop_price = st["entry_price"] * (1 - R2G_STOP_PCT / 100)
                    if c_low <= stop_price:
                        _close_position(st, stop_price, "STOP", ts)
                        continue

                # 3. Activate trailing stop
                if not st["r2g_trailing_active"]:
                    unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                    if unrealized_pct >= R2G_TRAIL_ACTIVATE_PCT:
                        st["r2g_trailing_active"] = True

                # 4. Partial sell (if enabled)
                if not st["r2g_partial_taken"] and R2G_PARTIAL_SELL_PCT > 0:
                    partial_tgt = st["entry_price"] * (1 + R2G_PARTIAL_TARGET_PCT / 100)
                    if c_high >= partial_tgt:
                        partial_shares = st["shares"] * (R2G_PARTIAL_SELL_PCT / 100)
                        sell_price = partial_tgt * (1 - SLIPPAGE_PCT / 100)
                        partial_proceeds = partial_shares * sell_price
                        st["r2g_partial_proceeds"] += partial_proceeds
                        st["shares"] -= partial_shares
                        st["r2g_partial_taken"] = True
                        _receive_proceeds(partial_proceeds)
                        if st["shares"] <= 0.001:
                            st["pnl"] = st["r2g_partial_proceeds"] - st["position_cost"]
                            st["exit_price"] = partial_tgt
                            st["exit_time"] = ts
                            st["exit_reason"] = "TARGET"
                            st["entry_price"] = None
                            st["shares"] = 0
                            st["done"] = True
                            continue

                # 5. Full profit target
                target_price = st["entry_price"] * (1 + R2G_TARGET_PCT / 100)
                if c_high >= target_price:
                    _close_position(st, target_price, "TARGET", ts)
                    continue

                # 6. Time stop
                if minutes_in_trade >= R2G_TIME_LIMIT_MINUTES:
                    _close_position(st, c_close, "TIME_STOP", ts)
                    continue

                continue

            # --- NOT IN POSITION: signal detection ---
            st["candle_count"] += 1

            # Track day open price from first candle
            if st["candle_count"] == 1:
                st["day_open"] = c_open
                # Check if candle 1 goes red (close < open)
                if c_close < c_open:
                    st["went_red"] = True
                    dip_pct = (c_open - c_low) / c_open * 100
                    st["lowest_below_open"] = dip_pct
                # Even if candle 1 is green, low might have gone below open
                if c_low < c_open:
                    st["went_red"] = True
                    dip_pct = (c_open - c_low) / c_open * 100
                    st["lowest_below_open"] = max(st["lowest_below_open"], dip_pct)
                continue  # Don't enter on candle 1

            day_open = st["day_open"]
            if day_open is None or day_open <= 0:
                continue

            # Track if price goes below day open (any candle)
            if c_low < day_open:
                st["went_red"] = True
                dip_pct = (day_open - c_low) / day_open * 100
                st["lowest_below_open"] = max(st["lowest_below_open"], dip_pct)

            # Too early or too late
            if st["candle_count"] < R2G_EARLIEST_CANDLE:
                continue
            if st["candle_count"] > R2G_LATEST_CANDLE:
                st["done"] = True
                continue

            # --- ENTRY SIGNAL: Red-to-Green reclaim ---

            # 1. Must have gone red first
            if not st["went_red"]:
                continue

            # 2. Dip must be within min/max range
            if st["lowest_below_open"] < R2G_MIN_DIP_PCT:
                continue
            if st["lowest_below_open"] > R2G_MAX_DIP_PCT:
                st["done"] = True  # Too deep, broken stock
                continue

            # 3. Current candle must reclaim above day open
            if R2G_REQUIRE_CLOSE_ABOVE_OPEN:
                if c_close <= day_open:
                    continue
            else:
                if c_high <= day_open:
                    continue

            # 4. Green body check (reclaim candle should be green with min body)
            if c_open > 0:
                body_pct = (c_close / c_open - 1) * 100
            else:
                body_pct = 0
            if body_pct < R2G_MIN_RECLAIM_BODY_PCT:
                continue

            # 5. Volume surge check
            candle_idx = st["candle_count"] - 1
            if candle_idx >= 5:
                recent_vols = st["mh"].iloc[candle_idx-5:candle_idx]["Volume"].values.astype(float)
                avg_vol = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 0
            else:
                recent_vols = st["mh"].iloc[:candle_idx]["Volume"].values.astype(float)
                avg_vol = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 0

            if avg_vol > 0 and c_vol < avg_vol * R2G_VOL_SURGE_MULT:
                continue

            # 6. VWAP filter (optional)
            if R2G_REQUIRE_ABOVE_VWAP and candle_idx < len(st["vwap"]):
                if c_close < st["vwap"][candle_idx]:
                    continue

            # All conditions met — signal!
            st["signal"] = True
            st["signal_price"] = c_close
            entry_candidates.append(st)

        # --- ALLOCATION: priority by gap % (highest gap first) ---
        entry_candidates.sort(key=lambda s: -s["gap_pct"])

        tickers_taken = set()
        for st in entry_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue
            if st["ticker"] in tickers_taken:
                continue

            try:
                ts_et = ts.astimezone(ET_TZ)
            except Exception:
                ts_et = ts
            mins_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
            if mins_to_close <= EOD_EXIT_MINUTES:
                continue

            if cash < 100:
                continue
            trade_size = cash  # FULL_BALANCE_SIZING

            fill_price = st["signal_price"]
            if fill_price is None or fill_price <= 0:
                continue

            # Volume cap
            if VOL_CAP_PCT > 0:
                pre_entry = st["mh"].loc[st["mh"].index <= ts]
                vol_shares = pre_entry["Volume"].sum() if len(pre_entry) > 0 else 0
                dollar_vol = fill_price * vol_shares
                vol_limit = dollar_vol * (VOL_CAP_PCT / 100)
                if vol_limit > 0 and trade_size > vol_limit:
                    trade_size = vol_limit
                    st["vol_capped"] = True
                if trade_size < 50:
                    continue

            entry_price = fill_price * (1 + SLIPPAGE_PCT / 100)
            st["entry_price"] = entry_price
            st["_orig_entry_price"] = entry_price
            st["entry_time"] = ts
            st["position_cost"] = trade_size
            st["shares"] = trade_size / entry_price
            st["_orig_shares"] = trade_size / entry_price
            st["r2g_highest_since_entry"] = entry_price
            cash -= trade_size
            tickers_taken.add(st["ticker"])

    # EOD: close remaining
    for st in states:
        if st["entry_price"] is not None and st["shares"] > 0:
            last_ts = st["mh"].index[-1]
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["shares"] * sell_price
            partial_procs = st.get("r2g_partial_proceeds", 0)
            st["pnl"] = partial_procs + proceeds - st["position_cost"]
            st["exit_price"] = last_close
            st["exit_time"] = last_ts
            st["exit_reason"] = "EOD_CLOSE"
            st["entry_price"] = None
            st["shares"] = 0
            st["done"] = True
            _receive_proceeds(proceeds)

    return states, cash, unsettled


# --- MAIN ---
if __name__ == "__main__":
    no_charts = "--no-charts" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-charts"]
    data_dirs = args if args else ["stored_data_combined"]

    print(f"Strategy R2G: Red-to-Green Move Backtest")
    print(f"{'='*70}")
    print(f"  Min gap:            {R2G_MIN_GAP_PCT}%")
    print(f"  Dip range:          {R2G_MIN_DIP_PCT}% - {R2G_MAX_DIP_PCT}% below open")
    print(f"  Entry window:       candle {R2G_EARLIEST_CANDLE} - {R2G_LATEST_CANDLE}")
    print(f"  Vol surge:          >= {R2G_VOL_SURGE_MULT}x avg last 5")
    print(f"  Reclaim body:       >= {R2G_MIN_RECLAIM_BODY_PCT}% green")
    print(f"  Close above open:   {R2G_REQUIRE_CLOSE_ABOVE_OPEN}")
    print(f"  Above VWAP:         {R2G_REQUIRE_ABOVE_VWAP}")
    print(f"  Target:             +{R2G_TARGET_PCT}%")
    print(f"  Stop:               -{R2G_STOP_PCT}%")
    print(f"  Trail:              {R2G_TRAIL_PCT}% (activates at +{R2G_TRAIL_ACTIVATE_PCT}%)")
    if R2G_PARTIAL_SELL_PCT > 0:
        print(f"  Partial sell:       {R2G_PARTIAL_SELL_PCT}% at +{R2G_PARTIAL_TARGET_PCT}%")
    print(f"  Time limit:         {R2G_TIME_LIMIT_MINUTES}m")
    print(f"  Data: {data_dirs}")
    print(f"{'='*70}\n")

    print("Loading data...")
    all_dates, daily_picks = load_all_picks(data_dirs)
    all_dates = [d for d in all_dates if "2024-01-01" <= d <= "2026-02-28"]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")

    # Count eligible picks (gap >= min)
    total_picks = 0
    eligible_picks = 0
    for d in all_dates:
        for p in daily_picks.get(d, []):
            total_picks += 1
            if p["gap_pct"] >= R2G_MIN_GAP_PCT:
                eligible_picks += 1
    print(f"  Total picks: {total_picks}, R2G-eligible (gap >= {R2G_MIN_GAP_PCT}%): "
          f"{eligible_picks} ({eligible_picks/max(total_picks,1)*100:.1f}%)\n")

    cash = float(STARTING_CASH)
    unsettled = 0.0
    all_results = []

    print(f"{'Date':<12} {'Trades':>6} {'Win':>4} {'Loss':>5} "
          f"{'Day P&L':>12} {'Balance':>14}")
    print("-" * 60)

    for d in all_dates:
        cash += unsettled
        unsettled = 0.0

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        states, cash, unsettled = simulate_day_r2g(picks, cash, cash_account)

        day_pnl = 0.0
        day_trades = 0
        day_wins = 0
        day_losses = 0

        for st in states:
            if st["exit_reason"] is not None:
                day_trades += 1
                day_pnl += st["pnl"]
                if st["pnl"] > 0:
                    day_wins += 1
                else:
                    day_losses += 1

        equity = cash + unsettled

        if day_trades > 0:
            print(f"{d:<12} {day_trades:>6} {day_wins:>4} {day_losses:>5} "
                  f"${day_pnl:>+11,.0f} ${equity:>13,.0f}")

            for st in states:
                if st["exit_reason"] is not None:
                    pct = (st["pnl"] / st["position_cost"] * 100) if st["position_cost"] > 0 else 0
                    reason_short = {"TARGET": "T", "TIME_STOP": "TS", "EOD_CLOSE": "EOD",
                                    "STOP": "SL", "TRAIL": "TR"}.get(
                        st["exit_reason"], st["exit_reason"][:3])
                    vc_tag = " VC" if st.get("vol_capped") else ""
                    dip = st["lowest_below_open"]
                    print(f"  -> {st['ticker']:<6} {reason_short:<3}  "
                          f"${st['pnl']:>+10,.0f}  ({pct:>+6.2f}%)  "
                          f"gap={st['gap_pct']:.0f}% dip={dip:.1f}%{vc_tag}")

        all_results.append({
            "date": d, "states": states,
            "day_pnl": day_pnl, "equity": equity,
            "trades": day_trades, "wins": day_wins, "losses": day_losses,
        })

    # --- Summary ---
    final_equity = cash + unsettled
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    daily_pnls = [r["day_pnl"] for r in all_results if r["trades"] > 0]
    green = sum(1 for p in daily_pnls if p > 0) if daily_pnls else 0
    red = sum(1 for p in daily_pnls if p <= 0) if daily_pnls else 0
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if daily_pnls and np.std(daily_pnls) > 0 else 0

    all_trade_pnls = []
    all_exits = {}
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None:
                all_trade_pnls.append(st["pnl"])
                all_exits[st["exit_reason"]] = all_exits.get(st["exit_reason"], 0) + 1

    avg_win = np.mean([p for p in all_trade_pnls if p > 0]) if total_wins > 0 else 0
    avg_loss = np.mean([p for p in all_trade_pnls if p <= 0]) if total_losses > 0 else 0

    all_trade_pcts = []
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None and st["position_cost"] > 0:
                all_trade_pcts.append(st["pnl"] / st["position_cost"] * 100)

    daily_eq_pcts = []
    for i in range(1, len(all_results)):
        prev_eq = all_results[i-1]["equity"]
        curr_eq = all_results[i]["equity"]
        if prev_eq > 0:
            daily_eq_pcts.append((curr_eq / prev_eq - 1) * 100)

    print(f"\n{'='*70}")
    print(f"  STRATEGY R2G: RED-TO-GREEN MOVE SUMMARY")
    print(f"{'='*70}")
    print(f"  Starting Cash:    ${STARTING_CASH:,}")
    print(f"  Ending Equity:    ${final_equity:,.0f}  ({(final_equity/STARTING_CASH - 1)*100:+.1f}%)")
    print(f"  Trading Days:     {len(all_dates)}")
    print(f"  Total Trades:     {total_trades}")
    print(f"    Winners:        {total_wins} ({total_wins/max(total_trades,1)*100:.1f}%)")
    print(f"    Losers:         {total_losses}")
    print(f"  Avg Win:          ${avg_win:+,.0f}")
    print(f"  Avg Loss:         ${avg_loss:+,.0f}")
    pf = abs(avg_win * total_wins / (avg_loss * total_losses)) if total_losses > 0 and avg_loss != 0 else 0
    print(f"  Profit Factor:    {pf:.2f}" if pf > 0 else "")

    if daily_eq_pcts:
        print(f"\n  Daily % Change:")
        print(f"    Avg Daily:      {np.mean(daily_eq_pcts):+.2f}%")
        print(f"    Median Daily:   {np.median(daily_eq_pcts):+.2f}%")
        print(f"    Best Day:       {max(daily_eq_pcts):+.2f}%")
        print(f"    Worst Day:      {min(daily_eq_pcts):+.2f}%")
        print(f"    Std Dev:        {np.std(daily_eq_pcts):.2f}%")

    if all_trade_pcts:
        win_pcts = [p for p in all_trade_pcts if p > 0]
        loss_pcts = [p for p in all_trade_pcts if p <= 0]
        print(f"\n  Per-Trade % Return:")
        print(f"    Avg Trade:      {np.mean(all_trade_pcts):+.2f}%")
        print(f"    Median Trade:   {np.median(all_trade_pcts):+.2f}%")
        if win_pcts:
            print(f"    Avg Winner:     {np.mean(win_pcts):+.2f}%")
        if loss_pcts:
            print(f"    Avg Loser:      {np.mean(loss_pcts):+.2f}%")

    print(f"\n  Exit Reasons:")
    for reason, count in sorted(all_exits.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20} {count:>4} ({count/max(total_trades,1)*100:.1f}%)")

    if daily_pnls:
        print(f"\n  Green Days:       {green}/{len(daily_pnls)} ({green/len(daily_pnls)*100:.1f}%)")
        print(f"  Red Days:         {red}/{len(daily_pnls)}")
        print(f"  Best Day:         ${max(daily_pnls):+,.0f}")
        print(f"  Worst Day:        ${min(daily_pnls):+,.0f}")
        print(f"  Avg P&L/Day:      ${np.mean(daily_pnls):+,.0f}")
        print(f"  Sharpe (ann.):    {sharpe:.2f}")

    # Dip depth breakdown
    dip_buckets = {"1-3%": [], "3-5%": [], "5-8%": [], "8%+": []}
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None:
                dip = st["lowest_below_open"]
                if dip < 3:
                    dip_buckets["1-3%"].append(st["pnl"])
                elif dip < 5:
                    dip_buckets["3-5%"].append(st["pnl"])
                elif dip < 8:
                    dip_buckets["5-8%"].append(st["pnl"])
                else:
                    dip_buckets["8%+"].append(st["pnl"])
    print(f"\n  Dip Depth Breakdown:")
    for bucket, pnls in dip_buckets.items():
        if pnls:
            w = sum(1 for p in pnls if p > 0)
            print(f"    {bucket:>5}: {len(pnls)} trades, {w/len(pnls)*100:.0f}% WR, "
                  f"${sum(pnls):+,.0f} total, ${np.mean(pnls):+,.0f} avg")

    # Entry candle distribution
    entry_candles = []
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None and st["entry_time"] is not None:
                entry_candles.append(st["candle_count"])
    if entry_candles:
        print(f"\n  Entry Candle Distribution:")
        print(f"    Min: {min(entry_candles)}, Max: {max(entry_candles)}, "
              f"Median: {np.median(entry_candles):.0f}, Mean: {np.mean(entry_candles):.1f}")

    # Gap size breakdown
    gap_buckets = {"10-20%": [], "20-40%": [], "40-60%": [], "60%+": []}
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None:
                g = st["gap_pct"]
                if g < 20:
                    gap_buckets["10-20%"].append(st["pnl"])
                elif g < 40:
                    gap_buckets["20-40%"].append(st["pnl"])
                elif g < 60:
                    gap_buckets["40-60%"].append(st["pnl"])
                else:
                    gap_buckets["60%+"].append(st["pnl"])
    print(f"\n  Gap Size Breakdown:")
    for bucket, pnls in gap_buckets.items():
        if pnls:
            w = sum(1 for p in pnls if p > 0)
            print(f"    {bucket:>6}: {len(pnls)} trades, {w/len(pnls)*100:.0f}% WR, "
                  f"${sum(pnls):+,.0f} total, ${np.mean(pnls):+,.0f} avg")

    # --- STRESS TEST ---
    print(f"\n  {'='*50}")
    print(f"  STRESS TEST")
    print(f"  {'='*50}")

    equities = [r["equity"] for r in all_results]
    peak = equities[0]
    max_dd_dollar = 0.0
    max_dd_pct = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = peak - eq
        dd_pct = dd / peak if peak > 0 else 0
        if dd > max_dd_dollar:
            max_dd_dollar = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    print(f"  Max Drawdown:     ${max_dd_dollar:,.0f} ({max_dd_pct*100:.1f}%)")

    streak = 0
    max_streak = 0
    for pnl in daily_pnls:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"  Max Losing Streak: {max_streak} days")

    if daily_pnls:
        sorted_pnls = sorted(daily_pnls, reverse=True)
        n_remove = max(1, int(len(daily_pnls) * 0.10))
        remaining = sorted_pnls[n_remove:]
        rem_total = sum(remaining)
        rem_std = np.std(remaining) if len(remaining) > 1 else 1.0
        rem_sharpe = (np.mean(remaining) / rem_std) * np.sqrt(252) if rem_std > 0 else 0
        print(f"  Remove Top 10%:   ${rem_total:+,.0f} (Sharpe {rem_sharpe:.2f}) "
              f"{'PASS' if rem_total > 0 else 'FAIL'}")

    wins_pnls = [p for p in daily_pnls if p > 0]
    loss_pnls_abs = [abs(p) for p in daily_pnls if p < 0]
    if wins_pnls and loss_pnls_abs:
        avg_w = np.mean(wins_pnls)
        avg_l = np.mean(loss_pnls_abs)
        p_w = len(wins_pnls) / len(daily_pnls)
        b = avg_w / avg_l if avg_l > 0 else float('inf')
        kelly = (b * p_w - (1 - p_w)) / b if b > 0 else 0
        print(f"  Kelly Criterion:  {kelly*100:.1f}% {'PASS' if kelly > 0 else 'FAIL'}")
    else:
        kelly = 0

    vc_count = sum(1 for r in all_results for s in r["states"]
                   if s["exit_reason"] is not None and s.get("vol_capped"))
    if vc_count > 0:
        print(f"  Vol-Capped Trades: {vc_count}")

    print(f"{'='*70}")

    # --- CHARTS ---
    if no_charts:
        print("\n  Skipping charts (--no-charts)")
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("charts", f"r2g_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nGenerating charts -> {run_dir}/")

        def _to_et(ts_val):
            try:
                return ts_val.astimezone(ET_TZ)
            except Exception:
                return ts_val

        COLORS_R2G = ["#2196F3", "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB"]
        DAYS_PER_PAGE = 5

        trade_results = [r for r in all_results if r["trades"] > 0]

        num_pages = math.ceil(len(trade_results) / DAYS_PER_PAGE) if trade_results else 0
        for page in range(num_pages):
            start = page * DAYS_PER_PAGE
            end = min(start + DAYS_PER_PAGE, len(trade_results))
            page_results = trade_results[start:end]
            n_rows = len(page_results)

            fig, axes = plt.subplots(
                n_rows, 2, figsize=(20, 4.5 * n_rows),
                gridspec_kw={"width_ratios": [2.5, 1]},
            )
            if n_rows == 1:
                axes = [axes]

            fig.suptitle(
                f"Strategy R2G: Red-to-Green Move - Page {page+1}/{num_pages}\n"
                f"Gap >= {R2G_MIN_GAP_PCT}% | Dip {R2G_MIN_DIP_PCT}-{R2G_MAX_DIP_PCT}% | "
                f"Vol {R2G_VOL_SURGE_MULT}x | "
                f"Target: +{R2G_TARGET_PCT}% | Stop: -{R2G_STOP_PCT}% | "
                f"Trail: {R2G_TRAIL_PCT}% (at +{R2G_TRAIL_ACTIVATE_PCT}%)",
                fontsize=10, fontweight="bold", y=1.01,
            )

            for i, res in enumerate(page_results):
                row_axes = axes[i] if n_rows > 1 else axes[0]
                ax_price, ax_pnl = row_axes[0], row_axes[1]

                traded = [s for s in res["states"] if s["exit_reason"] is not None]
                if traded:
                    for si, st in enumerate(traded):
                        mh = st["mh"]
                        if mh.index.tz is not None:
                            et_times = mh.index.tz_convert(ET_TZ)
                        else:
                            et_times = mh.index.tz_localize("UTC").tz_convert(ET_TZ)

                        color = COLORS_R2G[si % len(COLORS_R2G)]
                        first_candle_close = float(mh.iloc[0]["Close"])
                        pct_change = (mh["Close"].values.astype(float) / first_candle_close - 1) * 100

                        vc_tag = " VC" if st.get("vol_capped") else ""
                        cost_k = st["position_cost"] / 1000
                        dip = st["lowest_below_open"]
                        label = f"{st['ticker']} ${cost_k:.0f}K (gap {st['gap_pct']:.0f}%, dip {dip:.1f}%){vc_tag}"
                        ax_price.plot(et_times, pct_change, color=color, linewidth=1.2,
                                      label=label, alpha=0.85)

                        # Day open line (the key R2G level)
                        if st["day_open"] and first_candle_close > 0:
                            open_pct = (st["day_open"] / first_candle_close - 1) * 100
                            ax_price.axhline(y=open_pct, color=color, linestyle="--",
                                             alpha=0.4, linewidth=0.8)

                        # BUY marker
                        if st["entry_time"] is not None:
                            et_buy = _to_et(st["entry_time"])
                            orig_entry = st.get("_orig_entry_price") or (st["position_cost"] / max(st.get("_orig_shares", 1), 0.001))
                            buy_pct = (orig_entry / first_candle_close - 1) * 100
                            ax_price.plot(et_buy, buy_pct, marker="^", color=color,
                                          markersize=10, zorder=5)
                            ax_price.annotate(
                                "BUY", xy=(et_buy, buy_pct), xytext=(0, 12),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=7, fontweight="bold", color=color,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          ec=color, alpha=0.8, lw=0.5),
                            )

                        # SELL marker
                        if st["exit_time"] is not None and st["exit_price"] is not None:
                            et_sell = _to_et(st["exit_time"])
                            sell_pct = (st["exit_price"] / first_candle_close - 1) * 100
                            is_win = st["pnl"] > 0
                            marker = "v" if not is_win else "s"
                            sell_color = "#4CAF50" if is_win else "#f44336"
                            ax_price.plot(et_sell, sell_pct, marker=marker, color=sell_color,
                                          markersize=10, zorder=5,
                                          markeredgecolor="white", markeredgewidth=1)
                            reason_short = {"TARGET": "T", "STOP": "SL",
                                            "EOD_CLOSE": "EOD", "TIME_STOP": "TS",
                                            "TRAIL": "TR"}.get(
                                st["exit_reason"], st["exit_reason"][:3])
                            ax_price.annotate(
                                reason_short, xy=(et_sell, sell_pct),
                                xytext=(0, -14 if sell_pct >= 0 else 12),
                                textcoords="offset points", ha="center",
                                va="top" if sell_pct >= 0 else "bottom",
                                fontsize=7, fontweight="bold", color=sell_color,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          ec=sell_color, alpha=0.8, lw=0.5),
                            )

                    ax_price.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                    ax_price.axhline(y=R2G_TARGET_PCT, color="#4CAF50", linestyle=":", alpha=0.35,
                                     label=f"+{R2G_TARGET_PCT:.0f}% target")
                    ax_price.axhline(y=-R2G_STOP_PCT, color="#f44336", linestyle=":", alpha=0.4,
                                     label=f"-{R2G_STOP_PCT:.0f}% stop")
                    ax_price.set_title(f"{res['date']} - Price (% from 1st candle close)",
                                       fontsize=10, fontweight="bold")
                    ax_price.set_ylabel("% from Entry", fontsize=8)
                    ax_price.legend(fontsize=6, loc="upper left", ncol=2)
                    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p", tz=ET_TZ))
                    ax_price.tick_params(axis="x", rotation=30, labelsize=7)
                    ax_price.grid(alpha=0.3)

                    # P&L bar chart
                    tickers = []
                    for s in traded:
                        vc = " VC" if s.get("vol_capped") else ""
                        dip = s["lowest_below_open"]
                        tickers.append(f"{s['ticker']} (dip {dip:.1f}%){vc}")
                    pnls = [s["pnl"] for s in traded]
                    pct_profits = [(s["pnl"] / s["position_cost"] * 100) if s["position_cost"] > 0 else 0 for s in traded]
                    reasons = [s["exit_reason"] for s in traded]
                    bar_colors = ["#4CAF50" if p > 0 else "#f44336" for p in pnls]
                    y_pos = range(len(tickers))
                    bars = ax_pnl.barh(y_pos, pnls, color=bar_colors, edgecolor="white", height=0.6)
                    ax_pnl.set_yticks(y_pos)
                    ax_pnl.set_yticklabels([f"{t} ({r[:3]})" for t, r in zip(tickers, reasons)], fontsize=7)
                    ax_pnl.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
                    ax_pnl.invert_yaxis()
                    for j, (bar, pnl, pct) in enumerate(zip(bars, pnls, pct_profits)):
                        x_pos = bar.get_width()
                        align = "left" if pnl >= 0 else "right"
                        offset = 5 if pnl >= 0 else -5
                        ax_pnl.annotate(f"${pnl:+.0f} ({pct:+.1f}%)", xy=(x_pos, j), xytext=(offset, 0),
                                        textcoords="offset points", ha=align, va="center",
                                        fontsize=7, fontweight="bold", color=bar_colors[j])
                    day_total = sum(pnls)
                    tc = "#4CAF50" if day_total >= 0 else "#f44336"
                    bal = res["equity"]
                    ax_pnl.set_title(f"P&L: ${day_total:+,.0f} | Bal: ${bal:,.0f}",
                                     fontsize=10, fontweight="bold", color=tc)
                    ax_pnl.set_xlabel("P&L ($)", fontsize=8)
                    ax_pnl.grid(alpha=0.3, axis="x")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            chart_path = os.path.join(run_dir, f"r2g_page_{page+1:02d}.png")
            plt.savefig(chart_path, dpi=120, bbox_inches="tight")
            plt.close()
            sys.stdout.write(f"\r  Charts: page {page+1}/{num_pages}")
            sys.stdout.flush()

        if num_pages > 0:
            print(f"\r  {num_pages} chart pages saved to {run_dir}/          ")

        # --- SUMMARY CHART ---
        fig2 = plt.figure(figsize=(20, 14))
        gs = fig2.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 0.8],
                               height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.3)
        fig2.suptitle(
            f"Strategy R2G: Red-to-Green Summary: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)",
            fontsize=16, fontweight="bold",
        )

        # 1. Equity curve
        ax = fig2.add_subplot(gs[0, :])
        equities = [r["equity"] for r in all_results]
        dates_list = [r["date"] for r in all_results]
        ax.plot(range(len(dates_list)), equities, color="#2196F3", linewidth=2, label="Strategy R2G")
        ax.fill_between(range(len(dates_list)), STARTING_CASH, equities,
                        where=[e >= STARTING_CASH for e in equities], alpha=0.15, color="#4CAF50")
        ax.fill_between(range(len(dates_list)), STARTING_CASH, equities,
                        where=[e < STARTING_CASH for e in equities], alpha=0.15, color="#f44336")
        ax.axhline(y=STARTING_CASH, color="gray", linestyle="--", alpha=0.5,
                   label=f"Start: ${STARTING_CASH:,}")

        milestones = [(50_000, "$50K", "#FF9800"), (100_000, "$100K", "#E91E63"),
                      (1_000_000, "$1M", "#9C27B0"), (10_000_000, "$10M", "#F44336")]
        y_max = max(equities)
        for mi, (mlevel, mlabel, mcolor) in enumerate(milestones):
            if y_max >= mlevel:
                ax.axhline(y=mlevel, color=mcolor, linestyle="--", alpha=0.3, linewidth=1)
                for midx, eq in enumerate(equities):
                    if eq >= mlevel:
                        label_y = y_max * (0.85 + mi * 0.04)
                        ax.plot([midx, midx], [mlevel, label_y], color=mcolor,
                                linestyle="-", linewidth=1, alpha=0.6)
                        ax.plot(midx, mlevel, marker="o", color=mcolor, markersize=6, zorder=5)
                        ax.annotate(f"{mlabel}\n{dates_list[midx]}",
                                    xy=(midx, label_y), xytext=(0, 6),
                                    textcoords="offset points", ha="center",
                                    fontsize=8, fontweight="bold", color=mcolor)
                        break

        tick_interval = max(1, len(dates_list) // 20)
        ax.set_xticks(range(0, len(dates_list), tick_interval))
        ax.set_xticklabels([dates_list[i] for i in range(0, len(dates_list), tick_interval)],
                           rotation=45, fontsize=7)
        ax.set_ylabel("Equity ($)", fontsize=10)
        ax.set_title(f"Equity Curve: ${STARTING_CASH:,} -> ${final_equity:,.0f}", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        if y_max > 100_000:
            ax.set_yscale("log")

        # 2. Daily P&L distribution
        ax2 = fig2.add_subplot(gs[1, 0])
        if daily_pnls:
            colors_hist = ["#4CAF50" if p > 0 else "#f44336" for p in sorted(daily_pnls)]
            ax2.hist(daily_pnls, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
            ax2.axvline(x=0, color="red", linestyle="-", alpha=0.5)
            ax2.axvline(x=np.mean(daily_pnls), color="orange", linestyle="--",
                        label=f"Avg: ${np.mean(daily_pnls):+,.0f}")
            ax2.set_title("Daily P&L Distribution", fontsize=10)
            ax2.set_xlabel("Daily P&L ($)")
            ax2.legend(fontsize=7)
            ax2.grid(alpha=0.3)

        # 3. Win rate by month
        ax3 = fig2.add_subplot(gs[1, 1])
        monthly_wr = {}
        for r in all_results:
            month = r["date"][:7]
            if r["trades"] > 0:
                if month not in monthly_wr:
                    monthly_wr[month] = {"wins": 0, "total": 0, "pnl": 0}
                monthly_wr[month]["wins"] += r["wins"]
                monthly_wr[month]["total"] += r["trades"]
                monthly_wr[month]["pnl"] += r["day_pnl"]
        if monthly_wr:
            months = sorted(monthly_wr.keys())
            wrs = [monthly_wr[m]["wins"] / monthly_wr[m]["total"] * 100 for m in months]
            bar_colors = ["#4CAF50" if wr >= 50 else "#f44336" for wr in wrs]
            ax3.bar(range(len(months)), wrs, color=bar_colors, alpha=0.8, edgecolor="white")
            ax3.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
            ax3.set_xticks(range(len(months)))
            ax3.set_xticklabels(months, rotation=45, fontsize=6)
            ax3.set_ylabel("Win Rate %")
            ax3.set_title("Monthly Win Rate", fontsize=10)
            ax3.grid(alpha=0.3)

        # 4. Exit reason breakdown
        ax4 = fig2.add_subplot(gs[1, 2])
        if all_exits:
            reasons = list(all_exits.keys())
            counts = [all_exits[r] for r in reasons]
            reason_colors = {"TARGET": "#4CAF50", "STOP": "#f44336", "TRAIL": "#FF9800",
                             "EOD_CLOSE": "#9E9E9E", "TIME_STOP": "#2196F3"}
            colors = [reason_colors.get(r, "#607D8B") for r in reasons]
            ax4.pie(counts, labels=reasons, colors=colors, autopct="%1.0f%%",
                    startangle=90, textprops={"fontsize": 8})
            ax4.set_title("Exit Reasons", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        summary_path = os.path.join(run_dir, "r2g_summary.png")
        plt.savefig(summary_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Summary chart saved to {summary_path}")

    print(f"\nDone.")
