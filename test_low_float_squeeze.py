"""
Strategy L: Low Float Squeeze Backtest (standalone)
====================================================
Gap-up + low float + HOD break with volume surge -> squeeze play.

Low-float stocks (< 10M shares) that gap up have extreme intraday ranges:
- 61% move 50%+ from open
- 30% double from open
This strategy targets the squeeze: buying HOD breaks with volume confirmation.

Entry: New HOD + volume surge + price acceleration + (optional) above VWAP
Exit:  Trailing stop + partial sell + hard stop + time stop + EOD

Usage:
  python test_low_float_squeeze.py
  python test_low_float_squeeze.py --no-charts
"""

import os
import sys
import json
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

# --- FLOAT DATA ---
FLOAT_DATA = {}
_float_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "float_data.json")
if os.path.exists(_float_path):
    with open(_float_path) as _f:
        _raw = json.load(_f)
    for _tk, _v in _raw.items():
        if isinstance(_v, dict) and _v.get("floatShares"):
            FLOAT_DATA[_tk] = _v["floatShares"]

# --- STRATEGY L CONFIG (optimized trial #486: $78.6M PnL, PF 4.33, 84.9% WR) ---
L_MAX_FLOAT = 15_000_000        # Float shares threshold
L_MIN_GAP_PCT = 30.0            # Minimum gap %
L_EARLIEST_CANDLE = 8           # Don't enter too early
L_LATEST_CANDLE = 115           # Latest possible entry candle
L_HOD_BREAK_REQUIRED = True     # Must break to new HOD for entry
L_VOL_SURGE_MULT = 1.5          # Current candle vol >= Nx avg of last 10
L_MIN_PRICE_ACCEL_PCT = 1.0     # Min green candle body % for entry candle
L_REQUIRE_ABOVE_VWAP = True     # Must be above VWAP at entry
# Float-tiered targets: low float stocks move more, so bigger targets
L_TIER1_FLOAT = 1_000_000       # Ultra-low float boundary
L_TIER2_FLOAT = 5_000_000       # Low float boundary
# Tier 1: < 1M float (ultra-low)
L_TIER1_TARGET1_PCT = 30.0
L_TIER1_TARGET2_PCT = 40.0
# Tier 2: 1-5M float (low)
L_TIER2_TARGET1_PCT = 15.0
L_TIER2_TARGET2_PCT = 40.0
# Tier 3: 5M+ float (mid)
L_TIER3_TARGET1_PCT = 9.0
L_TIER3_TARGET2_PCT = 32.0
L_STOP_PCT = 14.0               # Hard stop (wide for low float volatility)
L_PARTIAL_SELL_PCT = 0.0        # No partial sell — let winners run
L_TRAIL_PCT = 1.0               # Tight trailing stop %
L_TRAIL_ACTIVATE_PCT = 2.0      # Start trailing early at +2%
L_TIME_LIMIT_MINUTES = 70       # Time limit in minutes

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


def _get_tiered_targets(float_shares):
    """Return (target1_pct, target2_pct) based on float size."""
    if float_shares < L_TIER1_FLOAT:
        return L_TIER1_TARGET1_PCT, L_TIER1_TARGET2_PCT
    elif float_shares < L_TIER2_FLOAT:
        return L_TIER2_TARGET1_PCT, L_TIER2_TARGET2_PCT
    else:
        return L_TIER3_TARGET1_PCT, L_TIER3_TARGET2_PCT


def simulate_day_l(picks, cash, cash_account=False):
    """Simulate L strategy for one day.
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

    # Initialize states - only for low float stocks
    states = []
    for pick in picks:
        ticker = pick["ticker"]
        float_shares = FLOAT_DATA.get(ticker)
        if float_shares is None or float_shares > L_MAX_FLOAT:
            continue
        if pick["gap_pct"] < L_MIN_GAP_PCT:
            continue

        mh = pick.get("market_hour_candles")
        if mh is None or len(mh) < 10:
            continue

        vwap = _compute_vwap(mh)
        tgt1_pct, tgt2_pct = _get_tiered_targets(float_shares)
        states.append({
            "ticker": ticker,
            "float_shares": float_shares,
            "gap_pct": pick["gap_pct"],
            "premarket_high": pick["premarket_high"],
            "pm_volume": pick.get("pm_volume", 0),
            "mh": mh,
            "vwap": vwap,
            # Tracking
            "candle_count": 0,
            "open_price": None,
            "running_hod": 0.0,         # Highest high seen so far
            "strategy": "L",
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
            # L exit management
            "l_target1_pct": tgt1_pct,
            "l_target2_pct": tgt2_pct,
            "l_highest_since_entry": 0.0,
            "l_trailing_active": False,
            "l_partial_taken": False,
            "l_partial_proceeds": 0.0,
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
                    partial_procs = st.get("l_partial_proceeds", 0)
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
                if c_high > st["l_highest_since_entry"]:
                    st["l_highest_since_entry"] = c_high

                # 1. Trailing stop (if active)
                if st["l_trailing_active"]:
                    trail_stop = st["l_highest_since_entry"] * (1 - L_TRAIL_PCT / 100)
                    if c_low <= trail_stop:
                        _close_position(st, trail_stop, "TRAIL", ts)
                        continue
                else:
                    # 2. Hard stop (before trail activates)
                    stop_price = st["entry_price"] * (1 - L_STOP_PCT / 100)
                    if c_low <= stop_price:
                        _close_position(st, stop_price, "STOP", ts)
                        continue

                # 3. Activate trailing stop
                if not st["l_trailing_active"]:
                    unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                    if unrealized_pct >= L_TRAIL_ACTIVATE_PCT:
                        st["l_trailing_active"] = True

                # 4. Partial sell at target1 (float-tiered)
                if not st["l_partial_taken"] and L_PARTIAL_SELL_PCT > 0:
                    tgt1 = st["entry_price"] * (1 + st["l_target1_pct"] / 100)
                    if c_high >= tgt1:
                        partial_shares = st["shares"] * (L_PARTIAL_SELL_PCT / 100)
                        sell_price = tgt1 * (1 - SLIPPAGE_PCT / 100)
                        partial_proceeds = partial_shares * sell_price
                        st["l_partial_proceeds"] += partial_proceeds
                        st["shares"] -= partial_shares
                        st["l_partial_taken"] = True
                        _receive_proceeds(partial_proceeds)
                        if st["shares"] <= 0.001:
                            st["pnl"] = st["l_partial_proceeds"] - st["position_cost"]
                            st["exit_price"] = tgt1
                            st["exit_time"] = ts
                            st["exit_reason"] = "TARGET"
                            st["entry_price"] = None
                            st["shares"] = 0
                            st["done"] = True
                            continue

                # 5. Runner target2 (float-tiered)
                tgt2 = st["entry_price"] * (1 + st["l_target2_pct"] / 100)
                if c_high >= tgt2:
                    _close_position(st, tgt2, "TARGET", ts)
                    continue

                # 6. Time stop
                if minutes_in_trade >= L_TIME_LIMIT_MINUTES:
                    _close_position(st, c_close, "TIME_STOP", ts)
                    continue

                continue

            # --- NOT IN POSITION: signal detection ---
            st["candle_count"] += 1

            # Track open price
            if st["candle_count"] == 1:
                st["open_price"] = c_open

            # Update running HOD
            if c_high > st["running_hod"]:
                prev_hod = st["running_hod"]
                st["running_hod"] = c_high
                is_new_hod = prev_hod > 0  # Not first candle
            else:
                is_new_hod = False

            # Too early or too late
            if st["candle_count"] < L_EARLIEST_CANDLE:
                continue
            if st["candle_count"] > L_LATEST_CANDLE:
                st["done"] = True
                continue

            # --- ENTRY SIGNAL: HOD break + volume surge + price acceleration ---
            candle_idx = st["candle_count"] - 1

            # 1. HOD break check
            if L_HOD_BREAK_REQUIRED and not is_new_hod:
                continue

            # 2. Volume surge check
            if candle_idx >= 10:
                recent_vols = st["mh"].iloc[candle_idx-10:candle_idx]["Volume"].values.astype(float)
                avg_vol = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 0
            else:
                recent_vols = st["mh"].iloc[:candle_idx]["Volume"].values.astype(float)
                avg_vol = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 0

            if avg_vol > 0 and c_vol < avg_vol * L_VOL_SURGE_MULT:
                continue

            # 3. Price acceleration (green candle with min body %)
            if c_open > 0:
                body_pct = (c_close / c_open - 1) * 100
            else:
                body_pct = 0
            if body_pct < L_MIN_PRICE_ACCEL_PCT:
                continue

            # 4. VWAP filter
            if L_REQUIRE_ABOVE_VWAP and candle_idx < len(st["vwap"]):
                if c_close < st["vwap"][candle_idx]:
                    continue

            # All conditions met
            st["signal"] = True
            st["signal_price"] = c_close
            entry_candidates.append(st)

        # --- ALLOCATION: priority by gap % (highest gap = most squeeze potential) ---
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
            st["l_highest_since_entry"] = entry_price
            cash -= trade_size
            tickers_taken.add(st["ticker"])

    # EOD: close remaining
    for st in states:
        if st["entry_price"] is not None and st["shares"] > 0:
            last_ts = st["mh"].index[-1]
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["shares"] * sell_price
            partial_procs = st.get("l_partial_proceeds", 0)
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

    print(f"Strategy L: Low Float Squeeze Backtest")
    print(f"{'='*70}")
    print(f"  Float data loaded: {len(FLOAT_DATA)} tickers")
    low_ct = sum(1 for v in FLOAT_DATA.values() if v < L_MAX_FLOAT)
    print(f"  Low float (< {L_MAX_FLOAT/1e6:.0f}M): {low_ct} tickers")
    print(f"  Min gap:          {L_MIN_GAP_PCT}%")
    print(f"  Entry window:     candle {L_EARLIEST_CANDLE} - {L_LATEST_CANDLE}")
    print(f"  HOD break:        {L_HOD_BREAK_REQUIRED}")
    print(f"  Vol surge:        >= {L_VOL_SURGE_MULT}x avg last 10")
    print(f"  Price accel:      >= {L_MIN_PRICE_ACCEL_PCT}% green body")
    print(f"  Above VWAP:       {L_REQUIRE_ABOVE_VWAP}")
    print(f"  Tiered Targets:")
    print(f"    <{L_TIER1_FLOAT/1e6:.0f}M flt:    +{L_TIER1_TARGET1_PCT}% / +{L_TIER1_TARGET2_PCT}% (sell {L_PARTIAL_SELL_PCT:.0f}%)")
    print(f"    {L_TIER1_FLOAT/1e6:.0f}-{L_TIER2_FLOAT/1e6:.0f}M flt:  +{L_TIER2_TARGET1_PCT}% / +{L_TIER2_TARGET2_PCT}%")
    print(f"    {L_TIER2_FLOAT/1e6:.0f}M+ flt:    +{L_TIER3_TARGET1_PCT}% / +{L_TIER3_TARGET2_PCT}%")
    print(f"  Stop:             -{L_STOP_PCT}%")
    print(f"  Trail:            {L_TRAIL_PCT}% (activates at +{L_TRAIL_ACTIVATE_PCT}%)")
    print(f"  Time limit:       {L_TIME_LIMIT_MINUTES}m")
    print(f"  Data: {data_dirs}")
    print(f"{'='*70}\n")

    print("Loading data...")
    all_dates, daily_picks = load_all_picks(data_dirs)
    all_dates = [d for d in all_dates if "2024-01-01" <= d <= "2026-02-28"]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")

    # Count eligible picks
    total_picks = 0
    eligible_picks = 0
    for d in all_dates:
        for p in daily_picks.get(d, []):
            total_picks += 1
            flt = FLOAT_DATA.get(p["ticker"])
            if flt and flt < L_MAX_FLOAT and p["gap_pct"] >= L_MIN_GAP_PCT:
                eligible_picks += 1
    print(f"  Total picks: {total_picks}, L-eligible: {eligible_picks} ({eligible_picks/max(total_picks,1)*100:.1f}%)\n")

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

        states, cash, unsettled = simulate_day_l(picks, cash, cash_account)

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
                    flt_m = st["float_shares"] / 1e6
                    print(f"  -> {st['ticker']:<6} {reason_short:<3}  "
                          f"${st['pnl']:>+10,.0f}  ({pct:>+6.2f}%)  "
                          f"gap={st['gap_pct']:.0f}% float={flt_m:.1f}M{vc_tag}")

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

    # Per-trade % returns
    all_trade_pcts = []
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None and st["position_cost"] > 0:
                all_trade_pcts.append(st["pnl"] / st["position_cost"] * 100)

    # Daily equity % changes
    daily_eq_pcts = []
    for i in range(1, len(all_results)):
        prev_eq = all_results[i-1]["equity"]
        curr_eq = all_results[i]["equity"]
        if prev_eq > 0:
            daily_eq_pcts.append((curr_eq / prev_eq - 1) * 100)

    print(f"\n{'='*70}")
    print(f"  STRATEGY L: LOW FLOAT SQUEEZE SUMMARY")
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

    # Daily % metrics
    if daily_eq_pcts:
        print(f"\n  Daily % Change:")
        print(f"    Avg Daily:      {np.mean(daily_eq_pcts):+.2f}%")
        print(f"    Median Daily:   {np.median(daily_eq_pcts):+.2f}%")
        print(f"    Best Day:       {max(daily_eq_pcts):+.2f}%")
        print(f"    Worst Day:      {min(daily_eq_pcts):+.2f}%")
        print(f"    Std Dev:        {np.std(daily_eq_pcts):.2f}%")

    # Per-trade % metrics
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

    # Float breakdown of trades
    float_buckets = {"<2M": [], "2-5M": [], "5-10M": []}
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None:
                f = st["float_shares"]
                if f < 2_000_000:
                    float_buckets["<2M"].append(st["pnl"])
                elif f < 5_000_000:
                    float_buckets["2-5M"].append(st["pnl"])
                else:
                    float_buckets["5-10M"].append(st["pnl"])
    print(f"\n  Float Breakdown:")
    for bucket, pnls in float_buckets.items():
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
        run_dir = os.path.join("charts", f"low_float_squeeze_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nGenerating charts -> {run_dir}/")

        def _to_et(ts_val):
            try:
                return ts_val.astimezone(ET_TZ)
            except Exception:
                return ts_val

        COLORS_L = ["#FF5722", "#FF7043", "#FF8A65", "#FFAB91", "#FFCCBC"]
        DAYS_PER_PAGE = 5

        # Only pages with trades
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
                f"Strategy L: Low Float Squeeze - Page {page+1}/{num_pages}\n"
                f"Float < {L_MAX_FLOAT/1e6:.0f}M | Gap >= {L_MIN_GAP_PCT}% | "
                f"HOD break + Vol {L_VOL_SURGE_MULT}x | "
                f"Tiered Targets (<{L_TIER1_FLOAT/1e6:.0f}M: +{L_TIER1_TARGET1_PCT:.0f}/{L_TIER1_TARGET2_PCT:.0f}%, "
                f"{L_TIER1_FLOAT/1e6:.0f}-{L_TIER2_FLOAT/1e6:.0f}M: +{L_TIER2_TARGET1_PCT:.0f}/{L_TIER2_TARGET2_PCT:.0f}%, "
                f"{L_TIER2_FLOAT/1e6:.0f}M+: +{L_TIER3_TARGET1_PCT:.0f}/{L_TIER3_TARGET2_PCT:.0f}%) | "
                f"Stop: -{L_STOP_PCT}% | Trail: {L_TRAIL_PCT}% (at +{L_TRAIL_ACTIVATE_PCT}%)",
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

                        color = COLORS_L[si % len(COLORS_L)]
                        first_candle_close = float(mh.iloc[0]["Close"])
                        pct_change = (mh["Close"].values.astype(float) / first_candle_close - 1) * 100

                        flt_m = st["float_shares"] / 1e6
                        vc_tag = " VC" if st.get("vol_capped") else ""
                        cost_k = st["position_cost"] / 1000
                        label = f"{st['ticker']} ${cost_k:.0f}K (gap {st['gap_pct']:.0f}%, {flt_m:.1f}M flt){vc_tag}"
                        ax_price.plot(et_times, pct_change, color=color, linewidth=1.2,
                                      label=label, alpha=0.85)

                        # BUY marker
                        if st["entry_time"] is not None:
                            et_buy = _to_et(st["entry_time"])
                            # entry_price is None after close; reconstruct from position_cost
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
                    # Show tiered target lines for each traded stock's tier
                    shown_tiers = set()
                    for s in traded:
                        t1p, t2p = s["l_target1_pct"], s["l_target2_pct"]
                        tier_key = (t1p, t2p)
                        if tier_key not in shown_tiers:
                            shown_tiers.add(tier_key)
                            flt_m = s["float_shares"] / 1e6
                            ax_price.axhline(y=t1p, color="#FF5722", linestyle=":", alpha=0.35,
                                             label=f"+{t1p:.0f}% tgt1 ({flt_m:.0f}M)")
                            ax_price.axhline(y=t2p, color="#E64A19", linestyle=":", alpha=0.25,
                                             label=f"+{t2p:.0f}% tgt2")
                    ax_price.axhline(y=-L_STOP_PCT, color="#f44336", linestyle=":", alpha=0.4,
                                     label=f"-{L_STOP_PCT}% stop")
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
                        flt_m = s["float_shares"] / 1e6
                        tickers.append(f"{s['ticker']} ({flt_m:.1f}M){vc}")
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
            chart_path = os.path.join(run_dir, f"lfs_page_{page+1:02d}.png")
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
            f"Strategy L: Low Float Squeeze Summary: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)",
            fontsize=16, fontweight="bold",
        )

        # 1. Equity curve
        ax = fig2.add_subplot(gs[0, :])
        equities = [r["equity"] for r in all_results]
        dates_list = [r["date"] for r in all_results]
        ax.plot(range(len(dates_list)), equities, color="#FF5722", linewidth=2, label="Strategy L")
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
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=8, color=mcolor, fontweight="bold",
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                              ec=mcolor, alpha=0.9, lw=1))
                        break

        ax.set_xticks(range(0, len(dates_list), max(1, len(dates_list) // 15)))
        ax.set_xticklabels([dates_list[i] for i in range(0, len(dates_list), max(1, len(dates_list) // 15))],
                           rotation=45, fontsize=8, ha="right")
        ax.set_title(f"Equity: ${STARTING_CASH:,} -> ${equities[-1]:,.0f} ({(equities[-1]/STARTING_CASH-1)*100:+.1f}%)",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Balance ($)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 2. Exit reasons pie
        ax = fig2.add_subplot(gs[1, 0])
        if all_exits:
            reason_colors = {
                "TARGET": "#4CAF50", "TRAIL": "#FF9800", "STOP": "#f44336",
                "TIME_STOP": "#2196F3", "EOD_CLOSE": "#9E9E9E",
            }
            labels = list(all_exits.keys())
            sizes = list(all_exits.values())
            colors = [reason_colors.get(r, "#999") for r in labels]
            ax.pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
                   colors=colors, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
        ax.set_title("Exit Reasons", fontweight="bold")

        # 3. Float bucket performance
        ax = fig2.add_subplot(gs[1, 1])
        bucket_names = list(float_buckets.keys())
        bucket_pnls = [sum(v) for v in float_buckets.values()]
        bucket_counts = [len(v) for v in float_buckets.values()]
        bucket_wr = [sum(1 for p in v if p > 0) / max(len(v), 1) * 100 for v in float_buckets.values()]
        bar_colors = ["#4CAF50" if p > 0 else "#f44336" for p in bucket_pnls]
        bars = ax.bar(bucket_names, bucket_pnls, color=bar_colors, edgecolor="white")
        for j, (bar, pnl, ct, wr) in enumerate(zip(bars, bucket_pnls, bucket_counts, bucket_wr)):
            ax.annotate(f"${pnl:+,.0f}\n{ct} trades\n{wr:.0f}% WR",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8, fontweight="bold")
        ax.set_title("Performance by Float Size", fontweight="bold")
        ax.set_ylabel("Total P&L ($)")
        ax.grid(alpha=0.3, axis="y")

        # 4. Stats panel
        ax = fig2.add_subplot(gs[1, 2])
        ax.axis("off")
        stats_text = (
            f"PERFORMANCE\n{'='*32}\n"
            f"Starting:     ${STARTING_CASH:,}\n"
            f"Ending:       ${final_equity:,.0f}\n"
            f"Return:       {(final_equity/STARTING_CASH-1)*100:+.1f}%\n"
            f"{'_'*32}\n"
            f"Total Trades: {total_trades}\n"
            f"Win Rate:     {total_wins/max(total_trades,1)*100:.1f}%\n"
            f"Avg Win:      ${avg_win:+,.0f}\n"
            f"Avg Loss:     ${avg_loss:+,.0f}\n"
            f"Profit Fac:   {pf:.2f}\n"
            f"Sharpe:       {sharpe:.2f}\n"
            f"Green Days:   {green}/{len(daily_pnls)}\n"
            f"\n{'='*32}\n"
            f"PARAMETERS\n{'='*32}\n"
            f"Float < {L_MAX_FLOAT/1e6:.0f}M\n"
            f"Gap >= {L_MIN_GAP_PCT:.0f}%\n"
            f"Entry: c{L_EARLIEST_CANDLE}-{L_LATEST_CANDLE}\n"
            f"HOD break: {L_HOD_BREAK_REQUIRED}\n"
            f"Vol surge: {L_VOL_SURGE_MULT}x\n"
            f"Accel: {L_MIN_PRICE_ACCEL_PCT}%\n"
            f"VWAP: {L_REQUIRE_ABOVE_VWAP}\n"
            f"Tiered Targets:\n"
            f"  <{L_TIER1_FLOAT/1e6:.0f}M: +{L_TIER1_TARGET1_PCT:.0f}%/+{L_TIER1_TARGET2_PCT:.0f}%\n"
            f"  {L_TIER1_FLOAT/1e6:.0f}-{L_TIER2_FLOAT/1e6:.0f}M: +{L_TIER2_TARGET1_PCT:.0f}%/+{L_TIER2_TARGET2_PCT:.0f}%\n"
            f"  {L_TIER2_FLOAT/1e6:.0f}M+: +{L_TIER3_TARGET1_PCT:.0f}%/+{L_TIER3_TARGET2_PCT:.0f}%\n"
            f"Partial: {L_PARTIAL_SELL_PCT:.0f}%\n"
            f"Stop: -{L_STOP_PCT}%\n"
            f"Trail: {L_TRAIL_PCT}% (at +{L_TRAIL_ACTIVATE_PCT}%)\n"
            f"Time: {L_TIME_LIMIT_MINUTES}m\n"
            f"\n{'='*32}\n"
            f"STRESS TEST\n"
            f"Max DD: ${max_dd_dollar:,.0f} ({max_dd_pct*100:.1f}%)\n"
            f"Streak: {max_streak} days\n"
        )
        if vc_count > 0:
            stats_text += f"Vol-Capped: {vc_count}\n"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#FBE9E7", alpha=0.7))

        plt.tight_layout()
        summary_path = os.path.join(run_dir, "lfs_summary.png")
        plt.savefig(summary_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Summary chart saved to {summary_path}")
