"""
Unified Strategy: Green Candle (G+A+F) + test_full Breakout on shared balance
==============================================================================
Both strategies run simultaneously on the same day's candles, sharing one cash pool.
MAX_POSITIONS=1 globally, 100% balance sizing.

Green candle: No SPY filter. Fires on candle 2 (~10 min). Exits via target/time stop.
test_full:    SPY SMA(40) filter. Enters on confirmed breakout+pullback+bounce.
              Exits via partial sell (+15%), trailing stop, stop loss (-16%).

Priority: Green candle enters first (earlier signal). Once it exits, freed cash
          becomes available for test_full entries.

Usage:
  python test_unified.py
  python test_unified.py --no-charts
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
from datetime import datetime

from test_full import (
    load_all_picks,
    _is_warrant_or_unit,
    _get_trade_pct,
    SLIPPAGE_PCT,
    STARTING_CASH,
    TRADE_PCT,
    TOP_N,
    SPY_SMA_PERIOD,
    MARGIN_THRESHOLD,
    VOL_CAP_PCT,
    ET_TZ,
)
from regime_filters import RegimeFilter

import io, sys as _sys
if hasattr(_sys.stdout, 'buffer'):
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8",
                                    errors="replace", line_buffering=True)

# ─── GREEN CANDLE CONFIG (Optuna-optimized) ──────────────────────────────────
GC_G_MIN_GAP = 25.0;  GC_G_TARGET = 8.0;  GC_G_TIME = 20
GC_A_MIN_GAP = 15.0;  GC_A_BODY = 0.0;  GC_A_TARGET = 3.0;  GC_A_TIME = 10
GC_F_MIN_GAP = 10.0;  GC_F_TARGET = 10.0;  GC_F_TIME = 3

# ─── TEST_FULL CONFIG (Phase 3 Optuna / run_111559) ─────────────────────────
TF_STOP_LOSS_PCT = 16.0
TF_PARTIAL_SELL_PCT = 15.0
TF_PARTIAL_SELL_FRAC = 0.90
TF_ATR_PERIOD = 8
TF_ATR_MULTIPLIER = 4.25
TF_CONFIRM_ABOVE = 2
TF_CONFIRM_WINDOW = 4
TF_PULLBACK_PCT = 4.0
TF_PULLBACK_TIMEOUT = 24
TF_EOD_EXIT_MINUTES = 30

# ─── SHARED CONFIG ───────────────────────────────────────────────────────────
EOD_EXIT_MINUTES = 15           # Green candle EOD
MAX_POSITIONS = 1               # Global: only 1 position at a time
FULL_BALANCE_SIZING = True


def _gc_classify(gap_pct, body_pct, second_green, second_new_high):
    """Classify candle 2 for green candle strategies G > A > F."""
    if gap_pct >= GC_G_MIN_GAP and second_green and second_new_high:
        return "G"
    if gap_pct >= GC_A_MIN_GAP and body_pct >= GC_A_BODY and second_green and second_new_high:
        return "A"
    if gap_pct >= GC_F_MIN_GAP and second_green:
        return "F"
    return None


def _gc_params(strategy):
    """Return (target_pct, time_limit) for green candle strategy."""
    return {
        "G": (GC_G_TARGET, GC_G_TIME),
        "A": (GC_A_TARGET, GC_A_TIME),
        "F": (GC_F_TARGET, GC_F_TIME),
    }.get(strategy, (GC_A_TARGET, GC_A_TIME))


def simulate_day_unified(picks, starting_cash, cash_account=False, spy_ok=True):
    """Simulate both strategies on shared cash for one day."""
    cash = starting_cash
    unsettled = 0.0

    def _receive(amount):
        nonlocal cash, unsettled
        if cash_account:
            unsettled += amount
        else:
            cash += amount

    # Build unified timestamp index
    all_ts = set()
    for p in picks:
        mh = p.get("market_hour_candles")
        if mh is not None and len(mh) > 0:
            all_ts.update(mh.index.tolist())
    all_ts = sorted(all_ts)
    if not all_ts:
        return [], [], cash, unsettled

    # ─── Initialize green candle states ──────────────────────────────────
    gc_states = []
    for p in picks:
        mh = p.get("market_hour_candles")
        if mh is None or len(mh) == 0:
            continue
        gc_states.append({
            "ticker": p["ticker"], "gap_pct": p["gap_pct"],
            "pm_volume": p.get("pm_volume", 0), "mh": mh,
            "candle_count": 0, "first_candle_ok": False,
            "first_candle_high": 0.0, "first_candle_body_pct": 0.0,
            "signal": False, "signal_price": None, "open_price": None,
            "strategy": None, "strat_type": "GC",
            "entry_price": None, "entry_time": None,
            "exit_price": None, "exit_time": None, "exit_reason": None,
            "shares": 0, "position_cost": 0.0, "pnl": 0.0, "done": False,
        })

    # ─── Initialize test_full states ─────────────────────────────────────
    tf_states = []
    for p in picks:
        mh = p.get("market_hour_candles")
        if mh is None or len(mh) == 0:
            continue
        tf_states.append({
            "ticker": p["ticker"], "gap_pct": p["gap_pct"],
            "premarket_high": p["premarket_high"],
            "pm_volume": p.get("pm_volume", 0), "mh": mh,
            "strat_type": "TF",
            # Breakout detection
            "recent_closes_above": [], "breakout_confirmed": False,
            "candles_since_confirm": 0, "pullback_detected": False,
            # Position
            "signal": False, "signal_price": None,
            "entry_price": None, "original_entry_price": None,
            "entry_time": None,
            "exit_price": None, "exit_time": None, "exit_reason": None,
            "shares": 0, "remaining_shares": 0,
            "position_cost": 0.0, "total_exit_value": 0.0,
            "pnl": 0.0, "done": False,
            # Partial sells / trailing
            "partial_sold": False, "trailing_active": False,
            "highest_since_entry": 0.0,
            "first_candle_body_pct": 0.0,
        })

    # ─── MAIN LOOP: process each timestamp ───────────────────────────────
    for ts in all_ts:
        try:
            ts_et = ts.astimezone(ET_TZ)
        except Exception:
            ts_et = ts
        minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)

        # ── PASS 1: Exits (frees cash) ──────────────────────────────────
        # Green candle exits
        for st in gc_states:
            if st["done"] or st["entry_price"] is None:
                continue
            if ts not in st["mh"].index:
                continue
            candle = st["mh"].loc[ts]
            c_high, c_close = float(candle["High"]), float(candle["Close"])

            entry_et = st["entry_time"]
            try:
                entry_et = entry_et.astimezone(ET_TZ)
            except Exception:
                pass
            mins_in = (ts_et.hour * 60 + ts_et.minute) - (entry_et.hour * 60 + entry_et.minute)
            target_pct, time_lim = _gc_params(st["strategy"])

            # EOD
            if minutes_to_close <= EOD_EXIT_MINUTES:
                sell_p = c_close * (1 - SLIPPAGE_PCT / 100)
                proceeds = st["shares"] * sell_p
                st.update(pnl=proceeds - st["position_cost"], exit_price=c_close,
                          exit_time=ts, exit_reason="GC_EOD", entry_price=None, shares=0, done=True)
                _receive(proceeds); continue

            # Target
            tp = st["entry_price"] * (1 + target_pct / 100)
            if c_high >= tp:
                sell_p = tp * (1 - SLIPPAGE_PCT / 100)
                proceeds = st["shares"] * sell_p
                st.update(pnl=proceeds - st["position_cost"], exit_price=tp,
                          exit_time=ts, exit_reason="GC_TARGET", entry_price=None, shares=0, done=True)
                _receive(proceeds); continue

            # Time stop
            if mins_in >= time_lim:
                sell_p = c_close * (1 - SLIPPAGE_PCT / 100)
                proceeds = st["shares"] * sell_p
                st.update(pnl=proceeds - st["position_cost"], exit_price=c_close,
                          exit_time=ts, exit_reason="GC_TIME", entry_price=None, shares=0, done=True)
                _receive(proceeds); continue

        # test_full exits
        for st in tf_states:
            if st["done"] or st["entry_price"] is None:
                continue
            if ts not in st["mh"].index:
                continue
            candle = st["mh"].loc[ts]
            c_high, c_low, c_close = float(candle["High"]), float(candle["Low"]), float(candle["Close"])
            c_vol = float(candle["Volume"])

            if c_high > st["highest_since_entry"]:
                st["highest_since_entry"] = c_high

            base_price = st["original_entry_price"]

            # EOD exit (test_full uses 30 min)
            if minutes_to_close <= TF_EOD_EXIT_MINUTES:
                sell_p = c_close * (1 - SLIPPAGE_PCT / 100)
                proceeds = st["remaining_shares"] * sell_p
                st["total_exit_value"] += proceeds
                st["pnl"] = st["total_exit_value"] - st["position_cost"]
                st.update(exit_price=c_close, exit_time=ts, exit_reason="TF_EOD",
                          entry_price=None, shares=0, remaining_shares=0, done=True)
                _receive(proceeds); continue

            # Stop loss
            stop_price = base_price * (1 - TF_STOP_LOSS_PCT / 100)
            if c_low <= stop_price:
                sell_p = stop_price * (1 - SLIPPAGE_PCT / 100)
                proceeds = st["remaining_shares"] * sell_p
                st["total_exit_value"] += proceeds
                st["pnl"] = st["total_exit_value"] - st["position_cost"]
                st.update(exit_price=stop_price, exit_time=ts, exit_reason="TF_STOP",
                          entry_price=None, shares=0, remaining_shares=0, done=True)
                _receive(proceeds); continue

            # Partial sell at +15% (sell 90%)
            if not st["partial_sold"]:
                partial_price = base_price * (1 + TF_PARTIAL_SELL_PCT / 100)
                if c_high >= partial_price:
                    sell_shares = st["remaining_shares"] * TF_PARTIAL_SELL_FRAC
                    sell_p = partial_price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = sell_shares * sell_p
                    st["remaining_shares"] -= sell_shares
                    st["total_exit_value"] += proceeds
                    st["partial_sold"] = True
                    st["trailing_active"] = True
                    _receive(proceeds)
                    # If no shares left, done
                    if st["remaining_shares"] < 0.01:
                        st["pnl"] = st["total_exit_value"] - st["position_cost"]
                        st.update(exit_price=partial_price, exit_time=ts,
                                  exit_reason="TF_TARGET", entry_price=None, done=True)
                        continue

            # Trailing stop (ATR-based) after partial sell
            if st["trailing_active"] and st["remaining_shares"] > 0:
                # Compute ATR from recent candles
                mh_up_to = st["mh"].loc[st["mh"].index <= ts]
                if len(mh_up_to) >= TF_ATR_PERIOD:
                    highs = mh_up_to["High"].values[-TF_ATR_PERIOD:].astype(float)
                    lows = mh_up_to["Low"].values[-TF_ATR_PERIOD:].astype(float)
                    closes = mh_up_to["Close"].values[-TF_ATR_PERIOD:].astype(float)
                    tr = np.maximum(highs - lows, np.maximum(
                        np.abs(highs - np.roll(closes, 1)),
                        np.abs(lows - np.roll(closes, 1))))
                    tr[0] = highs[0] - lows[0]
                    atr = np.mean(tr)
                    trail_stop = st["highest_since_entry"] - (atr * TF_ATR_MULTIPLIER)
                    if c_low <= trail_stop and trail_stop > 0:
                        sell_p = trail_stop * (1 - SLIPPAGE_PCT / 100)
                        proceeds = st["remaining_shares"] * sell_p
                        st["total_exit_value"] += proceeds
                        st["pnl"] = st["total_exit_value"] - st["position_cost"]
                        st.update(exit_price=trail_stop, exit_time=ts, exit_reason="TF_TRAIL",
                                  entry_price=None, shares=0, remaining_shares=0, done=True)
                        _receive(proceeds); continue

        # ── PASS 2: Signal detection ────────────────────────────────────
        gc_candidates = []
        tf_candidates = []

        # Green candle signals (candle counting)
        for st in gc_states:
            if st["done"] or st["entry_price"] is not None:
                continue
            if ts not in st["mh"].index:
                continue
            candle = st["mh"].loc[ts]
            c_open, c_high, c_close = float(candle["Open"]), float(candle["High"]), float(candle["Close"])
            st["candle_count"] += 1

            if st["candle_count"] == 1:
                if c_open > 0:
                    body_pct = (c_close / c_open - 1) * 100
                    st["first_candle_body_pct"] = body_pct
                    if body_pct > 0:
                        st["first_candle_ok"] = True
                        st["first_candle_high"] = c_high
                        st["signal_price"] = c_close
                if not st["first_candle_ok"]:
                    st["done"] = True

            elif st["candle_count"] == 2 and st["first_candle_ok"]:
                second_green = c_close > c_open
                second_new_high = c_high > st["first_candle_high"]
                strategy = _gc_classify(st["gap_pct"], st["first_candle_body_pct"],
                                        second_green, second_new_high)
                if strategy:
                    st["strategy"] = strategy
                    st["signal"] = True
                    st["signal_price"] = c_close
                    gc_candidates.append(st)
                else:
                    st["done"] = True

            elif st["signal"] and st["entry_price"] is None and not st["done"]:
                st["signal_price"] = c_close
                gc_candidates.append(st)

        # test_full signals (breakout → pullback → bounce) — only if SPY ok
        if spy_ok:
            for st in tf_states:
                if st["done"] or st["entry_price"] is not None:
                    continue
                if ts not in st["mh"].index:
                    continue
                candle = st["mh"].loc[ts]
                c_high, c_low, c_close = float(candle["High"]), float(candle["Low"]), float(candle["Close"])
                pm_high = st["premarket_high"]

                # Phase 1: Breakout confirmation
                if not st["breakout_confirmed"]:
                    st["recent_closes_above"].append(1 if c_close > pm_high else 0)
                    if len(st["recent_closes_above"]) > TF_CONFIRM_WINDOW:
                        st["recent_closes_above"] = st["recent_closes_above"][-TF_CONFIRM_WINDOW:]
                    if sum(st["recent_closes_above"]) >= TF_CONFIRM_ABOVE:
                        st["breakout_confirmed"] = True
                    continue

                # Phase 2: Pullback detection
                if not st["pullback_detected"]:
                    st["candles_since_confirm"] += 1
                    pullback_zone = pm_high * (1 + TF_PULLBACK_PCT / 100)
                    if c_low <= pullback_zone:
                        st["pullback_detected"] = True
                    elif st["candles_since_confirm"] >= TF_PULLBACK_TIMEOUT:
                        # Timeout: force entry
                        st["signal"] = True
                        st["signal_price"] = c_close
                        tf_candidates.append(st)
                    continue

                # Phase 3: Bounce (close above PM high)
                if c_close > pm_high:
                    st["signal"] = True
                    st["signal_price"] = c_close
                    tf_candidates.append(st)
                    continue

        # ── PASS 3: Allocate capital (green candle first) ────────────────
        positions_active = (
            sum(1 for s in gc_states if s["entry_price"] is not None) +
            sum(1 for s in tf_states if s["entry_price"] is not None)
        )

        # Green candle entries (priority)
        gc_priority = {"G": 0, "A": 1, "F": 2}
        gc_candidates.sort(key=lambda s: (gc_priority.get(s["strategy"], 9), -s["first_candle_body_pct"]))

        for st in gc_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue
            if positions_active >= MAX_POSITIONS:
                break
            if minutes_to_close <= EOD_EXIT_MINUTES:
                continue
            if cash < 100:
                break

            trade_size = cash if FULL_BALANCE_SIZING else starting_cash * 0.5
            if trade_size > cash:
                trade_size = cash

            fill_price = st["signal_price"]
            if fill_price is None or fill_price <= 0:
                continue

            entry_price = fill_price * (1 + SLIPPAGE_PCT / 100)
            st["entry_price"] = entry_price
            st["entry_time"] = ts
            st["position_cost"] = trade_size
            st["shares"] = trade_size / entry_price
            cash -= trade_size
            positions_active += 1

        # test_full entries (only if cash available)
        tf_candidates.sort(key=lambda s: -s["pm_volume"])  # Best liquidity first

        for st in tf_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue
            if positions_active >= MAX_POSITIONS:
                break
            if minutes_to_close <= TF_EOD_EXIT_MINUTES:
                continue
            if cash < 100:
                break

            trade_size = cash if FULL_BALANCE_SIZING else starting_cash * TRADE_PCT
            if trade_size > cash:
                trade_size = cash

            fill_price = st["signal_price"]
            if fill_price is None or fill_price <= 0:
                continue

            entry_price = fill_price * (1 + SLIPPAGE_PCT / 100)
            st["entry_price"] = entry_price
            st["original_entry_price"] = entry_price
            st["entry_time"] = ts
            st["position_cost"] = trade_size
            st["shares"] = trade_size / entry_price
            st["remaining_shares"] = st["shares"]
            st["highest_since_entry"] = float(st["mh"].loc[ts]["High"])
            cash -= trade_size
            positions_active += 1

    # ── EOD: close remaining ─────────────────────────────────────────────
    for st in gc_states:
        if st["entry_price"] is not None and st["shares"] > 0:
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_p = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["shares"] * sell_p
            st.update(pnl=proceeds - st["position_cost"], exit_price=last_close,
                      exit_time=st["mh"].index[-1], exit_reason="GC_EOD",
                      entry_price=None, shares=0, done=True)
            _receive(proceeds)

    for st in tf_states:
        if st["entry_price"] is not None and st["remaining_shares"] > 0:
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_p = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["remaining_shares"] * sell_p
            st["total_exit_value"] += proceeds
            st["pnl"] = st["total_exit_value"] - st["position_cost"]
            st.update(exit_price=last_close, exit_time=st["mh"].index[-1],
                      exit_reason="TF_EOD", entry_price=None, remaining_shares=0, done=True)
            _receive(proceeds)

    return gc_states, tf_states, cash, unsettled


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    no_charts = "--no-charts" in sys.argv
    data_dirs = ["stored_data_oos", "stored_data"]

    print(f"Unified Strategy: Green Candle (G+A+F) + test_full Breakout")
    print(f"{'='*70}")
    print(f"  Green Candle: G(gap>={GC_G_MIN_GAP}%,+{GC_G_TARGET}%/{GC_G_TIME}m) "
          f"A(gap>={GC_A_MIN_GAP}%,+{GC_A_TARGET}%/{GC_A_TIME}m) "
          f"F(gap>={GC_F_MIN_GAP}%,+{GC_F_TARGET}%/{GC_F_TIME}m)")
    print(f"  test_full:    Confirm({TF_CONFIRM_ABOVE}/{TF_CONFIRM_WINDOW}) + "
          f"Pullback({TF_PULLBACK_PCT}%) + Bounce")
    print(f"                Partial +{TF_PARTIAL_SELL_PCT}%/{TF_PARTIAL_SELL_FRAC*100:.0f}% "
          f"| Stop -{TF_STOP_LOSS_PCT}% | Trail ATR*{TF_ATR_MULTIPLIER}")
    print(f"  Priority:     Green candle > test_full")
    print(f"  SPY filter:   Green candle=OFF, test_full=ON (SMA{SPY_SMA_PERIOD})")
    print(f"  Max Positions: {MAX_POSITIONS} (shared, 100% balance)")
    print(f"{'='*70}\n")

    print("Loading data...")
    all_dates, daily_picks = load_all_picks(data_dirs)
    all_dates = [d for d in all_dates if "2025-10-01" <= d <= "2026-02-28"]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}\n")

    print("Loading regime filter...")
    rf = RegimeFilter(spy_ma_period=SPY_SMA_PERIOD, enable_vix=False,
                      enable_spy_trend=True, enable_adaptive=False)
    rf.load_data(all_dates[0], all_dates[-1])
    tradeable_spy = sum(1 for d in all_dates if rf.check(d)[0])
    skipped_spy = len(all_dates) - tradeable_spy
    print(f"  SPY filter: {tradeable_spy} tradeable, {skipped_spy} skipped for test_full\n")

    cash = STARTING_CASH
    unsettled_cash = 0.0
    all_results = []

    print(f"{'Date':<12} {'Strat':>8} {'Trades':>6} {'Win':>4} {'Loss':>5} "
          f"{'Day P&L':>12} {'Balance':>14}")
    print("-" * 75)

    for d in all_dates:
        if unsettled_cash > 0:
            cash += unsettled_cash
            unsettled_cash = 0.0

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD
        spy_ok, _, _ = rf.check(d)

        gc_states, tf_states, new_cash, new_unsettled = simulate_day_unified(
            picks, cash, cash_account, spy_ok
        )

        day_pnl = 0.0
        day_trades = 0
        day_wins = 0
        day_losses = 0
        gc_count = 0
        tf_count = 0
        gc_wins = 0
        tf_wins = 0

        for st in gc_states:
            if st["exit_reason"] is not None:
                day_trades += 1; gc_count += 1
                day_pnl += st["pnl"]
                if st["pnl"] > 0:
                    day_wins += 1; gc_wins += 1
                else:
                    day_losses += 1

        for st in tf_states:
            if st["exit_reason"] is not None:
                day_trades += 1; tf_count += 1
                day_pnl += st["pnl"]
                if st["pnl"] > 0:
                    day_wins += 1; tf_wins += 1
                else:
                    day_losses += 1

        cash = new_cash
        unsettled_cash = new_unsettled
        equity = cash + unsettled_cash

        parts = []
        if gc_count > 0:
            parts.append(f"GC{gc_count}")
        if tf_count > 0:
            parts.append(f"TF{tf_count}")
        strat_label = "+".join(parts) if parts else ""
        spy_tag = "" if spy_ok else " [noSPY]"

        print(f"{d:<12} {strat_label:>8} {day_trades:>6} {day_wins:>4} {day_losses:>5} "
              f"${day_pnl:>+11,.0f} ${equity:>13,.0f}{spy_tag}")

        all_results.append({
            "date": d, "gc_states": gc_states, "tf_states": tf_states,
            "day_pnl": day_pnl, "equity": equity,
            "trades": day_trades, "wins": day_wins, "losses": day_losses,
            "gc_count": gc_count, "tf_count": tf_count,
            "gc_wins": gc_wins, "tf_wins": tf_wins,
            "spy_ok": spy_ok,
        })

    # ─── Summary ─────────────────────────────────────────────────────────
    final_equity = cash + unsettled_cash
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    total_gc = sum(r["gc_count"] for r in all_results)
    total_tf = sum(r["tf_count"] for r in all_results)
    total_gc_wins = sum(r["gc_wins"] for r in all_results)
    total_tf_wins = sum(r["tf_wins"] for r in all_results)

    gc_pnls = []
    tf_pnls = []
    for r in all_results:
        for st in r["gc_states"]:
            if st["exit_reason"] is not None:
                gc_pnls.append(st["pnl"])
        for st in r["tf_states"]:
            if st["exit_reason"] is not None:
                tf_pnls.append(st["pnl"])

    daily_pnls = [r["day_pnl"] for r in all_results if r["trades"] > 0]
    green = sum(1 for p in daily_pnls if p > 0)
    red = sum(1 for p in daily_pnls if p <= 0)
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if daily_pnls and np.std(daily_pnls) > 0 else 0

    all_pnls = gc_pnls + tf_pnls
    avg_win = np.mean([p for p in all_pnls if p > 0]) if total_wins > 0 else 0
    avg_loss = np.mean([p for p in all_pnls if p <= 0]) if total_losses > 0 else 0
    pf = abs(avg_win * total_wins / (avg_loss * total_losses)) if total_losses > 0 and avg_loss != 0 else 0

    print(f"\n{'='*70}")
    print(f"  UNIFIED STRATEGY SUMMARY")
    print(f"{'='*70}")
    print(f"  Starting Cash:    ${STARTING_CASH:,}")
    print(f"  Ending Equity:    ${final_equity:,.0f}  ({(final_equity/STARTING_CASH - 1)*100:+.1f}%)")
    print(f"  Trading Days:     {len(all_dates)}")
    print(f"  Total Trades:     {total_trades}")
    print(f"    Winners:        {total_wins} ({total_wins/max(total_trades,1)*100:.1f}%)")
    print(f"    Losers:         {total_losses}")
    print(f"  Avg Win:          ${avg_win:+,.0f}")
    print(f"  Avg Loss:         ${avg_loss:+,.0f}")
    if pf > 0:
        print(f"  Profit Factor:    {pf:.2f}")

    print(f"\n  {'---'*17}")
    print(f"  GREEN CANDLE (G+A+F): {total_gc} trades, {total_gc_wins} wins ({total_gc_wins/max(total_gc,1)*100:.1f}%)")
    if gc_pnls:
        print(f"    Total PnL: ${sum(gc_pnls):+,.0f}")
        gc_w = [p for p in gc_pnls if p > 0]
        gc_l = [p for p in gc_pnls if p <= 0]
        if gc_w: print(f"    Avg Win:   ${np.mean(gc_w):+,.0f}")
        if gc_l: print(f"    Avg Loss:  ${np.mean(gc_l):+,.0f}")

    print(f"\n  {'---'*17}")
    print(f"  TEST_FULL (Breakout): {total_tf} trades, {total_tf_wins} wins ({total_tf_wins/max(total_tf,1)*100:.1f}%)")
    if tf_pnls:
        print(f"    Total PnL: ${sum(tf_pnls):+,.0f}")
        tf_w = [p for p in tf_pnls if p > 0]
        tf_l = [p for p in tf_pnls if p <= 0]
        if tf_w: print(f"    Avg Win:   ${np.mean(tf_w):+,.0f}")
        if tf_l: print(f"    Avg Loss:  ${np.mean(tf_l):+,.0f}")

    # Exit reasons
    all_exits = {}
    for r in all_results:
        for st in r["gc_states"] + r["tf_states"]:
            if st["exit_reason"]:
                all_exits[st["exit_reason"]] = all_exits.get(st["exit_reason"], 0) + 1
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
    print(f"{'='*70}")

    if not no_charts:
        print("\n  (Charts not yet implemented for unified — use --no-charts)")
