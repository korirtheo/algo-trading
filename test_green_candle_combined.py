"""
Combined Green Candle Strategy Backtest (H + G + A + F + P on shared balance)
=============================================================================
Strategy H: Gap>=35% + body>=4% + 2nd green + new hi + vol confirm -> +16% target, 15m
Strategy G: Gap>=30% + 2nd green + new hi       -> +11% target, 10m time stop
Strategy A: Gap>=15% + body>=4% + 2nd green + new hi -> +6% target, 12m time stop
Strategy F: Gap>=10% + 2nd green                 -> +8% target, 3m time stop (catch-all)
Strategy P: Gap>=10% + PM high breakout + pullback + bounce -> partial +9%, runner +13%
            Trailing stop 2% (activates at +2%), hard stop -12%, 40m time limit
            (fallback: only fires when H/G/A/F don't classify the stock)

Priority: H > G > A > F > P (highest conviction first).
All strategies share the same $25K cash pool, MAX_POSITIONS=1.
No SPY regime filter.

Usage:
  python test_green_candle_combined.py
  python test_green_candle_combined.py --no-charts
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
    MARGIN_THRESHOLD,
    VOL_CAP_PCT,
    ET_TZ,
)

import io, sys as _sys
if hasattr(_sys.stdout, 'buffer'):
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8",
                                    errors="replace", line_buffering=True)

# --- STRATEGY H CONFIG: High Conviction (filtered G) ---
H_MIN_GAP_PCT = 35.0             # Optuna v2: tightened from 25%
H_MIN_BODY_PCT = 4.0             # Optuna v2: loosened from 5%
H_REQUIRE_VOL_CONFIRM = True     # c2 volume > c1 volume
H_TARGET_PCT = 16.0              # Optuna v2: raised from 12%
H_TIME_LIMIT_MINUTES = 15

# --- STRATEGY G CONFIG: Big Gap Runner ---
G_MIN_GAP_PCT = 30.0             # Optuna v2: tightened from 25%
G_MIN_BODY_PCT = 0.0             # No body filter
G_REQUIRE_2ND_GREEN = True
G_REQUIRE_2ND_NEW_HIGH = True
G_TARGET_PCT = 11.0              # Optuna v2: raised from 8%
G_TIME_LIMIT_MINUTES = 10        # Optuna v2: cut from 20m

# --- STRATEGY A CONFIG: Quick Scalp ---
A_MIN_GAP_PCT = 15.0             # Unchanged
A_MIN_BODY_PCT = 4.0             # Optuna v2: added body filter
A_MAX_BODY_PCT = 999
A_REQUIRE_2ND_GREEN = True
A_REQUIRE_2ND_NEW_HIGH = True
A_TARGET_PCT = 6.0               # Optuna v2: raised from 3%
A_TIME_LIMIT_MINUTES = 12        # Optuna v2: raised from 10m

# --- STRATEGY F CONFIG: Catch-All ---
F_MIN_GAP_PCT = 10.0             # Unchanged
F_MIN_BODY_PCT = 0.0             # No body filter
F_REQUIRE_2ND_GREEN = True
F_REQUIRE_2ND_NEW_HIGH = False
F_TARGET_PCT = 8.0               # Optuna v2: lowered from 10%
F_TIME_LIMIT_MINUTES = 3         # Unchanged

# --- STRATEGY P CONFIG: PM High Breakout + Pullback + Bounce (Optuna v2) ---
# Only fires when H/G/A/F did not classify the stock (fallback)
# Advanced exits: trailing stop + partial selling
P_MIN_GAP_PCT = 10.0            # Optuna v2: widened from 15%
P_CONFIRM_ABOVE = 4             # Optuna v2: tightened from 3
P_CONFIRM_WINDOW = 6            # Optuna v2: widened from 5
P_PULLBACK_PCT = 7.0            # Optuna v2: widened from 5.5%
P_PULLBACK_TIMEOUT = 30         # Optuna v2: widened from 20
P_MAX_ENTRY_CANDLE = 75         # Optuna v2: reduced from 90
P_TARGET1_PCT = 9.0             # Partial target: sell 25% here
P_TARGET2_PCT = 13.0            # Runner target: sell remaining here
P_STOP_PCT = 12.0               # Optuna v2: widened from 10%
P_TIME_LIMIT_MINUTES = 40       # Optuna v2: reduced from 50
P_PARTIAL_SELL_PCT = 25.0       # Sell 25% of position at target1
P_TRAIL_PCT = 2.0               # Fixed trailing stop %
P_TRAIL_ACTIVATE_PCT = 2.0      # Start trailing after +2% unrealized

# --- SHARED CONFIG ---
EOD_EXIT_MINUTES = 15
MAX_POSITIONS = 1               # 100% of balance per trade
FULL_BALANCE_SIZING = True      # Use full cash for each trade


def _classify_candle2(gap_pct, body_pct, second_green, second_new_high, vol_confirm=False):
    """Classify on candle 2 for strategies H, G, A, F.
    Priority: H > G > A > F (highest conviction first)."""
    # H: high conviction - gap>=25%, body>=5%, 2nd green + new hi + vol confirm
    if (gap_pct >= H_MIN_GAP_PCT
            and body_pct >= H_MIN_BODY_PCT
            and second_green and second_new_high
            and (not H_REQUIRE_VOL_CONFIRM or vol_confirm)):
        return "H"
    # G: gap>=25%, 2nd green + new hi (strong runners)
    if (gap_pct >= G_MIN_GAP_PCT
            and body_pct >= G_MIN_BODY_PCT
            and (not G_REQUIRE_2ND_GREEN or second_green)
            and (not G_REQUIRE_2ND_NEW_HIGH or second_new_high)):
        return "G"
    # A: gap>=15%, 2nd green + new hi (quick scalp)
    if (gap_pct >= A_MIN_GAP_PCT
            and A_MIN_BODY_PCT <= body_pct <= A_MAX_BODY_PCT
            and (not A_REQUIRE_2ND_GREEN or second_green)
            and (not A_REQUIRE_2ND_NEW_HIGH or second_new_high)):
        return "A"
    # F: gap>=10%, 2nd green (catch-all)
    if (gap_pct >= F_MIN_GAP_PCT
            and body_pct >= F_MIN_BODY_PCT
            and (not F_REQUIRE_2ND_GREEN or second_green)
            and (not F_REQUIRE_2ND_NEW_HIGH or second_new_high)):
        return "F"
    return None


def simulate_day_combined(picks, starting_cash, cash_account=False):
    """Simulate combined A+G+F strategy for one day on shared capital."""
    cash = starting_cash
    unsettled = 0.0
    trade_pct = _get_trade_pct(starting_cash)

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

    # Initialize states
    states = []
    for pick in picks:
        mh = pick.get("market_hour_candles")
        if mh is None or len(mh) == 0:
            continue
        states.append({
            "ticker": pick["ticker"],
            "premarket_high": pick["premarket_high"],
            "gap_pct": pick["gap_pct"],
            "pm_volume": pick.get("pm_volume", 0),
            "mh": mh,
            # State — H/G/A/F candle classification
            "candle_count": 0,
            "first_candle_ok": False,
            "first_candle_high": 0.0,
            "first_candle_body_pct": 0.0,
            "first_candle_volume": 0.0,      # For vol confirm (H)
            "signal": False,
            "signal_price": None,
            "open_price": None,
            "strategy": None,  # "H", "G", "A", "F", or "P"
            # State — P (PM high breakout fallback)
            "p_eligible": False,         # Set True after candle 2 if H/G/A/F failed
            "p_recent_closes": [],       # Track closes above PM high
            "p_breakout_confirmed": False,
            "p_pullback_detected": False,
            "p_candles_since_confirm": 0,
            # State — P advanced exit management
            "p_highest_since_entry": 0.0,
            "p_trailing_active": False,
            "p_partial_taken": False,
            "p_partial_proceeds": 0.0,
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

            # --- IN POSITION: check exits using strategy-specific params ---
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

                # --- Helper: close entire remaining position ---
                def _close_position(st, price, reason, ts_now):
                    sell_price = price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["shares"] * sell_price
                    st["pnl"] = st["p_partial_proceeds"] + proceeds - st["position_cost"]
                    st["exit_price"] = price
                    st["exit_time"] = ts_now
                    st["exit_reason"] = reason
                    st["entry_price"] = None
                    st["shares"] = 0
                    st["done"] = True
                    _receive_proceeds(proceeds)

                # EOD forced exit (all strategies)
                if minutes_to_close <= EOD_EXIT_MINUTES:
                    _close_position(st, c_close, "EOD_CLOSE", ts)
                    continue

                # ===== STRATEGY P: Advanced exit (trailing + partial) =====
                if st["strategy"] == "P":
                    # Update highest since entry
                    if c_high > st["p_highest_since_entry"]:
                        st["p_highest_since_entry"] = c_high

                    # 1. Trailing stop (if active)
                    if st["p_trailing_active"]:
                        trail_stop = st["p_highest_since_entry"] * (1 - P_TRAIL_PCT / 100)
                        if c_low <= trail_stop:
                            _close_position(st, trail_stop, "TRAIL", ts)
                            continue
                    else:
                        # 2. Hard stop (before trail activates)
                        stop_price = st["entry_price"] * (1 - P_STOP_PCT / 100)
                        if c_low <= stop_price:
                            _close_position(st, stop_price, "STOP", ts)
                            continue

                    # 3. Activate trailing stop at +X% unrealized
                    if not st["p_trailing_active"]:
                        unrealized_pct = (c_high / st["entry_price"] - 1) * 100
                        if unrealized_pct >= P_TRAIL_ACTIVATE_PCT:
                            st["p_trailing_active"] = True

                    # 4. Partial sell at target1
                    if not st["p_partial_taken"] and P_PARTIAL_SELL_PCT > 0:
                        tgt1 = st["entry_price"] * (1 + P_TARGET1_PCT / 100)
                        if c_high >= tgt1:
                            partial_shares = st["shares"] * (P_PARTIAL_SELL_PCT / 100)
                            sell_price = tgt1 * (1 - SLIPPAGE_PCT / 100)
                            partial_proceeds = partial_shares * sell_price
                            st["p_partial_proceeds"] += partial_proceeds
                            st["shares"] -= partial_shares
                            st["p_partial_taken"] = True
                            _receive_proceeds(partial_proceeds)
                            if st["shares"] <= 0.001:
                                # Sold everything (shouldn't happen with 25%)
                                st["pnl"] = st["p_partial_proceeds"] - st["position_cost"]
                                st["exit_price"] = tgt1
                                st["exit_time"] = ts
                                st["exit_reason"] = "TARGET"
                                st["entry_price"] = None
                                st["shares"] = 0
                                st["done"] = True
                                continue
                            # Fall through to check target2 on same candle

                    # 5. Runner target2 (full exit of remaining)
                    tgt2 = st["entry_price"] * (1 + P_TARGET2_PCT / 100)
                    if c_high >= tgt2:
                        _close_position(st, tgt2, "TARGET", ts)
                        continue

                    # 6. Time stop
                    if minutes_in_trade >= P_TIME_LIMIT_MINUTES:
                        _close_position(st, c_close, "TIME_STOP", ts)
                        continue

                    continue

                # ===== STRATEGIES H/G/A/F: Simple target + time stop =====
                strat_map = {
                    "H": (H_TARGET_PCT, H_TIME_LIMIT_MINUTES),
                    "G": (G_TARGET_PCT, G_TIME_LIMIT_MINUTES),
                    "A": (A_TARGET_PCT, A_TIME_LIMIT_MINUTES),
                    "F": (F_TARGET_PCT, F_TIME_LIMIT_MINUTES),
                }
                target_pct, time_limit = strat_map.get(st["strategy"], (A_TARGET_PCT, A_TIME_LIMIT_MINUTES))

                # Target hit
                target_price = st["entry_price"] * (1 + target_pct / 100)
                if c_high >= target_price:
                    _close_position(st, target_price, "TARGET", ts)
                    continue

                # Time stop
                if minutes_in_trade >= time_limit:
                    _close_position(st, c_close, "TIME_STOP", ts)
                    continue

                continue

            # --- NOT IN POSITION: candle-by-candle signal detection ---
            st["candle_count"] += 1

            # CANDLE 1: Check first candle is green
            if st["candle_count"] == 1:
                if c_open > 0:
                    body_pct = (c_close / c_open - 1) * 100
                    st["first_candle_body_pct"] = body_pct
                    st["first_candle_high"] = c_high
                    st["first_candle_volume"] = float(candle["Volume"])
                    st["signal_price"] = c_close
                    st["open_price"] = c_open
                    if body_pct > 0:
                        st["first_candle_ok"] = True
                if not st["first_candle_ok"]:
                    st["done"] = True

            # CANDLE 2: Classify for H, G, A, F
            elif st["candle_count"] == 2 and st["first_candle_ok"]:
                second_green = c_close > c_open
                second_new_high = c_high > st["first_candle_high"]
                vol_confirm = float(candle["Volume"]) > st["first_candle_volume"]

                strategy = _classify_candle2(
                    st["gap_pct"], st["first_candle_body_pct"],
                    second_green, second_new_high, vol_confirm,
                )
                if strategy:
                    st["strategy"] = strategy
                    st["signal"] = True
                    entry_candidates.append(st)
                else:
                    # H/G/A/F rejected — enable P fallback if gap qualifies
                    if st["gap_pct"] >= P_MIN_GAP_PCT:
                        st["p_eligible"] = True
                    else:
                        st["done"] = True

            # Retry fill on later candles (H/G/A/F)
            elif st["signal"] and st["entry_price"] is None and not st["done"]:
                if c_close > 0:
                    st["signal_price"] = c_close
                    entry_candidates.append(st)

            # --- P: PM high breakout + pullback + bounce (candle 3+) ---
            elif st["p_eligible"] and st["entry_price"] is None and not st["done"]:
                if st["candle_count"] > P_MAX_ENTRY_CANDLE:
                    st["done"] = True
                    continue

                pm_high = st["premarket_high"]

                # Phase 1: Breakout confirmation
                if not st["p_breakout_confirmed"]:
                    st["p_recent_closes"].append(c_close > pm_high)
                    if len(st["p_recent_closes"]) > P_CONFIRM_WINDOW:
                        st["p_recent_closes"] = st["p_recent_closes"][-P_CONFIRM_WINDOW:]
                    if sum(st["p_recent_closes"]) >= P_CONFIRM_ABOVE:
                        st["p_breakout_confirmed"] = True

                # Phase 2: Pullback detection
                elif not st["p_pullback_detected"]:
                    st["p_candles_since_confirm"] += 1
                    pullback_zone = pm_high * (1 + P_PULLBACK_PCT / 100)
                    if c_low <= pullback_zone:
                        st["p_pullback_detected"] = True
                        if c_close > pm_high:
                            st["strategy"] = "P"
                            st["signal"] = True
                            st["signal_price"] = c_close
                            entry_candidates.append(st)
                    elif st["p_candles_since_confirm"] >= P_PULLBACK_TIMEOUT:
                        if c_close > pm_high:
                            st["strategy"] = "P"
                            st["signal"] = True
                            st["signal_price"] = c_close
                            entry_candidates.append(st)
                        else:
                            st["done"] = True

                # Phase 3: Bounce
                else:
                    if c_close > pm_high:
                        st["strategy"] = "P"
                        st["signal"] = True
                        st["signal_price"] = c_close
                        entry_candidates.append(st)

        # --- PASS 2: Allocate capital ---
        # Priority: H > G > A > F
        strat_priority = {"H": 0, "G": 1, "A": 2, "F": 3, "P": 4}
        entry_candidates.sort(key=lambda s: (strat_priority.get(s["strategy"], 9), -s["first_candle_body_pct"]))

        positions_today = sum(1 for s in states if s["entry_price"] is not None)

        for st in entry_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue
            if positions_today >= MAX_POSITIONS:
                break

            try:
                ts_et = ts.astimezone(ET_TZ)
            except Exception:
                ts_et = ts
            mins_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
            if mins_to_close <= EOD_EXIT_MINUTES:
                continue

            if FULL_BALANCE_SIZING:
                trade_size = cash
            else:
                trade_size = starting_cash * trade_pct
            if cash < 100:
                continue
            if trade_size > cash:
                trade_size = cash

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
            st["entry_time"] = ts
            st["position_cost"] = trade_size
            st["shares"] = trade_size / entry_price
            cash -= trade_size
            positions_today += 1

    # EOD: close remaining
    for st in states:
        if st["entry_price"] is not None and st["shares"] > 0:
            last_ts = st["mh"].index[-1]
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["shares"] * sell_price
            st["pnl"] = st.get("p_partial_proceeds", 0) + proceeds - st["position_cost"]
            st["exit_price"] = last_close
            st["exit_time"] = last_ts
            st["exit_reason"] = "EOD_CLOSE"
            st["entry_price"] = None
            st["shares"] = 0
            st["done"] = True
            _receive_proceeds(proceeds)

    return states, cash, unsettled


# --- MAIN ----

if __name__ == "__main__":
    no_charts = "--no-charts" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-charts"]
    data_dirs = args if args else ["stored_data_combined"]

    print(f"Combined Green Candle Strategy (H + G + A + F + P) Backtest")
    print(f"{'='*70}")
    print(f"  Strategy H: Gap>={H_MIN_GAP_PCT}% + body>={H_MIN_BODY_PCT}% + 2nd green + new hi + vol confirm")
    print(f"              Target: +{H_TARGET_PCT}% | Time Stop: {H_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy G: Gap>={G_MIN_GAP_PCT}% + 2nd green + new hi")
    print(f"              Target: +{G_TARGET_PCT}% | Time Stop: {G_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy A: Gap>{A_MIN_GAP_PCT}% + body>={A_MIN_BODY_PCT}% + 2nd green + new hi")
    print(f"              Target: +{A_TARGET_PCT}% | Time Stop: {A_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy F: Gap>{F_MIN_GAP_PCT}% + 2nd green (catch-all)")
    print(f"              Target: +{F_TARGET_PCT}% | Time Stop: {F_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy P: Gap>{P_MIN_GAP_PCT}% + PM high breakout + pullback + bounce (fallback)")
    print(f"              Partial: sell {P_PARTIAL_SELL_PCT:.0f}% at +{P_TARGET1_PCT}% | Runner: +{P_TARGET2_PCT}%")
    print(f"              Trail: {P_TRAIL_PCT}% (activates +{P_TRAIL_ACTIVATE_PCT}%) | Stop: -{P_STOP_PCT}% | {P_TIME_LIMIT_MINUTES}m")
    print(f"  Priority:   H > G > A > F > P")
    print(f"  Max Positions: {MAX_POSITIONS} (shared, 100% balance)")
    print(f"  No SPY regime filter")
    print(f"  Data: {data_dirs}")
    print(f"{'='*70}\n")

    print("Loading data...")
    all_dates, daily_picks = load_all_picks(data_dirs)
    all_dates = [d for d in all_dates if "2024-01-01" <= d <= "2026-02-28"]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}\n")

    cash = STARTING_CASH
    unsettled_cash = 0.0
    all_results = []

    print(f"{'Date':<12} {'Strat':>5} {'Trades':>6} {'Win':>4} {'Loss':>5} "
          f"{'Day P&L':>12} {'Balance':>14}")
    print("-" * 72)

    for d in all_dates:
        if unsettled_cash > 0:
            cash += unsettled_cash
            unsettled_cash = 0.0

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        states, new_cash, new_unsettled = simulate_day_combined(
            picks, cash, cash_account
        )

        day_pnl = 0.0
        day_trades = 0
        day_wins = 0
        day_losses = 0
        counts = {"H": [0, 0], "G": [0, 0], "A": [0, 0], "F": [0, 0], "P": [0, 0]}

        for st in states:
            if st["exit_reason"] is not None:
                day_trades += 1
                day_pnl += st["pnl"]
                if st["pnl"] > 0:
                    day_wins += 1
                else:
                    day_losses += 1
                s = st["strategy"]
                if s in counts:
                    counts[s][0] += 1
                    if st["pnl"] > 0:
                        counts[s][1] += 1

        cash = new_cash
        unsettled_cash = new_unsettled
        equity = cash + unsettled_cash

        # Build strat label like "H1G1A2F1"
        parts = []
        for key in ["H", "G", "A", "F", "P"]:
            if counts[key][0] > 0:
                parts.append(f"{key}{counts[key][0]}")
        strat_label = "".join(parts) if parts else ""

        print(f"{d:<12} {strat_label:>5} {day_trades:>6} {day_wins:>4} {day_losses:>5} "
              f"${day_pnl:>+11,.0f} ${equity:>13,.0f}")

        # Per-trade detail with % profit
        traded_states = [s for s in states if s["exit_reason"] is not None]
        for st in traded_states:
            pct = (st["pnl"] / st["position_cost"] * 100) if st["position_cost"] > 0 else 0
            reason_short = {"TARGET": "T", "TIME_STOP": "TS", "EOD_CLOSE": "EOD",
                            "STOP": "SL", "TRAIL": "TR"}.get(
                st["exit_reason"], st["exit_reason"][:3])
            vc_tag = " VC" if st.get("vol_capped") else ""
            print(f"  -> [{st['strategy']}] {st['ticker']:<6} {reason_short:<3}  "
                  f"${st['pnl']:>+10,.0f}  ({pct:>+6.2f}%){vc_tag}")

        all_results.append({
            "date": d, "picks": picks, "states": states,
            "day_pnl": day_pnl, "equity": equity, "regime_skip": False,
            "trades": day_trades, "wins": day_wins, "losses": day_losses,
            "h_trades": counts["H"][0], "g_trades": counts["G"][0],
            "a_trades": counts["A"][0], "f_trades": counts["F"][0],
            "p_trades": counts["P"][0],
            "h_wins": counts["H"][1], "g_wins": counts["G"][1],
            "a_wins": counts["A"][1], "f_wins": counts["F"][1],
            "p_wins": counts["P"][1],
        })

    # --- Summary ---
    final_equity = cash + unsettled_cash
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    total_h = sum(r["h_trades"] for r in all_results)
    total_g = sum(r["g_trades"] for r in all_results)
    total_a = sum(r["a_trades"] for r in all_results)
    total_f = sum(r["f_trades"] for r in all_results)
    total_p = sum(r["p_trades"] for r in all_results)
    total_h_wins = sum(r["h_wins"] for r in all_results)
    total_g_wins = sum(r["g_wins"] for r in all_results)
    total_a_wins = sum(r["a_wins"] for r in all_results)
    total_f_wins = sum(r["f_wins"] for r in all_results)
    total_p_wins = sum(r["p_wins"] for r in all_results)
    daily_pnls = [r["day_pnl"] for r in all_results if r["trades"] > 0]
    green = sum(1 for p in daily_pnls if p > 0) if daily_pnls else 0
    red = sum(1 for p in daily_pnls if p <= 0) if daily_pnls else 0
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if daily_pnls and np.std(daily_pnls) > 0 else 0

    all_exits = {}
    all_trade_pnls = []
    strat_pnls = {"H": [], "G": [], "A": [], "F": [], "P": []}
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None:
                reason_key = f"{st['strategy']}_{st['exit_reason']}"
                all_exits[reason_key] = all_exits.get(reason_key, 0) + 1
                all_trade_pnls.append(st["pnl"])
                s = st["strategy"]
                if s in strat_pnls:
                    strat_pnls[s].append(st["pnl"])

    avg_win = np.mean([p for p in all_trade_pnls if p > 0]) if total_wins > 0 else 0
    avg_loss = np.mean([p for p in all_trade_pnls if p <= 0]) if total_losses > 0 else 0

    print(f"\n{'='*70}")
    print(f"  COMBINED STRATEGY SUMMARY (H + G + A + F + P)")
    print(f"{'='*70}")
    print(f"  Starting Cash:    ${STARTING_CASH:,}")
    print(f"  Ending Equity:    ${final_equity:,.0f}  ({(final_equity/STARTING_CASH - 1)*100:+.1f}%)")
    if unsettled_cash > 0:
        print(f"    (Cash: ${cash:,.0f} + Unsettled: ${unsettled_cash:,.0f})")
    print(f"  Trading Days:     {len(all_dates)}")
    print(f"  Total Trades:     {total_trades}")
    print(f"    Winners:        {total_wins} ({total_wins/max(total_trades,1)*100:.1f}%)")
    print(f"    Losers:         {total_losses}")
    print(f"  Avg Win:          ${avg_win:+,.0f}")
    print(f"  Avg Loss:         ${avg_loss:+,.0f}")
    pf = abs(avg_win * total_wins / (avg_loss * total_losses)) if total_losses > 0 and avg_loss != 0 else 0
    print(f"  Profit Factor:    {pf:.2f}" if pf > 0 else "")

    # Per-strategy breakdown
    for label, key, total, wins, target, time_lim in [
        ("H (High Conviction)", "H", total_h, total_h_wins, H_TARGET_PCT, H_TIME_LIMIT_MINUTES),
        ("G (Big Gap Runner)", "G", total_g, total_g_wins, G_TARGET_PCT, G_TIME_LIMIT_MINUTES),
        ("A (Quick Scalp)", "A", total_a, total_a_wins, A_TARGET_PCT, A_TIME_LIMIT_MINUTES),
        ("F (Catch-All)", "F", total_f, total_f_wins, F_TARGET_PCT, F_TIME_LIMIT_MINUTES),
        ("P (PM High Breakout)", "P", total_p, total_p_wins, P_TARGET1_PCT, P_TIME_LIMIT_MINUTES),
    ]:
        print(f"\n  {'---'*17}")
        if key == "P":
            print(f"  STRATEGY {label}: +{P_TARGET1_PCT}/{P_TARGET2_PCT}%, trail {P_TRAIL_PCT}%, {time_lim}m")
        else:
            print(f"  STRATEGY {label}: +{target}%, {time_lim}m")
        print(f"  {'---'*17}")
        print(f"    Trades:  {total}")
        print(f"    Winners: {wins} ({wins/max(total,1)*100:.1f}%)")
        pnls = strat_pnls[key]
        if pnls:
            w = [p for p in pnls if p > 0]
            l = [p for p in pnls if p <= 0]
            print(f"    Total PnL: ${sum(pnls):+,.0f}")
            if w:
                print(f"    Avg Win:   ${np.mean(w):+,.0f}")
            if l:
                print(f"    Avg Loss:  ${np.mean(l):+,.0f}")

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

    # --- STRESS TEST ---
    print(f"\n  {'='*50}")
    print(f"  STRESS TEST")
    print(f"  {'='*50}")

    # Max drawdown
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

    # Consecutive losing days
    streak = 0
    max_streak = 0
    for pnl in daily_pnls:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"  Max Losing Streak: {max_streak} days")

    # Remove top 10% of winning days
    if daily_pnls:
        sorted_pnls = sorted(daily_pnls, reverse=True)
        n_remove = max(1, int(len(daily_pnls) * 0.10))
        remaining = sorted_pnls[n_remove:]
        rem_total = sum(remaining)
        rem_std = np.std(remaining) if len(remaining) > 1 else 1.0
        rem_sharpe = (np.mean(remaining) / rem_std) * np.sqrt(252) if rem_std > 0 else 0
        rem_pass = rem_total > 0
        print(f"  Remove Top 10%:   ${rem_total:+,.0f} (Sharpe {rem_sharpe:.2f}) "
              f"{'PASS' if rem_pass else 'FAIL'}")

    # Kelly criterion
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
        print(f"  Kelly Criterion:  N/A")

    # Monte Carlo: remove random 10% of days, check if still profitable
    np.random.seed(42)
    n_sims = 1000
    n_remove_mc = max(1, int(len(daily_pnls) * 0.10))
    mc_profitable = 0
    all_daily = np.array(daily_pnls)
    for _ in range(n_sims):
        indices = np.random.choice(len(all_daily), size=len(all_daily) - n_remove_mc, replace=False)
        if all_daily[indices].sum() > 0:
            mc_profitable += 1
    mc_pct = mc_profitable / n_sims * 100
    mc_verdict = "PASS" if mc_pct > 80 else ("WEAK" if mc_pct > 50 else "FAIL")
    print(f"  Monte Carlo:      {mc_pct:.1f}% profitable ({mc_verdict})")

    # Volume-capped trade count
    vc_count = sum(1 for r in all_results for s in r["states"]
                   if s["exit_reason"] is not None and s.get("vol_capped"))
    if vc_count > 0:
        print(f"  Vol-Capped Trades: {vc_count}")

    stress_pass = (rem_total > 0 and kelly > 0 and mc_pct > 80) if daily_pnls else False
    print(f"\n  OVERALL: {'ALL PASS' if stress_pass else 'SOME CONCERNS'}")
    print(f"{'='*70}")

    # --- CHARTS ---
    if no_charts:
        print("\n  Skipping charts (--no-charts)")
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("charts", f"gc_combined_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nGenerating charts -> {run_dir}/")

        def _to_et(ts_val):
            try:
                return ts_val.astimezone(ET_TZ)
            except Exception:
                return ts_val

        COLORS_H = ["#9C27B0", "#AB47BC", "#BA68C8"]   # Purple for H
        COLORS_G = ["#4CAF50", "#66BB6A", "#81C784"]   # Greens for G
        COLORS_A = ["#2196F3", "#42A5F5", "#64B5F6"]   # Blues for A
        COLORS_F = ["#FF9800", "#FFA726", "#FFB74D"]    # Oranges for F
        COLORS_P = ["#E91E63", "#EC407A", "#F06292"]    # Pink for P
        DAYS_PER_PAGE = 5

        num_pages = math.ceil(len(all_dates) / DAYS_PER_PAGE)
        for page in range(num_pages):
            start = page * DAYS_PER_PAGE
            end = min(start + DAYS_PER_PAGE, len(all_dates))
            page_results = all_results[start:end]
            n_rows = len(page_results)

            fig, axes = plt.subplots(
                n_rows, 2, figsize=(20, 4.5 * n_rows),
                gridspec_kw={"width_ratios": [2.5, 1]},
            )
            if n_rows == 1:
                axes = [axes]

            fig.suptitle(
                f"Combined H+G+A+F+P Page {page+1}/{num_pages}: "
                f"{page_results[0]['date']} to {page_results[-1]['date']}\n"
                f"H: +{H_TARGET_PCT}%/{H_TIME_LIMIT_MINUTES}m (purple) | "
                f"G: +{G_TARGET_PCT}%/{G_TIME_LIMIT_MINUTES}m (green) | "
                f"A: +{A_TARGET_PCT}%/{A_TIME_LIMIT_MINUTES}m (blue) | "
                f"F: +{F_TARGET_PCT}%/{F_TIME_LIMIT_MINUTES}m (orange) | "
                f"P: +{P_TARGET1_PCT}/{P_TARGET2_PCT}%/trail{P_TRAIL_PCT}%/{P_TIME_LIMIT_MINUTES}m (pink)",
                fontsize=11, fontweight="bold", y=1.01,
            )

            for i, res in enumerate(page_results):
                row_axes = axes[i] if n_rows > 1 else axes[0]
                ax_price, ax_pnl = row_axes[0], row_axes[1]

                traded = [s for s in res["states"] if s["exit_reason"] is not None]

                if traded:
                    color_idx = {"H": 0, "G": 0, "A": 0, "F": 0, "P": 0}
                    color_map = {"H": COLORS_H, "G": COLORS_G, "A": COLORS_A, "F": COLORS_F, "P": COLORS_P}
                    for si, st in enumerate(traded):
                        mh = st["mh"]
                        if mh.index.tz is not None:
                            et_times = mh.index.tz_convert(ET_TZ)
                        else:
                            et_times = mh.index.tz_localize("UTC").tz_convert(ET_TZ)

                        s = st["strategy"] or "A"
                        colors_list = color_map.get(s, COLORS_A)
                        color = colors_list[color_idx.get(s, 0) % len(colors_list)]
                        color_idx[s] = color_idx.get(s, 0) + 1

                        first_candle_close = float(mh.iloc[0]["Close"])
                        ref_price = first_candle_close
                        pct_change = (mh["Close"].values.astype(float) / ref_price - 1) * 100

                        strat_tag = st["strategy"]
                        vc_tag = " VC" if st.get("vol_capped") else ""
                        cost_k = st["position_cost"] / 1000
                        label = (f"[{strat_tag}] {st['ticker']} ${cost_k:.0f}K{vc_tag} "
                                 f"(gap {st['gap_pct']:.0f}%)")
                        ax_price.plot(et_times, pct_change, color=color, linewidth=1.2,
                                      label=label, alpha=0.85)

                        # BUY marker
                        if st["entry_time"] is not None:
                            et_buy = _to_et(st["entry_time"])
                            ax_price.plot(et_buy, 0, marker="^", color=color,
                                          markersize=10, zorder=5)
                            ax_price.annotate(
                                f"BUY({strat_tag})", xy=(et_buy, 0), xytext=(0, 12),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=6, fontweight="bold", color=color,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          ec=color, alpha=0.8, lw=0.5),
                            )

                        # SELL marker
                        if st["exit_time"] is not None and st["exit_price"] is not None:
                            et_sell = _to_et(st["exit_time"])
                            sell_pct = (st["exit_price"] / ref_price - 1) * 100
                            is_win = st["pnl"] > 0
                            marker = "v" if not is_win else "s"
                            sell_color = "#4CAF50" if is_win else "#f44336"
                            ax_price.plot(et_sell, sell_pct, marker=marker, color=sell_color,
                                          markersize=10, zorder=5,
                                          markeredgecolor="white", markeredgewidth=1)
                            reason_short = {"TARGET": "T", "STOP_LOSS": "SL", "STOP": "SL",
                                            "EOD_CLOSE": "EOD", "TIME_STOP": "TS",
                                            "TRAIL": "TR"}.get(
                                st["exit_reason"], st["exit_reason"][:3])
                            ax_price.annotate(
                                reason_short, xy=(et_sell, sell_pct),
                                xytext=(0, -14 if sell_pct >= 0 else 12),
                                textcoords="offset points", ha="center",
                                va="top" if sell_pct >= 0 else "bottom",
                                fontsize=6, fontweight="bold", color=sell_color,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          ec=sell_color, alpha=0.8, lw=0.5),
                            )

                    ax_price.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                    # Show target lines for active strategies this day
                    day_strats = set(s["strategy"] for s in traded)
                    target_lines = {
                        "H": (H_TARGET_PCT, "#9C27B0"), "G": (G_TARGET_PCT, "#4CAF50"),
                        "A": (A_TARGET_PCT, "#2196F3"), "F": (F_TARGET_PCT, "#FF9800"),
                        "P": (P_TARGET1_PCT, "#E91E63"),
                    }
                    shown_targets = set()
                    for sk, (tv, tc_line) in target_lines.items():
                        if sk in day_strats and tv not in shown_targets:
                            ax_price.axhline(y=tv, color=tc_line, linestyle=":", alpha=0.4)
                            shown_targets.add(tv)
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
                        cost_k = s["position_cost"] / 1000
                        tickers.append(f"[{s['strategy']}] {s['ticker']} ${cost_k:.0f}K{vc}")
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
                else:
                    bal = res["equity"]
                    ax_price.text(0.5, 0.5, f"{res['date']}\nNo signals",
                                  ha="center", va="center", fontsize=12,
                                  transform=ax_price.transAxes, color="gray")
                    ax_price.set_title(f"{res['date']} - No Trades", fontsize=10)
                    ax_pnl.text(0.5, 0.5, "0 trades", ha="center", va="center",
                                fontsize=10, transform=ax_pnl.transAxes, color="gray")
                    ax_pnl.set_title(f"P&L: $0 | Bal: ${bal:,.0f}", fontsize=10, color="gray")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            chart_path = os.path.join(run_dir, f"gc_page_{page+1:02d}.png")
            plt.savefig(chart_path, dpi=120, bbox_inches="tight")
            plt.close()
            sys.stdout.write(f"\r  Charts: page {page+1}/{num_pages}")
            sys.stdout.flush()

        print(f"\r  {num_pages} chart pages saved to {run_dir}/          ")

        # --- SUMMARY CHART ---
        fig2 = plt.figure(figsize=(22, 18))
        gs = fig2.add_gridspec(3, 3, width_ratios=[1.2, 1.2, 0.8],
                               height_ratios=[1.0, 1.0, 1.0], hspace=0.35, wspace=0.3)
        fig2.suptitle(
            f"Combined H+G+A+F+P Summary: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)",
            fontsize=16, fontweight="bold",
        )

        # 1. Daily P&L bars
        ax = fig2.add_subplot(gs[0, :])
        dates_list = [r["date"] for r in all_results]
        pnls_list = [r["day_pnl"] for r in all_results]
        bar_c = ["#4CAF50" if p >= 0 else "#f44336" for p in pnls_list]
        ax.bar(range(len(dates_list)), pnls_list, color=bar_c, edgecolor="none", width=0.8)
        ax.set_xticks(range(0, len(dates_list), max(1, len(dates_list) // 15)))
        ax.set_xticklabels([dates_list[i] for i in range(0, len(dates_list), max(1, len(dates_list) // 15))],
                           rotation=45, fontsize=8, ha="right")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        tp = sum(pnls_list)
        tc = "#4CAF50" if tp >= 0 else "#f44336"
        ax.set_title(f"Daily P&L | Total: ${tp:+,.0f}", fontsize=13, fontweight="bold", color=tc)
        ax.set_ylabel("P&L ($)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

        # 2. Equity curve
        ax = fig2.add_subplot(gs[1, :])
        equities = [r["equity"] for r in all_results]
        ax.plot(range(len(dates_list)), equities, color="#2196F3", linewidth=2, label="Combined H+G+A+F+P")
        ax.fill_between(range(len(dates_list)), STARTING_CASH, equities,
                        where=[e >= STARTING_CASH for e in equities], alpha=0.15, color="#4CAF50")
        ax.fill_between(range(len(dates_list)), STARTING_CASH, equities,
                        where=[e < STARTING_CASH for e in equities], alpha=0.15, color="#f44336")
        ax.axhline(y=STARTING_CASH, color="gray", linestyle="--", alpha=0.5,
                   label=f"Start: ${STARTING_CASH:,}")
        ax.axhline(y=MARGIN_THRESHOLD, color="#FF9800", linestyle=":", alpha=0.6,
                   label=f"Margin ${MARGIN_THRESHOLD/1000:.0f}K")
        ax.set_xticks(range(0, len(dates_list), max(1, len(dates_list) // 15)))
        ax.set_xticklabels([dates_list[i] for i in range(0, len(dates_list), max(1, len(dates_list) // 15))],
                           rotation=45, fontsize=8, ha="right")
        ax.set_title(f"Equity: ${STARTING_CASH:,} -> ${equities[-1]:,.0f} ({(equities[-1]/STARTING_CASH-1)*100:+.1f}%)",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Balance ($)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 3. Exit reasons pie (split by strategy)
        ax = fig2.add_subplot(gs[2, 0])
        if all_exits:
            reason_colors = {
                "H_TARGET": "#9C27B0", "H_TIME_STOP": "#BA68C8", "H_EOD_CLOSE": "#CE93D8",
                "G_TARGET": "#4CAF50", "G_TIME_STOP": "#81C784", "G_EOD_CLOSE": "#C8E6C9",
                "A_TARGET": "#2196F3", "A_TIME_STOP": "#64B5F6", "A_EOD_CLOSE": "#BBDEFB",
                "F_TARGET": "#FF9800", "F_TIME_STOP": "#FFB74D", "F_EOD_CLOSE": "#FFE0B2",
                "P_TARGET": "#E91E63", "P_STOP": "#F48FB1", "P_TRAIL": "#F06292",
                "P_TIME_STOP": "#F8BBD0", "P_EOD_CLOSE": "#FCE4EC",
            }
            labels = list(all_exits.keys())
            sizes = list(all_exits.values())
            colors = [reason_colors.get(r, "#999") for r in labels]
            ax.pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
                   colors=colors, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
        ax.set_title("Exit Reasons by Strategy", fontweight="bold")

        # 4. Stats + Stress Test
        ax = fig2.add_subplot(gs[2, 1])
        ax.axis("off")
        stats_text = (
            f"PERFORMANCE\n{'='*38}\n"
            f"Starting:     ${STARTING_CASH:,}\n"
            f"Ending:       ${final_equity:,.0f}\n"
            f"Return:       {(final_equity/STARTING_CASH-1)*100:+.1f}%\n"
            f"{'─'*38}\n"
            f"Total Trades: {total_trades}\n"
            f"  H: {total_h:>3} ({total_h_wins}W {total_h_wins/max(total_h,1)*100:.0f}%)"
            f"  ${sum(strat_pnls['H']):+,.0f}\n"
            f"  G: {total_g:>3} ({total_g_wins}W {total_g_wins/max(total_g,1)*100:.0f}%)"
            f"  ${sum(strat_pnls['G']):+,.0f}\n"
            f"  A: {total_a:>3} ({total_a_wins}W {total_a_wins/max(total_a,1)*100:.0f}%)"
            f"  ${sum(strat_pnls['A']):+,.0f}\n"
            f"  F: {total_f:>3} ({total_f_wins}W {total_f_wins/max(total_f,1)*100:.0f}%)"
            f"  ${sum(strat_pnls['F']):+,.0f}\n"
            f"  P: {total_p:>3} ({total_p_wins}W {total_p_wins/max(total_p,1)*100:.0f}%)"
            f"  ${sum(strat_pnls['P']):+,.0f}\n"
            f"{'─'*38}\n"
            f"Win Rate:     {total_wins/max(total_trades,1)*100:.1f}%\n"
            f"Avg Win:      ${avg_win:+,.0f}\n"
            f"Avg Loss:     ${avg_loss:+,.0f}\n"
            f"Profit Fac:   {pf:.2f}\n"
            f"Sharpe:       {sharpe:.2f}\n"
            f"Green Days:   {green}/{len(daily_pnls)} ({green/max(len(daily_pnls),1)*100:.0f}%)\n"
        )
        if daily_pnls:
            stats_text += (
                f"Best Day:     ${max(daily_pnls):+,.0f}\n"
                f"Worst Day:    ${min(daily_pnls):+,.0f}\n"
            )
        # Stress test section with pass/fail markers
        rem_mark = "OK" if rem_total > 0 else "XX"
        kelly_mark = "OK" if kelly > 0 else "XX"
        mc_mark = "OK" if mc_pct >= 90 else "XX"
        dd_mark = "OK" if max_dd_pct < 0.40 else "XX"
        overall_mark = "OK" if stress_pass else "XX"
        stats_text += (
            f"\n{'='*38}\n"
            f"STRESS TEST\n"
            f"{'='*38}\n"
            f"[{dd_mark}] Max DD:      ${max_dd_dollar:,.0f} ({max_dd_pct*100:.1f}%)\n"
            f"     Loss Streak:  {max_streak} days\n"
        )
        if daily_pnls:
            rem_sharpe = 0.0
            if np.std(daily_pnls) > 0:
                sorted_pnls = sorted(daily_pnls, reverse=True)
                cut = max(1, len(sorted_pnls) // 10)
                rem_pnls = sorted_pnls[cut:]
                if np.std(rem_pnls) > 0:
                    rem_sharpe = (np.mean(rem_pnls) / np.std(rem_pnls)) * np.sqrt(252)
            stats_text += (
                f"[{rem_mark}] Rm Top 10%: ${rem_total:+,.0f} (Sh {rem_sharpe:.2f})\n"
                f"[{kelly_mark}] Kelly:      {kelly*100:.1f}%\n"
                f"[{mc_mark}] Monte Carlo:{mc_pct:.0f}% profitable\n"
            )
        if vc_count > 0:
            stats_text += f"   Vol-Capped:  {vc_count}/{total_trades} trades\n"
        stats_text += f"\n[{overall_mark}] OVERALL: {'ALL PASS' if stress_pass else 'CONCERNS'}\n"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # 5. Config Parameters
        ax = fig2.add_subplot(gs[2, 2])
        ax.axis("off")
        config_text = (
            f"PARAMETERS\n{'='*32}\n"
            f"\nH (High Conviction)\n{'-'*32}\n"
            f"  Gap >= {H_MIN_GAP_PCT:.0f}%\n"
            f"  Body >= {H_MIN_BODY_PCT:.0f}%\n"
            f"  2nd green + new hi\n"
            f"  Vol confirm: {H_REQUIRE_VOL_CONFIRM}\n"
            f"  Target: +{H_TARGET_PCT:.0f}%  Stop: {H_TIME_LIMIT_MINUTES}m\n"
            f"\nG (Runner)\n{'-'*32}\n"
            f"  Gap >= {G_MIN_GAP_PCT:.0f}%\n"
            f"  2nd green + new hi\n"
            f"  Target: +{G_TARGET_PCT:.0f}%  Stop: {G_TIME_LIMIT_MINUTES}m\n"
            f"\nA (Scalp)\n{'-'*32}\n"
            f"  Gap >= {A_MIN_GAP_PCT:.0f}%\n"
            f"  Body >= {A_MIN_BODY_PCT:.0f}%\n"
            f"  2nd green + new hi\n"
            f"  Target: +{A_TARGET_PCT:.0f}%  Stop: {A_TIME_LIMIT_MINUTES}m\n"
            f"\nF (Catch-All)\n{'-'*32}\n"
            f"  Gap >= {F_MIN_GAP_PCT:.0f}%\n"
            f"  2nd green (no hi req)\n"
            f"  Target: +{F_TARGET_PCT:.0f}%  Stop: {F_TIME_LIMIT_MINUTES}m\n"
            f"\nP (PM High Breakout)\n{'-'*32}\n"
            f"  Gap >= {P_MIN_GAP_PCT:.0f}%\n"
            f"  Confirm: {P_CONFIRM_ABOVE}/{P_CONFIRM_WINDOW} closes\n"
            f"  Pullback: {P_PULLBACK_PCT:.1f}%\n"
            f"  Max entry: c{P_MAX_ENTRY_CANDLE}\n"
            f"  Target1: +{P_TARGET1_PCT:.0f}% (sell {P_PARTIAL_SELL_PCT:.0f}%)\n"
            f"  Target2: +{P_TARGET2_PCT:.0f}% (runner)\n"
            f"  Stop: -{P_STOP_PCT:.0f}%  Trail: {P_TRAIL_PCT:.0f}% (at +{P_TRAIL_ACTIVATE_PCT:.0f}%)\n"
            f"  Time: {P_TIME_LIMIT_MINUTES}m\n"
            f"\nSHARED\n{'-'*32}\n"
            f"  Priority:  H > G > A > F > P\n"
            f"  Max pos:   {MAX_POSITIONS}\n"
            f"  Sizing:    100% balance\n"
            f"  Slippage:  {SLIPPAGE_PCT}%\n"
            f"  Vol cap:   {VOL_CAP_PCT}%\n"
            f"  Start:     ${STARTING_CASH:,}\n"
            f"  Margin:    ${MARGIN_THRESHOLD:,}\n"
            f"\n  {all_dates[0]} to {all_dates[-1]}\n"
            f"  {len(all_dates)} trading days\n"
        )
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.7))

        plt.tight_layout()
        summary_path = os.path.join(run_dir, "gc_summary.png")
        plt.savefig(summary_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Summary chart saved to {summary_path}")
