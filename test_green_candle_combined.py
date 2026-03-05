"""
Combined Green Candle Strategy Backtest (A + G + F on shared balance)
====================================================================
Strategy G: Gap>=30% + 2nd green + new hi       -> +10% target, 20m time stop
Strategy A: Gap>20%  + body>=2% + 2nd green + new hi -> +3% target, 10m time stop
Strategy F: Gap>10%  + 2nd green                 -> +7% target, 14m time stop (catch-all)

Priority: G > A > F (highest conviction first).
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

# --- STRATEGY G CONFIG: Big Gap Runner ---
G_MIN_GAP_PCT = 25.0             # Optuna: lowered from 30% to catch more runners
G_MIN_BODY_PCT = 0.0             # No body filter
G_REQUIRE_2ND_GREEN = True
G_REQUIRE_2ND_NEW_HIGH = True
G_TARGET_PCT = 8.0               # Optuna: lowered from 10% (take profit sooner)
G_TIME_LIMIT_MINUTES = 20

# --- STRATEGY A CONFIG: Quick Scalp ---
A_MIN_GAP_PCT = 15.0             # Optuna: lowered from 20% (wider net)
A_MIN_BODY_PCT = 0.0             # Optuna: removed body filter
A_MAX_BODY_PCT = 999
A_REQUIRE_2ND_GREEN = True
A_REQUIRE_2ND_NEW_HIGH = True
A_TARGET_PCT = 3.0
A_TIME_LIMIT_MINUTES = 10

# --- STRATEGY F CONFIG: Catch-All ---
F_MIN_GAP_PCT = 10.0
F_MIN_BODY_PCT = 0.0             # No body filter
F_REQUIRE_2ND_GREEN = True
F_REQUIRE_2ND_NEW_HIGH = False
F_TARGET_PCT = 10.0              # Optuna: raised from 7% (bigger swings)
F_TIME_LIMIT_MINUTES = 3         # Optuna: 3m (slightly more room than 2m)

# --- SHARED CONFIG ---
EOD_EXIT_MINUTES = 15
MAX_POSITIONS = 1               # 100% of balance per trade
FULL_BALANCE_SIZING = True      # Use full cash for each trade


def _classify_candle2(gap_pct, body_pct, second_green, second_new_high):
    """Classify on candle 2 for strategies G, A, F.
    Priority: G > A > F (highest target first)."""
    # G: gap>=30%, 2nd green + new hi (strongest runners)
    if (gap_pct >= G_MIN_GAP_PCT
            and body_pct >= G_MIN_BODY_PCT
            and (not G_REQUIRE_2ND_GREEN or second_green)
            and (not G_REQUIRE_2ND_NEW_HIGH or second_new_high)):
        return "G"
    # A: gap>20%, body>=2%, 2nd green + new hi (quick scalp)
    if (gap_pct >= A_MIN_GAP_PCT
            and A_MIN_BODY_PCT <= body_pct <= A_MAX_BODY_PCT
            and (not A_REQUIRE_2ND_GREEN or second_green)
            and (not A_REQUIRE_2ND_NEW_HIGH or second_new_high)):
        return "A"
    # F: gap>10%, 2nd green (catch-all)
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
            # State
            "candle_count": 0,
            "first_candle_ok": False,
            "first_candle_high": 0.0,
            "first_candle_body_pct": 0.0,
            "signal": False,
            "signal_price": None,
            "open_price": None,
            "strategy": None,  # "G", "A", or "F"
            # Position
            "entry_price": None,
            "entry_time": None,
            "exit_price": None,
            "exit_time": None,
            "exit_reason": None,
            "shares": 0,
            "position_cost": 0.0,
            "pnl": 0.0,
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

                # Strategy-specific params
                strat_map = {
                    "G": (G_TARGET_PCT, G_TIME_LIMIT_MINUTES),
                    "A": (A_TARGET_PCT, A_TIME_LIMIT_MINUTES),
                    "F": (F_TARGET_PCT, F_TIME_LIMIT_MINUTES),
                }
                target_pct, time_limit = strat_map.get(st["strategy"], (A_TARGET_PCT, A_TIME_LIMIT_MINUTES))

                # EOD forced exit
                if minutes_to_close <= EOD_EXIT_MINUTES:
                    sell_price = c_close * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["shares"] * sell_price
                    st["pnl"] = proceeds - st["position_cost"]
                    st["exit_price"] = c_close
                    st["exit_time"] = ts
                    st["exit_reason"] = "EOD_CLOSE"
                    st["entry_price"] = None
                    st["shares"] = 0
                    st["done"] = True
                    _receive_proceeds(proceeds)
                    continue

                # Target hit
                target_price = st["entry_price"] * (1 + target_pct / 100)
                if c_high >= target_price:
                    sell_price = target_price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["shares"] * sell_price
                    st["pnl"] = proceeds - st["position_cost"]
                    st["exit_price"] = target_price
                    st["exit_time"] = ts
                    st["exit_reason"] = "TARGET"
                    st["entry_price"] = None
                    st["shares"] = 0
                    st["done"] = True
                    _receive_proceeds(proceeds)
                    continue

                # Time stop
                if minutes_in_trade >= time_limit:
                    sell_price = c_close * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["shares"] * sell_price
                    st["pnl"] = proceeds - st["position_cost"]
                    st["exit_price"] = c_close
                    st["exit_time"] = ts
                    st["exit_reason"] = "TIME_STOP"
                    st["entry_price"] = None
                    st["shares"] = 0
                    st["done"] = True
                    _receive_proceeds(proceeds)
                    continue

                continue

            # --- NOT IN POSITION: candle-by-candle signal detection ---
            st["candle_count"] += 1

            # CANDLE 1: Check first candle is green
            if st["candle_count"] == 1:
                if c_open > 0:
                    body_pct = (c_close / c_open - 1) * 100
                    st["first_candle_body_pct"] = body_pct
                    if body_pct > 0:  # First candle must be green
                        st["first_candle_ok"] = True
                        st["first_candle_high"] = c_high
                        st["signal_price"] = c_close
                        st["open_price"] = c_open
                if not st["first_candle_ok"]:
                    st["done"] = True

            # CANDLE 2: Classify for G, A, or F
            elif st["candle_count"] == 2 and st["first_candle_ok"]:
                second_green = c_close > c_open
                second_new_high = c_high > st["first_candle_high"]

                strategy = _classify_candle2(
                    st["gap_pct"], st["first_candle_body_pct"],
                    second_green, second_new_high,
                )
                if strategy:
                    st["strategy"] = strategy
                    st["signal"] = True
                    entry_candidates.append(st)
                else:
                    st["done"] = True

            # Retry fill on later candles
            elif st["signal"] and st["entry_price"] is None and not st["done"]:
                if c_close > 0:
                    st["signal_price"] = c_close
                    entry_candidates.append(st)

        # --- PASS 2: Allocate capital ---
        # Sort: G first (biggest target), then A (scalp), then F (catch-all)
        strat_priority = {"G": 0, "A": 1, "F": 2}
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
            st["pnl"] = proceeds - st["position_cost"]
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
    data_dirs = args if args else ["stored_data_oos", "stored_data"]

    print(f"Combined Green Candle Strategy (G + A + F) Backtest")
    print(f"{'='*70}")
    print(f"  Strategy G: Gap>={G_MIN_GAP_PCT}% + 2nd green + new hi")
    print(f"              Target: +{G_TARGET_PCT}% | Time Stop: {G_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy A: Gap>{A_MIN_GAP_PCT}% + body>={A_MIN_BODY_PCT}% + 2nd green + new hi")
    print(f"              Target: +{A_TARGET_PCT}% | Time Stop: {A_TIME_LIMIT_MINUTES} min")
    print(f"  Strategy F: Gap>{F_MIN_GAP_PCT}% + 2nd green (catch-all)")
    print(f"              Target: +{F_TARGET_PCT}% | Time Stop: {F_TIME_LIMIT_MINUTES} min")
    print(f"  Priority:   G > A > F")
    print(f"  Max Positions: {MAX_POSITIONS} (shared, 100% balance)")
    print(f"  No SPY regime filter")
    print(f"  Data: {data_dirs}")
    print(f"{'='*70}\n")

    print("Loading data...")
    all_dates, daily_picks = load_all_picks(data_dirs)
    all_dates = [d for d in all_dates if "2025-10-01" <= d <= "2026-02-28"]
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
        counts = {"G": [0, 0], "A": [0, 0], "F": [0, 0]}  # [trades, wins]

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

        # Build strat label like "G1A2F1"
        parts = []
        for key in ["G", "A", "F"]:
            if counts[key][0] > 0:
                parts.append(f"{key}{counts[key][0]}")
        strat_label = "".join(parts) if parts else ""

        print(f"{d:<12} {strat_label:>5} {day_trades:>6} {day_wins:>4} {day_losses:>5} "
              f"${day_pnl:>+11,.0f} ${equity:>13,.0f}")

        all_results.append({
            "date": d, "picks": picks, "states": states,
            "day_pnl": day_pnl, "equity": equity, "regime_skip": False,
            "trades": day_trades, "wins": day_wins, "losses": day_losses,
            "g_trades": counts["G"][0], "a_trades": counts["A"][0], "f_trades": counts["F"][0],
            "g_wins": counts["G"][1], "a_wins": counts["A"][1], "f_wins": counts["F"][1],
        })

    # --- Summary ---
    final_equity = cash + unsettled_cash
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    total_g = sum(r["g_trades"] for r in all_results)
    total_a = sum(r["a_trades"] for r in all_results)
    total_f = sum(r["f_trades"] for r in all_results)
    total_g_wins = sum(r["g_wins"] for r in all_results)
    total_a_wins = sum(r["a_wins"] for r in all_results)
    total_f_wins = sum(r["f_wins"] for r in all_results)
    daily_pnls = [r["day_pnl"] for r in all_results if r["trades"] > 0]
    green = sum(1 for p in daily_pnls if p > 0) if daily_pnls else 0
    red = sum(1 for p in daily_pnls if p <= 0) if daily_pnls else 0
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if daily_pnls and np.std(daily_pnls) > 0 else 0

    all_exits = {}
    all_trade_pnls = []
    strat_pnls = {"G": [], "A": [], "F": []}
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
    print(f"  COMBINED STRATEGY SUMMARY (G + A + F)")
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
        ("G (Big Gap Runner)", "G", total_g, total_g_wins, G_TARGET_PCT, G_TIME_LIMIT_MINUTES),
        ("A (Quick Scalp)", "A", total_a, total_a_wins, A_TARGET_PCT, A_TIME_LIMIT_MINUTES),
        ("F (Catch-All)", "F", total_f, total_f_wins, F_TARGET_PCT, F_TIME_LIMIT_MINUTES),
    ]:
        print(f"\n  {'---'*17}")
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

        COLORS_G = ["#4CAF50", "#66BB6A", "#81C784"]   # Greens for G
        COLORS_A = ["#2196F3", "#42A5F5", "#64B5F6"]   # Blues for A
        COLORS_F = ["#FF9800", "#FFA726", "#FFB74D"]    # Oranges for F
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
                f"Combined G+A+F Page {page+1}/{num_pages}: "
                f"{page_results[0]['date']} to {page_results[-1]['date']}\n"
                f"G: +{G_TARGET_PCT}%/{G_TIME_LIMIT_MINUTES}m (green) | "
                f"A: +{A_TARGET_PCT}%/{A_TIME_LIMIT_MINUTES}m (blue) | "
                f"F: +{F_TARGET_PCT}%/{F_TIME_LIMIT_MINUTES}m (orange)",
                fontsize=12, fontweight="bold", y=1.01,
            )

            for i, res in enumerate(page_results):
                row_axes = axes[i] if n_rows > 1 else axes[0]
                ax_price, ax_pnl = row_axes[0], row_axes[1]

                traded = [s for s in res["states"] if s["exit_reason"] is not None]

                if traded:
                    color_idx = {"G": 0, "A": 0, "F": 0}
                    color_map = {"G": COLORS_G, "A": COLORS_A, "F": COLORS_F}
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
                        label = (f"[{strat_tag}] {st['ticker']} "
                                 f"(gap {st['gap_pct']:.0f}%, body {st['first_candle_body_pct']:.1f}%)")
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
                            reason_short = {"TARGET": "T", "STOP_LOSS": "SL",
                                            "EOD_CLOSE": "EOD", "TIME_STOP": "TS"}.get(
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
                    # Show target lines for each strategy
                    ax_price.axhline(y=G_TARGET_PCT, color="#4CAF50", linestyle=":",
                                     alpha=0.4, label=f"G: +{G_TARGET_PCT}%")
                    ax_price.axhline(y=F_TARGET_PCT, color="#FF9800", linestyle=":",
                                     alpha=0.4, label=f"F: +{F_TARGET_PCT}%")
                    ax_price.axhline(y=A_TARGET_PCT, color="#2196F3", linestyle=":",
                                     alpha=0.4, label=f"A: +{A_TARGET_PCT}%")
                    ax_price.set_title(f"{res['date']} - Price (% from 1st candle close)",
                                       fontsize=10, fontweight="bold")
                    ax_price.set_ylabel("% from Entry", fontsize=8)
                    ax_price.legend(fontsize=6, loc="upper left", ncol=2)
                    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p", tz=ET_TZ))
                    ax_price.tick_params(axis="x", rotation=30, labelsize=7)
                    ax_price.grid(alpha=0.3)

                    # P&L bar chart
                    tickers = [f"[{s['strategy']}] {s['ticker']}" for s in traded]
                    pnls = [s["pnl"] for s in traded]
                    reasons = [s["exit_reason"] for s in traded]
                    bar_colors = ["#4CAF50" if p > 0 else "#f44336" for p in pnls]
                    y_pos = range(len(tickers))
                    bars = ax_pnl.barh(y_pos, pnls, color=bar_colors, edgecolor="white", height=0.6)
                    ax_pnl.set_yticks(y_pos)
                    ax_pnl.set_yticklabels([f"{t} ({r[:3]})" for t, r in zip(tickers, reasons)], fontsize=7)
                    ax_pnl.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
                    ax_pnl.invert_yaxis()
                    for j, (bar, pnl) in enumerate(zip(bars, pnls)):
                        x_pos = bar.get_width()
                        align = "left" if pnl >= 0 else "right"
                        offset = 5 if pnl >= 0 else -5
                        ax_pnl.annotate(f"${pnl:+.0f}", xy=(x_pos, j), xytext=(offset, 0),
                                        textcoords="offset points", ha=align, va="center",
                                        fontsize=7, fontweight="bold", color=bar_colors[j])
                    day_total = sum(pnls)
                    tc = "#4CAF50" if day_total >= 0 else "#f44336"
                    ax_pnl.set_title(f"P&L | Day: ${day_total:+,.0f}",
                                     fontsize=10, fontweight="bold", color=tc)
                    ax_pnl.set_xlabel("P&L ($)", fontsize=8)
                    ax_pnl.grid(alpha=0.3, axis="x")
                else:
                    ax_price.text(0.5, 0.5, f"{res['date']}\nNo signals",
                                  ha="center", va="center", fontsize=12,
                                  transform=ax_price.transAxes, color="gray")
                    ax_price.set_title(f"{res['date']} - No Trades", fontsize=10)
                    ax_pnl.text(0.5, 0.5, "0 trades", ha="center", va="center",
                                fontsize=10, transform=ax_pnl.transAxes, color="gray")
                    ax_pnl.set_title("P&L: $0", fontsize=10, color="gray")

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
            f"Combined G+A+F Summary: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)",
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
        ax.plot(range(len(dates_list)), equities, color="#2196F3", linewidth=2, label="Combined G+A+F")
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
                "G_TARGET": "#4CAF50", "G_TIME_STOP": "#81C784", "G_EOD_CLOSE": "#C8E6C9",
                "A_TARGET": "#2196F3", "A_TIME_STOP": "#64B5F6", "A_EOD_CLOSE": "#BBDEFB",
                "F_TARGET": "#FF9800", "F_TIME_STOP": "#FFB74D", "F_EOD_CLOSE": "#FFE0B2",
            }
            labels = list(all_exits.keys())
            sizes = list(all_exits.values())
            colors = [reason_colors.get(r, "#999") for r in labels]
            ax.pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
                   colors=colors, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
        ax.set_title("Exit Reasons by Strategy", fontweight="bold")

        # 4. Stats
        ax = fig2.add_subplot(gs[2, 1])
        ax.axis("off")
        cum_pnl = np.cumsum(pnls_list)
        max_dd = min(cum_pnl) if len(cum_pnl) > 0 else 0
        stats_text = (
            f"COMBINED PERFORMANCE\n{'='*35}\n"
            f"Starting:    ${STARTING_CASH:,}\n"
            f"Ending:      ${final_equity:,.0f}\n"
            f"Return:      {(final_equity/STARTING_CASH-1)*100:+.1f}%\n"
            f"Total P&L:   ${tp:+,.0f}\n"
            f"{'='*35}\n"
            f"Total Trades: {total_trades}\n"
            f"  Strategy G: {total_g} ({total_g_wins}W)\n"
            f"  Strategy A: {total_a} ({total_a_wins}W)\n"
            f"  Strategy F: {total_f} ({total_f_wins}W)\n"
            f"Win Rate:    {total_wins/max(total_trades,1)*100:.1f}%\n"
            f"{'='*35}\n"
            f"Avg Win:     ${avg_win:+,.0f}\n"
            f"Avg Loss:    ${avg_loss:+,.0f}\n"
            f"Profit Fac:  {pf:.2f}\n"
            f"Sharpe:      {sharpe:.2f}\n"
            f"{'='*35}\n"
            f"Green Days:  {green}/{len(daily_pnls)}\n"
        )
        if daily_pnls:
            stats_text += (
                f"Best Day:    ${max(daily_pnls):+,.0f}\n"
                f"Worst Day:   ${min(daily_pnls):+,.0f}\n"
            )
        stats_text += f"Max DD:      ${max_dd:+,.0f}\n"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # 5. Config
        ax = fig2.add_subplot(gs[2, 2])
        ax.axis("off")
        config_text = (
            f"STRATEGY CONFIG\n{'='*30}\n"
            f"\nSTRATEGY G (Runner)\n{'-'*30}\n"
            f"Gap:     >={G_MIN_GAP_PCT}%\n"
            f"2nd:     green + new hi\n"
            f"Target:  +{G_TARGET_PCT}%\n"
            f"Time:    {G_TIME_LIMIT_MINUTES} min\n"
            f"\nSTRATEGY A (Quick Scalp)\n{'-'*30}\n"
            f"Gap:     >{A_MIN_GAP_PCT}%\n"
            f"Body:    >={A_MIN_BODY_PCT}%\n"
            f"2nd:     green + new hi\n"
            f"Target:  +{A_TARGET_PCT}%\n"
            f"Time:    {A_TIME_LIMIT_MINUTES} min\n"
            f"\nSTRATEGY F (Catch-All)\n{'-'*30}\n"
            f"Gap:     >{F_MIN_GAP_PCT}%\n"
            f"2nd:     green\n"
            f"Target:  +{F_TARGET_PCT}%\n"
            f"Time:    {F_TIME_LIMIT_MINUTES} min\n"
            f"\nSHARED\n{'-'*30}\n"
            f"Max pos: {MAX_POSITIONS}\n"
            f"EOD:     {EOD_EXIT_MINUTES}min\n"
            f"Size:    100% of cash\n"
            f"Start:   ${STARTING_CASH:,}\n"
            f"Slip:    {SLIPPAGE_PCT}%\n"
            f"Regime:  NONE\n"
            f"\nPriority: G > A > F\n"
            f"\nDATA\n{'-'*30}\n"
            f"{all_dates[0]} to\n"
            f"{all_dates[-1]}\n"
            f"{len(all_dates)} days\n"
            f"Run: {run_ts}\n"
        )
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.7))

        plt.tight_layout()
        summary_path = os.path.join(run_dir, "gc_summary.png")
        plt.savefig(summary_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Summary chart saved to {summary_path}")
