"""
First Green Candle Strategy Backtest
=====================================
Entry: Gap >10%, first candle (9:30-9:32) closes green with >2% body → buy at 9:32 close.
Exit:  Sell 100% at +5% from entry price, or stop loss, or EOD.

Based on analysis showing 94.5% of these stocks hit +5% within 15 min.

Usage:
  python test_green_candle.py                     # uses stored_data_oos + stored_data
  python test_green_candle.py stored_data_combined # specific dir
  python test_green_candle.py --no-charts          # skip charts
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

# --- STRATEGY CONFIG ---
GREEN_CANDLE_MIN_BODY_PCT = 2.0  # First candle body >= 2%
GREEN_CANDLE_MAX_BODY_PCT = 999  # No cap
MIN_GAP_PCT = 20.0               # Gap filter (>20%)
TARGET_PCT = 3.0                 # Sell 100% at +3%
STOP_PCT = 0                     # No price stop — time-based exit instead
TIME_LIMIT_MINUTES = 5           # Sell everything that hasn't hit target after 5 min
EOD_EXIT_MINUTES = 15            # Close positions 15 min before close
MIN_PM_VOLUME = 0                # Use all candidates
MAX_POSITIONS = 3                # Max simultaneous positions per day
REQUIRE_2ND_GREEN = True         # 2nd candle must also be green
REQUIRE_2ND_NEW_HIGH = True      # 2nd candle must make new high above 1st candle


def simulate_day_green_candle(picks, starting_cash, cash_account=False):
    """Simulate First Green Candle strategy for one day.

    Logic:
    1. Wait for first candle (9:30-9:32) to complete
    2. If first candle body > GREEN_CANDLE_MIN_BODY_PCT%, signal entry
    3. Rank signals by first_candle_body (strongest green first)
    4. Enter up to MAX_POSITIONS at first candle close price
    5. Exit at +TARGET_PCT%, -STOP_PCT%, or EOD

    Returns: (states, ending_cash, unsettled_cash)
    """
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
            "first_candle_ok": False,   # 1st candle passed body filter
            "first_candle_high": 0.0,   # 1st candle high for new high check
            "signal": False,            # Entry confirmed (after 2nd candle check)
            "first_candle_body_pct": 0.0,
            "signal_price": None,       # Entry price (1st candle close)
            "open_price": None,
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

    first_ts = all_timestamps[0] if all_timestamps else None

    for ts in all_timestamps:
        # --- PASS 1: Process exits + detect signals ---
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

            # --- IN POSITION: check exits ---
            if st["entry_price"] is not None:
                try:
                    ts_et = ts.astimezone(ET_TZ)
                except Exception:
                    ts_et = ts
                minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)

                # Minutes since entry
                entry_et = st["entry_time"]
                try:
                    entry_et = entry_et.astimezone(ET_TZ)
                except Exception:
                    pass
                minutes_in_trade = (ts_et.hour * 60 + ts_et.minute) - (entry_et.hour * 60 + entry_et.minute)

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

                # Target hit: +5% from entry
                target_price = st["entry_price"] * (1 + TARGET_PCT / 100)
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

                # Time stop: sell at market after TIME_LIMIT_MINUTES
                if minutes_in_trade >= TIME_LIMIT_MINUTES:
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

                continue  # Position stays open

            # --- NOT IN POSITION: candle-by-candle signal detection ---
            st["candle_count"] += 1

            # CANDLE 1: Check body filter
            if st["candle_count"] == 1:
                if c_open > 0:
                    body_pct = (c_close / c_open - 1) * 100
                    st["first_candle_body_pct"] = body_pct
                    if GREEN_CANDLE_MIN_BODY_PCT <= body_pct <= GREEN_CANDLE_MAX_BODY_PCT:
                        st["first_candle_ok"] = True
                        st["first_candle_high"] = c_high
                        st["signal_price"] = c_close  # Will enter at 1st candle close
                        st["open_price"] = c_open
                if not st["first_candle_ok"]:
                    st["done"] = True

            # CANDLE 2: Check 2nd candle confirmation
            elif st["candle_count"] == 2 and st["first_candle_ok"]:
                passed = True
                if REQUIRE_2ND_GREEN and c_close <= c_open:
                    passed = False
                if REQUIRE_2ND_NEW_HIGH and c_high <= st["first_candle_high"]:
                    passed = False
                if passed:
                    st["signal"] = True
                    entry_candidates.append(st)
                else:
                    st["done"] = True

            # If signaled but couldn't fill (no cash), try next candle
            elif st["signal"] and st["entry_price"] is None and not st["done"]:
                if c_close > 0:
                    st["signal_price"] = c_close
                    entry_candidates.append(st)

        # --- PASS 2: Allocate capital ---
        # Rank by first candle body strength (strongest green first)
        entry_candidates.sort(key=lambda s: -s["first_candle_body_pct"])

        positions_today = sum(1 for s in states if s["entry_price"] is not None)

        for st in entry_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue
            if positions_today >= MAX_POSITIONS:
                break

            # No entries near EOD
            try:
                ts_et = ts.astimezone(ET_TZ)
            except Exception:
                ts_et = ts
            mins_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
            if mins_to_close <= EOD_EXIT_MINUTES:
                continue

            trade_size = starting_cash * trade_pct
            if cash < trade_size:
                if cash > 100:
                    trade_size = cash
                else:
                    continue

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

    # EOD: close any remaining positions
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


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    no_charts = "--no-charts" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-charts"]
    data_dirs = args if args else ["stored_data_oos", "stored_data"]

    print(f"First Green Candle Strategy Backtest")
    print(f"{'='*70}")
    entry_desc = f"Gap>{MIN_GAP_PCT}% + 1st body>={GREEN_CANDLE_MIN_BODY_PCT}%"
    if REQUIRE_2ND_GREEN:
        entry_desc += " + 2nd green"
    if REQUIRE_2ND_NEW_HIGH:
        entry_desc += " + new hi"
    print(f"  Entry: {entry_desc}")
    print(f"  Target: +{TARGET_PCT}%  |  Time Stop: {TIME_LIMIT_MINUTES} min  |  Max Positions: {MAX_POSITIONS}")
    print(f"  EOD Exit: {EOD_EXIT_MINUTES} min before close")
    print(f"  Data: {data_dirs}")
    print(f"{'='*70}\n")

    print("Loading data...")
    all_dates, daily_picks = load_all_picks(data_dirs)

    # Filter to Oct-Feb
    all_dates = [d for d in all_dates if "2025-10-01" <= d <= "2026-02-28"]
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}\n")

    # Regime filter
    print("Loading regime filter...")
    rf = RegimeFilter(
        spy_ma_period=SPY_SMA_PERIOD,
        enable_vix=False,
        enable_spy_trend=True,
        enable_adaptive=False,
    )
    rf.load_data(all_dates[0], all_dates[-1])
    tradeable = sum(1 for d in all_dates if rf.check(d)[0])
    skipped = len(all_dates) - tradeable
    print(f"  {tradeable} tradeable, {skipped} skipped (SPY < SMA{SPY_SMA_PERIOD})\n")

    # Rolling cash simulation
    cash = STARTING_CASH
    unsettled_cash = 0.0
    all_results = []

    print(f"{'Date':<12} {'Trades':>6} {'Win':>4} {'Loss':>5} "
          f"{'Size':>5} {'Day P&L':>12} {'Balance':>14}")
    print("-" * 72)

    for d in all_dates:
        # Settle T+1 cash
        if unsettled_cash > 0:
            cash += unsettled_cash
            unsettled_cash = 0.0

        # Regime check
        should_trade, _, _ = rf.check(d)
        if not should_trade:
            equity = cash + unsettled_cash
            print(f"{d:<12} {'SKIP':>6} {'':>4} {'':>5} "
                  f"{'':>5} {'$+0':>12} ${equity:>13,.0f}  <-- SPY < SMA{SPY_SMA_PERIOD}")
            all_results.append({
                "date": d, "picks": daily_picks.get(d, []), "states": [],
                "day_pnl": 0.0, "equity": equity, "regime_skip": True,
                "trades": 0, "wins": 0, "losses": 0,
            })
            continue

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        states, new_cash, new_unsettled = simulate_day_green_candle(
            picks, cash, cash_account
        )

        # Compute stats
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

        cash = new_cash
        unsettled_cash = new_unsettled
        equity = cash + unsettled_cash

        trade_pct_label = f"{_get_trade_pct(cash)*100:.0f}%"
        print(f"{d:<12} {day_trades:>6} {day_wins:>4} {day_losses:>5} "
              f"{trade_pct_label:>5} ${day_pnl:>+11,.0f} ${equity:>13,.0f}")

        all_results.append({
            "date": d, "picks": picks, "states": states,
            "day_pnl": day_pnl, "equity": equity, "regime_skip": False,
            "trades": day_trades, "wins": day_wins, "losses": day_losses,
        })

    # ─── Summary ─────────────────────────────────────────────────────────
    final_equity = cash + unsettled_cash
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    daily_pnls = [r["day_pnl"] for r in all_results if r["trades"] > 0]
    green = sum(1 for p in daily_pnls if p > 0) if daily_pnls else 0
    red = sum(1 for p in daily_pnls if p <= 0) if daily_pnls else 0
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if daily_pnls and np.std(daily_pnls) > 0 else 0

    # Exit reason breakdown
    all_exits = {}
    all_trade_pnls = []
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None:
                all_exits[st["exit_reason"]] = all_exits.get(st["exit_reason"], 0) + 1
                all_trade_pnls.append(st["pnl"])

    avg_win = np.mean([p for p in all_trade_pnls if p > 0]) if total_wins > 0 else 0
    avg_loss = np.mean([p for p in all_trade_pnls if p <= 0]) if total_losses > 0 else 0

    print(f"\n{'='*70}")
    print(f"  FIRST GREEN CANDLE STRATEGY SUMMARY")
    print(f"{'='*70}")
    print(f"  Starting Cash:    ${STARTING_CASH:,}")
    print(f"  Ending Equity:    ${final_equity:,.0f}  ({(final_equity/STARTING_CASH - 1)*100:+.1f}%)")
    if unsettled_cash > 0:
        print(f"    (Cash: ${cash:,.0f} + Unsettled: ${unsettled_cash:,.0f})")
    print(f"  Trading Days:     {len(all_dates)} ({tradeable} traded, {skipped} skipped)")
    print(f"  Total Trades:     {total_trades}")
    print(f"    Winners:        {total_wins} ({total_wins/max(total_trades,1)*100:.1f}%)")
    print(f"    Losers:         {total_losses}")
    print(f"  Avg Win:          ${avg_win:+,.0f}")
    print(f"  Avg Loss:         ${avg_loss:+,.0f}")
    print(f"  Profit Factor:    {abs(avg_win*total_wins/(avg_loss*total_losses)):.2f}" if total_losses > 0 and avg_loss != 0 else "")
    print(f"\n  Exit Reasons:")
    for reason, count in sorted(all_exits.items(), key=lambda x: -x[1]):
        print(f"    {reason:<15} {count:>4} ({count/max(total_trades,1)*100:.1f}%)")

    if daily_pnls:
        print(f"\n  Green Days:       {green}/{len(daily_pnls)} ({green/len(daily_pnls)*100:.1f}%)")
        print(f"  Red Days:         {red}/{len(daily_pnls)}")
        print(f"  Best Day:         ${max(daily_pnls):+,.0f}")
        print(f"  Worst Day:        ${min(daily_pnls):+,.0f}")
        print(f"  Avg P&L/Day:      ${np.mean(daily_pnls):+,.0f}")
        print(f"  Sharpe (ann.):    {sharpe:.2f}")

    print(f"\n  CONFIG:")
    print(f"    Entry:          {entry_desc}")
    print(f"    Target:         +{TARGET_PCT}%")
    print(f"    Time Stop:      {TIME_LIMIT_MINUTES} min (no price stop)")
    print(f"    Max Positions:  {MAX_POSITIONS}")
    print(f"    EOD Exit:       {EOD_EXIT_MINUTES} min before close")
    print(f"    Regime:         SPY > SMA({SPY_SMA_PERIOD})")
    print(f"{'='*70}")

    # ─── CHARTS ──────────────────────────────────────────────────────────
    if no_charts:
        print("\n  Skipping charts (--no-charts)")
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("charts", f"green_candle_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nGenerating charts -> {run_dir}/")

        def _to_et(ts_val):
            try:
                return ts_val.astimezone(ET_TZ)
            except Exception:
                return ts_val

        COLORS = [
            "#2196F3", "#FF9800", "#4CAF50", "#f44336", "#9C27B0",
            "#00BCD4", "#795548", "#607D8B", "#E91E63", "#CDDC39",
        ]
        DAYS_PER_PAGE = 5

        # --- Per-day chart pages ---
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
                f"Green Candle Page {page+1}/{num_pages}: "
                f"{page_results[0]['date']} to {page_results[-1]['date']}\n"
                f"Entry: 1st body>={GREEN_CANDLE_MIN_BODY_PCT}% + 2nd green | "
                f"Target: +{TARGET_PCT}% | Time Stop: {TIME_LIMIT_MINUTES} min",
                fontsize=12, fontweight="bold", y=1.01,
            )

            for i, res in enumerate(page_results):
                row_axes = axes[i] if n_rows > 1 else axes[0]
                ax_price, ax_pnl = row_axes[0], row_axes[1]

                if res["regime_skip"]:
                    ax_price.text(0.5, 0.5, f"{res['date']}\nSKIPPED (SPY < SMA{SPY_SMA_PERIOD})",
                                  ha="center", va="center", fontsize=12,
                                  transform=ax_price.transAxes, color="#FF9800")
                    ax_price.set_title(f"{res['date']} - Regime Skip", fontsize=10, color="#FF9800")
                    ax_pnl.text(0.5, 0.5, "Regime filter\nNo trades", ha="center", va="center",
                                fontsize=10, transform=ax_pnl.transAxes, color="#FF9800")
                    ax_pnl.set_title("P&L: $0 (skipped)", fontsize=10, color="#FF9800")
                    continue

                traded = [s for s in res["states"] if s["exit_reason"] is not None]

                if traded:
                    for si, st in enumerate(traded):
                        mh = st["mh"]
                        if mh.index.tz is not None:
                            et_times = mh.index.tz_convert(ET_TZ)
                        else:
                            et_times = mh.index.tz_localize("UTC").tz_convert(ET_TZ)

                        color = COLORS[si % len(COLORS)]
                        ref_price = st.get("exit_price", st.get("position_cost", 1))
                        # Use entry price from position cost / shares
                        orig_entry = st["position_cost"] / st["shares"] if st["shares"] > 0 else (
                            st["position_cost"] / max(st.get("position_cost", 1) / max(st.get("exit_price", 1), 1), 1)
                        )
                        # Better: reconstruct entry price from exit_price and pnl
                        if st["entry_time"] is not None and st["exit_price"] is not None:
                            # entry_price was stored but then cleared; reconstruct
                            orig_entry = st["position_cost"] / (st["position_cost"] / st["exit_price"] * (1 + SLIPPAGE_PCT/100)) if st["exit_price"] > 0 else st["exit_price"]

                        # Simplest: just use first candle close as reference
                        first_candle_close = float(mh.iloc[0]["Close"])
                        ref_price = first_candle_close

                        pct_change = (mh["Close"].values.astype(float) / ref_price - 1) * 100
                        label = f"{st['ticker']} (gap {st['gap_pct']:.0f}%, body {st['first_candle_body_pct']:.1f}%)"
                        ax_price.plot(et_times, pct_change, color=color, linewidth=1.2,
                                      label=label, alpha=0.85)

                        # BUY marker
                        if st["entry_time"] is not None:
                            et_buy = _to_et(st["entry_time"])
                            ax_price.plot(et_buy, 0, marker="^", color=color,
                                          markersize=10, zorder=5)
                            ax_price.annotate(
                                "BUY", xy=(et_buy, 0), xytext=(0, 12),
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
                            reason_short = {"TARGET": "T", "STOP_LOSS": "SL", "EOD_CLOSE": "EOD"}.get(
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
                    ax_price.axhline(y=TARGET_PCT, color="#4CAF50", linestyle=":",
                                     alpha=0.4, label=f"+{TARGET_PCT}% target")
                    ax_price.set_title(f"{res['date']} - Price (% from 1st candle close)",
                                       fontsize=10, fontweight="bold")
                    ax_price.set_ylabel("% from Entry", fontsize=8)
                    ax_price.legend(fontsize=6, loc="upper left", ncol=2)
                    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p", tz=ET_TZ))
                    ax_price.tick_params(axis="x", rotation=30, labelsize=7)
                    ax_price.grid(alpha=0.3)

                    # P&L bar chart
                    tickers = [s["ticker"] for s in traded]
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
                    ax_price.text(0.5, 0.5, f"{res['date']}\nNo green candle signals",
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
            f"Green Candle Summary: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)",
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
        ax.plot(range(len(dates_list)), equities, color="#2196F3", linewidth=2)
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
        ax.set_title(f"Equity: ${STARTING_CASH:,} -> ${equities[-1]:,.0f}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Balance ($)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 3. Exit reasons pie
        ax = fig2.add_subplot(gs[2, 0])
        if all_exits:
            reason_colors = {"TARGET": "#4CAF50", "STOP_LOSS": "#f44336", "EOD_CLOSE": "#2196F3"}
            labels = list(all_exits.keys())
            sizes = list(all_exits.values())
            colors = [reason_colors.get(r, "#999") for r in labels]
            ax.pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
                   colors=colors, autopct="%1.0f%%", startangle=90)
        ax.set_title("Exit Reasons", fontweight="bold")

        # 4. Stats
        ax = fig2.add_subplot(gs[2, 1])
        ax.axis("off")
        pf = abs(avg_win * total_wins / (avg_loss * total_losses)) if total_losses > 0 and avg_loss != 0 else 0
        cum_pnl = np.cumsum(pnls_list)
        max_dd = min(cum_pnl) if len(cum_pnl) > 0 else 0
        stats_text = (
            f"PERFORMANCE\n{'='*35}\n"
            f"Starting:    ${STARTING_CASH:,}\n"
            f"Ending:      ${final_equity:,.0f}\n"
            f"Return:      {(final_equity/STARTING_CASH-1)*100:+.1f}%\n"
            f"Total P&L:   ${tp:+,.0f}\n"
            f"{'='*35}\n"
            f"Days:        {len(all_dates)}\n"
            f"Trades:      {total_trades}\n"
            f"Winners:     {total_wins} ({total_wins/max(total_trades,1)*100:.1f}%)\n"
            f"Losers:      {total_losses}\n"
            f"{'='*35}\n"
            f"Avg Win:     ${avg_win:+,.0f}\n"
            f"Avg Loss:    ${avg_loss:+,.0f}\n"
            f"Profit Fac:  {pf:.2f}\n"
            f"Sharpe:      {sharpe:.2f}\n"
            f"{'='*35}\n"
            f"Green Days:  {green}/{len(daily_pnls)}\n"
            f"Best Day:    ${max(daily_pnls):+,.0f}\n" if daily_pnls else ""
            f"Worst Day:   ${min(daily_pnls):+,.0f}\n" if daily_pnls else ""
            f"Max DD:      ${max_dd:+,.0f}\n"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # 5. Config
        ax = fig2.add_subplot(gs[2, 2])
        ax.axis("off")
        config_text = (
            f"STRATEGY CONFIG\n{'='*30}\n"
            f"\nENTRY\n{'-'*30}\n"
            f"Gap:        >{MIN_GAP_PCT}%\n"
            f"1st candle: body>={GREEN_CANDLE_MIN_BODY_PCT}%\n"
            f"2nd candle: green\n"
            f"Max pos:    {MAX_POSITIONS}\n"
            f"\nEXIT\n{'-'*30}\n"
            f"Target:     +{TARGET_PCT}%\n"
            f"Time Stop:  {TIME_LIMIT_MINUTES} min\n"
            f"EOD:        {EOD_EXIT_MINUTES}min\n"
            f"\nSIZING\n{'-'*30}\n"
            f"Position:   {int(TRADE_PCT*100)}% of cash\n"
            f"Starting:   ${STARTING_CASH:,}\n"
            f"Slippage:   {SLIPPAGE_PCT}%\n"
            f"Vol Cap:    {VOL_CAP_PCT}%\n"
            f"\nFILTERS\n{'-'*30}\n"
            f"Regime:     SPY>SMA({SPY_SMA_PERIOD})\n"
            f"\nDATA\n{'-'*30}\n"
            f"{all_dates[0]} to\n"
            f"{all_dates[-1]}\n"
            f"{len(all_dates)} days\n"
            f"Run: {run_ts}\n"
        )
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.7))

        plt.tight_layout()
        summary_path = os.path.join(run_dir, "gc_summary.png")
        plt.savefig(summary_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Summary chart saved to {summary_path}")
