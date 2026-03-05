"""
Quick Scalp + Recycle Backtest
==============================
Variant strategy: buy breakouts, sell 100% at +3%, recycle into pullback.
Reuses data loading from test_full.py; own simulate_day_scalp().

Usage:
  python test_scalp.py                              # uses stored_data_oos + stored_data
  python test_scalp.py stored_data_combined          # specific dir
  python test_scalp.py --no-charts                   # skip charts
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
    CONFIRM_ABOVE,
    CONFIRM_WINDOW,
    PULLBACK_PCT,
    PULLBACK_TIMEOUT,
    SPY_SMA_PERIOD,
    MARGIN_THRESHOLD,
    VOL_CAP_PCT,
    ET_TZ,
)
from regime_filters import RegimeFilter

# --- SCALP CONFIG ---
SCALP_TARGET_PCT = 3.0           # Sell 100% at +3%
SCALP_STOP_PCT = 6.0             # Hard stop per cycle
SCALP_REENTRY_PULLBACK_PCT = 2.0 # Pullback % from last sell price for re-entry
SCALP_MAX_CYCLES = 5             # Max cycles per stock per day
SCALP_COOLDOWN_CANDLES = 2       # Min candles between sell and re-entry signal
SCALP_EOD_EXIT_MINUTES = 15      # Close positions 15 min before close
MIN_GAP_PCT = 12.0               # Own gap filter (can differ from test_full)
MIN_PM_VOLUME = 0                # Own volume filter


def simulate_day_scalp(picks, starting_cash, cash_account=False):
    """Simulate Quick Scalp + Recycle for one day.

    Returns: (states, ending_cash, unsettled_cash)
    """
    cash = starting_cash
    unsettled = 0.0
    trade_pct = _get_trade_pct(starting_cash)
    trade_size_base = starting_cash * trade_pct

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
            # 3-phase entry
            "breakout_confirmed": False,
            "pullback_detected": False,
            "candles_since_confirm": 0,
            "recent_closes": [],
            "signal_close_price": None,
            # Position
            "entry_price": None,
            "entry_time": None,
            "shares": 0,
            "position_cost": 0.0,
            "done": False,
            # Recycle
            "cycle_count": 0,
            "last_sell_price": None,
            "candles_since_sell": 0,
            "phase": "BREAKOUT_WAIT",
            "cycle_history": [],
        })

    for ts in all_timestamps:
        entry_candidates = []

        for st in states:
            if st["done"]:
                continue
            if ts not in st["mh"].index:
                continue

            candle = st["mh"].loc[ts]
            c_high = float(candle["High"])
            c_low = float(candle["Low"])
            c_close = float(candle["Close"])
            pm_high = st["premarket_high"]

            # ====== IN_POSITION: process exits ======
            if st["phase"] == "IN_POSITION" and st["entry_price"] is not None:

                # --- EOD forced exit ---
                try:
                    ts_et = ts.astimezone(ET_TZ)
                except:
                    ts_et = ts
                minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
                if minutes_to_close <= SCALP_EOD_EXIT_MINUTES:
                    sell_price = c_close * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["shares"] * sell_price
                    pnl = proceeds - st["position_cost"]
                    st["cycle_history"].append({
                        "entry_price": st["entry_price"],
                        "entry_time": st["entry_time"],
                        "exit_price": c_close,
                        "exit_time": ts,
                        "pnl": pnl,
                        "reason": "EOD_CLOSE",
                    })
                    st["entry_price"] = None
                    st["shares"] = 0
                    st["position_cost"] = 0.0
                    st["done"] = True
                    _receive_proceeds(proceeds)
                    continue

                # --- Stop loss check (BEFORE target) ---
                stop_price = st["entry_price"] * (1 - SCALP_STOP_PCT / 100)
                if c_low <= stop_price:
                    sell_price = stop_price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["shares"] * sell_price
                    pnl = proceeds - st["position_cost"]
                    st["cycle_history"].append({
                        "entry_price": st["entry_price"],
                        "entry_time": st["entry_time"],
                        "exit_price": stop_price,
                        "exit_time": ts,
                        "pnl": pnl,
                        "reason": "STOP_LOSS",
                    })
                    st["entry_price"] = None
                    st["shares"] = 0
                    st["position_cost"] = 0.0
                    st["done"] = True  # Stop loss = dead, no recycle
                    _receive_proceeds(proceeds)
                    continue

                # --- Target hit: +3% ---
                target_price = st["entry_price"] * (1 + SCALP_TARGET_PCT / 100)
                if c_high >= target_price:
                    sell_price = target_price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["shares"] * sell_price
                    pnl = proceeds - st["position_cost"]
                    st["cycle_history"].append({
                        "entry_price": st["entry_price"],
                        "entry_time": st["entry_time"],
                        "exit_price": target_price,
                        "exit_time": ts,
                        "pnl": pnl,
                        "reason": "TARGET",
                    })
                    st["last_sell_price"] = target_price
                    st["entry_price"] = None
                    st["shares"] = 0
                    st["position_cost"] = 0.0
                    st["cycle_count"] += 1
                    st["candles_since_sell"] = 0
                    _receive_proceeds(proceeds)

                    # Max cycles reached?
                    if st["cycle_count"] >= SCALP_MAX_CYCLES:
                        st["done"] = True
                        continue

                    # Transition to RECYCLING
                    st["phase"] = "RECYCLING"
                    st["pullback_detected"] = False
                    continue

                # Position stays open
                continue

            # ====== NO POSITION: detect entry/re-entry signals ======

            # --- BREAKOUT_WAIT ---
            if st["phase"] == "BREAKOUT_WAIT":
                st["recent_closes"].append(c_close > pm_high)
                if len(st["recent_closes"]) > CONFIRM_WINDOW:
                    st["recent_closes"] = st["recent_closes"][-CONFIRM_WINDOW:]
                if sum(st["recent_closes"]) >= CONFIRM_ABOVE:
                    st["breakout_confirmed"] = True
                    st["phase"] = "PULLBACK_WAIT"
                    st["candles_since_confirm"] = 0
                else:
                    continue

            # --- PULLBACK_WAIT ---
            if st["phase"] == "PULLBACK_WAIT":
                st["candles_since_confirm"] += 1
                pullback_zone = pm_high * (1 + PULLBACK_PCT / 100)
                if c_low <= pullback_zone:
                    st["pullback_detected"] = True
                    if c_close > pm_high:
                        # Pullback + bounce same candle
                        st["signal_close_price"] = c_close
                        st["phase"] = "BOUNCE_WAIT"
                        entry_candidates.append(st)
                    else:
                        st["phase"] = "BOUNCE_WAIT"
                elif st["candles_since_confirm"] >= PULLBACK_TIMEOUT:
                    if c_close > pm_high:
                        st["signal_close_price"] = c_close
                        st["phase"] = "BOUNCE_WAIT"
                        entry_candidates.append(st)
                continue

            # --- BOUNCE_WAIT ---
            if st["phase"] == "BOUNCE_WAIT":
                if c_close > pm_high:
                    st["signal_close_price"] = c_close
                    entry_candidates.append(st)
                continue

            # --- RECYCLING ---
            if st["phase"] == "RECYCLING":
                st["candles_since_sell"] += 1

                # Kill if price closes below PM high
                if c_close < pm_high:
                    st["done"] = True
                    continue

                # Cooldown
                if st["candles_since_sell"] < SCALP_COOLDOWN_CANDLES:
                    continue

                # Pullback detection from last sell price
                if not st["pullback_detected"]:
                    reentry_zone = st["last_sell_price"] * (1 + SCALP_REENTRY_PULLBACK_PCT / 100)
                    if c_low <= st["last_sell_price"] or c_close <= reentry_zone:
                        st["pullback_detected"] = True
                        if c_close > st["last_sell_price"]:
                            st["signal_close_price"] = c_close
                            entry_candidates.append(st)
                    continue

                # Bounce above last sell price
                if c_close > st["last_sell_price"]:
                    st["signal_close_price"] = c_close
                    entry_candidates.append(st)
                continue

        # ====== PASS 2: Allocate capital ======
        entry_candidates.sort(
            key=lambda s: -(s.get("pm_volume", 0) * s.get("premarket_high", 1))
        )

        for st in entry_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue

            # No entries near EOD
            try:
                ts_et = ts.astimezone(ET_TZ)
            except:
                ts_et = ts
            mins_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
            if mins_to_close <= SCALP_EOD_EXIT_MINUTES:
                continue

            position_size = trade_size_base
            if cash < position_size:
                if cash > 100:
                    position_size = cash
                else:
                    continue  # No cash, don't kill — may free later

            fill_price = st["signal_close_price"]
            if fill_price is None or fill_price <= 0:
                continue

            # Volume cap
            if VOL_CAP_PCT > 0:
                pre_entry = st["mh"].loc[st["mh"].index <= ts]
                vol_shares = pre_entry["Volume"].sum() if len(pre_entry) > 0 else 0
                dollar_vol = fill_price * vol_shares
                vol_limit = dollar_vol * (VOL_CAP_PCT / 100)
                if vol_limit > 0 and position_size > vol_limit:
                    position_size = vol_limit
                if position_size < 50:
                    continue

            entry_price = fill_price * (1 + SLIPPAGE_PCT / 100)
            st["entry_price"] = entry_price
            st["entry_time"] = ts
            st["position_cost"] = position_size
            st["shares"] = position_size / entry_price
            st["phase"] = "IN_POSITION"
            st["pullback_detected"] = False
            cash -= position_size

    # ====== EOD: close remaining positions ======
    for st in states:
        if st["entry_price"] is not None and st["shares"] > 0:
            last_ts = st["mh"].index[-1]
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["shares"] * sell_price
            pnl = proceeds - st["position_cost"]
            st["cycle_history"].append({
                "entry_price": st["entry_price"],
                "entry_time": st["entry_time"],
                "exit_price": last_close,
                "exit_time": last_ts,
                "pnl": pnl,
                "reason": "EOD_CLOSE",
            })
            st["entry_price"] = None
            st["shares"] = 0
            st["done"] = True
            _receive_proceeds(proceeds)

    return states, cash, unsettled


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse args
    no_charts = "--no-charts" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-charts"]
    data_dirs = args if args else ["stored_data_oos", "stored_data"]

    print(f"Quick Scalp + Recycle Backtest")
    print(f"{'='*70}")
    print(f"  Target: +{SCALP_TARGET_PCT}%  |  Stop: {SCALP_STOP_PCT}%  |  Max Cycles: {SCALP_MAX_CYCLES}")
    print(f"  Re-entry pullback: {SCALP_REENTRY_PULLBACK_PCT}% from last sell")
    print(f"  Cooldown: {SCALP_COOLDOWN_CANDLES} candles  |  EOD exit: {SCALP_EOD_EXIT_MINUTES} min")
    print(f"  Data: {data_dirs}")
    print(f"{'='*70}\n")

    print("Loading data...")
    all_dates, daily_picks = load_all_picks(data_dirs)
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
    total_cycles = 0
    total_winning_cycles = 0
    total_losing_cycles = 0
    total_trades = 0  # stocks entered at least once
    recycle_pnl = 0.0  # P&L from cycles 2+
    all_results = []

    print(f"{'Date':<12} {'Stocks':>6} {'Cycles':>7} {'Win':>4} {'Loss':>5} "
          f"{'Size':>5} {'Day P&L':>12} {'Balance':>14}")
    print("-" * 80)

    for d in all_dates:
        # Settle T+1 cash
        if unsettled_cash > 0:
            cash += unsettled_cash
            unsettled_cash = 0.0

        # Regime check
        should_trade, _, _ = rf.check(d)
        if not should_trade:
            equity = cash + unsettled_cash
            print(f"{d:<12} {'SKIP':>6} {'':>7} {'':>4} {'':>5} "
                  f"{'':>5} {'$+0':>12} ${equity:>13,.0f}  <-- SPY < SMA{SPY_SMA_PERIOD}")
            all_results.append({
                "date": d, "picks": daily_picks[d], "states": [],
                "day_pnl": 0.0, "equity": equity, "regime_skip": True,
                "stocks": 0, "cycles": 0, "wins": 0, "losses": 0,
            })
            continue

        picks = daily_picks[d]
        cash_account = cash < MARGIN_THRESHOLD

        states, new_cash, new_unsettled = simulate_day_scalp(
            picks, cash, cash_account
        )

        # Compute stats
        day_pnl = 0.0
        day_stocks = 0
        day_cycles = 0
        day_wins = 0
        day_losses = 0

        for st in states:
            if st["cycle_history"]:
                day_stocks += 1
                for i, cyc in enumerate(st["cycle_history"]):
                    day_cycles += 1
                    day_pnl += cyc["pnl"]
                    if cyc["pnl"] > 0:
                        day_wins += 1
                        total_winning_cycles += 1
                    else:
                        day_losses += 1
                        total_losing_cycles += 1
                    # Recycle P&L (cycles 2+)
                    if i > 0:
                        recycle_pnl += cyc["pnl"]

        total_cycles += day_cycles
        total_trades += day_stocks
        cash = new_cash
        unsettled_cash = new_unsettled

        trade_pct_label = f"{_get_trade_pct(cash)*100:.0f}%"

        equity = cash + unsettled_cash
        print(f"{d:<12} {day_stocks:>6} {day_cycles:>7} {day_wins:>4} {day_losses:>5} "
              f"{trade_pct_label:>5} ${day_pnl:>+11,.0f} ${equity:>13,.0f}")

        all_results.append({
            "date": d, "picks": picks, "states": states,
            "day_pnl": day_pnl, "equity": equity, "regime_skip": False,
            "stocks": day_stocks, "cycles": day_cycles,
            "wins": day_wins, "losses": day_losses,
        })

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  QUICK SCALP + RECYCLE SUMMARY")
    print(f"{'='*70}")
    final_equity = cash + unsettled_cash
    print(f"  Starting Cash:    ${STARTING_CASH:,}")
    print(f"  Ending Equity:    ${final_equity:,.0f}  ({(final_equity/STARTING_CASH - 1)*100:+.1f}%)")
    if unsettled_cash > 0:
        print(f"    (Cash: ${cash:,.0f} + Unsettled: ${unsettled_cash:,.0f})")
    print(f"  Trading Days:     {len(all_dates)} ({tradeable} traded, {skipped} skipped)")
    print(f"  Stocks Entered:   {total_trades}")
    print(f"  Total Cycles:     {total_cycles}")
    print(f"    Winning:        {total_winning_cycles} ({total_winning_cycles/max(total_cycles,1)*100:.1f}%)")
    print(f"    Losing:         {total_losing_cycles} ({total_losing_cycles/max(total_cycles,1)*100:.1f}%)")
    print(f"  Avg Cycles/Stock: {total_cycles/max(total_trades,1):.2f}")

    # Daily P&L stats
    daily_pnls = [r["day_pnl"] for r in all_results if r["stocks"] > 0]  # only traded days
    if daily_pnls:
        green = sum(1 for p in daily_pnls if p > 0)
        red = sum(1 for p in daily_pnls if p <= 0)
        sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if np.std(daily_pnls) > 0 else 0
        print(f"\n  Green Days:       {green}/{len(daily_pnls)} ({green/len(daily_pnls)*100:.1f}%)")
        print(f"  Red Days:         {red}/{len(daily_pnls)}")
        print(f"  Best Day:         ${max(daily_pnls):+,.0f}")
        print(f"  Worst Day:        ${min(daily_pnls):+,.0f}")
        print(f"  Avg P&L/Day:      ${np.mean(daily_pnls):+,.0f}")
        print(f"  Sharpe (ann.):    {sharpe:.2f}")

    # Recycle efficiency
    print(f"\n  RECYCLE EFFICIENCY:")
    cycle1_pnl = sum(r["day_pnl"] for r in all_results) - recycle_pnl
    print(f"    Cycle 1 P&L:    ${cycle1_pnl:+,.0f} (first entry only)")
    print(f"    Cycle 2+ P&L:   ${recycle_pnl:+,.0f} (recycled entries)")
    print(f"    Recycle added:  ${recycle_pnl:+,.0f} ({'helped' if recycle_pnl > 0 else 'hurt'})")

    print(f"\n  CONFIG:")
    print(f"    Target:         +{SCALP_TARGET_PCT}%")
    print(f"    Stop:           {SCALP_STOP_PCT}%")
    print(f"    Re-entry PB:    {SCALP_REENTRY_PULLBACK_PCT}% from last sell")
    print(f"    Max Cycles:     {SCALP_MAX_CYCLES}")
    print(f"    Cooldown:       {SCALP_COOLDOWN_CANDLES} candles")
    print(f"    EOD Exit:       {SCALP_EOD_EXIT_MINUTES} min before close")
    print(f"    Gap Filter:     >{MIN_GAP_PCT}%")
    print(f"    Regime:         SPY > SMA({SPY_SMA_PERIOD})")
    print(f"{'='*70}")

    # ─── CHARTS ─────────────────────────────────────────────────────────────
    if no_charts:
        print("\n  Skipping charts (--no-charts)")
    else:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("charts", f"scalp_{run_ts}")
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
                f"Scalp Backtest Page {page + 1}/{num_pages}: "
                f"{page_results[0]['date']} to {page_results[-1]['date']}\n"
                f"Target: +{SCALP_TARGET_PCT}% | Stop: {SCALP_STOP_PCT}% | "
                f"Max Cycles: {SCALP_MAX_CYCLES} | Cooldown: {SCALP_COOLDOWN_CANDLES}",
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

                traded = [s for s in res["states"] if s["cycle_history"]]

                if traded:
                    for si, st in enumerate(traded):
                        mh = st["mh"]
                        if mh.index.tz is not None:
                            et_times = mh.index.tz_convert(ET_TZ)
                        else:
                            et_times = mh.index.tz_localize("UTC").tz_convert(ET_TZ)

                        color = COLORS[si % len(COLORS)]
                        # Use first cycle entry as reference for % axis
                        ref_price = st["cycle_history"][0]["entry_price"]
                        pct_change = (mh["Close"].values.astype(float) / ref_price - 1) * 100
                        label = f"{st['ticker']} (gap {st['gap_pct']:.0f}%, {len(st['cycle_history'])} cyc)"
                        ax_price.plot(et_times, pct_change, color=color, linewidth=1.2,
                                      label=label, alpha=0.85)

                        # Plot cycle markers (BUY1, SELL1, BUY2, SELL2, ...)
                        for ci, cyc in enumerate(st["cycle_history"]):
                            cycle_num = ci + 1

                            # BUY marker
                            if cyc.get("entry_time") is not None:
                                et_buy = _to_et(cyc["entry_time"])
                                buy_pct = (cyc["entry_price"] / ref_price - 1) * 100
                                ax_price.plot(et_buy, buy_pct, marker="^", color=color,
                                              markersize=10, zorder=5)
                                ax_price.annotate(
                                    f"BUY{cycle_num}", xy=(et_buy, buy_pct),
                                    xytext=(0, 12), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=6, fontweight="bold",
                                    color=color,
                                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                              ec=color, alpha=0.8, lw=0.5),
                                )

                            # SELL marker
                            if cyc.get("exit_time") is not None:
                                et_sell = _to_et(cyc["exit_time"])
                                sell_pct = (cyc["exit_price"] / ref_price - 1) * 100
                                is_win = cyc["pnl"] > 0
                                marker = "v" if not is_win else "s"
                                sell_color = "#4CAF50" if is_win else "#f44336"
                                ax_price.plot(et_sell, sell_pct, marker=marker, color=sell_color,
                                              markersize=10, zorder=5,
                                              markeredgecolor="white", markeredgewidth=1)
                                reason_short = {
                                    "TARGET": "T", "STOP_LOSS": "SL", "EOD_CLOSE": "EOD"
                                }.get(cyc["reason"], cyc["reason"][:3])
                                ax_price.annotate(
                                    f"{reason_short}{cycle_num}", xy=(et_sell, sell_pct),
                                    xytext=(0, -14 if sell_pct >= 0 else 12),
                                    textcoords="offset points", ha="center",
                                    va="top" if sell_pct >= 0 else "bottom",
                                    fontsize=6, fontweight="bold", color=sell_color,
                                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                              ec=sell_color, alpha=0.8, lw=0.5),
                                )

                    ax_price.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                    ax_price.axhline(y=SCALP_TARGET_PCT, color="#4CAF50", linestyle=":",
                                     alpha=0.4, label=f"+{SCALP_TARGET_PCT}% target")
                    ax_price.axhline(y=-SCALP_STOP_PCT, color="#f44336", linestyle=":",
                                     alpha=0.4, label=f"-{SCALP_STOP_PCT}% stop")
                    ax_price.set_title(f"{res['date']} - Price Action (% from 1st entry)",
                                       fontsize=10, fontweight="bold")
                    ax_price.set_ylabel("% from Entry", fontsize=8)
                    ax_price.legend(fontsize=6, loc="upper left", ncol=2)
                    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p", tz=ET_TZ))
                    ax_price.tick_params(axis="x", rotation=30, labelsize=7)
                    ax_price.grid(alpha=0.3)

                    # P&L bar chart (per cycle)
                    bar_labels = []
                    bar_pnls = []
                    for st in traded:
                        for ci, cyc in enumerate(st["cycle_history"]):
                            bar_labels.append(f"{st['ticker']} C{ci+1} ({cyc['reason'][:3]})")
                            bar_pnls.append(cyc["pnl"])
                    bar_colors = ["#4CAF50" if p > 0 else "#f44336" for p in bar_pnls]
                    y_pos = range(len(bar_labels))
                    bars = ax_pnl.barh(y_pos, bar_pnls, color=bar_colors,
                                       edgecolor="white", height=0.6)
                    ax_pnl.set_yticks(y_pos)
                    ax_pnl.set_yticklabels(bar_labels, fontsize=7)
                    ax_pnl.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
                    ax_pnl.invert_yaxis()
                    for j, (bar, pnl) in enumerate(zip(bars, bar_pnls)):
                        x_pos = bar.get_width()
                        align = "left" if pnl >= 0 else "right"
                        offset = 5 if pnl >= 0 else -5
                        ax_pnl.annotate(f"${pnl:+.0f}", xy=(x_pos, j), xytext=(offset, 0),
                                        textcoords="offset points", ha=align, va="center",
                                        fontsize=7, fontweight="bold", color=bar_colors[j])
                    day_total = sum(bar_pnls)
                    total_color = "#4CAF50" if day_total >= 0 else "#f44336"
                    ax_pnl.set_title(f"P&L | Day: ${day_total:+,.0f} | {res['cycles']} cycles",
                                     fontsize=10, fontweight="bold", color=total_color)
                    ax_pnl.set_xlabel("P&L ($)", fontsize=8)
                    ax_pnl.grid(alpha=0.3, axis="x")
                else:
                    ax_price.text(0.5, 0.5, f"{res['date']}\nNo trades triggered",
                                  ha="center", va="center", fontsize=12,
                                  transform=ax_price.transAxes, color="gray")
                    ax_price.set_title(f"{res['date']} - No Trades", fontsize=10)
                    ax_pnl.text(0.5, 0.5, "0 trades", ha="center", va="center",
                                fontsize=10, transform=ax_pnl.transAxes, color="gray")
                    ax_pnl.set_title("P&L: $0", fontsize=10, color="gray")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            chart_path = os.path.join(run_dir, f"scalp_page_{page + 1:02d}.png")
            plt.savefig(chart_path, dpi=120, bbox_inches="tight")
            plt.close()
            sys.stdout.write(f"\r  Charts: page {page + 1}/{num_pages}")
            sys.stdout.flush()

        print(f"\r  {num_pages} chart pages saved to {run_dir}/          ")

        # --- SUMMARY CHART ---
        fig2 = plt.figure(figsize=(22, 18))
        gs = fig2.add_gridspec(3, 3, width_ratios=[1.2, 1.2, 0.8],
                               height_ratios=[1.0, 1.0, 1.0], hspace=0.35, wspace=0.3)
        fig2.suptitle(
            f"Scalp Backtest Summary: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)",
            fontsize=16, fontweight="bold",
        )

        # 1. Daily P&L bars — full width
        ax = fig2.add_subplot(gs[0, :])
        dates_list = [r["date"] for r in all_results]
        pnls_list = [r["day_pnl"] for r in all_results]
        bar_c = ["#4CAF50" if p >= 0 else "#f44336" for p in pnls_list]
        ax.bar(range(len(dates_list)), pnls_list, color=bar_c, edgecolor="none", width=0.8)
        ax.set_xticks(range(0, len(dates_list), max(1, len(dates_list) // 15)))
        ax.set_xticklabels(
            [dates_list[i] for i in range(0, len(dates_list), max(1, len(dates_list) // 15))],
            rotation=45, fontsize=8, ha="right",
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        tp = sum(pnls_list)
        tc = "#4CAF50" if tp >= 0 else "#f44336"
        ax.set_title(f"Daily P&L | Total: ${tp:+,.0f}", fontsize=13, fontweight="bold", color=tc)
        ax.set_ylabel("P&L ($)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

        # 2. Equity curve — full width
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
                   label=f"Margin ${MARGIN_THRESHOLD / 1000:.0f}K")
        ax.set_xticks(range(0, len(dates_list), max(1, len(dates_list) // 15)))
        ax.set_xticklabels(
            [dates_list[i] for i in range(0, len(dates_list), max(1, len(dates_list) // 15))],
            rotation=45, fontsize=8, ha="right",
        )
        ax.set_title(f"Equity Curve: ${STARTING_CASH:,} -> ${equities[-1]:,.0f}",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Account Balance ($)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # 3. Exit reasons pie
        ax = fig2.add_subplot(gs[2, 0])
        all_exits = {}
        for r in all_results:
            for st in r["states"]:
                for cyc in st["cycle_history"]:
                    reason = cyc["reason"]
                    all_exits[reason] = all_exits.get(reason, 0) + 1
        if all_exits:
            reason_colors = {
                "TARGET": "#4CAF50", "STOP_LOSS": "#f44336", "EOD_CLOSE": "#2196F3",
            }
            labels = list(all_exits.keys())
            sizes = list(all_exits.values())
            colors = [reason_colors.get(r, "#999") for r in labels]
            ax.pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
                   colors=colors, autopct="%1.0f%%", startangle=90)
        ax.set_title("Exit Reasons", fontweight="bold")

        # 4. Stats text
        ax = fig2.add_subplot(gs[2, 1])
        ax.axis("off")
        avg_win_pnl = np.mean([
            cyc["pnl"] for r in all_results for st in r["states"]
            for cyc in st["cycle_history"] if cyc["pnl"] > 0
        ]) if total_winning_cycles > 0 else 0
        avg_loss_pnl = np.mean([
            cyc["pnl"] for r in all_results for st in r["states"]
            for cyc in st["cycle_history"] if cyc["pnl"] <= 0
        ]) if total_losing_cycles > 0 else 0
        cum_pnl = np.cumsum(pnls_list)
        max_dd = min(cum_pnl) if len(cum_pnl) > 0 else 0
        pf = abs(avg_win_pnl * total_winning_cycles / (avg_loss_pnl * total_losing_cycles)) \
            if total_losing_cycles > 0 and avg_loss_pnl != 0 else 0
        stats_text = (
            f"SCALP PERFORMANCE\n"
            f"{'=' * 35}\n"
            f"Starting Cash:   ${STARTING_CASH:,}\n"
            f"Ending Equity:   ${final_equity:,.0f}\n"
            f"Total Return:    {(final_equity / STARTING_CASH - 1) * 100:+.1f}%\n"
            f"Total P&L:       ${tp:+,.0f}\n"
            f"{'=' * 35}\n"
            f"Trading Days:    {len(all_dates)}\n"
            f"Stocks Entered:  {total_trades}\n"
            f"Total Cycles:    {total_cycles}\n"
            f"  Winning:       {total_winning_cycles} ({total_winning_cycles / max(total_cycles, 1) * 100:.1f}%)\n"
            f"  Losing:        {total_losing_cycles}\n"
            f"Avg Cyc/Stock:   {total_cycles / max(total_trades, 1):.2f}\n"
            f"{'=' * 35}\n"
            f"Avg Win:         ${avg_win_pnl:+,.0f}\n"
            f"Avg Loss:        ${avg_loss_pnl:+,.0f}\n"
            f"Profit Factor:   {pf:.2f}\n"
            f"Sharpe (ann.):   {sharpe:.2f}\n"
            f"{'=' * 35}\n"
            f"Green Days:      {green}/{len(daily_pnls)}\n"
            f"Red Days:        {red}/{len(daily_pnls)}\n"
            f"Best Day:        ${max(daily_pnls):+,.0f}\n"
            f"Worst Day:       ${min(daily_pnls):+,.0f}\n"
            f"Max Drawdown:    ${max_dd:+,.0f}\n"
            f"{'=' * 35}\n"
            f"Cycle 1 P&L:     ${cycle1_pnl:+,.0f}\n"
            f"Cycle 2+ P&L:    ${recycle_pnl:+,.0f}\n"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # 5. Config text
        ax = fig2.add_subplot(gs[2, 2])
        ax.axis("off")
        config_text = (
            f"SCALP CONFIG\n"
            f"{'=' * 30}\n"
            f"\nENTRY\n{'-' * 30}\n"
            f"Confirm:      {CONFIRM_ABOVE}/{CONFIRM_WINDOW} candles\n"
            f"Pullback:     {PULLBACK_PCT}%\n"
            f"PB Timeout:   {PULLBACK_TIMEOUT} candles\n"
            f"\nEXIT\n{'-' * 30}\n"
            f"Target:       +{SCALP_TARGET_PCT}% (100% sell)\n"
            f"Stop Loss:    {SCALP_STOP_PCT}%\n"
            f"EOD Exit:     {SCALP_EOD_EXIT_MINUTES}min\n"
            f"\nRECYCLE\n{'-' * 30}\n"
            f"Max Cycles:   {SCALP_MAX_CYCLES}\n"
            f"Cooldown:     {SCALP_COOLDOWN_CANDLES} candles\n"
            f"Re-entry PB:  {SCALP_REENTRY_PULLBACK_PCT}%\n"
            f"Kill if:      close < PM high\n"
            f"\nSIZING\n{'-' * 30}\n"
            f"Position:     {int(TRADE_PCT * 100)}% of cash\n"
            f"Starting:     ${STARTING_CASH:,}\n"
            f"Slippage:     {SLIPPAGE_PCT}%\n"
            f"Vol Cap:      {VOL_CAP_PCT}%\n"
            f"\nFILTERS\n{'-' * 30}\n"
            f"Min Gap:      {MIN_GAP_PCT}%\n"
            f"Min PM Vol:   {MIN_PM_VOLUME:,}\n"
            f"Regime:       SPY > SMA({SPY_SMA_PERIOD})\n"
            f"\nRUN INFO\n{'-' * 30}\n"
            f"Dates:        {all_dates[0]}\n"
            f"              {all_dates[-1]}\n"
            f"Days:         {len(all_dates)}\n"
            f"Run:          {run_ts}\n"
        )
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.7))

        plt.tight_layout()
        summary_path = os.path.join(run_dir, "scalp_summary.png")
        plt.savefig(summary_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Summary chart saved to {summary_path}")
