"""
Weekly backtest with per-day visual charts.
Shows price action, buy/sell markers, P&L per trade, and daily totals.
"""
import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo


def _is_warrant_or_unit(ticker):
    """Filter out warrants (.WS, W suffix), rights (R suffix), units (U suffix)."""
    if '.WS' in ticker or '.RT' in ticker:
        return True
    if re.match(r'^[A-Z]{3,}W$', ticker):
        return True
    if ticker.endswith('WW'):
        return True
    if re.match(r'^[A-Z]{3,}U$', ticker):
        return True
    if re.match(r'^[A-Z]{3,}R$', ticker):
        return True
    return False

# --- CONFIG (Optuna Phase 2 optimized) ---
SLIPPAGE_PCT = 0.05
STOP_LOSS_PCT = 12.0
DAILY_CASH = 10_000
TRADE_PCT = 0.50
MIN_GAP_PCT = 2.0
MIN_PM_VOLUME = 0  # no volume filter
TOP_N = 10
PARTIAL_SELL_FRAC = 0.25
PARTIAL_SELL_PCT = 35.0
ATR_PERIOD = 20
ATR_MULTIPLIER = 2.75
CONFIRM_ABOVE = 3
CONFIRM_WINDOW = 4
PULLBACK_PCT = 10.0
PULLBACK_TIMEOUT = 7
SCALE_IN = 1
SCALE_IN_TRIGGER_PCT = 9.0

DATA_DIR = "stored_data"
ET_TZ = ZoneInfo("America/New_York")

# Last 5 trading days
TEST_DATES = ["2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27"]

intraday_dir = os.path.join(DATA_DIR, "intraday")
daily_dir = os.path.join(DATA_DIR, "daily")
all_tickers = [f.replace(".csv", "") for f in os.listdir(intraday_dir) if f.endswith(".csv")]
print(f"Available tickers: {len(all_tickers)}")


def load_day_candidates(test_date):
    """Load and rank top 10 premarket gainers for a single day."""
    candidates = []
    for ticker in all_tickers:
        if _is_warrant_or_unit(ticker):
            continue
        ipath = os.path.join(intraday_dir, f"{ticker}.csv")
        dpath = os.path.join(daily_dir, f"{ticker}.csv")
        try:
            idf = pd.read_csv(ipath, index_col=0, parse_dates=True)
        except Exception:
            continue
        if idf.index.tz is not None:
            et_index = idf.index.tz_convert(ET_TZ)
        else:
            et_index = idf.index.tz_localize("UTC").tz_convert(ET_TZ)

        day_mask = et_index.strftime("%Y-%m-%d") == test_date
        day_candles = idf[day_mask]
        et_day = et_index[day_mask]
        if len(day_candles) == 0:
            continue

        pm_mask = (et_day.hour < 9) | ((et_day.hour == 9) & (et_day.minute < 30))
        mh_mask = (
            ((et_day.hour == 9) & (et_day.minute >= 30))
            | ((et_day.hour >= 10) & (et_day.hour < 16))
        )
        premarket = day_candles[pm_mask]
        market_hours = day_candles[mh_mask]
        if len(market_hours) == 0:
            continue

        market_open = float(market_hours.iloc[0]["Open"])
        premarket_high = float(premarket["High"].max()) if len(premarket) > 0 else market_open

        # Premarket volume check
        pm_volume = int(premarket["Volume"].sum()) if len(premarket) > 0 else 0
        if pm_volume < MIN_PM_VOLUME:
            continue

        prev_close = None
        if os.path.exists(dpath):
            try:
                ddf = pd.read_csv(dpath, index_col=0, parse_dates=True)
                date_naive = pd.Timestamp(test_date)
                ddf_dates = ddf.index.tz_localize(None) if ddf.index.tz else ddf.index
                prev_mask = ddf_dates < date_naive
                if prev_mask.any():
                    prev_close = float(ddf.loc[ddf.index[prev_mask][-1], "Close"])
            except Exception:
                pass
        if prev_close is None or prev_close <= 0:
            continue

        gap_pct = (market_open - prev_close) / prev_close * 100
        if gap_pct < MIN_GAP_PCT:
            continue

        candidates.append({
            "ticker": ticker,
            "gap_pct": gap_pct,
            "market_open": market_open,
            "premarket_high": premarket_high,
            "prev_close": prev_close,
            "market_hour_candles": market_hours,
        })

    candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
    return candidates[:TOP_N]


def simulate_day(picks):
    """Simulate trades for one day, return list of trade results with event history."""
    cash = DAILY_CASH
    trade_size_base = DAILY_CASH * TRADE_PCT
    trades_taken = 0

    all_timestamps = set()
    for pick in picks:
        all_timestamps.update(pick["market_hour_candles"].index.tolist())
    all_timestamps = sorted(all_timestamps)

    states = []
    for pick in picks:
        states.append({
            "ticker": pick["ticker"],
            "premarket_high": pick["premarket_high"],
            "gap_pct": pick["gap_pct"],
            "mh": pick["market_hour_candles"],
            "entry_price": None,
            "position_cost": 0.0,
            "shares": 0,
            "remaining_shares": 0,
            "total_exit_value": 0.0,
            "exit_reason": "NO_BREAKOUT",
            "partial_sold": False,
            "trailing_active": False,
            "highest_since_entry": 0.0,
            "entry_time": None,
            "exit_price": None,
            "exit_time": None,
            "partial_sell_time": None,
            "partial_sell_price": None,
            "done": False,
            "recent_closes": [],
            "breakout_confirmed": False,
            "pullback_detected": False,
            "candles_since_confirm": 0,
            "true_ranges": [],
            "prev_candle_close": None,
            "scaled_in": False,
            "original_position_size": 0.0,
            "scale_in_time": None,
            "scale_in_price": None,
        })

    for ts in all_timestamps:
        for st in states:
            if st["done"]:
                continue
            if ts not in st["mh"].index:
                continue

            candle = st["mh"].loc[ts]
            c_high = float(candle["High"])
            c_low = float(candle["Low"])
            c_close = float(candle["Close"])

            # --- Accumulate True Range for ATR ---
            if st["prev_candle_close"] is not None:
                tr = max(c_high - c_low,
                         abs(c_high - st["prev_candle_close"]),
                         abs(c_low - st["prev_candle_close"]))
            else:
                tr = c_high - c_low
            st["true_ranges"].append(tr)
            if len(st["true_ranges"]) > ATR_PERIOD:
                st["true_ranges"] = st["true_ranges"][-ATR_PERIOD:]
            st["prev_candle_close"] = c_close

            pm_high = st["premarket_high"]

            # --- NOT YET IN TRADE: 3-phase entry ---
            if st["entry_price"] is None:
                entered = False

                # Phase 1: confirmation candles (2/3 above PM high)
                if not st["breakout_confirmed"]:
                    st["recent_closes"].append(c_close > pm_high)
                    if len(st["recent_closes"]) > CONFIRM_WINDOW:
                        st["recent_closes"] = st["recent_closes"][-CONFIRM_WINDOW:]
                    above_count = sum(st["recent_closes"])
                    if above_count >= CONFIRM_ABOVE:
                        st["breakout_confirmed"] = True

                # Phase 2: wait for pullback near PM high (or timeout)
                elif not st["pullback_detected"]:
                    st["candles_since_confirm"] += 1
                    pullback_zone = pm_high * (1 + PULLBACK_PCT / 100)
                    if c_low <= pullback_zone:
                        st["pullback_detected"] = True
                        if c_close > pm_high:
                            entered = True
                    elif st["candles_since_confirm"] >= PULLBACK_TIMEOUT:
                        if c_close > pm_high:
                            entered = True

                # Phase 3: wait for bounce (close above PM high after pullback)
                else:
                    if c_close > pm_high:
                        entered = True

                if entered:
                    trades_taken += 1
                    if trades_taken <= 3:
                        position_size = trade_size_base
                    else:
                        position_size = DAILY_CASH * 0.10
                    if cash < position_size:
                        if cash > 0:
                            position_size = cash
                        else:
                            st["exit_reason"] = "NO_CASH"
                            st["done"] = True
                            trades_taken -= 1
                            continue
                    fill_price = c_close
                    st["entry_price"] = fill_price * (1 + SLIPPAGE_PCT / 100)
                    st["position_cost"] = position_size
                    st["original_position_size"] = position_size
                    st["shares"] = position_size / st["entry_price"]
                    st["remaining_shares"] = st["shares"]
                    st["entry_time"] = ts
                    st["highest_since_entry"] = c_high
                    cash -= position_size
                else:
                    continue

            if c_high > st["highest_since_entry"]:
                st["highest_since_entry"] = c_high

            # --- SCALE-IN ---
            if (SCALE_IN and not st["scaled_in"]
                    and st["remaining_shares"] > 0
                    and not st["partial_sold"]):
                trigger_price = st["entry_price"] * (1 + SCALE_IN_TRIGGER_PCT / 100)
                if c_high >= trigger_price:
                    add_size = st["original_position_size"] * 0.50
                    if cash >= add_size and add_size > 0:
                        add_price = trigger_price * (1 + SLIPPAGE_PCT / 100)
                        add_shares = add_size / add_price
                        total_cost = st["position_cost"] + add_size
                        total_shares = st["remaining_shares"] + add_shares
                        st["entry_price"] = total_cost / total_shares
                        st["position_cost"] = total_cost
                        st["shares"] = total_shares
                        st["remaining_shares"] = total_shares
                        st["scale_in_time"] = ts
                        st["scale_in_price"] = trigger_price
                        cash -= add_size
                    st["scaled_in"] = True

            if st["remaining_shares"] > 0 and not st["trailing_active"]:
                stop_price = st["entry_price"] * (1 - STOP_LOSS_PCT / 100)
                if c_low <= stop_price:
                    sell_price = stop_price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["remaining_shares"] * sell_price
                    st["total_exit_value"] += proceeds
                    st["remaining_shares"] = 0
                    st["exit_reason"] = "STOP_LOSS"
                    st["exit_price"] = stop_price
                    st["exit_time"] = ts
                    st["done"] = True
                    cash += proceeds
                    continue

            if st["remaining_shares"] > 0 and st["trailing_active"]:
                atr = sum(st["true_ranges"]) / len(st["true_ranges"]) if st["true_ranges"] else 0
                trail_stop = st["highest_since_entry"] - (atr * ATR_MULTIPLIER)
                if c_low <= trail_stop:
                    sell_price = trail_stop * (1 - SLIPPAGE_PCT / 100)
                    proceeds = st["remaining_shares"] * sell_price
                    st["total_exit_value"] += proceeds
                    st["remaining_shares"] = 0
                    st["exit_reason"] = "TRAIL_STOP"
                    st["exit_price"] = trail_stop
                    st["exit_time"] = ts
                    st["done"] = True
                    cash += proceeds
                    continue

            if st["remaining_shares"] > 0 and not st["partial_sold"]:
                target_price = st["entry_price"] * (1 + PARTIAL_SELL_PCT / 100)
                if c_high >= target_price:
                    sell_shares = st["shares"] * PARTIAL_SELL_FRAC
                    if sell_shares > st["remaining_shares"]:
                        sell_shares = st["remaining_shares"]
                    sell_price = target_price * (1 - SLIPPAGE_PCT / 100)
                    proceeds = sell_shares * sell_price
                    st["total_exit_value"] += proceeds
                    st["remaining_shares"] -= sell_shares
                    st["partial_sold"] = True
                    st["trailing_active"] = True
                    st["partial_sell_time"] = ts
                    st["partial_sell_price"] = target_price
                    cash += proceeds

            if st["remaining_shares"] <= 0:
                st["done"] = True

    # EOD close
    for st in states:
        if st["entry_price"] is not None and st["remaining_shares"] > 0:
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["remaining_shares"] * sell_price
            st["total_exit_value"] += proceeds
            st["remaining_shares"] = 0
            if st["exit_reason"] not in ("STOP_LOSS", "TRAIL_STOP"):
                st["exit_reason"] = "EOD_CLOSE"
            st["exit_price"] = last_close
            st["exit_time"] = st["mh"].index[-1]
            cash += proceeds

    # Compute P&L
    for st in states:
        if st["entry_price"] is not None:
            cost = st["position_cost"]
            st["pnl"] = st["total_exit_value"] - cost
            st["pnl_pct"] = (st["pnl"] / cost) * 100 if cost > 0 else 0
        else:
            st["pnl"] = 0
            st["pnl_pct"] = 0

    return states, cash


def plot_day(date_str, picks, states, day_idx, axes_row):
    """Plot traded stocks for one day: price chart with buy/sell markers + P&L bar chart."""
    traded = [s for s in states if s["entry_price"] is not None]
    not_traded = [s for s in states if s["entry_price"] is None]

    ax_price = axes_row[0]
    ax_pnl = axes_row[1]

    # --- PRICE CHARTS FOR TRADED STOCKS ---
    if traded:
        colors_cycle = ["#2196F3", "#FF9800", "#4CAF50", "#f44336", "#9C27B0",
                        "#00BCD4", "#795548", "#607D8B", "#E91E63", "#CDDC39"]
        for i, st in enumerate(traded):
            mh = st["mh"]
            # Convert index to ET for display
            if mh.index.tz is not None:
                et_times = mh.index.tz_convert(ET_TZ)
            else:
                et_times = mh.index.tz_localize("UTC").tz_convert(ET_TZ)

            color = colors_cycle[i % len(colors_cycle)]
            si_tag = " S+" if st.get("scaled_in") else ""
            label = f"{st['ticker']} (gap {st['gap_pct']:.0f}%{si_tag})"

            # Normalize price to % change from entry for comparison
            entry_p = st["entry_price"]
            pct_change = (mh["Close"].values.astype(float) / entry_p - 1) * 100
            ax_price.plot(et_times, pct_change, color=color, linewidth=1.2, label=label, alpha=0.85)

            # Entry marker
            if st["entry_time"] is not None:
                et_entry = st["entry_time"].tz_convert(ET_TZ) if st["entry_time"].tzinfo else st["entry_time"]
                ax_price.axvline(x=et_entry, color=color, linestyle="--", alpha=0.3, linewidth=0.8)
                ax_price.plot(et_entry, 0, marker="^", color=color, markersize=10, zorder=5)

            # Scale-in marker
            if st.get("scale_in_time") is not None:
                et_si = st["scale_in_time"].tz_convert(ET_TZ) if st["scale_in_time"].tzinfo else st["scale_in_time"]
                si_pct = (st["scale_in_price"] / entry_p - 1) * 100
                ax_price.plot(et_si, si_pct, marker="P", color=color, markersize=9, zorder=5,
                             markeredgecolor="white", markeredgewidth=1)

            # Partial sell marker
            if st["partial_sell_time"] is not None:
                et_ps = st["partial_sell_time"].tz_convert(ET_TZ) if st["partial_sell_time"].tzinfo else st["partial_sell_time"]
                ps_pct = (st["partial_sell_price"] / entry_p - 1) * 100
                ax_price.plot(et_ps, ps_pct, marker="D", color=color, markersize=8, zorder=5,
                             markeredgecolor="white", markeredgewidth=1)

            # Exit marker
            if st["exit_time"] is not None and st["exit_price"] is not None:
                et_exit = st["exit_time"].tz_convert(ET_TZ) if st["exit_time"].tzinfo else st["exit_time"]
                exit_pct = (st["exit_price"] / entry_p - 1) * 100
                marker = "v" if st["pnl"] < 0 else "s"
                ax_price.plot(et_exit, exit_pct, marker=marker, color=color, markersize=10, zorder=5,
                             markeredgecolor="white", markeredgewidth=1)

        # Reference lines
        ax_price.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
        ax_price.axhline(y=-STOP_LOSS_PCT, color="#f44336", linestyle=":", alpha=0.4, label=f"Stop Loss (-{STOP_LOSS_PCT}%)")
        ax_price.axhline(y=PARTIAL_SELL_PCT, color="#4CAF50", linestyle=":", alpha=0.4, label=f"Partial Sell (+{PARTIAL_SELL_PCT}%)")

        ax_price.set_title(f"{date_str} - Price Action (% from entry)", fontsize=11, fontweight="bold")
        ax_price.set_ylabel("% from Entry")
        ax_price.legend(fontsize=7, loc="upper left", ncol=2)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p", tz=ET_TZ))
        ax_price.tick_params(axis="x", rotation=30, labelsize=8)
        ax_price.grid(alpha=0.3)
    else:
        ax_price.text(0.5, 0.5, f"{date_str}\nNo trades triggered", ha="center", va="center",
                     fontsize=14, transform=ax_price.transAxes, color="gray")
        ax_price.set_title(f"{date_str} - No Trades", fontsize=11)

    # --- P&L BAR CHART ---
    all_with_trades = [s for s in states if s["entry_price"] is not None]
    if all_with_trades:
        tickers = [s["ticker"] for s in all_with_trades]
        pnls = [s["pnl"] for s in all_with_trades]
        bar_colors = ["#4CAF50" if p > 0 else "#f44336" for p in pnls]
        reasons = [s["exit_reason"] for s in all_with_trades]

        y_pos = range(len(tickers))
        bars = ax_pnl.barh(y_pos, pnls, color=bar_colors, edgecolor="white", height=0.6)
        ax_pnl.set_yticks(y_pos)
        ax_pnl.set_yticklabels([f"{t} ({r})" for t, r in zip(tickers, reasons)], fontsize=8)
        ax_pnl.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
        ax_pnl.invert_yaxis()

        # Annotate bars with dollar amounts
        for j, (bar, pnl) in enumerate(zip(bars, pnls)):
            x_pos = bar.get_width()
            align = "left" if pnl >= 0 else "right"
            offset = 5 if pnl >= 0 else -5
            ax_pnl.annotate(f"${pnl:+.0f}", xy=(x_pos, j), xytext=(offset, 0),
                           textcoords="offset points", ha=align, va="center", fontsize=8,
                           fontweight="bold", color=bar_colors[j])

        day_total = sum(pnls)
        total_color = "#4CAF50" if day_total >= 0 else "#f44336"
        ax_pnl.set_title(f"P&L per Trade | Day Total: ${day_total:+,.0f}", fontsize=11,
                         fontweight="bold", color=total_color)
        ax_pnl.set_xlabel("P&L ($)")
        ax_pnl.grid(alpha=0.3, axis="x")
    else:
        no_bo_count = len([s for s in states if s["exit_reason"] == "NO_BREAKOUT"])
        ax_pnl.text(0.5, 0.5, f"0 trades\n{no_bo_count} stocks watched\n(none broke PM high)",
                   ha="center", va="center", fontsize=11, transform=ax_pnl.transAxes, color="gray")
        ax_pnl.set_title(f"P&L: $0", fontsize=11, color="gray")


# ─── MAIN ────────────────────────────────────────────────────────────────────
print(f"\nRunning weekly backtest: {TEST_DATES[0]} to {TEST_DATES[-1]}")
print(f"Config: ${DAILY_CASH:,}/day | {TRADE_PCT*100:.0f}%/trade | Stop: {STOP_LOSS_PCT}%")
print(f"        Sell {int(PARTIAL_SELL_FRAC*100)}% @ +{PARTIAL_SELL_PCT}% | Trail: ATR({ATR_PERIOD})x{ATR_MULTIPLIER} | Hard sell 4 PM")
print(f"        Confirm: {CONFIRM_ABOVE}/{CONFIRM_WINDOW} candles | Pullback: {PULLBACK_PCT}% near PM high")
print(f"        Scale-in: {'ON (+' + str(SCALE_IN_TRIGGER_PCT) + '%)' if SCALE_IN else 'OFF'}\n")

week_results = []

for date_str in TEST_DATES:
    sys.stdout.write(f"  Processing {date_str}...")
    sys.stdout.flush()
    picks = load_day_candidates(date_str)
    states, ending_cash = simulate_day(picks)
    traded_count = sum(1 for s in states if s["entry_price"] is not None)
    day_pnl = sum(s["pnl"] for s in states if s["entry_price"] is not None)
    week_results.append({
        "date": date_str,
        "picks": picks,
        "states": states,
        "ending_cash": ending_cash,
        "traded": traded_count,
        "day_pnl": day_pnl,
    })
    print(f" {traded_count} trades, P&L: ${day_pnl:+,.2f}")

# --- GENERATE CHARTS ---
print("\nGenerating charts...")

# Per-day charts: 5 rows x 2 columns (price action + P&L bars)
fig, axes = plt.subplots(len(TEST_DATES), 2, figsize=(20, 5 * len(TEST_DATES)),
                          gridspec_kw={"width_ratios": [2.5, 1]})
fig.suptitle(
    f"Weekly Backtest: {TEST_DATES[0]} to {TEST_DATES[-1]}\n"
    f"Stop: {STOP_LOSS_PCT}% | Sell {int(PARTIAL_SELL_FRAC*100)}% @ +{PARTIAL_SELL_PCT}% | Trail: ATR({ATR_PERIOD})x{ATR_MULTIPLIER} | Scale-in: +{SCALE_IN_TRIGGER_PCT}%\n"
    f"$^ = Buy  |  P = Scale-in  |  $\\diamondsuit$ = Partial Sell  |  $\\blacktriangledown$/■ = Exit",
    fontsize=14, fontweight="bold", y=1.01
)

for i, res in enumerate(week_results):
    row_axes = axes[i] if len(TEST_DATES) > 1 else axes
    plot_day(res["date"], res["picks"], res["states"], i, row_axes)

plt.tight_layout(rect=[0, 0, 1, 0.97])
chart_path = "weekly_backtest.png"
plt.savefig(chart_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Per-day charts saved to: {chart_path}")

# --- WEEKLY SUMMARY CHART ---
fig2, (ax_daily, ax_cum, ax_exits) = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle(f"Weekly Summary: {TEST_DATES[0]} to {TEST_DATES[-1]}", fontsize=13, fontweight="bold")

# Daily P&L bars
dates = [r["date"] for r in week_results]
daily_pnls = [r["day_pnl"] for r in week_results]
bar_colors = ["#4CAF50" if p >= 0 else "#f44336" for p in daily_pnls]
bars = ax_daily.bar(dates, daily_pnls, color=bar_colors, edgecolor="white", width=0.5)
for bar, pnl in zip(bars, daily_pnls):
    ax_daily.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                  f"${pnl:+,.0f}", ha="center", va="bottom" if pnl >= 0 else "top",
                  fontsize=9, fontweight="bold")
ax_daily.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
week_total = sum(daily_pnls)
total_color = "#4CAF50" if week_total >= 0 else "#f44336"
ax_daily.set_title(f"Daily P&L | Week Total: ${week_total:+,.2f}", fontweight="bold", color=total_color)
ax_daily.set_ylabel("P&L ($)")
ax_daily.tick_params(axis="x", rotation=30)
ax_daily.grid(alpha=0.3, axis="y")

# Cumulative P&L
cum_pnl = np.cumsum(daily_pnls)
ax_cum.plot(dates, cum_pnl, marker="o", color="#2196F3", linewidth=2, markersize=8)
ax_cum.fill_between(dates, 0, cum_pnl, where=np.array(cum_pnl) >= 0, alpha=0.15, color="#4CAF50")
ax_cum.fill_between(dates, 0, cum_pnl, where=np.array(cum_pnl) < 0, alpha=0.15, color="#f44336")
ax_cum.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
for d, c in zip(dates, cum_pnl):
    ax_cum.annotate(f"${c:+,.0f}", (d, c), textcoords="offset points",
                   xytext=(0, 10), ha="center", fontsize=8)
ax_cum.set_title("Cumulative P&L", fontweight="bold")
ax_cum.set_ylabel("P&L ($)")
ax_cum.tick_params(axis="x", rotation=30)
ax_cum.grid(alpha=0.3)

# Exit reasons pie
all_exits = {}
for res in week_results:
    for st in res["states"]:
        if st["entry_price"] is not None:
            reason = st["exit_reason"]
            all_exits[reason] = all_exits.get(reason, 0) + 1
if all_exits:
    reason_colors = {"STOP_LOSS": "#f44336", "TRAIL_STOP": "#FF9800", "EOD_CLOSE": "#2196F3"}
    labels = list(all_exits.keys())
    sizes = list(all_exits.values())
    colors = [reason_colors.get(r, "#999") for r in labels]
    ax_exits.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
ax_exits.set_title("Exit Reasons (Week)", fontweight="bold")

plt.tight_layout()
summary_path = "weekly_summary.png"
plt.savefig(summary_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Summary chart saved to: {summary_path}")

# --- PRINT SUMMARY TABLE ---
print(f"\n{'='*80}")
print(f"  WEEKLY SUMMARY: {TEST_DATES[0]} to {TEST_DATES[-1]}")
print(f"{'='*80}")
print(f"{'Date':<14}{'Trades':>8}{'Winners':>10}{'Losers':>9}{'Day P&L':>12}{'Cum P&L':>12}")
print("-" * 65)
cum = 0
total_trades = 0
total_winners = 0
total_losers = 0
for res in week_results:
    traded_states = [s for s in res["states"] if s["entry_price"] is not None]
    winners = sum(1 for s in traded_states if s["pnl"] > 0)
    losers = sum(1 for s in traded_states if s["pnl"] <= 0)
    cum += res["day_pnl"]
    total_trades += res["traded"]
    total_winners += winners
    total_losers += losers
    print(f"{res['date']:<14}{res['traded']:>8}{winners:>10}{losers:>9}  ${res['day_pnl']:>+9,.2f}  ${cum:>+9,.2f}")

print("-" * 65)
win_rate = (total_winners / total_trades * 100) if total_trades > 0 else 0
print(f"{'TOTAL':<14}{total_trades:>8}{total_winners:>10}{total_losers:>9}  ${sum(r['day_pnl'] for r in week_results):>+9,.2f}")
print(f"\n  Win Rate: {win_rate:.1f}%  |  Week P&L: ${cum:+,.2f}")
print(f"{'='*80}")

# Detail per trade
print(f"\n  ALL TRADES:")
print(f"  {'Date':<12}{'Ticker':<8}{'Gap%':>7}{'Entry':>9}{'Exit':>9}{'Reason':<12}{'Size':>9}{'P&L':>10}{'P&L%':>8}")
print("  " + "-" * 84)
for res in week_results:
    for st in res["states"]:
        if st["entry_price"] is not None:
            print(f"  {res['date']:<12}{st['ticker']:<8}{st['gap_pct']:>6.1f}%"
                  f"  ${st['entry_price']:>7.2f}  ${st['exit_price']:>7.2f}"
                  f"  {st['exit_reason']:<12}${st['position_cost']:>7.0f}"
                  f"  ${st['pnl']:>+8.2f}  {st['pnl_pct']:>+6.1f}%")
