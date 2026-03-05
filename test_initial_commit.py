"""Run the EXACT initial commit simulate_day logic to see what it produces."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import test_full as tf
from regime_filters import RegimeFilter
from zoneinfo import ZoneInfo

ET_TZ = ZoneInfo("America/New_York")

# --- INITIAL COMMIT CONFIG ---
SLIPPAGE_PCT = 0.05
STOP_LOSS_PCT = 16.0
STARTING_CASH = 10_000
TRADE_PCT = 0.50
MIN_GAP_PCT = 2.0
MIN_PM_VOLUME = 250_000
TOP_N = 10
PARTIAL_SELL_FRAC = 0.90
PARTIAL_SELL_PCT = 15.0
ATR_PERIOD = 8
ATR_MULTIPLIER = 4.25
CONFIRM_ABOVE = 2
CONFIRM_WINDOW = 4
PULLBACK_PCT = 4.0
PULLBACK_TIMEOUT = 24
SCALE_IN = 1
SCALE_IN_TRIGGER_PCT = 19.0
N_EXIT_TRANCHES = 3
PARTIAL_SELL_FRAC_2 = 0.35
PARTIAL_SELL_PCT_2 = 25.0
SPY_SMA_PERIOD = 50  # initial commit used 50, not 40


def simulate_day_initial(picks, starting_cash):
    """EXACT copy of initial commit simulate_day — single-pass, no T+1."""
    cash = starting_cash
    trade_size_base = starting_cash * TRADE_PCT
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
            "partial_sold_2": False,
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

            # ATR
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

            # --- ENTRY LOGIC (3-phase) ---
            if st["entry_price"] is None:
                entered = False

                if not st["breakout_confirmed"]:
                    st["recent_closes"].append(c_close > pm_high)
                    if len(st["recent_closes"]) > CONFIRM_WINDOW:
                        st["recent_closes"] = st["recent_closes"][-CONFIRM_WINDOW:]
                    if sum(st["recent_closes"]) >= CONFIRM_ABOVE:
                        st["breakout_confirmed"] = True
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
                else:
                    if c_close > pm_high:
                        entered = True

                if entered:
                    trades_taken += 1
                    if trades_taken <= 3:
                        position_size = trade_size_base
                    else:
                        position_size = starting_cash * 0.10
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

            # --- SCALE-IN (inline) ---
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

            # Stop loss
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

            # Trailing stop
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

            # Multi-tranche exits
            if N_EXIT_TRANCHES == 3:
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
                        st["partial_sell_time"] = ts
                        st["partial_sell_price"] = target_price
                        cash += proceeds

                if st["remaining_shares"] > 0 and st["partial_sold"] and not st["partial_sold_2"]:
                    target_price_2 = st["entry_price"] * (1 + PARTIAL_SELL_PCT_2 / 100)
                    if c_high >= target_price_2:
                        sell_shares = st["shares"] * PARTIAL_SELL_FRAC_2
                        if sell_shares > st["remaining_shares"]:
                            sell_shares = st["remaining_shares"]
                        sell_price = target_price_2 * (1 - SLIPPAGE_PCT / 100)
                        proceeds = sell_shares * sell_price
                        st["total_exit_value"] += proceeds
                        st["remaining_shares"] -= sell_shares
                        st["partial_sold_2"] = True
                        st["trailing_active"] = True
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
            if st["exit_reason"] not in ("STOP_LOSS", "TRAIL_STOP", "TARGET"):
                st["exit_reason"] = "PARTIAL+EOD" if st["partial_sold"] else "EOD_CLOSE"
            st["exit_price"] = last_close
            st["exit_time"] = st["mh"].index[-1]
            cash += proceeds

    for st in states:
        if st["entry_price"] is not None:
            cost = st["position_cost"]
            st["pnl"] = st["total_exit_value"] - cost
            st["pnl_pct"] = (st["pnl"] / cost) * 100 if cost > 0 else 0
        else:
            st["pnl"] = 0
            st["pnl_pct"] = 0

    return states, cash


# Load data (use ALL 5 dirs for comparison with $71M run)
DATA_DIRS = tf.DEFAULT_DATA_DIRS
ALL_DATES, daily_picks = tf.load_all_picks(DATA_DIRS)

# Test with SMA(50) first (initial commit config)
for sma in [50, 40]:
    regime = RegimeFilter(
        spy_ma_period=sma,
        enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
    )
    regime.load_data(ALL_DATES[0], ALL_DATES[-1])

    rolling_cash = STARTING_CASH
    total_trades = 0
    wins = 0
    scale_ins = 0
    regime_skipped = 0

    for date_str in ALL_DATES:
        should_trade, _, _ = regime.check(date_str)
        if not should_trade:
            regime_skipped += 1
            continue
        picks = daily_picks.get(date_str, [])
        if not picks:
            continue
        states, ending_cash = simulate_day_initial(picks, rolling_cash)
        for s in states:
            if s["entry_price"] is not None:
                total_trades += 1
                if s["pnl"] > 0:
                    wins += 1
                if s.get("scaled_in") and s.get("scale_in_time"):
                    scale_ins += 1
        rolling_cash = ending_cash

    wr = wins / total_trades * 100 if total_trades > 0 else 0
    print(f"\n  INITIAL COMMIT CODE + SMA({sma})")
    print(f"  Config: 50% flat sizing, no T+1, single-pass, inline SI")
    print(f"  Data: {len(ALL_DATES)} days, {len(DATA_DIRS)} dirs")
    print(f"  Final Cash: ${rolling_cash:>15,.2f}")
    print(f"  Trades: {total_trades}  WR: {wr:.1f}%  SI: {scale_ins}  Regime Skip: {regime_skipped}")
