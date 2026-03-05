"""
Reproduce $71M by recreating the OLD two-pass structure:
  Pass 1: detect scale-in (before partial sell), then execute partial sell
  Pass 2: execute scale-in (after partial sell) on remaining shares
This is the "split" behavior that produced $71M.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import test_full as tf
from regime_filters import RegimeFilter
from zoneinfo import ZoneInfo

ET_TZ = ZoneInfo("America/New_York")
DATA_DIRS = tf.DEFAULT_DATA_DIRS
ALL_DATES, daily_picks = tf.load_all_picks(DATA_DIRS)

regime = RegimeFilter(
    spy_ma_period=tf.SPY_SMA_PERIOD,
    enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
)
regime.load_data(ALL_DATES[0], ALL_DATES[-1])


def simulate_day_old_twopass(picks, starting_cash, cash_account=False):
    """OLD two-pass structure that produced $71M:
    Pass 1: exits + scale-in DETECTION (not execution) + entry signal detection
    Pass 2: allocate entries + execute scale-ins (after partial sells freed cash)
    Key difference: scale-in executes AFTER partial sell, so it adds to the 10% remaining.
    """
    cash = starting_cash
    unsettled = 0.0
    trade_pct = tf._get_trade_pct(starting_cash)
    trade_size_base = starting_cash * trade_pct
    trades_taken = 0

    def _recv(amount):
        nonlocal cash, unsettled
        if cash_account:
            unsettled += amount
        else:
            cash += amount

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
            "pm_volume": pick.get("pm_volume", 0),
            "mh": pick["market_hour_candles"],
            "entry_price": None,
            "original_entry_price": None,
            "position_cost": 0.0,
            "shares": 0,
            "remaining_shares": 0,
            "total_exit_value": 0.0,
            "total_cash_spent": 0.0,
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
            "partial_sell_time_2": None,
            "partial_sell_price_2": None,
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
            "signal_close_price": None,
            "breakeven_active": False,
            "waiting_for_cash": False,
            "cash_wait_entry": False,
            "vol_capped": False,
        })

    for ts in all_timestamps:
        entry_candidates = []
        scale_in_candidates = []  # OLD behavior: deferred scale-in

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
            if len(st["true_ranges"]) > tf.ATR_PERIOD:
                st["true_ranges"] = st["true_ranges"][-tf.ATR_PERIOD:]
            st["prev_candle_close"] = c_close

            pm_high = st["premarket_high"]

            # === OPEN POSITION: exits ===
            if st["entry_price"] is not None:
                if c_high > st["highest_since_entry"]:
                    st["highest_since_entry"] = c_high

                # Scale-in DETECTION (Pass 1, before partial sell)
                # Only DETECT here — execution is deferred to Pass 2
                if (tf.SCALE_IN and not st["scaled_in"]
                        and st["remaining_shares"] > 0
                        and not st["partial_sold"]):  # gate check happens HERE (before partial sell)
                    trigger_price = st["entry_price"] * (1 + tf.SCALE_IN_TRIGGER_PCT / 100)
                    if c_high >= trigger_price:
                        scale_in_candidates.append((st, trigger_price))
                        st["scaled_in"] = True  # mark so it doesn't detect again

                # Stop loss (uses entry_price, not original)
                if st["remaining_shares"] > 0 and not st["trailing_active"]:
                    stop_price = st["entry_price"] * (1 - tf.STOP_LOSS_PCT / 100)
                    if c_low <= stop_price:
                        sell_price = stop_price * (1 - tf.SLIPPAGE_PCT / 100)
                        proceeds = st["remaining_shares"] * sell_price
                        st["total_exit_value"] += proceeds
                        st["remaining_shares"] = 0
                        st["exit_reason"] = "STOP_LOSS"
                        st["exit_price"] = stop_price
                        st["exit_time"] = ts
                        st["done"] = True
                        _recv(proceeds)
                        continue

                # Trailing stop
                if st["remaining_shares"] > 0 and st["trailing_active"]:
                    atr = sum(st["true_ranges"]) / len(st["true_ranges"]) if st["true_ranges"] else 0
                    trail_stop = st["highest_since_entry"] - (atr * tf.ATR_MULTIPLIER)
                    if c_low <= trail_stop:
                        sell_price = trail_stop * (1 - tf.SLIPPAGE_PCT / 100)
                        proceeds = st["remaining_shares"] * sell_price
                        st["total_exit_value"] += proceeds
                        st["remaining_shares"] = 0
                        st["exit_reason"] = "TRAIL_STOP"
                        st["exit_price"] = trail_stop
                        st["exit_time"] = ts
                        st["done"] = True
                        _recv(proceeds)
                        continue

                # Partial sells (uses entry_price, NOT original_entry_price)
                if tf.N_EXIT_TRANCHES == 3:
                    if st["remaining_shares"] > 0 and not st["partial_sold"]:
                        target_price = st["entry_price"] * (1 + tf.PARTIAL_SELL_PCT / 100)
                        if c_high >= target_price:
                            sell_shares = st["shares"] * tf.PARTIAL_SELL_FRAC
                            if sell_shares > st["remaining_shares"]:
                                sell_shares = st["remaining_shares"]
                            sell_price = target_price * (1 - tf.SLIPPAGE_PCT / 100)
                            proceeds = sell_shares * sell_price
                            st["total_exit_value"] += proceeds
                            st["remaining_shares"] -= sell_shares
                            st["partial_sold"] = True
                            st["partial_sell_time"] = ts
                            st["partial_sell_price"] = target_price
                            _recv(proceeds)

                    if st["remaining_shares"] > 0 and st["partial_sold"] and not st["partial_sold_2"]:
                        target_price_2 = st["entry_price"] * (1 + tf.PARTIAL_SELL_PCT_2 / 100)
                        if c_high >= target_price_2:
                            sell_shares = st["shares"] * tf.PARTIAL_SELL_FRAC_2
                            if sell_shares > st["remaining_shares"]:
                                sell_shares = st["remaining_shares"]
                            sell_price = target_price_2 * (1 - tf.SLIPPAGE_PCT / 100)
                            proceeds = sell_shares * sell_price
                            st["total_exit_value"] += proceeds
                            st["remaining_shares"] -= sell_shares
                            st["partial_sold_2"] = True
                            st["trailing_active"] = True
                            st["partial_sell_time_2"] = ts
                            st["partial_sell_price_2"] = target_price_2
                            _recv(proceeds)

                # EOD exit
                if tf.EOD_EXIT_MINUTES > 0 and st["remaining_shares"] > 0:
                    ts_et = ts.astimezone(ET_TZ) if ts.tzinfo else ts
                    minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
                    if minutes_to_close <= tf.EOD_EXIT_MINUTES:
                        sell_price = c_close * (1 - tf.SLIPPAGE_PCT / 100)
                        proceeds = st["remaining_shares"] * sell_price
                        st["total_exit_value"] += proceeds
                        st["remaining_shares"] = 0
                        if st["exit_reason"] not in ("STOP_LOSS", "TRAIL_STOP", "TARGET"):
                            st["exit_reason"] = "PARTIAL+EOD" if st["partial_sold"] else "EOD_EARLY"
                        st["exit_price"] = c_close
                        st["exit_time"] = ts
                        st["done"] = True
                        _recv(proceeds)
                        continue

                if st["remaining_shares"] <= 0:
                    st["done"] = True
                continue

            # === NO POSITION: entry signal detection ===
            entered = False
            if not st["breakout_confirmed"]:
                st["recent_closes"].append(c_close > pm_high)
                if len(st["recent_closes"]) > tf.CONFIRM_WINDOW:
                    st["recent_closes"] = st["recent_closes"][-tf.CONFIRM_WINDOW:]
                if sum(st["recent_closes"]) >= tf.CONFIRM_ABOVE:
                    st["breakout_confirmed"] = True
            elif not st["pullback_detected"]:
                st["candles_since_confirm"] += 1
                pullback_zone = pm_high * (1 + tf.PULLBACK_PCT / 100)
                if c_low <= pullback_zone:
                    st["pullback_detected"] = True
                    if c_close > pm_high:
                        entered = True
                elif st["candles_since_confirm"] >= tf.PULLBACK_TIMEOUT:
                    if c_close > pm_high:
                        entered = True
            else:
                if c_close > pm_high:
                    entered = True

            if entered:
                st["signal_close_price"] = c_close
                entry_candidates.append(st)

        # === PASS 2: Execute entries + deferred scale-ins ===

        # Build allocation queue: entries first, then scale-ins
        allocation_queue = []
        for st in entry_candidates:
            allocation_queue.append(("entry", st))
        for st, trigger_price in scale_in_candidates:
            allocation_queue.append(("scale_in", st, trigger_price))

        for item in allocation_queue:
            action = item[0]

            if action == "entry":
                st = item[1]
                if st["done"] or st["entry_price"] is not None:
                    continue

                trades_taken += 1
                position_size = trade_size_base

                fill_price = st["signal_close_price"]

                # Volume cap
                if tf.VOL_CAP_PCT > 0:
                    pre_entry = st["mh"].loc[st["mh"].index <= ts]
                    vol_shares = pre_entry["Volume"].sum() if len(pre_entry) > 0 else 0
                    dollar_vol = fill_price * vol_shares
                    vol_limit = dollar_vol * (tf.VOL_CAP_PCT / 100)
                    if vol_limit > 0 and position_size > vol_limit:
                        position_size = vol_limit
                        st["vol_capped"] = True
                    if position_size < 50:
                        trades_taken -= 1
                        st["exit_reason"] = "LOW_VOL"
                        st["done"] = True
                        continue

                if cash < position_size:
                    if cash > 0:
                        position_size = cash
                    else:
                        trades_taken -= 1
                        st["exit_reason"] = "NO_CASH"
                        st["done"] = True
                        continue

                st["entry_price"] = fill_price * (1 + tf.SLIPPAGE_PCT / 100)
                st["original_entry_price"] = st["entry_price"]
                st["position_cost"] = position_size
                st["total_cash_spent"] = position_size
                st["original_position_size"] = position_size
                st["shares"] = position_size / st["entry_price"]
                st["remaining_shares"] = st["shares"]
                st["entry_time"] = ts
                c_high_now = float(st["mh"].loc[ts]["High"])
                st["highest_since_entry"] = c_high_now
                cash -= position_size

            elif action == "scale_in":
                st, trigger_price = item[1], item[2]
                if st["done"] or st["remaining_shares"] <= 0:
                    continue
                add_size = st["original_position_size"] * tf.SCALE_IN_FRAC
                # Volume cap
                if tf.VOL_CAP_PCT > 0:
                    pre_si = st["mh"].loc[st["mh"].index <= ts]
                    si_vol = pre_si["Volume"].sum() if len(pre_si) > 0 else 0
                    si_dvol = trigger_price * si_vol
                    si_limit = si_dvol * (tf.VOL_CAP_PCT / 100) - st["position_cost"]
                    if si_limit > 0 and add_size > si_limit:
                        add_size = si_limit
                    if add_size < 50:
                        add_size = 0
                if cash >= add_size and add_size > 0:
                    add_price = trigger_price * (1 + tf.SLIPPAGE_PCT / 100)
                    add_shares = add_size / add_price
                    remaining_cost = st["remaining_shares"] * st["entry_price"]
                    total_cost = remaining_cost + add_size
                    total_shares = st["remaining_shares"] + add_shares
                    st["entry_price"] = total_cost / total_shares
                    st["position_cost"] = total_cost
                    st["total_cash_spent"] += add_size
                    st["shares"] = total_shares
                    st["remaining_shares"] = total_shares
                    st["scale_in_time"] = ts
                    st["scale_in_price"] = trigger_price
                    cash -= add_size

    # EOD close
    for st in states:
        if st["entry_price"] is not None and st["remaining_shares"] > 0:
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - tf.SLIPPAGE_PCT / 100)
            proceeds = st["remaining_shares"] * sell_price
            st["total_exit_value"] += proceeds
            st["remaining_shares"] = 0
            if st["exit_reason"] not in ("STOP_LOSS", "TRAIL_STOP", "TARGET"):
                st["exit_reason"] = "PARTIAL+EOD" if st["partial_sold"] else "EOD_CLOSE"
            st["exit_price"] = last_close
            st["exit_time"] = st["mh"].index[-1]
            _recv(proceeds)

    cash += unsettled

    for st in states:
        if st["entry_price"] is not None:
            cost = st["total_cash_spent"]
            st["pnl"] = st["total_exit_value"] - cost
            st["pnl_pct"] = (st["pnl"] / cost) * 100 if cost > 0 else 0
        else:
            st["pnl"] = 0
            st["pnl_pct"] = 0

    return states, cash


# Run with old config
tf.SCALE_IN = 1
tf.SCALE_IN_TRIGGER_PCT = 19.0
tf.SCALE_IN_GATE_PARTIAL = True
tf.BREAKEVEN_STOP_PCT = 0.0
tf.ENTRY_CUTOFF_MINUTES = 0

rolling_cash = tf.STARTING_CASH
total_trades = 0
wins = 0
scale_ins = 0

for date_str in ALL_DATES:
    should_trade, _, _ = regime.check(date_str)
    if not should_trade:
        continue
    picks = daily_picks.get(date_str, [])
    if not picks:
        continue
    is_cash = rolling_cash < tf.MARGIN_THRESHOLD
    states, ending_cash = simulate_day_old_twopass(picks, rolling_cash, cash_account=is_cash)
    for s in states:
        if s["entry_price"] is not None:
            total_trades += 1
            if s["pnl"] > 0:
                wins += 1
            if s.get("scaled_in") and s.get("scale_in_time") is not None:
                scale_ins += 1
    rolling_cash = ending_cash

wr = wins / total_trades * 100 if total_trades > 0 else 0
print(f"\n{'='*80}")
print(f"  OLD TWO-PASS (detect in Pass 1, execute in Pass 2)")
print(f"  SI 19%, gate ON (exact old behavior)")
print(f"{'='*80}")
print(f"  Final Cash: ${rolling_cash:>15,.2f}")
print(f"  Trades: {total_trades}  WR: {wr:.1f}%  Scale-ins: {scale_ins}")
print(f"\n  Target: ~$71,084,943 (old baseline)")
print(f"  Current two-pass (SI 19%, gate ON): $52,147")
print(f"  Current two-pass (SI OFF): $90,837")
