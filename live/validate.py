"""
Validate live strategy module against historical backtest.

Runs the extracted L strategy logic (strategies/low_float_squeeze.py) on the
same historical data as test_low_float_squeeze.py and compares trade-for-trade.

Usage:
    python -m live.validate
    python -m live.validate stored_data_2023
"""
import sys
import os
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_full import load_all_picks, SLIPPAGE_PCT, STARTING_CASH, VOL_CAP_PCT, ET_TZ
from strategies.low_float_squeeze import (
    DEFAULT_PARAMS, create_state, check_signal, check_exit,
    compute_vwap, is_eligible,
)
from config.settings import FLOAT_DATA, EOD_EXIT_MINUTES

ET = ZoneInfo("America/New_York")


def simulate_day_extracted(picks, cash):
    """Simulate one day using the EXTRACTED strategy module.
    This mirrors simulate_day_l() from test_low_float_squeeze.py
    but uses the modular functions from strategies/low_float_squeeze.py.
    """
    unsettled = 0.0
    params = DEFAULT_PARAMS

    # Build unified timestamp index
    all_timestamps = set()
    for pick in picks:
        mh = pick.get("market_hour_candles")
        if mh is not None and len(mh) > 0:
            all_timestamps.update(mh.index.tolist())
    all_timestamps = sorted(all_timestamps)

    if not all_timestamps:
        return [], cash, unsettled

    # Initialize states using extracted module
    states = []
    bar_histories = []
    for pick in picks:
        ticker = pick["ticker"]
        float_shares = FLOAT_DATA.get(ticker)
        if not is_eligible(ticker, pick["gap_pct"], float_shares, params):
            continue

        mh = pick.get("market_hour_candles")
        if mh is None or len(mh) < 10:
            continue

        st = create_state(
            ticker=ticker,
            gap_pct=pick["gap_pct"],
            float_shares=float_shares,
            premarket_high=pick["premarket_high"],
            pm_volume=pick.get("pm_volume", 0),
            params=params,
        )
        # Store mh reference for bar iteration
        st["mh"] = mh
        states.append(st)
        bar_histories.append({"highs": [], "lows": [], "closes": [], "volumes": []})

    for ts in all_timestamps:
        entry_candidates = []

        for idx, st in enumerate(states):
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

            # Update bar history for VWAP
            hist = bar_histories[idx]
            hist["highs"].append(c_high)
            hist["lows"].append(c_low)
            hist["closes"].append(c_close)
            hist["volumes"].append(c_vol)

            # Compute VWAP
            vwap_arr = compute_vwap(hist["highs"], hist["lows"], hist["closes"], hist["volumes"])
            vwap_value = vwap_arr[-1] if len(vwap_arr) > 0 else None

            # --- IN POSITION: check exits ---
            if st["entry_price"] is not None:
                try:
                    ts_et = ts.astimezone(ET)
                except Exception:
                    ts_et = ts

                entry_et = st["entry_time"]
                try:
                    entry_et = entry_et.astimezone(ET)
                except Exception:
                    pass
                minutes_in_trade = (ts_et.hour * 60 + ts_et.minute) - (entry_et.hour * 60 + entry_et.minute)
                minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)

                should_exit, exit_price, exit_reason = check_exit(
                    st, c_high, c_low, c_close,
                    minutes_in_trade, minutes_to_close,
                    slippage_pct=SLIPPAGE_PCT,
                    eod_exit_minutes=EOD_EXIT_MINUTES,
                    params=params,
                )

                if should_exit:
                    if exit_reason == "PARTIAL":
                        partial_shares = st["shares"] * (params["partial_sell_pct"] / 100)
                        sell_price = exit_price * (1 - SLIPPAGE_PCT / 100)
                        st["partial_proceeds"] += partial_shares * sell_price
                        st["shares"] -= partial_shares
                        cash += partial_shares * sell_price
                        if st["shares"] <= 0.001:
                            st["pnl"] = st["partial_proceeds"] - st["position_cost"]
                            st["exit_price"] = exit_price
                            st["exit_time"] = ts
                            st["exit_reason"] = "TARGET"
                            st["entry_price"] = None
                            st["shares"] = 0
                            st["done"] = True
                    else:
                        sell_price = exit_price * (1 - SLIPPAGE_PCT / 100)
                        proceeds = st["shares"] * sell_price
                        st["pnl"] = st["partial_proceeds"] + proceeds - st["position_cost"]
                        st["exit_price"] = exit_price
                        st["exit_time"] = ts
                        st["exit_reason"] = exit_reason
                        st["entry_price"] = None
                        st["shares"] = 0
                        st["done"] = True
                        cash += proceeds
                continue

            # --- NOT IN POSITION: check signal ---
            fired = check_signal(st, c_open, c_high, c_low, c_close, c_vol, vwap_value, params)
            if fired:
                entry_candidates.append(st)

        # --- ALLOCATION ---
        entry_candidates.sort(key=lambda s: -s["gap_pct"])

        tickers_taken = set()
        for st in entry_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue
            if st["ticker"] in tickers_taken:
                continue

            try:
                ts_et = ts.astimezone(ET)
            except Exception:
                ts_et = ts
            mins_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
            if mins_to_close <= EOD_EXIT_MINUTES:
                continue

            if cash < 100:
                continue
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
            st["_orig_entry_price"] = entry_price
            st["entry_time"] = ts
            st["position_cost"] = trade_size
            st["shares"] = trade_size / entry_price
            st["_orig_shares"] = trade_size / entry_price
            st["highest_since_entry"] = entry_price
            cash -= trade_size
            tickers_taken.add(st["ticker"])

    # EOD close remaining
    for st in states:
        if st["entry_price"] is not None and st["shares"] > 0:
            last_ts = st["mh"].index[-1]
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["shares"] * sell_price
            st["pnl"] = st["partial_proceeds"] + proceeds - st["position_cost"]
            st["exit_price"] = last_close
            st["exit_time"] = last_ts
            st["exit_reason"] = "EOD_CLOSE"
            st["entry_price"] = None
            st["shares"] = 0
            st["done"] = True
            cash += proceeds

    return states, cash, unsettled


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    data_dirs = args if args else ["stored_data_combined"]

    print("=" * 70)
    print("VALIDATION: Extracted L Strategy vs Historical Data")
    print("=" * 70)
    print(f"Data dirs: {data_dirs}")
    print(f"Float data: {len(FLOAT_DATA)} tickers")
    print(f"Params: min_gap={DEFAULT_PARAMS['min_gap_pct']}%, "
          f"max_float={DEFAULT_PARAMS['max_float']/1e6:.0f}M, "
          f"stop={DEFAULT_PARAMS['stop_pct']}%, "
          f"trail={DEFAULT_PARAMS['trail_pct']}%@+{DEFAULT_PARAMS['trail_activate_pct']}%")
    print()

    result = load_all_picks(data_dirs)
    if isinstance(result, tuple):
        picks_by_date = result[1]
    else:
        picks_by_date = result
    sorted_dates = sorted(picks_by_date.keys())
    print(f"Loaded {len(sorted_dates)} trading days")

    cash = STARTING_CASH
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    exit_reasons = {}
    daily_pnls = []

    for i, date_str in enumerate(sorted_dates):
        picks = picks_by_date[date_str]
        states, cash, unsettled = simulate_day_extracted(picks, cash)
        cash += unsettled

        day_pnl = sum(st["pnl"] for st in states if st.get("exit_price") is not None)
        day_trades = sum(1 for st in states if st.get("exit_price") is not None)
        day_wins = sum(1 for st in states if st.get("exit_price") is not None and st["pnl"] > 0)

        total_trades += day_trades
        total_wins += day_wins
        total_pnl += day_pnl
        daily_pnls.append(day_pnl)

        for st in states:
            if st.get("exit_reason"):
                exit_reasons[st["exit_reason"]] = exit_reasons.get(st["exit_reason"], 0) + 1

        if day_trades > 0 and (i < 5 or i >= len(sorted_dates) - 3):
            tickers = [st["ticker"] for st in states if st.get("exit_price")]
            print(f"  {date_str}: {day_trades} trades, PnL=${day_pnl:+,.0f}, "
                  f"cash=${cash:,.0f} | {tickers}")

    print()
    print("=" * 70)
    print("RESULTS (Extracted Module)")
    print("=" * 70)
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"  Total PnL:    ${total_pnl:+,.0f}")
    print(f"  Final equity: ${cash:,.0f}")
    print(f"  Trades:       {total_trades}")
    print(f"  Win rate:     {wr:.1f}%")

    if daily_pnls:
        green = sum(1 for p in daily_pnls if p > 0)
        print(f"  Green days:   {green}/{len(daily_pnls)} ({green/len(daily_pnls)*100:.1f}%)")

        pnl_arr = np.array(daily_pnls)
        if pnl_arr.std() > 0:
            sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)
            print(f"  Sharpe:       {sharpe:.2f}")

    print(f"\n  Exit reasons:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")

    print()
    print("Compare these numbers against: python test_low_float_squeeze.py --no-charts")
    print("They should match exactly (same logic, same data, same params).")


if __name__ == "__main__":
    main()
