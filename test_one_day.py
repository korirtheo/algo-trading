"""
Quick 1-day backtest to verify sell logic: 50% at +20%, trail rest, hard sell 4 PM.
Loads only the tickers needed for a single day from stored CSVs.
"""
import os
import re
import pandas as pd
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
TEST_DATE = "2026-02-27"
ET_TZ = ZoneInfo("America/New_York")

# Load all tickers that have intraday data
intraday_dir = os.path.join(DATA_DIR, "intraday")
daily_dir = os.path.join(DATA_DIR, "daily")

all_tickers = [f.replace(".csv", "") for f in os.listdir(intraday_dir) if f.endswith(".csv")]
print(f"Available tickers: {len(all_tickers)}")
print(f"Test date: {TEST_DATE}")

# --- LOAD DATA FOR THIS DATE ONLY ---
print(f"\nScanning for premarket gainers on {TEST_DATE}...")
day_candidates = []

loaded = 0
for ticker in all_tickers:
    if _is_warrant_or_unit(ticker):
        continue
    ipath = os.path.join(intraday_dir, f"{ticker}.csv")
    dpath = os.path.join(daily_dir, f"{ticker}.csv")

    try:
        idf = pd.read_csv(ipath, index_col=0, parse_dates=True)
    except Exception:
        continue

    # Convert to Eastern Time for correct hour comparisons
    if idf.index.tz is not None:
        et_index = idf.index.tz_convert(ET_TZ)
    else:
        et_index = idf.index.tz_localize("UTC").tz_convert(ET_TZ)

    # Filter to test date (using ET dates)
    day_mask = et_index.strftime("%Y-%m-%d") == TEST_DATE
    day_candles = idf[day_mask]
    et_day = et_index[day_mask]
    if len(day_candles) == 0:
        continue

    loaded += 1

    # Split premarket (before 9:30 AM ET) vs market hours (9:30 AM - 4:00 PM ET)
    pm_mask = (et_day.hour < 9) | ((et_day.hour == 9) & (et_day.minute < 30))
    mh_mask = (
        ((et_day.hour == 9) & (et_day.minute >= 30))
        | ((et_day.hour >= 10) & (et_day.hour < 16))
    )
    premarket = day_candles[pm_mask]
    market_hours = day_candles[mh_mask]

    if len(market_hours) == 0:
        continue

    # Market open price (first candle of market hours)
    market_open = float(market_hours.iloc[0]["Open"])

    # Premarket high (buy trigger only, NOT used for ranking)
    if len(premarket) > 0:
        premarket_high = float(premarket["High"].max())
    else:
        premarket_high = market_open

    # Premarket volume check
    pm_volume = int(premarket["Volume"].sum()) if len(premarket) > 0 else 0
    if pm_volume < MIN_PM_VOLUME:
        continue

    # Previous close from daily data
    prev_close = None
    if os.path.exists(dpath):
        try:
            ddf = pd.read_csv(dpath, index_col=0, parse_dates=True)
            date_naive = pd.Timestamp(TEST_DATE)
            ddf_dates = ddf.index.tz_localize(None) if ddf.index.tz else ddf.index
            prev_mask = ddf_dates < date_naive
            if prev_mask.any():
                prev_idx = ddf.index[prev_mask][-1]
                prev_close = float(ddf.loc[prev_idx, "Close"])
        except Exception:
            pass

    if prev_close is None or prev_close <= 0:
        continue

    # Gap % = (market open - prev close) / prev close
    gap_pct = (market_open - prev_close) / prev_close * 100
    if gap_pct < MIN_GAP_PCT:
        continue

    day_candidates.append({
        "ticker": ticker,
        "gap_pct": gap_pct,
        "market_open": market_open,
        "premarket_high": premarket_high,
        "prev_close": prev_close,
        "market_hour_candles": market_hours,
        "premarket_candles": premarket,
    })

print(f"  Tickers with data on {TEST_DATE}: {loaded}")
print(f"  Gap-up candidates (>{MIN_GAP_PCT}%): {len(day_candidates)}")

# Sort and pick top 10
day_candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
top_picks = day_candidates[:TOP_N]

print(f"\n{'='*80}")
print(f"  TOP {TOP_N} PREMARKET GAINERS for {TEST_DATE}")
print(f"{'='*80}")
print(f"{'Rank':<6}{'Ticker':<8}{'Gap%':>8}{'Prev Cl':>10}{'Mkt Open':>10}{'PM High':>10}{'MH Candles':>12}")
print("-" * 66)
for i, p in enumerate(top_picks, 1):
    print(f"{i:<6}{p['ticker']:<8}{p['gap_pct']:>7.1f}%{p['prev_close']:>10.4f}{p['market_open']:>10.4f}{p['premarket_high']:>10.4f}{len(p['market_hour_candles']):>12}")

# --- SIMULATE TRADES WITH CASH MANAGEMENT ---
print(f"\n{'='*80}")
print(f"  TRADE SIMULATION (${DAILY_CASH:,} daily, {TRADE_PCT*100:.0f}%/trade)")
print(f"  Sell {int(PARTIAL_SELL_FRAC*100)}% @ +{PARTIAL_SELL_PCT}% | Trail: ATR({ATR_PERIOD})x{ATR_MULTIPLIER} | Hard sell 4 PM")
print(f"  Confirm: {CONFIRM_ABOVE}/{CONFIRM_WINDOW} candles | Pullback: {PULLBACK_PCT}% near PM high")
print(f"{'='*80}")

cash = DAILY_CASH
trade_size_base = DAILY_CASH * TRADE_PCT
trades_taken = 0
total_pnl = 0

# Collect all candle timestamps across all picks, simulate in time order
all_timestamps = set()
for pick in top_picks:
    all_timestamps.update(pick["market_hour_candles"].index.tolist())
all_timestamps = sorted(all_timestamps)

# Track state per pick
states = []
for pick in top_picks:
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
        "done": False,
        "recent_closes": [],
        "breakout_confirmed": False,
        "pullback_detected": False,
        "candles_since_confirm": 0,
        "true_ranges": [],
        "prev_candle_close": None,
        "scaled_in": False,
        "original_position_size": 0.0,
    })

# Walk through every 2-min candle in time order
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

        # --- IN TRADE: update highest price since entry ---
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
                    cash -= add_size
                st["scaled_in"] = True

        # --- IN TRADE: check stop loss (hard stop before partial sell) ---
        if st["remaining_shares"] > 0 and not st["trailing_active"]:
            stop_price = st["entry_price"] * (1 - STOP_LOSS_PCT / 100)
            if c_low <= stop_price:
                sell_price = stop_price * (1 - SLIPPAGE_PCT / 100)
                proceeds = st["remaining_shares"] * sell_price
                st["total_exit_value"] += proceeds
                st["remaining_shares"] = 0
                st["exit_reason"] = "STOP_LOSS"
                st["exit_price"] = stop_price
                st["done"] = True
                cash += proceeds
                continue

        # --- IN TRADE: check trailing stop (after partial sell) ---
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
                st["done"] = True
                cash += proceeds
                continue

        # --- IN TRADE: check +20% partial sell target ---
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
                cash += proceeds

        if st["remaining_shares"] <= 0:
            st["done"] = True

# End of day: hard sell anything remaining at market close
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
        cash += proceeds

# Display results
for st in states:
    ticker = st["ticker"]
    gap_pct = st["gap_pct"]
    pm_high = st["premarket_high"]

    if st["entry_price"] is not None:
        cost = st["position_cost"]
        pnl = st["total_exit_value"] - cost
        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
        total_pnl += pnl

        et_entry = st["entry_time"].tz_convert(ET_TZ) if st["entry_time"].tzinfo else st["entry_time"]
        print(f"\n  {ticker} (gap {gap_pct:.1f}%): {st['exit_reason']}")
        print(f"    PM High:  ${pm_high:.4f}")
        print(f"    Entry:    ${st['entry_price']:.4f} at {et_entry.strftime('%I:%M %p ET')}")
        print(f"    Exit:     ${st['exit_price']:.4f}")
        print(f"    Size:     ${cost:,.2f}")
        print(f"    P&L:      ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    elif st["exit_reason"] == "NO_CASH":
        print(f"\n  {ticker} (gap {gap_pct:.1f}%): SKIPPED - no cash available")
    else:
        print(f"\n  {ticker} (gap {gap_pct:.1f}%): NO BREAKOUT - price never beat PM high")

print(f"\n{'='*80}")
print(f"  DAY TOTAL P&L: ${total_pnl:+.2f}")
print(f"  ENDING CASH:   ${cash:,.2f} (started with ${DAILY_CASH:,})")
print(f"{'='*80}")
