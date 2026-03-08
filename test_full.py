"""
Full backtest with per-day visual charts and rolling cash.
Generates paginated chart images (5 days per page) + overall summary.
Supports multiple data directories (stored_data + stored_data_oos).

Usage:
  python test_full.py                           # uses both dirs
  python test_full.py stored_data               # single dir
  python test_full.py stored_data stored_data_oos  # explicit
"""

import os
import sys
import re
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from stress_test import run_stress_tests as _run_stress_tests
from optimize import load_all_picks as _load_optimizer_picks


def _is_warrant_or_unit(ticker):
    if ".WS" in ticker or ".RT" in ticker:
        return True
    if re.match(r"^[A-Z]{3,}W$", ticker):
        return True
    if ticker.endswith("WW"):
        return True
    if re.match(r"^[A-Z]{3,}U$", ticker):
        return True
    if re.match(r"^[A-Z]{3,}R$", ticker):
        return True
    return False


# --- CONFIG (Old Phase 2 — pre-Optuna, wider stops, conservative scale-in) ---
SLIPPAGE_PCT = 0.05
STOP_LOSS_PCT = 16.0
STARTING_CASH = 25_000
TRADE_PCT = 0.50  # fallback if adaptive sizing disabled
MIN_GAP_PCT = 2.0
MIN_PM_VOLUME = 250_000
TOP_N = 20
PARTIAL_SELL_FRAC = 0.90
PARTIAL_SELL_PCT = 15.0
ATR_PERIOD = 8
ATR_MULTIPLIER = 4.25
CONFIRM_ABOVE = 2
CONFIRM_WINDOW = 4
PULLBACK_PCT = 4.0
PULLBACK_TIMEOUT = 24
SCALE_IN = 1
SCALE_IN_TRIGGER_PCT = 14.0
SCALE_IN_FRAC = 0.50  # fraction of original position to add
N_EXIT_TRANCHES = 3
PARTIAL_SELL_FRAC_2 = 0.35
PARTIAL_SELL_PCT_2 = 25.0
# Regime filter
SPY_SMA_PERIOD = 40  # skip days when SPY < SMA(40)
# Cash/margin account toggle:
#   Under $25K: cash account — sell proceeds settle T+1 (available next day)
#   At/above $25K: margin account — sell proceeds available instantly
MARGIN_THRESHOLD = 25_000
# Cash wait: keep cash-blocked entries alive for rest of day (True=enabled)
CASH_WAIT_ENABLED = False
# Volume cap: limit position to X% of dollar volume traded up to entry
# Prevents unrealistic fills in thin stocks
VOL_CAP_PCT = 5.0  # 0 = disabled
# Reset stops after partial sell: treat remaining shares as fresh entry
RESET_STOPS_ON_PARTIAL = False
# Early EOD exit: sell remaining shares X minutes before close if profitable
# 0 = disabled (sell at 3:59 PM as before)
EOD_EXIT_MINUTES = 30
# Breakeven stop: move stop to entry price once stock gains this % (0 = disabled)
BREAKEVEN_STOP_PCT = 0.0  # e.g. 7.0 = move stop to entry after +7%
# Runner mode: skip partial sells when stock runs (disabled by optimizer)
RUNNER_MODE = False
RUNNER_TRIGGER_PCT = 25.0  # +25% above entry to activate
RUNNER_VOL_MULT = 3.0  # current candle volume > 3x avg volume to confirm
# Liquidity vacuum: require large candle + volume spike to confirm entry (disabled by optimizer)
LIQ_VACUUM = False
LIQ_CANDLE_ATR_MULT = 2.5  # candle range must be > X * ATR(14)
LIQ_VOL_MULT = 3.0  # candle volume must be > X * avg volume(10)
# Structural stop: use PM_high - X*ATR instead of fixed % stop
STRUCTURAL_STOP = False
STRUCTURAL_STOP_ATR_MULT = 1.5  # stop = max(PM_high - 1.5*ATR, entry - 16%)
# Entry time cutoff: no new entries after this many minutes before close (0 = disabled)
ENTRY_CUTOFF_MINUTES = 0
# Scale-in requires no partial sell first (True = old behavior where scale-in is dead code)
SCALE_IN_GATE_PARTIAL = False  # allow scale-in regardless of partial sell status

# Adaptive position sizing: (balance_threshold, trade_pct)
# Uses the highest tier where balance >= threshold
# Sweep-optimized: 25/40/50% beats both fixed 25% and fixed 50%
ADAPTIVE_SIZING = False
SIZING_TIERS = [
    (0, 0.25),  # $0 - $25K: 25% per trade (cash account phase)
    (25_000, 0.40),  # $25K - $50K: 40% (margin unlocked, ramp up)
    (50_000, 0.50),  # $50K+: 50% (full size)
]


def _get_trade_pct(balance, tiers=None):
    """Return position size % based on account balance."""
    if not ADAPTIVE_SIZING:
        return TRADE_PCT
    if tiers is None:
        tiers = SIZING_TIERS
    pct = tiers[0][1]
    for threshold, tier_pct in tiers:
        if balance >= threshold:
            pct = tier_pct
    return pct


# Data directories
DEFAULT_DATA_DIRS = [
    "stored_data_jan_mar_2025",
    "stored_data_apr_jun_2025",
    "stored_data_jul_2025",
    "stored_data",
    "stored_data_oos",
]
ET_TZ = ZoneInfo("America/New_York")
DAYS_PER_PAGE = 5


def _process_one_day(args):
    """Worker: scan all tickers for one day (runs in separate process)."""
    test_date, ticker_list, idir, ddir = args
    et_tz = ZoneInfo("America/New_York")
    candidates = []
    for ticker in ticker_list:
        if _is_warrant_or_unit(ticker):
            continue
        ipath = os.path.join(idir, f"{ticker}.csv")
        dpath = os.path.join(ddir, f"{ticker}.csv")
        try:
            idf = pd.read_csv(ipath, index_col=0, parse_dates=True)
        except Exception:
            continue
        if idf.index.tz is not None:
            et_index = idf.index.tz_convert(et_tz)
        else:
            et_index = idf.index.tz_localize("UTC").tz_convert(et_tz)

        day_mask = et_index.strftime("%Y-%m-%d") == test_date
        day_candles = idf[day_mask]
        et_day = et_index[day_mask]
        if len(day_candles) == 0:
            continue

        pm_mask = (et_day.hour < 9) | ((et_day.hour == 9) & (et_day.minute < 30))
        mh_mask = ((et_day.hour == 9) & (et_day.minute >= 30)) | (
            (et_day.hour >= 10) & (et_day.hour < 16)
        )
        premarket = day_candles[pm_mask]
        market_hours = day_candles[mh_mask]
        if len(market_hours) == 0:
            continue

        market_open = float(market_hours.iloc[0]["Open"])
        premarket_high = (
            float(premarket["High"].max()) if len(premarket) > 0 else market_open
        )

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

        candidates.append(
            {
                "ticker": ticker,
                "gap_pct": gap_pct,
                "market_open": market_open,
                "premarket_high": premarket_high,
                "prev_close": prev_close,
                "pm_volume": pm_volume,
                "market_hour_candles": market_hours,
            }
        )

    candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
    return test_date, candidates[:TOP_N]


def load_picks_for_dir(data_dir):
    """Load or build picks for a single data directory.
    Cache is keyed by MIN_GAP_PCT and MIN_PM_VOLUME so different configs
    get separate caches instead of overwriting each other.
    """
    cache_tag = f"gap{int(MIN_GAP_PCT)}_vol{int(MIN_PM_VOLUME//1000)}k"
    pickle_path = os.path.join(data_dir, f"fulltest_picks_{cache_tag}.pkl")
    if os.path.exists(pickle_path):
        print(f"  Loading cached picks from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    intraday_dir = os.path.join(data_dir, "intraday")
    daily_dir = os.path.join(data_dir, "daily")
    all_tickers = [
        f.replace(".csv", "") for f in os.listdir(intraday_dir) if f.endswith(".csv")
    ]
    gainers_csv = os.path.join(data_dir, "daily_top_gainers.csv")
    gdf = pd.read_csv(gainers_csv)
    all_dates = sorted(gdf["date"].unique().tolist())

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(
        f"  Scanning {len(all_tickers)} tickers x {len(all_dates)} days in {data_dir}..."
    )

    task_args = [(d, all_tickers, intraday_dir, daily_dir) for d in all_dates]
    daily_picks = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_one_day, args): args[0] for args in task_args
        }
        for future in as_completed(futures):
            date_str, picks = future.result()
            daily_picks[date_str] = picks
            completed += 1
            sys.stdout.write(f"\r    [{completed}/{len(all_dates)}] {date_str}...")
            sys.stdout.flush()

    print(f"\r    Loaded {len(daily_picks)} days.                        ")
    with open(pickle_path, "wb") as f:
        pickle.dump(daily_picks, f)
    print(f"    Cached to {pickle_path}")
    return daily_picks


def load_all_picks(data_dirs):
    """Load picks from all data directories, merge by date."""
    merged = {}
    for d in data_dirs:
        if not os.path.isdir(d):
            print(f"  Skipping {d} (not found)")
            continue
        picks = load_picks_for_dir(d)
        for date_str, day_picks in picks.items():
            if date_str not in merged:
                merged[date_str] = day_picks
            else:
                existing_tickers = {p["ticker"] for p in merged[date_str]}
                for p in day_picks:
                    if p["ticker"] not in existing_tickers:
                        merged[date_str].append(p)
                merged[date_str].sort(key=lambda x: x["gap_pct"], reverse=True)
                merged[date_str] = merged[date_str][:TOP_N]
    all_dates = sorted(merged.keys())
    return all_dates, merged


def simulate_day(picks, starting_cash, cash_account=False):
    """Simulate trades for one day with two-pass structure.
    Pass 1: Process exits (frees cash), execute scale-ins inline, detect entry signals.
    Pass 2: Allocate capital to new entries (ranked by liquidity).

    cash_account=True:  T+1 settlement — sell proceeds available next day only
    cash_account=False: margin — sell proceeds available instantly for new entries
    """
    cash = starting_cash
    unsettled = 0.0  # T+1: proceeds from today's sells, available tomorrow
    trade_pct = _get_trade_pct(starting_cash)
    trade_size_base = starting_cash * trade_pct
    trades_taken = 0

    def _receive_proceeds(amount):
        """Route sell proceeds based on account type."""
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
        states.append(
            {
                "ticker": pick["ticker"],
                "premarket_high": pick["premarket_high"],
                "gap_pct": pick["gap_pct"],
                "pm_volume": pick.get("pm_volume", 0),
                "mh": pick["market_hour_candles"],
                "entry_price": None,
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
                "original_entry_price": None,
                "done": False,
                "recent_closes": [],
                "breakout_confirmed": False,
                "pullback_detected": False,
                "candles_since_confirm": 0,
                "true_ranges": [],
                "prev_candle_close": None,
                "candle_volumes": [],
                "runner_mode": False,
                # Scale-in tracking
                "scaled_in": False,
                "original_position_size": 0.0,
                "scale_in_time": None,
                "scale_in_price": None,
                "signal_close_price": None,
                # Breakeven stop tracking
                "breakeven_active": False,
                # Cash wait tracking
                "waiting_for_cash": False,
                "cash_wait_entry": False,
                "vol_capped": False,
            }
        )

    for ts in all_timestamps:
        # ── PASS 1: Exits + signal detection ──
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
            c_vol = float(candle.get("Volume", 0))

            # ATR accumulation (always, all stocks)
            if st["prev_candle_close"] is not None:
                tr = max(
                    c_high - c_low,
                    abs(c_high - st["prev_candle_close"]),
                    abs(c_low - st["prev_candle_close"]),
                )
            else:
                tr = c_high - c_low
            st["true_ranges"].append(tr)
            if len(st["true_ranges"]) > ATR_PERIOD:
                st["true_ranges"] = st["true_ranges"][-ATR_PERIOD:]
            st["prev_candle_close"] = c_close

            # Track candle volumes for runner detection
            st["candle_volumes"].append(c_vol)
            if len(st["candle_volumes"]) > 20:
                st["candle_volumes"] = st["candle_volumes"][-20:]

            pm_high = st["premarket_high"]

            # === OPEN POSITION: process exits ===
            if st["entry_price"] is not None:
                if c_high > st["highest_since_entry"]:
                    st["highest_since_entry"] = c_high

                # Breakeven stop: once stock hits +X%, move stop to entry price
                if (
                    BREAKEVEN_STOP_PCT > 0
                    and not st.get("breakeven_active")
                    and st["remaining_shares"] > 0
                    and st["original_entry_price"] is not None
                ):
                    be_trigger = st["original_entry_price"] * (1 + BREAKEVEN_STOP_PCT / 100)
                    if c_high >= be_trigger:
                        st["breakeven_active"] = True

                # Scale-in: execute INLINE before partial sells
                # so +14% scale-in fires before +15% partial sell on same candle
                if (
                    SCALE_IN
                    and not st["scaled_in"]
                    and st["remaining_shares"] > 0
                    and (not SCALE_IN_GATE_PARTIAL or not st["partial_sold"])
                ):
                    trigger_price = st["entry_price"] * (1 + SCALE_IN_TRIGGER_PCT / 100)
                    if c_high >= trigger_price:
                        add_size = st["original_position_size"] * SCALE_IN_FRAC
                        # Volume cap on scale-in
                        if VOL_CAP_PCT > 0:
                            pre_si = st["mh"].loc[st["mh"].index <= ts]
                            si_vol = pre_si["Volume"].sum() if len(pre_si) > 0 else 0
                            si_dvol = trigger_price * si_vol
                            si_limit = si_dvol * (VOL_CAP_PCT / 100) - st["position_cost"]
                            if si_limit > 0 and add_size > si_limit:
                                add_size = si_limit
                            if add_size < 50:
                                add_size = 0
                        if cash >= add_size and add_size > 0:
                            add_price = trigger_price * (1 + SLIPPAGE_PCT / 100)
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
                        st["scaled_in"] = True

                # Stop loss (use original entry for pre-partial-sell so scale-in doesn't tighten stop)
                if st["remaining_shares"] > 0 and not st["trailing_active"]:
                    if STRUCTURAL_STOP and st["true_ranges"]:
                        atr_val = sum(st["true_ranges"]) / len(st["true_ranges"])
                        struct_stop = st["premarket_high"] - STRUCTURAL_STOP_ATR_MULT * atr_val
                        base_price = st["original_entry_price"] if not st["partial_sold"] else st["entry_price"]
                        pct_stop = base_price * (1 - STOP_LOSS_PCT / 100)
                        stop_price = max(struct_stop, pct_stop)  # use tighter of the two
                    else:
                        base_price = st["original_entry_price"] if not st["partial_sold"] else st["entry_price"]
                        stop_price = base_price * (1 - STOP_LOSS_PCT / 100)
                    # Breakeven override: if activated, stop is at entry price (0% loss)
                    if st.get("breakeven_active") and st["original_entry_price"] is not None:
                        stop_price = st["original_entry_price"]
                    if c_low <= stop_price:
                        sell_price = stop_price * (1 - SLIPPAGE_PCT / 100)
                        proceeds = st["remaining_shares"] * sell_price
                        st["total_exit_value"] += proceeds
                        st["remaining_shares"] = 0
                        st["exit_reason"] = "STOP_LOSS"
                        st["exit_price"] = stop_price
                        st["exit_time"] = ts
                        st["done"] = True
                        _receive_proceeds(proceeds)
                        continue

                # Trailing stop
                if st["remaining_shares"] > 0 and st["trailing_active"]:
                    atr = (
                        sum(st["true_ranges"]) / len(st["true_ranges"])
                        if st["true_ranges"]
                        else 0
                    )
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
                        _receive_proceeds(proceeds)
                        continue

                # Runner mode detection: skip partial sells, trail only
                if RUNNER_MODE and st["entry_price"] is not None and not st["runner_mode"]:
                    gain_pct = (c_close / st["entry_price"] - 1) * 100
                    if gain_pct >= RUNNER_TRIGGER_PCT:
                        st["runner_mode"] = True
                        st["trailing_active"] = True

                # Multi-tranche exits (frees cash for Pass 2)
                if st.get("runner_mode"):
                    pass  # skip partial sells, trail stop handles exit
                elif N_EXIT_TRANCHES == 1:
                    if st["remaining_shares"] > 0 and not st["partial_sold"]:
                        target_price = st["original_entry_price"] * (1 + PARTIAL_SELL_PCT / 100)
                        if c_high >= target_price:
                            sell_price = target_price * (1 - SLIPPAGE_PCT / 100)
                            proceeds = st["remaining_shares"] * sell_price
                            st["total_exit_value"] += proceeds
                            st["remaining_shares"] = 0
                            st["partial_sold"] = True
                            st["done"] = True
                            st["exit_reason"] = "TARGET"
                            st["exit_price"] = target_price
                            st["exit_time"] = ts
                            st["partial_sell_time"] = ts
                            st["partial_sell_price"] = target_price
                            _receive_proceeds(proceeds)

                elif N_EXIT_TRANCHES == 2:
                    if st["remaining_shares"] > 0 and not st["partial_sold"]:
                        target_price = st["original_entry_price"] * (1 + PARTIAL_SELL_PCT / 100)
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
                            _receive_proceeds(proceeds)
                            if RESET_STOPS_ON_PARTIAL and st["remaining_shares"] > 0:
                                st["entry_price"] = target_price
                                st["position_cost"] = st["remaining_shares"] * target_price
                                st["highest_since_entry"] = c_high

                elif N_EXIT_TRANCHES == 3:
                    if st["remaining_shares"] > 0 and not st["partial_sold"]:
                        target_price = st["original_entry_price"] * (1 + PARTIAL_SELL_PCT / 100)
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
                            _receive_proceeds(proceeds)
                            if RESET_STOPS_ON_PARTIAL and st["remaining_shares"] > 0:
                                st["entry_price"] = target_price
                                st["position_cost"] = st["remaining_shares"] * target_price
                                st["highest_since_entry"] = c_high

                    if (
                        st["remaining_shares"] > 0
                        and st["partial_sold"]
                        and not st["partial_sold_2"]
                    ):
                        target_price_2 = st["entry_price"] * (
                            1 + PARTIAL_SELL_PCT_2 / 100
                        )
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
                            st["partial_sell_time_2"] = ts
                            st["partial_sell_price_2"] = target_price_2
                            _receive_proceeds(proceeds)
                            if RESET_STOPS_ON_PARTIAL and st["remaining_shares"] > 0:
                                st["entry_price"] = target_price_2
                                st["position_cost"] = st["remaining_shares"] * target_price_2
                                st["highest_since_entry"] = c_high

                # Early EOD exit: sell remaining shares if within exit window
                if EOD_EXIT_MINUTES > 0 and st["remaining_shares"] > 0:
                    ts_et = ts.astimezone(ET_TZ) if ts.tzinfo else ts
                    minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
                    if minutes_to_close <= EOD_EXIT_MINUTES:
                        sell_price = c_close * (1 - SLIPPAGE_PCT / 100)
                        proceeds = st["remaining_shares"] * sell_price
                        st["total_exit_value"] += proceeds
                        st["remaining_shares"] = 0
                        if st["exit_reason"] not in ("STOP_LOSS", "TRAIL_STOP", "TARGET"):
                            if st["partial_sold"]:
                                st["exit_reason"] = "PARTIAL+EOD"
                            else:
                                st["exit_reason"] = "EOD_EARLY"
                        st["exit_price"] = c_close
                        st["exit_time"] = ts
                        st["done"] = True
                        _receive_proceeds(proceeds)
                        continue

                if st["remaining_shares"] <= 0:
                    st["done"] = True
                continue  # done with this open-position stock for Pass 1

            # === NO POSITION: check entry signals ===

            # 3-phase entry detection
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

            # Liquidity vacuum filter: require large candle + volume spike
            if entered and LIQ_VACUUM:
                candle_range = c_high - c_low
                trs = st["true_ranges"]
                vols = st["candle_volumes"]
                atr14 = sum(trs) / len(trs) if len(trs) >= 3 else candle_range
                avg_vol = sum(vols[:-1]) / len(vols[:-1]) if len(vols) > 1 else (c_vol if c_vol > 0 else 1)
                if avg_vol > 0 and atr14 > 0:
                    if candle_range < LIQ_CANDLE_ATR_MULT * atr14 or c_vol < LIQ_VOL_MULT * avg_vol:
                        entered = False

            if entered:
                st["signal_close_price"] = c_close
                entry_candidates.append(st)

            # Cash-wait: re-add waiting stocks if setup still valid
            elif st["waiting_for_cash"] and CASH_WAIT_ENABLED:
                if c_close > pm_high:
                    st["signal_close_price"] = c_close  # update to current price
                    entry_candidates.append(st)

        # ── PASS 2: Allocate capital to new entries ──
        # (Scale-ins already executed inline in Pass 1 before partial sells)
        # Sort by PM dollar volume desc (best liquidity first)
        entry_candidates.sort(key=lambda st: -(st.get("pm_volume", 0) * st.get("premarket_high", 1)))

        for st in entry_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue

            # Entry time cutoff: skip if too close to market close
            if ENTRY_CUTOFF_MINUTES > 0:
                ts_et = ts.astimezone(ET_TZ) if ts.tzinfo else ts
                mins_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
                if mins_to_close <= ENTRY_CUTOFF_MINUTES:
                    st["exit_reason"] = "LATE_ENTRY"
                    st["done"] = True
                    continue

            trades_taken += 1
            position_size = trade_size_base

            if cash < position_size:
                if cash > 0:
                    position_size = cash
                else:
                    trades_taken -= 1
                    if CASH_WAIT_ENABLED:
                        st["waiting_for_cash"] = True
                    else:
                        st["exit_reason"] = "NO_CASH"
                        st["done"] = True
                    continue

            fill_price = st["signal_close_price"]

            # Volume cap: limit position to VOL_CAP_PCT% of dollar volume at buy
            # dollar_vol = current price * total shares traded up to now
            if VOL_CAP_PCT > 0:
                pre_entry = st["mh"].loc[st["mh"].index <= ts]
                vol_shares = pre_entry["Volume"].sum() if len(pre_entry) > 0 else 0
                dollar_vol = fill_price * vol_shares
                vol_limit = dollar_vol * (VOL_CAP_PCT / 100)
                if vol_limit > 0 and position_size > vol_limit:
                    position_size = vol_limit
                    st["vol_capped"] = True
                if position_size < 50:  # skip if too small to trade
                    trades_taken -= 1
                    st["exit_reason"] = "LOW_VOL"
                    st["done"] = True
                    continue

            st["entry_price"] = fill_price * (1 + SLIPPAGE_PCT / 100)
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
            # Track if this was a cash-wait entry
            if st["waiting_for_cash"]:
                st["cash_wait_entry"] = True
                st["waiting_for_cash"] = False

    # EOD: mark stocks still waiting for cash as NO_CASH
    for st in states:
        if st["waiting_for_cash"]:
            st["exit_reason"] = "NO_CASH"
            st["done"] = True
            st["waiting_for_cash"] = False

    # EOD: close remaining positions
    for st in states:
        if st["entry_price"] is not None and st["remaining_shares"] > 0:
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["remaining_shares"] * sell_price
            st["total_exit_value"] += proceeds
            st["remaining_shares"] = 0
            if st["exit_reason"] not in ("STOP_LOSS", "TRAIL_STOP", "TARGET"):
                if st["partial_sold"]:
                    st["exit_reason"] = "PARTIAL+EOD"
                else:
                    st["exit_reason"] = "EOD_CLOSE"
            st["exit_price"] = last_close
            st["exit_time"] = st["mh"].index[-1]
            _receive_proceeds(proceeds)

    # Settle unsettled cash (available next day)
    cash += unsettled

    # Compute P&L using total cash spent (original cost + scale-in cost)
    for st in states:
        if st["entry_price"] is not None:
            cost = st["total_cash_spent"]
            st["pnl"] = st["total_exit_value"] - cost
            st["pnl_pct"] = (st["pnl"] / cost) * 100 if cost > 0 else 0
        else:
            st["pnl"] = 0
            st["pnl_pct"] = 0

    return states, cash


def _to_et(ts):
    """Convert a timestamp to Eastern Time."""
    if ts is None:
        return None
    return ts.tz_convert(ET_TZ) if ts.tzinfo else ts


def plot_day(date_str, picks, states, day_idx, axes_row):
    """Plot traded stocks for one day."""
    traded = [s for s in states if s.get("original_entry_price") is not None or s["entry_price"] is not None]
    ax_price = axes_row[0]
    ax_pnl = axes_row[1]

    if traded:
        colors_cycle = [
            "#2196F3",
            "#FF9800",
            "#4CAF50",
            "#f44336",
            "#9C27B0",
            "#00BCD4",
            "#795548",
            "#607D8B",
            "#E91E63",
            "#CDDC39",
        ]
        for i, st in enumerate(traded):
            mh = st["mh"]
            if mh.index.tz is not None:
                et_times = mh.index.tz_convert(ET_TZ)
            else:
                et_times = mh.index.tz_localize("UTC").tz_convert(ET_TZ)

            color = colors_cycle[i % len(colors_cycle)]
            si = " S+" if st.get("scaled_in") else ""
            vc = " VC" if st.get("vol_capped") else ""
            pos_str = f"${st['position_cost']:,.0f}" if st["entry_price"] else ""
            label = f"{st['ticker']} (gap {st['gap_pct']:.0f}%{si}{vc}) {pos_str}"
            # Use original_entry_price for consistent % axis (unaffected by reset_stops)
            chart_entry_p = st.get("original_entry_price") or st["entry_price"]
            pct_change = (mh["Close"].values.astype(float) / chart_entry_p - 1) * 100
            ax_price.plot(
                et_times,
                pct_change,
                color=color,
                linewidth=1.2,
                label=label,
                alpha=0.85,
            )

            # --- BUY marker ---
            if st["entry_time"] is not None:
                et_entry = _to_et(st["entry_time"])
                ax_price.axvline(
                    x=et_entry, color=color, linestyle="--", alpha=0.3, linewidth=0.8
                )
                ax_price.plot(
                    et_entry, 0, marker="^", color=color, markersize=10, zorder=5
                )
                buy_label = "BUY"
                if st.get("cash_wait_entry"):
                    buy_label = "BUY(W)"
                ax_price.annotate(
                    buy_label, xy=(et_entry, 0), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8, lw=0.5),
                )

            # --- Scale-in marker ---
            if st.get("scale_in_time") is not None:
                et_si = _to_et(st["scale_in_time"])
                si_pct = (st["scale_in_price"] / chart_entry_p - 1) * 100
                ax_price.plot(
                    et_si, si_pct, marker="P", color=color, markersize=9, zorder=5,
                    markeredgecolor="white", markeredgewidth=1,
                )
                ax_price.annotate(
                    "S+", xy=(et_si, si_pct), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8, lw=0.5),
                )

            # --- SELL 1 marker (first partial sell) ---
            if st["partial_sell_time"] is not None:
                et_ps = _to_et(st["partial_sell_time"])
                ps_pct = (st["partial_sell_price"] / chart_entry_p - 1) * 100
                ax_price.plot(
                    et_ps, ps_pct, marker="D", color=color, markersize=8, zorder=5,
                    markeredgecolor="white", markeredgewidth=1,
                )
                sell1_label = "SELL 1" if N_EXIT_TRANCHES > 1 else "SELL"
                ax_price.annotate(
                    sell1_label, xy=(et_ps, ps_pct), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=6, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8, lw=0.5),
                )

            # --- SELL 2 marker (second partial sell, 3-tranche only) ---
            if st.get("partial_sell_time_2") is not None:
                et_ps2 = _to_et(st["partial_sell_time_2"])
                ps2_pct = (st["partial_sell_price_2"] / chart_entry_p - 1) * 100
                ax_price.plot(
                    et_ps2, ps2_pct, marker="D", color=color, markersize=8, zorder=5,
                    markeredgecolor="white", markeredgewidth=1,
                )
                ax_price.annotate(
                    "SELL 2", xy=(et_ps2, ps2_pct), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=6, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8, lw=0.5),
                )

            # --- EXIT marker (final exit: stop loss, trailing stop, EOD, target) ---
            if st["exit_time"] is not None and st["exit_price"] is not None:
                et_exit = _to_et(st["exit_time"])
                exit_pct = (st["exit_price"] / chart_entry_p - 1) * 100
                marker = "v" if st["pnl"] < 0 else "s"
                ax_price.plot(
                    et_exit, exit_pct, marker=marker, color=color, markersize=10, zorder=5,
                    markeredgecolor="white", markeredgewidth=1,
                )
                # Label based on exit reason
                reason = st["exit_reason"]
                exit_labels = {
                    "STOP_LOSS": "STOP",
                    "TRAIL_STOP": "TRAIL",
                    "EOD_CLOSE": "EOD",
                    "EOD_EARLY": "EOD*",
                    "PARTIAL+EOD": "EOD",
                    "TARGET": "TARGET",
                }
                exit_label = exit_labels.get(reason, reason[:6])
                ax_price.annotate(
                    exit_label, xy=(et_exit, exit_pct), xytext=(0, -14 if exit_pct >= 0 else 12),
                    textcoords="offset points", ha="center",
                    va="top" if exit_pct >= 0 else "bottom",
                    fontsize=6, fontweight="bold",
                    color="#f44336" if st["pnl"] < 0 else "#4CAF50",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#f44336" if st["pnl"] < 0 else "#4CAF50", alpha=0.8, lw=0.5),
                )

        ax_price.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
        ax_price.axhline(
            y=-STOP_LOSS_PCT,
            color="#f44336",
            linestyle=":",
            alpha=0.4,
            label=f"Stop (-{STOP_LOSS_PCT}%)",
        )
        ax_price.axhline(
            y=PARTIAL_SELL_PCT,
            color="#4CAF50",
            linestyle=":",
            alpha=0.4,
            label=f"+{PARTIAL_SELL_PCT}%",
        )

        ax_price.set_title(
            f"{date_str} - Price Action (% from entry)", fontsize=10, fontweight="bold"
        )
        ax_price.set_ylabel("% from Entry", fontsize=8)
        ax_price.legend(fontsize=6, loc="upper left", ncol=2)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p", tz=ET_TZ))
        ax_price.tick_params(axis="x", rotation=30, labelsize=7)
        ax_price.grid(alpha=0.3)
    else:
        ax_price.text(
            0.5,
            0.5,
            f"{date_str}\nNo trades triggered",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax_price.transAxes,
            color="gray",
        )
        ax_price.set_title(f"{date_str} - No Trades", fontsize=10)

    # P&L bar chart
    all_with_trades = [s for s in states if (s.get("original_entry_price") or s["entry_price"]) is not None]
    if all_with_trades:
        tickers = [s["ticker"] for s in all_with_trades]
        pnls = [s["pnl"] for s in all_with_trades]
        bar_colors = ["#4CAF50" if p > 0 else "#f44336" for p in pnls]
        reasons = []
        for s in all_with_trades:
            r = s["exit_reason"]
            if s.get("cash_wait_entry"):
                r += " W"
            if s.get("vol_capped"):
                r += f" VC${s['position_cost']:,.0f}"
            reasons.append(r)

        y_pos = range(len(tickers))
        bars = ax_pnl.barh(y_pos, pnls, color=bar_colors, edgecolor="white", height=0.6)
        ax_pnl.set_yticks(y_pos)
        ax_pnl.set_yticklabels(
            [f"{t} ({r})" for t, r in zip(tickers, reasons)], fontsize=7
        )
        ax_pnl.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
        ax_pnl.invert_yaxis()

        for j, (bar, pnl) in enumerate(zip(bars, pnls)):
            x_pos = bar.get_width()
            align = "left" if pnl >= 0 else "right"
            offset = 5 if pnl >= 0 else -5
            ax_pnl.annotate(
                f"${pnl:+.0f}",
                xy=(x_pos, j),
                xytext=(offset, 0),
                textcoords="offset points",
                ha=align,
                va="center",
                fontsize=7,
                fontweight="bold",
                color=bar_colors[j],
            )

        day_total = sum(pnls)
        total_color = "#4CAF50" if day_total >= 0 else "#f44336"
        ax_pnl.set_title(
            f"P&L | Day: ${day_total:+,.0f}",
            fontsize=10,
            fontweight="bold",
            color=total_color,
        )
        ax_pnl.set_xlabel("P&L ($)", fontsize=8)
        ax_pnl.grid(alpha=0.3, axis="x")
    else:
        no_bo_count = len([s for s in states if s["exit_reason"] == "NO_BREAKOUT"])
        ax_pnl.text(
            0.5,
            0.5,
            f"0 trades\n{no_bo_count} watched",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax_pnl.transAxes,
            color="gray",
        )
        ax_pnl.set_title("P&L: $0", fontsize=10, color="gray")


def _run_sizing_sweep(data_dirs):
    """Sweep adaptive sizing tier configurations to find optimal breakpoints."""
    from itertools import product
    from regime_filters import RegimeFilter

    ALL_DATES, daily_picks = load_all_picks(data_dirs)
    regime = RegimeFilter(
        spy_ma_period=SPY_SMA_PERIOD,
        enable_vix=False,
        enable_spy_trend=True,
        enable_adaptive=False,
    )
    regime.load_data(ALL_DATES[0], ALL_DATES[-1])

    # Build list of days to simulate (pre-filter regime)
    tradeable_days = []
    for date_str in ALL_DATES:
        should_trade, _, _ = regime.check(date_str)
        if should_trade:
            tradeable_days.append(date_str)

    print(f"\nSizing Sweep: {ALL_DATES[0]} to {ALL_DATES[-1]}")
    print(f"  {len(ALL_DATES)} total days, {len(tradeable_days)} tradeable\n")

    # Tier configs to test: (pct_under_25K, pct_25K_50K, pct_50K_100K, pct_above_100K)
    pct_1_options = [0.20, 0.25, 0.30]  # cash account phase
    pct_2_options = [0.25, 0.30, 0.33, 0.40]  # $25K-$50K
    pct_3_options = [0.33, 0.40, 0.45, 0.50]  # $50K-$100K
    pct_4_options = [0.40, 0.45, 0.50]  # $100K+

    # Also test fixed sizing as baselines
    fixed_options = [0.25, 0.30, 0.33, 0.40, 0.50]

    results = []

    def _simulate_with_tiers(tiers, label):
        """Run full rolling backtest with given sizing tiers."""
        rolling = STARTING_CASH
        crossed = None
        daily_pnls = []
        for date_str in ALL_DATES:
            should_trade, _, _ = regime.check(date_str)
            if not should_trade:
                daily_pnls.append(0.0)
                continue
            is_cash = rolling < MARGIN_THRESHOLD
            # Compute trade_pct from tiers
            pct = tiers[0][1]
            for thresh, tier_pct in tiers:
                if rolling >= thresh:
                    pct = tier_pct
            # Temporarily override the global for simulate_day
            picks = daily_picks[date_str]
            # Inline: set trade_size_base via the tier
            old_adaptive = globals().get("ADAPTIVE_SIZING")
            globals()["ADAPTIVE_SIZING"] = False  # disable global adaptive
            old_pct = globals().get("TRADE_PCT")
            globals()["TRADE_PCT"] = pct
            states, ending = simulate_day(picks, rolling, cash_account=is_cash)
            globals()["TRADE_PCT"] = old_pct
            globals()["ADAPTIVE_SIZING"] = old_adaptive
            day_pnl = sum(s["pnl"] for s in states if s["entry_price"] is not None)
            daily_pnls.append(day_pnl)
            rolling = ending
            if crossed is None and rolling >= MARGIN_THRESHOLD:
                crossed = date_str

        arr = np.array(daily_pnls)
        total = arr.sum()
        mean_d = np.mean(arr)
        std_d = np.std(arr) if len(arr) > 1 else 1.0
        sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 0 else 0
        # Max drawdown from peak
        cumulative = np.cumsum(arr)
        peak = np.maximum.accumulate(cumulative + STARTING_CASH)
        dd = (peak - (cumulative + STARTING_CASH)) / peak * 100
        max_dd = dd.max()
        no_cash_count = 0  # can't easily count from here
        return {
            "label": label,
            "tiers": tiers,
            "final": rolling,
            "total_pnl": total,
            "sharpe": sharpe,
            "max_dd_pct": max_dd,
            "margin_date": crossed,
        }

    # Test fixed sizing baselines
    print("  Testing fixed sizing baselines...")
    for pct in fixed_options:
        tiers = [(0, pct)]
        label = f"Fixed {int(pct * 100)}%"
        r = _simulate_with_tiers(tiers, label)
        results.append(r)
        print(
            f"    {label:<30} Final: ${r['final']:>12,.0f}  Sharpe: {r['sharpe']:>5.2f}  MaxDD: {r['max_dd_pct']:>5.1f}%  Margin: {r['margin_date'] or 'never'}"
        )

    # Test adaptive tier combinations
    print("\n  Testing adaptive tier combinations...")
    combo_count = 0
    for p1, p2, p3, p4 in product(
        pct_1_options, pct_2_options, pct_3_options, pct_4_options
    ):
        # Enforce monotonically increasing
        if not (p1 <= p2 <= p3 <= p4):
            continue
        tiers = [(0, p1), (25_000, p2), (50_000, p3), (100_000, p4)]
        label = f"{int(p1 * 100)}/{int(p2 * 100)}/{int(p3 * 100)}/{int(p4 * 100)}%"
        r = _simulate_with_tiers(tiers, label)
        results.append(r)
        combo_count += 1
        if combo_count % 10 == 0:
            sys.stdout.write(f"\r    Tested {combo_count} combinations...")
            sys.stdout.flush()

    print(f"\r    Tested {combo_count} adaptive combinations.          ")

    # Also test different breakpoint levels with best base percentages
    print("\n  Testing alternate breakpoints...")
    for mid_break in [35_000, 50_000, 75_000]:
        for hi_break in [75_000, 100_000, 150_000]:
            if mid_break >= hi_break:
                continue
            tiers = [(0, 0.25), (25_000, 0.33), (mid_break, 0.40), (hi_break, 0.50)]
            label = f"25/33/40/50% @0/{25}K/{mid_break // 1000}K/{hi_break // 1000}K"
            r = _simulate_with_tiers(tiers, label)
            results.append(r)

    # Sort by final balance (primary), then Sharpe
    results.sort(key=lambda x: (-x["final"], -x["sharpe"]))

    print(f"\n{'=' * 95}")
    print(f"  SIZING SWEEP RESULTS — Top 20 by Final Balance")
    print(f"{'=' * 95}")
    print(
        f"  {'Rank':<5}{'Configuration':<40}{'Final':>14}{'Sharpe':>8}{'MaxDD%':>8}{'Margin Unlock':>16}"
    )
    print(f"  {'-' * 5}{'-' * 40}{'-' * 14}{'-' * 8}{'-' * 8}{'-' * 16}")
    for i, r in enumerate(results[:20]):
        print(
            f"  {i + 1:<5}{r['label']:<40}${r['final']:>12,.0f}{r['sharpe']:>8.2f}{r['max_dd_pct']:>7.1f}%  {r['margin_date'] or 'never':>14}"
        )

    # Best by Sharpe
    by_sharpe = sorted(results, key=lambda x: -x["sharpe"])
    print(f"\n  {'=' * 60}")
    print(f"  Top 5 by Sharpe:")
    for i, r in enumerate(by_sharpe[:5]):
        print(
            f"    {i + 1}. {r['label']:<40} Sharpe: {r['sharpe']:.2f}  Final: ${r['final']:,.0f}"
        )

    # Best overall (Sharpe * log(final))
    for r in results:
        r["score"] = r["sharpe"] * np.log(max(r["final"], 1))
    by_score = sorted(results, key=lambda x: -x["score"])
    print(f"\n  Top 5 by Combined Score (Sharpe * ln(Final)):")
    for i, r in enumerate(by_score[:5]):
        print(
            f"    {i + 1}. {r['label']:<40} Score: {r['score']:.1f}  Sharpe: {r['sharpe']:.2f}  Final: ${r['final']:,.0f}"
        )

    print(f"\n{'=' * 95}")
    return results


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check for flags
    sweep_mode = "--sweep-sizing" in sys.argv
    no_charts = "--no-charts" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    # Determine data directories
    if args:
        data_dirs = args
    else:
        data_dirs = DEFAULT_DATA_DIRS

    if sweep_mode:
        print(f"Data directories: {data_dirs}")
        _run_sizing_sweep(data_dirs)
        sys.exit(0)

    print(f"Data directories: {data_dirs}")
    ALL_DATES, daily_picks = load_all_picks(data_dirs)

    # Load SPY regime filter
    from regime_filters import RegimeFilter

    regime = RegimeFilter(
        spy_ma_period=SPY_SMA_PERIOD,
        enable_vix=False,
        enable_spy_trend=True,
        enable_adaptive=False,
    )
    regime.load_data(ALL_DATES[0], ALL_DATES[-1])

    print(
        f"\nFull Backtest: {ALL_DATES[0]} to {ALL_DATES[-1]} ({len(ALL_DATES)} trading days)"
    )
    if ADAPTIVE_SIZING:
        tier_str = " -> ".join(
            f"{int(p * 100)}%@${t / 1000:.0f}K" for t, p in SIZING_TIERS
        )
        print(
            f"Config: ${STARTING_CASH:,} starting capital (rolling) | Adaptive sizing: {tier_str} | Stop: {STOP_LOSS_PCT}%"
        )
    else:
        print(
            f"Config: ${STARTING_CASH:,} starting capital (rolling) | {TRADE_PCT * 100:.0f}%/trade | Stop: {STOP_LOSS_PCT}%"
        )
    if N_EXIT_TRANCHES == 3:
        print(
            f"        T1: Sell {int(PARTIAL_SELL_FRAC * 100)}% @ +{PARTIAL_SELL_PCT}% | T2: Sell {int(PARTIAL_SELL_FRAC_2 * 100)}% @ +{PARTIAL_SELL_PCT_2}% | Trail rest: ATR({ATR_PERIOD})x{ATR_MULTIPLIER}"
        )
    else:
        print(
            f"        Sell {int(PARTIAL_SELL_FRAC * 100)}% @ +{PARTIAL_SELL_PCT}% | Trail: ATR({ATR_PERIOD})x{ATR_MULTIPLIER}"
        )
    print(
        f"        Confirm: {CONFIRM_ABOVE}/{CONFIRM_WINDOW} | Pullback: {PULLBACK_PCT}% (timeout: {PULLBACK_TIMEOUT}) | Vol: {MIN_PM_VOLUME:,}"
    )
    si_str = (
        f"Scale-in: +{SCALE_IN_TRIGGER_PCT}% (50% add)" if SCALE_IN else "Scale-in: off"
    )
    print(
        f"        {si_str} | Regime: SPY > SMA({SPY_SMA_PERIOD}) | Cash acct < ${MARGIN_THRESHOLD / 1000:.0f}K (T+1)\n"
    )

    all_results = []
    rolling_cash = STARTING_CASH
    regime_skipped = 0
    crossed_margin_date = None  # track first time we hit margin threshold

    for idx, date_str in enumerate(ALL_DATES):
        sys.stdout.write(
            f"\r  [{idx + 1}/{len(ALL_DATES)}] Simulating {date_str} (cash: ${rolling_cash:,.2f})..."
        )
        sys.stdout.flush()

        # Regime filter: skip day if SPY < SMA
        should_trade, _, regime_info = regime.check(date_str)
        if not should_trade:
            regime_skipped += 1
            all_results.append(
                {
                    "date": date_str,
                    "picks": daily_picks[date_str],
                    "states": [],
                    "starting_cash": rolling_cash,
                    "ending_cash": rolling_cash,
                    "traded": 0,
                    "day_pnl": 0.0,
                    "regime_skip": True,
                }
            )
            continue

        # Cash account (T+1) until we reach margin threshold
        is_cash_account = rolling_cash < MARGIN_THRESHOLD
        day_trade_pct = _get_trade_pct(rolling_cash)

        picks = daily_picks[date_str]
        states, ending_cash = simulate_day(
            picks, rolling_cash, cash_account=is_cash_account
        )
        traded_count = sum(1 for s in states if s["entry_price"] is not None)
        day_pnl = sum(s["pnl"] for s in states if s["entry_price"] is not None)
        rolling_cash = ending_cash

        # Track first margin threshold crossover
        if crossed_margin_date is None and rolling_cash >= MARGIN_THRESHOLD:
            crossed_margin_date = date_str

        all_results.append(
            {
                "date": date_str,
                "picks": picks,
                "states": states,
                "starting_cash": rolling_cash - day_pnl,
                "ending_cash": ending_cash,
                "traded": traded_count,
                "day_pnl": day_pnl,
                "regime_skip": False,
                "cash_account": is_cash_account,
                "trade_pct": day_trade_pct,
            }
        )

    print(
        f"\r  Processed {len(ALL_DATES)} days ({regime_skipped} skipped by regime filter).     "
    )

    # Compute stats used by both charts and print summary
    total_trades = 0
    total_winners = 0
    total_losers = 0
    all_exits = {}
    scale_in_count = 0
    cash_wait_count = 0
    vol_capped_count = 0
    low_vol_count = 0
    green_days = 0
    best_day = 0.0
    worst_day = 0.0
    cash_acct_days = 0
    for res in all_results:
        if res.get("regime_skip"):
            continue
        if res.get("cash_account"):
            cash_acct_days += 1
        dp = res["day_pnl"]
        if dp > 0:
            green_days += 1
        if dp > best_day:
            best_day = dp
        if dp < worst_day:
            worst_day = dp
        for st in res["states"]:
            if st["entry_price"] is not None:
                total_trades += 1
                if st["pnl"] > 0:
                    total_winners += 1
                else:
                    total_losers += 1
                reason = st["exit_reason"]
                all_exits[reason] = all_exits.get(reason, 0) + 1
                if st.get("scaled_in") and st.get("scale_in_time"):
                    scale_in_count += 1
                if st.get("cash_wait_entry"):
                    cash_wait_count += 1
                if st.get("vol_capped"):
                    vol_capped_count += 1
            elif st.get("exit_reason") == "LOW_VOL":
                low_vol_count += 1
    win_rate = (total_winners / total_trades * 100) if total_trades > 0 else 0
    margin_str = crossed_margin_date if crossed_margin_date else "never"
    total_pnl = sum(r["day_pnl"] for r in all_results)

    # --- GENERATE PAGINATED CHARTS (versioned) ---
    run_dir = None
    stress_results = None
    if not no_charts:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("charts", f"run_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\nGenerating charts -> {run_dir}/")

    if no_charts:
        print("\n  Skipping charts and stress tests (--no-charts)")

    if not no_charts:
        num_pages = math.ceil(len(ALL_DATES) / DAYS_PER_PAGE)
        chart_files = []

        for page in range(num_pages):
            start = page * DAYS_PER_PAGE
            end = min(start + DAYS_PER_PAGE, len(ALL_DATES))
            page_dates = ALL_DATES[start:end]
            n_rows = len(page_dates)

            fig, axes = plt.subplots(
                n_rows,
                2,
                figsize=(20, 4.5 * n_rows),
                gridspec_kw={"width_ratios": [2.5, 1]},
            )
            if n_rows == 1:
                axes = [axes]

            exit_str = (
                f"T1:{int(PARTIAL_SELL_FRAC * 100)}%@+{PARTIAL_SELL_PCT}% T2:{int(PARTIAL_SELL_FRAC_2 * 100)}%@+{PARTIAL_SELL_PCT_2}%"
                if N_EXIT_TRANCHES == 3
                else f"Sell {int(PARTIAL_SELL_FRAC * 100)}%@+{PARTIAL_SELL_PCT}%"
            )
            fig.suptitle(
                f"Backtest Page {page + 1}/{num_pages}: {page_dates[0]} to {page_dates[-1]}\n"
                f"Stop: {STOP_LOSS_PCT}% | {exit_str} | Trail: ATR({ATR_PERIOD})x{ATR_MULTIPLIER} | "
                f"PB: {PULLBACK_PCT}%/t{PULLBACK_TIMEOUT} | Vol: {MIN_PM_VOLUME:,} | Scale: +{SCALE_IN_TRIGGER_PCT}%",
                fontsize=12,
                fontweight="bold",
                y=1.01,
            )

            for i, date_str in enumerate(page_dates):
                res = all_results[start + i]
                row_axes = axes[i] if n_rows > 1 else axes[0]
                if res.get("regime_skip"):
                    row_axes[0].text(
                        0.5,
                        0.5,
                        f"{res['date']}\nSKIPPED (SPY < SMA{SPY_SMA_PERIOD})",
                        ha="center",
                        va="center",
                        fontsize=12,
                        transform=row_axes[0].transAxes,
                        color="#FF9800",
                    )
                    row_axes[0].set_title(
                        f"{res['date']} - Regime Skip", fontsize=10, color="#FF9800"
                    )
                    row_axes[1].text(
                        0.5,
                        0.5,
                        "Regime filter\nNo trades",
                        ha="center",
                        va="center",
                        fontsize=10,
                        transform=row_axes[1].transAxes,
                        color="#FF9800",
                    )
                    row_axes[1].set_title("P&L: $0 (skipped)", fontsize=10, color="#FF9800")
                else:
                    plot_day(res["date"], res["picks"], res["states"], i, row_axes)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            chart_path = os.path.join(run_dir, f"backtest_page_{page + 1:02d}.png")
            plt.savefig(chart_path, dpi=120, bbox_inches="tight")
            plt.close()
            chart_files.append(chart_path)
            sys.stdout.write(f"\r  Charts: page {page + 1}/{num_pages}")
            sys.stdout.flush()

        print(f"\r  {num_pages} chart pages saved to {run_dir}/          ")

        # --- RUN STRESS TESTS ---
        print("  Running stress tests...")
        optimizer_picks = _load_optimizer_picks(data_dirs)
        stress_results = _run_stress_tests(optimizer_picks)
        stress_pass = stress_results["all_pass"]
        print(f"  Stress tests: {'ALL PASSED' if stress_pass else 'SOME FAILED'}")

        # --- OVERALL SUMMARY CHART ---
        fig2 = plt.figure(figsize=(24, 24))
        gs = fig2.add_gridspec(
            4,
            3,
            width_ratios=[1.2, 1.2, 0.8],
            height_ratios=[1.0, 1.0, 1.0, 0.7],
            hspace=0.35,
            wspace=0.3,
        )
        fig2.suptitle(
            f"Full Backtest Summary: {ALL_DATES[0]} to {ALL_DATES[-1]} ({len(ALL_DATES)} days)",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Daily P&L bars — FULL WIDTH
        ax = fig2.add_subplot(gs[0, :])
        dates = [r["date"] for r in all_results]
        daily_pnls = [r["day_pnl"] for r in all_results]
        bar_colors = ["#4CAF50" if p >= 0 else "#f44336" for p in daily_pnls]
        ax.bar(range(len(dates)), daily_pnls, color=bar_colors, edgecolor="none", width=0.8)
        ax.set_xticks(range(0, len(dates), max(1, len(dates) // 15)))
        ax.set_xticklabels(
            [dates[i] for i in range(0, len(dates), max(1, len(dates) // 15))],
            rotation=45,
            fontsize=8,
            ha="right",
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        total_pnl = sum(daily_pnls)
        total_color = "#4CAF50" if total_pnl >= 0 else "#f44336"
        ax.set_title(
            f"Daily P&L | Total: ${total_pnl:+,.2f}", fontsize=13, fontweight="bold", color=total_color
        )
        ax.set_ylabel("P&L ($)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

        # 2. Equity Curve — FULL WIDTH
        ax = fig2.add_subplot(gs[1, :])
        balances = [r["ending_cash"] for r in all_results]
        ax.plot(range(len(dates)), balances, color="#2196F3", linewidth=2)
        ax.fill_between(
            range(len(dates)),
            STARTING_CASH,
            balances,
            where=[b >= STARTING_CASH for b in balances],
            alpha=0.15,
            color="#4CAF50",
        )
        ax.fill_between(
            range(len(dates)),
            STARTING_CASH,
            balances,
            where=[b < STARTING_CASH for b in balances],
            alpha=0.15,
            color="#f44336",
        )
        ax.axhline(
            y=STARTING_CASH,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Start: ${STARTING_CASH:,}",
        )
        # Mark margin threshold ($25K)
        ax.axhline(
            y=MARGIN_THRESHOLD,
            color="#FF9800",
            linestyle=":",
            alpha=0.6,
            label=f"Margin ${MARGIN_THRESHOLD / 1000:.0f}K",
        )
        if crossed_margin_date:
            cross_idx = dates.index(crossed_margin_date)
            ax.axvline(x=cross_idx, color="#FF9800", linestyle="--", alpha=0.4)
            ax.annotate(
                f"Margin unlocked\n{crossed_margin_date}",
                xy=(cross_idx, MARGIN_THRESHOLD),
                fontsize=8,
                xytext=(10, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#FF9800"),
                color="#FF9800",
                fontweight="bold",
            )
        ax.set_xticks(range(0, len(dates), max(1, len(dates) // 15)))
        ax.set_xticklabels(
            [dates[i] for i in range(0, len(dates), max(1, len(dates) // 15))],
            rotation=45,
            fontsize=8,
            ha="right",
        )
        ax.set_title(
            f"Equity Curve: ${STARTING_CASH:,} -> ${balances[-1]:,.0f}", fontsize=13, fontweight="bold"
        )
        ax.set_ylabel("Account Balance ($)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        cum_pnl = np.cumsum(daily_pnls)

        # 3. Exit reasons pie
        ax = fig2.add_subplot(gs[2, 0])
        if all_exits:
            reason_colors = {
                "STOP_LOSS": "#f44336",
                "TRAIL_STOP": "#FF9800",
                "EOD_CLOSE": "#2196F3",
                "PARTIAL+EOD": "#8BC34A",
                "TARGET": "#4CAF50",
            }
            labels = list(all_exits.keys())
            sizes = list(all_exits.values())
            colors = [reason_colors.get(r, "#999") for r in labels]
            ax.pie(
                sizes,
                labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
                colors=colors,
                autopct="%1.0f%%",
                startangle=90,
            )
        ax.set_title("Exit Reasons", fontweight="bold")

        # 4. Stats text
        ax = fig2.add_subplot(gs[2, 1])
        ax.axis("off")
        # Chart-specific stats (reuse total_trades etc. computed above)
        avg_win = (
            np.mean([s["pnl"] for r in all_results for s in r["states"]
                     if s["entry_price"] is not None and s["pnl"] > 0])
            if total_winners > 0 else 0
        )
        avg_loss = (
            np.mean([s["pnl"] for r in all_results for s in r["states"]
                     if s["entry_price"] is not None and s["pnl"] <= 0])
            if total_losers > 0 else 0
        )
        max_dd = min(cum_pnl) if len(cum_pnl) > 0 else 0
        red_days = sum(1 for p in daily_pnls if p <= 0)
        final_cash = all_results[-1]["ending_cash"]
        total_return_pct = (final_cash - STARTING_CASH) / STARTING_CASH * 100
        profit_factor_val = (
            abs(avg_win * total_winners / (avg_loss * total_losers))
            if (total_losers > 0 and avg_loss != 0) else 0
        )
        daily_pct_gains = []
        for r in all_results:
            if r["starting_cash"] > 0 and not r.get("regime_skip"):
                daily_pct_gains.append(r["day_pnl"] / r["starting_cash"] * 100)
        avg_daily_pct = np.mean(daily_pct_gains) if daily_pct_gains else 0
        trade_pct_gains = [
            s["pnl_pct"] for r in all_results for s in r["states"]
            if s["entry_price"] is not None
        ]
        avg_trade_pct = np.mean(trade_pct_gains) if trade_pct_gains else 0

        stats_text = (
            f"PERFORMANCE STATS\n"
            f"{'=' * 35}\n"
            f"Starting Cash:   ${STARTING_CASH:,.2f}\n"
            f"Ending Cash:     ${final_cash:,.2f}\n"
            f"Total Return:    {total_return_pct:+.1f}%\n"
            f"Total P&L:       ${total_pnl:+,.2f}\n"
            f"{'=' * 35}\n"
            f"Trading Days:    {len(ALL_DATES)}\n"
            f"Total Trades:    {total_trades}\n"
            f"Winners:         {total_winners}  ({win_rate:.1f}%)\n"
            f"Losers:          {total_losers}\n"
            f"Scale-ins:       {scale_in_count}\n"
            f"Vol-capped:      {vol_capped_count}\n"
            f"Skipped (low vol): {low_vol_count}\n"
            f"Regime skipped:  {regime_skipped} days\n"
            f"{'=' * 35}\n"
            f"Cash acct days:  {cash_acct_days}\n"
            f"Margin from:     {margin_str}\n"
            f"{'=' * 35}\n"
            f"Avg Daily Gain:  {avg_daily_pct:+.2f}%\n"
            f"Avg Trade Gain:  {avg_trade_pct:+.2f}%\n"
            f"Avg Win:         ${avg_win:+,.2f}\n"
            f"Avg Loss:        ${avg_loss:+,.2f}\n"
            f"Profit Factor:   {profit_factor_val:.2f}\n"
            f"{'=' * 35}\n"
            f"Green Days:      {green_days}/{len(ALL_DATES)}\n"
            f"Red Days:        {red_days}/{len(ALL_DATES)}\n"
            f"Best Day:        ${best_day:+,.2f}\n"
            f"Worst Day:       ${worst_day:+,.2f}\n"
            f"Max Drawdown:    ${max_dd:+,.2f}\n"
            f"Avg P&L/Day:     ${total_pnl / len(ALL_DATES):+,.2f}\n"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 5. Config details (right column, spans rows 2-3)
        ax = fig2.add_subplot(gs[2:, 2])
        ax.axis("off")
        si_label = f"+{SCALE_IN_TRIGGER_PCT}% ({int(SCALE_IN_FRAC*100)}% add)" if SCALE_IN else "off"
        struct_label = f"PM_high - {STRUCTURAL_STOP_ATR_MULT}*ATR" if STRUCTURAL_STOP else "off"
        runner_label = f"+{RUNNER_TRIGGER_PCT}%" if RUNNER_MODE else "off"
        liq_label = f"{LIQ_CANDLE_ATR_MULT}x ATR + {LIQ_VOL_MULT}x vol" if LIQ_VACUUM else "off"
        be_label = f"+{BREAKEVEN_STOP_PCT}%" if BREAKEVEN_STOP_PCT > 0 else "off"
        cutoff_label = f"{ENTRY_CUTOFF_MINUTES}min" if ENTRY_CUTOFF_MINUTES > 0 else "off"
        config_text = (
            f"STRATEGY CONFIG\n"
            f"{'=' * 32}\n"
            f"\n"
            f"ENTRY\n"
            f"{'-' * 32}\n"
            f"Confirm:        {CONFIRM_ABOVE}/{CONFIRM_WINDOW} candles\n"
            f"Pullback:       {PULLBACK_PCT}%\n"
            f"PB Timeout:     {PULLBACK_TIMEOUT} candles\n"
            f"Entry Cutoff:   {cutoff_label}\n"
            f"Liq Vacuum:     {liq_label}\n"
            f"\n"
            f"EXIT\n"
            f"{'-' * 32}\n"
            f"Stop Loss:      {STOP_LOSS_PCT}%\n"
            f"Struct Stop:    {struct_label}\n"
            f"Breakeven:      {be_label}\n"
            f"Tranche 1:      {int(PARTIAL_SELL_FRAC * 100)}% @ +{PARTIAL_SELL_PCT}%\n"
            f"Tranche 2:      {int(PARTIAL_SELL_FRAC_2 * 100)}% @ +{PARTIAL_SELL_PCT_2}%\n"
            f"Trail:          ATR({ATR_PERIOD}) x {ATR_MULTIPLIER}\n"
            f"Runner Mode:    {runner_label}\n"
            f"EOD Exit:       {EOD_EXIT_MINUTES}min before close\n"
            f"Reset Stops:    {'yes' if RESET_STOPS_ON_PARTIAL else 'no'}\n"
            f"\n"
            f"SIZING\n"
            f"{'-' * 32}\n"
            f"{'Adaptive:' if ADAPTIVE_SIZING else 'Position:':<16}"
            f"{'yes' if ADAPTIVE_SIZING else f'{int(TRADE_PCT * 100)}% of cash'}\n"
            + (
                "".join(
                    f"  ${t / 1000:>5.0f}K+:      {int(p * 100)}%\n" for t, p in SIZING_TIERS
                )
                if ADAPTIVE_SIZING
                else ""
            )
            + f"Scale-in:       {si_label}\n"
            f"Starting Cash:  ${STARTING_CASH:,}\n"
            f"Slippage:       {SLIPPAGE_PCT}%\n"
            f"Cash -> Margin: ${MARGIN_THRESHOLD / 1000:.0f}K\n"
            f"\n"
            f"FILTERS\n"
            f"{'-' * 32}\n"
            f"Min Gap:        {MIN_GAP_PCT}%\n"
            f"Min PM Vol:     {MIN_PM_VOLUME:,}\n"
            f"Top N:          {TOP_N}\n"
            f"Regime:         SPY > SMA({SPY_SMA_PERIOD})\n"
            f"Cash Wait:      {'all day' if CASH_WAIT_ENABLED else 'off'}\n"
            f"Vol Cap:        {VOL_CAP_PCT}% of $vol\n"
            f"\n"
            f"RUN INFO\n"
            f"{'-' * 32}\n"
            f"Date range:     {ALL_DATES[0]}\n"
            f"                {ALL_DATES[-1]}\n"
            f"Trading days:   {len(ALL_DATES)}\n"
            f"Run time:       {run_ts}\n"
        )
        ax.text(
            0.05,
            0.95,
            config_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.7),
        )

        # 6. Stress test results (bottom row, spans 2 columns)
        ax = fig2.add_subplot(gs[3, 0:2])
        ax.axis("off")
        verdict_color = "#4CAF50" if stress_results["all_pass"] else "#f44336"
        overall_label = "ALL PASSED" if stress_results["all_pass"] else "SOME FAILED"
        st_lines = f"STRESS TESTS (flat $10K/day)        [{overall_label}]\n"
        st_lines += f"{'=' * 52}\n"
        st_lines += f"{'Test':<28} {'P&L':>10} {'Sharpe':>7} {'Result':>7}\n"
        st_lines += f"{'-' * 28} {'-' * 10} {'-' * 7} {'-' * 7}\n"
        for row in stress_results["rows"]:
            pnl_str = (
                f"${row['pnl']:>+9,.0f}"
                if row["pnl"] is not None
                else f"{row['sharpe']:>9.1f}%"
            )
            sharpe_str = f"{row['sharpe']:>7.2f}" if row["pnl"] is not None else f"{'':>7}"
            st_lines += f"{row['test']:<28} {pnl_str} {sharpe_str} {row['verdict']:>7}\n"
        st_lines += f"{'-' * 52}\n"
        st_lines += f"Max losing streak: {stress_results['max_streak']} days\n"

        ax.text(
            0.05,
            0.95,
            st_lines,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor="#E8F5E9" if stress_results["all_pass"] else "#FFEBEE",
                alpha=0.7,
            ),
        )

        plt.tight_layout()
        summary_path = os.path.join(run_dir, "backtest_summary.png")
        plt.savefig(summary_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Summary chart saved to: {summary_path}")

    # --- PRINT SUMMARY TABLE ---
    print(f"\n{'=' * 90}")
    print(
        f"  FULL BACKTEST: {ALL_DATES[0]} to {ALL_DATES[-1]} ({len(ALL_DATES)} trading days)"
    )
    print(f"{'=' * 90}")
    print(
        f"{'Date':<14}{'Trades':>8}{'Winners':>10}{'Losers':>9}{'Size%':>7}{'Day P&L':>12}{'Balance':>14}"
    )
    print("-" * 82)
    for res in all_results:
        if res.get("regime_skip"):
            print(
                f"{res['date']:<14}{'SKIP':>8}{'':>10}{'':>9}{'':>7}  {'$+0.00':>11}  ${res['ending_cash']:>12,.2f}  <-- SPY < SMA{SPY_SMA_PERIOD}"
            )
            continue
        traded_states = [s for s in res["states"] if s["entry_price"] is not None]
        winners = sum(1 for s in traded_states if s["pnl"] > 0)
        losers = sum(1 for s in traded_states if s["pnl"] <= 0)
        pct_str = f"{res.get('trade_pct', TRADE_PCT) * 100:.0f}%"
        acct_tag = "  <-- cash acct" if res.get("cash_account") else ""
        if res["date"] == crossed_margin_date:
            acct_tag = "  <-- MARGIN UNLOCKED"
        print(
            f"{res['date']:<14}{res['traded']:>8}{winners:>10}{losers:>9}{pct_str:>7}  ${res['day_pnl']:>+9,.2f}  ${res['ending_cash']:>12,.2f}{acct_tag}"
        )

    final_cash = all_results[-1]["ending_cash"]
    total_return_pct = (final_cash - STARTING_CASH) / STARTING_CASH * 100
    print("-" * 82)
    print(
        f"{'TOTAL':<14}{total_trades:>8}{total_winners:>10}{total_losers:>9}{'':>7}  ${total_pnl:>+9,.2f}  ${final_cash:>12,.2f}"
    )
    print(
        f"\n  Starting Cash: ${STARTING_CASH:,}  ->  Ending: ${final_cash:,.2f}  ({total_return_pct:+.1f}%)"
    )
    print(
        f"  Win Rate: {win_rate:.1f}%  |  Total P&L: ${total_pnl:+,.2f}  |  Avg/Day: ${total_pnl / len(ALL_DATES):+,.2f}"
    )
    print(
        f"  Green Days: {green_days}/{len(ALL_DATES)}  |  Best: ${best_day:+,.2f}  |  Worst: ${worst_day:+,.2f}"
    )
    cw_str = f"  |  Cash-wait entries: {cash_wait_count}" if cash_wait_count > 0 else ""
    vc_str = f"  |  Vol-capped: {vol_capped_count}" if vol_capped_count > 0 else ""
    lv_str = f"  |  Skipped (low vol): {low_vol_count}" if low_vol_count > 0 else ""
    print(
        f"  Scale-ins: {scale_in_count}{cw_str}{vc_str}{lv_str}  |  Regime skipped: {regime_skipped} days (SPY < SMA{SPY_SMA_PERIOD})"
    )
    print(f"  Cash acct: {cash_acct_days} days (T+1)  |  Margin from: {margin_str}")
    print(f"{'=' * 90}")

    # --- STRESS TEST CONSOLE SUMMARY ---
    if stress_results is None:
        print(f"{'=' * 90}")
        sys.exit(0)
    overall_label = "ALL PASSED" if stress_results["all_pass"] else "SOME FAILED"
    print(f"\n  STRESS TESTS (flat $10K/day)  [{overall_label}]")
    print(f"  {'-' * 55}")
    print(f"  {'Test':<28} {'P&L':>10} {'Sharpe':>7} {'Result':>7}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 7} {'-' * 7}")
    for row in stress_results["rows"]:
        pnl_str = (
            f"${row['pnl']:>+9,.0f}"
            if row["pnl"] is not None
            else f"{row['sharpe']:>9.1f}%"
        )
        sharpe_str = f"{row['sharpe']:>7.2f}" if row["pnl"] is not None else f"{'':>7}"
        print(f"  {row['test']:<28} {pnl_str} {sharpe_str} {row['verdict']:>7}")
    print(f"  {'-' * 55}")
    print(f"  Max losing streak: {stress_results['max_streak']} days")
    print(f"{'=' * 90}")
