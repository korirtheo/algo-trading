"""
Optuna Parameter Optimizer for Premarket Gap Strategy (Phase 2)
===============================================================
Step 1: Precompute daily picks (load CSVs once, save to pickle) — no volume filter
Step 2: Run Optuna optimization using cached data — fast

Parameters optimized (16 total):
  Original 10:
    - STOP_LOSS_PCT, PARTIAL_SELL_PCT, PARTIAL_SELL_FRAC
    - ATR_PERIOD, ATR_MULTIPLIER
    - CONFIRM_ABOVE, CONFIRM_WINDOW
    - PULLBACK_PCT, PULLBACK_TIMEOUT
    - TRADE_PCT
  New 6:
    - N_EXIT_TRANCHES (1-3): number of partial sell levels
    - PARTIAL_SELL_PCT_2, PARTIAL_SELL_FRAC_2: second tranche (when n_tranches=3)
    - MIN_PM_VOLUME: optimizable volume filter (categorical)
    - SCALE_IN: enable/disable adding to position
    - SCALE_IN_TRIGGER_PCT: % above entry to trigger scale-in

Matches test_full.py: SPY regime filter, adaptive sizing tiers, risk-adjusted objective.

Usage:
  python optimize.py                           # uses both data dirs
  python optimize.py stored_data               # single dir
  python optimize.py stored_data stored_data_oos  # explicit dirs
"""
import os
import sys
import re
import pickle
import time
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import optuna
from regime_filters import RegimeFilter


def _is_warrant_or_unit(ticker):
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


optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- FIXED CONFIG ---
DAILY_CASH = 10_000
SLIPPAGE_PCT = 0.05
MIN_GAP_PCT = 2.0
TOP_N = 10
ET_TZ = ZoneInfo("America/New_York")

N_TRIALS = 500

# Regime filter (matches test_full.py)
SPY_SMA_PERIOD = 40

# Adaptive sizing tiers (matches test_full.py)
SIZING_TIERS = [
    (0, 0.25),       # $0 - $25K: 25%
    (25_000, 0.40),  # $25K - $50K: 40%
    (50_000, 0.50),  # $50K+: 50%
]

# Default data directories (uses both if no args)
DEFAULT_DATA_DIRS = ["stored_data_combined"]


# ─── STEP 1: PRECOMPUTE PICKS (NO volume filter) ─────────────────────────────

def _process_one_day(args):
    """Worker: scan all tickers for one day. No volume filter — saves pm_volume."""
    test_date, all_tickers, intraday_dir, daily_dir = args
    et_tz = ZoneInfo("America/New_York")

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
            et_index = idf.index.tz_convert(et_tz)
        else:
            et_index = idf.index.tz_localize("UTC").tz_convert(et_tz)

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
        pm_volume = int(premarket["Volume"].sum()) if len(premarket) > 0 else 0

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
            "premarket_high": premarket_high,
            "pm_volume": pm_volume,
            "market_hour_candles": market_hours,
        })

    candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
    return test_date, candidates[:TOP_N]


def precompute_picks_for_dir(data_dir):
    """Precompute picks for a single data dir (no volume filter)."""
    pickle_path = os.path.join(data_dir, "optimize_picks_novol.pkl")
    if os.path.exists(pickle_path):
        print(f"  Loading cached picks from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    intraday_dir = os.path.join(data_dir, "intraday")
    daily_dir = os.path.join(data_dir, "daily")
    all_tickers = [f.replace(".csv", "") for f in os.listdir(intraday_dir) if f.endswith(".csv")]

    gainers_csv = os.path.join(data_dir, "daily_top_gainers.csv")
    gdf = pd.read_csv(gainers_csv)
    all_dates = sorted(gdf["date"].unique().tolist())

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"  Scanning {len(all_tickers)} tickers x {len(all_dates)} days in {data_dir}...")

    task_args = [(d, all_tickers, intraday_dir, daily_dir) for d in all_dates]
    daily_picks = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_day, args): args[0] for args in task_args}
        for future in as_completed(futures):
            date_str, picks = future.result()
            daily_picks[date_str] = picks
            completed += 1
            sys.stdout.write(f"\r    [{completed}/{len(all_dates)}] {date_str}...")
            sys.stdout.flush()

    print(f"\r    Precomputed {len(daily_picks)} days.                        ")

    with open(pickle_path, "wb") as f:
        pickle.dump(daily_picks, f)
    print(f"    Saved to {pickle_path}")

    return daily_picks


def load_all_picks(data_dirs):
    """Load and merge picks from multiple data directories."""
    merged = {}
    for d in data_dirs:
        if not os.path.isdir(d):
            print(f"  Skipping {d} (not found)")
            continue
        picks = precompute_picks_for_dir(d)
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
    return merged


# ─── STEP 2: FAST SIMULATION ─────────────────────────────────────────────────

def simulate_day_fast(picks, params):
    """Simulate one day with two-pass structure.
    Pass 1: Process exits (frees cash) + detect entry signals.
    Pass 2: Allocate capital to entries, then scale-ins.
    No file I/O — pure computation.
    """
    # Apply volume filter dynamically
    min_vol = params.get("min_pm_volume", 0)
    if min_vol > 0:
        picks = [p for p in picks if p.get("pm_volume", 0) >= min_vol]

    # Apply gap filter dynamically
    min_gap = params.get("min_gap_pct", MIN_GAP_PCT)
    if min_gap > MIN_GAP_PCT:
        picks = [p for p in picks if p.get("gap_pct", 0) >= min_gap]

    if not picks:
        return 0.0

    cash = DAILY_CASH
    unsettled = 0.0
    cash_account = params.get("cash_account", False)
    cash_wait = params.get("cash_wait", False)
    vol_cap_pct = params.get("vol_cap_pct", 5.0)
    eod_exit_minutes = params.get("eod_exit_minutes", 30)
    reset_stops = params.get("reset_stops_on_partial", False)

    # Adaptive sizing: pick tier based on DAILY_CASH (matches test_full.py)
    trade_pct = SIZING_TIERS[0][1]  # default to lowest tier
    for threshold, pct in SIZING_TIERS:
        if DAILY_CASH >= threshold:
            trade_pct = pct
    trade_size_base = DAILY_CASH * trade_pct
    trades_taken = 0
    day_pnl = 0.0

    def _receive_proceeds(amount):
        nonlocal cash, unsettled
        if cash_account:
            unsettled += amount
        else:
            cash += amount

    all_timestamps = set()
    for pick in picks:
        all_timestamps.update(pick["market_hour_candles"].index.tolist())
    all_timestamps = sorted(all_timestamps)

    n_tranches = params.get("n_exit_tranches", 2)
    scale_in_enabled = params.get("scale_in", 0)
    scale_in_trigger = params.get("scale_in_trigger_pct", 15.0)
    scale_in_frac = params.get("scale_in_frac", 0.50)
    runner_mode_enabled = params.get("runner_mode", 0)
    runner_trigger = params.get("runner_trigger_pct", 25.0)
    liq_vacuum = params.get("liq_vacuum", 0)
    liq_candle_mult = params.get("liq_candle_atr_mult", 2.5)
    liq_vol_mult = params.get("liq_vol_mult", 3.0)
    structural_stop = params.get("structural_stop", 0)
    structural_stop_atr = params.get("structural_stop_atr_mult", 1.5)
    stop_pct = params["stop_loss_pct"]
    partial_frac_1 = params["partial_sell_frac"]
    partial_pct_1 = params["partial_sell_pct"]
    partial_frac_2 = params.get("partial_sell_frac_2", 0.50)
    partial_pct_2 = params.get("partial_sell_pct_2", 30.0)
    atr_period = params["atr_period"]
    atr_mult = params["atr_multiplier"]
    confirm_above = params["confirm_above"]
    confirm_window = params["confirm_window"]
    pullback_pct = params["pullback_pct"]
    pullback_timeout = params["pullback_timeout"]

    states = []
    for pick in picks:
        states.append({
            "premarket_high": pick["premarket_high"],
            "pm_volume": pick.get("pm_volume", 0),
            "gap_pct": pick.get("gap_pct", 0),
            "mh": pick["market_hour_candles"],
            "entry_price": None,
            "position_cost": 0.0,
            "shares": 0,
            "remaining_shares": 0,
            "total_exit_value": 0.0,
            "total_cash_spent": 0.0,
            "partial_sold_1": False,
            "partial_sold_2": False,
            "trailing_active": False,
            "highest_since_entry": 0.0,
            "done": False,
            "recent_closes": [],
            "breakout_confirmed": False,
            "pullback_detected": False,
            "candles_since_confirm": 0,
            "true_ranges": [],
            "prev_candle_close": None,
            "candle_volumes": [],
            # Scale-in tracking
            "scaled_in": False,
            "original_position_size": 0.0,
            "original_entry_price": None,
            "signal_close_price": None,
            "runner_mode": False,
            "waiting_for_cash": False,
        })

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

            # ATR accumulation
            if st["prev_candle_close"] is not None:
                tr = max(c_high - c_low,
                         abs(c_high - st["prev_candle_close"]),
                         abs(c_low - st["prev_candle_close"]))
            else:
                tr = c_high - c_low
            st["true_ranges"].append(tr)
            if len(st["true_ranges"]) > atr_period:
                st["true_ranges"] = st["true_ranges"][-atr_period:]
            st["prev_candle_close"] = c_close

            st["candle_volumes"].append(c_vol)
            if len(st["candle_volumes"]) > 20:
                st["candle_volumes"] = st["candle_volumes"][-20:]

            pm_high = st["premarket_high"]

            # === OPEN POSITION: process exits ===
            if st["entry_price"] is not None:
                if c_high > st["highest_since_entry"]:
                    st["highest_since_entry"] = c_high

                # Scale-in: execute INLINE before partial sells
                if (scale_in_enabled and not st["scaled_in"]
                        and st["remaining_shares"] > 0):
                    trigger_price = st["original_entry_price"] * (1 + scale_in_trigger / 100)
                    if c_high >= trigger_price:
                        add_size = st["original_position_size"] * scale_in_frac
                        if vol_cap_pct > 0:
                            pre_si = st["mh"].loc[st["mh"].index <= ts]
                            si_vol = pre_si["Volume"].sum() if len(pre_si) > 0 else 0
                            si_dvol = trigger_price * si_vol
                            si_limit = si_dvol * (vol_cap_pct / 100) - st["position_cost"]
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
                            cash -= add_size
                        st["scaled_in"] = True

                # Stop loss (use original entry for pre-partial-sell)
                if st["remaining_shares"] > 0 and not st["trailing_active"]:
                    if structural_stop and st["true_ranges"]:
                        atr_val = sum(st["true_ranges"]) / len(st["true_ranges"])
                        struct_stop = st["premarket_high"] - structural_stop_atr * atr_val
                        base_price = st["original_entry_price"] if not st["partial_sold_1"] else st["entry_price"]
                        pct_stop = base_price * (1 - stop_pct / 100)
                        stop_price = max(struct_stop, pct_stop)
                    else:
                        base_price = st["original_entry_price"] if not st["partial_sold_1"] else st["entry_price"]
                        stop_price = base_price * (1 - stop_pct / 100)
                    if c_low <= stop_price:
                        sell_price = stop_price * (1 - SLIPPAGE_PCT / 100)
                        proceeds = st["remaining_shares"] * sell_price
                        st["total_exit_value"] += proceeds
                        st["remaining_shares"] = 0
                        st["done"] = True
                        _receive_proceeds(proceeds)
                        continue

                # Trailing stop
                if st["remaining_shares"] > 0 and st["trailing_active"]:
                    atr = sum(st["true_ranges"]) / len(st["true_ranges"]) if st["true_ranges"] else 0
                    trail_stop = st["highest_since_entry"] - (atr * atr_mult)
                    if c_low <= trail_stop:
                        sell_price = trail_stop * (1 - SLIPPAGE_PCT / 100)
                        proceeds = st["remaining_shares"] * sell_price
                        st["total_exit_value"] += proceeds
                        st["remaining_shares"] = 0
                        st["done"] = True
                        _receive_proceeds(proceeds)
                        continue

                # Runner mode detection
                if runner_mode_enabled and st["entry_price"] is not None and not st["runner_mode"]:
                    gain_pct = (c_close / st["entry_price"] - 1) * 100
                    if gain_pct >= runner_trigger:
                        st["runner_mode"] = True
                        st["trailing_active"] = True

                # Multi-tranche exits
                if st["runner_mode"]:
                    pass  # skip partial sells, trail stop handles exit
                elif n_tranches == 1:
                    if st["remaining_shares"] > 0 and not st["partial_sold_1"]:
                        target_price = st["original_entry_price"] * (1 + partial_pct_1 / 100)
                        if c_high >= target_price:
                            sell_price = target_price * (1 - SLIPPAGE_PCT / 100)
                            proceeds = st["remaining_shares"] * sell_price
                            st["total_exit_value"] += proceeds
                            st["remaining_shares"] = 0
                            st["partial_sold_1"] = True
                            st["done"] = True
                            _receive_proceeds(proceeds)

                elif n_tranches == 2:
                    if st["remaining_shares"] > 0 and not st["partial_sold_1"]:
                        target_price = st["original_entry_price"] * (1 + partial_pct_1 / 100)
                        if c_high >= target_price:
                            sell_shares = st["shares"] * partial_frac_1
                            if sell_shares > st["remaining_shares"]:
                                sell_shares = st["remaining_shares"]
                            sell_price = target_price * (1 - SLIPPAGE_PCT / 100)
                            proceeds = sell_shares * sell_price
                            st["total_exit_value"] += proceeds
                            st["remaining_shares"] -= sell_shares
                            st["partial_sold_1"] = True
                            st["trailing_active"] = True
                            _receive_proceeds(proceeds)
                            if reset_stops and st["remaining_shares"] > 0:
                                st["entry_price"] = target_price
                                st["position_cost"] = st["remaining_shares"] * target_price
                                st["highest_since_entry"] = c_high

                elif n_tranches == 3:
                    if st["remaining_shares"] > 0 and not st["partial_sold_1"]:
                        target_price = st["original_entry_price"] * (1 + partial_pct_1 / 100)
                        if c_high >= target_price:
                            sell_shares = st["shares"] * partial_frac_1
                            if sell_shares > st["remaining_shares"]:
                                sell_shares = st["remaining_shares"]
                            sell_price = target_price * (1 - SLIPPAGE_PCT / 100)
                            proceeds = sell_shares * sell_price
                            st["total_exit_value"] += proceeds
                            st["remaining_shares"] -= sell_shares
                            st["partial_sold_1"] = True
                            _receive_proceeds(proceeds)
                            if reset_stops and st["remaining_shares"] > 0:
                                st["entry_price"] = target_price
                                st["position_cost"] = st["remaining_shares"] * target_price
                                st["highest_since_entry"] = c_high

                    if st["remaining_shares"] > 0 and st["partial_sold_1"] and not st["partial_sold_2"]:
                        target_price_2 = st["entry_price"] * (1 + partial_pct_2 / 100)
                        if c_high >= target_price_2:
                            sell_shares = st["shares"] * partial_frac_2
                            if sell_shares > st["remaining_shares"]:
                                sell_shares = st["remaining_shares"]
                            sell_price = target_price_2 * (1 - SLIPPAGE_PCT / 100)
                            proceeds = sell_shares * sell_price
                            st["total_exit_value"] += proceeds
                            st["remaining_shares"] -= sell_shares
                            st["partial_sold_2"] = True
                            st["trailing_active"] = True
                            _receive_proceeds(proceeds)
                            if reset_stops and st["remaining_shares"] > 0:
                                st["entry_price"] = target_price_2
                                st["position_cost"] = st["remaining_shares"] * target_price_2
                                st["highest_since_entry"] = c_high

                # Early EOD exit: sell remaining shares within exit window
                if eod_exit_minutes > 0 and st["remaining_shares"] > 0:
                    ts_et = ts.astimezone(ET_TZ) if ts.tzinfo else ts
                    minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
                    if minutes_to_close <= eod_exit_minutes:
                        sell_price = c_close * (1 - SLIPPAGE_PCT / 100)
                        proceeds = st["remaining_shares"] * sell_price
                        st["total_exit_value"] += proceeds
                        st["remaining_shares"] = 0
                        st["done"] = True
                        _receive_proceeds(proceeds)
                        continue

                if st["remaining_shares"] <= 0:
                    st["done"] = True
                continue

            # === NO POSITION: check entry signals ===

            # 3-phase entry detection
            entered = False

            if not st["breakout_confirmed"]:
                st["recent_closes"].append(c_close > pm_high)
                if len(st["recent_closes"]) > confirm_window:
                    st["recent_closes"] = st["recent_closes"][-confirm_window:]
                if sum(st["recent_closes"]) >= confirm_above:
                    st["breakout_confirmed"] = True

            elif not st["pullback_detected"]:
                st["candles_since_confirm"] += 1
                pullback_zone = pm_high * (1 + pullback_pct / 100)
                if c_low <= pullback_zone:
                    st["pullback_detected"] = True
                    if c_close > pm_high:
                        entered = True
                elif st["candles_since_confirm"] >= pullback_timeout:
                    if c_close > pm_high:
                        entered = True
            else:
                if c_close > pm_high:
                    entered = True

            # Liquidity vacuum filter
            if entered and liq_vacuum:
                candle_range = c_high - c_low
                trs = st["true_ranges"]
                vols = st["candle_volumes"]
                atr_v = sum(trs) / len(trs) if len(trs) >= 3 else candle_range
                avg_v = sum(vols[:-1]) / len(vols[:-1]) if len(vols) > 1 else (c_vol if c_vol > 0 else 1)
                if avg_v > 0 and atr_v > 0:
                    if candle_range < liq_candle_mult * atr_v or c_vol < liq_vol_mult * avg_v:
                        entered = False

            if entered:
                st["signal_close_price"] = c_close
                entry_candidates.append(st)
            elif st["waiting_for_cash"] and cash_wait:
                if c_close > pm_high:
                    st["signal_close_price"] = c_close
                    entry_candidates.append(st)

        # ── PASS 2: Allocate capital to new entries ──
        # (Scale-ins already executed inline in Pass 1)
        entry_candidates.sort(key=lambda st: -(st.get("pm_volume", 0) * st.get("premarket_high", 1)))
        entry_cutoff = params.get("entry_cutoff_minutes", 0)

        for st in entry_candidates:
            if st["done"] or st["entry_price"] is not None:
                continue

            # Entry time cutoff
            if entry_cutoff > 0:
                ts_et = ts.astimezone(ET_TZ) if ts.tzinfo else ts
                mins_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)
                if mins_to_close <= entry_cutoff:
                    st["done"] = True
                    continue

            trades_taken += 1
            position_size = trade_size_base

            if cash < position_size:
                if cash > 0:
                    position_size = cash
                else:
                    trades_taken -= 1
                    if cash_wait:
                        st["waiting_for_cash"] = True
                    else:
                        st["done"] = True
                    continue

            fill_price = st["signal_close_price"]

            # Volume cap: limit position to vol_cap_pct% of dollar volume at buy
            if vol_cap_pct > 0:
                pre_entry = st["mh"].loc[st["mh"].index <= ts]
                vol_shares = pre_entry["Volume"].sum() if len(pre_entry) > 0 else 0
                dollar_vol = fill_price * vol_shares
                vol_limit = dollar_vol * (vol_cap_pct / 100)
                if vol_limit > 0 and position_size > vol_limit:
                    position_size = vol_limit
                if position_size < 50:
                    trades_taken -= 1
                    st["done"] = True
                    continue

            st["entry_price"] = fill_price * (1 + SLIPPAGE_PCT / 100)
            st["original_entry_price"] = st["entry_price"]
            st["position_cost"] = position_size
            st["total_cash_spent"] = position_size
            st["original_position_size"] = position_size
            st["shares"] = position_size / st["entry_price"]
            st["remaining_shares"] = st["shares"]
            st["highest_since_entry"] = float(st["mh"].loc[ts]["High"])
            cash -= position_size
            if st["waiting_for_cash"]:
                st["waiting_for_cash"] = False

    # EOD: clear waiting stocks
    for st in states:
        if st["waiting_for_cash"]:
            st["done"] = True
            st["waiting_for_cash"] = False

    # EOD close remaining positions
    for st in states:
        if st["entry_price"] is not None and st["remaining_shares"] > 0:
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["remaining_shares"] * sell_price
            st["total_exit_value"] += proceeds
            st["remaining_shares"] = 0
            _receive_proceeds(proceeds)

    # Total P&L and trade count
    day_trades = 0
    for st in states:
        if st["entry_price"] is not None:
            day_pnl += st["total_exit_value"] - st["total_cash_spent"]
            day_trades += 1

    return day_pnl, day_trades


def run_full_backtest(daily_picks, params, regime=None):
    """Run backtest across all days. Returns total P&L, stats, and trade count.
    If regime filter provided, skip days when SPY < SMA (matches test_full.py).
    """
    total_pnl = 0.0
    total_trades = 0
    daily_pnls = []

    for date_str in sorted(daily_picks.keys()):
        # Regime filter: skip bear days (matches test_full.py)
        if regime is not None:
            should_trade, _, _ = regime.check(date_str)
            if not should_trade:
                continue

        picks = daily_picks[date_str]
        day_pnl, day_trades = simulate_day_fast(picks, params)
        total_pnl += day_pnl
        total_trades += day_trades
        daily_pnls.append(day_pnl)

    daily_arr = np.array(daily_pnls)
    mean_daily = np.mean(daily_arr) if len(daily_arr) > 0 else 0
    std_daily = np.std(daily_arr) if len(daily_arr) > 1 else 1.0
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0

    return total_pnl, sharpe, daily_pnls, total_trades


# ─── STEP 3: OPTUNA OBJECTIVE ────────────────────────────────────────────────

VOLUME_CHOICES = [0, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]


def create_objective(daily_picks, regime=None):
    def objective(trial):
        # --- Original 9 params (trade_pct removed — uses adaptive sizing tiers) ---
        params = {
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 5.0, 25.0, step=1.0),
            "partial_sell_frac": trial.suggest_float("partial_sell_frac", 0.25, 1.00, step=0.05),
            "partial_sell_pct": trial.suggest_float("partial_sell_pct", 5.0, 50.0, step=1.0),
            "atr_period": trial.suggest_int("atr_period", 7, 28),
            "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 5.0, step=0.25),
            "confirm_above": trial.suggest_int("confirm_above", 1, 3),
            "confirm_window": trial.suggest_int("confirm_window", 2, 5),
            "pullback_pct": trial.suggest_float("pullback_pct", 1.0, 15.0, step=0.5),
            "pullback_timeout": trial.suggest_int("pullback_timeout", 3, 25),
        }

        # Constraint: confirm_above must be <= confirm_window
        if params["confirm_above"] > params["confirm_window"]:
            return float("-inf")

        # --- New: exit tranches ---
        n_tranches = trial.suggest_int("n_exit_tranches", 1, 3)
        params["n_exit_tranches"] = n_tranches

        if n_tranches == 1:
            # Sell 100% at target (partial_sell_frac forced to 1.0)
            params["partial_sell_frac"] = 1.0

        if n_tranches == 3:
            # Second tranche params (must have pct_2 > pct_1)
            params["partial_sell_frac_2"] = trial.suggest_float("partial_sell_frac_2", 0.25, 0.50, step=0.05)
            params["partial_sell_pct_2"] = trial.suggest_float("partial_sell_pct_2", 15.0, 80.0, step=5.0)
            # Ensure pct_2 > pct_1
            if params["partial_sell_pct_2"] <= params["partial_sell_pct"]:
                return float("-inf")

        # --- New: volume filter ---
        min_vol = trial.suggest_categorical("min_pm_volume", VOLUME_CHOICES)
        params["min_pm_volume"] = min_vol

        # --- New: scale-in ---
        scale_in = trial.suggest_int("scale_in", 0, 1)
        params["scale_in"] = scale_in
        if scale_in:
            params["scale_in_trigger_pct"] = trial.suggest_float("scale_in_trigger_pct", 5.0, 25.0, step=1.0)
            params["scale_in_frac"] = trial.suggest_float("scale_in_frac", 0.10, 1.00, step=0.05)
        else:
            params["scale_in_trigger_pct"] = 15.0
            params["scale_in_frac"] = 0.50

        # --- New: EOD exit timing ---
        params["eod_exit_minutes"] = trial.suggest_int("eod_exit_minutes", 0, 60, step=5)

        # --- New: entry cutoff ---
        params["entry_cutoff_minutes"] = trial.suggest_int("entry_cutoff_minutes", 0, 180, step=15)

        # --- Gap quality filter ---
        params["min_gap_pct"] = trial.suggest_float("min_gap_pct", 2.0, 15.0, step=1.0)

        # --- Runner mode: skip partial sells when stock runs ---
        runner_mode = trial.suggest_int("runner_mode", 0, 1)
        params["runner_mode"] = runner_mode
        if runner_mode:
            params["runner_trigger_pct"] = trial.suggest_float("runner_trigger_pct", 15.0, 50.0, step=5.0)
        else:
            params["runner_trigger_pct"] = 25.0

        # --- Liquidity vacuum: large candle + volume spike entry filter ---
        liq_vac = trial.suggest_int("liq_vacuum", 0, 1)
        params["liq_vacuum"] = liq_vac
        if liq_vac:
            params["liq_candle_atr_mult"] = trial.suggest_float("liq_candle_atr_mult", 1.5, 4.0, step=0.5)
            params["liq_vol_mult"] = trial.suggest_float("liq_vol_mult", 2.0, 5.0, step=0.5)
        else:
            params["liq_candle_atr_mult"] = 2.5
            params["liq_vol_mult"] = 3.0

        # --- Structural stop: PM_high - X*ATR instead of fixed % ---
        struct_stop = trial.suggest_int("structural_stop", 0, 1)
        params["structural_stop"] = struct_stop
        if struct_stop:
            params["structural_stop_atr_mult"] = trial.suggest_float("structural_stop_atr_mult", 1.0, 3.0, step=0.25)
        else:
            params["structural_stop_atr_mult"] = 1.5

        total_pnl, sharpe, _, total_trades = run_full_backtest(daily_picks, params, regime=regime)
        # Expectancy: average P&L per trade — rewards quality over quantity
        if total_trades == 0:
            return float("-inf")
        expectancy = total_pnl / total_trades
        return expectancy

    return objective


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Determine data directories
    # Parse --trials N from argv
    args = sys.argv[1:]
    n_trials = N_TRIALS
    if "--trials" in args:
        idx = args.index("--trials")
        n_trials = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]
    if args:
        data_dirs = args
    else:
        data_dirs = DEFAULT_DATA_DIRS

    print(f"Optuna Optimizer — Phase 4 (15 params + regime filter, {n_trials} trials)")
    print(f"Data directories: {data_dirs}\n")

    # Step 1: Load picks (no volume filter — filter applied dynamically)
    daily_picks = load_all_picks(data_dirs)
    all_dates = sorted(daily_picks.keys())
    total_candidates = sum(len(v) for v in daily_picks.values())
    print(f"\nLoaded {len(all_dates)} trading days, {total_candidates} total candidates")
    print(f"Date range: {all_dates[0]} to {all_dates[-1]}")

    # Load SPY regime filter (matches test_full.py)
    print("\nLoading regime filter...")
    regime = RegimeFilter(
        spy_ma_period=SPY_SMA_PERIOD,
        enable_vix=False,
        enable_spy_trend=True,
        enable_adaptive=False,
    )
    regime.load_data(all_dates[0], all_dates[-1])
    tradeable = sum(1 for d in all_dates if regime.check(d)[0])
    skipped = len(all_dates) - tradeable
    print(f"  Regime: {tradeable} tradeable days, {skipped} skipped (SPY < SMA({SPY_SMA_PERIOD}))")

    # Benchmark
    print("\nBenchmarking single backtest run...")
    # Current best params (matching test_full.py defaults)
    current_best = {
        "stop_loss_pct": 16.0,
        "partial_sell_frac": 0.90,
        "partial_sell_pct": 15.0,
        "atr_period": 8,
        "atr_multiplier": 4.25,
        "confirm_above": 2,
        "confirm_window": 4,
        "pullback_pct": 4.0,
        "pullback_timeout": 24,
        "n_exit_tranches": 3,
        "partial_sell_frac_2": 0.35,
        "partial_sell_pct_2": 25.0,
        "min_pm_volume": 250_000,
        "scale_in": 1,
        "scale_in_trigger_pct": 14.0,
        "scale_in_frac": 0.50,
        "eod_exit_minutes": 30,
        "entry_cutoff_minutes": 120,
    }
    t0 = time.time()
    baseline_pnl, baseline_sharpe, baseline_daily, baseline_trades = run_full_backtest(daily_picks, current_best, regime=regime)
    baseline_expectancy = baseline_pnl / baseline_trades if baseline_trades > 0 else 0
    t1 = time.time()
    print(f"  Current best P&L: ${baseline_pnl:+,.2f} | Expectancy: ${baseline_expectancy:+,.2f}/trade | Trades: {baseline_trades} | Time: {t1-t0:.2f}s")
    est_total = (t1 - t0) * n_trials / 60
    print(f"  Estimated optimization: {est_total:.0f} min for {n_trials} trials\n")

    # Step 2: Run Optuna
    print(f"{'='*70}")
    print(f"  RUNNING OPTUNA OPTIMIZATION ({n_trials} trials, 15 params)")
    print(f"  Params: 9 core + exit_tranches + volume_filter + scale_in + eod_exit + entry_cutoff")
    print(f"{'='*70}")

    study = optuna.create_study(direction="maximize", study_name="gap_strategy_v2")

    # Enqueue known good starting points
    # 1. Current best from test_full.py
    study.enqueue_trial({
        "stop_loss_pct": 16.0, "partial_sell_frac": 0.90, "partial_sell_pct": 15.0,
        "atr_period": 8, "atr_multiplier": 4.25, "confirm_above": 2, "confirm_window": 4,
        "pullback_pct": 4.0, "pullback_timeout": 24,
        "n_exit_tranches": 3, "partial_sell_frac_2": 0.35, "partial_sell_pct_2": 25.0,
        "min_pm_volume": 250_000, "scale_in": 1,
        "scale_in_trigger_pct": 14.0, "scale_in_frac": 0.50,
        "eod_exit_minutes": 30, "entry_cutoff_minutes": 120,
    })
    # 2. Previous optimizer best
    study.enqueue_trial({
        "stop_loss_pct": 17.0, "partial_sell_frac": 0.75, "partial_sell_pct": 42.0,
        "atr_period": 11, "atr_multiplier": 4.0, "confirm_above": 2, "confirm_window": 3,
        "pullback_pct": 8.5, "pullback_timeout": 14,
        "n_exit_tranches": 2, "min_pm_volume": 1_000_000, "scale_in": 0,
        "eod_exit_minutes": 30, "entry_cutoff_minutes": 0,
    })
    # 3. Scale-in 13% with cutoff
    study.enqueue_trial({
        "stop_loss_pct": 16.0, "partial_sell_frac": 0.90, "partial_sell_pct": 15.0,
        "atr_period": 8, "atr_multiplier": 4.25, "confirm_above": 2, "confirm_window": 4,
        "pullback_pct": 4.0, "pullback_timeout": 24,
        "n_exit_tranches": 3, "partial_sell_frac_2": 0.35, "partial_sell_pct_2": 25.0,
        "min_pm_volume": 250_000, "scale_in": 1,
        "scale_in_trigger_pct": 13.0, "scale_in_frac": 0.50,
        "eod_exit_minutes": 30, "entry_cutoff_minutes": 120,
    })
    # 4. No scale-in, with cutoff
    study.enqueue_trial({
        "stop_loss_pct": 16.0, "partial_sell_frac": 0.90, "partial_sell_pct": 15.0,
        "atr_period": 8, "atr_multiplier": 4.25, "confirm_above": 2, "confirm_window": 4,
        "pullback_pct": 4.0, "pullback_timeout": 24,
        "n_exit_tranches": 3, "partial_sell_frac_2": 0.35, "partial_sell_pct_2": 25.0,
        "min_pm_volume": 250_000, "scale_in": 0,
        "eod_exit_minutes": 30, "entry_cutoff_minutes": 120,
    })

    t_start = time.time()
    completed = [0]

    def progress_callback(study, trial):
        completed[0] += 1
        if completed[0] % 10 == 0 or completed[0] == n_trials:
            elapsed = time.time() - t_start
            best = study.best_value
            eta = (elapsed / completed[0]) * (n_trials - completed[0]) / 60
            sys.stdout.write(
                f"\r  Trial {completed[0]}/{n_trials} | "
                f"Best Expectancy: ${best:+,.2f}/trade | "
                f"ETA: {eta:.1f} min"
            )
            sys.stdout.flush()

    study.optimize(
        create_objective(daily_picks, regime=regime),
        n_trials=n_trials,
        callbacks=[progress_callback],
        show_progress_bar=False,
    )

    elapsed = time.time() - t_start
    print(f"\n\n  Optimization complete in {elapsed/60:.1f} minutes.\n")

    # Step 3: Results
    best = study.best_params
    best_pnl = study.best_value

    print(f"{'='*70}")
    print(f"  OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  Previous Best P&L:  ${baseline_pnl:+,.2f}")
    print(f"  Optimized P&L:      ${best_pnl:+,.2f}")
    improvement = best_pnl - baseline_pnl
    print(f"  Improvement:        ${improvement:+,.2f}\n")

    # Print all optimized params
    print(f"  {'Parameter':<26}{'Value':>14}")
    print(f"  {'-'*40}")
    all_param_names = [
        "stop_loss_pct", "partial_sell_frac", "partial_sell_pct",
        "atr_period", "atr_multiplier", "confirm_above", "confirm_window",
        "pullback_pct", "pullback_timeout",
        "n_exit_tranches", "min_pm_volume", "scale_in", "eod_exit_minutes",
        "entry_cutoff_minutes", "min_gap_pct", "runner_mode",
        "liq_vacuum", "structural_stop",
    ]
    for key in all_param_names:
        if key in best:
            val = best[key]
            if isinstance(val, float):
                print(f"  {key:<26}{val:>14.2f}")
            else:
                print(f"  {key:<26}{val:>14}")

    # Conditional params
    if best.get("n_exit_tranches", 2) == 3:
        for key in ["partial_sell_frac_2", "partial_sell_pct_2"]:
            if key in best:
                print(f"  {key:<26}{best[key]:>14.2f}")
    if best.get("scale_in", 0) == 1:
        if "scale_in_trigger_pct" in best:
            print(f"  {'scale_in_trigger_pct':<26}{best['scale_in_trigger_pct']:>14.2f}")
        if "scale_in_frac" in best:
            print(f"  {'scale_in_frac':<26}{best['scale_in_frac']:>14.2f}")
    if best.get("runner_mode", 0) == 1:
        if "runner_trigger_pct" in best:
            print(f"  {'runner_trigger_pct':<26}{best['runner_trigger_pct']:>14.2f}")
    if best.get("liq_vacuum", 0) == 1:
        if "liq_candle_atr_mult" in best:
            print(f"  {'liq_candle_atr_mult':<26}{best['liq_candle_atr_mult']:>14.2f}")
        if "liq_vol_mult" in best:
            print(f"  {'liq_vol_mult':<26}{best['liq_vol_mult']:>14.2f}")
    if best.get("structural_stop", 0) == 1:
        if "structural_stop_atr_mult" in best:
            print(f"  {'structural_stop_atr_mult':<26}{best['structural_stop_atr_mult']:>14.2f}")

    # Run final stats
    opt_pnl, opt_sharpe, opt_daily, opt_trades = run_full_backtest(daily_picks, best, regime=regime)
    opt_expectancy = opt_pnl / opt_trades if opt_trades > 0 else 0
    green_days = sum(1 for d in opt_daily if d > 0)
    red_days = sum(1 for d in opt_daily if d <= 0)
    cum = np.cumsum(opt_daily)
    max_dd = min(cum) if len(cum) > 0 else 0
    # Peak-to-trough drawdown
    peak = np.maximum.accumulate(cum)
    drawdowns = cum - peak
    max_dd_ptot = min(drawdowns) if len(drawdowns) > 0 else 0

    print(f"\n  OPTIMIZED STRATEGY STATS:")
    print(f"  {'-'*40}")
    print(f"  Total P&L:         ${opt_pnl:+,.2f}")
    print(f"  Expectancy/Trade:  ${opt_expectancy:+,.2f}")
    print(f"  Total Trades:      {opt_trades}")
    print(f"  Sharpe Ratio:      {opt_sharpe:.2f}")
    print(f"  Green Days:        {green_days}/{len(opt_daily)}")
    print(f"  Red Days:          {red_days}/{len(opt_daily)}")
    print(f"  Best Day:          ${max(opt_daily):+,.2f}")
    print(f"  Worst Day:         ${min(opt_daily):+,.2f}")
    print(f"  Max Drawdown (PT): ${max_dd_ptot:+,.2f}")
    print(f"  Avg P&L/Day:       ${np.mean(opt_daily):+,.2f}")

    # Copy-paste config
    print(f"\n  COPY-PASTE CONFIG FOR test_full.py:")
    print(f"  {'-'*40}")
    print(f"  STOP_LOSS_PCT = {best['stop_loss_pct']}")
    print(f"  PARTIAL_SELL_FRAC = {best['partial_sell_frac']}")
    print(f"  PARTIAL_SELL_PCT = {best['partial_sell_pct']}")
    print(f"  ATR_PERIOD = {best['atr_period']}")
    print(f"  ATR_MULTIPLIER = {best['atr_multiplier']}")
    print(f"  CONFIRM_ABOVE = {best['confirm_above']}")
    print(f"  CONFIRM_WINDOW = {best['confirm_window']}")
    print(f"  PULLBACK_PCT = {best['pullback_pct']}")
    print(f"  PULLBACK_TIMEOUT = {best['pullback_timeout']}")
    print(f"  N_EXIT_TRANCHES = {best.get('n_exit_tranches', 2)}")
    vol = best.get('min_pm_volume', 0)
    print(f"  MIN_PM_VOLUME = {vol:,}" if isinstance(vol, int) else f"  MIN_PM_VOLUME = {int(vol):,}")
    print(f"  SCALE_IN = {best.get('scale_in', 0)}")
    if best.get("scale_in", 0) == 1:
        print(f"  SCALE_IN_TRIGGER_PCT = {best.get('scale_in_trigger_pct', 15.0)}")
        print(f"  SCALE_IN_FRAC = {best.get('scale_in_frac', 0.50)}")
    if best.get("n_exit_tranches", 2) == 3:
        print(f"  PARTIAL_SELL_FRAC_2 = {best.get('partial_sell_frac_2', 0.50)}")
        print(f"  PARTIAL_SELL_PCT_2 = {best.get('partial_sell_pct_2', 30.0)}")
    print(f"  EOD_EXIT_MINUTES = {best.get('eod_exit_minutes', 0)}")
    print(f"  ENTRY_CUTOFF_MINUTES = {best.get('entry_cutoff_minutes', 0)}")
    print(f"  MIN_GAP_PCT = {best.get('min_gap_pct', 2.0)}")
    print(f"  RUNNER_MODE = {bool(best.get('runner_mode', 0))}")
    if best.get("runner_mode", 0) == 1:
        print(f"  RUNNER_TRIGGER_PCT = {best.get('runner_trigger_pct', 25.0)}")
    print(f"  LIQ_VACUUM = {bool(best.get('liq_vacuum', 0))}")
    if best.get("liq_vacuum", 0) == 1:
        print(f"  LIQ_CANDLE_ATR_MULT = {best.get('liq_candle_atr_mult', 2.5)}")
        print(f"  LIQ_VOL_MULT = {best.get('liq_vol_mult', 3.0)}")
    print(f"  STRUCTURAL_STOP = {bool(best.get('structural_stop', 0))}")
    if best.get("structural_stop", 0) == 1:
        print(f"  STRUCTURAL_STOP_ATR_MULT = {best.get('structural_stop_atr_mult', 1.5)}")
    print(f"  # Sizing: adaptive tiers {SIZING_TIERS}")
    print(f"{'='*70}")

    # Top 10 trials
    print(f"\n  TOP 10 PARAMETER COMBINATIONS:")
    print(f"  {'Rank':<6}{'P&L':>12}  Key Params")
    print(f"  {'-'*80}")
    trials_sorted = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else float("-inf"),
        reverse=True,
    )
    for i, trial in enumerate(trials_sorted[:10]):
        p = trial.params
        vol_str = f"{p.get('min_pm_volume', 0)/1000:.0f}K" if p.get('min_pm_volume', 0) > 0 else "none"
        tranche_str = f"T{p.get('n_exit_tranches', 2)}"
        scale_str = "S+" if p.get('scale_in', 0) else ""
        eod_str = f"EOD:{p.get('eod_exit_minutes', 30)}m"
        print(f"  {i+1:<6}${trial.value:>+10,.2f}  "
              f"Stop:{p['stop_loss_pct']:.0f}% Sell:{p['partial_sell_pct']:.0f}%@{p['partial_sell_frac']:.2f} "
              f"ATR:{p['atr_period']}x{p['atr_multiplier']:.1f} "
              f"Vol:{vol_str} {tranche_str} {scale_str} {eod_str}")
