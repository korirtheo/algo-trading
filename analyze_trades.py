"""
Trade Analysis Script — Phase 1
================================
Runs the backtest on all available data (stored_data + stored_data_oos),
collects detailed per-trade metadata, then outputs analysis tables.

Usage:
  python analyze_trades.py                  # uses both data dirs
  python analyze_trades.py stored_data      # single dir
"""
import os
import sys
import re
import math
import pickle
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- CONFIG (current optimized params from test_full.py) ---
SLIPPAGE_PCT = 0.05
STOP_LOSS_PCT = 17.0
DAILY_CASH = 10_000
TRADE_PCT = 0.50
MIN_GAP_PCT = 2.0
TOP_N = 10
PARTIAL_SELL_FRAC = 0.75
PARTIAL_SELL_PCT = 42.0
ATR_PERIOD = 11
ATR_MULTIPLIER = 4.0
CONFIRM_ABOVE = 2
CONFIRM_WINDOW = 3
PULLBACK_PCT = 8.5
PULLBACK_TIMEOUT = 14

ET_TZ = ZoneInfo("America/New_York")

# We do NOT filter by volume here — we save pm_volume so we can bucket later
MIN_PM_VOLUME = 0


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


def _process_one_day(args):
    """Worker: scan all tickers for one day. Saves pm_volume in picks (no volume filter)."""
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
            "market_open": market_open,
            "premarket_high": premarket_high,
            "prev_close": prev_close,
            "pm_volume": pm_volume,
            "market_hour_candles": market_hours,
        })

    candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
    return test_date, candidates[:TOP_N]


def load_picks_for_dir(data_dir):
    """Load or build picks for a single data directory (no volume filter)."""
    pickle_path = os.path.join(data_dir, "analyze_picks_novol.pkl")
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
    print(f"  Using {n_workers} workers")

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
                # Merge and re-sort by gap%, keep top N
                existing_tickers = {p["ticker"] for p in merged[date_str]}
                for p in day_picks:
                    if p["ticker"] not in existing_tickers:
                        merged[date_str].append(p)
                merged[date_str].sort(key=lambda x: x["gap_pct"], reverse=True)
                merged[date_str] = merged[date_str][:TOP_N]
    return merged


def simulate_day_enriched(picks):
    """Simulate one day with enriched per-trade metadata collection.
    Returns list of trade records with detailed features."""
    cash = DAILY_CASH
    trade_size_base = DAILY_CASH * TRADE_PCT
    trades_taken = 0
    trade_records = []

    all_timestamps = set()
    for pick in picks:
        all_timestamps.update(pick["market_hour_candles"].index.tolist())
    all_timestamps = sorted(all_timestamps)

    # Market open timestamp (first candle)
    market_open_ts = all_timestamps[0] if all_timestamps else None

    states = []
    for pick in picks:
        states.append({
            "ticker": pick["ticker"],
            "gap_pct": pick["gap_pct"],
            "market_open": pick.get("market_open", 0),
            "premarket_high": pick["premarket_high"],
            "prev_close": pick.get("prev_close", 0),
            "pm_volume": pick.get("pm_volume", 0),
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
            "lowest_since_entry": float("inf"),
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
            # Enriched tracking
            "max_favorable_excursion": 0.0,  # highest unrealized profit %
            "max_adverse_excursion": 0.0,     # deepest unrealized loss %
            "candles_in_trade": 0,
            "confirm_type": None,  # "pullback" or "timeout"
            "peak_price_at_exit": 0.0,
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

            # ATR accumulation
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
                        st["confirm_type"] = "pullback"
                        if c_close > pm_high:
                            entered = True
                    elif st["candles_since_confirm"] >= PULLBACK_TIMEOUT:
                        st["confirm_type"] = "timeout"
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
                    st["shares"] = position_size / st["entry_price"]
                    st["remaining_shares"] = st["shares"]
                    st["entry_time"] = ts
                    st["highest_since_entry"] = c_high
                    st["lowest_since_entry"] = c_low
                    cash -= position_size
                else:
                    continue

            # Track MFE/MAE
            st["candles_in_trade"] += 1
            if c_high > st["highest_since_entry"]:
                st["highest_since_entry"] = c_high
            if c_low < st["lowest_since_entry"]:
                st["lowest_since_entry"] = c_low

            unrealized_high = (c_high / st["entry_price"] - 1) * 100
            unrealized_low = (c_low / st["entry_price"] - 1) * 100
            if unrealized_high > st["max_favorable_excursion"]:
                st["max_favorable_excursion"] = unrealized_high
            if unrealized_low < st["max_adverse_excursion"]:
                st["max_adverse_excursion"] = unrealized_low

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
                    st["peak_price_at_exit"] = st["highest_since_entry"]
                    st["done"] = True
                    cash += proceeds
                    continue

            # Trail stop
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
                    st["peak_price_at_exit"] = st["highest_since_entry"]
                    st["done"] = True
                    cash += proceeds
                    continue

            # Partial sell
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
                st["peak_price_at_exit"] = st["highest_since_entry"]
                st["done"] = True

    # EOD close
    for st in states:
        if st["entry_price"] is not None and st["remaining_shares"] > 0:
            last_close = float(st["mh"].iloc[-1]["Close"])
            sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
            proceeds = st["remaining_shares"] * sell_price
            st["total_exit_value"] += proceeds
            st["remaining_shares"] = 0
            if st["partial_sold"]:
                st["exit_reason"] = "PARTIAL+EOD"
            else:
                st["exit_reason"] = "EOD_CLOSE"
            st["exit_price"] = last_close
            st["exit_time"] = st["mh"].index[-1]
            st["peak_price_at_exit"] = st["highest_since_entry"]
            cash += proceeds

    # Build trade records
    for st in states:
        if st["entry_price"] is not None:
            cost = st["position_cost"]
            pnl = st["total_exit_value"] - cost
            pnl_pct = (pnl / cost) * 100 if cost > 0 else 0

            # Minutes from market open to entry
            entry_minutes = None
            if st["entry_time"] is not None and market_open_ts is not None:
                try:
                    delta = (st["entry_time"] - market_open_ts).total_seconds() / 60
                    entry_minutes = max(0, delta)
                except Exception:
                    pass

            # Profit left on table (for trail/EOD exits)
            profit_left = 0.0
            if st["exit_price"] is not None and st["peak_price_at_exit"] > 0:
                profit_left = (st["peak_price_at_exit"] - st["exit_price"]) / st["entry_price"] * 100

            trade_records.append({
                "ticker": st["ticker"],
                "gap_pct": st["gap_pct"],
                "pm_volume": st["pm_volume"],
                "market_open": st["market_open"],
                "premarket_high": st["premarket_high"],
                "entry_price": st["entry_price"],
                "entry_time": st["entry_time"],
                "entry_minutes": entry_minutes,
                "exit_reason": st["exit_reason"],
                "exit_price": st["exit_price"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "mfe": st["max_favorable_excursion"],
                "mae": st["max_adverse_excursion"],
                "partial_sold": st["partial_sold"],
                "candles_in_trade": st["candles_in_trade"],
                "confirm_type": st["confirm_type"],
                "profit_left_on_table": profit_left,
            })

    return trade_records


# ─── ANALYSIS OUTPUTS ────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_exit_types(df):
    """1. Exit type distribution — how often does each exit type fire?"""
    print_section("EXIT TYPE DISTRIBUTION")
    grouped = df.groupby("exit_reason").agg(
        count=("pnl", "size"),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_mfe=("mfe", "mean"),
        avg_mae=("mae", "mean"),
    ).sort_values("count", ascending=False)

    print(f"\n  {'Exit Type':<16}{'Count':>8}{'Avg P&L':>12}{'Total P&L':>14}{'Win%':>8}{'Avg MFE':>10}{'Avg MAE':>10}")
    print(f"  {'-'*78}")
    for reason, row in grouped.iterrows():
        print(f"  {reason:<16}{row['count']:>8.0f}  ${row['avg_pnl']:>+9.2f}  ${row['total_pnl']:>+11.2f}{row['win_rate']:>7.1f}%{row['avg_mfe']:>+9.1f}%{row['avg_mae']:>+9.1f}%")
    total = df["pnl"].sum()
    print(f"\n  Total trades: {len(df)}  |  Total P&L: ${total:+,.2f}")


def analyze_partial_sells(df):
    """2. Partial sell hit rate."""
    print_section("PARTIAL SELL ANALYSIS")
    traded = df[df["exit_reason"] != "NO_CASH"]
    total = len(traded)
    partial_hit = traded["partial_sold"].sum()
    partial_miss = total - partial_hit

    print(f"\n  Partial sell target: +{PARTIAL_SELL_PCT}% (sell {PARTIAL_SELL_FRAC*100:.0f}% of position)")
    print(f"  Hit partial target:  {partial_hit}/{total} ({partial_hit/total*100:.1f}%)")
    print(f"  Never reached target: {partial_miss}/{total} ({partial_miss/total*100:.1f}%)")

    # Of those that missed partial, where did they exit?
    missed = traded[~traded["partial_sold"]]
    if len(missed) > 0:
        print(f"\n  Trades that MISSED partial target ({len(missed)} trades):")
        missed_exits = missed.groupby("exit_reason").agg(
            count=("pnl", "size"), avg_pnl=("pnl", "mean"), avg_mfe=("mfe", "mean"),
        )
        for reason, row in missed_exits.iterrows():
            print(f"    {reason:<16} {row['count']:>4.0f} trades  avg P&L: ${row['avg_pnl']:>+8.2f}  avg MFE: {row['avg_mfe']:>+6.1f}%")

    # Of those that hit partial, how did the remainder exit?
    hit = traded[traded["partial_sold"]]
    if len(hit) > 0:
        print(f"\n  Trades that HIT partial target ({len(hit)} trades):")
        hit_exits = hit.groupby("exit_reason").agg(
            count=("pnl", "size"), avg_pnl=("pnl", "mean"),
        )
        for reason, row in hit_exits.iterrows():
            print(f"    {reason:<16} {row['count']:>4.0f} trades  avg P&L: ${row['avg_pnl']:>+8.2f}")


def analyze_volume_buckets(df):
    """3. P&L and win rate by PM volume bucket."""
    print_section("PREMARKET VOLUME ANALYSIS")
    bins = [0, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000, float("inf")]
    labels = ["0-100K", "100K-250K", "250K-500K", "500K-1M", "1M-2M", "2M-5M", "5M+"]
    df["vol_bucket"] = pd.cut(df["pm_volume"], bins=bins, labels=labels, right=False)

    print(f"\n  {'Volume Bucket':<14}{'Trades':>8}{'Win%':>8}{'Avg P&L':>12}{'Total P&L':>14}{'Avg Gap%':>10}")
    print(f"  {'-'*66}")
    for label in labels:
        bucket = df[df["vol_bucket"] == label]
        if len(bucket) == 0:
            print(f"  {label:<14}{'0':>8}{'—':>8}{'—':>12}{'—':>14}{'—':>10}")
            continue
        wr = (bucket["pnl"] > 0).mean() * 100
        print(f"  {label:<14}{len(bucket):>8}{wr:>7.1f}%  ${bucket['pnl'].mean():>+9.2f}  ${bucket['pnl'].sum():>+11.2f}{bucket['gap_pct'].mean():>+9.1f}%")


def analyze_gap_buckets(df):
    """4. P&L by gap size."""
    print_section("GAP SIZE ANALYSIS")
    bins = [2, 5, 10, 20, 50, 100, float("inf")]
    labels = ["2-5%", "5-10%", "10-20%", "20-50%", "50-100%", "100%+"]
    df["gap_bucket"] = pd.cut(df["gap_pct"], bins=bins, labels=labels, right=False)

    print(f"\n  {'Gap Bucket':<12}{'Trades':>8}{'Win%':>8}{'Avg P&L':>12}{'Total P&L':>14}{'Avg MFE':>10}{'Avg MAE':>10}")
    print(f"  {'-'*74}")
    for label in labels:
        bucket = df[df["gap_bucket"] == label]
        if len(bucket) == 0:
            continue
        wr = (bucket["pnl"] > 0).mean() * 100
        print(f"  {label:<12}{len(bucket):>8}{wr:>7.1f}%  ${bucket['pnl'].mean():>+9.2f}  ${bucket['pnl'].sum():>+11.2f}{bucket['mfe'].mean():>+9.1f}%{bucket['mae'].mean():>+9.1f}%")


def analyze_entry_time(df):
    """5. P&L by entry time bucket."""
    print_section("ENTRY TIME ANALYSIS (minutes after open)")
    valid = df.dropna(subset=["entry_minutes"])
    if len(valid) == 0:
        print("  No entry time data available.")
        return

    bins = [0, 30, 60, 120, 195, float("inf")]
    labels = ["0-30m", "30-60m", "60-120m", "120-195m", "195m+"]
    valid = valid.copy()
    valid["time_bucket"] = pd.cut(valid["entry_minutes"], bins=bins, labels=labels, right=False)

    print(f"\n  {'Time Bucket':<14}{'Trades':>8}{'Win%':>8}{'Avg P&L':>12}{'Total P&L':>14}{'Avg Gap%':>10}")
    print(f"  {'-'*66}")
    for label in labels:
        bucket = valid[valid["time_bucket"] == label]
        if len(bucket) == 0:
            continue
        wr = (bucket["pnl"] > 0).mean() * 100
        print(f"  {label:<14}{len(bucket):>8}{wr:>7.1f}%  ${bucket['pnl'].mean():>+9.2f}  ${bucket['pnl'].sum():>+11.2f}{bucket['gap_pct'].mean():>+9.1f}%")


def analyze_mfe_mae(df):
    """6. MFE/MAE analysis — how deep do winners draw down? How high do losers get?"""
    print_section("MFE / MAE ANALYSIS")

    winners = df[df["pnl"] > 0]
    losers = df[df["pnl"] <= 0]

    print(f"\n  WINNERS ({len(winners)} trades):")
    if len(winners) > 0:
        print(f"    Avg MFE (max unrealized profit): {winners['mfe'].mean():>+.1f}%")
        print(f"    Avg MAE (max drawdown before win): {winners['mae'].mean():>+.1f}%")
        print(f"    Median MFE: {winners['mfe'].median():>+.1f}%")
        print(f"    Median MAE: {winners['mae'].median():>+.1f}%")
        print(f"    Max MFE: {winners['mfe'].max():>+.1f}%  |  Worst MAE: {winners['mae'].min():>+.1f}%")

    print(f"\n  LOSERS ({len(losers)} trades):")
    if len(losers) > 0:
        print(f"    Avg MFE (how high before reversing): {losers['mfe'].mean():>+.1f}%")
        print(f"    Avg MAE (how deep they fell): {losers['mae'].mean():>+.1f}%")
        print(f"    Median MFE: {losers['mfe'].median():>+.1f}%")
        print(f"    Median MAE: {losers['mae'].median():>+.1f}%")
        print(f"    Max MFE: {losers['mfe'].max():>+.1f}%  |  Worst MAE: {losers['mae'].min():>+.1f}%")

    # MFE distribution for losers (what % of losers had >X% unrealized profit?)
    if len(losers) > 0:
        print(f"\n  LOSERS: Unrealized profit they had before losing:")
        for threshold in [5, 10, 15, 20, 30]:
            count = (losers["mfe"] >= threshold).sum()
            pct = count / len(losers) * 100
            print(f"    Had >{threshold:>2}% unrealized profit: {count:>4} ({pct:.1f}%)")

    # MAE distribution for winners (what % of winners drew down >X%?)
    if len(winners) > 0:
        print(f"\n  WINNERS: How deep they drew down before winning:")
        for threshold in [-3, -5, -8, -10, -15]:
            count = (winners["mae"] <= threshold).sum()
            pct = count / len(winners) * 100
            print(f"    Drew down >{abs(threshold):>2}%: {count:>4} ({pct:.1f}%)")


def analyze_profit_left(df):
    """7. Unrealized profit left on table for trailing stop and EOD exits."""
    print_section("PROFIT LEFT ON TABLE")
    trail_exits = df[df["exit_reason"].isin(["TRAIL_STOP", "PARTIAL+EOD", "EOD_CLOSE"])]
    if len(trail_exits) == 0:
        print("  No trailing/EOD exits to analyze.")
        return

    for reason in ["TRAIL_STOP", "PARTIAL+EOD", "EOD_CLOSE"]:
        subset = trail_exits[trail_exits["exit_reason"] == reason]
        if len(subset) == 0:
            continue
        avg_left = subset["profit_left_on_table"].mean()
        med_left = subset["profit_left_on_table"].median()
        max_left = subset["profit_left_on_table"].max()
        print(f"\n  {reason} ({len(subset)} trades):")
        print(f"    Avg profit left on table: {avg_left:>+.1f}%")
        print(f"    Median: {med_left:>+.1f}%  |  Max: {max_left:>+.1f}%")


def analyze_confirmation_type(df):
    """8. Entry confirmation: pullback vs timeout."""
    print_section("ENTRY CONFIRMATION TYPE")
    valid = df.dropna(subset=["confirm_type"])
    if len(valid) == 0:
        print("  No confirmation type data.")
        return

    print(f"\n  {'Type':<14}{'Trades':>8}{'Win%':>8}{'Avg P&L':>12}{'Total P&L':>14}")
    print(f"  {'-'*56}")
    for ctype in ["pullback", "timeout"]:
        subset = valid[valid["confirm_type"] == ctype]
        if len(subset) == 0:
            continue
        wr = (subset["pnl"] > 0).mean() * 100
        print(f"  {ctype:<14}{len(subset):>8}{wr:>7.1f}%  ${subset['pnl'].mean():>+9.2f}  ${subset['pnl'].sum():>+11.2f}")


def analyze_candles_in_trade(df):
    """9. Time in trade analysis (candles)."""
    print_section("TIME IN TRADE (2-min candles)")
    traded = df[df["candles_in_trade"] > 0]
    if len(traded) == 0:
        return

    bins = [0, 15, 30, 60, 100, 195, float("inf")]
    labels = ["0-15", "15-30", "30-60", "60-100", "100-195", "195+"]
    traded = traded.copy()
    traded["candle_bucket"] = pd.cut(traded["candles_in_trade"], bins=bins, labels=labels, right=False)

    print(f"\n  {'Candles':<12}{'Trades':>8}{'Win%':>8}{'Avg P&L':>12}{'Total P&L':>14}")
    print(f"  {'-'*54}")
    for label in labels:
        bucket = traded[traded["candle_bucket"] == label]
        if len(bucket) == 0:
            continue
        wr = (bucket["pnl"] > 0).mean() * 100
        print(f"  {label:<12}{len(bucket):>8}{wr:>7.1f}%  ${bucket['pnl'].mean():>+9.2f}  ${bucket['pnl'].sum():>+11.2f}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Determine which data directories to use
    if len(sys.argv) > 1:
        data_dirs = sys.argv[1:]
    else:
        data_dirs = ["stored_data", "stored_data_oos"]

    print(f"Trade Analysis — Phase 1")
    print(f"Data directories: {data_dirs}")
    print(f"Config: Stop={STOP_LOSS_PCT}% | Sell {PARTIAL_SELL_FRAC*100:.0f}%@+{PARTIAL_SELL_PCT}% | "
          f"ATR({ATR_PERIOD})x{ATR_MULTIPLIER} | Confirm {CONFIRM_ABOVE}/{CONFIRM_WINDOW} | "
          f"PB {PULLBACK_PCT}%/t{PULLBACK_TIMEOUT}")
    print(f"NOTE: No volume filter applied — all candidates included for analysis\n")

    # Load picks from all directories
    daily_picks = load_all_picks(data_dirs)
    all_dates = sorted(daily_picks.keys())
    print(f"\nTotal: {len(all_dates)} trading days ({all_dates[0]} to {all_dates[-1]})")

    # Simulate all days
    print(f"\nSimulating trades...")
    all_trades = []
    for idx, date_str in enumerate(all_dates):
        sys.stdout.write(f"\r  [{idx+1}/{len(all_dates)}] {date_str}...")
        sys.stdout.flush()
        picks = daily_picks[date_str]
        if not picks:
            continue
        records = simulate_day_enriched(picks)
        for r in records:
            r["date"] = date_str
        all_trades.extend(records)

    print(f"\r  Simulated {len(all_dates)} days, {len(all_trades)} trades entered.\n")

    # Build DataFrame
    df = pd.DataFrame(all_trades)

    if len(df) == 0:
        print("No trades found! Check your data.")
        sys.exit(1)

    # Print overall summary
    print_section("OVERALL SUMMARY")
    total_pnl = df["pnl"].sum()
    winners = (df["pnl"] > 0).sum()
    losers = (df["pnl"] <= 0).sum()
    win_rate = winners / len(df) * 100
    avg_win = df[df["pnl"] > 0]["pnl"].mean() if winners > 0 else 0
    avg_loss = df[df["pnl"] <= 0]["pnl"].mean() if losers > 0 else 0
    print(f"  Total Trades: {len(df)}")
    print(f"  Winners: {winners} ({win_rate:.1f}%)  |  Losers: {losers}")
    print(f"  Total P&L: ${total_pnl:+,.2f}")
    print(f"  Avg Win: ${avg_win:+,.2f}  |  Avg Loss: ${avg_loss:+,.2f}")
    if losers > 0 and avg_loss != 0:
        pf = abs(avg_win * winners / (avg_loss * losers))
        print(f"  Profit Factor: {pf:.2f}")
    print(f"  Days: {len(all_dates)}  |  Avg trades/day: {len(df)/len(all_dates):.1f}")

    # Run all analyses
    analyze_exit_types(df)
    analyze_partial_sells(df)
    analyze_volume_buckets(df)
    analyze_gap_buckets(df)
    analyze_entry_time(df)
    analyze_mfe_mae(df)
    analyze_profit_left(df)
    analyze_confirmation_type(df)
    analyze_candles_in_trade(df)

    print(f"\n{'='*70}")
    print(f"  Analysis complete. {len(df)} trades across {len(all_dates)} days.")
    print(f"{'='*70}")
