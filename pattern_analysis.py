"""
Pattern Analysis Script — Phase 3
===================================
Computes intraday pattern features for each candidate ticker-day,
correlates with P&L, and identifies the strongest predictive features.

Uses precomputed picks (no volume filter) from analyze_trades.py
and runs the same simulation to get per-trade P&L.

Usage:
  python pattern_analysis.py                     # uses both data dirs
  python pattern_analysis.py stored_data         # single dir
"""
import os
import sys
import re
import pickle
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# --- CONFIG (same as analyze_trades.py / test_full.py) ---
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
    """Worker: scan tickers for one day. Returns picks with full candle data."""
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
            "premarket_candles": premarket,
            "market_hour_candles": market_hours,
        })

    candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
    return test_date, candidates[:TOP_N]


def load_picks_for_dir(data_dir):
    """Load or build picks for a directory (with full premarket candles for pattern analysis)."""
    pickle_path = os.path.join(data_dir, "pattern_picks.pkl")
    if os.path.exists(pickle_path):
        print(f"  Loading cached pattern picks from {pickle_path}...")
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

    print(f"\r    Loaded {len(daily_picks)} days.                        ")
    with open(pickle_path, "wb") as f:
        pickle.dump(daily_picks, f)
    print(f"    Cached to {pickle_path}")
    return daily_picks


def load_all_picks(data_dirs):
    merged = {}
    for d in data_dirs:
        if not os.path.isdir(d):
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
    return merged


# ─── SIMULATION (same as analyze_trades) ──────────────────────────────────────

def simulate_day_with_pnl(picks):
    """Simulate one day, return per-pick P&L dict {ticker: pnl_pct}."""
    cash = DAILY_CASH
    trade_size_base = DAILY_CASH * TRADE_PCT
    trades_taken = 0
    results = {}

    all_timestamps = set()
    for pick in picks:
        all_timestamps.update(pick["market_hour_candles"].index.tolist())
    all_timestamps = sorted(all_timestamps)

    states = []
    for pick in picks:
        states.append({
            "ticker": pick["ticker"],
            "mh": pick["market_hour_candles"],
            "premarket_high": pick["premarket_high"],
            "entry_price": None,
            "position_cost": 0.0,
            "shares": 0,
            "remaining_shares": 0,
            "total_exit_value": 0.0,
            "partial_sold": False,
            "trailing_active": False,
            "highest_since_entry": 0.0,
            "done": False,
            "recent_closes": [],
            "breakout_confirmed": False,
            "pullback_detected": False,
            "candles_since_confirm": 0,
            "true_ranges": [],
            "prev_candle_close": None,
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
                    position_size = trade_size_base if trades_taken <= 3 else DAILY_CASH * 0.10
                    if cash < position_size:
                        position_size = cash if cash > 0 else 0
                    if position_size <= 0:
                        st["done"] = True
                        trades_taken -= 1
                        continue
                    st["entry_price"] = c_close * (1 + SLIPPAGE_PCT / 100)
                    st["position_cost"] = position_size
                    st["shares"] = position_size / st["entry_price"]
                    st["remaining_shares"] = st["shares"]
                    cash -= position_size
                else:
                    continue

            if c_high > st["highest_since_entry"]:
                st["highest_since_entry"] = c_high

            if st["remaining_shares"] > 0 and not st["trailing_active"]:
                stop_price = st["entry_price"] * (1 - STOP_LOSS_PCT / 100)
                if c_low <= stop_price:
                    proceeds = st["remaining_shares"] * stop_price * (1 - SLIPPAGE_PCT / 100)
                    st["total_exit_value"] += proceeds
                    st["remaining_shares"] = 0
                    st["done"] = True
                    cash += proceeds
                    continue

            if st["remaining_shares"] > 0 and st["trailing_active"]:
                atr = sum(st["true_ranges"]) / len(st["true_ranges"]) if st["true_ranges"] else 0
                trail_stop = st["highest_since_entry"] - (atr * ATR_MULTIPLIER)
                if c_low <= trail_stop:
                    proceeds = st["remaining_shares"] * trail_stop * (1 - SLIPPAGE_PCT / 100)
                    st["total_exit_value"] += proceeds
                    st["remaining_shares"] = 0
                    st["done"] = True
                    cash += proceeds
                    continue

            if st["remaining_shares"] > 0 and not st["partial_sold"]:
                target = st["entry_price"] * (1 + PARTIAL_SELL_PCT / 100)
                if c_high >= target:
                    sell_shares = st["shares"] * PARTIAL_SELL_FRAC
                    if sell_shares > st["remaining_shares"]:
                        sell_shares = st["remaining_shares"]
                    proceeds = sell_shares * target * (1 - SLIPPAGE_PCT / 100)
                    st["total_exit_value"] += proceeds
                    st["remaining_shares"] -= sell_shares
                    st["partial_sold"] = True
                    st["trailing_active"] = True
                    cash += proceeds

            if st["remaining_shares"] <= 0:
                st["done"] = True

    for st in states:
        if st["entry_price"] is not None and st["remaining_shares"] > 0:
            last_close = float(st["mh"].iloc[-1]["Close"])
            proceeds = st["remaining_shares"] * last_close * (1 - SLIPPAGE_PCT / 100)
            st["total_exit_value"] += proceeds
            cash += proceeds

    for st in states:
        if st["entry_price"] is not None:
            cost = st["position_cost"]
            pnl = st["total_exit_value"] - cost
            pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
            results[st["ticker"]] = {"pnl": pnl, "pnl_pct": pnl_pct}

    return results


# ─── PATTERN FEATURES ─────────────────────────────────────────────────────────

def compute_pattern_features(pick):
    """Compute intraday pattern features for a single pick.
    Returns a dict of feature values."""
    mh = pick["market_hour_candles"]
    pm = pick.get("premarket_candles", pd.DataFrame())
    features = {}

    market_open = pick["market_open"]
    pm_high = pick["premarket_high"]
    prev_close = pick["prev_close"]
    gap_pct = pick["gap_pct"]
    pm_volume = pick["pm_volume"]

    features["gap_pct"] = gap_pct
    features["pm_volume"] = pm_volume
    features["pm_volume_log"] = np.log1p(pm_volume)

    # 1. First candle type (bullish/bearish, relative size)
    if len(mh) > 0:
        first = mh.iloc[0]
        first_open = float(first["Open"])
        first_close = float(first["Close"])
        first_high = float(first["High"])
        first_low = float(first["Low"])
        features["first_candle_bullish"] = 1 if first_close > first_open else 0
        first_body = abs(first_close - first_open)
        pm_range = pm_high - prev_close if pm_high > prev_close else 1.0
        features["first_candle_body_vs_pm"] = first_body / pm_range if pm_range > 0 else 0
        features["first_candle_range"] = (first_high - first_low) / first_open * 100 if first_open > 0 else 0
    else:
        features["first_candle_bullish"] = 0
        features["first_candle_body_vs_pm"] = 0
        features["first_candle_range"] = 0

    # 2. Opening range (first 15 min = ~7-8 candles of 2-min data)
    or_candles = mh.iloc[:8] if len(mh) >= 8 else mh
    if len(or_candles) > 0:
        or_high = float(or_candles["High"].max())
        or_low = float(or_candles["Low"].min())
        features["opening_range_pct"] = (or_high - or_low) / market_open * 100 if market_open > 0 else 0
        or_close = float(or_candles.iloc[-1]["Close"])
        features["opening_range_breakout_up"] = 1 if or_close > (or_high + or_low) / 2 else 0
    else:
        features["opening_range_pct"] = 0
        features["opening_range_breakout_up"] = 0

    # 3. PM high vs market open ratio
    features["pm_high_vs_open"] = pm_high / market_open if market_open > 0 else 1.0

    # 4. Volume profile: first 30-min volume vs rest-of-day
    first_30_candles = mh.iloc[:15] if len(mh) >= 15 else mh
    rest_candles = mh.iloc[15:] if len(mh) > 15 else pd.DataFrame()
    vol_first_30 = float(first_30_candles["Volume"].sum()) if len(first_30_candles) > 0 else 0
    vol_rest = float(rest_candles["Volume"].sum()) if len(rest_candles) > 0 else 0
    total_mh_vol = vol_first_30 + vol_rest
    features["vol_first_30_ratio"] = vol_first_30 / total_mh_vol if total_mh_vol > 0 else 0.5

    # 5. Premarket candle count (proxy for PM activity)
    features["pm_candle_count"] = len(pm) if pm is not None and len(pm) > 0 else 0

    # 6. Gap fill attempt: does price touch prev_close within first hour?
    first_hour = mh.iloc[:30] if len(mh) >= 30 else mh
    if len(first_hour) > 0 and prev_close > 0:
        lowest_first_hour = float(first_hour["Low"].min())
        features["gap_fill_attempt"] = 1 if lowest_first_hour <= prev_close * 1.01 else 0
    else:
        features["gap_fill_attempt"] = 0

    # 7. VWAP slope: compute VWAP for first hour, check if price stays above
    if len(mh) > 5:
        prices = mh["Close"].values.astype(float)
        volumes = mh["Volume"].values.astype(float)
        cum_pv = np.cumsum(prices * volumes)
        cum_vol = np.cumsum(volumes)
        # Avoid division by zero
        cum_vol_safe = np.where(cum_vol > 0, cum_vol, 1)
        vwap = cum_pv / cum_vol_safe
        # Fraction of candles above VWAP in first 30 candles
        compare_len = min(30, len(prices))
        above_vwap = np.sum(prices[:compare_len] > vwap[:compare_len])
        features["pct_above_vwap_first_hour"] = above_vwap / compare_len
    else:
        features["pct_above_vwap_first_hour"] = 0.5

    # 8. Consolidation before breakout: range tightness in 30 min before PM high breakout
    # Use first 15 candles as proxy for pre-breakout zone
    pre_breakout = mh.iloc[:15] if len(mh) >= 15 else mh
    if len(pre_breakout) > 2:
        pb_high = float(pre_breakout["High"].max())
        pb_low = float(pre_breakout["Low"].min())
        features["pre_breakout_range_pct"] = (pb_high - pb_low) / market_open * 100 if market_open > 0 else 0
    else:
        features["pre_breakout_range_pct"] = 0

    # 9. Price relative to PM high at open
    features["open_vs_pm_high_pct"] = (market_open - pm_high) / pm_high * 100 if pm_high > 0 else 0

    # 10. Market hours volume (total)
    features["mh_volume_log"] = np.log1p(total_mh_vol)

    # 11. Gap direction momentum: first 5 candles trend
    first_5 = mh.iloc[:5] if len(mh) >= 5 else mh
    if len(first_5) >= 2:
        closes = first_5["Close"].values.astype(float)
        features["first_5_trend"] = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0
    else:
        features["first_5_trend"] = 0

    # 12. Premarket high distance from open (as % of open)
    features["pm_high_distance_pct"] = (pm_high - market_open) / market_open * 100 if market_open > 0 else 0

    return features


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dirs = sys.argv[1:]
    else:
        data_dirs = ["stored_data", "stored_data_oos"]

    print(f"Pattern Analysis — Phase 3")
    print(f"Data directories: {data_dirs}\n")

    # Load picks
    daily_picks = load_all_picks(data_dirs)
    all_dates = sorted(daily_picks.keys())
    print(f"\nTotal: {len(all_dates)} trading days ({all_dates[0]} to {all_dates[-1]})")

    # Simulate + extract features
    print(f"\nComputing features and simulating trades...")
    all_rows = []
    for idx, date_str in enumerate(all_dates):
        sys.stdout.write(f"\r  [{idx+1}/{len(all_dates)}] {date_str}...")
        sys.stdout.flush()
        picks = daily_picks[date_str]
        if not picks:
            continue

        # Get P&L for each ticker
        pnl_results = simulate_day_with_pnl(picks)

        # Compute features for each pick
        for pick in picks:
            ticker = pick["ticker"]
            features = compute_pattern_features(pick)
            features["date"] = date_str
            features["ticker"] = ticker
            if ticker in pnl_results:
                features["pnl"] = pnl_results[ticker]["pnl"]
                features["pnl_pct"] = pnl_results[ticker]["pnl_pct"]
                features["traded"] = 1
            else:
                features["pnl"] = 0
                features["pnl_pct"] = 0
                features["traded"] = 0
            all_rows.append(features)

    print(f"\r  Processed {len(all_dates)} days, {len(all_rows)} candidates.\n")

    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        print("No data found!")
        sys.exit(1)

    # Only analyze traded candidates
    traded = df[df["traded"] == 1].copy()
    print(f"Total candidates: {len(df)} | Traded: {len(traded)} | Not traded: {len(df) - len(traded)}")

    # Feature columns (numeric only, exclude metadata)
    feature_cols = [c for c in df.columns if c not in
                    ["date", "ticker", "pnl", "pnl_pct", "traded"]]

    # ─── 1. CORRELATION MATRIX ────────────────────────────────────────────────
    print_section("FEATURE CORRELATIONS WITH P&L")

    # Correlate among traded stocks
    if len(traded) > 5:
        corr_pnl = traded[feature_cols + ["pnl_pct"]].corr()["pnl_pct"].drop("pnl_pct")
        corr_sorted = corr_pnl.abs().sort_values(ascending=False)

        print(f"\n  {'Feature':<35}{'Corr w/ P&L%':>14}{'|Corr|':>10}")
        print(f"  {'-'*59}")
        for feat in corr_sorted.index:
            c = corr_pnl[feat]
            print(f"  {feat:<35}{c:>+14.4f}{abs(c):>10.4f}")

    # ─── 2. FEATURE CORRELATIONS WITH WIN/LOSS ────────────────────────────────
    print_section("FEATURE MEANS: WINNERS vs LOSERS")
    winners = traded[traded["pnl"] > 0]
    losers = traded[traded["pnl"] <= 0]

    if len(winners) > 0 and len(losers) > 0:
        print(f"\n  {'Feature':<35}{'Winners':>12}{'Losers':>12}{'Diff':>12}")
        print(f"  {'-'*71}")
        for feat in feature_cols:
            w_mean = winners[feat].mean()
            l_mean = losers[feat].mean()
            diff = w_mean - l_mean
            print(f"  {feat:<35}{w_mean:>12.3f}{l_mean:>12.3f}{diff:>+12.3f}")

    # ─── 3. FEATURE IMPORTANCE (Decision Tree + Random Forest) ────────────────
    print_section("FEATURE IMPORTANCE (ML Models)")

    if len(traded) > 20:
        X = traded[feature_cols].fillna(0).values
        y = traded["pnl_pct"].values

        # Decision Tree
        dt = DecisionTreeRegressor(max_depth=5, random_state=42)
        dt.fit(X, y)
        dt_importance = dt.feature_importances_

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_

        # Score = average of both
        combined = (dt_importance + rf_importance) / 2
        order = np.argsort(combined)[::-1]

        print(f"\n  {'Rank':<6}{'Feature':<35}{'DTree':>10}{'RForest':>10}{'Combined':>10}")
        print(f"  {'-'*71}")
        for rank, idx in enumerate(order):
            feat = feature_cols[idx]
            print(f"  {rank+1:<6}{feat:<35}{dt_importance[idx]:>10.4f}{rf_importance[idx]:>10.4f}{combined[idx]:>10.4f}")

        # ─── 4. TOP 5 STRONGEST FEATURES ──────────────────────────────────────
        print_section("TOP 5 PREDICTIVE FEATURES")
        top5_indices = order[:5]
        for rank, idx in enumerate(top5_indices):
            feat = feature_cols[idx]
            corr = corr_pnl[feat] if feat in corr_pnl else 0
            w_mean = winners[feat].mean() if len(winners) > 0 else 0
            l_mean = losers[feat].mean() if len(losers) > 0 else 0
            print(f"\n  #{rank+1}: {feat}")
            print(f"    Importance: {combined[idx]:.4f} | Corr w/ P&L: {corr:+.4f}")
            print(f"    Winners avg: {w_mean:.3f} | Losers avg: {l_mean:.3f}")

            # Show P&L by tercile
            vals = traded[feat].values
            try:
                tercile_edges = np.percentile(vals[~np.isnan(vals)], [33, 67])
                low = traded[traded[feat] <= tercile_edges[0]]
                mid = traded[(traded[feat] > tercile_edges[0]) & (traded[feat] <= tercile_edges[1])]
                high = traded[traded[feat] > tercile_edges[1]]
                print(f"    Bottom third ({len(low)} trades): avg P&L ${low['pnl'].mean():+.2f}")
                print(f"    Middle third ({len(mid)} trades): avg P&L ${mid['pnl'].mean():+.2f}")
                print(f"    Top third ({len(high)} trades): avg P&L ${high['pnl'].mean():+.2f}")
            except Exception:
                pass

    # ─── 5. RECOMMENDED FILTERS ──────────────────────────────────────────────
    print_section("RECOMMENDED FILTERS")
    print("\n  Based on feature analysis, consider these filters:\n")

    # Auto-detect features where one extreme is clearly better
    if len(traded) > 20 and len(winners) > 5 and len(losers) > 5:
        recommendations = []
        for feat in feature_cols:
            try:
                vals = traded[feat].dropna().values
                if len(vals) < 10:
                    continue
                tercile_edges = np.percentile(vals, [33, 67])
                low = traded[traded[feat] <= tercile_edges[0]]
                high = traded[traded[feat] > tercile_edges[1]]
                low_pnl = low["pnl"].mean()
                high_pnl = high["pnl"].mean()
                diff = high_pnl - low_pnl
                if abs(diff) > 100:  # meaningful difference ($100+)
                    direction = "high" if diff > 0 else "low"
                    better_pnl = high_pnl if diff > 0 else low_pnl
                    worse_pnl = low_pnl if diff > 0 else high_pnl
                    threshold = tercile_edges[1] if diff > 0 else tercile_edges[0]
                    recommendations.append({
                        "feature": feat,
                        "direction": direction,
                        "diff": abs(diff),
                        "better_pnl": better_pnl,
                        "worse_pnl": worse_pnl,
                        "threshold": threshold,
                    })
            except Exception:
                continue

        recommendations.sort(key=lambda x: x["diff"], reverse=True)
        for r in recommendations[:8]:
            op = ">" if r["direction"] == "high" else "<"
            print(f"  - {r['feature']} {op} {r['threshold']:.2f}")
            print(f"    Better tercile avg: ${r['better_pnl']:+.2f} | Worse: ${r['worse_pnl']:+.2f} | Diff: ${r['diff']:.2f}")
    else:
        print("  Not enough data to generate recommendations.")

    # ─── 6. CORRELATION BETWEEN ALL FEATURES ──────────────────────────────────
    print_section("FEATURE CROSS-CORRELATIONS (top pairs)")
    if len(traded) > 10:
        corr_matrix = traded[feature_cols].corr()
        # Find highly correlated pairs
        pairs = []
        for i, f1 in enumerate(feature_cols):
            for j, f2 in enumerate(feature_cols):
                if i < j:
                    c = corr_matrix.loc[f1, f2]
                    if abs(c) > 0.5:
                        pairs.append((f1, f2, c))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if pairs:
            print(f"\n  Highly correlated feature pairs (|r| > 0.5):")
            for f1, f2, c in pairs[:10]:
                print(f"    {f1} <-> {f2}: {c:+.3f}")
        else:
            print("\n  No highly correlated feature pairs found.")

    print(f"\n{'='*70}")
    print(f"  Pattern analysis complete. {len(traded)} traded candidates analyzed.")
    print(f"{'='*70}")
