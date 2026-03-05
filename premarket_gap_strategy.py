"""
Premarket Gap-Up Breakout Strategy Backtester
==============================================
How it works (example: Monday):
  1. By 9:30 AM, scan for the top 10 premarket % gainers (small caps < $50M)
  2. For each, record its PREMARKET HIGH (highest price before 9:30 AM)
     e.g. ticker TKR had a premarket high of $4.50 at 8:30 AM
  3. During market hours (9:30-10:30), if TKR's price beats $4.50
     (e.g. hits $4.60 at 10:00 AM) -> that's the BUY
  4. Stop loss at -5% from entry price
  5. Sell 50% at +20% profit, trail remaining 50% with 5% trailing stop
  6. Hard sell anything still held at 4:00 PM (market close)
  7. Backtest over the past year

Data approach:
  - Download hourly data with prepost=True to get actual premarket candles
  - Premarket high = highest High in candles BEFORE 9:30 AM that day
  - First-hour candle (9:30-10:30) used for trade simulation
  - Gap-up % = (premarket high - previous close) / previous close
"""

import datetime
import os
import re
import warnings
import sys
import time
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
BACKTEST_DAYS = 60  # 2 months -- using 2-minute candles (max 60 days)
END_DATE = datetime.date.today()
START_DATE = END_DATE - datetime.timedelta(days=BACKTEST_DAYS)

MIN_GAP_PCT = 2.0  # minimum premarket gap-up % to consider
MIN_PM_VOLUME = 1_000_000  # minimum premarket volume on gap day
TOP_N = 10  # pick top 10 premarket gainers each day
MAX_MARKET_CAP = 500e6  # ONLY trade small caps below $500M market cap
DAILY_CASH = 10_000  # start each day with $10,000
TRADE_PCT = 0.50  # each trade uses 50% of daily starting cash ($5,000)
SLIPPAGE_PCT = 0.05  # 0.05% slippage on entry/exit
STOP_LOSS_PCT = 17.0  # 17% stop loss from entry price

# Profit-taking: sell 75% at +42%, trail the remaining 25%
PARTIAL_SELL_FRAC = 0.75  # sell 75% of position at target
PARTIAL_SELL_PCT = 42.0  # +42% target for partial sell
ATR_PERIOD = 11       # lookback candles for Average True Range
ATR_MULTIPLIER = 4.0  # trail stop = highest - ATR * multiplier
CONFIRM_ABOVE = 2  # need this many candles closing above PM high...
CONFIRM_WINDOW = 3  # ...within the last this many candles before entering
PULLBACK_PCT = 8.5  # after confirmation, wait for low within this % of PM high
PULLBACK_TIMEOUT = 14  # if no pullback within N candles after confirm, enter anyway
# anything still held gets sold at 4:00 PM (market close)


# ─── STEP 1: PULL EVERY TICKER FROM NASDAQ + NYSE ───────────────────────────────
def get_ticker_universe():
    """
    Pull the FULL list of every stock on NASDAQ and NYSE/AMEX.
    Filter to common stocks only (no ETFs, warrants, units, preferred).
    This gives us ~8000+ tickers -- the same universe a real scanner would use.
    """
    print("Pulling FULL ticker lists from NASDAQ + NYSE...")
    tickers = set()

    # 1. NASDAQ-listed stocks
    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        df = pd.read_csv(url, sep="|")
        # Filter: not a test issue, not an ETF, only normal financial status
        df = df[df["Test Issue"] == "N"]
        df = df[df["ETF"] == "N"]
        # Remove warrants, units, rights (symbols with W, U, R suffixes or special chars)
        symbols = df["Symbol"].dropna().tolist()
        # Keep only clean symbols (letters only, max 5 chars = common stock)
        clean = [s for s in symbols if s.isalpha() and len(s) <= 5]
        tickers.update(clean)
        print(f"  NASDAQ: {len(clean)} common stocks")
    except Exception as e:
        print(f"  NASDAQ fetch failed: {e}")

    # 2. NYSE / AMEX listed stocks
    try:
        url2 = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        df2 = pd.read_csv(url2, sep="|")
        df2 = df2[df2["Test Issue"] == "N"]
        df2 = df2[df2["ETF"] == "N"]
        symbols2 = df2["ACT Symbol"].dropna().tolist()
        clean2 = [s for s in symbols2 if s.isalpha() and len(s) <= 5]
        tickers.update(clean2)
        print(f"  NYSE/AMEX: {len(clean2)} common stocks")
    except Exception as e:
        print(f"  NYSE/AMEX fetch failed: {e}")

    tickers = sorted(tickers)
    print(f"  TOTAL UNIVERSE: {len(tickers)} stocks (every listed common stock)")
    return tickers


# ─── STEP 1b: FAST MARKET CAP FILTER USING BULK DOWNLOAD ─────────────────────────
def filter_by_market_cap(tickers):
    """
    Fast filter: download 5 days of daily data in bulk for ALL tickers,
    then check market cap via yfinance for only the cheapest stocks
    (price < $20 as a quick pre-filter, since >$20 stocks are rarely <$50M mcap).
    """
    print(f"\nFiltering for small caps only (< ${MAX_MARKET_CAP / 1e6:.0f}M)...")
    print(f"  Step 1: Quick price pre-filter (download recent prices)...")

    # Download last 5 days of data for ALL tickers in one bulk call
    # This is fast because yfinance batches it
    cheap_tickers = []
    batch_size = 500
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i : i + batch_size]
        pct = min(100, int((i + batch_size) / len(tickers) * 100))
        sys.stdout.write(f"\r    Downloading prices... {pct}%")
        sys.stdout.flush()
        try:
            raw = yf.download(
                chunk,
                period="5d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
            for t in chunk:
                try:
                    if len(chunk) == 1:
                        df = raw
                    else:
                        df = raw[t]
                    last_close = df["Close"].dropna().iloc[-1]
                    # Pre-filter: stocks under $100 are candidates for <$500M mcap
                    if 0.001 < float(last_close) < 100:
                        cheap_tickers.append(t)
                except Exception:
                    pass
        except Exception:
            pass

    print(f"\n    Price pre-filter: {len(cheap_tickers)} stocks under $20")

    # Step 2: Check actual market cap for the pre-filtered set
    print(f"  Step 2: Checking market caps for {len(cheap_tickers)} candidates...")
    small_caps = []
    too_big = 0
    failed = 0

    for i, t in enumerate(cheap_tickers):
        if (i + 1) % 100 == 0 or i == len(cheap_tickers) - 1:
            sys.stdout.write(
                f"\r    Checking market caps... {i + 1}/{len(cheap_tickers)}"
            )
            sys.stdout.flush()
        try:
            info = yf.Ticker(t).fast_info
            mcap = getattr(info, "market_cap", None)
            if mcap is None:
                mcap = getattr(info, "marketCap", None)
            if mcap is not None and mcap <= MAX_MARKET_CAP:
                small_caps.append(t)
            elif mcap is not None:
                too_big += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"\n  Small caps found (< ${MAX_MARKET_CAP / 1e6:.0f}M): {len(small_caps)}")
    print(f"  Filtered out (too large): {too_big}")
    print(f"  No data / failed: {failed}")

    if small_caps:
        # Show first 30
        display = small_caps[:30]
        suffix = f"... and {len(small_caps) - 30} more" if len(small_caps) > 30 else ""
        print(f"  Tickers: {', '.join(display)} {suffix}")
    return small_caps


# ─── STEP 2: DOWNLOAD 2-MINUTE DATA WITH PREMARKET (+ SAVE TO FILES) ────────────
DATA_DIR = "stored_data"  # saved as CSV files in the project directory
INTERVAL = "2m"  # 2-minute candles -- supports up to 60 days of history


def download_all_data(tickers):
    """
    Download 2-MINUTE candles with prepost=True for all tickers.
    This gives us candles every 2 minutes from 4:00 AM to 8:00 PM ET.

    Data is SAVED as CSV files in stored_data/ folder:
      - stored_data/intraday/TICKER.csv  (one file per ticker, 2-min candles)
      - stored_data/daily/TICKER.csv     (one file per ticker, daily OHLC)
    Re-runs load from these files instead of re-downloading.
    """
    import os

    intraday_dir = os.path.join(DATA_DIR, "intraday")
    daily_dir = os.path.join(DATA_DIR, "daily")
    os.makedirs(intraday_dir, exist_ok=True)
    os.makedirs(daily_dir, exist_ok=True)

    intraday = {}
    daily = {}

    # --- CHECK IF WE ALREADY HAVE STORED FILES ---
    existing_intraday = set()
    for f in os.listdir(intraday_dir):
        if f.endswith(".csv"):
            existing_intraday.add(f.replace(".csv", ""))

    tickers_to_download = [t for t in tickers if t not in existing_intraday]
    tickers_from_file = [t for t in tickers if t in existing_intraday]

    # Load existing files
    if tickers_from_file:
        print(
            f"\nLoading {len(tickers_from_file)} tickers from stored_data/intraday/..."
        )
        for t in tickers_from_file:
            try:
                fpath = os.path.join(intraday_dir, f"{t}.csv")
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                if len(df) > 0:
                    intraday[t] = df
            except Exception:
                tickers_to_download.append(t)  # re-download if file is corrupt

    # Download missing tickers
    if tickers_to_download:
        print(
            f"\nDownloading {INTERVAL} data for {len(tickers_to_download)} new tickers (prepost=True)..."
        )
        print(f"  Period: {START_DATE} to {END_DATE} ({BACKTEST_DAYS} days)")

        batch_size = 5
        for i in range(0, len(tickers_to_download), batch_size):
            chunk = tickers_to_download[i : i + batch_size]
            pct = min(100, int((i + batch_size) / len(tickers_to_download) * 100))
            sys.stdout.write(
                f"\r  Downloading... {pct}% ({min(i + batch_size, len(tickers_to_download))}/{len(tickers_to_download)})"
            )
            sys.stdout.flush()

            try:
                raw = yf.download(
                    chunk,
                    start=str(START_DATE),
                    end=str(END_DATE),
                    interval=INTERVAL,
                    prepost=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                for t in chunk:
                    try:
                        if len(chunk) == 1:
                            df = raw.copy()
                        else:
                            df = raw[t].copy()
                        df = df.dropna(subset=["Open", "Close", "High", "Low"])
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        if len(df) > 0:
                            intraday[t] = df
                            # Save to CSV file
                            df.to_csv(os.path.join(intraday_dir, f"{t}.csv"))
                    except Exception:
                        pass
            except Exception:
                pass

            time.sleep(0.2)

        print(
            f"\n  Downloaded and saved {len(tickers_to_download)} tickers to stored_data/intraday/"
        )

    print(f"  Total intraday data: {len(intraday)} tickers")

    # --- DAILY DATA (for previous close) ---
    existing_daily = set()
    for f in os.listdir(daily_dir):
        if f.endswith(".csv"):
            existing_daily.add(f.replace(".csv", ""))

    daily_to_download = [t for t in tickers if t not in existing_daily]
    daily_from_file = [t for t in tickers if t in existing_daily]

    if daily_from_file:
        print(f"  Loading {len(daily_from_file)} tickers from stored_data/daily/...")
        for t in daily_from_file:
            try:
                fpath = os.path.join(daily_dir, f"{t}.csv")
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                if len(df) > 0:
                    daily[t] = df
            except Exception:
                daily_to_download.append(t)

    if daily_to_download:
        print(f"  Downloading daily data for {len(daily_to_download)} tickers...")
        batch_size = 50
        for i in range(0, len(daily_to_download), batch_size):
            chunk = daily_to_download[i : i + batch_size]
            try:
                raw_daily = yf.download(
                    chunk,
                    start=str(START_DATE - datetime.timedelta(days=10)),
                    end=str(END_DATE),
                    interval="1d",
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                for t in chunk:
                    try:
                        if len(chunk) == 1:
                            df = raw_daily.copy()
                        else:
                            df = raw_daily[t].copy()
                        df = df.dropna(subset=["Close"])
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        if len(df) > 0:
                            daily[t] = df
                            df.to_csv(os.path.join(daily_dir, f"{t}.csv"))
                    except Exception:
                        pass
            except Exception:
                pass

    print(f"  Total daily data: {len(daily)} tickers")
    total_files = len(os.listdir(intraday_dir)) + len(os.listdir(daily_dir))
    print(f"  All data stored in stored_data/ ({total_files} CSV files)")
    return intraday, daily


# ─── STEP 3: EXTRACT PREMARKET HIGHS & IDENTIFY TOP GAINERS ─────────────────────
def find_daily_premarket_gainers(intraday_data, daily_data):
    """
    For each trading day, using 2-min candles:
      1. Find all premarket candles (before 9:30 AM) -> PREMARKET HIGH
      2. Compute gap % = (premarket_high - prev_close) / prev_close
      3. Rank and pick top 10 -> these are the day's premarket gainer watchlist
      4. Also collect all market-hours candles (9:30-10:30) for trade sim
    """
    print("\nExtracting premarket highs and finding daily top gainers...")

    all_dates = set()
    for t, df in intraday_data.items():
        dates = df.index.normalize().unique()
        all_dates.update(dates)
    all_dates = sorted(all_dates)

    daily_picks = {}
    days_with_picks = 0
    total_candidates = 0

    for date in all_dates:
        date_str = date.strftime("%Y-%m-%d")
        day_candidates = []

        for ticker in intraday_data:
            if _is_warrant_or_unit(ticker):
                continue
            idf = intraday_data[ticker]

            # Convert UTC timestamps to Eastern Time for correct hour comparisons
            # yfinance stores data in UTC (+00:00), but we need ET for market hours
            et_tz = ZoneInfo("America/New_York")
            if idf.index.tz is not None:
                et_index = idf.index.tz_convert(et_tz)
            else:
                et_index = idf.index.tz_localize("UTC").tz_convert(et_tz)

            day_mask = et_index.strftime("%Y-%m-%d") == date_str
            day_candles = idf[day_mask]
            et_day = et_index[day_mask]
            if len(day_candles) == 0:
                continue

            # Split into premarket (before 9:30 AM ET) and market hours (9:30 AM - 4:00 PM ET)
            pm_mask = (et_day.hour < 9) | ((et_day.hour == 9) & (et_day.minute < 30))
            mh_mask = ((et_day.hour == 9) & (et_day.minute >= 30)) | (
                (et_day.hour >= 10) & (et_day.hour < 16)
            )
            premarket = day_candles[pm_mask]
            market_hours = day_candles[mh_mask]

            if len(market_hours) == 0:
                continue

            # Market open price = first candle of market hours
            market_open = float(market_hours.iloc[0]["Open"])

            # Premarket high (buy trigger level only, NOT used for ranking)
            if len(premarket) > 0:
                premarket_high = float(premarket["High"].max())
            else:
                premarket_high = market_open

            # Premarket volume check
            pm_volume = int(premarket["Volume"].sum()) if len(premarket) > 0 else 0
            if pm_volume < MIN_PM_VOLUME:
                continue

            # Previous close
            prev_close = None
            if ticker in daily_data:
                ddf = daily_data[ticker]
                date_naive = pd.Timestamp(date_str)
                ddf_dates = ddf.index.tz_localize(None) if ddf.index.tz else ddf.index
                prev_mask = ddf_dates < date_naive
                if prev_mask.any():
                    prev_idx = ddf.index[prev_mask][-1]
                    prev_close = float(ddf.loc[prev_idx, "Close"])

            if prev_close is None or prev_close <= 0:
                continue

            # Gap % = how much the stock gapped up at the OPEN vs prev close
            # This is what determines the top 10 premarket gainers ranking
            gap_pct = (market_open - prev_close) / prev_close * 100
            if gap_pct < MIN_GAP_PCT:
                continue

            day_candidates.append(
                {
                    "ticker": ticker,
                    "gap_pct": gap_pct,
                    "premarket_high": premarket_high,
                    "market_open": market_open,
                    "prev_close": prev_close,
                    "market_hour_candles": market_hours,
                }
            )

        if day_candidates:
            day_candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
            top_picks = day_candidates[:TOP_N]
            daily_picks[date] = top_picks
            days_with_picks += 1
            total_candidates += len(top_picks)

    print(f"  Trading days with qualifying gap-ups: {days_with_picks}")
    print(f"  Total trade candidates: {total_candidates}")

    # Save daily top 10 premarket gainers to CSV for future reference
    gainers_rows = []
    for date, picks in sorted(daily_picks.items()):
        for rank, pick in enumerate(picks, 1):
            gainers_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "rank": rank,
                    "ticker": pick["ticker"],
                    "gap_pct": round(pick["gap_pct"], 2),
                    "market_open": round(pick["market_open"], 4),
                    "premarket_high": round(pick["premarket_high"], 4),
                    "prev_close": round(pick["prev_close"], 4),
                }
            )
    if gainers_rows:
        gainers_df = pd.DataFrame(gainers_rows)
        gainers_path = os.path.join(DATA_DIR, "daily_top_gainers.csv")
        gainers_df.to_csv(gainers_path, index=False)
        print(f"  Saved daily top gainers to {gainers_path}")

    return daily_picks


# ─── STEP 4: SIMULATE TRADES (CANDLE BY CANDLE WITH CASH MANAGEMENT) ────────────
def simulate_trades(daily_picks):
    """
    Walk through every 2-minute candle in the first hour to simulate trades.
    Cash management:
      - Start each day with DAILY_CASH ($10,000)
      - Each trade uses TRADE_PCT (30%) of daily starting amount = $3,000
      - Max 3 trades at 30% each; a 4th trade gets the remaining 10%
      - When a position is sold, proceeds return to available cash
      - No cash available = trade skipped
    Trades are processed candle-by-candle across all picks simultaneously
    so that sells on one stock free up cash for another.
    """
    print("\nSimulating trades (candle-by-candle through market hours)...")
    trades = []
    trade_size_base = DAILY_CASH * TRADE_PCT  # $3,000 per trade

    for date, picks in daily_picks.items():
        # --- DAILY CASH RESET ---
        cash = DAILY_CASH
        trades_taken_today = 0

        # Collect all unique candle timestamps across all picks for this day
        all_timestamps = set()
        for pick in picks:
            all_timestamps.update(pick["market_hour_candles"].index.tolist())
        all_timestamps = sorted(all_timestamps)

        # Track state per pick
        states = []
        for pick in picks:
            states.append(
                {
                    "ticker": pick["ticker"],
                    "premarket_high": pick["premarket_high"],
                    "gap_pct": pick["gap_pct"],
                    "mh_candles": pick["market_hour_candles"],
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
                    "recent_closes": [],  # track last N candle closes vs PM high
                    "breakout_confirmed": False,  # True after 2/3 candles above PM high
                    "pullback_detected": False,   # True after price pulls back near PM high
                    "candles_since_confirm": 0,   # timeout counter for Phase 2
                    "true_ranges": [],    # rolling TR values for ATR calculation
                    "prev_candle_close": None,  # previous candle close for TR
                }
            )

        # --- WALK THROUGH EVERY 2-MIN CANDLE IN ORDER ---
        for ts in all_timestamps:
            for st in states:
                if st["done"]:
                    continue

                # Skip if this pick doesn't have a candle at this timestamp
                if ts not in st["mh_candles"].index:
                    continue

                candle = st["mh_candles"].loc[ts]
                c_open = float(candle["Open"])
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

                premarket_high = st["premarket_high"]

                # --- NOT YET IN TRADE: 3-phase entry ---
                if st["entry_price"] is None:
                    entered = False

                    # Phase 1: confirmation candles (2/3 above PM high)
                    if not st["breakout_confirmed"]:
                        st["recent_closes"].append(c_close > premarket_high)
                        if len(st["recent_closes"]) > CONFIRM_WINDOW:
                            st["recent_closes"] = st["recent_closes"][-CONFIRM_WINDOW:]
                        above_count = sum(st["recent_closes"])
                        if above_count >= CONFIRM_ABOVE:
                            st["breakout_confirmed"] = True

                    # Phase 2: wait for pullback near PM high (or timeout)
                    elif not st["pullback_detected"]:
                        st["candles_since_confirm"] += 1
                        pullback_zone = premarket_high * (1 + PULLBACK_PCT / 100)
                        if c_low <= pullback_zone:
                            st["pullback_detected"] = True
                            if c_close > premarket_high:
                                entered = True
                        elif st["candles_since_confirm"] >= PULLBACK_TIMEOUT:
                            # Timeout: no pullback came, enter on next close above PM high
                            if c_close > premarket_high:
                                entered = True

                    # Phase 3: wait for bounce (close above PM high after pullback)
                    else:
                        if c_close > premarket_high:
                            entered = True

                    if entered:
                        trades_taken_today += 1
                        if trades_taken_today <= 3:
                            position_size = trade_size_base
                        else:
                            position_size = DAILY_CASH * 0.10

                        if cash < position_size:
                            if cash > 0:
                                position_size = cash
                            else:
                                st["exit_reason"] = "NO_CASH"
                                st["done"] = True
                                trades_taken_today -= 1
                                continue

                        fill_price = c_close
                        st["entry_price"] = fill_price * (1 + SLIPPAGE_PCT / 100)
                        st["position_cost"] = position_size
                        st["shares"] = position_size / st["entry_price"]
                        st["remaining_shares"] = st["shares"]
                        st["entry_time"] = ts
                        cash -= position_size
                    else:
                        continue

                # --- IN TRADE: update highest price since entry ---
                if c_high > st["highest_since_entry"]:
                    st["highest_since_entry"] = c_high

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

        # --- END OF DAY: sell anything still held at market close ---
        for st in states:
            if st["entry_price"] is not None and st["remaining_shares"] > 0:
                last_close = float(st["mh_candles"].iloc[-1]["Close"])
                sell_price = last_close * (1 - SLIPPAGE_PCT / 100)
                proceeds = st["remaining_shares"] * sell_price
                st["total_exit_value"] += proceeds
                st["remaining_shares"] = 0
                if st["exit_reason"] not in ("STOP_LOSS", "TRAIL_STOP"):
                    st["exit_reason"] = "EOD_CLOSE"
                st["exit_price"] = last_close
                cash += proceeds

        # --- RECORD ALL TRADES FOR THIS DAY ---
        for st in states:
            if st["entry_price"] is not None:
                cost = st["position_cost"]
                pnl_dollar = st["total_exit_value"] - cost
                pnl_pct = (st["total_exit_value"] / cost - 1) * 100 if cost > 0 else 0
                trades.append(
                    {
                        "date": date,
                        "ticker": st["ticker"],
                        "gap_pct": round(st["gap_pct"], 2),
                        "premarket_high": round(st["premarket_high"], 4),
                        "entry_price": round(st["entry_price"], 4),
                        "exit_price": round(st["exit_price"], 4)
                        if st["exit_price"]
                        else None,
                        "position_size": round(cost, 2),
                        "pnl_pct": round(pnl_pct, 4),
                        "pnl_dollar": round(pnl_dollar, 2),
                        "status": "TRADED",
                        "exit_reason": st["exit_reason"],
                    }
                )
            else:
                status = "NO_CASH" if st["exit_reason"] == "NO_CASH" else "NO_BREAKOUT"
                trades.append(
                    {
                        "date": date,
                        "ticker": st["ticker"],
                        "gap_pct": round(st["gap_pct"], 2),
                        "premarket_high": round(st["premarket_high"], 4),
                        "entry_price": None,
                        "exit_price": None,
                        "position_size": 0,
                        "pnl_pct": 0.0,
                        "pnl_dollar": 0.0,
                        "status": status,
                        "exit_reason": st["exit_reason"],
                    }
                )

    trades_df = pd.DataFrame(trades)
    traded = trades_df[trades_df["status"] == "TRADED"]
    no_breakout = trades_df[trades_df["status"] == "NO_BREAKOUT"]
    no_cash = trades_df[trades_df["status"] == "NO_CASH"]
    print(f"  Total candidates: {len(trades_df)}")
    print(f"  Trades executed (price beat premarket high): {len(traded)}")
    print(f"  No breakout (price stayed below premarket high): {len(no_breakout)}")
    if len(no_cash) > 0:
        print(f"  Skipped (no cash available): {len(no_cash)}")
    return trades_df


# ─── STEP 5: CALCULATE RESULTS ──────────────────────────────────────────────────
def calculate_results(trades_df):
    """Compute strategy performance metrics."""
    traded = trades_df[trades_df["status"] == "TRADED"].copy()

    if len(traded) == 0:
        print(
            "\nNo trades were executed. Try lowering MIN_GAP_PCT or expanding universe."
        )
        return None

    total_trades = len(traded)
    winners = traded[traded["pnl_dollar"] > 0]
    losers = traded[traded["pnl_dollar"] <= 0]
    win_rate = len(winners) / total_trades * 100

    total_pnl = traded["pnl_dollar"].sum()
    avg_pnl = traded["pnl_dollar"].mean()
    avg_pnl_pct = traded["pnl_pct"].mean()
    avg_win = winners["pnl_dollar"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl_dollar"].mean() if len(losers) > 0 else 0
    best_trade = traded["pnl_dollar"].max()
    worst_trade = traded["pnl_dollar"].min()

    gross_profit = winners["pnl_dollar"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_dollar"].sum()) if len(losers) > 0 else 0.01
    profit_factor = gross_profit / gross_loss

    daily_pnl = traded.groupby("date")["pnl_dollar"].sum()
    cumulative = daily_pnl.cumsum()

    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    daily_returns = daily_pnl / DAILY_CASH
    sharpe = (
        np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        if daily_returns.std() > 0
        else 0
    )

    return {
        "total_trades": total_trades,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "daily_pnl": daily_pnl,
        "cumulative": cumulative,
        "traded": traded,
    }


# ─── STEP 6: DISPLAY RESULTS ────────────────────────────────────────────────────
def display_results(results, trades_df):
    """Print summary tables and generate charts with daily P&L bars."""
    if results is None:
        return None

    traded = results["traded"]

    # ─── SUMMARY TABLE ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PREMARKET GAP-UP BREAKOUT STRATEGY -- BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(
        f"  Universe: Small Caps < ${MAX_MARKET_CAP / 1e6:.0f}M | Min Gap: {MIN_GAP_PCT}%"
    )
    print(f"  Top {TOP_N} premarket gainers per day")
    print(f"  Buy trigger: price beats PREMARKET HIGH")
    print(
        f"  Stop Loss: {STOP_LOSS_PCT}% | Daily Cash: ${DAILY_CASH:,.0f} | {TRADE_PCT * 100:.0f}%/trade"
    )
    print(
        f"  Sell {int(PARTIAL_SELL_FRAC * 100)}% @ +{PARTIAL_SELL_PCT}% | Trail: ATR({ATR_PERIOD})×{ATR_MULTIPLIER} | Hard sell at 4 PM"
    )
    print("-" * 70)

    summary = [
        ["Total Trades", f"{results['total_trades']}"],
        ["Winners / Losers", f"{results['winners']} / {results['losers']}"],
        ["Win Rate", f"{results['win_rate']:.1f}%"],
        ["", ""],
        ["Total P&L", f"${results['total_pnl']:,.2f}"],
        [
            "Avg P&L per Trade",
            f"${results['avg_pnl']:,.2f} ({results['avg_pnl_pct']:.2f}%)",
        ],
        ["Avg Winner", f"${results['avg_win']:,.2f}"],
        ["Avg Loser", f"${results['avg_loss']:,.2f}"],
        ["Best Trade", f"${results['best_trade']:,.2f}"],
        ["Worst Trade", f"${results['worst_trade']:,.2f}"],
        ["", ""],
        ["Profit Factor", f"{results['profit_factor']:.2f}"],
        ["Max Drawdown", f"${results['max_drawdown']:,.2f}"],
        ["Sharpe Ratio (ann.)", f"{results['sharpe_ratio']:.2f}"],
    ]
    print(tabulate(summary, headers=["Metric", "Value"], tablefmt="simple"))
    print("=" * 70)

    # ─── TOP TRADES ─────────────────────────────────────────────────────────
    print("\nTop 10 Best Trades:")
    best = traded.nlargest(10, "pnl_dollar")[
        [
            "date",
            "ticker",
            "gap_pct",
            "premarket_high",
            "entry_price",
            "pnl_pct",
            "pnl_dollar",
        ]
    ]
    best_disp = best.copy()
    best_disp["date"] = best_disp["date"].apply(lambda x: str(x)[:10])
    best_disp.columns = [
        "Date",
        "Ticker",
        "Gap%",
        "PM High",
        "Entry",
        "Return%",
        "P&L $",
    ]
    print(
        tabulate(
            best_disp.values.tolist(),
            headers=best_disp.columns.tolist(),
            tablefmt="simple",
            floatfmt=".2f",
        )
    )

    print("\nTop 10 Worst Trades:")
    worst = traded.nsmallest(10, "pnl_dollar")[
        [
            "date",
            "ticker",
            "gap_pct",
            "premarket_high",
            "entry_price",
            "pnl_pct",
            "pnl_dollar",
        ]
    ]
    worst_disp = worst.copy()
    worst_disp["date"] = worst_disp["date"].apply(lambda x: str(x)[:10])
    worst_disp.columns = [
        "Date",
        "Ticker",
        "Gap%",
        "PM High",
        "Entry",
        "Return%",
        "P&L $",
    ]
    print(
        tabulate(
            worst_disp.values.tolist(),
            headers=worst_disp.columns.tolist(),
            tablefmt="simple",
            floatfmt=".2f",
        )
    )

    # ─── MONTHLY BREAKDOWN ──────────────────────────────────────────────────
    traded_copy = traded.copy()
    traded_copy["month"] = pd.to_datetime(traded_copy["date"]).dt.to_period("M")
    monthly = (
        traded_copy.groupby("month")
        .agg(
            trades=("pnl_dollar", "count"),
            total_pnl=("pnl_dollar", "sum"),
            avg_pnl=("pnl_dollar", "mean"),
            win_rate=("pnl_dollar", lambda x: (x > 0).sum() / len(x) * 100),
        )
        .reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)
    print("\nMonthly Breakdown:")
    print(
        tabulate(
            monthly.values.tolist(),
            headers=["Month", "Trades", "Total P&L", "Avg P&L", "Win%"],
            tablefmt="simple",
            floatfmt=".2f",
        )
    )

    # ─── EXIT REASON BREAKDOWN ──────────────────────────────────────────────
    if "exit_reason" in traded.columns:
        print("\nExit Reason Breakdown:")
        for reason in ["STOP_LOSS", "TRAIL_STOP", "EOD_CLOSE"]:
            subset = traded[traded["exit_reason"] == reason]
            if len(subset) > 0:
                avg = subset["pnl_dollar"].mean()
                print(f"  {reason:15s}: {len(subset):4d} trades, avg P&L ${avg:,.2f}")

    # ─── CHARTS ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(
        f"Premarket Gap-Up Breakout -- Small Caps < ${MAX_MARKET_CAP / 1e6:.0f}M\n"
        f"Buy when price beats premarket high | Stop: {STOP_LOSS_PCT}% | "
        f"Sell 50% @ +{PARTIAL_SELL_PCT}% | Trail: ATR({ATR_PERIOD})×{ATR_MULTIPLIER} | Hard sell 4 PM",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    # 1. DAILY P&L BAR CHART (full width)
    ax1 = fig.add_subplot(4, 1, 1)
    daily_pnl = results["daily_pnl"]
    colors = ["#4CAF50" if x > 0 else "#f44336" for x in daily_pnl.values]
    ax1.bar(
        daily_pnl.index, daily_pnl.values, color=colors, width=1.5, edgecolor="none"
    )
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Daily P&L -- Each Bar = One Trading Day", fontsize=11)
    ax1.set_ylabel("P&L ($)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(alpha=0.3, axis="y")

    # 2. EQUITY CURVE
    ax2 = fig.add_subplot(4, 2, 3)
    cum = results["cumulative"]
    ax2.plot(cum.index, cum.values, color="#2196F3", linewidth=1.5)
    ax2.fill_between(
        cum.index, 0, cum.values, where=cum.values >= 0, alpha=0.15, color="#2196F3"
    )
    ax2.fill_between(
        cum.index, 0, cum.values, where=cum.values < 0, alpha=0.15, color="#f44336"
    )
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("Equity Curve (Cumulative P&L)")
    ax2.set_ylabel("P&L ($)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(alpha=0.3)

    # 3. RETURN DISTRIBUTION
    ax3 = fig.add_subplot(4, 2, 4)
    pnl_vals = traded["pnl_pct"].values
    ax3.hist(pnl_vals, bins=40, color="#2196F3", edgecolor="white", alpha=0.8)
    ax3.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax3.axvline(
        x=np.mean(pnl_vals),
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {np.mean(pnl_vals):.2f}%",
    )
    ax3.set_title("Trade Return Distribution")
    ax3.set_xlabel("Return (%)")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. MONTHLY P&L
    ax4 = fig.add_subplot(4, 2, 5)
    monthly_pnl = monthly.set_index("month")["total_pnl"]
    mcolors = ["#4CAF50" if x > 0 else "#f44336" for x in monthly_pnl.values]
    ax4.bar(
        range(len(monthly_pnl)), monthly_pnl.values, color=mcolors, edgecolor="white"
    )
    ax4.set_xticks(range(len(monthly_pnl)))
    ax4.set_xticklabels(monthly_pnl.index, rotation=45, ha="right")
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_title("Monthly P&L")
    ax4.set_ylabel("P&L ($)")
    ax4.grid(alpha=0.3)

    # 5. WIN RATE BY GAP SIZE
    ax5 = fig.add_subplot(4, 2, 6)
    traded_copy["gap_bin"] = pd.cut(
        traded_copy["gap_pct"],
        bins=[2, 3, 5, 7, 10, 15, 100],
        labels=["2-3%", "3-5%", "5-7%", "7-10%", "10-15%", "15%+"],
    )
    gap_stats = traded_copy.groupby("gap_bin", observed=True).agg(
        win_rate=(
            "pnl_dollar",
            lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0,
        ),
        count=("pnl_dollar", "count"),
    )
    if len(gap_stats) > 0:
        x_pos = range(len(gap_stats))
        ax5.bar(
            x_pos, gap_stats["win_rate"], color="#FF9800", edgecolor="white", alpha=0.8
        )
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(gap_stats.index, rotation=0)
        for i, (wr, cnt) in enumerate(zip(gap_stats["win_rate"], gap_stats["count"])):
            ax5.text(i, wr + 1.5, f"n={cnt}", ha="center", fontsize=8)
    ax5.set_title("Win Rate by Gap-Up Size")
    ax5.set_xlabel("Premarket Gap %")
    ax5.set_ylabel("Win Rate (%)")
    ax5.set_ylim(0, 100)
    ax5.grid(alpha=0.3)

    # 6. EXIT REASON PIE
    ax6 = fig.add_subplot(4, 2, 7)
    if "exit_reason" in traded.columns:
        reason_counts = traded["exit_reason"].value_counts()
        reason_colors = {
            "STOP_LOSS": "#f44336",
            "TRAIL_STOP": "#FF9800",
            "EOD_CLOSE": "#2196F3",
        }
        pie_colors = [reason_colors.get(r, "#999") for r in reason_counts.index]
        ax6.pie(
            reason_counts.values,
            labels=reason_counts.index,
            colors=pie_colors,
            autopct="%1.0f%%",
            startangle=90,
        )
    ax6.set_title("Exit Reasons")

    # 7. TICKER FREQUENCY
    ax7 = fig.add_subplot(4, 2, 8)
    ticker_counts = traded["ticker"].value_counts().head(15)
    ticker_pnl = traded.groupby("ticker")["pnl_dollar"].sum()
    tick_colors = [
        "#4CAF50" if ticker_pnl.get(t, 0) > 0 else "#f44336"
        for t in ticker_counts.index
    ]
    ax7.barh(range(len(ticker_counts)), ticker_counts.values, color=tick_colors)
    ax7.set_yticks(range(len(ticker_counts)))
    ax7.set_yticklabels(ticker_counts.index, fontsize=8)
    ax7.set_xlabel("Number of Trades")
    ax7.set_title("Most Traded Tickers (green=profitable)")
    ax7.invert_yaxis()
    ax7.grid(alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = "backtest_results.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCharts saved to: {chart_path}")

    # ─── EXPORT TRADE LOG ───────────────────────────────────────────────────
    csv_path = "trade_log.csv"
    trades_df.to_csv(csv_path, index=False)
    print(f"Trade log saved to: {csv_path}")

    return chart_path


# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PREMARKET GAP-UP BREAKOUT STRATEGY BACKTESTER")
    print(f"  Small Caps Only (< ${MAX_MARKET_CAP / 1e6:.0f}M)")
    print(f"  Buy when price beats PREMARKET HIGH")
    print(f"  Stop Loss: {STOP_LOSS_PCT}% | Top {TOP_N} Gainers/Day")
    print("=" * 70)

    # Step 1: Check if we have a cached small cap ticker list
    ticker_cache_path = os.path.join(DATA_DIR, "small_cap_tickers.txt")
    if os.path.exists(ticker_cache_path):
        with open(ticker_cache_path, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(
            f"\nLoaded {len(tickers)} small-cap tickers from cache ({ticker_cache_path})"
        )
        print(f"  (Delete this file to force a fresh market cap scan)")
    else:
        # Step 1a: Build universe of small-cap / micro-cap tickers
        all_tickers = get_ticker_universe()

        # Step 1b: Filter to only small caps below $500M
        tickers = filter_by_market_cap(all_tickers)
        if not tickers:
            print("ERROR: No small-cap tickers found. Try raising MAX_MARKET_CAP.")
            sys.exit(1)

        # Save the filtered list for future runs
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(ticker_cache_path, "w") as f:
            f.write("\n".join(tickers))
        print(f"\n  Saved small-cap ticker list to {ticker_cache_path}")

    # Step 2: Download hourly data WITH premarket candles + daily for prev close
    hourly_data, daily_data = download_all_data(tickers)
    if not hourly_data:
        print("ERROR: No hourly data downloaded.")
        sys.exit(1)

    # Step 3: For each day, extract premarket highs and find top 10 gainers
    daily_picks = find_daily_premarket_gainers(hourly_data, daily_data)
    if not daily_picks:
        print("ERROR: No premarket gap-ups found. Try lowering MIN_GAP_PCT.")
        sys.exit(1)

    # Step 4: Simulate trades (buy when price > premarket high)
    trades_df = simulate_trades(daily_picks)

    # Step 5: Calculate and display results
    results = calculate_results(trades_df)
    chart_path = display_results(results, trades_df)

    if chart_path:
        print(f"\nDone! Open '{chart_path}' to see the visual results.")


if __name__ == "__main__":
    main()
