"""
Backtest using Alpaca historical data.

Fetches intraday (2-min) and daily bars from Alpaca, builds picks,
and runs the combined strategy backtest with trial 432 params.

Usage:
  python backtest_alpaca.py                          # last 5 trading days
  python backtest_alpaca.py --days 20                # last 20 trading days
  python backtest_alpaca.py --start 2026-02-01 --end 2026-03-07
"""
import os
import sys
import re
import json
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED, FLOAT_DATA

# Import the combined strategy backtest
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params

ET = ZoneInfo("America/New_York")

# Scanner/filter settings
MIN_GAP_PCT = 8.0
MIN_PM_VOLUME = 250_000
TOP_N = 20
MAX_PRICE = 50.0
STARTING_CASH = 25_000


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


def load_trial_params(path=None):
    """Load trial 432 params and apply to tgc module."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config", "trial_432_params.json")
    with open(path) as f:
        params = json.load(f)
    set_strategy_params(params)
    enabled = []
    for s in "HGAFDVPMRWOBKCEIJNL":
        gap_attr = f"{s}_MIN_GAP_PCT" if s != "R" else "R_DAY1_MIN_GAP"
        gap_val = getattr(tgc, gap_attr, 9999)
        if gap_val < 9000:
            enabled.append(s)
    print(f"Loaded trial params: {len(params)} params, enabled: {', '.join(enabled)}")
    return params


def get_trading_days(data_client, start_date, end_date):
    """Get trading days by fetching SPY daily bars."""
    req = StockBarsRequest(
        symbol_or_symbols=["SPY"],
        timeframe=TimeFrame.Day,
        start=datetime.combine(start_date, datetime.min.time()),
        end=datetime.combine(end_date + timedelta(days=1), datetime.min.time()),
        feed=ALPACA_FEED,
    )
    bars = data_client.get_stock_bars(req)
    if bars.df.empty:
        return []
    df = bars.df.reset_index()
    dates = sorted(df["timestamp"].dt.tz_convert(ET).dt.date.unique().tolist())
    return dates


def get_all_tickers(trading_client):
    """Get tradeable US equity tickers."""
    assets = trading_client.get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    )
    tickers = [
        a.symbol for a in assets
        if a.tradable
        and not _is_warrant_or_unit(a.symbol)
        and "." not in a.symbol
        and len(a.symbol) <= 5
    ]
    return tickers


def fetch_prev_closes(data_client, tickers, date):
    """Fetch previous day's close for a list of tickers."""
    prev_closes = {}
    batch_size = 500
    start_dt = datetime.combine(date - timedelta(days=7), datetime.min.time())
    end_dt = datetime.combine(date, datetime.min.time())

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                adjustment="raw",
                feed=ALPACA_FEED,
                limit=5,
            )
            bars = data_client.get_stock_bars(req)
            if not bars.df.empty:
                df = bars.df.reset_index()
                for ticker in df["symbol"].unique():
                    tdf = df[df["symbol"] == ticker].sort_values("timestamp")
                    if len(tdf) > 0:
                        prev_closes[ticker] = float(tdf.iloc[-1]["close"])
        except Exception as e:
            pass
        time.sleep(0.25)
    return prev_closes


def fetch_intraday_bars(data_client, tickers, date):
    """Fetch 2-min intraday bars for given tickers on a specific date."""
    all_bars = {}
    batch_size = 200
    start_dt = datetime.combine(date, datetime.min.time().replace(hour=4), tzinfo=ET)
    end_dt = datetime.combine(date, datetime.min.time().replace(hour=20), tzinfo=ET)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(amount=2, unit="Min"),
                start=start_dt,
                end=end_dt,
                adjustment="raw",
                feed=ALPACA_FEED,
            )
            bars = data_client.get_stock_bars(req)
            if not bars.df.empty:
                df = bars.df.reset_index()
                for ticker in df["symbol"].unique():
                    tdf = df[df["symbol"] == ticker].sort_values("timestamp")
                    all_bars[ticker] = tdf
        except Exception as e:
            pass
        time.sleep(0.25)
        sys.stdout.write(f"\r    Intraday: {min(i + batch_size, len(tickers))}/{len(tickers)} tickers")
        sys.stdout.flush()
    return all_bars


def build_picks_for_day(intraday_bars, prev_closes):
    """Build picks list from Alpaca data for one day."""
    candidates = []

    for ticker, idf in intraday_bars.items():
        if _is_warrant_or_unit(ticker):
            continue
        if ticker not in prev_closes:
            continue

        prev_close = prev_closes[ticker]
        if prev_close <= 0 or prev_close > MAX_PRICE:
            continue

        # Convert timestamps to ET
        idf = idf.copy()
        idf["ts_et"] = pd.to_datetime(idf["timestamp"]).dt.tz_convert(ET)

        # Split premarket vs market hours
        pm_mask = (idf["ts_et"].dt.hour < 9) | ((idf["ts_et"].dt.hour == 9) & (idf["ts_et"].dt.minute < 30))
        mh_mask = ((idf["ts_et"].dt.hour == 9) & (idf["ts_et"].dt.minute >= 30)) | (
            (idf["ts_et"].dt.hour >= 10) & (idf["ts_et"].dt.hour < 16)
        )

        premarket = idf[pm_mask]
        market_hours = idf[mh_mask]

        if len(market_hours) == 0:
            continue

        market_open = float(market_hours.iloc[0]["open"])
        premarket_high = float(premarket["high"].max()) if len(premarket) > 0 else market_open
        pm_volume = int(premarket["volume"].sum()) if len(premarket) > 0 else 0

        if pm_volume < MIN_PM_VOLUME:
            continue

        gap_pct = (market_open - prev_close) / prev_close * 100
        if gap_pct < MIN_GAP_PCT:
            continue

        # Build market hours DataFrame in expected format
        mh_df = market_hours[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        mh_df.columns = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
        mh_df = mh_df.set_index("timestamp").sort_index()

        candidates.append({
            "ticker": ticker,
            "gap_pct": gap_pct,
            "market_open": market_open,
            "premarket_high": premarket_high,
            "prev_close": prev_close,
            "pm_volume": pm_volume,
            "market_hour_candles": mh_df,
        })

    candidates.sort(key=lambda x: x["gap_pct"], reverse=True)
    return candidates[:TOP_N]


def main():
    parser = argparse.ArgumentParser(description="Backtest on Alpaca historical data")
    parser.add_argument("--days", type=int, default=5, help="Number of trading days to test")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--params", type=str, help="Path to trial params JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("ALPACA HISTORICAL BACKTEST - Combined Strategy (Trial 432)")
    print("=" * 70)

    # Load trial params
    load_trial_params(args.params)

    # Set up Alpaca clients
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

    # Determine date range
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = datetime.now(ET).date() - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days * 2)

    print(f"\nFetching trading days from {start_date} to {end_date}...")
    trading_days = get_trading_days(data_client, start_date, end_date)

    if not args.start:
        trading_days = trading_days[-args.days:]

    print(f"Trading days to test: {len(trading_days)}")
    if not trading_days:
        print("No trading days found.")
        return

    # Get all tradeable tickers once
    print("\nFetching tradeable tickers...")
    all_tickers = get_all_tickers(trading_client)
    print(f"Found {len(all_tickers)} tradeable US equities")

    # Run backtest day by day
    cash = STARTING_CASH
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    daily_pnls = []
    exit_reasons = {}
    strat_stats = {}

    print(f"\nStarting backtest with ${STARTING_CASH:,} cash...")
    print("-" * 70)

    for day_num, date in enumerate(trading_days):
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n  [{day_num+1}/{len(trading_days)}] {date_str}")

        # Step 1: Get previous closes for all tickers
        print(f"    Fetching prev closes...", end="", flush=True)
        prev_closes = fetch_prev_closes(data_client, all_tickers, date)
        # Filter to tickers under MAX_PRICE
        cheap_tickers = [t for t in all_tickers if t in prev_closes and 0 < prev_closes[t] <= MAX_PRICE]
        print(f" {len(cheap_tickers)} under ${MAX_PRICE}")

        # Step 2: Fetch intraday bars for cheap tickers
        intraday = fetch_intraday_bars(data_client, cheap_tickers, date)
        print(f"\n    Got intraday for {len(intraday)} tickers")

        # Step 3: Build picks
        picks = build_picks_for_day(intraday, prev_closes)

        if not picks:
            print(f"    0 picks, skipping")
            daily_pnls.append(0)
            continue

        print(f"    {len(picks)} picks: {', '.join(p['ticker'] + f'({p[\"gap_pct\"]:.0f}%)' for p in picks[:8])}"
              + ("..." if len(picks) > 8 else ""))

        # Step 4: Run combined strategy simulation
        states, new_cash, unsettled, _ = tgc.simulate_day_combined(picks, cash)
        cash = new_cash + unsettled

        # Tally results
        day_pnl = 0.0
        day_trades = 0
        day_wins = 0
        day_tickers = []

        for st in states:
            if st.get("exit_price") is not None:
                pnl = st.get("pnl", 0)
                day_pnl += pnl
                day_trades += 1
                if pnl > 0:
                    day_wins += 1
                day_tickers.append(f"{st['ticker']}({st.get('strategy','?')})")

                reason = st.get("exit_reason", "UNKNOWN")
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

                strat = st.get("strategy", "?")
                if strat not in strat_stats:
                    strat_stats[strat] = {"n": 0, "pnl": 0.0, "wins": 0}
                strat_stats[strat]["n"] += 1
                strat_stats[strat]["pnl"] += pnl
                if pnl > 0:
                    strat_stats[strat]["wins"] += 1

        total_trades += day_trades
        total_wins += day_wins
        total_pnl += day_pnl
        daily_pnls.append(day_pnl)

        print(f"    >> {day_trades} trades, PnL=${day_pnl:+,.0f}, cash=${cash:,.0f}"
              + (f" | {', '.join(day_tickers[:5])}" if day_tickers else ""))

    # Summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"  Total PnL:     ${total_pnl:+,.0f}")
    print(f"  Final equity:  ${cash:,.0f}")
    print(f"  Trades:        {total_trades}")
    print(f"  Win rate:      {wr:.1f}%")
    print(f"  Trading days:  {len(trading_days)}")

    if daily_pnls:
        green = sum(1 for p in daily_pnls if p > 0)
        print(f"  Green days:    {green}/{len(daily_pnls)} ({green/len(daily_pnls)*100:.1f}%)")

        pnl_arr = np.array(daily_pnls)
        if pnl_arr.std() > 0:
            sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)
            print(f"  Sharpe:        {sharpe:.2f}")

    if strat_stats:
        print(f"\n  Strategy breakdown:")
        for strat in sorted(strat_stats.keys()):
            s = strat_stats[strat]
            swr = (s["wins"] / s["n"] * 100) if s["n"] > 0 else 0
            print(f"    {strat}: {s['n']} trades, ${s['pnl']:+,.0f}, {swr:.0f}% WR")

    if exit_reasons:
        print(f"\n  Exit reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
