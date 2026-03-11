"""
Backtest Today: Fetch today's top 10 pre-market gainers from Webull,
download their 2-min bars from Alpaca, then run simulate_day_combined()
with the exact same params as the live engine (trial_432_params.json).

Usage:
  python backtest_today.py
  python backtest_today.py --date 2026-03-11   # replay a specific date (uses stored data)
"""
import sys
import json
import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

ALPACA_API_KEY = "PK34OGXUBAOCLG7E6KYTI6QMZ3"
ALPACA_API_SECRET = "DBn7mXAKdTBAR9XZnkhnu1CykZDYNZEVzkBojKDtoYbJ"
ALPACA_FEED = "iex"
MAX_PRICE = 50.0
MIN_GAP_PCT = 10.0
TOP_N = 10

PARAMS_PATH = os.path.join("config", "trial_432_params.json")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_live_params():
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    set_strategy_params(params)
    return params


def get_webull_premarket():
    """Fetch top 30 pre-market gainers from Webull."""
    url = "https://quotes-gw.webullfintech.com/api/wlas/ranking/topGainers"
    params = {"regionId": 6, "rankType": "preMarket", "pageIndex": 1, "pageSize": 30}
    resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    movers = []
    ticker_ids = {}
    for item in data.get("data", []):
        t = item.get("ticker", {})
        v = item.get("values", {})
        symbol = t.get("symbol", "")
        if not symbol or len(symbol) > 5 or "." in symbol:
            continue
        change_pct = float(v.get("changeRatio", 0)) * 100
        price = float(v.get("price", 0))
        if price <= 0 or price > MAX_PRICE or change_pct < MIN_GAP_PCT or change_pct > 500:
            continue
        ticker_id = t.get("tickerId")
        movers.append({
            "symbol": symbol,
            "price": price,
            "gap_pct": change_pct,
            "pm_volume": int(float(t.get("volume", 0))),
            "prev_close": float(t.get("preClose", 0)),
        })
        if ticker_id:
            ticker_ids[symbol] = ticker_id

    # Batch fetch float
    if ticker_ids:
        ids_str = ",".join(str(v) for v in ticker_ids.values())
        try:
            r2 = requests.get(
                "https://quotes-gw.webullfintech.com/api/bgw/quote/realtime",
                params={"ids": ids_str, "includeSecu": 1, "delay": 0, "more": 1},
                headers={"User-Agent": "Mozilla/5.0"}, timeout=10
            )
            id_to_sym = {v: k for k, v in ticker_ids.items()}
            for item in r2.json():
                sym = id_to_sym.get(item.get("tickerId"), item.get("symbol", ""))
                for m in movers:
                    if m["symbol"] == sym:
                        fs = int(float(item.get("outstandingShares", 0)))
                        ts = int(float(item.get("totalShares", 0)))
                        m["float_shares"] = fs or ts
        except Exception as e:
            print(f"  Warning: float fetch failed: {e}")

    movers.sort(key=lambda x: x["gap_pct"], reverse=True)
    return movers[:TOP_N]


def fetch_intraday_bars(tickers, trade_date):
    """Download 2-min market-hours bars for given tickers on trade_date."""
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    d = date.fromisoformat(trade_date) if isinstance(trade_date, str) else trade_date
    start = datetime.combine(d, datetime.min.time().replace(hour=9, minute=30)).replace(tzinfo=ET)
    end   = datetime.combine(d, datetime.min.time().replace(hour=16, minute=0)).replace(tzinfo=ET)

    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(2, TimeFrameUnit.Minute),
        start=start,
        end=end,
        adjustment="raw",
        feed=ALPACA_FEED,
    )
    bars = client.get_stock_bars(req)
    if bars.df.empty:
        return {}

    result = {}
    df = bars.df.reset_index()
    for ticker in df["symbol"].unique():
        tdf = df[df["symbol"] == ticker].copy()
        tdf = tdf.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        tdf = tdf.set_index("timestamp")[["Open", "High", "Low", "Close", "Volume"]]
        result[ticker] = tdf
    return result


def fetch_pm_high(tickers, trade_date):
    """Download pre-market bars (4am–9:30am) to get PM high per ticker."""
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    d = date.fromisoformat(trade_date) if isinstance(trade_date, str) else trade_date
    start = datetime.combine(d, datetime.min.time().replace(hour=4, minute=0)).replace(tzinfo=ET)
    end   = datetime.combine(d, datetime.min.time().replace(hour=9, minute=30)).replace(tzinfo=ET)

    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        adjustment="raw",
        feed=ALPACA_FEED,
    )
    try:
        bars = client.get_stock_bars(req)
        if bars.df.empty:
            return {}
        df = bars.df.reset_index()
        result = {}
        for ticker in df["symbol"].unique():
            tdf = df[df["symbol"] == ticker]
            result[ticker] = float(tdf["high"].max())
        return result
    except Exception:
        return {}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    today_str = datetime.now(ET).strftime("%Y-%m-%d")

    # Parse --date arg
    trade_date = today_str
    for arg in sys.argv[1:]:
        if arg.startswith("--date="):
            trade_date = arg.split("=", 1)[1]
        elif arg == "--date" and len(sys.argv) > sys.argv.index(arg) + 1:
            trade_date = sys.argv[sys.argv.index(arg) + 1]

    print(f"\nBacktest Today: {trade_date}")
    print("=" * 60)

    # 1. Load live params
    print("Loading live params (trial_432_params.json)...")
    params = load_live_params()
    print(f"  Loaded {len(params)} params")

    # Show enabled strategies
    enabled = []
    for s in tgc.STRAT_KEYS if hasattr(tgc, "STRAT_KEYS") else list("HGAFDVPMRWOBKCEIJNL"):
        gap_attr = f"{s}_MIN_GAP_PCT" if s != "R" else "R_DAY1_MIN_GAP"
        gap_val = getattr(tgc, gap_attr, 9999)
        if gap_val < 9000:
            enabled.append(s)
    print(f"  Enabled: {', '.join(enabled)}")

    # 2. Get candidates
    print(f"\nFetching top {TOP_N} pre-market gainers from Webull...")
    try:
        movers = get_webull_premarket()
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    if not movers:
        print("  No candidates found — market may be closed or pre-market not active.")
        sys.exit(0)

    print(f"\n  {'Ticker':<7} {'Gap%':>7} {'PM Vol':>12} {'Float':>10} {'Prev Close':>10}")
    print("  " + "-" * 50)
    for m in movers:
        vol = m.get("pm_volume", 0)
        vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K"
        fl = m.get("float_shares", 0)
        fl_str = f"{fl/1e6:.1f}M" if fl >= 1e6 else ("N/A" if not fl else str(fl))
        print(f"  {m['symbol']:<7} {m['gap_pct']:>6.1f}% {vol_str:>12} {fl_str:>10} ${m['prev_close']:>9.2f}")

    tickers = [m["symbol"] for m in movers]

    # 3. Fetch market-hours 2-min bars
    print(f"\nDownloading 2-min bars for {trade_date}...")
    bars_map = fetch_intraday_bars(tickers, trade_date)
    print(f"  Got data for {len(bars_map)}/{len(tickers)} tickers")

    if not bars_map:
        print("  ERROR: No intraday data available (market not open yet or holiday?)")
        sys.exit(0)

    # 4. Fetch PM highs
    print("Fetching pre-market highs...")
    pm_highs = fetch_pm_high(tickers, trade_date)

    # 5. Build picks in backtest format
    picks = []
    for m in movers:
        tk = m["symbol"]
        bars = bars_map.get(tk)
        if bars is None or len(bars) < 3:
            print(f"  Skipping {tk}: insufficient bar data ({0 if bars is None else len(bars)} bars)")
            continue
        picks.append({
            "ticker": tk,
            "gap_pct": m["gap_pct"],
            "market_open": float(bars.iloc[0]["Open"]),
            "premarket_high": pm_highs.get(tk, float(bars.iloc[0]["Open"])),
            "prev_close": m["prev_close"],
            "pm_volume": m.get("pm_volume", 0),
            "float_shares": m.get("float_shares"),
            "market_hour_candles": bars,
        })

    if not picks:
        print("\nNo picks with sufficient data.")
        sys.exit(0)

    print(f"\nRunning backtest on {len(picks)} tickers...")
    print()

    # 6. Run simulation
    STRAT_KEYS = ["H","G","A","F","D","V","P","M","R","W","O","B","K","C","S","E","I","J","N","L"]
    tgc.STRAT_KEYS = STRAT_KEYS  # ensure it's set

    cash = 25_000.0
    states, cash_end, unsettled, sel_log = tgc.simulate_day_combined(picks, cash, cash_account=False)

    # 7. Print results
    print(f"  {'Ticker':<7} {'Strat':>5} {'Entry':>8} {'Exit':>8} {'PnL':>12} {'%':>7} {'Reason':<12} {'Bars':>5}")
    print("  " + "-" * 70)

    traded = [s for s in states if s["exit_reason"] is not None]
    traded_tickers = {s["ticker"] for s in traded}
    # "No signal" = states that never triggered a signal at all
    no_signal = [s for s in states if s["exit_reason"] is None and not s.get("signal")]
    # Deduplicate by ticker (one per ticker is enough)
    no_signal_tickers = sorted({s["ticker"] for s in no_signal} - traded_tickers)

    day_pnl = 0.0
    wins = 0
    for st in traded:
        pct = (st["pnl"] / st["position_cost"] * 100) if st["position_cost"] > 0 else 0
        reason = {"TARGET": "TARGET", "TIME_STOP": "TIME STOP", "EOD_CLOSE": "EOD",
                  "STOP": "STOP LOSS", "TRAIL": "TRAIL STOP"}.get(st["exit_reason"], st["exit_reason"])
        n_bars = len(st["mh"])
        entry_time = st["entry_time"].astimezone(ET).strftime("%H:%M") if st.get("entry_time") else "--"
        exit_time  = st["exit_time"].astimezone(ET).strftime("%H:%M")  if st.get("exit_time")  else "--"
        print(f"  {st['ticker']:<7} {st['strategy']:>5} {entry_time:>8} {exit_time:>8}  "
              f"${st['pnl']:>+10,.0f}  ({pct:>+6.2f}%)  {reason:<12} {n_bars:>5}")
        day_pnl += st["pnl"]
        if st["pnl"] > 0:
            wins += 1

    print()
    if no_signal_tickers:
        print("  NO SIGNAL (no strategy triggered):")
        for tk in no_signal_tickers:
            pick = next((p for p in picks if p["ticker"] == tk), {})
            print(f"    {tk:<7}  gap={pick.get('gap_pct', 0):.1f}%")
    print()
    print(f"  {'─'*50}")
    print(f"  Trades:       {len(traded)} | Wins: {wins} | Losses: {len(traded)-wins}")
    wr = (wins / len(traded) * 100) if traded else 0
    print(f"  Win Rate:     {wr:.1f}%")
    print(f"  Day P&L:      ${day_pnl:+,.2f}")
    print(f"  End Equity:   ${cash_end + unsettled + day_pnl:,.2f}")
    print()


if __name__ == "__main__":
    main()
