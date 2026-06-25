"""Backtest #124 on TODAY's actual market data (2026-06-18).

Pulls intraday bars for today's live-bot watchlist from Alpaca SIP
(historical access works for data >15min old), constructs a picks list
matching the backtester's expected format, then runs simulate_day_combined
with #124's params to see what the bot WOULD have done if the IEX feed
issue + bracket bug weren't blocking it.

Compares to actual live trade (APWC -$19.34) and projects what the day
should have looked like.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import MARGIN_THRESHOLD

# Today's watchlist from the live bot log @ 09:30:00 ET
# Format: (ticker, gap_pct, pm_volume, float_shares_M, pm_high)
WATCHLIST = [
    ("ADTX", 270.4, 894_865_803, 0.8, None),
    ("CAST", 129.1,  21_249_623,16.3, None),
    ("CDT",   93.3,  44_675_516, 1.1, None),
    ("LNKS",  86.3,  12_476_132, 1.3, None),
    ("WKSP",  59.7,  47_702_700,11.7, None),
    ("BYAH",  39.2,   3_670_585, 1.0, 1.79),
    ("LPA",   39.1,   2_473_835, 3.1, 4.37),
    ("APWC",  37.4,  15_589_182, 6.6, 1.94),
    ("WOK",   31.2,   1_204_484, 1.3, 7.18),
    ("GPUS",  28.5,  66_333_834, 457.8, 0.51),
    ("BFLY",  26.4,     761_909, 206.6, 7.21),
    ("CRVO",  22.3,   7_580_083, 5.3, 5.35),
    ("CTNT",  22.3,   3_228_761, 2.8, 1.90),
    ("QTEX",  21.6,  10_836_308, 44.8, 2.37),
    ("WPRT",  21.0,  18_185_359, 15.4, 2.58),
    ("ATPC",  18.3,   1_656_654, 1.0, 3.18),
    ("SPRO",  17.5,   1_328_438, 44.9, 2.48),
    ("BIRD",  17.3,   2_163_960, 5.8, 6.57),
    ("RUM",   16.5,     380_739, 329.5, 8.49),
    ("AVD",   15.8,         162, 26.6, 3.00),
]

ET = ZoneInfo("America/New_York")
TODAY = datetime.now(ET).strftime("%Y-%m-%d")
PARENT_BASE = "config/trial_432_params.json"
TRIAL_124 = "config/trial_124_microcap_pump_extracted.json"
STARTING_CASH = 25_000


def _merged(params_in):
    with open(PARENT_BASE) as f: b = json.load(f)
    m = dict(b); m.update(params_in); return m


def fetch_bars_for_today(client, ticker):
    """Get 2-min bars from 9:30 ET to now for ticker."""
    start_et = ET.localize(datetime.now(ET).replace(hour=9, minute=30, second=0, microsecond=0)) \
        if hasattr(ET, "localize") else datetime.now(ET).replace(hour=9, minute=30, second=0, microsecond=0)
    # current time minus 20 min so all bars are accessible (SIP delayed access)
    end_et = datetime.now(ET) - timedelta(minutes=20)
    start_utc = start_et.astimezone(timezone.utc)
    end_utc = end_et.astimezone(timezone.utc)

    try:
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame(2, "Minute") if False else TimeFrame.Minute,
            start=start_utc, end=end_utc,
            feed=DataFeed.SIP,
        )
        bars = client.get_stock_bars(req)
        df = bars.df.reset_index()
        if len(df) == 0:
            return None
        # Convert 1-min to 2-min
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index).tz_convert(ET)
        # Resample to 2-min
        df_2 = df.resample("2min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum"
        }).dropna()
        df_2.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df_2
    except Exception as e:
        print(f"  [warn] {ticker}: {e}")
        return None


def main():
    print(f"=== Backtesting #124 on TODAY ({TODAY}) ===\n")

    client = StockHistoricalDataClient(
        "PKIPXFIETM7H4BAGQ64FQV3IWJ",
        "25RY682kuN9EBcFr6SFxgbSpFjdZkn613PLKPy1TdYzG"
    )

    # Load #124 config
    with open(TRIAL_124) as f: cfg = json.load(f)
    set_strategy_params(_merged(cfg["params"]))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    # Build picks from watchlist + Alpaca SIP intraday bars
    picks = []
    print(f"Fetching intraday bars for {len(WATCHLIST)} watchlist tickers...")
    for ticker, gap, pm_vol, float_m, pm_high in WATCHLIST:
        print(f"  {ticker}...", end=" ")
        df = fetch_bars_for_today(client, ticker)
        if df is None or len(df) == 0:
            print("no bars")
            continue
        prev_close = (pm_high or 0) / (1 + gap/100) if pm_high else df["Open"].iloc[0] / (1 + gap/100)
        picks.append({
            "ticker": ticker,
            "gap_pct": gap,
            "pm_volume": pm_vol,
            "premarket_high": pm_high or 0,
            "prev_close": prev_close,
            "market_hour_candles": df,
        })
        print(f"{len(df)} 2-min bars")

    print(f"\n{len(picks)} picks ready. Running simulate_day_combined...\n")

    cash = STARTING_CASH
    states, end_c, unset, _ = tgc.simulate_day_combined(picks, cash, cash_account=False)

    # Report results
    trades = [s for s in states if s.get("exit_reason") is not None and s.get("position_cost", 0) > 0]
    final = end_c + unset
    pnl = final - STARTING_CASH

    print(f"=== Backtest result for #124 on {TODAY} ===")
    print(f"  Starting cash: ${STARTING_CASH:,}")
    print(f"  Final equity:  ${final:,.0f}")
    print(f"  PnL:           ${pnl:+,.0f}  ({pnl/STARTING_CASH*100:+.2f}%)")
    print(f"  Trades:        {len(trades)}")
    if trades:
        print()
        print(f"  {'ticker':<6} {'strategy':<8} {'entry':>8} {'exit':>8} {'$cost':>10} {'$pnl':>10} {'pnl%':>7} {'exit_reason':<14}")
        for s in trades:
            ep = s.get("entry_price") or 0
            xp = s.get("exit_price") or 0
            cost = s.get("position_cost") or 0
            spnl = s.get("pnl") or 0
            pct = (spnl/cost*100) if cost > 0 else 0
            shares = s.get("shares") or 0
            print(f"  {s['ticker']:<6} {s.get('strategy','?'):<8} ${ep:>7.3f} ${xp:>7.3f} {shares:>6.0f}sh ${cost:>9,.0f} ${spnl:>+9,.0f} {pct:>+6.2f}% {s.get('exit_reason','?'):<14}")

    print(f"\n=== Live actual today (PKIPX): 1 trade APWC (H), -$19.34 ===")
    print(f"=== Backtest projects ${pnl:+,.0f} if sized properly per #124's logic ===")


if __name__ == "__main__":
    main()
