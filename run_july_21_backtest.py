import json
import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add current directory to path
sys.path.insert(0, ".")

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params, _build_param_snapshot, _param_lock

API_KEY = "o9vwR8VA1nPulvAo9A5Dk8FGvoJV"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

# The 20 watchlist candidates for 2026-07-21
CANDIDATES = [
    {
        "ticker": "VIVK",
        "gap_pct": 168.77,
        "pm_volume": 8387101,
        "premarket_high": 5.48,
        "prev_close": 1.73,
        "float_shares": 404703,
    },
    {
        "ticker": "KIDZ",
        "gap_pct": 135.79,
        "pm_volume": 80835171,
        "premarket_high": 1.17,
        "prev_close": 0.4121,
        "float_shares": 1121097,
    },
    {
        "ticker": "SLGB",
        "gap_pct": 119.98,
        "pm_volume": 45772697,
        "premarket_high": 1.32,
        "prev_close": 0.5091,
        "float_shares": 5000000,
    },
    {
        "ticker": "UTZ",
        "gap_pct": 88.87,
        "pm_volume": 13070765,
        "premarket_high": 14.07,
        "prev_close": 7.45,
        "float_shares": 72083785,
    },
    {
        "ticker": "HIHO",
        "gap_pct": 46.3,
        "pm_volume": 14543298,
        "premarket_high": 1.38,
        "prev_close": 0.9296,
        "float_shares": 3049609,
    },
    {
        "ticker": "JUNS",
        "gap_pct": 36.0,
        "pm_volume": 124106835,
        "premarket_high": 0.228,
        "prev_close": 0.165,
        "float_shares": 30121940,
    },
    {
        "ticker": "FWRD",
        "gap_pct": 17.12,
        "pm_volume": 41584,
        "premarket_high": 15.46,
        "prev_close": 13.2,
        "float_shares": 21792171,
    },
    {
        "ticker": "RPT",
        "gap_pct": 16.65,
        "pm_volume": 50973,
        "premarket_high": 12.4,
        "prev_close": 10.63,
        "float_shares": 7759875,
    },
    {
        "ticker": "PKE",
        "gap_pct": 13.2,
        "pm_volume": 6248,
        "premarket_high": 36.97,
        "prev_close": 32.66,
        "float_shares": 19791921,
    },
    {
        "ticker": "LASE",
        "gap_pct": 12.87,
        "pm_volume": 6939634,
        "premarket_high": 1.26,
        "prev_close": 1.09,
        "float_shares": 35483946,
    },
    {
        "ticker": "NIKI",
        "gap_pct": 12.86,
        "pm_volume": 6043,
        "premarket_high": 7.9,
        "prev_close": 7.0,
        "float_shares": 530529,
    },
    {
        "ticker": "AKTX",
        "gap_pct": 12.48,
        "pm_volume": 1157,
        "premarket_high": 11.99,
        "prev_close": 10.66,
        "float_shares": 1450477,
    },
    {
        "ticker": "EPRX",
        "gap_pct": 12.18,
        "pm_volume": 1539,
        "premarket_high": 7.0,
        "prev_close": 6.24,
        "float_shares": 55934848,
    },
    {
        "ticker": "RDGT",
        "gap_pct": 12.01,
        "pm_volume": 5992613,
        "premarket_high": 1.79,
        "prev_close": 1.58,
        "float_shares": 883913,
    },
    {
        "ticker": "LMFA",
        "gap_pct": 11.84,
        "pm_volume": 82225,
        "premarket_high": 2.74,
        "prev_close": 2.45,
        "float_shares": 868568,
    },
    {
        "ticker": "ELOX",
        "gap_pct": 11.59,
        "pm_volume": 1,
        "premarket_high": 13.0,
        "prev_close": 11.65,
        "float_shares": 2974080,
    },
    {
        "ticker": "CCTG",
        "gap_pct": 11.1,
        "pm_volume": 1462684,
        "premarket_high": 0.861,
        "prev_close": 0.775,
        "float_shares": 12675639,
    },
    {
        "ticker": "VIVS",
        "gap_pct": 10.93,
        "pm_volume": 28170350,
        "premarket_high": 0.6858,
        "prev_close": 0.5049,
        "float_shares": 3179725,
    },
    {
        "ticker": "ULBI",
        "gap_pct": 10.9,
        "pm_volume": 51,
        "premarket_high": 5.84,
        "prev_close": 5.27,
        "float_shares": 16120403,
    },
    {
        "ticker": "CBIO",
        "gap_pct": 10.76,
        "pm_volume": 345,
        "premarket_high": 16.67,
        "prev_close": 15.05,
        "float_shares": 35346824,
    },
]

DATE_STR = "2026-07-21"


def fetch_tradier_1min_bars(symbol):
    url = "https://api.tradier.com/v1/markets/timesales"
    params = {
        "symbol": symbol,
        "interval": "1min",
        "start": f"{DATE_STR} 09:30",
        "end": f"{DATE_STR} 16:00",
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "series" in data and data["series"] and "data" in data["series"]:
                    return data["series"]["data"]
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return []


def aggregate_1min_to_2min(one_min_bars):
    if not one_min_bars:
        return pd.DataFrame()

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(one_min_bars)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Convert timezone to UTC for compatibility with simulation expectations
    df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")

    # Aggregate to 2-minute bars
    resampled = (
        df.resample("2min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    # Rename columns to camelcase expected by the simulator
    resampled.columns = ["Open", "High", "Low", "Close", "Volume"]
    return resampled


def main():
    print(f"Downloading 1-minute historical data from Tradier for {DATE_STR}...")
    picks = []

    for cand in CANDIDATES:
        ticker = cand["ticker"]
        print(f"  Fetching {ticker}...")
        one_min = fetch_tradier_1min_bars(ticker)
        if not one_min:
            print(f"    No data found for {ticker}")
            continue

        two_min_df = aggregate_1min_to_2min(one_min)
        if two_min_df.empty:
            continue

        pick = {
            "ticker": ticker,
            "gap_pct": cand["gap_pct"],
            "pm_volume": cand["pm_volume"],
            "premarket_high": cand["premarket_high"],
            "prev_close": cand["prev_close"],
            "market_open": float(two_min_df.iloc[0]["Open"]),
            "market_hour_candles": two_min_df,
        }
        picks.append(pick)

    print(f"\nSuccessfully downloaded and processed {len(picks)}/20 picks.")

    # Load deployed parameters from config/trial_gl_trail_538_deploy.json
    print("\nLoading Deployed parameters from config/trial_gl_trail_538_deploy.json...")
    with open("config/trial_gl_trail_538_deploy.json") as f:
        config_data = json.load(f)

    params = {
        k: v
        for k, v in config_data["params"].items()
        if k.startswith("g_") or k.startswith("l_")
    }

    # Configure simulator
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    tgc.NEWS_FILTER_ENABLED = False

    # Ensure H/A/F are disabled, G and L enabled
    p = {}
    base_config_path = "config/trial_gl_trail_final_best.json"
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            base_data = json.load(f)
        p = dict(base_data.get("params", {}))

    p.update(params)
    p["enable_g"] = True
    p["enable_l"] = True
    for prefix in "hafdvmrpwobkcsexijn":
        p[f"enable_{prefix}"] = False

    with _param_lock:
        set_strategy_params(p)
        snapshot = _build_param_snapshot()

    print("\nRunning G+L simulation on July 21st, 2026 data...")
    cash = 25000.0
    cash_account = True
    states, cash_next, unsettled, _ = tgc.simulate_day_combined(
        picks, cash, cash_account, params=snapshot
    )

    print("\n" + "=" * 80)
    print(f"             BACKTEST SIMULATION TRADES ON {DATE_STR}")
    print("=" * 80)
    trades = []
    for st in states:
        if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
            print(
                f"TRADE: Ticker={st['ticker']:<5} | Strategy={st['strategy']:<2} | PnL=${st['pnl']:,.2f} | Reason={st['exit_reason']}"
            )
            trades.append(st)

    print(f"\nTotal simulated trades: {len(trades)}")
    print(f"Ending cash: ${cash_next + unsettled:,.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
