"""Replay 2026-06-24 backtest with same watchlist + bars as live engine.

Compares backtest predicted result with live actual (-$42.77 on CCXI from +153bp slip).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import pickle
import pandas as pd

DATA_DIR = "stored_data_2026_06_24"
INTRA_DIR = os.path.join(DATA_DIR, "intraday")
DAILY_DIR = os.path.join(DATA_DIR, "daily")

WATCHLIST = {
    "PLSM": {"gap_pct": 206.5, "pm_vol": 9_615_683, "float": 3_700_000},
    "FRTT": {"gap_pct": 77.4,  "pm_vol":12_029_744, "float": None},
    "MGN":  {"gap_pct": 58.6,  "pm_vol":79_280_162, "float":26_200_000},
    "VTAK": {"gap_pct": 55.4,  "pm_vol":18_340_420, "float": 2_300_000},
    "ORISF":{"gap_pct": 33.5,  "pm_vol":12_021_163, "float": 1_100_000},
    "QNRX": {"gap_pct": 32.6,  "pm_vol": 1_657_710, "float": 1_900_000},
    "ABSI": {"gap_pct": 28.3,  "pm_vol": 2_352_060, "float":135_300_000},
    "RCT":  {"gap_pct": 28.0,  "pm_vol": 1_126_384, "float":17_200_000},
    "WEN":  {"gap_pct": 25.2,  "pm_vol":15_509_683, "float":173_000_000},
    "GPUS": {"gap_pct": 25.1,  "pm_vol":106_991_694,"float":457_800_000},
    "DGICB":{"gap_pct": 24.9,  "pm_vol":          6, "float":36_400_000},
    "RUN":  {"gap_pct": 20.7,  "pm_vol": 3_130_753, "float":231_600_000},
    "CCXI": {"gap_pct": 20.5,  "pm_vol": 2_988_302, "float":40_500_000},
    "UBXG": {"gap_pct": 19.2,  "pm_vol":     86_243, "float": 1_200_000},
    "EHGO": {"gap_pct": 17.2,  "pm_vol": 3_728_163, "float": 1_300_000},
    "FISN": {"gap_pct": 16.8,  "pm_vol":     31_190, "float": 2_500_000},
    "ENRD": {"gap_pct": 16.6,  "pm_vol":     10_628, "float":139_400_000},
    "FCEL": {"gap_pct": 15.9,  "pm_vol": 1_300_000, "float":67_400_000},
    "BNC":  {"gap_pct": 13.2,  "pm_vol":    399_000, "float":37_700_000},
    "MMED": {"gap_pct": 13.0,  "pm_vol":         95, "float":26_700_000},
}

TEST_DATE = "2026-06-24"


def normalize_bars():
    print("Normalizing per-ticker bars...")
    n = 0
    for fn in sorted(os.listdir(INTRA_DIR)):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(INTRA_DIR, fn)
        df = pd.read_csv(path)
        if df.empty or "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df = df.sort_values(["timestamp", "Volume"]).drop_duplicates(subset=["timestamp"], keep="last")
        df = df.set_index("timestamp")[["Open", "High", "Low", "Close", "Volume"]]
        df.index.name = "Datetime"
        df.to_csv(path)
        n += 1
    print(f"  Normalized {n} ticker CSVs")


def build_daily():
    print("Building daily CSVs (prev_close from gap_pct)...")
    import pytz
    et = pytz.timezone("America/New_York")
    n = 0
    for ticker, info in WATCHLIST.items():
        intra_path = os.path.join(INTRA_DIR, f"{ticker}.csv")
        if not os.path.exists(intra_path):
            continue
        df = pd.read_csv(intra_path, index_col=0, parse_dates=True)
        if df.empty:
            continue
        et_index = df.index.tz_convert(et) if df.index.tz else df.index.tz_localize("UTC").tz_convert(et)
        mh_mask = ((et_index.hour == 9) & (et_index.minute >= 30)) | ((et_index.hour >= 10) & (et_index.hour < 16))
        market_hours = df[mh_mask]
        if market_hours.empty:
            continue
        market_open = float(market_hours.iloc[0]["Open"])
        prev_close = market_open / (1 + info["gap_pct"] / 100)
        daily_df = pd.DataFrame({
            "Open": [prev_close, prev_close, market_open],
            "High": [prev_close, prev_close, float(df["High"].max())],
            "Low":  [prev_close, prev_close, float(df["Low"].min())],
            "Close":[prev_close, prev_close, float(df["Close"].iloc[-1])],
            "Volume":[100_000, 100_000, int(df["Volume"].sum())],
        }, index=pd.to_datetime(["2026-06-22", "2026-06-23", TEST_DATE]))
        daily_df.index.name = "Date"
        daily_df.to_csv(os.path.join(DAILY_DIR, f"{ticker}.csv"))
        n += 1
    print(f"  Wrote {n} daily CSVs")


def build_gainers():
    rows = [{"date": TEST_DATE, "ticker": t, "gap_pct": info["gap_pct"]}
            for t, info in WATCHLIST.items()]
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "daily_top_gainers.csv"), index=False)


def main():
    normalize_bars()
    build_daily()
    build_gainers()

    # Build picks DIRECTLY (skip _process_one_day — our audit bars lack premarket)
    import pytz
    et = pytz.timezone("America/New_York")
    day_picks = []
    for ticker, info in WATCHLIST.items():
        intra_path = os.path.join(INTRA_DIR, f"{ticker}.csv")
        if not os.path.exists(intra_path):
            continue
        idf = pd.read_csv(intra_path, index_col=0, parse_dates=True)
        if idf.empty:
            continue
        et_index = idf.index.tz_convert(et) if idf.index.tz else idf.index.tz_localize("UTC").tz_convert(et)
        mh_mask = ((et_index.hour == 9) & (et_index.minute >= 30)) | ((et_index.hour >= 10) & (et_index.hour < 16))
        market_hours = idf[mh_mask]
        if market_hours.empty:
            continue
        market_open = float(market_hours.iloc[0]["Open"])
        prev_close = market_open / (1 + info["gap_pct"] / 100)
        gap_pct = (market_open / prev_close - 1) * 100
        day_picks.append({
            "ticker": ticker,
            "gap_pct": gap_pct,
            "market_open": market_open,
            "premarket_high": market_open,  # approx — we lack PM bars
            "prev_close": prev_close,
            "pm_volume": info["pm_vol"],
            "market_hour_candles": market_hours,
        })
    day_picks.sort(key=lambda x: -x["gap_pct"])
    day_picks = day_picks[:20]
    print(f"\n--- {TEST_DATE} picks (constructed directly) ---")
    print(f"  picks for {TEST_DATE}: {len(day_picks)}")
    for p in day_picks[:10]:
        print(f"    {p['ticker']:>6}  gap={p['gap_pct']:>6.1f}%  pm_vol={p['pm_volume']:>11,}  mkt_open={p['market_open']:.4f}  bars={len(p['market_hour_candles'])}")

    print("\n" + "=" * 92)
    print("  REPLAY: #511 deployed params on 2026-06-24 bars")
    print("=" * 92)
    with open("config/trial_432_params.json") as f:
        baseline = json.load(f)
    with open("config/trial_w21b_511_deploy.json") as f:
        p511 = json.load(f)["params"]
    merged = {**baseline, **p511}
    ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})

    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0
    tgc.NEWS_MODULATOR_ENABLED = False

    STARTING_CASH = 25_000
    states, end_cash, unset, _ = tgc.simulate_day_combined(day_picks, STARTING_CASH, cash_account=True)
    total = end_cash + unset
    print(f"  End cash: ${end_cash:,.2f}  unset: ${unset:,.2f}  total: ${total:,.2f}")
    print(f"  Backtest PnL vs $25K start: ${total - STARTING_CASH:+,.2f}")
    print()
    print("  Trades:")
    for st in states:
        if st.get("exit_reason") and st.get("position_cost", 0) > 0:
            entry = st.get("entry_price") or 0
            exit_p = st.get("exit_price") or 0
            cost = st.get("position_cost") or 0
            pnl = st.get("pnl") or 0
            print(f"    {st.get('ticker','?'):>6} {st.get('strategy','?'):>2}  "
                  f"entry=${entry:.4f}  exit=${exit_p:.4f}  "
                  f"cost=${cost:,.0f}  pnl=${pnl:+,.2f}  "
                  f"reason={st.get('exit_reason')}")

    print("\n--- LIVE COMPARISON ---")
    print("  Live actual: CCXI entry=$12.83 (+153bp slip from signal $12.6363), exit=$12.76, pnl=$-42.77")


if __name__ == "__main__":
    main()
