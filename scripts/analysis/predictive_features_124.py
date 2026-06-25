"""Find pre-9:30-AM features that predict #124's daily PnL.

Combines internal day-features (from picks data) + external macro
(VIX, SPY overnight gap, BTC trend, DXY) and tests which features
have the strongest discrimination between WIN days and LOSS days.

The goal: a same-day prediction at 9:25 AM ET that flags
"this is a good day to trade #124" vs "stay out today."

Outputs:
  - results/regime_trends/predictive_features.csv  (per-day data)
  - results/regime_trends/feature_signal_report.txt (heterogeneity verdicts)
  - Console: ranked feature importance
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
from strategies.regime_gate import classify_regime, compute_features as _regime_feats

CONFIG = "config/trial_124_microcap_pump_extracted.json"
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000
OUT_DIR = "results/regime_trends"

# All available years
ALL_DIRS = [
    "stored_data_2021", "stored_data_2022", "stored_data_2023",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026",
]


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def _classify(picks):
    try:
        regime = classify_regime(_regime_feats(picks))
    except Exception:
        regime = "NORMAL"
    sig = day_signature(picks)
    if sig is None: return "empty"
    return classify_shape(sig, regime)


def _gini(values):
    """Concentration measure (0 = even spread, 1 = single dominant)."""
    v = np.array(sorted(values))
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    n = len(v)
    cum = np.cumsum(v)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def compute_day_features(picks):
    """Compute pre-9:30 features from picks list (the data the bot would have at 9:25)."""
    if not picks:
        return {}
    gaps = [p.get("gap_pct", 0) or 0 for p in picks]
    prices = [p.get("prev_close", 0) or 0 for p in picks]
    pm_vols = [p.get("pm_volume", 0) or 0 for p in picks]
    pm_dollars = [pv * pr for pv, pr in zip(pm_vols, prices)]

    return {
        "n_picks": len(picks),
        "n_above_30": sum(1 for g in gaps if g >= 30),
        "n_above_50": sum(1 for g in gaps if g >= 50),
        "max_gap": max(gaps) if gaps else 0,
        "median_gap": float(np.median(gaps)),
        "gap_std": float(np.std(gaps)),
        "leader_prev_close": prices[int(np.argmax(gaps))] if gaps else 0,
        "leader_pm_vol": pm_vols[int(np.argmax(gaps))] if gaps else 0,
        "median_prev_close": float(np.median(prices)),
        "total_pm_dollar_vol": float(sum(pm_dollars)),
        "median_pm_dollar_vol": float(np.median(pm_dollars)),
        # Concentration: how much PM $vol is in the top-3 vs total
        "top3_pm_dvol_share": (sum(sorted(pm_dollars, reverse=True)[:3]) /
                               sum(pm_dollars)) if sum(pm_dollars) > 0 else 0,
        "pm_dvol_gini": _gini(pm_dollars),
    }


def download_external():
    """Pull VIX, SPY, QQQ, BTC, DXY daily bars 2021-now via yfinance."""
    import yfinance as yf
    tickers = {
        "VIX": "^VIX",
        "SPY": "SPY",
        "QQQ": "QQQ",
        "DXY": "DX-Y.NYB",  # US Dollar Index
        "BTC": "BTC-USD",
        "IWM": "IWM",       # Russell 2000 (small-caps)
    }
    out = {}
    for name, sym in tickers.items():
        try:
            print(f"  Downloading {name} ({sym})...")
            df = yf.download(sym, start="2020-12-15", end="2026-06-18",
                            interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) == 0:
                print(f"    [warn] {name} returned 0 rows")
                continue
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            df.index = df.index.normalize()
            out[name] = df
            print(f"    Got {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as e:
            print(f"    [err] {name}: {e}")
    return out


def macro_features_for_date(ext, date_str):
    """Get macro context as of 9:30 AM ET on date_str (using PREVIOUS close)."""
    d = pd.Timestamp(date_str).normalize()
    feats = {}
    for name, df in ext.items():
        prior = df[df.index < d]
        if len(prior) < 2:
            continue
        last_close = float(prior["Close"].iloc[-1])
        prev_close = float(prior["Close"].iloc[-2])
        chg_1d = (last_close - prev_close) / prev_close * 100
        # 5-day and 20-day momentum
        if len(prior) >= 6:
            chg_5d = (last_close - float(prior["Close"].iloc[-6])) / float(prior["Close"].iloc[-6]) * 100
        else:
            chg_5d = 0
        if len(prior) >= 21:
            chg_20d = (last_close - float(prior["Close"].iloc[-21])) / float(prior["Close"].iloc[-21]) * 100
        else:
            chg_20d = 0
        feats[f"{name}_close"] = last_close
        feats[f"{name}_1d"] = chg_1d
        feats[f"{name}_5d"] = chg_5d
        feats[f"{name}_20d"] = chg_20d
    return feats


def run_day_pnl(picks, cash):
    is_cash = cash < MARGIN_THRESHOLD
    try:
        states, end_c, unset, _ = tgc.simulate_day_combined(picks, cash, cash_account=is_cash)
    except Exception:
        return cash, 0, 0
    n_trades = sum(1 for s in states if s.get("exit_reason") is not None
                   and s.get("position_cost", 0) > 0)
    final_cash = end_c + (unset if is_cash else 0)
    day_pnl = final_cash - cash
    return final_cash, day_pnl, n_trades


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CONFIG) as f: cfg = json.load(f)
    set_strategy_params(_merged(cfg["params"]))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    print("Downloading macro data...")
    ext = download_external()
    print(f"\nGot {len(ext)} macro series.")

    present = [d for d in ALL_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(present)

    # Only use 2021-2026 (matches #124's universe)
    test_dates = [d for d in all_dates if d[:4] >= "2021"]

    print(f"\nRunning #124 backtest with per-day feature collection...")
    cash = STARTING_CASH
    rows = []
    for d in test_dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            continue
        shape = _classify(day_picks)
        intl_feats = compute_day_features(day_picks)
        macro = macro_features_for_date(ext, d)

        new_cash, day_pnl, n_trades = run_day_pnl(day_picks, cash)

        row = {
            "date": d,
            "year": d[:4],
            "dow": pd.Timestamp(d).day_name(),
            "month": int(d[5:7]),
            "shape": shape,
            "starting_cash": cash,
            "ending_cash": new_cash,
            "day_pnl": day_pnl,
            "day_pnl_pct": (day_pnl / cash * 100) if cash > 0 else 0,
            "n_trades": n_trades,
            "n_picks": intl_feats.get("n_picks", 0),
            "n_above_30": intl_feats.get("n_above_30", 0),
            "n_above_50": intl_feats.get("n_above_50", 0),
            "max_gap": intl_feats.get("max_gap", 0),
            "median_gap": intl_feats.get("median_gap", 0),
            "gap_std": intl_feats.get("gap_std", 0),
            "leader_prev_close": intl_feats.get("leader_prev_close", 0),
            "leader_pm_vol": intl_feats.get("leader_pm_vol", 0),
            "median_prev_close": intl_feats.get("median_prev_close", 0),
            "total_pm_dollar_vol": intl_feats.get("total_pm_dollar_vol", 0),
            "median_pm_dollar_vol": intl_feats.get("median_pm_dollar_vol", 0),
            "top3_pm_dvol_share": intl_feats.get("top3_pm_dvol_share", 0),
            "pm_dvol_gini": intl_feats.get("pm_dvol_gini", 0),
            **macro,
        }
        rows.append(row)
        cash = new_cash

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "predictive_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}: {len(df)} rows")

    # --- Analysis ---
    # Only days where the bot actually traded
    traded = df[df["n_trades"] > 0].copy()
    print(f"\nDays with trades: {len(traded)}")
    print(f"  Win days: {(traded['day_pnl'] > 0).sum()} ({(traded['day_pnl']>0).mean()*100:.1f}%)")
    print(f"  Median PnL: ${traded['day_pnl'].median():,.0f}")

    # Per-feature: split into 3 buckets (low / mid / high) and report mean PnL per bucket
    print(f"\n{'='*78}")
    print(f"FEATURE DISCRIMINATION (continuous features by tercile)")
    print(f"{'='*78}")
    print(f"  {'feature':<28} {'low_pnl':>10} {'mid_pnl':>10} {'high_pnl':>10} {'spread':>8}")

    feature_cols = [c for c in df.columns if c not in (
        "date","year","dow","month","shape","starting_cash","ending_cash",
        "day_pnl","day_pnl_pct","n_trades")]

    rankings = []
    for col in feature_cols:
        vals = traded[col].dropna()
        if len(vals) < 30 or vals.std() == 0:
            continue
        try:
            qs = pd.qcut(vals, 3, labels=["low","mid","high"], duplicates="drop")
        except Exception:
            continue
        if qs.isna().sum() == len(qs):
            continue
        bucket_pnls = traded.loc[vals.index].assign(b=qs).groupby("b")["day_pnl_pct"].mean()
        if len(bucket_pnls) < 2:
            continue
        spread = bucket_pnls.max() - bucket_pnls.min()
        rankings.append((col, bucket_pnls, spread))

    rankings.sort(key=lambda r: -r[2])
    for col, bucket_pnls, spread in rankings[:30]:
        lo = bucket_pnls.get("low", float("nan"))
        md = bucket_pnls.get("mid", float("nan"))
        hi = bucket_pnls.get("high", float("nan"))
        print(f"  {col:<28} {lo:>+9.2f}% {md:>+9.2f}% {hi:>+9.2f}% {spread:>+7.2f}pp")

    # Categorical: day-of-week, month, shape
    print(f"\n{'='*78}")
    print(f"CATEGORICAL FEATURES")
    print(f"{'='*78}")
    for col in ("dow", "month", "shape", "year"):
        g = traded.groupby(col).agg(
            n=("day_pnl_pct", "count"),
            mean=("day_pnl_pct", "mean"),
            win_rate=("day_pnl", lambda s: (s > 0).mean() * 100),
        ).sort_values("mean", ascending=False)
        print(f"\n  {col.upper()}:")
        for k, r in g.iterrows():
            print(f"    {str(k):<14} n={int(r['n']):>4}  mean={r['mean']:>+7.2f}%  WR={r['win_rate']:>5.1f}%")


if __name__ == "__main__":
    main()
