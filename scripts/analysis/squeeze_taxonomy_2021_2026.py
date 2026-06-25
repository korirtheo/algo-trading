"""Deep-dive taxonomy of 2019-2026 gap-up days.

Looks at EVERY trading day (dead/normal/squeeze) and builds a day-shape
signature using:
  - n_above_20 / n_above_50 (how many tickers are gapping)
  - max_gap (leader's gap)
  - leader prev_close, median pick prev_close
  - total premarket volume (shares) + total premarket dollar volume
  - count of picks per price tier (<$1, $1-3, $3-10, $10-30, >$30)

Then classifies each day into a "shape bucket":
  - dead              : <2 picks gapping >20%
  - microcap-thin     : 2+ gappers but leader sub-$1 (likely pump-and-dump)
  - corp-action       : leader sub-$1 AND gap >500% (likely reverse-split)
  - thin-normal       : 2-5 gappers, leader $1-$3, low volume
  - liquid-normal     : 2-5 gappers, leader $3+, decent volume
  - broad-squeeze     : 4+ gappers above 50%, leader $3+, real multi-name push
  - mega-cap-day      : leader >$30 (large-cap gap)

Outputs:
  results/squeeze_taxonomy_summary.csv      — one row per day
  cross-year tables in stdout
  optional --backtest mode: runs #587/#541/#6 per shape-bucket on 2021-2026.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import os
import numpy as np
import pandas as pd

from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime, regime_features

DATA_DIRS_BY_YEAR = {
    "2019": ["stored_data_2019"],
    "2020": ["stored_data_2020"],
    "2021": ["stored_data_2021"],
    "2022": ["stored_data_2022"],
    "2023": ["stored_data_2023"],
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026"],
}

PRICE_TIERS = [
    ("<$1",     0,    1),
    ("$1-3",    1,    3),
    ("$3-10",   3,   10),
    ("$10-30", 10,   30),
    (">$30",   30, 1e9),
]


def tier_of(p):
    for label, lo, hi in PRICE_TIERS:
        if lo <= p < hi:
            return label
    return ">$30"


def _pm_vol(p):
    return float(p.get("pm_volume") or p.get("premarket_volume") or 0)


def day_signature(picks):
    if not picks:
        return None
    prices  = [float(p.get("prev_close") or 0) for p in picks]
    gaps    = [float(p.get("gap_pct") or 0) for p in picks]
    pm_vols = [_pm_vol(p) for p in picks]
    pm_dollars = [pv * pr for pv, pr in zip(pm_vols, prices)]

    # leader = biggest gap
    li = int(np.argmax(gaps))
    leader = picks[li]
    leader_prev = float(leader.get("prev_close", 0) or 0)
    leader_gap  = float(leader.get("gap_pct", 0) or 0)
    leader_pm_vol = _pm_vol(leader)

    sig = {
        "n_picks":         len(picks),
        "leader":          leader.get("ticker", ""),
        "leader_gap":      leader_gap,
        "leader_prev":     leader_prev,
        "leader_pm_vol":   leader_pm_vol,
        "leader_pm_$vol":  leader_prev * leader_pm_vol,
        "leader_tier":     tier_of(leader_prev),
        "median_prev":     float(np.median(prices)),
        "min_prev":        float(np.min(prices)),
        "max_gap":         float(np.max(gaps)),
        "median_gap":      float(np.median(gaps)),
        "total_pm_vol":    float(np.sum(pm_vols)),
        "total_pm_$vol":   float(np.sum(pm_dollars)),
        "median_pm_vol":   float(np.median(pm_vols)),
    }
    for label, lo, hi in PRICE_TIERS:
        sig[f"n_{label}"] = sum(1 for x in prices if lo <= x < hi)

    # n_above_20 / n_above_50 — mirror the regime gate
    rf = regime_features(picks)
    sig["n_above_20"] = rf.get("n_above_20", 0)
    sig["n_above_50"] = rf.get("n_above_50", 0)
    return sig


def classify_shape(sig, regime):
    """Stratify the day into a structural bucket."""
    if sig is None:
        return "empty"
    # 1. Corporate-action artifact: extreme gap from a sub-$1 stock
    if sig["leader_gap"] >= 500 and sig["leader_prev"] < 1.0:
        return "corp-action"
    # 2. Mega-cap day: leader is large-cap
    if sig["leader_prev"] >= 30.0:
        return "mega-cap"
    # 3. Microcap-thin: leader sub-$1 — likely pump-and-dump
    if sig["leader_prev"] < 1.0:
        return "microcap-thin"
    # 4. Broad squeeze: many strong gappers with respectable leader
    if sig["n_above_50"] >= 4 and sig["leader_prev"] >= 3.0:
        return "broad-squeeze"
    # 5. Liquid normal: leader $3+, decent pm dollar volume
    if sig["leader_prev"] >= 3.0 and sig["total_pm_$vol"] >= 1_000_000:
        return "liquid-normal"
    # 6. Thin normal: 1-3 range, low volume
    if sig["leader_prev"] >= 1.0 and sig["leader_prev"] < 3.0:
        return "thin-microcap"
    # 7. Catch-all
    return "other"


def year_report(year, dirs, args):
    print("=" * 100)
    print(f"YEAR {year}  ({', '.join(dirs)})")
    print("=" * 100)
    dirs_present = [d for d in dirs if os.path.exists(d)]
    if not dirs_present:
        print(f"[skip] no data dirs present")
        return None
    dates, dp = load_all_picks(dirs_present)
    if not dates:
        print("[skip] no dates loaded"); return None

    rows = []
    for d in dates:
        picks = dp.get(d, [])
        regime = classify_regime(picks)
        sig = day_signature(picks) or {}
        sig.update({
            "date": d, "year": year, "regime": regime, "n_picks_raw": len(picks),
            "shape": classify_shape(sig if sig else None, regime),
        })
        rows.append(sig)
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} trading days")

    regime_counts = df["regime"].value_counts().reindex(
        ["dead", "normal", "squeeze"], fill_value=0)
    print(f"\nregime: {regime_counts.to_dict()}")
    print(f"\nshape bucket counts:")
    sc = df["shape"].value_counts()
    for s, n in sc.items():
        print(f"  {s:<16}  {n:>4}  ({100*n/len(df):.1f}%)")

    # Median stats per shape bucket
    print(f"\nmedians per shape bucket:")
    cols = ["n_picks", "leader_gap", "leader_prev", "median_prev",
            "n_above_50", "total_pm_vol", "total_pm_$vol"]
    g = df.groupby("shape")[cols].median().round(2)
    print(g.to_string())

    if args.verbose:
        print(f"\nshape examples (first 3 each):")
        for shape in sorted(df["shape"].unique()):
            sub = df[df["shape"] == shape].head(3)
            print(f"\n  [{shape}]")
            for _, r in sub.iterrows():
                print(f"    {r['date']}  leader={r['leader']:<6} "
                      f"gap={r['leader_gap']:>6.0f}% prev=${r['leader_prev']:>6.2f} "
                      f"n50={r['n_above_50']:>2} pm_vol={r['total_pm_vol']:>10,.0f}")
    return df


def run_one_day(picks, cash, cfg_params):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    set_strategy_params(cfg_params)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.VOL_CAP_PCT = 5.0
    is_cash = cash < MARGIN_THRESHOLD
    try:
        _, ending_cash, unsettled, _ = tgc.simulate_day_combined(
            picks, cash, cash_account=is_cash)
        return ending_cash, unsettled, is_cash
    except Exception:
        return cash, 0.0, is_cash


def _load_cfg(path, baseline):
    with open(path) as f:
        p = json.load(f)
    if isinstance(p, dict) and "params" in p:
        p = p["params"]
    with open(baseline) as f:
        base = json.load(f)
    m = dict(base); m.update(p); return m


def backtest_per_shape(df, daily_picks, configs, baseline, starting_cash=25_000):
    """For each (shape, config) pair, sum PnL across all days in that shape."""
    print("\n" + "=" * 100)
    print("PER-SHAPE BACKTEST (each config restarted from $25K within each shape)")
    print("=" * 100)
    rows = []
    for shape in sorted(df["shape"].unique()):
        days = df[df["shape"] == shape]["date"].tolist()
        if not days:
            continue
        for cfg_path, label in configs:
            cfg = _load_cfg(cfg_path, baseline)
            cash = starting_cash
            n_pos = n_neg = 0
            for d in days:
                picks = daily_picks.get(d, [])
                if not picks: continue
                end_cash, unsettled, is_cash = run_one_day(picks, cash, cfg)
                pnl = end_cash - cash
                cash = end_cash + (unsettled if is_cash else 0)
                if pnl > 0: n_pos += 1
                elif pnl < 0: n_neg += 1
            total_pnl = cash - starting_cash
            rows.append({
                "shape": shape, "config": label, "n_days": len(days),
                "total_pnl": total_pnl, "wins": n_pos, "losses": n_neg,
                "pnl_per_day": total_pnl / len(days) if days else 0,
            })
    rdf = pd.DataFrame(rows)
    print(f"\n{rdf.to_string(index=False)}")
    return rdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2019,2021,2022,2023,2024,2025,2026")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--backtest", action="store_true",
                    help="Also run #6/#326/#541/#587 per shape bucket on 2021-2026")
    ap.add_argument("--save", default="results/squeeze_taxonomy_summary.csv")
    args = ap.parse_args()

    all_rows = []
    picks_by_year = {}
    for y in args.years.split(","):
        y = y.strip()
        if y not in DATA_DIRS_BY_YEAR:
            print(f"[skip] unknown year {y}")
            continue
        dirs_present = [d for d in DATA_DIRS_BY_YEAR[y] if os.path.exists(d)]
        if not dirs_present:
            print(f"[skip] {y}: no data dirs present")
            continue
        df_y = year_report(y, DATA_DIRS_BY_YEAR[y], args)
        if df_y is not None and not df_y.empty:
            all_rows.append(df_y)
            # Cache the picks dict for the per-shape backtest later
            _, dp = load_all_picks(dirs_present)
            picks_by_year[y] = dp

    if not all_rows:
        return
    big = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    big.to_csv(args.save, index=False)
    print(f"\nSaved {len(big)} day-rows -> {args.save}")

    # Cross-year tables — only on shape (regime is already shown per-year)
    print("\n" + "=" * 100)
    print("CROSS-YEAR SHAPE MATRIX (count)")
    print("=" * 100)
    ct = pd.crosstab(big["year"], big["shape"], margins=True, margins_name="total")
    print(ct.to_string())

    print("\n" + "=" * 100)
    print("CROSS-YEAR SHAPE MATRIX (% of year)")
    print("=" * 100)
    ct2 = pd.crosstab(big["year"], big["shape"], normalize="index") * 100
    print(ct2.round(1).to_string())

    print("\n" + "=" * 100)
    print("REGIME MATRIX (count)")
    print("=" * 100)
    ct3 = pd.crosstab(big["year"], big["regime"], margins=True, margins_name="total")
    print(ct3.to_string())

    # Volume by shape — what does each shape look like?
    print("\n" + "=" * 100)
    print("VOLUME / PRICE SUMMARY BY SHAPE (medians across ALL years)")
    print("=" * 100)
    cols = ["n_picks", "n_above_50", "leader_gap", "leader_prev",
            "median_prev", "total_pm_vol", "total_pm_$vol"]
    g = big.groupby("shape")[cols].median().round(2)
    print(g.to_string())

    if args.backtest:
        # Combined 2021-2026 backtest per shape
        train_years = ["2021", "2022", "2023", "2024", "2025", "2026"]
        train_df = big[big["year"].isin(train_years)].copy()
        # Combine all training-year picks
        all_picks = {}
        for y in train_years:
            if y in picks_by_year:
                all_picks.update(picks_by_year[y])
        configs = [
            ("config/trial_6_extracted.json",            "#6   generalist"),
            ("config/trial_326_squeeze_extracted.json",  "#326 sq2.09"),
            ("config/trial_541_squeeze_extracted.json",  "#541 sq2.73"),
            ("config/trial_587_squeeze_extracted.json",  "#587 sq2.91"),
        ]
        configs = [(p, l) for p, l in configs if os.path.exists(p)]
        backtest_per_shape(train_df, all_picks, configs,
                           baseline="config/trial_432_params.json")


if __name__ == "__main__":
    main()
