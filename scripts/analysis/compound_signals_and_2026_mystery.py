"""Two analyses in one:

A) COMPOUND PREDICTIVE POWER — what's WR when N signals align?
   Combines the strongest signals into a "favorability count" and reports
   outcomes by count.

B) 2026 MICROCAP-THIN MYSTERY — what's structurally different about 2026's
   microcap-thin days compared to 2021-2025's? Same regime, opposite PnL.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

CSV = "results/regime_trends/predictive_features.csv"


def analysis_a_compound_signals(df):
    """Compound predictive power on microcap-pump days."""
    sub = df[df["shape"].isin(["microcap-thin", "thin-microcap"])
             & (df["n_trades"] > 0)].copy()
    print(f"\n{'='*80}")
    print(f"A) COMPOUND PREDICTIVE POWER — microcap-pump days")
    print(f"   Baseline: n={len(sub)}, WR={(sub['day_pnl']>0).mean()*100:.1f}%, "
          f"mean_pnl%={sub['day_pnl_pct'].mean():+.2f}%")
    print(f"{'='*80}")

    # Define each signal as a binary (1 = favorable)
    signals = {
        "not_wed":      sub["dow"] != "Wednesday",
        "not_jun_aug":  ~sub["month"].isin([6, 8]),
        "dxy_up":       sub["DXY_1d"] > 0,
        "btc_flat":     sub["BTC_5d"].abs() < 5,
        "vix_mid":      (sub["VIX_close"] >= sub["VIX_close"].quantile(1/3)) &
                        (sub["VIX_close"] <= sub["VIX_close"].quantile(2/3)),
        "iwm_mid":      (sub["IWM_close"] >= sub["IWM_close"].quantile(1/3)) &
                        (sub["IWM_close"] <= sub["IWM_close"].quantile(2/3)),
        "good_pm_dvol": (sub["median_pm_dollar_vol"] >= sub["median_pm_dollar_vol"].quantile(1/3)) &
                        (sub["median_pm_dollar_vol"] <= sub["median_pm_dollar_vol"].quantile(2/3)),
    }
    print(f"\n   Each signal's individual effect:")
    print(f"   {'signal':<16} {'true_n':>7} {'true_mean':>10} {'true_WR':>8}  "
          f"{'false_n':>7} {'false_mean':>10} {'false_WR':>8}")
    for name, mask in signals.items():
        t = sub[mask]; f = sub[~mask]
        print(f"   {name:<16} {len(t):>7} {t['day_pnl_pct'].mean():>+9.2f}%  "
              f"{(t['day_pnl']>0).mean()*100:>6.1f}%  "
              f"{len(f):>7} {f['day_pnl_pct'].mean():>+9.2f}%  "
              f"{(f['day_pnl']>0).mean()*100:>6.1f}%")

    # Sum of signals each day
    sub["n_signals"] = sum(mask.astype(int) for mask in signals.values())
    n_max = len(signals)
    print(f"\n   COMPOUND ALIGNMENT (out of {n_max} signals):")
    print(f"   {'n_aligned':<11} {'n_days':>7} {'WR':>7} {'mean_pnl%':>11} {'median_pnl%':>13}")
    for n in range(0, n_max + 1):
        g = sub[sub["n_signals"] == n]
        if len(g) < 3: continue
        print(f"   {n}/{n_max:<8} {len(g):>7} {(g['day_pnl']>0).mean()*100:>6.1f}% "
              f"{g['day_pnl_pct'].mean():>+10.2f}% {g['day_pnl_pct'].median():>+12.2f}%")

    # Top buckets
    print(f"\n   By favorability percentile bucket:")
    sub["bucket"] = pd.cut(sub["n_signals"], bins=[-1, 2, 4, 7],
                           labels=["LOW(0-2)", "MID(3-4)", "HIGH(5-7)"])
    g = sub.groupby("bucket", observed=True).agg(
        n=("day_pnl_pct", "count"),
        wr=("day_pnl", lambda s: (s > 0).mean() * 100),
        mean=("day_pnl_pct", "mean"),
        median=("day_pnl_pct", "median"),
    )
    print(f"   {g.to_string()}")


def analysis_b_2026_mystery(df):
    """What's different about 2026 microcap-thin days?"""
    sub = df[df["shape"] == "microcap-thin"].copy()
    sub["year"] = sub["year"].astype(int)
    sub["era"] = sub["year"].apply(
        lambda y: "2021-2025" if y < 2026 else "2026")

    by_era = sub[sub["n_trades"] > 0].groupby("era").agg(
        n=("day_pnl_pct", "count"),
        wr=("day_pnl", lambda s: (s > 0).mean() * 100),
        mean=("day_pnl_pct", "mean"),
        median=("day_pnl", "median"),
    )
    print(f"\n{'='*80}")
    print(f"B) 2026 MICROCAP-THIN MYSTERY — what's different?")
    print(f"{'='*80}")
    print(f"\n   Era comparison:")
    for era, r in by_era.iterrows():
        print(f"     {era:<10}  n={int(r['n']):>4}  WR={r['wr']:>5.1f}%  "
              f"mean={r['mean']:>+6.2f}%  median=${r['median']:,.0f}")

    # Compare distributions of every feature
    feature_cols = [c for c in df.columns if c not in (
        "date","year","dow","month","shape","starting_cash","ending_cash",
        "day_pnl","day_pnl_pct","n_trades","era")]

    pre = sub[(sub["era"] == "2021-2025") & (sub["n_trades"] > 0)]
    cur = sub[(sub["era"] == "2026") & (sub["n_trades"] > 0)]
    if len(pre) < 30 or len(cur) < 10:
        print(f"   Insufficient sample for comparison."); return

    print(f"\n   Feature drift 2021-2025 vs 2026 (mean values, sorted by relative change):")
    print(f"   {'feature':<26} {'2021-25':>12} {'2026':>12} {'delta':>10} {'rel%':>8}")
    drifts = []
    for col in feature_cols:
        a = pre[col].dropna()
        b = cur[col].dropna()
        if len(a) < 20 or len(b) < 10 or a.std() == 0: continue
        ma, mb = a.mean(), b.mean()
        rel = ((mb - ma) / abs(ma) * 100) if abs(ma) > 1e-9 else 0
        drifts.append((col, ma, mb, mb - ma, rel))
    drifts.sort(key=lambda r: -abs(r[4]))
    for col, ma, mb, delta, rel in drifts[:20]:
        print(f"   {col:<26} {ma:>+11.2f} {mb:>+11.2f} {delta:>+9.2f} {rel:>+7.1f}%")

    # Same DOW analysis
    print(f"\n   2026 microcap-thin by DOW:")
    g = cur.groupby("dow").agg(
        n=("day_pnl_pct", "count"),
        wr=("day_pnl", lambda s: (s > 0).mean() * 100),
        mean=("day_pnl_pct", "mean"),
    )
    for k, r in g.iterrows():
        print(f"     {str(k):<12} n={int(r['n']):>3}  WR={r['wr']:>5.1f}%  mean={r['mean']:>+6.2f}%")

    # Compare specific suspects
    print(f"\n   KEY SUSPECT FEATURES (mean by era):")
    suspects = ["DXY_1d", "DXY_close", "BTC_close", "VIX_close", "IWM_close",
                "leader_pm_vol", "top3_pm_dvol_share", "pm_dvol_gini",
                "leader_prev_close", "max_gap", "n_picks"]
    for col in suspects:
        if col not in pre.columns: continue
        pa = pre[col].mean(); pb = cur[col].mean()
        print(f"     {col:<28} 2021-25 avg={pa:>+10.2f}  2026 avg={pb:>+10.2f}  diff={pb-pa:+.2f}")


def main():
    df = pd.read_csv(CSV)
    analysis_a_compound_signals(df)
    analysis_b_2026_mystery(df)


if __name__ == "__main__":
    main()
