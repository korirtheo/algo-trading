"""Same predictive-features analysis run separately per regime.

Compares the discriminating features across:
  - microcap-pump (combined microcap-thin + thin-microcap)
  - liquid-normal
  - broad-squeeze (if enough samples)
  - corp-action / mega-cap (note insufficient sample if applicable)

For each regime, reports:
  - Top continuous features (terciles)
  - Day-of-week, month
  - Macro combinations
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

CSV = "results/regime_trends/predictive_features.csv"

REGIME_GROUPS = {
    "microcap-pump":  ["microcap-thin", "thin-microcap"],
    "liquid-normal":  ["liquid-normal"],
    "broad-squeeze":  ["broad-squeeze"],
    "mega-cap":       ["mega-cap"],
    "corp-action":    ["corp-action"],
}

MIN_SAMPLE = 30  # need at least 30 trade-days to be meaningful


def analyze_regime(df, regime_name, shapes, n_top=10):
    sub = df[df["shape"].isin(shapes) & (df["n_trades"] > 0)].copy()
    n = len(sub)
    if n < MIN_SAMPLE:
        print(f"\n{'#'*72}\n# {regime_name.upper()}: only {n} trade-days — SKIPPED (need {MIN_SAMPLE}+)\n{'#'*72}")
        return None

    wr = (sub["day_pnl"] > 0).mean() * 100
    median_pnl = sub["day_pnl"].median()
    mean_pnl_pct = sub["day_pnl_pct"].mean()
    print(f"\n{'#'*78}")
    print(f"# {regime_name.upper()}")
    print(f"# n_days={n}, WR={wr:.1f}%, median_pnl=${median_pnl:,.0f}, mean_pnl%={mean_pnl_pct:+.2f}%")
    print(f"{'#'*78}")

    feature_cols = [c for c in df.columns if c not in (
        "date","year","dow","month","shape","starting_cash","ending_cash",
        "day_pnl","day_pnl_pct","n_trades")]

    rankings = []
    for col in feature_cols:
        vals = sub[col].dropna()
        if len(vals) < 20 or vals.std() == 0: continue
        try:
            qs = pd.qcut(vals, 3, labels=["low","mid","high"], duplicates="drop")
        except Exception: continue
        if qs.isna().all(): continue
        bp = sub.loc[vals.index].assign(b=qs).groupby("b")["day_pnl_pct"].mean()
        if len(bp) < 2: continue
        spread = bp.max() - bp.min()
        rankings.append((col, bp, spread))

    rankings.sort(key=lambda r: -r[2])
    print(f"\n  TOP CONTINUOUS FEATURES (mean PnL% by tercile):")
    print(f"  {'feature':<26} {'low':>8} {'mid':>8} {'high':>8} {'spread':>8}")
    for col, bp, spread in rankings[:n_top]:
        lo = bp.get('low', np.nan); md = bp.get('mid', np.nan); hi = bp.get('high', np.nan)
        print(f"  {col:<26} {lo:>+7.2f}% {md:>+7.2f}% {hi:>+7.2f}% {spread:>+7.2f}pp")

    print(f"\n  BY DAY-OF-WEEK:")
    g = sub.groupby("dow").agg(
        n=("day_pnl_pct", "count"),
        mean=("day_pnl_pct", "mean"),
        wr=("day_pnl", lambda s: (s > 0).mean() * 100)
    ).sort_values("mean", ascending=False)
    for k, r in g.iterrows():
        marker = "  ** BEST **" if r["mean"] == g["mean"].max() else ("  ** WORST **" if r["mean"] == g["mean"].min() else "")
        print(f"    {str(k):<12} n={int(r['n']):>3}  mean={r['mean']:>+6.2f}%  WR={r['wr']:>5.1f}%{marker}")

    print(f"\n  BY MONTH (sorted by mean):")
    g = sub.groupby("month").agg(
        n=("day_pnl_pct", "count"),
        mean=("day_pnl_pct", "mean"),
        wr=("day_pnl", lambda s: (s > 0).mean() * 100)
    ).sort_values("mean", ascending=False)
    for k, r in g.iterrows():
        if r["n"] < 10: continue  # skip thin months
        print(f"    month {int(k):>2}  n={int(r['n']):>3}  mean={r['mean']:>+6.2f}%  WR={r['wr']:>5.1f}%")

    print(f"\n  BY YEAR:")
    g = sub.groupby("year").agg(
        n=("day_pnl_pct", "count"),
        mean=("day_pnl_pct", "mean"),
        wr=("day_pnl", lambda s: (s > 0).mean() * 100)
    )
    for k, r in g.iterrows():
        print(f"    {str(k):<6} n={int(r['n']):>3}  mean={r['mean']:>+6.2f}%  WR={r['wr']:>5.1f}%")

    # Macro buckets
    if len(sub) >= 60:
        print(f"\n  MACRO COMBINATIONS:")
        try:
            sub["vix_bucket"] = pd.qcut(sub["VIX_close"], 3, labels=["lowVIX","midVIX","highVIX"], duplicates="drop")
            sub["dxy_dir"] = pd.qcut(sub["DXY_1d"], 2, labels=["DXYdown","DXYup"], duplicates="drop")
            g = sub.groupby(["vix_bucket", "dxy_dir"]).agg(
                n=("day_pnl_pct", "count"), mean=("day_pnl_pct", "mean"),
                wr=("day_pnl", lambda s: (s > 0).mean() * 100)).reset_index()
            g = g[g["n"] >= 8].sort_values("mean", ascending=False)
            print(f"    VIX x DXY:")
            for _, r in g.iterrows():
                print(f"      {str(r['vix_bucket']):<10} {str(r['dxy_dir']):<10} n={int(r['n']):>3}  "
                      f"mean={r['mean']:>+6.2f}%  WR={r['wr']:>5.1f}%")

            sub["btc_bucket"] = pd.qcut(sub["BTC_5d"], 3, labels=["BTCdown","BTCflat","BTCup"], duplicates="drop")
            g = sub.groupby("btc_bucket").agg(
                n=("day_pnl_pct", "count"), mean=("day_pnl_pct", "mean"),
                wr=("day_pnl", lambda s: (s > 0).mean() * 100))
            print(f"\n    BTC 5d momentum:")
            for k, r in g.iterrows():
                print(f"      {str(k):<10} n={int(r['n']):>3}  mean={r['mean']:>+6.2f}%  WR={r['wr']:>5.1f}%")
        except Exception as e:
            print(f"    Macro analysis failed: {e}")

    return {
        "regime": regime_name,
        "n_days": n,
        "wr": wr,
        "mean_pnl_pct": mean_pnl_pct,
        "top_features": [(c, sp) for c, _, sp in rankings[:5]],
    }


def main():
    df = pd.read_csv(CSV)
    print(f"Total day-rows: {len(df)}")
    print(f"Shapes seen: {dict(df['shape'].value_counts())}")

    summaries = []
    for regime_name, shapes in REGIME_GROUPS.items():
        result = analyze_regime(df, regime_name, shapes)
        if result: summaries.append(result)

    # Compare top features across regimes
    print(f"\n\n{'='*86}")
    print(f"CROSS-REGIME COMPARISON: top-5 features by spread")
    print(f"{'='*86}")
    print(f"  {'regime':<18} {'n':>5} {'WR':>6} {'mean_pnl%':>10}  top features")
    for s in summaries:
        feats = "  ".join(f"{c}={sp:+.2f}pp" for c, sp in s["top_features"])
        print(f"  {s['regime']:<18} {s['n_days']:>5} {s['wr']:>5.1f}% {s['mean_pnl_pct']:>+8.2f}%  {feats}")


if __name__ == "__main__":
    main()
