"""Same predictive-features analysis but ONLY microcap-pump days.

Filters the existing predictive_features.csv to microcap-thin + thin-microcap,
runs the same heterogeneity/discrimination analysis. Answers the question:
"WITHIN microcap-pump days, what predicts a winning vs losing day?"
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

CSV = "results/regime_trends/predictive_features.csv"
MICROCAP_SHAPES = ["microcap-thin", "thin-microcap"]


def main():
    df = pd.read_csv(CSV)
    df = df[df["shape"].isin(MICROCAP_SHAPES)].copy()
    df = df[df["n_trades"] > 0].copy()
    print(f"Microcap-pump days with trades: {len(df)}")
    print(f"  Win days: {(df['day_pnl']>0).sum()} ({(df['day_pnl']>0).mean()*100:.1f}%)")
    print(f"  Median day PnL: ${df['day_pnl'].median():,.0f}")
    print(f"  Days by year: {dict(df.year.value_counts().sort_index())}")
    print(f"  Days by shape: {dict(df['shape'].value_counts())}")

    feature_cols = [c for c in df.columns if c not in (
        "date","year","dow","month","shape","starting_cash","ending_cash",
        "day_pnl","day_pnl_pct","n_trades")]

    print(f"\n{'='*78}")
    print(f"CONTINUOUS FEATURES (terciles within microcap-pump days only)")
    print(f"{'='*78}")
    print(f"  {'feature':<28} {'low_pnl':>10} {'mid_pnl':>10} {'high_pnl':>10} {'spread':>8}")

    rankings = []
    for col in feature_cols:
        vals = df[col].dropna()
        if len(vals) < 30 or vals.std() == 0:
            continue
        try:
            qs = pd.qcut(vals, 3, labels=["low","mid","high"], duplicates="drop")
        except Exception:
            continue
        if qs.isna().all():
            continue
        bucket_pnls = df.loc[vals.index].assign(b=qs).groupby("b")["day_pnl_pct"].mean()
        bucket_wr = df.loc[vals.index].assign(b=qs).groupby("b").apply(
            lambda s: (s["day_pnl"] > 0).mean() * 100)
        if len(bucket_pnls) < 2: continue
        spread = bucket_pnls.max() - bucket_pnls.min()
        rankings.append((col, bucket_pnls, bucket_wr, spread))

    rankings.sort(key=lambda r: -r[3])
    for col, bp, bwr, spread in rankings[:25]:
        lo = bp.get("low", float("nan")); md = bp.get("mid", float("nan")); hi = bp.get("high", float("nan"))
        wl = bwr.get("low", float("nan")); wm = bwr.get("mid", float("nan")); wh = bwr.get("high", float("nan"))
        print(f"  {col:<28} {lo:>+9.2f}% {md:>+9.2f}% {hi:>+9.2f}% {spread:>+7.2f}pp  "
              f"WR: {wl:.0f}/{wm:.0f}/{wh:.0f}%")

    print(f"\n{'='*78}")
    print(f"CATEGORICAL (microcap-pump days only)")
    print(f"{'='*78}")
    for col in ("dow", "month", "year", "shape"):
        g = df.groupby(col).agg(
            n=("day_pnl_pct", "count"),
            mean=("day_pnl_pct", "mean"),
            win_rate=("day_pnl", lambda s: (s > 0).mean() * 100),
        ).sort_values("mean", ascending=False)
        print(f"\n  {col.upper()}:")
        for k, r in g.iterrows():
            print(f"    {str(k):<14} n={int(r['n']):>4}  mean={r['mean']:>+7.2f}%  WR={r['win_rate']:>5.1f}%")

    # Year x DOW cross-tab — is the day-of-week effect stable across years?
    print(f"\n{'='*78}")
    print(f"YEAR x DOW (mean day_pnl_pct, microcap-pump only)")
    print(f"{'='*78}")
    pivot = df.pivot_table(index="year", columns="dow", values="day_pnl_pct", aggfunc="mean")
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    pivot = pivot.reindex(columns=[c for c in dow_order if c in pivot.columns])
    print(pivot.to_string(float_format=lambda v: f"{v:+.2f}%"))

    # Macro x microcap-pump deep dive
    print(f"\n{'='*78}")
    print(f"MACRO COMBINATIONS (look for compound effects)")
    print(f"{'='*78}")
    # VIX regime + DXY direction
    df["vix_bucket"] = pd.qcut(df["VIX_close"], 3, labels=["lowVIX","midVIX","highVIX"], duplicates="drop")
    df["dxy_dir"] = pd.qcut(df["DXY_1d"], 2, labels=["DXYdown","DXYup"], duplicates="drop")
    print("\n  VIX bucket x DXY direction (mean PnL%):")
    g = df.groupby(["vix_bucket", "dxy_dir"]).agg(
        n=("day_pnl_pct", "count"), mean=("day_pnl_pct", "mean"),
        wr=("day_pnl", lambda s: (s > 0).mean() * 100)).reset_index()
    g = g[g["n"] >= 10].sort_values("mean", ascending=False)
    for _, r in g.iterrows():
        print(f"    {r['vix_bucket']:<10} {r['dxy_dir']:<10} n={int(r['n']):>3}  mean={r['mean']:>+6.2f}%  WR={r['wr']:>5.1f}%")

    # BTC trend buckets
    df["btc_5d_bucket"] = pd.qcut(df["BTC_5d"], 3, labels=["BTCdown","BTCflat","BTCup"], duplicates="drop")
    print("\n  BTC 5-day momentum:")
    g = df.groupby("btc_5d_bucket").agg(
        n=("day_pnl_pct", "count"), mean=("day_pnl_pct", "mean"),
        wr=("day_pnl", lambda s: (s > 0).mean() * 100))
    for k, r in g.iterrows():
        print(f"    {str(k):<10} n={int(r['n']):>4}  mean={r['mean']:>+7.2f}%  WR={r['wr']:>5.1f}%")


if __name__ == "__main__":
    main()
