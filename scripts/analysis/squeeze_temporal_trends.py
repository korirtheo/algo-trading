"""Temporal trend analysis on the squeeze taxonomy.

Operates on results/squeeze_taxonomy_summary.csv (produced by
squeeze_taxonomy_2021_2026.py).

Questions answered:
  1. Rolling 5d / 20d shape mix — is today's regime different from
     last quarter's?
  2. Shape transition matrix — given yesterday's shape, what's today?
     (high diagonal = sticky regime; high off-diag = noisy day-to-day)
  3. Autocorrelation of `n_above_50` — is the squeeze signal serially
     correlated? If yes, "what was yesterday" is a feature.
  4. Quarter-by-quarter shape mix — discrete view of regime drift.
  5. Last-30-days vs all-time baseline — current regime call-out.

Writes:
  results/temporal_shape_quarterly.csv
  results/temporal_shape_rolling20.png
  results/temporal_shape_transition.csv
"""
import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHAPE_COLORS = {
    "broad-squeeze":  "#d62728",
    "liquid-normal":  "#2ca02c",
    "thin-microcap":  "#ff7f0e",
    "microcap-thin":  "#9467bd",
    "mega-cap":       "#17becf",
    "corp-action":    "#888888",
    "empty":          "#dddddd",
    "other":          "#444444",
}


def load(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def quarterly_mix(df):
    df = df.copy()
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    pivot = (df.groupby(["quarter", "shape"]).size()
             .unstack(fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    return pivot, pivot_pct


def rolling_mix(df, win=20):
    df = df.copy()
    shapes = sorted(df["shape"].dropna().unique())
    # one-hot encode shapes
    for s in shapes:
        df[f"is_{s}"] = (df["shape"] == s).astype(int)
    cols = [f"is_{s}" for s in shapes]
    df[cols] = df[cols].rolling(win, min_periods=1).mean() * 100
    return df[["date"] + cols], shapes


def transition_matrix(df):
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["prev_shape"] = df["shape"].shift(1)
    tx = pd.crosstab(df["prev_shape"], df["shape"], normalize="index") * 100
    return tx.round(1)


def autocorr_n50(df, lags=(1, 2, 3, 5, 10)):
    s = df.sort_values("date")["n_above_50"].fillna(0)
    out = {}
    for k in lags:
        s1, s2 = s.iloc[k:].values, s.iloc[:-k].values
        if len(s1) > 5 and np.std(s1) > 0 and np.std(s2) > 0:
            out[k] = float(np.corrcoef(s1, s2)[0, 1])
        else:
            out[k] = None
    return out


def recent_vs_all(df, recent_days=30):
    """Last-30-trading-days shape distribution vs all-time."""
    recent = df.sort_values("date").tail(recent_days)
    all_pct = df["shape"].value_counts(normalize=True) * 100
    rec_pct = recent["shape"].value_counts(normalize=True) * 100
    out = pd.DataFrame({"all_time_%": all_pct, "recent_30d_%": rec_pct}).fillna(0)
    out["delta_pp"] = out["recent_30d_%"] - out["all_time_%"]
    return out.sort_values("delta_pp", ascending=False)


def plot_rolling(df_roll, shapes, out_path, title):
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(df_roll))
    for s in shapes:
        if f"is_{s}" not in df_roll.columns:
            continue
        vals = df_roll[f"is_{s}"].values
        ax.fill_between(df_roll["date"], bottom, bottom + vals,
                        label=s, color=SHAPE_COLORS.get(s, "#444"),
                        alpha=0.85, linewidth=0)
        bottom = bottom + vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of rolling-20-day window (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/squeeze_taxonomy_summary.csv")
    ap.add_argument("--win", type=int, default=20)
    ap.add_argument("--recent-days", type=int, default=30)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    df = load(args.csv)
    print(f"Loaded {len(df)} day-rows: {df['date'].min().date()} -> {df['date'].max().date()}")

    # --- Quarterly mix ---
    print("\n" + "=" * 88)
    print("QUARTERLY SHAPE MIX (%)")
    print("=" * 88)
    pivot, pivot_pct = quarterly_mix(df)
    print(pivot_pct.round(1).to_string())
    qcsv = os.path.join(args.outdir, "temporal_shape_quarterly.csv")
    pivot_pct.round(2).to_csv(qcsv)
    print(f"\n  wrote {qcsv}")

    # --- Rolling 20d ---
    df_roll, shapes = rolling_mix(df, win=args.win)
    plot_rolling(df_roll, shapes,
                 os.path.join(args.outdir, f"temporal_shape_rolling{args.win}.png"),
                 f"Rolling-{args.win}-day shape mix (2019-2026)")

    # --- Transition matrix ---
    print("\n" + "=" * 88)
    print("YESTERDAY -> TODAY SHAPE TRANSITION (% of rows; diagonal = sticky)")
    print("=" * 88)
    tx = transition_matrix(df)
    print(tx.to_string())
    tx.to_csv(os.path.join(args.outdir, "temporal_shape_transition.csv"))

    # --- Autocorrelation on n_above_50 ---
    print("\n" + "=" * 88)
    print("n_above_50 AUTOCORRELATION (1=perfect persistence, 0=random)")
    print("=" * 88)
    ac = autocorr_n50(df)
    for k, v in ac.items():
        if v is None:
            print(f"  lag-{k}: n/a")
        else:
            print(f"  lag-{k}: {v:+.3f}")

    # --- Recent vs all-time ---
    print("\n" + "=" * 88)
    print(f"LAST {args.recent_days} DAYS vs ALL-TIME SHAPE MIX")
    print("=" * 88)
    rva = recent_vs_all(df, recent_days=args.recent_days)
    print(rva.round(1).to_string())

    # --- Sticky-regime callout ---
    diag = pd.Series({s: tx.loc[s, s] if s in tx.index else 0 for s in shapes})
    print("\n" + "=" * 88)
    print("SHAPE STICKINESS (today's-shape = yesterday's-shape %)")
    print("=" * 88)
    for s, v in diag.sort_values(ascending=False).items():
        baseline = (df["shape"] == s).mean() * 100
        ratio = v / baseline if baseline > 0 else 0
        print(f"  {s:<16}  P(today=this | yesterday=this)={v:>5.1f}%  "
              f"baseline={baseline:>5.1f}%  lift={ratio:.2f}x")


if __name__ == "__main__":
    main()
