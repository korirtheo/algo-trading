"""Do the shape buckets each have a characteristic intraday signature?

For each day in results/squeeze_taxonomy_summary.csv we load the leader's
2-min intraday CSV and compute per-day features:
  - max_gain_pct_from_open
  - time_to_peak_min (regular-hours minutes between 09:30 ET open and HOD)
  - eod_pct_from_open      (% above/below open at 15:45 ET cutoff)
  - reversal_pct           (1 - eod/peak)  — how much of the peak faded
  - max_drawdown_after_peak
  - first_30min_vol_share  (volume in first 30 min / day volume)
  - n_halt_resumes         (# of bars with vol > 5x rolling avg following
                            a flat-line gap > 60s)

Then we aggregate per shape (median + IQR) and chart the median normalized
price path per shape. Output:
  results/shape_intraday_signatures.csv
  results/shape_intraday_path_median.png
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

YEAR_TO_DIRS = {
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

RTH_START = "09:30"
RTH_END   = "15:45"


def _find_intraday(year, leader):
    if not isinstance(leader, str) or not leader:
        return None
    for d in YEAR_TO_DIRS.get(year, []):
        path = os.path.join(d, "intraday", f"{leader}.csv")
        if os.path.exists(path):
            return path
    return None


def _load_day_bars(path, date):
    """Return DataFrame of bars for the given date in regular hours (09:30-15:45 ET)."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None
    # find timestamp col (these CSVs use 'Datetime')
    ts_col = next((c for c in df.columns
                   if c.lower() in ("timestamp", "time", "t", "date_time", "datetime")), None)
    if ts_col is None:
        return None
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    # Convert to ET
    df["et"] = df[ts_col].dt.tz_convert("America/New_York")
    df = df[df["et"].dt.strftime("%Y-%m-%d") == date]
    if df.empty:
        return None
    df["hhmm"] = df["et"].dt.strftime("%H:%M")
    df = df[(df["hhmm"] >= RTH_START) & (df["hhmm"] <= RTH_END)]
    df = df.sort_values("et")
    # normalize column names (CSVs use Capitalized OHLCV)
    cols_lower = {c.lower(): c for c in df.columns}
    df = df.rename(columns={
        cols_lower.get("open", "open"):  "open",
        cols_lower.get("high", "high"):  "high",
        cols_lower.get("low",  "low"):   "low",
        cols_lower.get("close","close"): "close",
        cols_lower.get("volume","volume"): "volume",
    })
    keep = [c for c in ["open","high","low","close","volume","et"] if c in df.columns]
    return df[keep].reset_index(drop=True) if not df.empty else None


def _day_signature(bars):
    """Return per-day intraday features from RTH bars."""
    if bars is None or len(bars) < 3:
        return None
    open_px = float(bars["open"].iloc[0])
    if open_px <= 0:
        return None
    high_idx = int(bars["high"].idxmax())
    peak_px = float(bars["high"].iloc[high_idx])
    eod_px  = float(bars["close"].iloc[-1])
    minutes_from_open = (bars["et"] - bars["et"].iloc[0]).dt.total_seconds() / 60
    time_to_peak = float(minutes_from_open.iloc[high_idx])

    max_gain_pct = (peak_px - open_px) / open_px * 100
    eod_pct = (eod_px - open_px) / open_px * 100
    reversal_pct = (peak_px - eod_px) / peak_px * 100 if peak_px > 0 else 0
    # max drawdown after peak
    after = bars.iloc[high_idx:]
    if len(after) > 1:
        dd = (after["low"].min() - peak_px) / peak_px * 100
    else:
        dd = 0.0

    total_vol = bars["volume"].sum() or 1
    first_30 = bars[minutes_from_open <= 30]["volume"].sum()
    first_30_share = first_30 / total_vol * 100

    return {
        "open": open_px,
        "peak": peak_px,
        "eod": eod_px,
        "max_gain_pct": max_gain_pct,
        "time_to_peak_min": time_to_peak,
        "eod_pct_from_open": eod_pct,
        "reversal_pct_from_peak": reversal_pct,
        "max_dd_after_peak": dd,
        "first_30min_vol_share": first_30_share,
        "n_bars": len(bars),
    }


def _process(row):
    date, leader, year, shape = row["date"], row["leader"], row["year"], row["shape"]
    if not isinstance(leader, str) or shape in ("empty",):
        return None
    path = _find_intraday(year, leader)
    if path is None:
        return None
    bars = _load_day_bars(path, date)
    sig = _day_signature(bars)
    if sig is None:
        return None
    sig.update({"date": date, "leader": leader, "year": year, "shape": shape})
    # also keep a sparse 26-point normalized path (each 15-min slot from open)
    if bars is not None and len(bars) > 5:
        open_px = bars["open"].iloc[0]
        bars["min_since"] = (bars["et"] - bars["et"].iloc[0]).dt.total_seconds() // 60
        # bucket into 15-min slots 0..375  (6h 15m)
        bars["slot"] = (bars["min_since"] // 15).astype(int)
        slot_close = bars.groupby("slot")["close"].last()
        for s in range(0, 26):
            sig[f"slot_{s:02d}"] = float(slot_close.get(s, np.nan) / open_px - 1.0) * 100 \
                                    if s in slot_close.index else np.nan
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/squeeze_taxonomy_summary.csv")
    ap.add_argument("--out", default="results/shape_intraday_signatures.csv")
    ap.add_argument("--chart", default="results/shape_intraday_path_median.png")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["shape"].notna() & df["leader"].notna()]
    df["year"] = df["year"].astype(str)
    print(f"Processing {len(df)} days (parallel x{args.workers})...")

    rows = []
    records = df[["date","leader","year","shape"]].to_dict(orient="records")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process, r) for r in records]
        for i, fut in enumerate(as_completed(futs), 1):
            sig = fut.result()
            if sig is not None:
                rows.append(sig)
            if i % 100 == 0:
                print(f"  {i}/{len(records)} processed, {len(rows)} kept")

    if not rows:
        print("No usable rows.")
        return
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nSaved {len(out)} rows -> {args.out}")

    # --- Per-shape aggregate (medians + IQR) ---
    key_cols = ["max_gain_pct","time_to_peak_min","eod_pct_from_open",
                "reversal_pct_from_peak","max_dd_after_peak",
                "first_30min_vol_share"]
    print("\n" + "=" * 92)
    print("PER-SHAPE INTRADAY SIGNATURE (median values across leader's RTH bars)")
    print("=" * 92)
    agg_med = out.groupby("shape")[key_cols].median().round(1)
    agg_n   = out.groupby("shape").size().rename("n_days")
    summary = pd.concat([agg_n, agg_med], axis=1)
    print(summary.to_string())

    # Also IQR width
    iqr = out.groupby("shape")[key_cols].quantile(0.75) - out.groupby("shape")[key_cols].quantile(0.25)
    print("\n" + "=" * 92)
    print("IQR WIDTH PER SHAPE (75th - 25th percentile) — small = consistent")
    print("=" * 92)
    print(iqr.round(1).to_string())

    # --- Chart: median normalized price path per shape ---
    slot_cols = [c for c in out.columns if c.startswith("slot_")]
    if slot_cols:
        med_path = out.groupby("shape")[slot_cols].median()
        n_min = [int(c.split("_")[1]) * 15 for c in slot_cols]
        fig, ax = plt.subplots(figsize=(13, 6))
        for shape in med_path.index:
            if shape in ("empty","other"): continue
            n = (out["shape"] == shape).sum()
            ax.plot(n_min, med_path.loc[shape].values,
                    color=SHAPE_COLORS.get(shape, "#444"),
                    label=f"{shape} (n={n})", linewidth=2.2, marker="o", markersize=4)
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.set_xlabel("Minutes from 09:30 ET open")
        ax.set_ylabel("Median % change from open (leader)")
        ax.set_title("Leader's median intraday path by shape bucket")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.chart, dpi=140)
        plt.close(fig)
        print(f"\nWrote {args.chart}")


if __name__ == "__main__":
    main()
