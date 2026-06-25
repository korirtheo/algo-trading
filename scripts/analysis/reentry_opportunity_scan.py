"""Scan 2024 picks for mid-day re-entry opportunities.

For each pick we look at the intraday 2-min bars and ask:
  1. Did the stock have a clear "first leg" up (peak high within first 30 min)?
  2. Did it then pull back (low somewhere between peak time and afterward)?
  3. Did it RECOVER to >= 90% of the first-leg high?
  4. Did it BREAK the first-leg high (= true second leg)?

Aggregates:
  - % of picks with measurable first-leg highs
  - % of those with a >=10% pullback
  - % of those that recovered to >=90% of peak (worth a re-entry attempt)
  - % of those that broke the peak (the actual second-leg trades)
  - Median PnL opportunity from a perfect re-entry trade
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

YEAR_DIRS = {
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
             "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
}

RTH_START = "09:30"
RTH_END   = "15:45"


def _load_day_bars(path, date):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if df.empty: return None
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df["et"] = df["Datetime"].dt.tz_convert("America/New_York")
    df = df[df["et"].dt.strftime("%Y-%m-%d") == date]
    if df.empty: return None
    df["hhmm"] = df["et"].dt.strftime("%H:%M")
    df = df[(df["hhmm"] >= RTH_START) & (df["hhmm"] <= RTH_END)]
    df = df.sort_values("et").reset_index(drop=True)
    return df if len(df) >= 5 else None


def _find_first_leg(bars):
    """Return (peak_idx, peak_high) for the first significant first-leg peak.

    Defines first leg as: peak found in first 30 RTH minutes (15 bars).
    """
    if len(bars) < 5: return None
    first_window = bars.iloc[:min(15, len(bars))]
    peak_idx = int(first_window["High"].idxmax())
    return peak_idx, float(bars["High"].iloc[peak_idx])


def _analyze_one(args):
    path, date, ticker, prev_close = args
    bars = _load_day_bars(path, date)
    if bars is None: return None
    open_px = float(bars["Open"].iloc[0])
    if open_px <= 0: return None
    fl = _find_first_leg(bars)
    if fl is None: return None
    peak_idx, peak_high = fl
    if peak_idx >= len(bars) - 5: return None  # peak too late, no room for second leg

    # First-leg gain
    fl_gain_pct = (peak_high - open_px) / open_px * 100

    # Post-peak section
    post = bars.iloc[peak_idx + 1:]
    if len(post) < 3: return None
    trough_idx = int(post["Low"].idxmin())
    trough_low = float(post["Low"].iloc[trough_idx - (peak_idx + 1)]) if trough_idx >= peak_idx + 1 else None
    if trough_low is None or trough_low <= 0: return None

    # Pullback depth from peak
    pullback_pct = (peak_high - trough_low) / peak_high * 100

    # Post-trough recovery — was there a re-high?
    post_trough = bars.iloc[trough_idx:]
    if len(post_trough) < 2: return None
    second_peak = float(post_trough["High"].max())
    recovery_pct = (second_peak - trough_low) / trough_low * 100 if trough_low > 0 else 0
    broke_first_leg = second_peak > peak_high

    return {
        "date": date,
        "ticker": ticker,
        "open": open_px,
        "first_leg_peak": peak_high,
        "first_leg_gain_pct": fl_gain_pct,
        "trough_after_first": trough_low,
        "pullback_pct": pullback_pct,
        "second_peak": second_peak,
        "recovery_pct": recovery_pct,
        "broke_first_leg": broke_first_leg,
        "second_leg_gain_pct": (second_peak - trough_low) / trough_low * 100,
        "ideal_reentry_pnl_pct": (second_peak - trough_low) / trough_low * 100 if recovery_pct > 5 else 0,
    }


def main():
    print("Scanning 2024 picks for re-entry opportunities...")
    all_args = []
    for y, dirs in YEAR_DIRS.items():
        for ddir in dirs:
            if not os.path.exists(ddir): continue
            # Get picks
            picks_csv = os.path.join(ddir, "daily_top_gainers.csv")
            if not os.path.exists(picks_csv): continue
            gdf = pd.read_csv(picks_csv)
            for _, row in gdf.iterrows():
                tk = row.get("ticker", "")
                date = str(row.get("date", ""))
                prev = row.get("prev_close", 0)
                path = os.path.join(ddir, "intraday", f"{tk}.csv")
                if os.path.exists(path):
                    all_args.append((path, date, tk, prev))

    print(f"  {len(all_args)} ticker-days to analyze")

    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_analyze_one, a) for a in all_args[:5000]]  # cap for speed
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is not None:
                rows.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(futs)} done, {len(rows)} valid")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid analysis rows.")
        return
    out_path = "results/reentry_opportunity_2024.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} ticker-days to {out_path}")

    # Aggregate stats
    n = len(df)
    has_first_leg = (df["first_leg_gain_pct"] >= 5).sum()
    has_pullback = (df["pullback_pct"] >= 10).sum()
    decent_recovery = ((df["recovery_pct"] >= 5) & (df["pullback_pct"] >= 10)).sum()
    broke_high = (df["broke_first_leg"] & (df["pullback_pct"] >= 10)).sum()

    print(f"\nRESULTS — 2024 picks ({n} ticker-days analyzed)")
    print("=" * 72)
    print(f"  Had clear first-leg gain >=5%:         {has_first_leg:>5} ({100*has_first_leg/n:.1f}%)")
    print(f"  Of those: >=10% pullback from peak:    {has_pullback:>5} ({100*has_pullback/n:.1f}% of all)")
    print(f"  Of those: recovered >=5% from trough:  {decent_recovery:>5} ({100*decent_recovery/n:.1f}% of all)")
    print(f"  Of those: broke the first-leg high:    {broke_high:>5} ({100*broke_high/n:.1f}% of all)")

    if has_pullback > 0:
        eligible = df[(df["pullback_pct"] >= 10) & (df["recovery_pct"] >= 5)]
        print(f"\nFor the {len(eligible)} eligible re-entry candidates:")
        print(f"  Median pullback depth:           {eligible['pullback_pct'].median():.1f}%")
        print(f"  Median recovery from trough:     {eligible['recovery_pct'].median():.1f}%")
        print(f"  Median second-peak vs first-pk:  {(eligible['second_peak']/eligible['first_leg_peak']*100).median():.1f}%")
        print(f"  Median ideal-reentry PnL %:      {eligible['ideal_reentry_pnl_pct'].median():.1f}%")

    # By month
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%Y-%m")
    print(f"\nMonthly trend (% of days with eligible re-entry):")
    monthly = df.groupby("month").apply(
        lambda g: 100 * ((g["pullback_pct"] >= 10) & (g["recovery_pct"] >= 5)).sum() / len(g),
        include_groups=False
    )
    for m, v in monthly.items():
        print(f"  {m}  {v:5.1f}%")


if __name__ == "__main__":
    main()
