"""Tune the regime-classifier thresholds analytically.

Approach: each "trial" of a threshold combo doesn't re-run the simulator.
Instead we re-label our existing 1,286 per-day records (each tagged with
trial #6's actual PnL on that day) using the candidate thresholds, then
aggregate per-regime PnL/day. Sweep is sub-second; lets us see exactly
which thresholds produce the cleanest squeeze-vs-normal-vs-dead separation.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
import glob
import numpy as np
import pandas as pd

from test_full import load_all_picks

DATA_DIRS = [
    "stored_data_2021", "stored_data_2022", "stored_data_2023",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data", "stored_data_mar_may_2026",
]


def local_classify(n20, n50, max_gap, med_vol, *, dead_n20_max, dead_max_gap_max,
                    dead_med_vol_max, squeeze_n50_min, squeeze_max_gap_min):
    """Local copy of the classifier with explicit thresholds."""
    if squeeze_n50_min == 0 and squeeze_max_gap_min == 0:
        pass  # disabled
    if (n50 >= squeeze_n50_min) or (max_gap > squeeze_max_gap_min):
        return "squeeze"
    cond_low_density = (n20 < dead_n20_max) and (max_gap < dead_max_gap_max)
    cond_low_vol = med_vol < dead_med_vol_max
    if cond_low_density and cond_low_vol:
        return "dead"
    return "normal"


def main():
    # 1. Load all picks, compute per-day features
    print("Loading all picks and computing per-day features...")
    all_dates, daily_picks = load_all_picks(DATA_DIRS)

    rows = []
    for d in all_dates:
        picks = daily_picks.get(d, [])
        if not picks:
            rows.append({"date": d, "n20": 0, "n50": 0, "max_gap": 0.0, "med_vol": 0.0})
            continue
        gaps = [float(p.get("gap_pct", 0.0)) for p in picks]
        vols = [float(p.get("pm_volume", p.get("volume", 0.0))) for p in picks]
        rows.append({
            "date": d,
            "n20": sum(1 for g in gaps if g >= 20),
            "n50": sum(1 for g in gaps if g >= 50),
            "max_gap": max(gaps),
            "med_vol": float(np.median(vols)) if vols else 0.0,
        })
    features = pd.DataFrame(rows)
    print(f"  {len(features)} days")

    # 2. Load baseline run (trial #6 on every day) for per-day PnL
    base_csvs = sorted(glob.glob("results/regime_gate/baseline_*.csv"))
    if not base_csvs:
        print("ERROR: no baseline_*.csv found in results/regime_gate/")
        sys.exit(1)
    pnl_csv = base_csvs[-1]
    print(f"Loading baseline PnL from {pnl_csv}")
    pnl_df = pd.read_csv(pnl_csv)
    pnl_df['date'] = pnl_df['date'].astype(str)
    features['date'] = features['date'].astype(str)
    df = features.merge(pnl_df[['date', 'pnl']], on='date', how='inner')
    print(f"  Merged: {len(df)} days with both features and PnL")

    # 3. Sweep thresholds
    print("\nSweeping thresholds...")
    grid_dead_n20 = [0, 1, 2, 3, 4]
    grid_dead_max_gap = [20.0, 25.0, 30.0, 40.0, 50.0]
    grid_dead_med_vol = [200_000.0, 500_000.0, 1_000_000.0]
    grid_sq_n50 = [1, 2, 3, 4, 5]
    grid_sq_max_gap = [50.0, 75.0, 100.0, 150.0, 200.0]

    results = []
    total = (len(grid_dead_n20) * len(grid_dead_max_gap) * len(grid_dead_med_vol)
             * len(grid_sq_n50) * len(grid_sq_max_gap))
    print(f"  {total} threshold combinations...")

    # Pull arrays once
    n20 = df['n20'].to_numpy()
    n50 = df['n50'].to_numpy()
    max_gap = df['max_gap'].to_numpy()
    med_vol = df['med_vol'].to_numpy()
    pnl = df['pnl'].to_numpy()

    for d_n20 in grid_dead_n20:
        for d_mg in grid_dead_max_gap:
            for d_mv in grid_dead_med_vol:
                for s_n50 in grid_sq_n50:
                    for s_mg in grid_sq_max_gap:
                        # Vectorized classification
                        is_squeeze = (n50 >= s_n50) | (max_gap > s_mg)
                        is_dead = (~is_squeeze) & (n20 < d_n20) & (max_gap < d_mg) & (med_vol < d_mv)
                        is_normal = ~(is_squeeze | is_dead)

                        dead_days = int(is_dead.sum())
                        normal_days = int(is_normal.sum())
                        squeeze_days = int(is_squeeze.sum())
                        dead_ppd = float(pnl[is_dead].mean()) if dead_days else 0.0
                        normal_ppd = float(pnl[is_normal].mean()) if normal_days else 0.0
                        squeeze_ppd = float(pnl[is_squeeze].mean()) if squeeze_days else 0.0

                        # Separation score: how much extra does squeeze make over normal,
                        # rewarding strong separation and large day counts
                        separation = squeeze_ppd - normal_ppd
                        # Also: skip-quality (how little we lose by skipping dead)
                        dead_skip_quality = -dead_ppd  # higher = better skip
                        # Combined: prefer high separation + low dead PnL/day
                        score = (squeeze_ppd - normal_ppd) - 0.5 * dead_ppd

                        # Minimum-day-count constraint
                        valid = (dead_days >= 30 and normal_days >= 100 and squeeze_days >= 100)

                        results.append({
                            "d_n20": d_n20, "d_mg": d_mg, "d_mv_K": int(d_mv / 1000),
                            "s_n50": s_n50, "s_mg": s_mg,
                            "dead_days": dead_days, "normal_days": normal_days,
                            "squeeze_days": squeeze_days,
                            "dead_ppd": dead_ppd, "normal_ppd": normal_ppd,
                            "squeeze_ppd": squeeze_ppd,
                            "separation": separation,
                            "score": score,
                            "valid": valid,
                        })

    out = pd.DataFrame(results)
    out_valid = out[out['valid']].copy()

    print(f"\nTotal combinations: {len(out)}")
    print(f"Valid (min day counts met): {len(out_valid)}")

    # 4. Report top 10 by score
    print("\n" + "=" * 100)
    print("TOP 10 by separation score (squeeze$/day - normal$/day - 0.5*dead$/day)")
    print("=" * 100)
    cols_show = ['d_n20', 'd_mg', 'd_mv_K', 's_n50', 's_mg',
                  'dead_days', 'normal_days', 'squeeze_days',
                  'dead_ppd', 'normal_ppd', 'squeeze_ppd', 'separation', 'score']
    print(out_valid.nlargest(10, 'score')[cols_show].round(0).to_string(index=False))

    print("\nTOP 10 by raw separation (squeeze - normal $/day, ignoring dead)")
    print(out_valid.nlargest(10, 'separation')[cols_show].round(0).to_string(index=False))

    print("\nTOP 10 by HIGHEST squeeze $/day (most concentrated squeezes)")
    print(out_valid.nlargest(10, 'squeeze_ppd')[cols_show].round(0).to_string(index=False))

    print("\nCURRENT THRESHOLDS (regime_gate.py defaults):")
    cur = out[(out['d_n20'] == 2) & (out['d_mg'] == 30.0) & (out['d_mv_K'] == 500)
              & (out['s_n50'] == 3) & (out['s_mg'] == 100.0)]
    if not cur.empty:
        print(cur[cols_show].round(0).to_string(index=False))
    else:
        print("  (current thresholds not in grid)")

    # Save full results
    out.to_csv("results/regime_gate/threshold_sweep.csv", index=False)
    print("\nSaved full sweep to results/regime_gate/threshold_sweep.csv")


if __name__ == "__main__":
    import sys
    main()
