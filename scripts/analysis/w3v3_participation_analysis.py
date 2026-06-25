"""Analyze the actual participation rate of W3 v3 #488 trades on forward 2024.

For each trade taken by the best W3 v3 trial:
  - Look up the bar at entry time
  - Compute V_eff_adj (the multiwindow effective liquidity)
  - Compute participation = position_cost / V_eff_adj
  - Aggregate distribution

Reports:
  - Median, 75th, 90th, 99th percentile participation
  - Count of vol_capped trades (cap was hit)
  - Distribution by bucket (<2%, 2-5%, 5-10%, 10-15%, >15%)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

W3_V3_BEST = "results/walk_forward_v3/W3_train_2021_2022_2023_test_2024_best.json"
BASELINE = "config/trial_432_params.json"
TEST_YEAR_DIRS = ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
                   "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"]
STARTING_CASH = 25_000


def _merged_params(best):
    with open(BASELINE) as f:
        b = json.load(f)
    m = dict(b)
    m.update(best["params"])
    return m


def main():
    print("Loading W3 v3 best params...")
    with open(W3_V3_BEST) as f:
        best = json.load(f)
    print(f"  Trial #{best['trial_number']}  score=${best['score']:,.0f}")

    print("\nLoading 2024 picks...")
    dirs = [d for d in TEST_YEAR_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    test_dates = [d for d in all_dates if d.startswith("2024")]
    print(f"  {len(test_dates)} test days")

    set_strategy_params(_merged_params(best))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08

    cash = STARTING_CASH
    trades = []
    for d in test_dates:
        day_picks = picks.get(d, [])
        if not day_picks: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception as e:
            continue
        for s in states:
            if s.get("exit_reason") is None or s.get("position_cost", 0) <= 0:
                continue
            # Find the entry bar and compute V_eff_adj at that bar
            mh = s["mh"]
            ts_entry = s.get("entry_time")
            if ts_entry is None: continue
            fill_price = float(mh.loc[ts_entry]["Open"]) if ts_entry in mh.index else 0
            if fill_price <= 0: continue
            v_eff_adj, v_2min, v_local, v_regime = tgc._multi_window_effective_volume(
                mh, ts_entry, fill_price)
            cum_vol = 0
            pre = mh.loc[mh.index <= ts_entry]
            if len(pre) > 0:
                cum_vol = float((pre["Volume"] * pre["Close"]).sum())
            pos = s["position_cost"]
            part_v_eff = pos / v_eff_adj * 100 if v_eff_adj > 0 else 0
            part_cum = pos / cum_vol * 100 if cum_vol > 0 else 0
            part_regime = pos / v_regime * 100 if v_regime > 0 else 0
            trades.append({
                "date": d, "ticker": s["ticker"], "strategy": s["strategy"],
                "position": pos, "v_eff_adj": v_eff_adj, "v_cum": cum_vol,
                "v_regime": v_regime,
                "part_v_eff_pct": part_v_eff,
                "part_cum_pct": part_cum,
                "part_regime_pct": part_regime,
                "vol_capped": s.get("vol_capped", False),
                "pct_pnl": (s["pnl"] / pos * 100) if pos > 0 else 0,
            })
        cash = end_c
        if is_cash: cash += unset

    if not trades:
        print("No trades to analyze.")
        return

    print(f"\n{len(trades)} trades analyzed.")
    print("=" * 88)
    print("PARTICIPATION DISTRIBUTION — W3 v3 #488 forward 2024")
    print("=" * 88)

    for label, key in [
        ("% of V_eff_adj (execution cap=15%)", "part_v_eff_pct"),
        ("% of V_regime  (regime cap=8%)",     "part_regime_pct"),
        ("% of cumulative $vol (sanity=5%)",   "part_cum_pct"),
    ]:
        vals = np.array([t[key] for t in trades])
        print(f"\n  {label}")
        print(f"    Median:    {np.median(vals):>6.2f}%")
        print(f"    Mean:      {vals.mean():>6.2f}%")
        print(f"    75th pct:  {np.percentile(vals, 75):>6.2f}%")
        print(f"    90th pct:  {np.percentile(vals, 90):>6.2f}%")
        print(f"    99th pct:  {np.percentile(vals, 99):>6.2f}%")
        print(f"    Max:       {vals.max():>6.2f}%")

    # Buckets for V_eff (execution measure)
    vals = np.array([t["part_v_eff_pct"] for t in trades])
    print(f"\n  Trade count by V_eff_adj participation bucket:")
    buckets = [
        ("<2%",       0,  2),
        ("2-5%",      2,  5),
        ("5-10%",     5, 10),
        ("10-15%",   10, 15),
        ("15-25%",   15, 25),
        (">25%",     25, 999),
    ]
    for label, lo, hi in buckets:
        n = ((vals >= lo) & (vals < hi)).sum()
        pct = n / len(vals) * 100
        print(f"    {label:<10} {n:>4}  ({pct:.1f}%)")

    vol_capped = sum(1 for t in trades if t["vol_capped"])
    print(f"\n  Vol-capped trades: {vol_capped}/{len(trades)} ({100*vol_capped/len(trades):.1f}%)")

    # Per-strategy breakdown
    by_strat = {}
    for t in trades:
        by_strat.setdefault(t["strategy"], []).append(t)
    print(f"\n  Per-strategy participation (median % of V_eff_adj):")
    for k, ts in sorted(by_strat.items()):
        vs = np.array([t["part_v_eff_pct"] for t in ts])
        pnls = np.array([t["pct_pnl"] for t in ts])
        print(f"    {k}: {len(ts):>3} trades  median_part={np.median(vs):>5.2f}%  "
              f"median_pnl={np.median(pnls):+5.2f}%")


if __name__ == "__main__":
    main()
