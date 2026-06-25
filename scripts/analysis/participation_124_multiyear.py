"""Participation + position-size analysis of #124 across 2024, 2025, 2026.

Reports per year:
  - Median, 75th, 90th percentile of position size ($)
  - Same for participation as % of V_eff_adj
  - Number / share of trades hitting the 15% cap
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

CONFIG = "config/trial_124_microcap_pump_extracted.json"
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000

YEAR_DIRS = {
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026"],
}


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def analyze_year(year, dirs):
    dirs_present = [d for d in dirs if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs_present)
    test_dates = [d for d in all_dates if d.startswith(year)]

    cash = STARTING_CASH
    trades = []
    for d in test_dates:
        day_picks = picks.get(d, [])
        if not day_picks: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
        except Exception:
            continue
        for s in states:
            if s.get("exit_reason") is None or s.get("position_cost", 0) <= 0:
                continue
            mh = s["mh"]; ts_entry = s.get("entry_time")
            if ts_entry is None: continue
            fill_price = float(mh.loc[ts_entry]["Open"]) if ts_entry in mh.index else 0
            if fill_price <= 0: continue
            v_eff_adj, v_2min, v_local, v_regime = tgc._multi_window_effective_volume(
                mh, ts_entry, fill_price)
            pos = s["position_cost"]
            part = pos / v_eff_adj * 100 if v_eff_adj > 0 else 0
            trades.append({
                "date": d, "ticker": s["ticker"], "strategy": s["strategy"],
                "position_cost": pos, "entry_equity": cash,
                "position_pct_equity": pos / cash * 100 if cash > 0 else 0,
                "part_v_eff_pct": part,
                "vol_capped": s.get("vol_capped", False),
            })
        cash = end_c
        if is_cash: cash += unset
    return trades, cash


def report_stats(year, trades, final_cash):
    if not trades:
        print(f"\n{year}: no trades"); return
    pos = np.array([t["position_cost"] for t in trades])
    part = np.array([t["part_v_eff_pct"] for t in trades])
    pos_pct_eq = np.array([t["position_pct_equity"] for t in trades])
    capped = sum(1 for t in trades if t["vol_capped"])

    print(f"\n{'='*72}")
    print(f"YEAR {year}  —  {len(trades)} trades  —  end equity ${final_cash:,.0f} "
          f"({final_cash/STARTING_CASH:.2f}x)")
    print(f"{'='*72}")
    print(f"  Position size ($):")
    print(f"    Median:    ${np.median(pos):>11,.0f}")
    print(f"    Mean:      ${pos.mean():>11,.0f}")
    print(f"    75th pct:  ${np.percentile(pos, 75):>11,.0f}")
    print(f"    90th pct:  ${np.percentile(pos, 90):>11,.0f}")
    print(f"    Max:       ${pos.max():>11,.0f}")
    print(f"\n  Position size as % of equity (per-trade):")
    print(f"    Median:    {np.median(pos_pct_eq):>6.1f}%")
    print(f"    75th pct:  {np.percentile(pos_pct_eq, 75):>6.1f}%")
    print(f"    90th pct:  {np.percentile(pos_pct_eq, 90):>6.1f}%")
    print(f"    Max:       {pos_pct_eq.max():>6.1f}%")
    print(f"\n  Participation as % of V_eff_adj  (execution cap 15%):")
    print(f"    Median:    {np.median(part):>6.2f}%")
    print(f"    Mean:      {part.mean():>6.2f}%")
    print(f"    75th pct:  {np.percentile(part, 75):>6.2f}%")
    print(f"    90th pct:  {np.percentile(part, 90):>6.2f}%")
    print(f"    99th pct:  {np.percentile(part, 99):>6.2f}%")
    print(f"    Max:       {part.max():>6.2f}%")
    print(f"\n  Vol-capped trades: {capped}/{len(trades)} ({100*capped/len(trades):.1f}%)")

    # Bucket counts
    buckets = [("<2%",0,2),("2-5%",2,5),("5-10%",5,10),
               ("10-15%",10,15),(">=15%",15,999)]
    print(f"\n  Trade count by participation bucket:")
    for label, lo, hi in buckets:
        n = ((part >= lo) & (part < hi)).sum()
        print(f"    {label:<8} {n:>4} ({100*n/len(trades):.1f}%)")


def main():
    with open(CONFIG) as f: cfg = json.load(f)
    print(f"Config: {CONFIG}")
    print(f"  Trial #{cfg.get('trial_number')}")

    set_strategy_params(_merged(cfg["params"]))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    all_summaries = []
    for year, dirs in YEAR_DIRS.items():
        print(f"\nAnalyzing {year}...")
        trades, final = analyze_year(year, dirs)
        report_stats(year, trades, final)
        if trades:
            all_summaries.append((year, len(trades), final,
                                   np.median([t["position_cost"] for t in trades]),
                                   np.median([t["part_v_eff_pct"] for t in trades])))

    print(f"\n{'='*72}")
    print(f"CROSS-YEAR HEADLINE SUMMARY")
    print(f"{'='*72}")
    print(f"  {'year':<6} {'trades':<8} {'final':<12} {'median_pos':<14} {'median_part':<12}")
    for y, n, fin, mpos, mpart in all_summaries:
        print(f"  {y:<6} {n:<8} ${fin:<11,.0f} ${mpos:<13,.0f} {mpart:>5.2f}%")


if __name__ == "__main__":
    main()
