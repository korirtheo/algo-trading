"""Test whether X (Range Reversion) overlaps with G (Big Gap Runner) on 2026.

Forward-test:
  1. G only (with #254's tuned G params)
  2. X only (with #254's tuned X params, enable_x flipped on)
  3. G + X together

Then compare:
  - Trade counts each
  - PnL each
  - (date, ticker) pairs shared between G-only and X-only trades
  - Per-strategy attribution in G+X combined run
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict
import numpy as np

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
SOURCE_CONFIG = "config/trial_254_w7_extracted.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]


def make_only(base_params, enabled_set):
    p = dict(base_params)
    for s in ALL_STRATS:
        p[f"enable_{s}"] = (s in enabled_set)
    return p


def forward_with_trades(params, label):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(params)
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    trades = []
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks: continue
        if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as nfp
            day_picks = nfp(day_picks, d, tgc.NEWS_MIN_ARTICLES, tgc.NEWS_REQUIRE_CATALYST)
        if not day_picks: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                entry_time = str(st.get("entry_time", ""))
                trades.append({
                    "date": d,
                    "ticker": st["ticker"],
                    "strategy": st.get("strategy", "?"),
                    "entry_time": entry_time,
                    "pnl": float(st["pnl"]),
                    "cost": float(st["position_cost"]),
                    "pct": float(st["pnl"]) / float(st["position_cost"]) * 100 if st["position_cost"] > 0 else 0,
                    "exit_reason": str(st.get("exit_reason")),
                })
        cash = end_c + (unset if is_cash else 0)
    print(f"  {label}: {len(trades)} trades, final ${cash:,.0f}, PnL ${cash-STARTING_CASH:+,.0f}")
    return trades, cash


def main():
    with open(SOURCE_CONFIG) as f: data = json.load(f)
    src_params = data.get("params", data)

    print("Building 3 configs from #254's tuned params:")
    print("  G-only:    enable_g=True, all others False")
    print("  X-only:    enable_x=True, all others False")
    print("  G+X:       enable_g=True, enable_x=True, all others False")
    print()

    g_only = make_only(src_params, {"g"})
    x_only = make_only(src_params, {"x"})
    gx = make_only(src_params, {"g", "x"})

    print("Forward-testing on 2026:")
    g_trades, g_final = forward_with_trades(g_only, "G only")
    x_trades, x_final = forward_with_trades(x_only, "X only")
    gx_trades, gx_final = forward_with_trades(gx, "G + X")

    # Overlap analysis: which (date, ticker) pairs appear in BOTH G-only and X-only?
    g_keys = {(t["date"], t["ticker"]) for t in g_trades}
    x_keys = {(t["date"], t["ticker"]) for t in x_trades}
    shared = g_keys & x_keys
    only_g = g_keys - x_keys
    only_x = x_keys - g_keys

    print(f"\n{'='*92}")
    print(f"  OVERLAP ANALYSIS")
    print(f"{'='*92}")
    print(f"  Unique (date, ticker) pairs:")
    print(f"    G-only trades touched : {len(g_keys):>4}")
    print(f"    X-only trades touched : {len(x_keys):>4}")
    print(f"    Shared by BOTH        : {len(shared):>4}  ({len(shared)/max(len(g_keys),1)*100:.1f}% of G's tickers)")
    print(f"    Only G traded         : {len(only_g):>4}")
    print(f"    Only X traded         : {len(only_x):>4}")

    # For shared tickers — look at entry times and PnL
    if shared:
        print(f"\n  Shared (date, ticker) pairs — entry times + PnL:")
        print(f"  {'date':<12} {'ticker':<8} {'G entry':<22} {'G pnl':>8} {'X entry':<22} {'X pnl':>8}")
        for d, tk in sorted(shared):
            gt = next((t for t in g_trades if t["date"] == d and t["ticker"] == tk), None)
            xt = next((t for t in x_trades if t["date"] == d and t["ticker"] == tk), None)
            if gt and xt:
                print(f"  {d:<12} {tk:<8} {gt['entry_time'][:19]:<22} ${gt['pnl']:>+6,.0f} {xt['entry_time'][:19]:<22} ${xt['pnl']:>+6,.0f}")

    # In the combined G+X run, see strategy mix
    by_strat = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in gx_trades:
        by_strat[t["strategy"]]["n"] += 1
        by_strat[t["strategy"]]["pnl"] += t["pnl"]
    print(f"\n  G+X COMBINED RUN — per-strategy attribution:")
    for s, v in sorted(by_strat.items()):
        avg = v["pnl"] / v["n"] if v["n"] else 0
        print(f"    {s:<6} {v['n']:>4} trades, ${v['pnl']:>+11,.0f} total, ${avg:>+8,.0f}/trade")

    # Did combined exceed sum of independent runs?
    print(f"\n  IS X COMPLEMENTARY TO G?")
    g_pnl = g_final - STARTING_CASH
    x_pnl = x_final - STARTING_CASH
    gx_pnl = gx_final - STARTING_CASH
    naive_sum = g_pnl + x_pnl
    print(f"    G alone:    ${g_pnl:>+12,.0f}")
    print(f"    X alone:    ${x_pnl:>+12,.0f}")
    print(f"    Naive sum:  ${naive_sum:>+12,.0f}")
    print(f"    G+X actual: ${gx_pnl:>+12,.0f}")
    print(f"    Synergy:    ${gx_pnl - naive_sum:>+12,.0f}  (positive = independent edges; negative = trades cannibalize)")
    print(f"    G+X vs G:   ${gx_pnl - g_pnl:>+12,.0f}  (positive = X adds value on top of G)")

    out_path = "results/x_vs_g_overlap.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "g_only": {"trades": g_trades, "final": g_final},
            "x_only": {"trades": x_trades, "final": x_final},
            "g_x": {"trades": gx_trades, "final": gx_final},
            "shared_keys": sorted(list(shared)),
        }, f, indent=2, default=str)
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()
