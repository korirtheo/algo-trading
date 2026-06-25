"""Test different cash allocation %s on full 2026 with #511 G+L.

100% = current backtest behavior (full remaining cash per trade)
50%  = use only half of remaining cash per trade
30%  = live engine behavior (LIVE_DISABLE_VOL_CAPS=True)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
import numpy as np
from collections import defaultdict

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = ["stored_data", "stored_data_oos", "stored_data_mar_may_2026",
             "stored_data_jun_2026", "stored_data_2026_gap_fill"]


def run_with_allocation(alloc_frac, label):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)
    p511 = {**bp, **base["params"]}
    for s in ALL_STRATS: p511[f"enable_{s}"] = (s in {"g","l"})
    set_strategy_params(p511)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False
    tgc.MARGIN_MULTIPLIER = alloc_frac  # <-- the key change

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if "2026-01-01" <= d <= "2026-06-30"])

    cash = STARTING_CASH
    daily_eq = [cash]
    n_fired = {"G":0, "L":0}; n_skipped = {"G":0, "L":0}; n_capped = {"G":0, "L":0}
    pnl_by_strat = {"G": 0.0, "L": 0.0}
    for d in dates:
        dp = picks.get(d, [])
        if not dp: daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, sel_log = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                if s in ("G","L"):
                    n_fired[s] += 1
                    pnl_by_strat[s] += float(st.get("pnl") or 0)
                    if st.get("vol_capped"): n_capped[s] += 1
        for sel in sel_log:
            for s in sel.get("skipped", []):
                if s in ("G","L"): n_skipped[s] += 1
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)

    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0
    return {"label": label, "alloc": alloc_frac, "final_cash": cash,
            "pnl": cash - STARTING_CASH, "multiplier": cash/STARTING_CASH, "max_dd_pct": dd,
            "n_fired_G": n_fired["G"], "n_fired_L": n_fired["L"],
            "n_skipped_G": n_skipped["G"], "n_skipped_L": n_skipped["L"],
            "n_capped_G": n_capped["G"], "n_capped_L": n_capped["L"],
            "g_pnl": pnl_by_strat["G"], "l_pnl": pnl_by_strat["L"]}


def main():
    print("Running 2026 backtest at 3 allocation levels...")
    results = []
    for alloc, label in [(1.0, "100% (current)"), (0.5, "50%"), (0.3, "30% (live)"), (2.0, "200% (2x margin)"), (4.0, "400% (4x margin)")]:
        print(f"\n  Running {label}...")
        r = run_with_allocation(alloc, label)
        results.append(r)

    print(f"\n{'='*100}")
    print(f"  ALLOCATION % COMPARISON on full 2026 (G+L only, no v3)")
    print(f"{'='*100}")
    print(f"  {'alloc':<16} {'PnL':>14} {'Mult':>9} {'DD%':>8} {'G fire':>7} {'G skip':>7} {'G cap':>7} {'L fire':>7}")
    print(f"  {'-'*16} {'-'*14} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        sig_g = r['n_fired_G'] + r['n_skipped_G']
        fire_pct = r['n_fired_G']/sig_g*100 if sig_g else 0
        print(f"  {r['label']:<16} ${r['pnl']:>+12,.0f}  {r['multiplier']:>7.2f}x {r['max_dd_pct']:>+7.1f}% "
              f"{r['n_fired_G']:>7} {r['n_skipped_G']:>7} {r['n_capped_G']:>7} {r['n_fired_L']:>7}")

    print(f"\n  Per-strategy PnL by allocation:")
    print(f"  {'alloc':<16} {'G PnL':>14} {'L PnL':>14}")
    for r in results:
        print(f"  {r['label']:<16} ${r['g_pnl']:>+12,.0f}  ${r['l_pnl']:>+12,.0f}")

    print(f"\n  vs 100% baseline:")
    base = results[0]['pnl']
    for r in results[1:]:
        delta = (r['pnl'] - base) / base * 100
        print(f"  {r['label']:<16} Δ PnL ${r['pnl']-base:>+13,.0f} ({delta:+.1f}%)  "
              f"G fired Δ {r['n_fired_G']-results[0]['n_fired_G']:+d}  G skipped Δ {r['n_skipped_G']-results[0]['n_skipped_G']:+d}")

    out = "results/cash_allocation_2026_comparison.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
