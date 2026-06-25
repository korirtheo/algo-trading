"""Count trades on 2026 that were:
  - skipped due to no cash (cash < $100)
  - vol-capped (vol limit reduced trade size from 100% cash)
  - fired at full cash (vol cap NOT binding)

For #511 G+L on full 2026.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
from collections import defaultdict

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = ["stored_data", "stored_data_oos", "stored_data_mar_may_2026",
             "stored_data_jun_2026", "stored_data_2026_gap_fill"]


def main():
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

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if "2026-01-01" <= d <= "2026-06-30"])
    print(f"Days: {len(dates)}")

    cash = STARTING_CASH
    n_fired = {"G": 0, "L": 0}
    n_vol_capped = {"G": 0, "L": 0}
    n_skipped_no_cash = {"G": 0, "L": 0}
    capped_sizes = []
    full_sizes = []
    monthly = defaultdict(lambda: {"fired_G":0,"fired_L":0,"capped_G":0,"capped_L":0,
                                    "skipped_G":0,"skipped_L":0,"total_signals_G":0,"total_signals_L":0})
    for d in dates:
        dp = picks.get(d, [])
        if not dp: continue
        m = d[:7]
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, sel_log = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception: continue
        # Fired trades + vol-capped
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                if s not in ("G","L"): continue
                n_fired[s] += 1
                monthly[m][f"fired_{s}"] += 1
                if st.get("vol_capped"):
                    n_vol_capped[s] += 1
                    monthly[m][f"capped_{s}"] += 1
                    capped_sizes.append(st["position_cost"])
                else:
                    full_sizes.append(st["position_cost"])
        # Skipped-no-cash from selection log
        for sel in sel_log:
            for s in sel.get("skipped", []):
                if s in ("G","L"):
                    n_skipped_no_cash[s] += 1
                    monthly[m][f"skipped_{s}"] += 1
        cash = end_c + (unset if is_cash else 0)

    print(f"\nFinal cash: ${cash:,.0f}")

    print(f"\n{'='*80}")
    print(f"  SIZING BREAKDOWN — full 2026")
    print(f"{'='*80}")
    for s in ["G","L"]:
        fired = n_fired[s]
        capped = n_vol_capped[s]
        full = fired - capped
        skipped = n_skipped_no_cash[s]
        total_signals = fired + skipped
        print(f"\n  {s} strategy:")
        print(f"    Total signals attempted:    {total_signals}")
        print(f"    Fired:                      {fired}  ({fired/total_signals*100:.1f}% of signals)" if total_signals else f"    Fired: {fired}")
        print(f"    Skipped (cash < $100):      {skipped}  ({skipped/total_signals*100:.1f}% of signals)" if total_signals else "")
        print(f"    Of fired:")
        print(f"      Full-cash deployment:     {full}  ({full/fired*100:.1f}% of fired)" if fired else "")
        print(f"      Vol-cap reduced:          {capped}  ({capped/fired*100:.1f}% of fired)" if fired else "")

    if capped_sizes:
        cs = sorted(capped_sizes)
        print(f"\n  Vol-capped trade size stats:")
        print(f"    n={len(cs)}, median=${cs[len(cs)//2]:>10,.0f}, mean=${sum(cs)/len(cs):>10,.0f}")
        print(f"    p10=${cs[len(cs)//10]:>10,.0f}, p90=${cs[len(cs)*9//10]:>10,.0f}")
    if full_sizes:
        fs = sorted(full_sizes)
        print(f"\n  Full-cash trade size stats:")
        print(f"    n={len(fs)}, median=${fs[len(fs)//2]:>10,.0f}, mean=${sum(fs)/len(fs):>10,.0f}")
        print(f"    p10=${fs[len(fs)//10]:>10,.0f}, p90=${fs[len(fs)*9//10]:>10,.0f}")

    print(f"\n{'='*80}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"{'='*80}")
    print(f"  {'month':<10} {'G_sig':>6} {'G_fire':>7} {'G_cap':>6} {'G_skip':>7} {'L_sig':>6} {'L_fire':>7} {'L_cap':>6} {'L_skip':>7}")
    for m in sorted(monthly.keys()):
        x = monthly[m]
        gs = x["fired_G"] + x["skipped_G"]; ls = x["fired_L"] + x["skipped_L"]
        print(f"  {m:<10} {gs:>6} {x['fired_G']:>7} {x['capped_G']:>6} {x['skipped_G']:>7} "
              f"{ls:>6} {x['fired_L']:>7} {x['capped_L']:>6} {x['skipped_L']:>7}")


if __name__ == "__main__":
    main()
