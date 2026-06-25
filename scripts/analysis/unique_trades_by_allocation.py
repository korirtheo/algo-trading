"""Count UNIQUE (ticker, date) G+L trades fired at 100% vs 50% vs 30% allocation.

If 50%/30% fires MORE unique trades than 100%, then cash exhaustion is blocking unique signals.
If all same 418, then the 260 'skipped' at 100% are bar-event duplicates of the same signal.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os

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
    tgc.MARGIN_MULTIPLIER = alloc_frac

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if "2026-01-01" <= d <= "2026-06-30"])

    cash = STARTING_CASH
    unique_g = set(); unique_l = set()
    skip_g_events = 0; skip_l_events = 0
    skipped_unique = set()  # (ticker, date, ts, strat)
    for d in dates:
        dp = picks.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, sel_log = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception: continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                if s == "G": unique_g.add((st.get("ticker"), d))
                elif s == "L": unique_l.add((st.get("ticker"), d))
        for sel in sel_log:
            for s in sel.get("skipped", []):
                if s == "G": skip_g_events += 1
                elif s == "L": skip_l_events += 1
        cash = end_c + (unset if is_cash else 0)

    return {"label": label, "alloc": alloc_frac, "final_cash": cash,
            "unique_g_trades": len(unique_g), "unique_l_trades": len(unique_l),
            "skip_g_events": skip_g_events, "skip_l_events": skip_l_events,
            "g_set": unique_g, "l_set": unique_l}


def main():
    results = []
    for alloc, label in [(1.0, "100%"), (0.5, "50%"), (0.3, "30% (live)")]:
        print(f"\nRunning {label}...")
        r = run_with_allocation(alloc, label)
        results.append(r)

    print(f"\n{'='*92}")
    print(f"  UNIQUE (ticker, date) TRADES FIRED by allocation")
    print(f"{'='*92}")
    print(f"  {'alloc':<15} {'unique G':>10} {'unique L':>10} {'G skip events':>15} {'L skip events':>15}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*15} {'-'*15}")
    for r in results:
        print(f"  {r['label']:<15} {r['unique_g_trades']:>10} {r['unique_l_trades']:>10} {r['skip_g_events']:>15} {r['skip_l_events']:>15}")

    print(f"\n{'='*92}")
    print(f"  SET DIFFERENCE: trades fired at 50% but NOT at 100%")
    print(f"{'='*92}")
    g_100 = results[0]["g_set"]; g_50 = results[1]["g_set"]; g_30 = results[2]["g_set"]
    new_at_50 = g_50 - g_100
    new_at_30 = g_30 - g_100
    lost_at_50 = g_100 - g_50
    print(f"  G unique trades at 100%: {len(g_100)}")
    print(f"  G unique trades at 50%:  {len(g_50)}")
    print(f"  Trades fired at 50% but NOT 100% (unlocked by less aggressive sizing): {len(new_at_50)}")
    print(f"  Trades fired at 100% but NOT 50% (signals lost because trade too small?): {len(lost_at_50)}")
    if new_at_50:
        print(f"\n  Sample unlocked trades: {list(new_at_50)[:10]}")
    if lost_at_50:
        print(f"\n  Sample lost trades: {list(lost_at_50)[:10]}")


if __name__ == "__main__":
    main()
