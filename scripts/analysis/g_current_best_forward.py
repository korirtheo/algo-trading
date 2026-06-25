"""Forward-test G's CURRENT best (mid-Optuna) on 2026 Mar-Jun OOS."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
import numpy as np
import psycopg2

STARTING_CASH = 25_000
BASELINE_PATH = "config/trial_432_params.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = ["stored_data_mar_may_2026", "stored_data_jun_2026"]
DATE_LO = "2026-03-01"
DATE_HI = "2026-06-30"


def fetch_best():
    c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_g_only')
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value FROM trials t
        JOIN trial_values tv ON tv.trial_id=t.trial_id
        WHERE t.state='COMPLETE' AND t.study_id=(SELECT study_id FROM studies WHERE study_name='g_only_w21b')
        ORDER BY tv.value DESC LIMIT 1
    """)
    r = cur.fetchone()
    if not r: c.close(); return None
    tid, num, score = r
    cur.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s", (tid,))
    params = {}
    for name, val, dist in cur.fetchall():
        try:
            d = json.loads(dist) if dist else {}
            kind = d.get("name", "")
            if kind == "CategoricalDistribution":
                params[name] = d["attributes"]["choices"][int(val)]
            elif "Int" in kind:
                params[name] = int(val)
            else:
                params[name] = float(val)
        except Exception:
            params[name] = float(val)
    cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
    ua = dict(cur.fetchall())
    c.close()
    return {"number": num, "score": float(score), "params": params, "user_attrs": ua}


def main():
    best = fetch_best()
    if best is None: print("No completed G trials yet."); return
    print(f"G CURRENT BEST trial #{best['number']}: train score = ${best['score']:,.0f}")
    print(f"  Train user_attrs: {best['user_attrs']}")

    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE_PATH) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(best["params"])
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s == "g")
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
    print(f"Forward window: {DATE_LO} .. {DATE_HI}  ({len(dates)} days)")

    cash = STARTING_CASH
    trades = []
    daily_eq = [cash]
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
                trades.append({"date": d, "ticker": st.get("ticker"),
                               "pnl": float(st.get("pnl") or 0),
                               "exit_reason": st.get("exit_reason")})
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)

    if not trades:
        print("\nG fired 0 trades."); return

    n = len(trades); tot = sum(t["pnl"] for t in trades)
    w = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    l = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    wr = sum(1 for t in trades if t["pnl"] > 0)/n*100
    pf = w/l if l > 0 else 99.0
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = float(((eq - peak) / peak).min() * 100)

    print(f"\n{'='*80}")
    print(f"  G FORWARD on 2026 Mar-Jun OOS (with current Optuna best #{best['number']})")
    print(f"{'='*80}")
    print(f"  Trades:        {n}")
    print(f"  Total PnL:    ${tot:>+12,.0f}")
    print(f"  Win Rate:      {wr:.1f}%")
    print(f"  PF:            {pf:.2f}")
    print(f"  Final cash:   ${cash:>+12,.0f}  ({cash/STARTING_CASH:.2f}x)")
    print(f"  Max DD:        {dd_pct:.1f}%")

    by_reason = {}
    for t in trades: by_reason.setdefault(t["exit_reason"], []).append(t)
    print(f"\n  Exit reason breakdown:")
    print(f"  {'reason':<15} {'n':>5} {'pnl':>14} {'mean':>10} {'WR%':>6}")
    for r, ts in sorted(by_reason.items()):
        rn = len(ts); rt = sum(t["pnl"] for t in ts); rm = rt/rn
        rwr = sum(1 for t in ts if t["pnl"] > 0)/rn*100
        print(f"  {str(r):<15} {rn:>5} ${rt:>+12,.0f} ${rm:>+8,.0f} {rwr:>5.1f}%")

    print(f"\n  Tunable G params from trial #{best['number']}:")
    for k in sorted(best['params'].keys()):
        if k.startswith("g_"):
            print(f"    {k} = {best['params'][k]}")

    out = "results/g_current_best_2026_mar_jun.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"trial_number": best["number"], "train_score": best["score"],
                   "params": best["params"],
                   "forward_pnl": tot, "n_trades": n, "wr": wr, "pf": pf,
                   "max_dd_pct": dd_pct, "trades": trades}, f, indent=2, default=str)
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
