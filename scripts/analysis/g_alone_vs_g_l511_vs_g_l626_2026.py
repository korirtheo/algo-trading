"""Compare on full 2026 with joint compounding:
  A) #511 G alone (no L)
  B) #511 G + #511 L (current deploy)
  C) #511 G + #626 L (proposed L swap)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
import numpy as np
import psycopg2

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = ["stored_data", "stored_data_oos", "stored_data_mar_may_2026",
             "stored_data_jun_2026", "stored_data_2026_gap_fill"]


def fetch_l626():
    c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_l_only')
    cur = c.cursor()
    cur.execute('SELECT trial_id FROM trials WHERE number=626 AND study_id=(SELECT study_id FROM studies WHERE study_name=%s)', ('l_only_w21b',))
    tid = cur.fetchone()[0]
    cur.execute('SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s', (tid,))
    params = {}
    for name, val, dist in cur.fetchall():
        try:
            d = json.loads(dist) if dist else {}
            kind = d.get("name", "")
            if kind == "CategoricalDistribution": params[name] = d["attributes"]["choices"][int(val)]
            elif "Int" in kind: params[name] = int(val)
            else: params[name] = float(val)
        except: params[name] = float(val)
    c.close()
    return params


def run(params, label, allocation=1.0):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD
    set_strategy_params(params)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False
    tgc.MARGIN_MULTIPLIER = allocation

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if "2026-01-01" <= d <= "2026-06-30"])
    cash = STARTING_CASH
    n_g = n_l = 0; g_pnl = l_pnl = 0.0
    daily_eq = [cash]
    for d in dates:
        dp = picks.get(d, [])
        if not dp: daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try: states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except: daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy"); p = float(st.get("pnl") or 0)
                if s == "G": n_g += 1; g_pnl += p
                elif s == "L": n_l += 1; l_pnl += p
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0
    return {"label": label, "pnl": cash - STARTING_CASH, "multiplier": cash/STARTING_CASH,
            "max_dd_pct": dd, "n_g": n_g, "g_pnl": g_pnl, "n_l": n_l, "l_pnl": l_pnl}


def main():
    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)

    p_g_alone = {**bp, **base["params"]}
    for s in ALL_STRATS: p_g_alone[f"enable_{s}"] = (s == "g")

    p_gl511 = {**bp, **base["params"]}
    for s in ALL_STRATS: p_gl511[f"enable_{s}"] = (s in {"g","l"})

    l626 = fetch_l626()
    p_gl626 = {**bp, **base["params"]}
    for k, v in l626.items():
        if k.startswith("l_"): p_gl626[k] = v
    for s in ALL_STRATS: p_gl626[f"enable_{s}"] = (s in {"g","l"})

    print("Running at 100% allocation (backtest standard)...")
    a100 = [run(p_g_alone, "G alone (no L)", 1.0),
            run(p_gl511, "G + L #511 (current)", 1.0),
            run(p_gl626, "G + L #626 (proposed)", 1.0)]

    print("\nRunning at 30% allocation (live behavior)...")
    a30 = [run(p_g_alone, "G alone (no L)", 0.3),
           run(p_gl511, "G + L #511 (current)", 0.3),
           run(p_gl626, "G + L #626 (proposed)", 0.3)]

    for alloc_label, results in [("100%", a100), ("30% (live)", a30)]:
        print(f"\n{'='*102}")
        print(f"  Full 2026 — joint compounded at {alloc_label} allocation")
        print(f"{'='*102}")
        print(f"  {'config':<22} {'Total PnL':>14} {'Mult':>9} {'DD%':>8} {'G n':>5} {'G PnL':>13} {'L n':>5} {'L PnL':>13}")
        print(f"  {'-'*22} {'-'*14} {'-'*9} {'-'*8} {'-'*5} {'-'*13} {'-'*5} {'-'*13}")
        for r in results:
            print(f"  {r['label']:<22} ${r['pnl']:>+12,.0f}  {r['multiplier']:>7.2f}x {r['max_dd_pct']:>+7.1f}% "
                  f"{r['n_g']:>5} ${r['g_pnl']:>+11,.0f} {r['n_l']:>5} ${r['l_pnl']:>+11,.0f}")


if __name__ == "__main__":
    main()
