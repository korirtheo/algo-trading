"""Apples-to-apples wide-forward: #511 G vs #159 G.

Each config runs G-only (no L, no v3) on each window with fresh $25K.
Same methodology as the original wide-forward used.
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

WINDOWS = {
    "2022":         {"dirs": ["stored_data_2022"], "lo": "2022-01-01", "hi": "2022-12-31"},
    "2026_MarJun":  {"dirs": ["stored_data_mar_may_2026", "stored_data_jun_2026"],
                    "lo": "2026-03-01", "hi": "2026-06-30"},
}


def fetch_g159():
    c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_g_only')
    cur = c.cursor()
    cur.execute('SELECT trial_id FROM trials WHERE number=159 AND study_id=(SELECT study_id FROM studies WHERE study_name=%s)', ('g_only_w21b',))
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


def run(params, dirs_arg, date_lo, date_hi):
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
    tgc.MARGIN_MULTIPLIER = 1.0

    dirs = [d for d in dirs_arg if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])
    cash = STARTING_CASH
    n = 0; pnl = 0.0; wins = 0; losses_v = 0.0; wins_v = 0.0
    daily_eq = [cash]
    for d in dates:
        dp = picks.get(d, [])
        if not dp: daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try: states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except: daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
                n += 1
                p = float(st.get("pnl") or 0); pnl += p
                if p > 0: wins += 1; wins_v += p
                else: losses_v += -p
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq); peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0
    wr = wins/n*100 if n else 0
    pf = wins_v/losses_v if losses_v > 0 else 99
    return {"final_cash": cash, "pnl": cash - STARTING_CASH, "multiplier": cash/STARTING_CASH,
            "max_dd_pct": dd, "n_trades": n, "wr": wr, "pf": pf}


def main():
    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)

    # #511 G alone (G enabled, no L)
    p511g = {**bp, **base["params"]}
    for s in ALL_STRATS: p511g[f"enable_{s}"] = (s == "g")

    # #159 G alone (G enabled, params from g_only trial #159, no L)
    g159 = fetch_g159()
    p159 = dict(bp)
    for k, v in g159.items(): p159[k] = v  # override with #159 params
    for s in ALL_STRATS: p159[f"enable_{s}"] = (s == "g")

    print(f"{'='*100}")
    print(f"  APPLES-TO-APPLES: #511 G alone vs #159 G alone on wide-forward windows")
    print(f"  (G only, no L, no v3, fresh $25K each window, 100% allocation)")
    print(f"{'='*100}")
    print(f"\n  {'window':<14} {'config':<14} {'PnL':>13} {'Mult':>8} {'DD%':>7} {'n':>5} {'WR%':>6} {'PF':>6}")
    print(f"  {'-'*14} {'-'*14} {'-'*13} {'-'*8} {'-'*7} {'-'*5} {'-'*6} {'-'*6}")

    results = {}
    for wname, w in WINDOWS.items():
        r511 = run(p511g, w["dirs"], w["lo"], w["hi"])
        r159 = run(p159, w["dirs"], w["lo"], w["hi"])
        results[wname] = {"r511": r511, "r159": r159}
        print(f"  {wname:<14} {'#511 G':<14} ${r511['pnl']:>+11,.0f}  {r511['multiplier']:>6.2f}x {r511['max_dd_pct']:>+6.1f}% "
              f"{r511['n_trades']:>5} {r511['wr']:>5.1f}% {r511['pf']:>5.2f}")
        print(f"  {'':<14} {'#159 G':<14} ${r159['pnl']:>+11,.0f}  {r159['multiplier']:>6.2f}x {r159['max_dd_pct']:>+6.1f}% "
              f"{r159['n_trades']:>5} {r159['wr']:>5.1f}% {r159['pf']:>5.2f}")
        print()

    sum_511 = sum(r["r511"]["pnl"] for r in results.values())
    sum_159 = sum(r["r159"]["pnl"] for r in results.values())
    print(f"  {'TOTAL':<14} {'#511 G':<14} ${sum_511:>+11,.0f}")
    print(f"  {'':<14} {'#159 G':<14} ${sum_159:>+11,.0f}  (Δ vs #511: ${sum_159-sum_511:+,.0f}, "
          f"{(sum_159-sum_511)/sum_511*100 if sum_511 else 0:+.1f}%)")


if __name__ == "__main__":
    main()
