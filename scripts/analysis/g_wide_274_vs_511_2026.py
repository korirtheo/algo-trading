"""Test g_wide current best #274 vs #511 G on full 2026 and wide-forward windows."""
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
    "2026_full":    {"dirs": ["stored_data", "stored_data_oos", "stored_data_mar_may_2026",
                              "stored_data_jun_2026", "stored_data_2026_gap_fill"],
                    "lo": "2026-01-01", "hi": "2026-06-30"},
}


def fetch_g_wide(num):
    c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_g_wide')
    cur = c.cursor()
    cur.execute('SELECT trial_id FROM trials WHERE number=%s AND study_id=(SELECT study_id FROM studies WHERE study_name=%s)', (num, 'g_wide_w21b'))
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


def run(params, dirs_arg, date_lo, date_hi, alloc=1.0):
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
    tgc.MARGIN_MULTIPLIER = alloc
    dirs = [d for d in dirs_arg if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])
    cash = STARTING_CASH; n = 0; pnl = 0.0; wins = 0; losses_v = 0.0; wins_v = 0.0
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
    return {"pnl": cash - STARTING_CASH, "multiplier": cash/STARTING_CASH,
            "max_dd_pct": dd, "n": n, "wr": wr, "pf": pf}


def main():
    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)

    p511 = {**bp, **base["params"]}
    for s in ALL_STRATS: p511[f"enable_{s}"] = (s == "g")

    # 3-way: #511 (deploy), #527 (g_wide train winner), #521 (g_wide forward winner)
    g527 = fetch_g_wide(527)
    g521 = fetch_g_wide(521)
    g274 = g527  # legacy var name
    print(f"g_wide #527 (TRAIN winner) params:")
    for k, v in sorted(g527.items()): print(f"  {k} = {v}")
    print(f"\ng_wide #521 (FORWARD winner) params:")
    for k, v in sorted(g521.items()): print(f"  {k} = {v}")

    p274 = {**bp, **base["params"]}
    # Fix entries to #511's (g_wide locked entries)
    p274["g_min_gap_pct"] = 15.0
    p274["g_require_2nd_green"] = True
    p274["g_require_2nd_new_high"] = False
    # Override exits from trial
    for k, v in g274.items(): p274[k] = v
    for s in ALL_STRATS: p274[f"enable_{s}"] = (s == "g")

    print(f"\n{'='*102}")
    print(f"  g_wide #274 vs #511 — G ONLY, 100% allocation")
    print(f"{'='*102}")
    print(f"  {'window':<14} {'config':<14} {'PnL':>13} {'Mult':>8} {'DD%':>7} {'n':>5} {'WR%':>6} {'PF':>6}")
    print(f"  {'-'*14} {'-'*14} {'-'*13} {'-'*8} {'-'*7} {'-'*5} {'-'*6} {'-'*6}")
    totals = {"#511": 0, "#274": 0}
    for wname, w in WINDOWS.items():
        r511 = run(p511, w["dirs"], w["lo"], w["hi"], 1.0)
        r274 = run(p274, w["dirs"], w["lo"], w["hi"], 1.0)
        totals["#511"] += r511["pnl"]; totals["#274"] += r274["pnl"]
        print(f"  {wname:<14} {'#511 G':<14} ${r511['pnl']:>+11,.0f}  {r511['multiplier']:>6.2f}x {r511['max_dd_pct']:>+6.1f}% "
              f"{r511['n']:>5} {r511['wr']:>5.1f}% {r511['pf']:>5.2f}")
        print(f"  {'':<14} {'g_wide #274':<14} ${r274['pnl']:>+11,.0f}  {r274['multiplier']:>6.2f}x {r274['max_dd_pct']:>+6.1f}% "
              f"{r274['n']:>5} {r274['wr']:>5.1f}% {r274['pf']:>5.2f}")
        print()
    print(f"  {'TOTAL':<14} {'#511 G':<14} ${totals['#511']:>+11,.0f}")
    print(f"  {'':<14} {'g_wide #274':<14} ${totals['#274']:>+11,.0f}  (Δ vs #511: ${totals['#274']-totals['#511']:+,.0f}, "
          f"{(totals['#274']-totals['#511'])/totals['#511']*100 if totals['#511'] else 0:+.1f}%)")


if __name__ == "__main__":
    main()
