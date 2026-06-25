"""Compare on 2026:
  A) #511 G + #511 L (current deploy)
  B) #511 G + #626 L (L swap surgical)
  C) #511 G + #626 L + v3 (L swap + R-O)
  D) #159 G + #626 L (best solos, no v3)
  E) #159 G + #626 L + v3 (best solos + R-O — the 'all best' stack)

Joint compounded, 100% AND 30% allocations.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
from collections import defaultdict
import numpy as np
import psycopg2

STARTING_CASH = 25_000
POSITION_PCT_V3 = 0.30
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = ["stored_data", "stored_data_oos", "stored_data_mar_may_2026",
             "stored_data_jun_2026", "stored_data_2026_gap_fill"]

V3_PARAMS = {"g_target_pct": 57.0, "g_stop_pct": 30.0, "g_time_limit_min": 27,
             "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}


def fetch_trial(db, study, num):
    c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname=db)
    cur = c.cursor()
    cur.execute('SELECT trial_id FROM trials WHERE number=%s AND study_id=(SELECT study_id FROM studies WHERE study_name=%s)', (num, study))
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


def setup_tgc(params, alloc):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
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
    return tgc


def _v3_candidate(mh, day_open, g_holds_list):
    if mh is None or len(mh) < 2: return None
    bar0_red = float(mh.iloc[0]["Close"]) <= day_open
    scan = 1
    if not bar0_red and g_holds_list:
        ge = max(x for _, x in g_holds_list)
        ns = None
        for i in range(1, len(mh)):
            if mh.index[i] > ge: ns = i; break
        if ns is None: return None
        scan = ns
    for i in range(scan, len(mh)):
        if float(mh.iloc[i]["Close"]) > day_open:
            ba = mh.iloc[i+1:]
            if len(ba) == 0: return None
            return (mh.index[i], float(mh.iloc[i]["Close"]), ba)
    return None


def _v3_trade(tgc, mh, ets, fp, ba, pos_dollars):
    pre = mh.loc[mh.index <= ets]
    vs = float(pre["Volume"].sum()) if len(pre) > 0 else 0
    dv = fp * vs
    if dv <= 0: return 0.0
    lim = dv * (tgc.VOL_CAP_PCT/100)
    ve, _, _, vr = tgc._multi_window_effective_volume(mh, ets, fp)
    if tgc.MAX_REGIME_PARTICIPATION > 0 and vr > 0: lim = min(lim, vr * tgc.MAX_REGIME_PARTICIPATION)
    if tgc.MAX_2MIN_PARTICIPATION > 0 and ve > 0: lim = min(lim, ve * tgc.MAX_2MIN_PARTICIPATION)
    cs = min(pos_dollars, lim)
    if cs < 50: return 0.0
    si = tgc._entry_slip_pct(fp, cs, ve)
    ae = fp * (1 + si/100)
    sh = cs / ae
    tp = V3_PARAMS["g_target_pct"]; sp = V3_PARAMS["g_stop_pct"]
    tm = int(V3_PARAMS["g_time_limit_min"]); trp = V3_PARAMS["g_trail_pct"]; tap = V3_PARAMS["g_trail_activate_pct"]
    tgt = ae*(1+tp/100); stp = ae*(1-sp/100); peak = ae; ts_ = None
    ep = None; et2 = None
    mb = max(1, tm//2)
    for i, (ts, row) in enumerate(ba.iterrows()):
        if i >= mb: ep = float(row["Close"]); et2 = ts; break
        h = float(row["High"]); lo_ = float(row["Low"])
        if h >= tgt: ep = tgt; et2 = ts; break
        if lo_ <= stp: ep = stp; et2 = ts; break
        if h > peak: peak = h
        if (peak/ae-1)*100 >= tap:
            nt = peak*(1-trp/100)
            if ts_ is None or nt > ts_: ts_ = nt
        if ts_ is not None and lo_ <= ts_: ep = ts_; et2 = ts; break
    if ep is None and len(ba) > 0: ep = float(ba.iloc[-1]["Close"]); et2 = ba.index[-1]
    if ep is None or et2 is None: return 0.0
    so = tgc._exit_slip_pct(ep, sh, {"mh": mh}, et2)
    return sh * (ep*(1-so/100) - ae)


def run(params, with_v3, label, alloc=1.0):
    from test_full import load_all_picks, MARGIN_THRESHOLD
    tgc = setup_tgc(params, alloc)
    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if "2026-01-01" <= d <= "2026-06-30"])
    cash = STARTING_CASH
    n_g = n_l = n_v3 = 0; g_pnl = l_pnl = v3_pnl = 0.0
    daily_eq = [cash]
    for d in dates:
        dp = picks.get(d, [])
        if not dp: daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try: states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except: daily_eq.append(cash); continue
        g_holds = defaultdict(list)
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy"); p = float(st.get("pnl") or 0)
                if s == "G":
                    n_g += 1; g_pnl += p
                    et, xt = st.get("entry_time"), st.get("exit_time")
                    if et and xt: g_holds[st["ticker"]].append((et, xt))
                elif s == "L":
                    n_l += 1; l_pnl += p
        gl_end_cash = end_c + (unset if is_cash else 0)
        v3_added = 0.0
        if with_v3:
            for p in dp:
                mh = p.get("market_hour_candles")
                if mh is None or len(mh) < 2: continue
                day_open = float(mh.iloc[0]["Open"])
                ghol = g_holds.get(p["ticker"], [])
                cand = _v3_candidate(mh, day_open, ghol)
                if cand is None: continue
                ets, fp, ba = cand
                pos = cash * POSITION_PCT_V3
                pnl = _v3_trade(tgc, mh, ets, fp, ba, pos)
                if pnl != 0.0:
                    n_v3 += 1; v3_pnl += pnl; v3_added += pnl
        cash = gl_end_cash + v3_added
        daily_eq.append(cash)
    eq = np.array(daily_eq); peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0
    return {"label": label, "pnl": cash - STARTING_CASH, "multiplier": cash/STARTING_CASH,
            "max_dd_pct": dd, "n_g": n_g, "g_pnl": g_pnl, "n_l": n_l, "l_pnl": l_pnl,
            "n_v3": n_v3, "v3_pnl": v3_pnl}


def main():
    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)

    g159 = fetch_trial("optuna_g_only", "g_only_w21b", 159)
    l626 = fetch_trial("optuna_l_only", "l_only_w21b", 626)

    # Config builders
    def build(g_params_dict, l_params_dict, label):
        p = {**bp, **base["params"]}  # start from baseline + #511
        if g_params_dict is not None:
            for k, v in g_params_dict.items():
                if k.startswith("g_"): p[k] = v
        if l_params_dict is not None:
            for k, v in l_params_dict.items():
                if k.startswith("l_"): p[k] = v
        for s in ALL_STRATS: p[f"enable_{s}"] = (s in {"g","l"})
        return p

    A = build(None, None, "A: #511 G + #511 L (current)")
    B = build(None, l626, "B: #511 G + #626 L")
    D = build(g159, l626, "D: #159 G + #626 L")

    print("Running 5 configs at 100% allocation...")
    rA100 = run(A, False, "A: #511 G + #511 L", 1.0)
    rB100 = run(B, False, "B: #511 G + #626 L", 1.0)
    rC100 = run(B, True,  "C: #511 G + #626 L + v3", 1.0)
    rD100 = run(D, False, "D: #159 G + #626 L", 1.0)
    rE100 = run(D, True,  "E: #159 G + #626 L + v3", 1.0)

    print("\nRunning 5 configs at 30% (live) allocation...")
    rA30 = run(A, False, "A: #511 G + #511 L", 0.3)
    rB30 = run(B, False, "B: #511 G + #626 L", 0.3)
    rC30 = run(B, True,  "C: #511 G + #626 L + v3", 0.3)
    rD30 = run(D, False, "D: #159 G + #626 L", 0.3)
    rE30 = run(D, True,  "E: #159 G + #626 L + v3", 0.3)

    for alloc_label, rs in [("100%", [rA100, rB100, rC100, rD100, rE100]),
                              ("30% (live)", [rA30, rB30, rC30, rD30, rE30])]:
        print(f"\n{'='*112}")
        print(f"  Full 2026 — joint compounded at {alloc_label} allocation")
        print(f"{'='*112}")
        print(f"  {'config':<32} {'Total PnL':>14} {'Mult':>9} {'DD%':>8} "
              f"{'G n':>4} {'G PnL':>13} {'L n':>4} {'L PnL':>13} {'v3 n':>5} {'v3 PnL':>12}")
        print(f"  {'-'*32} {'-'*14} {'-'*9} {'-'*8} {'-'*4} {'-'*13} {'-'*4} {'-'*13} {'-'*5} {'-'*12}")
        for r in rs:
            print(f"  {r['label']:<32} ${r['pnl']:>+12,.0f}  {r['multiplier']:>7.2f}x {r['max_dd_pct']:>+7.1f}% "
                  f"{r['n_g']:>4} ${r['g_pnl']:>+11,.0f} {r['n_l']:>4} ${r['l_pnl']:>+11,.0f} "
                  f"{r['n_v3']:>5} ${r['v3_pnl']:>+10,.0f}")


if __name__ == "__main__":
    main()
