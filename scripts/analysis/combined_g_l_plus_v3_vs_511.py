"""G #159 + L #626 (combined config) + v3 R-O #576 (separate $25K account) vs #511.

Assumes v3 runs in a separate Alpaca account from G+L until the engine
multi-position change is implemented. Each compounds independently.

Reports: 2022 OOS + 2026 Mar-Jun OOS.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
from collections import defaultdict
import numpy as np
import psycopg2

STARTING_CASH = 25_000
POSITION_PCT = 0.30
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

V3_PARAMS = {"g_target_pct": 57.0, "g_stop_pct": 30.0, "g_time_limit_min": 27,
             "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}

WINDOWS = {
    "2022":         {"dirs": ["stored_data_2022"],
                    "lo": "2022-01-01", "hi": "2022-12-31"},
    "2026_MarJun":  {"dirs": ["stored_data_mar_may_2026", "stored_data_jun_2026"],
                    "lo": "2026-03-01", "hi": "2026-06-30"},
}


def fetch_trial_params(db, study, trial_num):
    c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname=db)
    cur = c.cursor()
    cur.execute('SELECT trial_id FROM trials WHERE number=%s AND study_id=(SELECT study_id FROM studies WHERE study_name=%s)', (trial_num, study))
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


def setup_tgc(params):
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
    return tgc


def run_g_l_compounded(params, dirs_arg, date_lo, date_hi):
    from test_full import load_all_picks, MARGIN_THRESHOLD
    tgc = setup_tgc(params)
    dirs = [d for d in dirs_arg if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])
    cash = STARTING_CASH
    n_g = n_l = 0; g_pnl = l_pnl = 0.0
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
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                p = float(st.get("pnl") or 0)
                if s == "G": n_g += 1; g_pnl += p
                elif s == "L": n_l += 1; l_pnl += p
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0
    return {"final_cash": cash, "pnl": cash - STARTING_CASH,
            "multiplier": cash/STARTING_CASH, "max_dd_pct": dd,
            "n_g": n_g, "g_pnl": g_pnl, "n_l": n_l, "l_pnl": l_pnl}


def precompute_v3_candidates(tgc, picks_by_date, dates, g_holds_by_key):
    cands = []
    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2: continue
            day_open = float(mh.iloc[0]["Open"])
            bar0_close = float(mh.iloc[0]["Close"])
            bar0_red = bar0_close <= day_open
            ghol = g_holds_by_key.get((p["ticker"], d), [])
            scan = 1
            if not bar0_red and ghol:
                ge = max(x for _, x in ghol)
                ns = None
                for i in range(1, len(mh)):
                    if mh.index[i] > ge: ns = i; break
                if ns is None: continue
                scan = ns
            ei = ep = ets = None
            for i in range(scan, len(mh)):
                if float(mh.iloc[i]["Close"]) > day_open:
                    ei = i; ep = float(mh.iloc[i]["Close"]); ets = mh.index[i]; break
            if ei is None or ep <= 0: continue
            ba = mh.iloc[ei+1:]
            if len(ba) == 0: continue
            cands.append((d, ets, ep, mh, ba))
    return cands


def run_v3_compounded(dirs_arg, date_lo, date_hi):
    """v3 standalone with #511 baseline for G-window pre-compute. Compounds the $25K start."""
    from test_full import load_all_picks, MARGIN_THRESHOLD
    # Need #511 (G+L) baseline to determine G holds
    with open(W21B_DEPLOY) as f: base = json.load(f)
    with open(BASELINE) as f: bp = json.load(f)
    p511 = {**bp, **base["params"]}
    for s in ALL_STRATS:
        p511[f"enable_{s}"] = (s in {"g", "l"})
    tgc = setup_tgc(p511)
    dirs = [d for d in dirs_arg if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])
    # G holds from baseline #511 run (separate cash; doesn't matter for window detection)
    cash = STARTING_CASH
    g_holds = defaultdict(list)
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try: states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except: continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost",0)>0 and st.get("strategy")=="G":
                et, xt = st.get("entry_time"), st.get("exit_time")
                if et and xt: g_holds[(st["ticker"], d)].append((et, xt))
        cash = end_c + (unset if is_cash else 0)
    # Now pre-compute v3 candidates
    cands_by_date = defaultdict(list)
    cands = precompute_v3_candidates(tgc, picks_by_date, dates, g_holds)
    for c in cands:
        cands_by_date[c[0]].append(c)
    # Walk through dates, compounding cash. For each candidate that day, simulate v3 trade with current cash.
    cash = STARTING_CASH
    n_trades = 0; total_pnl_v3 = 0.0
    daily_eq = [cash]
    VC = tgc.VOL_CAP_PCT; MR = tgc.MAX_REGIME_PARTICIPATION; M2 = tgc.MAX_2MIN_PARTICIPATION
    def caps(mh, ts, fp, req):
        pre = mh.loc[mh.index <= ts]
        vs = float(pre["Volume"].sum()) if len(pre) > 0 else 0
        dv = fp * vs
        if dv <= 0: return 0, 0
        lim = dv * (VC/100)
        ve, _, _, vr = tgc._multi_window_effective_volume(mh, ts, fp)
        if MR > 0 and vr > 0: lim = min(lim, vr * MR)
        if M2 > 0 and ve > 0: lim = min(lim, ve * M2)
        return min(req, lim), ve
    tp = V3_PARAMS["g_target_pct"]; sp = V3_PARAMS["g_stop_pct"]
    tm = int(V3_PARAMS["g_time_limit_min"]); trp = V3_PARAMS["g_trail_pct"]
    tap = V3_PARAMS["g_trail_activate_pct"]
    for d in dates:
        day_cands = cands_by_date.get(d, [])
        if not day_cands: daily_eq.append(cash); continue
        pos_dollars = cash * POSITION_PCT  # CURRENT cash, compounded
        for (dd, ets, fp, mh, ba) in day_cands:
            cs, ve = caps(mh, ets, fp, pos_dollars)
            if cs < 50: continue
            si = tgc._entry_slip_pct(fp, cs, ve)
            ae = fp * (1 + si/100)
            sh = cs / ae
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
            if ep is None or et2 is None: continue
            so = tgc._exit_slip_pct(ep, sh, {"mh": mh}, et2)
            pnl = sh * (ep*(1-so/100) - ae)
            cash += pnl
            n_trades += 1
            total_pnl_v3 += pnl
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0
    return {"final_cash": cash, "pnl": cash - STARTING_CASH,
            "multiplier": cash/STARTING_CASH, "max_dd_pct": dd,
            "n_trades": n_trades, "v3_pnl": total_pnl_v3}


def main():
    # Build the G+L combined config (already done previously, re-fetch)
    with open(W21B_DEPLOY) as f: base = json.load(f)
    with open(BASELINE) as f: bp = json.load(f)
    g159 = fetch_trial_params("optuna_g_only", "g_only_w21b", 159)
    l626 = fetch_trial_params("optuna_l_only", "l_only_w21b", 626)
    new_params = dict(base["params"])
    for k, v in g159.items():
        if k.startswith("g_"): new_params[k] = v
    for k, v in l626.items():
        if k.startswith("l_"): new_params[k] = v
    for s in ALL_STRATS:
        new_params[f"enable_{s}"] = (s in {"g", "l"})
    new_full = {**bp, **new_params}

    p511 = {**bp, **base["params"]}
    for s in ALL_STRATS:
        p511[f"enable_{s}"] = (s in {"g", "l"})

    all_results = {}
    for wname, w in WINDOWS.items():
        print(f"\n--- Running on {wname} ---")
        print("  #511 baseline...")
        r511 = run_g_l_compounded(p511, w["dirs"], w["lo"], w["hi"])
        print("  G+L combined...")
        rgl = run_g_l_compounded(new_full, w["dirs"], w["lo"], w["hi"])
        print("  v3 R-O standalone (separate $25K)...")
        rv3 = run_v3_compounded(w["dirs"], w["lo"], w["hi"])
        all_results[wname] = {"r511": r511, "g_l_new": rgl, "v3": rv3}

        # Combined: G+L final cash + v3 final cash - 2 × $25K (each starts with $25K)
        gl_v3_total = (rgl["pnl"]) + (rv3["pnl"])
        new_with_v3_final = rgl["final_cash"] + rv3["final_cash"] - STARTING_CASH  # net 2-account treatment

        print(f"\n{'='*102}")
        print(f"  3-CONFIG COMPARISON on {wname} (compounded)")
        print(f"{'='*102}")
        print(f"  {'config':<32} {'PnL':>16} {'Multiplier':>12} {'Max DD%':>10}")
        print(f"  {'-'*32} {'-'*16} {'-'*12} {'-'*10}")
        print(f"  {'#511 (G+L)':<32} ${r511['pnl']:>+13,.0f}  {r511['multiplier']:>10.2f}x  {r511['max_dd_pct']:>8.1f}%")
        print(f"  {'NEW G+L (G#159 + L#626)':<32} ${rgl['pnl']:>+13,.0f}  {rgl['multiplier']:>10.2f}x  {rgl['max_dd_pct']:>8.1f}%")
        print(f"  {'v3 R-O only (standalone)':<32} ${rv3['pnl']:>+13,.0f}  {rv3['multiplier']:>10.2f}x  {rv3['max_dd_pct']:>8.1f}%")
        print(f"  {'-'*32} {'-'*16} {'-'*12} {'-'*10}")
        print(f"  {'NEW G+L + v3 R-O (2 accts)':<32} ${gl_v3_total:>+13,.0f}")
        print(f"\n  Breakdown of NEW G+L:")
        print(f"    G: {rgl['n_g']} trades, pnl ${rgl['g_pnl']:>+12,.0f}")
        print(f"    L: {rgl['n_l']} trades, pnl ${rgl['l_pnl']:>+12,.0f}")
        print(f"  v3 R-O: {rv3['n_trades']} trades, pnl ${rv3['v3_pnl']:>+12,.0f}")

    out = "results/combined_g_l_plus_v3_vs_511_compounded.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f: json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
