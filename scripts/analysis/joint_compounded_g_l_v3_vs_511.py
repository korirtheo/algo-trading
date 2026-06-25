"""G+L+v3 on ONE shared compounding account vs #511.

Per-day flow:
  - day starts with cash $X
  - G+L simulator runs through day -> end-of-day cash $Y
  - For each v3 candidate on that day, simulate the v3 trade using
    start-of-day cash $X for sizing (30% of $X). v3 P&Ls add to end cash.
  - Next day starts with $Y + sum(v3 P&Ls)

Approximates the multi-position engine where G+L and v3 can hold positions
on different tickers simultaneously (shared capital pool).
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
    "2022":        {"dirs": ["stored_data_2022"], "lo": "2022-01-01", "hi": "2022-12-31"},
    "2026_MarJun": {"dirs": ["stored_data_mar_may_2026", "stored_data_jun_2026"],
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


def _v3_candidate_for_day(mh, day_open, g_holds_list):
    """Return (bar_idx, ets, fp, ba) or None."""
    if mh is None or len(mh) < 2: return None
    bar0_close = float(mh.iloc[0]["Close"])
    bar0_red = bar0_close <= day_open
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
            return (i, mh.index[i], float(mh.iloc[i]["Close"]), ba)
    return None


def _v3_simulate_trade(tgc, mh, ets, fp, ba, position_dollars):
    VC = tgc.VOL_CAP_PCT; MR = tgc.MAX_REGIME_PARTICIPATION; M2 = tgc.MAX_2MIN_PARTICIPATION
    pre = mh.loc[mh.index <= ets]
    vs = float(pre["Volume"].sum()) if len(pre) > 0 else 0
    dv = fp * vs
    if dv <= 0: return 0.0
    lim = dv * (VC/100)
    ve, _, _, vr = tgc._multi_window_effective_volume(mh, ets, fp)
    if MR > 0 and vr > 0: lim = min(lim, vr * MR)
    if M2 > 0 and ve > 0: lim = min(lim, ve * M2)
    cs = min(position_dollars, lim)
    if cs < 50: return 0.0
    si = tgc._entry_slip_pct(fp, cs, ve)
    ae = fp * (1 + si/100)
    sh = cs / ae
    tp = V3_PARAMS["g_target_pct"]; sp = V3_PARAMS["g_stop_pct"]
    tm = int(V3_PARAMS["g_time_limit_min"]); trp = V3_PARAMS["g_trail_pct"]
    tap = V3_PARAMS["g_trail_activate_pct"]
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


def run_joint(params, with_v3, dirs_arg, date_lo, date_hi, label):
    from test_full import load_all_picks, MARGIN_THRESHOLD
    tgc = setup_tgc(params)
    dirs = [d for d in dirs_arg if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])

    cash = STARTING_CASH
    n_g = n_l = n_v3 = 0
    g_pnl = l_pnl = v3_pnl = 0.0
    daily_eq = [cash]

    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: daily_eq.append(cash); continue

        # G+L runs first using start-of-day cash
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash); continue

        # Capture G holds (for v3 candidate determination)
        g_holds_per_ticker = defaultdict(list)
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                p = float(st.get("pnl") or 0)
                if s == "G":
                    n_g += 1; g_pnl += p
                    et, xt = st.get("entry_time"), st.get("exit_time")
                    if et and xt: g_holds_per_ticker[st["ticker"]].append((et, xt))
                elif s == "L":
                    n_l += 1; l_pnl += p

        gl_end_cash = end_c + (unset if is_cash else 0)

        # v3 trades use start-of-day cash (same as G+L sizing). PnLs ADD to gl_end_cash.
        v3_added = 0.0
        if with_v3:
            for p in dp:
                mh = p.get("market_hour_candles")
                if mh is None or len(mh) < 2: continue
                day_open = float(mh.iloc[0]["Open"])
                ghol = g_holds_per_ticker.get(p["ticker"], [])
                cand = _v3_candidate_for_day(mh, day_open, ghol)
                if cand is None: continue
                bar_idx, ets, fp, ba = cand
                pos = cash * POSITION_PCT
                pnl = _v3_simulate_trade(tgc, mh, ets, fp, ba, pos)
                if pnl != 0.0:
                    n_v3 += 1; v3_pnl += pnl; v3_added += pnl

        cash = gl_end_cash + v3_added
        daily_eq.append(cash)

    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0
    return {"label": label, "final_cash": cash, "pnl": cash - STARTING_CASH,
            "multiplier": cash/STARTING_CASH, "max_dd_pct": dd,
            "n_g": n_g, "g_pnl": g_pnl, "n_l": n_l, "l_pnl": l_pnl,
            "n_v3": n_v3, "v3_pnl": v3_pnl}


def main():
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
        new_params[f"enable_{s}"] = (s in {"g","l"})
    new_full = {**bp, **new_params}

    p511 = {**bp, **base["params"]}
    for s in ALL_STRATS:
        p511[f"enable_{s}"] = (s in {"g","l"})

    all_results = {}
    for wname, w in WINDOWS.items():
        print(f"\n--- Running on {wname} ---")
        r511 = run_joint(p511, with_v3=False, dirs_arg=w["dirs"], date_lo=w["lo"], date_hi=w["hi"], label="#511 (G+L)")
        rgl  = run_joint(new_full, with_v3=False, dirs_arg=w["dirs"], date_lo=w["lo"], date_hi=w["hi"], label="NEW G+L")
        rfull = run_joint(new_full, with_v3=True, dirs_arg=w["dirs"], date_lo=w["lo"], date_hi=w["hi"], label="NEW G+L + v3")
        r511_v3 = run_joint(p511, with_v3=True, dirs_arg=w["dirs"], date_lo=w["lo"], date_hi=w["hi"], label="#511 + v3")
        all_results[wname] = {"r511": r511, "g_l_new": rgl, "g_l_new_v3": rfull, "r511_v3": r511_v3}

        print(f"\n{'='*100}")
        print(f"  JOINT COMPOUNDED COMPARISON on {wname}")
        print(f"{'='*100}")
        print(f"  {'config':<26} {'PnL':>16} {'Mult':>9} {'DD%':>8} {'G n':>5} {'L n':>5} {'v3 n':>6}")
        print(f"  {'-'*26} {'-'*16} {'-'*9} {'-'*8} {'-'*5} {'-'*5} {'-'*6}")
        for r in [r511, rgl, r511_v3, rfull]:
            print(f"  {r['label']:<26} ${r['pnl']:>+13,.0f}  {r['multiplier']:>7.2f}x {r['max_dd_pct']:>+7.1f}% "
                  f"{r['n_g']:>5} {r['n_l']:>5} {r['n_v3']:>6}")

    out = "results/joint_compounded_g_l_v3_vs_511.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f: json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
