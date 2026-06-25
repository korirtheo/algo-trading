"""Backtest #511 + v3 R-O on full 2026 (Jan 1 - Jun 24) with joint compounding.

Compares:
  A) #511 (G+L) alone
  B) #511 + v3 R-O #576 layered on same compounding account

Output: per-month breakdown so we can see when v3 helps vs hurts.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
from collections import defaultdict
import numpy as np

STARTING_CASH = 25_000
POSITION_PCT = 0.30
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

V3_PARAMS = {"g_target_pct": 57.0, "g_stop_pct": 30.0, "g_time_limit_min": 27,
             "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}

DATA_DIRS_2026 = ["stored_data", "stored_data_oos", "stored_data_mar_may_2026",
                  "stored_data_jun_2026", "stored_data_2026_gap_fill"]
DATE_LO = "2026-01-01"
DATE_HI = "2026-06-30"


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


def run(params, with_v3, label):
    from test_full import load_all_picks, MARGIN_THRESHOLD
    tgc = setup_tgc(params)
    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
    print(f"  {label}: {len(dates)} days from {dates[0]} to {dates[-1]}")

    cash = STARTING_CASH
    n_g = n_l = n_v3 = 0
    g_pnl = l_pnl = v3_pnl = 0.0
    daily_eq = [(dates[0] if dates else "", cash)]
    monthly = defaultdict(lambda: {"n_g": 0, "n_l": 0, "n_v3": 0, "g_pnl": 0, "l_pnl": 0, "v3_pnl": 0,
                                    "start_cash": 0, "end_cash": 0})
    cur_month = None
    for d in dates:
        m = d[:7]
        if m != cur_month:
            if cur_month is not None: monthly[cur_month]["end_cash"] = cash
            monthly[m]["start_cash"] = cash
            cur_month = m
        dp = picks_by_date.get(d, [])
        if not dp: daily_eq.append((d, cash)); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append((d, cash)); continue
        g_holds_per_ticker = defaultdict(list)
        day_g_pnl = day_l_pnl = day_v3_pnl = 0
        day_g = day_l = day_v3 = 0
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                p = float(st.get("pnl") or 0)
                if s == "G":
                    n_g += 1; g_pnl += p; day_g_pnl += p; day_g += 1
                    et, xt = st.get("entry_time"), st.get("exit_time")
                    if et and xt: g_holds_per_ticker[st["ticker"]].append((et, xt))
                elif s == "L":
                    n_l += 1; l_pnl += p; day_l_pnl += p; day_l += 1
        gl_end_cash = end_c + (unset if is_cash else 0)
        v3_added = 0.0
        if with_v3:
            for p in dp:
                mh = p.get("market_hour_candles")
                if mh is None or len(mh) < 2: continue
                day_open = float(mh.iloc[0]["Open"])
                ghol = g_holds_per_ticker.get(p["ticker"], [])
                cand = _v3_candidate(mh, day_open, ghol)
                if cand is None: continue
                ets, fp, ba = cand
                pos = cash * POSITION_PCT
                pnl = _v3_trade(tgc, mh, ets, fp, ba, pos)
                if pnl != 0.0:
                    n_v3 += 1; v3_pnl += pnl; v3_added += pnl
                    day_v3 += 1; day_v3_pnl += pnl
        cash = gl_end_cash + v3_added
        daily_eq.append((d, cash))
        monthly[m]["n_g"] += day_g; monthly[m]["n_l"] += day_l; monthly[m]["n_v3"] += day_v3
        monthly[m]["g_pnl"] += day_g_pnl; monthly[m]["l_pnl"] += day_l_pnl; monthly[m]["v3_pnl"] += day_v3_pnl
    if cur_month is not None: monthly[cur_month]["end_cash"] = cash

    eq = np.array([c for _, c in daily_eq])
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0
    return {"label": label, "pnl": cash - STARTING_CASH, "multiplier": cash/STARTING_CASH,
            "max_dd_pct": dd, "n_g": n_g, "g_pnl": g_pnl, "n_l": n_l, "l_pnl": l_pnl,
            "n_v3": n_v3, "v3_pnl": v3_pnl, "monthly": dict(monthly)}


def main():
    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)
    p511 = {**bp, **base["params"]}
    for s in ALL_STRATS: p511[f"enable_{s}"] = (s in {"g","l"})

    print("Running on full 2026 (Jan 1 - Jun 30)...")
    r511 = run(p511, False, "#511 (G+L)")
    r511_v3 = run(p511, True, "#511 + v3")

    print(f"\n{'='*100}")
    print(f"  #511 vs #511 + v3 on full 2026 (compounded)")
    print(f"{'='*100}")
    print(f"  {'config':<22} {'PnL':>16} {'Mult':>10} {'DD%':>8} {'G':>5} {'L':>5} {'v3':>5}")
    print(f"  {'-'*22} {'-'*16} {'-'*10} {'-'*8} {'-'*5} {'-'*5} {'-'*5}")
    for r in [r511, r511_v3]:
        print(f"  {r['label']:<22} ${r['pnl']:>+13,.0f}  {r['multiplier']:>8.2f}x {r['max_dd_pct']:>+7.1f}% "
              f"{r['n_g']:>5} {r['n_l']:>5} {r['n_v3']:>5}")
    lift = ((r511_v3['pnl'] - r511['pnl']) / r511['pnl']) * 100
    print(f"\n  v3 lift over #511: ${r511_v3['pnl'] - r511['pnl']:+,.0f}  ({lift:+.1f}%)")

    print(f"\n{'='*100}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"{'='*100}")
    print(f"  {'month':<10} {'#511 PnL':>13} {'#511+v3 PnL':>15} {'v3 contrib':>13} {'G':>4} {'L':>4} {'v3':>4}")
    for m in sorted(r511_v3["monthly"].keys()):
        a = r511["monthly"].get(m, {})
        b = r511_v3["monthly"][m]
        a_pnl = a.get("end_cash", 0) - a.get("start_cash", 0) if a else 0
        b_pnl = b.get("end_cash", 0) - b.get("start_cash", 0) if b else 0
        v3_c = b.get("v3_pnl", 0)
        print(f"  {m:<10} ${a_pnl:>+11,.0f}  ${b_pnl:>+13,.0f}  ${v3_c:>+11,.0f}  "
              f"{b.get('n_g',0):>4} {b.get('n_l',0):>4} {b.get('n_v3',0):>4}")

    out = "results/backtest_511_plus_v3_2026.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f: json.dump({"r511": r511, "r511_v3": r511_v3}, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
