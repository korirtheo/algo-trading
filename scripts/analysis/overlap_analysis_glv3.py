"""Analyze time-concurrent overlap between G, L, and v3 positions.

Measures what percentage of market minutes have 2+ strategies holding simultaneously.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
from collections import defaultdict
import pandas as pd

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


def _v3_trade(tgc, mh, ets, fp, ba, pos_dollars, day_open):
    pre = mh.loc[mh.index <= ets]
    vs = float(pre["Volume"].sum()) if len(pre) > 0 else 0
    dv = fp * vs
    if dv <= 0: return 0.0, None
    lim = dv * (tgc.VOL_CAP_PCT/100)
    ve, _, _, vr = tgc._multi_window_effective_volume(mh, ets, fp)
    if tgc.MAX_REGIME_PARTICIPATION > 0 and vr > 0: lim = min(lim, vr * tgc.MAX_REGIME_PARTICIPATION)
    if tgc.MAX_2MIN_PARTICIPATION > 0 and ve > 0: lim = min(lim, ve * tgc.MAX_2MIN_PARTICIPATION)
    cs = min(pos_dollars, lim)
    if cs < 50: return 0.0, None
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
    if ep is None or et2 is None: return 0.0, None
    so = tgc._exit_slip_pct(ep, sh, {"mh": mh}, et2)
    pnl = sh * (ep*(1-so/100) - ae)
    return pnl, (ets, et2, fp, ae, sh, pnl)


def minute_range(et, xt, day_open_ts):
    """Produce set of minute-level timestamps from entry to exit."""
    start_ts = max(et, day_open_ts)
    end_ts = min(xt, day_open_ts.replace(hour=15, minute=58))
    mins = set()
    cur = start_ts
    while cur <= end_ts:
        mins.add(cur.floor("min"))
        cur += pd.Timedelta(minutes=1)
    return mins


def _to_naive(ts):
    if hasattr(ts, 'tz') and ts.tz is not None:
        return ts.tz_localize(None)
    return ts


def run_overlap_analysis(params, label):
    import test_green_candle_combined as tgc
    from test_full import load_all_picks, MARGIN_THRESHOLD

    tgc = setup_tgc(params)
    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
    print(f"  {label}: {len(dates)} days")

    cash = STARTING_CASH
    g_minutes = set(); l_minutes = set(); v3_minutes = set()

    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            continue

        g_holds_per_ticker = defaultdict(list)
        day_g_positions = []    # (entry_ts, exit_ts, ticker)
        day_l_positions = []

        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                et = st.get("entry_time")
                xt = st.get("exit_time")
                if et and xt:
                    et_ts = pd.Timestamp(et)
                    xt_ts = pd.Timestamp(xt)
                    if s == "G":
                        day_g_positions.append((et_ts, xt_ts, st["ticker"]))
                        g_holds_per_ticker[st["ticker"]].append((et_ts, xt_ts))
                    elif s == "L":
                        day_l_positions.append((et_ts, xt_ts, st["ticker"]))

        # v3 overlay
        day_v3_positions = []
        for p in dp:
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2: continue
            day_open = float(mh.iloc[0]["Open"])
            ghol = g_holds_per_ticker.get(p["ticker"], [])
            cand = _v3_candidate(mh, day_open, ghol)
            if cand is None: continue
            ets, fp, ba = cand
            pos = cash * POSITION_PCT
            pnl, trade_data = _v3_trade(tgc, mh, ets, fp, ba, pos, day_open)
            if trade_data is not None:
                day_v3_positions.append((trade_data[0], trade_data[1], p["ticker"]))

        # Convert positions to minute-level sets for this day
        day_open_ts = pd.Timestamp(d + " 09:30")
        g_mins_day = set()
        for et, xt, ticker in day_g_positions:
            g_mins_day |= minute_range(_to_naive(pd.Timestamp(et)), _to_naive(pd.Timestamp(xt)), day_open_ts)
        l_mins_day = set()
        for et, xt, ticker in day_l_positions:
            l_mins_day |= minute_range(_to_naive(pd.Timestamp(et)), _to_naive(pd.Timestamp(xt)), day_open_ts)
        v3_mins_day = set()
        for et, xt, ticker in day_v3_positions:
            v3_mins_day |= minute_range(_to_naive(pd.Timestamp(et)), _to_naive(pd.Timestamp(xt)), day_open_ts)

        g_minutes |= g_mins_day
        l_minutes |= l_mins_day
        v3_minutes |= v3_mins_day

        cash = end_c + (unset if is_cash else 0)

    print(f"\n  === Overlap Analysis ===")
    print(f"  G total hold minutes: {len(g_minutes):,}")
    print(f"  L total hold minutes: {len(l_minutes):,}")
    print(f"  v3 total hold minutes: {len(v3_minutes):,}")
    print(f"  G+L overlap minutes: {len(g_minutes & l_minutes):,}")
    print(f"  G+v3 overlap minutes: {len(g_minutes & v3_minutes):,}")
    print(f"  L+v3 overlap minutes: {len(l_minutes & v3_minutes):,}")
    print(f"  G+L+v3 all three: {len(g_minutes & l_minutes & v3_minutes):,}")

    # Percentages relative to each strategy's total
    if len(g_minutes):
        print(f"\n  As % of G's hold time:")
        print(f"    G alone:        {len(g_minutes - l_minutes - v3_minutes)/len(g_minutes)*100:.1f}%")
        print(f"    G+L concurrent: {len(g_minutes & l_minutes)/len(g_minutes)*100:.1f}%")
        print(f"    G+v3 concurrent: {len(g_minutes & v3_minutes)/len(g_minutes)*100:.1f}%")
    if len(l_minutes):
        print(f"\n  As % of L's hold time:")
        print(f"    L alone:        {len(l_minutes - g_minutes - v3_minutes)/len(l_minutes)*100:.1f}%")
        print(f"    L+G concurrent: {len(l_minutes & g_minutes)/len(l_minutes)*100:.1f}%")
        print(f"    L+v3 concurrent: {len(l_minutes & v3_minutes)/len(l_minutes)*100:.1f}%")
    if len(v3_minutes):
        print(f"\n  As % of v3's hold time:")
        print(f"    v3 alone:       {len(v3_minutes - g_minutes - l_minutes)/len(v3_minutes)*100:.1f}%")
        print(f"    v3+G concurrent: {len(v3_minutes & g_minutes)/len(v3_minutes)*100:.1f}%")
        print(f"    v3+L concurrent: {len(v3_minutes & l_minutes)/len(v3_minutes)*100:.1f}%")

    # Total market minutes across all trading days
    n_days = len(dates)
    mkt_mins = n_days * 388  # 6.5 hrs = 390 min, minus first/last = 388 2-min candles
    any_held = len(g_minutes | l_minutes | v3_minutes)
    print(f"\n  Out of {n_days} trading days × 388 min = {mkt_mins:,} market minutes:")
    print(f"    Any position held: {any_held:,} ({any_held/mkt_mins*100:.1f}%)")
    print(f"    G+L overlap:       {len(g_minutes & l_minutes):,} ({len(g_minutes & l_minutes)/mkt_mins*100:.1f}%)")
    print(f"    G+v3 overlap:      {len(g_minutes & v3_minutes):,} ({len(g_minutes & v3_minutes)/mkt_mins*100:.1f}%)")
    print(f"    L+v3 overlap:      {len(l_minutes & v3_minutes):,} ({len(l_minutes & v3_minutes)/mkt_mins*100:.1f}%)")


def main():
    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)
    p511 = {**bp, **base["params"]}
    for s in ALL_STRATS: p511[f"enable_{s}"] = (s in {"g","l"})
    run_overlap_analysis(p511, "G511 + L626 + v3")


if __name__ == "__main__":
    main()
