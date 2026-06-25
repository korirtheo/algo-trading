"""Per-strategy decomposition backtest across 2022 OOS / Train / 2026 Mar-Jun OOS.

Reports n_trades, total_pnl, WR, PF for each:
  - G (from #511 baseline)
  - L (from #511 baseline)
  - v3 R-O (top trial #576, defer-bar-1 color-aware)
  - reclaim_after_dip R-O (top trial #551)
  - any_green_above_open R-O (top trial #597) [DISPLAY ONLY: has G overlap leakage]

Strategies are run in PARALLEL (independent) — does NOT model joint capital usage.
This is the marginal contribution per strategy assuming each has its own slot.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
from collections import defaultdict

STARTING_CASH = 25_000
POSITION_PCT = 0.30
BASELINE_PATH = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

V3_PARAMS = {"g_target_pct": 57.0, "g_stop_pct": 30.0, "g_time_limit_min": 27,
             "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}
RECLAIM_PARAMS = {"g_target_pct": 51.0, "g_stop_pct": 29.0, "g_time_limit_min": 27,
                  "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}
ANY_GREEN_PARAMS = {"g_target_pct": 91.0, "g_stop_pct": 28.0, "g_time_limit_min": 6,
                    "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}

WINDOWS = {
    "2022_OOS":    {"dirs": ["stored_data_2022"], "lo": "2022-01-01", "hi": "2022-12-31"},
    "TRAIN":       {"dirs": ["stored_data_combined", "stored_data_jan_mar_2024",
                             "stored_data_apr_jun_2024", "stored_data_jul_sep_2024",
                             "stored_data_oct_dec_2024", "stored_data_jan_mar_2025",
                             "stored_data_apr_jun_2025", "stored_data_jul_2025",
                             "stored_data_oos", "stored_data"],
                   "lo": "2024-01-01", "hi": "2026-02-28"},
    "2026_MarJun": {"dirs": ["stored_data_mar_may_2026", "stored_data_jun_2026"],
                   "lo": "2026-03-01", "hi": "2026-06-30"},
}


def setup_sim():
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    with open(BASELINE_PATH) as f: baseline = json.load(f)
    with open(W21B_DEPLOY) as f: p511 = json.load(f)["params"]
    merged = {**baseline, **p511}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g","l"})
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False
    return tgc


def run_511_baseline(tgc, dirs, lo, hi):
    """Run #511 and return per-strategy trade lists + (ticker,date)->[(strat, et, xt)]."""
    from test_full import load_all_picks, MARGIN_THRESHOLD
    dirs = [d for d in dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if lo <= d <= hi])

    by_strat = defaultdict(list)  # strategy -> list of pnls
    windows = defaultdict(list)  # (ticker,date) -> [(strat, et, xt)]
    cash = STARTING_CASH
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                if s: by_strat[s].append(st.get("pnl", 0.0))
                et, xt = st.get("entry_time"), st.get("exit_time")
                if s and et and xt:
                    windows[(st.get("ticker"), d)].append((s, et, xt))
        cash = end_c + (unset if is_cash else 0)
    return by_strat, windows, dates, picks_by_date


def make_simulator(tgc):
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
    def sim(ets, fp, mh, ba, tp, sp, tm, trp, tap, pd_):
        cs, ve = caps(mh, ets, fp, pd_)
        if cs < 50: return 0.0
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
        if ep is None or et2 is None: return 0.0
        so = tgc._exit_slip_pct(ep, sh, {"mh": mh}, et2)
        return sh * (ep*(1-so/100) - ae)
    return sim


def precompute(mode, dates, picks_by_date, windows):
    """Return list of (ets, fp, mh, ba) per mode."""
    out = []
    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2: continue
            day_open = float(mh.iloc[0]["Open"])

            if mode == "any_green":
                ei = ep = ets = None
                for i, (ts, row) in enumerate(mh.iterrows()):
                    if float(row["Close"]) > day_open:
                        ei = i; ep = float(row["Close"]); ets = ts; break
                if ei is None or ep <= 0: continue
                # mode (b): skip if entry inside G/L window
                hold = windows.get((p["ticker"], d), [])
                if any(et <= ets <= xt for _, et, xt in hold): continue
                ba = mh.iloc[ei+1:]
                if len(ba) == 0: continue
                out.append((ets, ep, mh, ba))

            elif mode == "reclaim":
                ei = ep = ets = None; ever = False
                for i, (ts, row) in enumerate(mh.iterrows()):
                    c = float(row["Close"])
                    if ever and c > day_open:
                        ei = i; ep = c; ets = ts; break
                    if c <= day_open: ever = True
                if ei is None or ep <= 0: continue
                hold = windows.get((p["ticker"], d), [])
                if any(et <= ets <= xt for _, et, xt in hold): continue
                ba = mh.iloc[ei+1:]
                if len(ba) == 0: continue
                out.append((ets, ep, mh, ba))

            elif mode == "v3":
                bar0_red = float(mh.iloc[0]["Close"]) <= float(mh.iloc[0]["Open"])
                g_only = [(et, xt) for s, et, xt in windows.get((p["ticker"], d), []) if s == "G"]
                scan = 1
                if not bar0_red and g_only:
                    ge = max(x for _, x in g_only)
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
                out.append((ets, ep, mh, ba))
    return out


def stats(pnls):
    if not pnls: return {"n":0,"pnl":0.0,"wr":0.0,"pf":0.0,"mean":0.0}
    n = len(pnls); tot = sum(pnls)
    w = sum(p for p in pnls if p > 0); l = abs(sum(p for p in pnls if p <= 0))
    wr = sum(1 for p in pnls if p > 0)/n*100
    pf = w/l if l > 0 else 99.0
    return {"n":n,"pnl":tot,"wr":wr,"pf":pf,"mean":tot/n}


def main():
    tgc = setup_sim()
    sim = make_simulator(tgc)
    pos_dollars = STARTING_CASH * POSITION_PCT

    results = {}
    for wname, w in WINDOWS.items():
        print(f"\n=== {wname} ({w['lo']} .. {w['hi']}) ===")
        by_strat, windows, dates, picks = run_511_baseline(tgc, w["dirs"], w["lo"], w["hi"])
        g_stats = stats(by_strat.get("G", []))
        l_stats = stats(by_strat.get("L", []))
        print(f"  #511 G:  n={g_stats['n']}  pnl=${g_stats['pnl']:>+10,.0f}  WR={g_stats['wr']:.1f}%  PF={g_stats['pf']:.2f}")
        print(f"  #511 L:  n={l_stats['n']}  pnl=${l_stats['pnl']:>+10,.0f}  WR={l_stats['wr']:.1f}%  PF={l_stats['pf']:.2f}")

        modes = {"any_green": ANY_GREEN_PARAMS, "reclaim": RECLAIM_PARAMS, "v3": V3_PARAMS}
        mode_stats = {}
        for mname, params in modes.items():
            cands = precompute(mname, dates, picks, windows)
            pnls = []
            for ets, ep, mh, ba in cands:
                p = sim(ets, ep, mh, ba, params["g_target_pct"], params["g_stop_pct"],
                       int(params["g_time_limit_min"]), params["g_trail_pct"], params["g_trail_activate_pct"], pos_dollars)
                if p != 0.0: pnls.append(p)
            s = stats(pnls)
            mode_stats[mname] = s
            print(f"  R-O {mname:<10} n={s['n']}  pnl=${s['pnl']:>+10,.0f}  WR={s['wr']:.1f}%  PF={s['pf']:.2f}")

        results[wname] = {"G": g_stats, "L": l_stats, **mode_stats}

    # Summary table
    print(f"\n{'='*110}")
    print(f"  DECOMPOSITION SUMMARY (per-strategy standalone PnL)")
    print(f"{'='*110}")
    print(f"  {'window':<14} {'G':>14} {'L':>12} {'v3':>14} {'reclaim':>14} {'any_green*':>14}")
    print(f"  {'-'*14} {'-'*14} {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
    for wname, r in results.items():
        print(f"  {wname:<14} ${r['G']['pnl']:>+12,.0f} ${r['L']['pnl']:>+10,.0f} "
              f"${r['v3']['pnl']:>+12,.0f} ${r['reclaim']['pnl']:>+12,.0f} ${r['any_green']['pnl']:>+12,.0f}")

    tot = {k: sum(r[k]['pnl'] for r in results.values()) for k in ["G","L","v3","reclaim","any_green"]}
    print(f"  {'-'*14} {'-'*14} {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'TOTAL':<14} ${tot['G']:>+12,.0f} ${tot['L']:>+10,.0f} ${tot['v3']:>+12,.0f} ${tot['reclaim']:>+12,.0f} ${tot['any_green']:>+12,.0f}")

    print(f"\n  * any_green has G-leakage (fires at bar 0 when G eventually fires) — DO NOT deploy")
    print(f"\n  G+L (current deploy):  ${tot['G']+tot['L']:>+,.0f}")
    print(f"  G+L+v3 (proposed):     ${tot['G']+tot['L']+tot['v3']:>+,.0f}  "
          f"(+{100*tot['v3']/(tot['G']+tot['L']):.0f}% over G+L)")
    print(f"  G+L+reclaim (alt):     ${tot['G']+tot['L']+tot['reclaim']:>+,.0f}  "
          f"(+{100*tot['reclaim']/(tot['G']+tot['L']):.0f}% over G+L)")

    out = "results/strategy_decomposition.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
