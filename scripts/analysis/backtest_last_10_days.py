"""
Backtest last 10 trading days with deployed G #511 + L #626 + v3 overlay.
Uses stored_data_jun_2026 + stored_data_jul_2026.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import defaultdict
import numpy as np
import test_green_candle_combined as tgc
from test_full import load_all_picks

STARTING_CASH = 25_000
POSITION_PCT = 0.30
DATA_DIRS = ["stored_data_combined", "stored_data_jun_2026"]

# 2026 year, excluding newly downloaded Jul data
DATE_LO = "2026-01-01"
DATE_HI = "2026-12-31"

# Load config
cfg_path = "config/trial_g511_l626_v3_overlay.json"
with open(cfg_path) as f:
    config = json.load(f)

params = config["params"]
v3_params = config.get("v3_overlay", {})

# Setup strategy params
from optimize_combined import set_strategy_params
set_strategy_params(params)

# Slippage settings
tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.NEWS_MODULATOR_ENABLED = False


def v3_candidate(mh, day_open, g_holds_list):
    """Find v3 entry after G hold expires."""
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


def v3_trade(mh, ets, fp, ba, pos_dollars):
    """Simulate v3 trade with exit logic."""
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
    tp = v3_params.get("target_pct", 57.0)
    sp = v3_params.get("stop_pct", 30.0)
    tm = int(v3_params.get("time_limit_min", 27))
    trp = v3_params.get("trail_pct", 0.5)
    tap = v3_params.get("trail_activate_pct", 0.0)
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


def run_backtest():
    print(f"Loading picks from {DATA_DIRS}...")
    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
    print(f"  {len(dates)} trading days: {dates[0]} to {dates[-1]}")
    print(f"  Config: {config['label']}")
    print()

    cash = STARTING_CASH
    trade_log = []
    daily_log = []
    v3_log = []

    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp:
            daily_log.append({
                "date": d, "trades": 0, "g": 0, "l": 0, "v3": 0,
                "g_pnl": 0, "l_pnl": 0, "v3_pnl": 0,
                "total_pnl": 0, "end_cash": round(cash, 2),
            })
            continue

        is_cash = cash < 25000  # MARGIN_THRESHOLD
        states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)

        # Track G/L trades and G hold times
        day_trades = []
        g_holds_per_ticker = defaultdict(list)
        day_g_pnl = day_l_pnl = day_v3_pnl = 0
        day_g = day_l = day_v3 = 0

        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                p = float(st.get("pnl") or 0)
                entry_p = float(st.get("entry_price") or 0)
                exit_p = float(st.get("exit_price") or 0)
                ret_pct = ((exit_p/entry_p)-1)*100 if entry_p else 0
                trade = {
                    "date": d, "ticker": st["ticker"], "strategy": s,
                    "entry": entry_p, "exit": exit_p,
                    "pnl": p, "ret_pct": round(ret_pct, 2),
                    "reason": st.get("exit_reason"),
                }
                if s == "G":
                    day_g += 1; day_g_pnl += p
                    et = st.get("entry_time"); xt = st.get("exit_time")
                    if et and xt: g_holds_per_ticker[st["ticker"]].append((et, xt))
                elif s == "L":
                    day_l += 1; day_l_pnl += p
                day_trades.append(trade)

        # v3 after G holds expire
        gl_end_cash = end_c + (unset if is_cash else 0)
        v3_added = 0.0
        for p in dp:
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2: continue
            day_open = float(mh.iloc[0]["Open"])
            ghol = g_holds_per_ticker.get(p["ticker"], [])
            cand = v3_candidate(mh, day_open, ghol)
            if cand is None: continue
            ets, fp, ba = cand
            pos = cash * POSITION_PCT
            pnl = v3_trade(mh, ets, fp, ba, pos)
            if pnl != 0.0:
                day_v3 += 1; day_v3_pnl += pnl; v3_added += pnl
                v3_log.append({
                    "date": d, "ticker": p["ticker"], "pnl": round(pnl, 2),
                })

        cash = gl_end_cash + v3_added
        day_pnl = day_g_pnl + day_l_pnl + v3_added
        daily_log.append({
            "date": d, "trades": len(day_trades),
            "g": day_g, "l": day_l, "v3": day_v3,
            "g_pnl": round(day_g_pnl, 2), "l_pnl": round(day_l_pnl, 2),
            "v3_pnl": round(v3_added, 2),
            "total_pnl": round(day_pnl, 2), "end_cash": round(cash, 2),
        })
        trade_log.extend(day_trades)

    # ── Results ──
    total_pnl = cash - STARTING_CASH
    total_mult = cash / STARTING_CASH

    print(f"{'='*90}")
    print(f"  BACKTEST: {config['label']}")
    print(f"  Period: {DATE_LO} to {DATE_HI} ({len(dates)} trading days)")
    print(f"{'='*90}")
    print()
    print(f"  Starting Cash: ${STARTING_CASH:,.2f}")
    print(f"  Ending Cash:   ${cash:,.2f}")
    print(f"  Total PnL:     ${total_pnl:+,.2f}  ({total_mult:.2f}x)")
    print()

    # Trade stats
    all_pnls = [t["pnl"] for t in trade_log]
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p < 0]
    n_total = len(trade_log)

    # Per-strategy
    print(f"{'─'*90}")
    print(f"{'Strategy':>10} {'Trades':>8} {'Wins':>6} {'Losses':>6} {'WR':>6} {'Total PnL':>12} {'Avg PnL':>10} {'Best':>10} {'Worst':>10}")
    print(f"{'─'*90}")
    for strat in ["G", "L", "V3"]:
        st = [t for t in trade_log if t["strategy"] == strat]
        sv = [t for t in v3_log]
        if strat == "V3":
            sp = [t["pnl"] for t in sv]
            sw = [p for p in sp if p > 0]
            sl = [p for p in sp if p < 0]
            sc = len(sv)
        else:
            sp = [t["pnl"] for t in st]
            sw = [p for p in sp if p > 0]
            sl = [p for p in sp if p < 0]
            sc = len(st)
        if sc > 0:
            wr = len(sw)/sc*100
            tot = sum(sp)
            avg = tot/sc
            best = max(sp) if sw else 0
            worst = min(sp) if sl else 0
            print(f"{strat:>10} {sc:>8} {len(sw):>6} {len(sl):>6} {wr:>5.1f}% {tot:>+10,.0f} {avg:>+9,.2f} {best:>+9,.2f} {worst:>+9,.2f}")
    print(f"{'─'*90}")
    tot = sum(all_pnls) + sum(t["pnl"] for t in v3_log)
    if n_total > 0:
        print(f"{'ALL':>10} {n_total:>8} {len(wins):>6} {len(losses):>6}"
              f" {len(wins)/n_total*100:>5.1f}% {total_pnl:>+10,.0f}"
              f" {total_pnl/n_total:>+9,.2f}" if n_total else "N/A")
    print()

    # Daily breakdown
    print(f"{'─'*90}")
    print(f"{'Date':<12} {'Trades':>7} {'G':>4} {'L':>4} {'V3':>4} {'G PnL':>10} {'L PnL':>10} {'V3 PnL':>10} {'Day PnL':>10} {'Equity':>10}")
    print(f"{'─'*90}")
    for dl in daily_log:
        print(f"{dl['date']:<12} {dl['trades']:>7} {dl['g']:>4} {dl['l']:>4} {dl['v3']:>4}"
              f" {dl['g_pnl']:>+9,.0f} {dl['l_pnl']:>+9,.0f} {dl['v3_pnl']:>+9,.0f}"
              f" {dl['total_pnl']:>+9,.0f} ${dl['end_cash']:>8,.0f}")

    # Trade list
    print(f"\n{'─'*90}")
    print(f"  ALL TRADES")
    print(f"{'─'*90}")
    print(f"{'Date':<12} {'Ticker':>8} {'Strat':>6} {'Entry':>8} {'Exit':>8} {'Ret%':>7} {'PnL':>10} {'Reason':>20}")
    print(f"{'─'*90}")
    for t in sorted(trade_log, key=lambda x: (x["date"], x["strategy"], x["ticker"])):
        print(f"{t['date']:<12} {t['ticker']:>8} {t['strategy']:>6}"
              f" ${t['entry']:<6.2f} ${t['exit']:<6.2f}"
              f" {t['ret_pct']:>+6.2f}% ${t['pnl']:>+8,.2f}  {t['reason']:>20}")
    for t in sorted(v3_log, key=lambda x: (x["date"], x["ticker"])):
        print(f"{t['date']:<12} {t['ticker']:>8} {'V3':>6} {'-':>8} {'-':>8} {'-':>7} ${t['pnl']:>+8,.2f}  {'V3_OVERLAY':>20}")

    # Save
    out = "results/backtest_2026_excl_new.json"
    os.makedirs("results", exist_ok=True)
    result = {
        "label": config["label"], "period": f"{DATE_LO}_{DATE_HI}",
        "starting_cash": STARTING_CASH, "end_cash": round(cash, 2),
        "total_pnl": round(total_pnl, 2), "multiplier": round(total_mult, 4),
        "n_trades": n_total, "n_wins": len(wins), "n_losses": len(losses),
        "win_rate": round(len(wins)/n_total*100, 1) if n_total else 0,
        "daily": daily_log, "trades": trade_log, "v3_trades": v3_log,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    run_backtest()
