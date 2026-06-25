"""Compare drawdowns of #124 (deployed) vs #254 (W7 top) on 2026 — both aggregate
equity-curve DD and per-ticker / per-trade DD.

Important caveat: stored_data_jun_2026 pkl has 17 dates listed but ALL empty
(2026-05-21 -> 2026-06-16). So the "forward 98 days" is really 73 days of
real data + 25 days of skip. The aggregate equity is correct, but the trade
sample is only May 15 and earlier.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np
from collections import defaultdict

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

STARTING_CASH = 25_000
DATA_DIRS = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
BASELINE = "config/trial_432_params.json"

CONFIGS = [
    ("#124 W3 deployed",  "config/trial_124_microcap_pump_extracted.json"),
    ("#217 W7 (prior)",   "config/trial_217_w7_extracted.json"),
    ("#254 W7 (current)", "config/trial_254_w7_extracted.json"),
]


def forward_with_trades(config_path, label):
    with open(config_path) as f: cfg = json.load(f)
    params = cfg.get("params", cfg)
    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(params)
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    daily_eq = [cash]
    all_trades = []
    real_days = 0
    empty_days = 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            empty_days += 1
            daily_eq.append(cash); continue
        if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as nfp
            day_picks = nfp(day_picks, d, tgc.NEWS_MIN_ARTICLES, tgc.NEWS_REQUIRE_CATALYST)
        if not day_picks:
            daily_eq.append(cash); continue
        real_days += 1
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") is None: continue
            cost = st.get("position_cost", 0)
            if cost <= 0: continue
            pnl = st["pnl"]
            pct = pnl / cost * 100
            all_trades.append({
                "date": d,
                "ticker": st["ticker"],
                "strategy": st.get("strategy", "?"),
                "cost": cost,
                "pnl": pnl,
                "pct": pct,
                "exit_reason": st.get("exit_reason"),
            })
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    return {
        "label": label,
        "daily_eq": np.array(daily_eq),
        "trades": all_trades,
        "real_days": real_days,
        "empty_days": empty_days,
    }


def report(r):
    label = r["label"]
    eq = r["daily_eq"]
    trades = r["trades"]
    final = eq[-1]
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    print(f"  real trading days : {r['real_days']}  (skipped {r['empty_days']} empty-pick days)")
    print(f"  final equity      : ${final:,.0f}  ({final/STARTING_CASH:.2f}x)")
    print(f"  total PnL         : ${final-STARTING_CASH:+,.0f}")
    print(f"  trades            : {len(trades)}")

    if len(trades) == 0:
        print("  (no trades — nothing to break down)")
        return

    # Aggregate equity DD
    peak = np.maximum.accumulate(eq)
    dd_pct = (eq - peak) / peak * 100
    dd_dollar = eq - peak
    worst_dd_idx = int(np.argmin(dd_pct))
    print(f"\n  Aggregate equity drawdown:")
    print(f"    max DD %        : {dd_pct.min():.1f}%")
    print(f"    max DD $        : ${dd_dollar.min():,.0f}")
    print(f"    at day index    : {worst_dd_idx} (peak ${peak[worst_dd_idx]:,.0f} -> ${eq[worst_dd_idx]:,.0f})")

    # Per-trade worst losses
    pnls = np.array([t["pnl"] for t in trades])
    pcts = np.array([t["pct"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    wr = len(wins) / len(pnls) * 100 if len(pnls) else 0
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
    print(f"\n  Per-trade stats:")
    print(f"    win rate        : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"    profit factor   : {pf:.2f}")
    print(f"    avg win $       : ${wins.mean():,.0f}  ({wins.mean()/STARTING_CASH*100:.1f}% of start)" if len(wins) else "    avg win $       : -")
    print(f"    avg loss $      : ${losses.mean():,.0f}  ({losses.mean()/STARTING_CASH*100:.1f}% of start)" if len(losses) else "    avg loss $      : -")
    print(f"    avg win %       : {pcts[pnls>0].mean():.2f}%" if len(wins) else "")
    print(f"    avg loss %      : {pcts[pnls<0].mean():.2f}%" if len(losses) else "")
    print(f"    worst single $  : ${pnls.min():,.0f}  ({pcts[pnls.argmin()]:.1f}% on ${trades[int(pnls.argmin())]['cost']:,.0f} cost — {trades[int(pnls.argmin())]['ticker']} {trades[int(pnls.argmin())]['date']})")
    print(f"    worst single %  : {pcts.min():.1f}%  (${pnls[pcts.argmin()]:,.0f} on ${trades[int(pcts.argmin())]['cost']:,.0f} cost — {trades[int(pcts.argmin())]['ticker']} {trades[int(pcts.argmin())]['date']})")

    # 5 worst trades
    print(f"\n  Top 5 worst trades by $:")
    worst5 = sorted(trades, key=lambda t: t["pnl"])[:5]
    for t in worst5:
        print(f"    {t['date']}  {t['ticker']:<6}  strat {t['strategy']:<2}  "
              f"${t['pnl']:>+10,.0f}  ({t['pct']:>+6.1f}% on ${t['cost']:>9,.0f})  exit={t['exit_reason']}")
    print(f"\n  Top 5 worst trades by %:")
    worst5p = sorted(trades, key=lambda t: t["pct"])[:5]
    for t in worst5p:
        print(f"    {t['date']}  {t['ticker']:<6}  strat {t['strategy']:<2}  "
              f"${t['pnl']:>+10,.0f}  ({t['pct']:>+6.1f}% on ${t['cost']:>9,.0f})  exit={t['exit_reason']}")

    # Per-ticker rollup
    by_ticker = defaultdict(list)
    for t in trades:
        by_ticker[t["ticker"]].append(t)
    # Worst tickers by net P/L
    ticker_net = [(tk, sum(x["pnl"] for x in ts), len(ts)) for tk, ts in by_ticker.items()]
    ticker_net_sorted = sorted(ticker_net, key=lambda x: x[1])[:8]
    print(f"\n  Worst tickers by net $ (cost-net P/L across all trades on that ticker):")
    for tk, net, n in ticker_net_sorted:
        print(f"    {tk:<6}  net ${net:>+10,.0f}  over {n:>2} trade(s)")

    # Per-strategy rollup
    by_strat = defaultdict(list)
    for t in trades:
        by_strat[t["strategy"]].append(t)
    print(f"\n  Per-strategy:")
    print(f"    {'strat':<6} {'n':>4} {'wr%':>6} {'avg$':>10} {'totalPnL$':>14} {'worst$':>11}")
    for s in sorted(by_strat.keys()):
        ts = by_strat[s]
        ps = np.array([x["pnl"] for x in ts])
        wins_s = (ps > 0).sum()
        print(f"    {s:<6} {len(ts):>4} {wins_s/len(ts)*100:>5.1f}% ${ps.mean():>+9,.0f} ${ps.sum():>+12,.0f} ${ps.min():>+9,.0f}")


def main():
    results = []
    for label, path in CONFIGS:
        if not os.path.exists(path):
            print(f"[skip] {path}"); continue
        print(f"Running forward: {label}...")
        results.append(forward_with_trades(path, label))
    for r in results:
        report(r)

    # Cross-comparison table
    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    print(f"  {'config':<20} {'final$':>12} {'PnL$':>12} {'maxDD%':>8} {'maxDD$':>12} {'worst$':>11}")
    for r in results:
        eq = r["daily_eq"]
        peak = np.maximum.accumulate(eq)
        ddp = (eq - peak) / peak * 100
        ddd = eq - peak
        worst_t = min((t["pnl"] for t in r["trades"]), default=0)
        print(f"  {r['label']:<20} ${eq[-1]:>10,.0f} ${eq[-1]-STARTING_CASH:>+10,.0f} {ddp.min():>7.1f}% ${ddd.min():>+10,.0f} ${worst_t:>+9,.0f}")


if __name__ == "__main__":
    main()
