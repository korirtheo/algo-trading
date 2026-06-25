"""L-strategy decay diagnostic — what changed in 2026 Mar-Jun?

Pulls all L trades from #511 baseline across full backtest, breaks down by month.
Surfaces: trade count, win rate, total pnl, mean pnl, biggest losers,
and the specific (ticker, date, pnl) tuples for 2026 Mar-Jun losses.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
from collections import defaultdict

STARTING_CASH = 25_000
BASELINE_PATH = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = [
    "stored_data_2022", "stored_data_combined",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos", "stored_data",
    "stored_data_mar_may_2026", "stored_data_jun_2026",
    "stored_data_2026_gap_fill",
]


def main():
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

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

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    print(f"Loading {len(dirs)} dirs...")
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted(all_dates)
    print(f"Trading days: {len(dates)}  ({dates[0]} .. {dates[-1]})")

    print("Running #511 baseline to capture all L trades...")
    cash = STARTING_CASH
    l_trades = []
    g_pnl_by_month = defaultdict(float)
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception: continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                if st.get("strategy") == "L":
                    l_trades.append({
                        "ticker": st.get("ticker"), "date": d, "month": d[:7],
                        "pnl": float(st.get("pnl", 0.0)),
                        "entry_time": str(st.get("entry_time")),
                        "exit_time": str(st.get("exit_time")),
                        "exit_reason": st.get("exit_reason"),
                        "entry_price": float(st.get("entry_price") or 0),
                        "exit_price": float(st.get("exit_price") or 0),
                    })
                if st.get("strategy") == "G":
                    g_pnl_by_month[d[:7]] += float(st.get("pnl", 0.0))
        cash = end_c + (unset if is_cash else 0)
    print(f"  Total L trades: {len(l_trades)}")

    # Monthly L breakdown
    by_month = defaultdict(list)
    for t in l_trades: by_month[t["month"]].append(t)

    print(f"\n{'='*92}")
    print(f"  L MONTHLY BREAKDOWN")
    print(f"{'='*92}")
    print(f"  {'month':<10} {'n':>4} {'pnl':>11} {'wr':>6} {'mean':>8} {'worst':>10} {'G_pnl':>12}")
    for m in sorted(by_month.keys()):
        ts = by_month[m]
        n = len(ts); pnl = sum(t["pnl"] for t in ts)
        wr = sum(1 for t in ts if t["pnl"] > 0)/n*100
        mean = pnl/n
        worst = min(t["pnl"] for t in ts)
        g_m = g_pnl_by_month.get(m, 0)
        mark = "  <-- LOSS" if pnl < 0 else ""
        print(f"  {m:<10} {n:>4} ${pnl:>+9,.0f} {wr:>5.1f}% ${mean:>+6,.0f} ${worst:>+8,.0f} ${g_m:>+10,.0f}{mark}")

    # 2026 Mar-Jun deep dive
    print(f"\n{'='*92}")
    print(f"  2026 Mar-Jun L trades — all of them")
    print(f"{'='*92}")
    mj_trades = [t for t in l_trades if t["date"] >= "2026-03-01" and t["date"] <= "2026-06-30"]
    mj_trades.sort(key=lambda t: t["pnl"])
    print(f"  {'date':<12} {'ticker':<8} {'entry':>8} {'exit':>8} {'pnl':>10} {'reason':<10}")
    for t in mj_trades:
        print(f"  {t['date']:<12} {t['ticker']:<8} ${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"${t['pnl']:>+8,.0f} {str(t['exit_reason']):<10}")

    # Exit reason distribution per period
    print(f"\n{'='*92}")
    print(f"  L EXIT REASON distribution (TRAIN vs 2026 Mar-Jun)")
    print(f"{'='*92}")
    train_trades = [t for t in l_trades if "2024-01-01" <= t["date"] <= "2026-02-28"]
    print(f"  {'reason':<15} {'TRAIN n':>8} {'TRAIN %':>8} {'TRAIN avg pnl':>15} {'26MJ n':>8} {'26MJ %':>8} {'26MJ avg pnl':>15}")
    all_reasons = set(t["exit_reason"] for t in l_trades)
    for r in sorted(all_reasons, key=lambda x: str(x)):
        tr = [t for t in train_trades if t["exit_reason"] == r]
        mj = [t for t in mj_trades if t["exit_reason"] == r]
        tr_pct = 100*len(tr)/len(train_trades) if train_trades else 0
        mj_pct = 100*len(mj)/len(mj_trades) if mj_trades else 0
        tr_avg = sum(t["pnl"] for t in tr)/len(tr) if tr else 0
        mj_avg = sum(t["pnl"] for t in mj)/len(mj) if mj else 0
        print(f"  {str(r):<15} {len(tr):>8} {tr_pct:>7.1f}% ${tr_avg:>+12,.0f} {len(mj):>8} {mj_pct:>7.1f}% ${mj_avg:>+12,.0f}")

    out = "results/l_decay_diagnostic.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"monthly": {m: {"n": len(by_month[m]), "pnl": sum(t["pnl"] for t in by_month[m]),
                                    "wr": sum(1 for t in by_month[m] if t["pnl"]>0)/len(by_month[m])*100}
                                for m in sorted(by_month.keys())},
                   "mar_jun_2026_trades": mj_trades},
                   f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
