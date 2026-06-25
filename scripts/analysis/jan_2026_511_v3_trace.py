"""Trace #511 (G+L) Jan 2026 trade-by-trade to verify the +$337K result."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
import numpy as np

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

DATA_DIRS = ["stored_data", "stored_data_oos", "stored_data_2026_gap_fill"]


def main():
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)
    p511 = {**bp, **base["params"]}
    for s in ALL_STRATS: p511[f"enable_{s}"] = (s in {"g","l"})
    set_strategy_params(p511)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if "2026-01-01" <= d <= "2026-01-31"])
    print(f"Jan 2026 trading days: {len(dates)}")

    cash = STARTING_CASH
    daily_logs = []
    all_trades = []
    for d in dates:
        dp = picks.get(d, [])
        if not dp: daily_logs.append((d, cash, 0, 0, 0, 0)); continue
        start_cash = cash
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            daily_logs.append((d, cash, 0, 0, 0, 0)); continue
        n_g = n_l = 0; g_pnl = l_pnl = 0
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                p = float(st.get("pnl") or 0)
                if s == "G":
                    n_g += 1; g_pnl += p
                    all_trades.append({"date": d, "ticker": st.get("ticker"), "strat": "G",
                                       "pnl": p, "reason": st.get("exit_reason"),
                                       "entry": float(st.get("entry_price") or 0),
                                       "exit": float(st.get("exit_price") or 0),
                                       "cost": float(st.get("position_cost", 0)),
                                       "cash_at_entry": start_cash})
                elif s == "L":
                    n_l += 1; l_pnl += p
                    all_trades.append({"date": d, "ticker": st.get("ticker"), "strat": "L",
                                       "pnl": p, "reason": st.get("exit_reason"),
                                       "entry": float(st.get("entry_price") or 0),
                                       "exit": float(st.get("exit_price") or 0),
                                       "cost": float(st.get("position_cost", 0)),
                                       "cash_at_entry": start_cash})
        cash = end_c + (unset if is_cash else 0)
        daily_logs.append((d, cash, n_g, n_l, g_pnl, l_pnl))

    print(f"\nFinal Jan 2026 cash: ${cash:,.0f} ({cash/STARTING_CASH:.1f}x)")
    print(f"Total Jan trades: {len(all_trades)}")
    print(f"\n{'='*90}")
    print(f"  DAILY CASH PROGRESSION")
    print(f"{'='*90}")
    print(f"  {'date':<12} {'end_cash':>14} {'mult':>6} {'G':>3} {'L':>3} {'G PnL':>12} {'L PnL':>12}")
    for d, c, n_g, n_l, gp, lp in daily_logs:
        print(f"  {d:<12} ${c:>+12,.0f}  {c/STARTING_CASH:>5.1f}x {n_g:>3} {n_l:>3} ${gp:>+10,.0f} ${lp:>+10,.0f}")

    print(f"\n{'='*90}")
    print(f"  TOP 10 BIGGEST WINS — Jan 2026")
    print(f"{'='*90}")
    print(f"  {'date':<12} {'ticker':<8} {'strat':<5} {'cost':>10} {'pnl':>11} {'return':>8} {'reason':<10}")
    for t in sorted(all_trades, key=lambda x: -x["pnl"])[:10]:
        ret = (t["pnl"] / t["cost"] * 100) if t["cost"] > 0 else 0
        print(f"  {t['date']:<12} {t['ticker']:<8} {t['strat']:<5} ${t['cost']:>+8,.0f} ${t['pnl']:>+9,.0f} {ret:>+6.1f}% {str(t['reason']):<10}")

    print(f"\n{'='*90}")
    print(f"  PER-TRADE STATS BY STRATEGY")
    print(f"{'='*90}")
    for strat in ["G", "L"]:
        ts = [t for t in all_trades if t["strat"] == strat]
        if not ts: continue
        wins = [t for t in ts if t["pnl"] > 0]
        losses = [t for t in ts if t["pnl"] <= 0]
        total = sum(t["pnl"] for t in ts)
        avg_cost = sum(t["cost"] for t in ts) / len(ts) if ts else 0
        avg_win_ret = sum(t["pnl"]/t["cost"]*100 for t in wins if t["cost"]>0)/max(1,len([t for t in wins if t["cost"]>0]))
        avg_loss_ret = sum(t["pnl"]/t["cost"]*100 for t in losses if t["cost"]>0)/max(1,len([t for t in losses if t["cost"]>0]))
        print(f"  {strat}: {len(ts)} trades, WR={len(wins)/len(ts)*100:.1f}%, total ${total:>+12,.0f}")
        print(f"     avg_cost=${avg_cost:>8,.0f}, avg_win={avg_win_ret:+.1f}%, avg_loss={avg_loss_ret:+.1f}%")
        print(f"     biggest win=${max(t['pnl'] for t in ts):>+12,.0f}, biggest loss=${min(t['pnl'] for t in ts):>+12,.0f}")


if __name__ == "__main__":
    main()
