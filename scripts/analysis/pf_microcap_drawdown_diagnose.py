"""Diagnose the day 100-150 drawdown in W3 PF microcap-pump #124 forward 2024.

For each day in the window:
  - Date, regime/shape label
  - Trades that fired (strategy, ticker, entry, exit, reason, PnL %)
  - Daily PnL ($ and %)
  - Equity at end of day

Highlights:
  - Top 10 worst losing days in the window
  - Pattern: were they regime-mismatched? Same strategies failing?
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape

DB = "results/wf_pf_microcap_pump_noX/W3_train_2021_2022_2023_test_2024.db"
BASELINE = "config/trial_432_params.json"
TEST_DIRS = ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
             "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"]
STARTING_CASH = 25_000

WIN_START = 95   # show a bit before the dropoff
WIN_END   = 155


def _merged_params(best_params):
    with open(BASELINE) as f:
        b = json.load(f)
    m = dict(b); m.update(best_params); return m


def main():
    s = optuna.load_study(study_name=optuna.get_all_study_names(f"sqlite:///{DB}")[0],
                          storage=f"sqlite:///{DB}")
    b = s.best_trial
    set_strategy_params(_merged_params(dict(b.params)))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    dirs = [d for d in TEST_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    test_dates = [d for d in all_dates if d.startswith("2024")]

    cash = STARTING_CASH
    daily = []
    for idx, d in enumerate(test_dates):
        day_picks = picks.get(d, [])
        regime = classify_regime(day_picks) if day_picks else "dead"
        sig = day_signature(day_picks) if day_picks else None
        shape = classify_shape(sig, regime) if sig else "empty"

        if not day_picks:
            daily.append({"idx": idx, "date": d, "regime": regime, "shape": shape,
                          "pnl": 0, "equity": cash, "trades": []})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception as e:
            daily.append({"idx": idx, "date": d, "regime": regime, "shape": shape,
                          "pnl": 0, "equity": cash, "trades": [], "error": str(e)[:60]})
            continue
        day_trades = []
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                day_trades.append({
                    "strategy": st.get("strategy"),
                    "ticker": st["ticker"],
                    "reason": st.get("exit_reason"),
                    "pnl": st["pnl"],
                    "pct": (st["pnl"] / st["position_cost"] * 100)
                           if st["position_cost"] > 0 else 0,
                    "position_cost": st["position_cost"],
                })
        pnl = end_c - cash
        cash = end_c
        if is_cash:
            cash += unset
        daily.append({"idx": idx, "date": d, "regime": regime, "shape": shape,
                      "pnl": pnl, "equity": cash, "trades": day_trades})

    # Slice to window
    window = [r for r in daily if WIN_START <= r["idx"] <= WIN_END]

    # Per-day printout
    print(f"\n{'idx':<4} {'date':<11} {'shape':<14} {'eq_start':>10} {'pnl':>10} {'eq_end':>10}  trades")
    print("-" * 110)
    eq_prev = daily[WIN_START - 1]["equity"] if WIN_START > 0 else STARTING_CASH
    for r in window:
        eq_start = eq_prev
        pct = (r["pnl"] / eq_start * 100) if eq_start > 0 else 0
        ntr = len(r["trades"])
        # Brief trade summary
        if ntr:
            strats = {}
            for t in r["trades"]:
                strats[t["strategy"]] = strats.get(t["strategy"], 0) + 1
            strat_str = ",".join(f"{k}({v})" for k, v in sorted(strats.items()))
        else:
            strat_str = "-"
        print(f"{r['idx']:<4} {r['date']:<11} {r['shape']:<14} "
              f"${eq_start:>9,.0f} ${r['pnl']:>+9,.0f} ({pct:>+5.1f}%) ${r['equity']:>9,.0f}  "
              f"{ntr:>2} [{strat_str}]")
        eq_prev = r["equity"]

    # Top losers
    losers = sorted([r for r in window if r["pnl"] < 0], key=lambda r: r["pnl"])[:10]
    print(f"\n{'='*88}")
    print(f"TOP 10 WORST DAYS IN WINDOW (days {WIN_START}-{WIN_END})")
    print(f"{'='*88}")
    print(f"{'date':<11} {'shape':<14} {'$ pnl':>11}  trades")
    for r in losers:
        ntr = len(r["trades"])
        print(f"{r['date']:<11} {r['shape']:<14} ${r['pnl']:>+10,.0f}  {ntr} trades")
        for t in r["trades"]:
            print(f"  └─ {t['strategy']:<3} {t['ticker']:<6} "
                  f"({t['reason']:<10}) pnl=${t['pnl']:>+9,.0f} ({t['pct']:>+6.2f}%)")

    # Shape distribution in window
    print(f"\n{'='*88}")
    print(f"SHAPE DISTRIBUTION IN WINDOW")
    print(f"{'='*88}")
    shape_counts = {}
    shape_pnl = {}
    for r in window:
        sh = r["shape"]
        shape_counts[sh] = shape_counts.get(sh, 0) + 1
        shape_pnl[sh] = shape_pnl.get(sh, 0) + r["pnl"]
    for sh in sorted(shape_counts, key=lambda x: shape_pnl.get(x, 0)):
        n = shape_counts[sh]; total = shape_pnl[sh]
        avg = total / n if n > 0 else 0
        print(f"  {sh:<14}  {n:>3} days  total ${total:>+11,.0f}  avg ${avg:>+8,.0f}/day")

    # Per-strategy in window
    print(f"\n{'='*88}")
    print(f"PER-STRATEGY IN WINDOW (which strategies were the losers?)")
    print(f"{'='*88}")
    strat_stats = {}
    for r in window:
        for t in r["trades"]:
            s = t["strategy"]
            if s not in strat_stats:
                strat_stats[s] = {"n": 0, "wins": 0, "pnl": 0.0}
            strat_stats[s]["n"] += 1
            strat_stats[s]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                strat_stats[s]["wins"] += 1
    for s in sorted(strat_stats, key=lambda x: -strat_stats[x]["pnl"]):
        v = strat_stats[s]
        wr = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
        print(f"  {s}  trades={v['n']:>3}  wins={v['wins']:>3} ({wr:>5.1f}%)  pnl=${v['pnl']:>+11,.0f}")


if __name__ == "__main__":
    main()
