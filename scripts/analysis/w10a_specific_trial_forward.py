"""Forward test a specific W10a trial number on 2026."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import sys
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
OUTDIR = "results/w10a_forward"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]


def fetch_trial_by_number(trial_num, db="optuna_w10"):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    cur.execute("""SELECT t.trial_id, t.number, tv.value
                   FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
                   WHERE t.state='COMPLETE' AND t.number=%s LIMIT 1""", (trial_num,))
    r = cur.fetchone()
    if not r:
        return None
    tid, num, score = r
    cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
    ua = dict(cur.fetchall())
    cur.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s", (tid,))
    params = {}
    for name, val, dist in cur.fetchall():
        try:
            d = json.loads(dist) if dist else {}
            kind = d.get("name", "")
            if kind == "CategoricalDistribution":
                params[name] = d["attributes"]["choices"][int(val)]
            elif "Int" in kind:
                params[name] = int(val)
            else:
                params[name] = float(val)
        except Exception:
            params[name] = val
    c.close()
    return {"number": num, "score": float(score), "user_attrs": ua, "params": params}


def forward(trial):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(trial["params"])
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    daily_eq = [cash]
    trades = []
    daily_records = []
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            daily_eq.append(cash)
            daily_records.append({"date": d, "pnl": 0, "equity": cash, "n_trades": 0})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash)
            daily_records.append({"date": d, "pnl": 0, "equity": cash, "n_trades": 0})
            continue
        day_pnl = 0; n_today = 0
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                day_pnl += st["pnl"]
                n_today += 1
                trades.append({"date": d, "ticker": st["ticker"],
                                "strategy": st.get("strategy", "?"),
                                "pnl": float(st["pnl"]), "cost": float(st["position_cost"])})
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
        daily_records.append({"date": d, "pnl": day_pnl, "equity": cash, "n_trades": n_today})

    return cash, daily_eq, trades, daily_records, dates


def main():
    if len(sys.argv) < 2:
        print("Usage: w10a_specific_trial_forward.py <trial_number>")
        sys.exit(1)
    trial_num = int(sys.argv[1])
    os.makedirs(OUTDIR, exist_ok=True)
    trial = fetch_trial_by_number(trial_num)
    if not trial:
        print(f"Trial #{trial_num} not found in optuna_w10")
        sys.exit(1)
    p = trial["params"]; ua = trial["user_attrs"]
    print(f"Trial #{trial['number']} score=${trial['score']:,.0f}")
    print(f"  pf={ua.get('pf','-')} n={ua.get('n','-')} wr={ua.get('wr','-')}")
    print(f"  g_min_gap_pct={p.get('g_min_gap_pct','-')}  l_min_gap={p.get('l_min_gap','-')}")

    print(f"\nForward 2026 (G+L only)...")
    final, daily_eq, trades, daily_records, dates = forward(trial)
    pnl = final - STARTING_CASH

    by_strat = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in trades:
        by_strat[t["strategy"]]["n"] += 1
        by_strat[t["strategy"]]["pnl"] += t["pnl"]

    print(f"\n=== RESULT ===")
    print(f"  Final: ${final:,.0f}  ({final/STARTING_CASH:.2f}x)")
    print(f"  PnL:   ${pnl:+,.0f}")
    wins = sum(1 for t in trades if t["pnl"] > 0)
    print(f"  Trades: {len(trades)}  ({wins/max(len(trades),1)*100:.1f}% WR)")
    eq_arr = np.array(daily_eq)
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (eq_arr - peak) / peak * 100 if peak.max() > 0 else 0
    print(f"  Max DD: {dd_pct.min():.1f}%")
    print(f"\n  Per strategy:")
    for s, v in sorted(by_strat.items()):
        avg = v["pnl"] / v["n"] if v["n"] else 0
        print(f"    {s}  {v['n']:>4} trades  ${v['pnl']:>+11,.0f}  (${avg:+,.0f}/trade)")

    # Save
    out = {"trial": trial["number"], "train_score": trial["score"], "params": p,
           "forward_pnl": float(pnl), "forward_multi": float(final / STARTING_CASH),
           "forward_max_dd_pct": float(dd_pct.min()), "n_trades": len(trades),
           "per_strategy": {k: {"n": v["n"], "pnl": v["pnl"]} for k, v in by_strat.items()}}
    with open(f"{OUTDIR}/forward_trial_{trial['number']}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved {OUTDIR}/forward_trial_{trial['number']}.json")


if __name__ == "__main__":
    main()
