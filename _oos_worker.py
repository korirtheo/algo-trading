# -*- coding: utf-8 -*-
"""
OOS forward test worker - runs a batch of trials.
Called by run_v5_oos_forward_parallel.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params, ALL_STRATS
from test_full import load_all_picks, MARGIN_THRESHOLD

DATA_DIRS = ["stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data_jul_2026"]
STARTING_CASH = 25_000

def run_one(trial_data):
    group, trial = trial_data
    num = trial["number"]
    params = trial["params"]
    is_score = trial["is_score"]

    # Apply params
    set_strategy_params(params)
    for s in ALL_STRATS:
        setattr(tgc, f"ENABLE_{s.upper()}", params.get(f"enable_{s}", False))

    # Run backtest
    cash = STARTING_CASH
    unsettled = 0.0
    daily = []
    all_trades = 0
    all_wins = 0
    strat_pnl = {}
    strat_trades = {}

    for d in oos_dates:
        picks = daily_picks.get(d, [])
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(picks, cash, is_live=False)
        except:
            end_c = cash
            unset = 0.0
            states = []
        pnl = end_c - cash
        cash = end_c
        if is_cash:
            cash += unset
        daily.append(pnl)

        for st in states:
            if st["exit_reason"] is not None:
                all_trades += 1
                s = st["strategy"]
                strat_pnl[s] = strat_pnl.get(s, 0) + st["pnl"]
                strat_trades[s] = strat_trades.get(s, 0) + 1
                if st["pnl"] > 0:
                    all_wins += 1

    equity = cash
    total_pnl = equity - STARTING_CASH
    daily_arr = np.array(daily)
    sharpe = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0
    wr = all_wins / max(all_trades, 1) * 100
    gross_wins = sum(v for v in strat_pnl.values() if v > 0)
    gross_losses = abs(sum(v for v in strat_pnl.values() if v <= 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    peak = STARTING_CASH
    max_dd = 0
    eq = STARTING_CASH
    for p in daily:
        eq += p
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    enabled = sorted([k.replace("enable_", "").upper() for k, v in params.items()
                      if k.startswith("enable_") and v is True])

    return {
        "group": group, "trial": num, "is_score": is_score,
        "oos_pnl": total_pnl, "oos_equity": equity, "oos_sharpe": sharpe,
        "oos_wr": wr, "oos_trades": all_trades, "oos_pf": pf,
        "oos_max_dd": max_dd, "enabled": enabled,
        "strat_pnl": strat_pnl, "strat_trades": strat_trades,
    }

if __name__ == "__main__":
    chunk_file = sys.argv[1]
    out_file = sys.argv[2]
    worker_id = sys.argv[3]

    with open(chunk_file) as f:
        trials_data = json.load(f)

    # Load data once per worker
    print(f"  Worker {worker_id}: loading data...", flush=True)
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    global oos_dates
    oos_dates = [d for d in all_dates if d >= "2026-03-01"]
    print(f"  Worker {worker_id}: {len(oos_dates)} OOS days, running {len(trials_data)} trials...", flush=True)

    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True

    results = []
    for i, td in enumerate(trials_data):
        if i % 10 == 0:
            print(f"  Worker {worker_id}: [{i+1}/{len(trials_data)}] trial #{td[1]['number']}", flush=True)
        results.append(run_one(td))

    with open(out_file, "w") as f:
        json.dump(results, f)
    print(f"  Worker {worker_id}: done - {len(results)} results saved", flush=True)
