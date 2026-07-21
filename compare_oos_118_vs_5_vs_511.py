#!/usr/bin/env python3
"""Compare Trial #118 vs #5 vs #511 on OOS window"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
from pathlib import Path

os.chdir(os.path.dirname(__file__))

# Import core logic
import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock

print("\n" + "="*80)
print("OOS COMPARISON: Trial #118 vs Trial #5 vs #511")
print("="*80)

STARTING_CASH = 25_000

def configure_simulator():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08

def run_backtest_by_strategy(dates, picks_by_date, snapshot):
    """Run backtest and return strategy breakdown and total PnL."""
    cash = float(STARTING_CASH)
    strat_stats = {}

    for d in dates:
        picks = picks_by_date.get(d, [])
        if not picks:
            continue

        cash_account = cash < MARGIN_THRESHOLD
        try:
            states, cash, unsettled, _ = tgc.simulate_day_combined(picks, cash, cash_account, params=snapshot)
        except Exception:
            continue

        cash += (unsettled if cash_account else 0)

        for st in states:
            if st.get("exit_reason") and st.get("pnl"):
                strat = st.get("strategy", "?")
                if strat not in strat_stats:
                    strat_stats[strat] = {"wins": 0, "losses": 0, "total_pnl": 0, "trades": 0}
                strat_stats[strat]["trades"] += 1
                strat_stats[strat]["total_pnl"] += st.get("pnl", 0)
                if st.get("pnl") > 0:
                    strat_stats[strat]["wins"] += 1
                else:
                    strat_stats[strat]["losses"] += 1

    total_pnl = cash - STARTING_CASH
    return strat_stats, total_pnl

configure_simulator()

# Load configs
with open('config/trial_g511_l626_v3_overlay.json') as f:
    config_511 = json.load(f)['params']

with open('config/trial_g511_l626_v3_full_no_trail_best.json') as f:
    data = json.load(f)
    config_118 = data['params']
    label_118 = data['source']

# Get Trial #5 from Optuna DB
try:
    import optuna
    from optuna.storages import RDBStorage
    storage = RDBStorage('postgresql://postgres@127.0.0.1:5432/optuna_g511_l626_v3_full_no_trail')
    study = optuna.load_study(study_name='g511_l626_v3_full_no_trail', storage=storage)
    trial_5_raw = study.trials[5]

    # Build complete params by loading deployed base and overlaying trial params
    base_params = dict(config_511)
    trial_5_overlay = dict(trial_5_raw.params)
    config_5 = base_params.copy()
    config_5.update(trial_5_overlay)

except Exception as e:
    print(f"ERROR loading Trial #5 from Optuna: {e}")
    sys.exit(1)

# Load OOS picks
oos_dirs = [d for d in ['stored_data_mar_may_2026', 'stored_data_jun_2026', 'stored_data'] if Path(d).exists()]
all_dates, picks_by_date = load_all_picks(oos_dirs)
dates = sorted([d for d in all_dates if any(x in d for x in ['2026-03', '2026-04', '2026-05', '2026-06'])])

print(f"\nOOS Window: {len(dates)} trading days ({dates[0]} to {dates[-1]})")
print(f"Total picks: {sum(len(picks_by_date.get(d, [])) for d in dates)}")

# Run backtests for each config
results = {}

# #511
with _param_lock:
    _oc_set_params(config_511)
    snapshot_511 = _build_param_snapshot()
strats_511, pnl_511 = run_backtest_by_strategy(dates, picks_by_date, snapshot_511)
total_trades_511 = sum(s['trades'] for s in strats_511.values())
results['#511'] = {'strats': strats_511, 'pnl': pnl_511, 'trades': total_trades_511}

# Trial #5
with _param_lock:
    _oc_set_params(config_5)
    snapshot_5 = _build_param_snapshot()
strats_5, pnl_5 = run_backtest_by_strategy(dates, picks_by_date, snapshot_5)
total_trades_5 = sum(s['trades'] for s in strats_5.values())
results['Trial #5'] = {'strats': strats_5, 'pnl': pnl_5, 'trades': total_trades_5}

# Trial #118
with _param_lock:
    _oc_set_params(config_118)
    snapshot_118 = _build_param_snapshot()
strats_118, pnl_118 = run_backtest_by_strategy(dates, picks_by_date, snapshot_118)
total_trades_118 = sum(s['trades'] for s in strats_118.values())
results['Trial #118'] = {'strats': strats_118, 'pnl': pnl_118, 'trades': total_trades_118}

print("\n" + "="*80)
print("OOS RESULTS (Mar-Jun 2026)")
print("="*80)

for label in ['#511', 'Trial #5', 'Trial #118']:
    r = results[label]
    print(f"\n{label}:")
    print(f"  PnL:       ${r['pnl']:>12,.0f}")
    print(f"  Trades:    {r['trades']:>12}")
    print(f"  Strategies:")
    for strat in sorted(r['strats'].keys()):
        s = r['strats'][strat]
        wr = s['wins'] / max(s['trades'], 1) * 100
        print(f"    {strat}: {s['trades']:3d} trades | {s['wins']}W-{s['losses']}L ({wr:5.1f}%) | ${s['total_pnl']:+9,.0f}")

# Deltas
pnl_5_vs_511 = results['Trial #5']['pnl'] - results['#511']['pnl']
pnl_118_vs_5 = results['Trial #118']['pnl'] - results['Trial #5']['pnl']
pnl_118_vs_511 = results['Trial #118']['pnl'] - results['#511']['pnl']

print("\n" + "="*80)
print("DELTAS")
print("="*80)
print(f"Trial #5 vs #511:     ${pnl_5_vs_511:>+10,.0f}")
print(f"Trial #118 vs #5:     ${pnl_118_vs_5:>+10,.0f}")
print(f"Trial #118 vs #511:   ${pnl_118_vs_511:>+10,.0f}")

print("\n")
