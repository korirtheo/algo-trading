
import sys
import os
import json
import argparse
import pandas as pd
import optuna
import random
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_full import load_all_picks
from optimize_combined import set_strategy_params, _build_param_snapshot
from test_green_candle_combined import simulate_day_combined, STARTING_CASH, MARGIN_THRESHOLD

def run_single_trial(args):
    params, all_dates, picks_by_date, trial_number = args
    set_strategy_params(params)
    param_snapshot = _build_param_snapshot()

    cash = STARTING_CASH
    daily_pnl = []

    for date in all_dates:
        day_picks = picks_by_date.get(date, [])
        if not day_picks:
            daily_pnl.append(0)
            continue

        is_cash_account = cash < MARGIN_THRESHOLD
        try:
            states, cash, unsettled_cash, _ = simulate_day_combined(day_picks, cash, is_cash_account, params=param_snapshot)
            day_pnl_val = 0
            for state in states:
                if state.get('pnl') is not None:
                    day_pnl_val += state['pnl']
            daily_pnl.append(day_pnl_val)
        except Exception as e:
            # print(f"Error simulating day {date}: {e}")
            daily_pnl.append(0)
    
    total_pnl = sum(daily_pnl)
    print(f"Trial #{trial_number}: PnL = ${total_pnl:,.2f}")
    return trial_number, total_pnl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date-lo', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--date-hi', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('data_dirs', nargs='+', help='List of data directories')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')

    args = parser.parse_args()

    with open('config/trial_432_params.json') as f:
        baseline_params = json.load(f)

    all_dates, picks_by_date = load_all_picks(args.data_dirs)
    dates_in_range = [d for d in all_dates if args.date_lo <= d <= args.date_hi]
    picks_in_range = {d: picks_by_date[d] for d in dates_in_range}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = 'postgresql://postgres@127.0.0.1:5432/optuna_gl_trail'
    study = optuna.load_study(study_name='gl_trail_v3', storage=storage)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    top_50_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:50]
    top_50_numbers = {t.number for t in top_50_trials}
    
    random_50_trials = random.sample(completed_trials, 50)

    trials_to_run = top_50_trials + random_50_trials

    tasks = []
    for trial in trials_to_run:
        params = baseline_params.copy()
        params.update(trial.params)
        tasks.append((params, dates_in_range, picks_in_range, trial.number))

    with multiprocessing.Pool(processes=args.workers) as pool:
        results = pool.map(run_single_trial, tasks)

    results.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Top 20 Performing Trials ---")
    for i, (trial_number, pnl) in enumerate(results[:20]):
        trial_type = "Top 50" if trial_number in top_50_numbers else "Random 50"
        print(f"{i+1}. Trial #{trial_number} ({trial_type}): PnL = ${pnl:,.2f}")

if __name__ == "__main__":
    main()
