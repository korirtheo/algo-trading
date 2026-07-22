"""
Flexible Wide Forward Test: G+L Only
=====================================

Usage:
    python oos_wide_forward_test_flex.py --study gl_trail_v3 --trials 538,591,592 --workers 4
    python oos_wide_forward_test_flex.py --study gl_trail_v3 --all --workers 4
    python oos_wide_forward_test_flex.py --study gl_trail_v3 --top 50 --workers 4

Features:
- Choose any study (gl_trail_v3, gl_trail_v3_clean, etc.)
- Always filter to G+L parameters only
- Run specific trials or top N by score
- Clean parameter handling for G+L only
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add current directory to path
sys.path.insert(0, ".")

import optuna

# Import the combined test module
import test_green_candle_combined as tgc
from test_full import MARGIN_THRESHOLD, load_all_picks

# Global constants (adjust as needed)
STARTING_CASH = 25_000
OUT_CSV = "oos_wide_forward_test_g_l_flex.csv"

# Data directories per window (OOS windows)
DIRS_2022_2023 = ["stored_data_2022", "stored_data_2023"]
DIRS_2026_OOS = ["stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data_oos"]

def _run_single_g_l_trial(trial_number, params, window_label, dates, picks_by_date):
    """Run a backtest for one trial on one window - G+L only"""
    import test_green_candle_combined as tgc
    from test_full import MARGIN_THRESHOLD
    from optimize_combined import (
        set_strategy_params,
        _build_param_snapshot,
        _param_lock,
    )
    import json

    # Configure simulator (same as training)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    tgc.NEWS_FILTER_ENABLED = False

    # Load complete baseline config
    base_config_path = "config/trial_gl_trail_final_best.json"
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            base_data = json.load(f)
        p = dict(base_data.get("params", {}))
    else:
        base_config_path = "config/trial_g511_l626_v3_full_no_trail_best.json"
        if os.path.exists(base_config_path):
            with open(base_config_path) as f:
                base_data = json.load(f)
            p = dict(base_data.get("params", {}))
        else:
            p = {}

    # Apply trial params - ONLY G and L strategies (ignore others)
    # Merge trial params directly (should already only contain G+L)
    p.update(params)

    # Explicitly enable G and L
    p["g_enabled"] = True
    p["l_enabled"] = True

    # Safety: disable ALL other strategies just in case
    for prefix in "hafdvmrpwobkcsexijn":
        p[f"{prefix}_enabled"] = False

    with _param_lock:
        set_strategy_params(p)
        snapshot = _build_param_snapshot()

    # Run backtest
    cash = float(STARTING_CASH)
    all_trades = []

    for d in dates:
        picks = picks_by_date.get(d, [])
        if not picks:
            continue
        cash_account = cash < MARGIN_THRESHOLD
        try:
            states, cash, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account, params=snapshot
            )
        except Exception:
            continue
        effective_cash = cash + (unsettled if cash_account else 0)
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0)
                if pnl is not None:
                    all_trades.append({"pnl": pnl})
        cash = effective_cash

    n = len(all_trades)
    if n == 0:
        return {
            "trial": trial_number,
            "window": window_label,
            "n": 0,
            "pnl": 0.0,
            "pf": 0.0,
            "wr": 0.0,
        }

    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in all_trades if t["pnl"] <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss
    wr = len(wins) / n * 100

    return {
        "trial": trial_number,
        "window": window_label,
        "n": n,
        "pnl": round(total_pnl, 2),
        "pf": round(pf, 3),
        "wr": round(wr, 1),
    }

def worker_g_l(args):
    trial_number, params, window_label, dates, picks_by_date = args
    try:
        return _run_single_g_l_trial(trial_number, params, window_label, dates, picks_by_date)
    except Exception as e:
        print(f"Error in trial {trial_number}: {e}")
        return {
            "trial": trial_number,
            "window": window_label,
            "n": -1,
            "pnl": 0.0,
            "pf": 0.0,
            "wr": 0.0,
            "error": str(e),
        }

def load_window_data(dirs, date_lo, date_hi):
    """Load data for a time window"""
    all_dirs = [d for d in dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(all_dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])
    return dates, picks_by_date

def main():
    parser = argparse.ArgumentParser(description="Flexible G+L wide forward test")

    # Study selection (THE KEY FEATURE)
    parser.add_argument("--study", default="gl_trail_v3",
                       help="Study name to run wide forward test on (gl_trail_v3, gl_trail_v3_clean, etc.)")

    parser.add_argument("--storage-url", default="postgresql://postgres@127.0.0.1:5432/optuna_gl_trail",
                       help="Optuna storage URL")

    # Trial selection (FLEXIBLE TRIAL CHOICE)
    parser.add_argument("--trials", type=str, default="538",
                       help="Comma-separated trial numbers (e.g., '538,591,592' or 'all' for all completed)")

    parser.add_argument("--top", type=int, default=0,
                       help="Run top N trials by training score (0 = use --trials list)")

    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker processes")

    # Window selection
    parser.add_argument("--windows", type=str, default="2026",
                       help="Windows to test: '2022', '2023', '2026', or comma-separated (e.g., '2022,2026')")

    args = parser.parse_args()

    print(f"Wide Forward Test: G+L Only (FLEXIBLE VERSION)")
    print(f"Study: {args.study}")
    print(f"Storage: {args.storage_url}")
    print(f"Trials: {args.trials}")
    print(f"Workers: {args.workers}")
    print()

    # Load study
    print("Loading study...", flush=True)
    study = optuna.load_study(study_name=args.study, storage=args.storage_url)
    completed = [
        t for t in study.trials
        if t.state.name == "COMPLETE" and t.value and t.value > 0
    ]
    print(f"  {len(completed)} completed trials with positive score")

    # Select trials based on user choice
    selected_trials = []

    if args.top > 0:
        # Top N by training score
        selected = sorted(completed, key=lambda t: t.value, reverse=True)[:args.top]
        selected_trials = selected
        print(f"  Selected top {args.top} trials")

    elif args.trials.lower() == "all":
        # All trials
        selected_trials = completed
        print(f"  Selected all {len(completed)} trials")

    else:
        # Specific trial numbers
        trial_numbers = [int(x.strip()) for x in args.trials.split(",")]
        for trial in completed:
            if trial.number in trial_numbers:
                selected_trials.append(trial)
        print(f"  Selected specific trials: {[t.number for t in selected_trials]}")

    if not selected_trials:
        print("ERROR: No trials selected!")
        return

    print(f"  Will run {len(selected_trials)} trials")

    # Define windows to test (OOS windows)
    windows_config = {
        "2022": ("2022-01-01", "2022-12-31", DIRS_2022_2023),
        "2023": ("2023-01-01", "2023-12-31", DIRS_2022_2023),
        "2026": ("2026-03-01", "2099-12-31", DIRS_2026_OOS),
    }

    # Parse windows
    if "," in args.windows:
        windows_to_test = [w.strip() for w in args.windows.split(",")]
    else:
        windows_to_test = [args.windows.strip()]

    available_windows = [w for w in windows_to_test if w in windows_config]
    if not available_windows:
        print(f"ERROR: No valid windows found. Available: {list(windows_config.keys())}")
        return

    print(f"Testing windows: {available_windows}")

    # Load data for each window
    print("Loading data windows...", flush=True)
    window_data = {}
    for wname in available_windows:
        date_lo, date_hi, dirs = windows_config[wname]
        dates, picks_by_date = load_window_data(dirs, date_lo, date_hi)
        window_data[wname] = (dates, picks_by_date)
        print(f"  {wname}: {len(dates)} trading days")

    # Build work list
    jobs = []
    for trial in selected_trials:
        for wname, (dates, picks_by_date) in window_data.items():
            jobs.append((trial.number, dict(trial.params), wname, dates, picks_by_date))

    print(f"\nRunning {len(jobs)} jobs on {args.workers} workers...")
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker_g_l, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            results.append(r)
            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                eta = (elapsed / done) * (len(jobs) - done) if done > 0 else 0
                print(f"  {done}/{len(jobs)} done  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")

    # Write results
    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = ["trial", "window", "n", "pnl", "pf", "wr"]
        # Remove 'error' field from results if present (only needed for debugging)
        clean_results = []
        for r in results:
            clean_r = {k: v for k, v in r.items() if k != "error"}
            clean_results.append(clean_r)

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(clean_results, key=lambda x: (x["trial"], x["window"])):
            w.writerow(r)

    print(f"\nResults saved to {OUT_CSV}", flush=True)

    # Summary
    print(f"\nSummary: Ran {len(jobs)} total jobs across {len(available_windows)} windows")

    # Stats by trial
    from collections import defaultdict
    by_trial = defaultdict(dict)
    for r in results:
        by_trial[r["trial"]][r["window"]] = r

    print("\nTop performing trials by total OOS PnL:")
    for trial_num, windows_data in by_trial.items():
        total_pnl = sum(windows_data[w].get("pnl", 0) for w in available_windows if w in windows_data)
        total_trades = sum(windows_data[w].get("n", 0) for w in available_windows if w in windows_data)
        print(f"  Trial {trial_num}: ${total_pnl:,.0f} PnL from {total_trades} trades")

    print(f"\nTotal runtime: {time.time() - t0:.0f}s")

if __name__ == "__main__":
    import json
    main()