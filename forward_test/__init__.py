import argparse
import csv
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, ".")

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _run_single(trial_number, params, window_label, dates, picks_by_date):
    """Run a backtest for one trial on one window. Must be importable at top level."""
    from backtest.engine import simulate_day_combined
    from test_full import MARGIN_THRESHOLD
    from optimize_combined import (
        set_strategy_params,
        _build_param_snapshot,
        _param_lock,
    )

    # Configure simulator (same as training)
    USE_DYNAMIC_SLIPPAGE = True
    USE_MULTIWINDOW_SLIPPAGE = True
    USE_VOLATILITY_ADJUSTMENT = True
    SLIP_IMPACT_K = 3.0
    VOL_CAP_PCT = 5.0
    MAX_2MIN_PARTICIPATION = 0.15
    MAX_REGIME_PARTICIPATION = 0.08
    NEWS_MODULATOR_ENABLED = False
    MIN_PRICE = 0.0
    MAX_MODELED_SLIP_BP = 0.0
    MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    MIN_ATR_PCT = 0.0
    MIN_FAVORABILITY_THRESHOLD = 0.0
    NEWS_FILTER_ENABLED = False

    # Load complete baseline param dict from a known-good deployed config
    # Then override only G/L/V params from the trial
    import json

    base_config_path = "config/trial_gl_trail_final_best.json"
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            base_data = json.load(f)
        p = dict(base_data.get("params", {}))
    else:
        # Fallback: try the new best config if it exists and is complete
        base_config_path = "config/trial_g511_l626_v3_full_no_trail_best.json"
        if os.path.exists(base_config_path):
            with open(base_config_path) as f:
                base_data = json.load(f)
            p = dict(base_data.get("params", {}))
        else:
            p = {}

    # Apply trial params (G, L, V only — these override)
    # Set default values for all possible parameters
    for s in "hgafdvmrpwobkcsexijnl":
        p.setdefault(f"{s}_enabled", False)
        p.setdefault(f"{s}_min_gap_pct", 0.0)
        p.setdefault(f"{s}_min_body_pct", 0.0)
        p.setdefault(f"{s}_require_2nd_green", False)
        p.setdefault(f"{s}_require_2nd_new_high", False)
        p.setdefault(f"{s}_require_vol_confirm", False)
        p.setdefault(f"{s}_target_pct", 0.0)
        p.setdefault(f"{s}_target2_pct", 0.0)
        p.setdefault(f"{s}_partial_sell_pct", 0.0)
        p.setdefault(f"{s}_time_limit_minutes", 0)
        p.setdefault(f"{s}_stop_pct", 0.0)
        p.setdefault(f"{s}_trail_pct", 0.0)
        p.setdefault(f"{s}_trail_activate_pct", 0.0)

    p.update(params)

    # Disable all strategies by default
    for prefix in "hgafdvmrpwobkcsexijnl":
        p[f"{prefix}_enabled"] = False

    # Explicitly enable G, L
    p["g_enabled"] = True
    p["l_enabled"] = True

    with _param_lock:
        set_strategy_params(p)
        snapshot = _build_param_snapshot()
        print(f"Snapshot for trial {trial_number}: {snapshot}")

    # Run backtest
    cash = float(25000)
    all_trades = []

    for d in dates:
        picks = picks_by_date.get(d, [])
        print(f"Picks for {d}: {picks}")
        if not picks:
            continue
        cash_account = cash < 100000
        try:
            states, cash, unsettled, _ = simulate_day_combined(
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


def worker(args):
    trial_number, params, window_label, dates, picks_by_date = args
    try:
        return _run_single(trial_number, params, window_label, dates, picks_by_date)
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


def load_window(dirs, date_lo, date_hi):
    from test_full import load_all_picks

    all_dirs = [d for d in dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(all_dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])
    return dates, picks_by_date


def main(config, args):
    study_name = config["study"]
    storage = "postgresql://postgres@127.0.0.1:5432/optuna_gl_trail"

    print("Loading study...", flush=True)
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [
        t
        for t in study.trials
        if t.state.name == "COMPLETE" and t.value and t.value > 0
    ]
    print(f"  {len(completed)} completed trials with positive score", flush=True)

    # Top N by training score
    top_n = sorted(completed, key=lambda t: t.value, reverse=True)[:50]
    top_ids = {t.number for t in top_n}

    # Random sample from the rest
    rest = [t for t in completed if t.number not in top_ids]
    random.seed(42)
    rand_n = random.sample(rest, min(50, len(rest)))

    selected = top_n + rand_n
    print(
        f"  Selected: {len(top_n)} top + {len(rand_n)} random = {len(selected)} trials",
        flush=True,
    )

    # Load data windows
    print("Loading data windows...", flush=True)
    windows = {
        "2026_oos": load_window(
            config["data_dirs"]["2026"], "2026-03-01", "2099-12-31"
        ),
    }
    for wname, (dates, _) in windows.items():
        print(f"  {wname}: {len(dates)} trading days", flush=True)

    # Build work list
    jobs = []
    for trial in selected:
        for wname, (dates, picks_by_date) in windows.items():
            jobs.append((trial.number, dict(trial.params), wname, dates, picks_by_date))

    print(f"\nRunning {len(jobs)} jobs on {args.workers} workers...", flush=True)
    t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            results.append(r)
            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                eta = (elapsed / done) * (len(jobs) - done)
                print(
                    f"  {done}/{len(jobs)} done  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
                    flush=True,
                )

    # Write CSV
    output_filename = f"forward_test_results_{study_name}_top50_rand50.csv"
    fieldnames = ["trial", "window", "n", "pnl", "pf", "wr"]
    with open(output_filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["trial"], x["window"])):
            w.writerow(r)
    print(f"\nResults saved to {output_filename}", flush=True)

    # ---- Summary table ----
    # Pivot: trial -> {window: result}
    from collections import defaultdict

    by_trial = defaultdict(dict)
    for r in results:
        by_trial[r["trial"]][r["window"]] = r

    # Also get training scores
    train_score = {t.number: t.value for t in selected}
    train_pnl = {t.number: t.user_attrs.get("total_pnl", 0) for t in selected}
    train_pf = {t.number: t.user_attrs.get("pf", 0) for t in selected}
    train_n = {t.number: t.user_attrs.get("n", 0) for t in selected}
    top_set = {t.number for t in top_n}

    # Print top 20 by 2026 OOS PnL
    print("\n" + "=" * 110)
    print(
        f"{'#':>5} {'src':>4}  {'train_score':>12} {'train_pnl':>10} {'train_pf':>8} "
        f"{'2026_pnl':>10} {'26_pf':>6} {'26_n':>5}"
    )
    print("-" * 110)

    def row_summary(tnum):
        r26 = by_trial[tnum].get("2026_oos", {})
        src = "TOP" if tnum in top_set else "RND"
        ts = train_score.get(tnum, 0)
        tp = train_pnl.get(tnum, 0)
        tf = train_pf.get(tnum, 0)
        return (
            tnum,
            src,
            ts,
            tp,
            tf,
            r26.get("pnl", 0),
            r26.get("pf", 0),
            r26.get("n", 0),
        )

    all_rows = [row_summary(t.number) for t in selected]

    # Sort by 2026 OOS PnL descending
    all_rows.sort(key=lambda x: x[5], reverse=True)

    for row in all_rows[:30]:
        tnum, src, ts, tp, tf, p26, f26, n26 = row
        print(
            f"{tnum:>5} {src:>4}  {ts:>12,.0f} {tp:>10,.0f} {tf:>8.3f} "
            f"{p26:>10,.0f} {f26:>6.2f} {n26:>5}"
        )

    print(f"\nTotal runtime: {time.time() - t0:.0f}s")
