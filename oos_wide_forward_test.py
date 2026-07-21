"""
Wide forward test: top-50 + 50-random trials from g511_l626_v3_full_no_trail
on three OOS windows: 2022, 2023, 2026-Mar+

Usage:
    python oos_wide_forward_test.py [--workers N]
"""
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

STORAGE_URL = "postgresql://postgres@127.0.0.1:5432/optuna_g511_l626_v3_full_no_trail"
STUDY_NAME = "g511_l626_v3_full_no_trail"
STARTING_CASH = 25_000
OUT_CSV = "oos_wide_forward_test_results.csv"

# Data directories per window
DIRS_2022 = ["stored_data_2022"]
DIRS_2023 = ["stored_data_2023"]
DIRS_2026 = ["stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data_oos"]


def _run_single(trial_number, params, window_label, dates, picks_by_date):
    """Run a backtest for one trial on one window. Must be importable at top level."""
    import test_green_candle_combined as tgc
    from test_full import MARGIN_THRESHOLD
    from optimize_combined import set_strategy_params, _build_param_snapshot, _param_lock

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

    # Load complete baseline param dict from a known-good deployed config
    # Then override only G/L/V params from the trial
    import json
    base_config_path = "config/trial_w21b_511_deploy.json"
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
    p.update(params)

    # Force trailing off (ensure all strategies have 0 trail)
    for prefix in ("g", "l", "v", "h", "a", "f", "d", "r", "w", "o", "b", "k", "c", "s", "e", "x", "i", "j", "n"):
        p[f"{prefix}_trail_pct"] = 0.0
        p[f"{prefix}_trail_activate_pct"] = 0.0

    # Disable unused strategies (only G, L, V enabled)
    for s in "hafdrwobkcsexijn":
        p[f"{s}_enabled"] = False

    # Explicitly enable G, L, V
    p["g_enabled"] = True
    p["l_enabled"] = True
    p["v_enabled"] = True

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
        return {"trial": trial_number, "window": window_label,
                "n": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0}

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
        return {"trial": trial_number, "window": window_label,
                "n": -1, "pnl": 0.0, "pf": 0.0, "wr": 0.0, "error": str(e)}


def load_window(dirs, date_lo, date_hi):
    from test_full import load_all_picks
    all_dirs = [d for d in dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(all_dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])
    return dates, picks_by_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top", type=int, default=50, help="Top N trials by training score")
    parser.add_argument("--random", type=int, default=50, help="Random trials to sample")
    args = parser.parse_args()

    print("Loading study...", flush=True)
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
    completed = [t for t in study.trials if t.state.name == "COMPLETE" and t.value and t.value > 0]
    print(f"  {len(completed)} completed trials with positive score", flush=True)

    # Top N by training score
    top_n = sorted(completed, key=lambda t: t.value, reverse=True)[:args.top]
    top_ids = {t.number for t in top_n}

    # Random sample from the rest
    rest = [t for t in completed if t.number not in top_ids]
    random.seed(42)
    rand_n = random.sample(rest, min(args.random, len(rest)))

    selected = top_n + rand_n
    print(f"  Selected: {len(top_n)} top + {len(rand_n)} random = {len(selected)} trials", flush=True)

    # Load data windows
    print("Loading data windows...", flush=True)
    windows = {
        "2022": load_window(DIRS_2022, "2022-01-01", "2022-12-31"),
        "2023": load_window(DIRS_2023, "2023-01-01", "2023-12-31"),
        "2026_oos": load_window(DIRS_2026, "2026-03-01", "2099-12-31"),
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
                print(f"  {done}/{len(jobs)} done  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

    # Write CSV
    fieldnames = ["trial", "window", "n", "pnl", "pf", "wr"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["trial"], x["window"])):
            w.writerow(r)
    print(f"\nResults saved to {OUT_CSV}", flush=True)

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

    WINDOWS = ["2022", "2023", "2026_oos"]

    # Print top 20 by 2026 OOS PnL
    print("\n" + "="*110)
    print(f"{'#':>5} {'src':>4}  {'train_score':>12} {'train_pnl':>10} {'train_pf':>8} "
          f"{'2022_pnl':>10} {'22_pf':>6} {'22_n':>5} "
          f"{'2023_pnl':>10} {'23_pf':>6} {'23_n':>5} "
          f"{'2026_pnl':>10} {'26_pf':>6} {'26_n':>5}")
    print("-"*110)

    def row_summary(tnum):
        r22 = by_trial[tnum].get("2022", {})
        r23 = by_trial[tnum].get("2023", {})
        r26 = by_trial[tnum].get("2026_oos", {})
        src = "TOP" if tnum in top_set else "RND"
        ts = train_score.get(tnum, 0)
        tp = train_pnl.get(tnum, 0)
        tf = train_pf.get(tnum, 0)
        return (tnum, src, ts, tp, tf,
                r22.get("pnl", 0), r22.get("pf", 0), r22.get("n", 0),
                r23.get("pnl", 0), r23.get("pf", 0), r23.get("n", 0),
                r26.get("pnl", 0), r26.get("pf", 0), r26.get("n", 0))

    all_rows = [row_summary(t.number) for t in selected]

    # Sort by 2026 OOS PnL descending
    all_rows.sort(key=lambda x: x[11], reverse=True)

    for row in all_rows[:30]:
        tnum, src, ts, tp, tf, p22, f22, n22, p23, f23, n23, p26, f26, n26 = row
        print(f"{tnum:>5} {src:>4}  {ts:>12,.0f} {tp:>10,.0f} {tf:>8.3f} "
              f"{p22:>10,.0f} {f22:>6.2f} {n22:>5} "
              f"{p23:>10,.0f} {f23:>6.2f} {n23:>5} "
              f"{p26:>10,.0f} {f26:>6.2f} {n26:>5}")

    print("\n--- Consistent top 10 (sum of all 3 OOS windows, trials with n>0 on all windows) ---")
    consistent = [(r, r[5]+r[8]+r[11]) for r in all_rows
                  if r[7] > 0 and r[10] > 0 and r[13] > 0]
    consistent.sort(key=lambda x: x[1], reverse=True)
    for row, total_oos in consistent[:10]:
        tnum, src, ts, tp, tf, p22, f22, n22, p23, f23, n23, p26, f26, n26 = row
        print(f"  #{tnum:3d} {src}  train={ts:>12,.0f}  OOS_total={total_oos:>10,.0f}  "
              f"2022={p22:>8,.0f}({f22:.2f}pf)  2023={p23:>8,.0f}({f23:.2f}pf)  "
              f"2026={p26:>8,.0f}({f26:.2f}pf)")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
