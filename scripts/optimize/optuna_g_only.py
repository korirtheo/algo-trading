"""
Optuna G-Only Optimization — Multi-Process Workers
====================================================
Optimizes 8 G strategy params (no V3 overlay) using ASK/TELL
with PostgreSQL so N workers can run in separate processes.

Usage:
  python scripts/optimize/optuna_g_only.py                    # main process
  python scripts/optimize/optuna_g_only.py --worker           # additional workers

Architecture
  Single-phase: each worker loads cached picks, runs G backtest.
  No V3 overlay, no candidate pre-computation.

  8 dimensions.
  n_startup_trials = 120  (~15x dims).
  Score floor: G trades >= 50, total_pnl > 0, pf >= 0.5.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
import signal

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, ".")

# ── Config ──────────────────────────────────────────────────────────────────
STARTING_CASH = 25_000
BASELINE_PATH = "config/trial_g511_l626_v3_overlay.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

DATA_DIRS = [
    "stored_data_2022", "stored_data_2023",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data", "stored_data_mar_may_2026",
    "stored_data_jun_2026",
]
DATE_LO = "2024-01-01"
DATE_HI = "2026-05-31"

# ── Lazy imports ────────────────────────────────────────────────────────────
import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock


def configure_simulator():
    """Set simulator flags (must be called before any simulation)."""
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False


def disable_adaptive_controls():
    """Lock all Phase 1A / adaptive controls to 0 (disabled)."""
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    for s in ALL_STRATS:
        setattr(tgc, f"{s.upper()}_PARTICIPATION_CAP", 0.0)
    tgc.NEWS_FILTER_ENABLED = False


def load_data():
    """Load picks and return (dates, picks_by_date)."""
    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
    print(f"  Training window: {DATE_LO} to {DATE_HI} ({len(dates)} days)", flush=True)
    print(f"  Data dirs: {dirs}", flush=True)
    if len(dates) == 0:
        print("ERROR: no trading days in range!", flush=True)
        sys.exit(1)
    return dates, picks_by_date


def run_g_backtest(dates, picks_by_date, g_params, snapshot=None):
    """Run G-only backtest. Returns detailed metrics dict."""
    cash = float(STARTING_CASH)
    all_pnls = []

    for d in dates:
        cash += 0  # unsettled handled within simulate_day
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
        # unsettled is already returned by simulate_day_combined — but the
        # function does NOT add it to cash (it returns both). We need to track
        # the "total cash + unsettled" ourselves.
        effective_cash = cash + (unsettled if cash_account else 0)
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0)
                if pnl is not None:
                    all_pnls.append(pnl)
        cash = effective_cash

    n = len(all_pnls)
    if n == 0:
        return {"n": 0, "total_pnl": -9999, "pf": 0.0, "equity": cash,
                "gross_win": 0, "gross_loss": 1e-9, "wr": 0.0}

    total_pnl = sum(all_pnls)
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    wr = len(wins) / n * 100 if n > 0 else 0.0

    return {
        "n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr,
        "equity": cash, "gross_win": gross_win, "gross_loss": gross_loss,
    }


# ── Optuna param spaces ─────────────────────────────────────────────────────

def suggest_g_params(trial):
    """8 G dims with widened search covering the #511 regime."""
    return {
        "g_min_gap_pct": trial.suggest_float("g_min_gap_pct", 15.0, 80.0, step=5.0),
        "g_require_2nd_green": trial.suggest_categorical("g_require_2nd_green", [True, False]),
        "g_require_2nd_new_high": trial.suggest_categorical("g_require_2nd_new_high", [True, False]),
        "g_target_pct": trial.suggest_float("g_target_pct", 4.0, 70.0, step=1.0),
        "g_time_limit_min": trial.suggest_int("g_time_limit_min", 3, 30),
        "g_stop_pct": trial.suggest_float("g_stop_pct", 0.0, 30.0, step=1.0),
        "g_trail_pct": trial.suggest_float("g_trail_pct", 0.0, 5.0, step=0.5),
        "g_trail_activate_pct": trial.suggest_float("g_trail_activate_pct", 0.0, 8.0, step=1.0),
    }


# ── Objective ───────────────────────────────────────────────────────────────

def objective(trial, data):
    """Optuna objective: suggest G params, run backtest."""
    dates = data["dates"]
    picks_by_date = data["picks_by_date"]
    bl_params = data["bl_params"]

    g_params = suggest_g_params(trial)

    # Build full param dict: baseline + only G enabled + trial overrides
    p = dict(bl_params)
    for s in ALL_STRATS:
        p[f"enable_{s}"] = (s == "g")
    p["enable_x"] = False
    p.update(g_params)

    with _param_lock:
        _oc_set_params(p)
        disable_adaptive_controls()
        snapshot = _build_param_snapshot()

    result = run_g_backtest(dates, picks_by_date, g_params, snapshot=snapshot)

    if result["n"] < 50 or result["total_pnl"] <= 0 or result["pf"] < 0.5:
        return -9999

    score = result["total_pnl"] * min(result["pf"], 3.0)
    if math.isnan(score) or math.isinf(score):
        return -9999
    score = max(-9.9e12, min(9.9e12, float(score)))

    trial.set_user_attr("g_n", result["n"])
    trial.set_user_attr("g_pnl", round(result["total_pnl"], 2))
    trial.set_user_attr("g_pf", round(result["pf"], 3))
    trial.set_user_attr("g_wr", round(result["wr"], 1))

    return score


# ── Worker loop ─────────────────────────────────────────────────────────────

def worker_loop(data, args):
    """Connect to study and run ask → objective → tell."""
    storage = optuna.storages.RDBStorage(url=args.db)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=args.startup_trials),
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    print(f"\nStudy state: {completed} complete, {running} running, {failed} failed "
          f"(target: {args.trials})", flush=True)

    if completed >= args.trials:
        print(f"Target {args.trials} trials already complete — nothing to do.", flush=True)
        return

    t_start = time.time()
    my_count = 0
    last_report = 0
    exiting = False

    def _handle_sigint(sig, frame):
        nonlocal exiting
        exiting = True
        print(f"\n[worker] SIGINT received — exiting after {my_count} trials...", flush=True)

    if os.name != "nt":
        signal.signal(signal.SIGINT, _handle_sigint)

    while not exiting:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if completed >= args.trials:
            break

        try:
            trial = study.ask()
        except Exception as e:
            print(f"  ask() error: {e} — retrying in 5s", flush=True)
            time.sleep(5)
            continue

        try:
            value = objective(trial, data)
            study.tell(trial, value)
            my_count += 1
        except KeyboardInterrupt:
            try:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            except Exception:
                pass
            print(f"\n[worker] KeyboardInterrupt after {my_count} trials", flush=True)
            break
        except Exception as e:
            try:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            except Exception:
                pass
            continue

        if my_count % 20 == 0 or my_count == last_report + 1:
            completed_now = len([t for t in study.trials
                                 if t.state == optuna.trial.TrialState.COMPLETE])
            elapsed = time.time() - t_start
            try:
                bt = study.best_trial
                print(f"  [{completed_now}/{args.trials}] {elapsed/60:.1f}m | "
                      f"best ${bt.value:,.0f} (#{bt.number}) "
                      f"G=${bt.user_attrs.get('g_pnl',0):,.0f} "
                      f"PF={bt.user_attrs.get('g_pf',0):.2f} "
                      f"WR={bt.user_attrs.get('g_wr',0):.1f}%",
                      flush=True)
            except Exception:
                print(f"  [{completed_now}/{args.trials}] {elapsed/60:.1f}m | "
                      f"no completed trials yet", flush=True)
            last_report = my_count

    total_elapsed = time.time() - t_start
    print(f"\n[worker] Done: {my_count} trials in {total_elapsed/60:.1f}m", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G-only Optuna optimization (multi-process)")
    parser.add_argument("--trials", type=int, default=600, help="Total trials across all workers")
    parser.add_argument("--startup-trials", type=int, default=120,
                        help="TPE n_startup_trials")
    parser.add_argument("--db", default="postgresql://postgres@127.0.0.1:5432/optuna_g_only",
                        help="PostgreSQL storage URL")
    parser.add_argument("--study", default="g_only",
                        help="Optuna study name")
    parser.add_argument("--params-out", default="config/trial_g_only_best.json",
                        help="Best params output path")
    parser.add_argument("--worker", action="store_true",
                        help="Run as worker only (skips data loading)")
    parser.add_argument("--dump-best", action="store_true",
                        help="Dump best params from existing study and exit")
    args = parser.parse_args()

    # ── Dump best mode ──────────────────────────────────
    if args.dump_best:
        use_pg = args.db.startswith("postgresql://") or args.db.startswith("postgres://")
        storage = optuna.storages.RDBStorage(url=args.db) if use_pg else f"sqlite:///{args.db}"
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study,
            storage=storage,
            load_if_exists=True,
        )
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
            print("No completed trials in study.", flush=True)
            sys.exit(1)
        dump_best(study, args)
        return

    # ── Setup: configure simulator + load data ──────────
    configure_simulator()
    print("="*60, flush=True)
    print("G-Only Optuna Optimization", flush=True)
    print("="*60, flush=True)

    if args.worker:
        # Worker mode: load picks only (already configured by main process)
        pass
    else:
        print("Loading data...", flush=True)

    dates, picks_by_date = load_data()

    # Load baseline config
    with open(BASELINE_PATH) as f:
        baseline_full = json.load(f)
    bl_params = dict(baseline_full["params"])

    data = {
        "dates": dates,
        "picks_by_date": picks_by_date,
        "bl_params": bl_params,
    }

    # ── Phase 2: Worker loop ───────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 2: Worker Loop", flush=True)
    print(f"  Study: {args.study}", flush=True)
    print(f"  DB:    {args.db}", flush=True)
    print(f"  Dims:  8 G", flush=True)
    print(f"  n_startup_trials: {args.startup_trials} ({args.startup_trials/8:.0f}x dims)", flush=True)
    print(f"  Target completed trials: {args.trials}", flush=True)
    print(f"{'='*60}", flush=True)

    worker_loop(data, args)

    # ── Report best ─────────────────────────────────────
    try:
        use_pg = args.db.startswith("postgresql://") or args.db.startswith("postgres://")
        storage = optuna.storages.RDBStorage(url=args.db) if use_pg else f"sqlite:///{args.db}"
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study,
            storage=storage,
            load_if_exists=True,
        )
        dump_best(study, args)
    except Exception as e:
        print(f"\nCould not report best: {e}", flush=True)


def dump_best(study, args):
    """Print best trial and save to file."""
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
        print("No completed trials.", flush=True)
        return

    best = study.best_trial
    ua = best.user_attrs

    print(f"\n{'='*60}", flush=True)
    print(f"BEST TRIAL #{best.number}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Score:        ${best.value:,.0f}", flush=True)
    for k, v in sorted(best.params.items()):
        print(f"  {k} = {v}", flush=True)
    print(f"  G:   n={ua.get('g_n')},  PnL=${ua.get('g_pnl',0):,.0f},  "
          f"PF={ua.get('g_pf',0):.2f},  WR={ua.get('g_wr',0):.1f}%", flush=True)

    best_params = {
        "label": f"G-only optuna #{best.number}",
        "params": dict(best.params),
        "source": f"optuna_g_only.py trial #{best.number}",
        "score": best.value,
        "g_pnl": ua.get("g_pnl", 0),
        "g_pf": ua.get("g_pf", 0),
        "g_n": ua.get("g_n", 0),
        "g_wr": ua.get("g_wr", 0),
    }
    with open(args.params_out, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {args.params_out}", flush=True)


if __name__ == "__main__":
    main()
