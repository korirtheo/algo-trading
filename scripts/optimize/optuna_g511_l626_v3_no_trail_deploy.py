"""
G511+L626+V3 No-Trail Variant — Optuna Study
==============================================
Optimizes deployed live config (G#511+L#626+V3#576 overlay) but with ALL trailing stops disabled.
Uses same entry logic and other exit params; only difference is all *_trail_pct=0, all *_trail_activate_pct=0.

Compare against trial_g511_l626_v3_overlay.json to measure impact of trailing.

Training window: 2024-01-01 to 2026-05-31
Blind OOS: 2026-06-01 onwards

Objective: total_pnl x min(pf, 3.0)
"""

import argparse
import json
import math
import os
import sys
import time

import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, ".")

# ── Config ──────────────────────────────────────────────────────────────────
STARTING_CASH = 25_000
DEPLOY_CONFIG_PATH = "config/trial_g511_l626_v3_overlay.json"
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
DATE_HI = "2026-02-28"  # Same as W21b #511 training window (Mar-Jun 2026 is blind OOS)

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


def run_backtest(dates, picks_by_date, params, snapshot=None):
    """Run combined backtest (G+L enabled, others disabled). Returns detailed metrics dict."""
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
                    all_trades.append({
                        "strategy": st.get("strategy"),
                        "pnl": pnl,
                        "position_cost": st.get("position_cost", 0),
                        "exit_reason": st.get("exit_reason"),
                    })
        cash = effective_cash

    n = len(all_trades)
    if n == 0:
        return {"n": 0, "total_pnl": -9999, "pf": 0.0, "equity": cash,
                "gross_win": 0, "gross_loss": 1e-9, "wr": 0.0}

    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in all_trades if t["pnl"] <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    wr = len(wins) / n * 100 if n > 0 else 0.0

    return {
        "n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr,
        "equity": cash, "gross_win": gross_win, "gross_loss": gross_loss,
    }


def objective(trial, data):
    """Optuna objective: use W21b #511 params but with all trails disabled."""
    dates = data["dates"]
    picks_by_date = data["picks_by_date"]
    deploy_params = data["deploy_params"]

    # Start with deployed config
    p = dict(deploy_params)

    # Force all trailing to 0
    for s in ALL_STRATS:
        p[f"{s}_trail_pct"] = 0.0
        p[f"{s}_trail_activate_pct"] = 0.0

    with _param_lock:
        _oc_set_params(p)
        disable_adaptive_controls()
        snapshot = _build_param_snapshot()

    result = run_backtest(dates, picks_by_date, p, snapshot=snapshot)

    if result["n"] < 100 or result["total_pnl"] <= 0 or result["pf"] < 0.5:
        return -9999

    score = result["total_pnl"] * min(result["pf"], 3.0)
    if math.isnan(score) or math.isinf(score):
        return -9999
    score = max(-9.9e12, min(9.9e12, float(score)))

    trial.set_user_attr("total_pnl", round(result["total_pnl"], 2))
    trial.set_user_attr("pf", round(result["pf"], 3))
    trial.set_user_attr("wr", round(result["wr"], 1))
    trial.set_user_attr("n", result["n"])
    trial.set_user_attr("equity", round(result["equity"], 0))

    return score


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

    while True:
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
                      f"best ${bt.value:,.0f} (#{bt.number}) | "
                      f"PnL=${bt.user_attrs.get('total_pnl',0):,.0f} "
                      f"PF={bt.user_attrs.get('pf',0):.2f} "
                      f"WR={bt.user_attrs.get('wr',0):.1f}%",
                      flush=True)
            except Exception:
                print(f"  [{completed_now}/{args.trials}] {elapsed/60:.1f}m | "
                      f"no completed trials yet", flush=True)
            last_report = my_count

    total_elapsed = time.time() - t_start
    print(f"\n[worker] Done: {my_count} trials in {total_elapsed/60:.1f}m", flush=True)


def dump_best(study, args):
    """Print best trial and save to file."""
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
        print("No completed trials.", flush=True)
        return

    best = study.best_trial
    ua = best.user_attrs

    print(f"\n{'='*60}", flush=True)
    print(f"BEST TRIAL #{best.number} (NO TRAIL)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Score:        ${best.value:,.0f}", flush=True)
    print(f"  PnL:          ${ua.get('total_pnl',0):,.0f}", flush=True)
    print(f"  PF:           {ua.get('pf',0):.3f}", flush=True)
    print(f"  WR:           {ua.get('wr',0):.1f}%", flush=True)
    print(f"  n:            {ua.get('n')}", flush=True)
    print(f"  Equity:       ${ua.get('equity',0):,.0f}", flush=True)

    best_params = {
        "label": f"G511+L626+V3 no-trail variant (optuna #{best.number})",
        "params": dict(best.params),
        "source": f"optuna_g511_l626_v3_no_trail.py trial #{best.number}",
        "score": best.value,
        "total_pnl": ua.get("total_pnl", 0),
        "pf": ua.get("pf", 0),
        "wr": ua.get("wr", 0),
        "n": ua.get("n", 0),
    }
    with open(args.params_out, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {args.params_out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="G511+L626+V3 no-trail Optuna optimization")
    parser.add_argument("--trials", type=int, default=600,
                        help="Total trials")
    parser.add_argument("--startup-trials", type=int, default=200,
                        help="TPE n_startup_trials")
    parser.add_argument("--db", default="postgresql://postgres:password@127.0.0.1:5432/optuna_g511_l626_v3_no_trail",
                        help="PostgreSQL storage URL")
    parser.add_argument("--study", default="g511_l626_v3_no_trail",
                        help="Optuna study name")
    parser.add_argument("--params-out", default="config/trial_g511_l626_v3_no_trail_best.json",
                        help="Best params output path")
    parser.add_argument("--worker", action="store_true",
                        help="Run as worker (skip data loading)")
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

    # ── Setup ────────────────────────────────────────────
    configure_simulator()
    print("="*60, flush=True)
    print("G511+L626+V3 No-Trail Optuna Study", flush=True)
    print("="*60, flush=True)

    print("Loading data...", flush=True)
    dates, picks_by_date = load_data()

    # Load deployed config
    with open(DEPLOY_CONFIG_PATH) as f:
        deploy_config = json.load(f)
    deploy_params = dict(deploy_config["params"])

    data = {
        "dates": dates,
        "picks_by_date": picks_by_date,
        "deploy_params": deploy_params,
    }

    # ── Phase: Worker loop ───────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Running Optuna Optimization (All Trails Disabled)", flush=True)
    print(f"  Study: {args.study}", flush=True)
    print(f"  DB:    {args.db}", flush=True)
    print(f"  Target trials: {args.trials}", flush=True)
    print(f"  Baseline: G511+L626+V3#576 overlay (live deployed)", flush=True)
    print(f"  Change: All *_trail_pct = 0, all *_trail_activate_pct = 0", flush=True)
    print(f"{'='*60}\n", flush=True)

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


if __name__ == "__main__":
    main()
