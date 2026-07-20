"""
G+L Trail Optimization Study
=============================
G strategy: tunable partial exit + optional trailing stop for remainder
L strategy: existing tiers + tunable trailing stop (on/off, range expanded)
V disabled (confirmed drag in 2026 OOS).

Exit architecture:
  G: sell g_partial_sell_pct% at g_target_pct, then either:
     (a) trail remainder with g_trail_pct/g_trail_activate_pct, OR
     (b) hold to g_target2_pct (flat exit)
     When g_partial_sell_pct=100, legacy behavior (sell all at target).

  L: existing tier system + l_trail_pct/l_trail_activate_pct (already in tgc)

Training: 2024-01-01 to 2026-02-28
Blind OOS: 2022, 2023, 2026-Mar+
Objective: total_pnl * min(pf, 3.0)
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

STARTING_CASH = 25_000
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
DATE_HI = "2026-02-28"

import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock


def configure_simulator():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False


def configure_for_optimization():
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    tgc.NEWS_FILTER_ENABLED = False


def load_data():
    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
    print(f"  Training window: {DATE_LO} to {DATE_HI} ({len(dates)} days)", flush=True)
    if len(dates) == 0:
        print("ERROR: no trading days in range!", flush=True)
        sys.exit(1)
    return dates, picks_by_date


def run_backtest(dates, picks_by_date, snapshot):
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
    pf = gross_win / gross_loss
    wr = len(wins) / n * 100
    return {"n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr,
            "equity": cash, "gross_win": gross_win, "gross_loss": gross_loss}


def objective(trial, data):
    """G+L with tunable partial exit and trailing stops."""
    dates = data["dates"]
    picks_by_date = data["picks_by_date"]

    # Start from deployed config so all required params are populated
    p = dict(data["base_params"])

    # ---- G strategy params ----
    p["g_min_gap_pct"] = trial.suggest_float("g_min_gap_pct", 5.0, 50.0, step=5.0)
    p["g_require_2nd_green"] = trial.suggest_categorical("g_require_2nd_green", [True, False])
    p["g_require_2nd_new_high"] = trial.suggest_categorical("g_require_2nd_new_high", [True, False])
    p["g_stop_pct"] = trial.suggest_float("g_stop_pct", 5.0, 30.0, step=1.0)
    p["g_time_limit_min"] = trial.suggest_int("g_time_limit_min", 6, 60, step=3)

    # Exit architecture: partial sell % (0=no partial, sell all at target1)
    g_partial = trial.suggest_float("g_partial_sell_pct", 0.0, 75.0, step=25.0)
    p["g_partial_sell_pct"] = g_partial
    p["g_target_pct"] = trial.suggest_float("g_target_pct", 5.0, 40.0, step=5.0)

    if g_partial > 0:
        # Runner exit: trail OR flat target2
        g_use_trail = trial.suggest_categorical("g_use_trail", [True, False])
        if g_use_trail:
            p["g_trail_activate_pct"] = trial.suggest_float("g_trail_activate_pct", 5.0, 40.0, step=5.0)
            p["g_trail_pct"] = trial.suggest_float("g_trail_pct", 3.0, 20.0, step=1.0)
            p["g_target2_pct"] = 999.0  # not used when trailing
        else:
            p["g_trail_activate_pct"] = 0.0
            p["g_trail_pct"] = 0.0
            p["g_target2_pct"] = trial.suggest_float("g_target2_pct", 20.0, 100.0, step=10.0)
    else:
        # Sell all at target1; optionally trail before target1 hits
        g_use_trail = trial.suggest_categorical("g_use_trail", [True, False])
        if g_use_trail:
            p["g_trail_activate_pct"] = trial.suggest_float("g_trail_activate_pct", 5.0, 40.0, step=5.0)
            p["g_trail_pct"] = trial.suggest_float("g_trail_pct", 3.0, 20.0, step=1.0)
        else:
            p["g_trail_activate_pct"] = 0.0
            p["g_trail_pct"] = 0.0
        p["g_target2_pct"] = 999.0  # n/a

    # ---- L strategy params ----
    p["l_earliest_candle"] = trial.suggest_int("l_earliest_candle", 3, 30, step=3)
    p["l_latest_candle"] = trial.suggest_int("l_latest_candle", 30, 180, step=15)
    p["l_max_float"] = trial.suggest_int("l_max_float", 5_000_000, 25_000_000, step=5_000_000)
    p["l_min_gap"] = trial.suggest_int("l_min_gap", 10, 80, step=5)
    p["l_min_price_accel_pct"] = trial.suggest_float("l_min_price_accel_pct", 0.5, 3.0, step=0.5)
    p["l_partial_sell_pct"] = trial.suggest_float("l_partial_sell_pct", 0.0, 75.0, step=25.0)
    p["l_stop_pct"] = trial.suggest_float("l_stop_pct", 10.0, 30.0, step=1.0)
    p["l_tier1_target1_pct"] = trial.suggest_float("l_tier1_target1_pct", 15.0, 50.0, step=5.0)
    p["l_tier1_target2_pct"] = trial.suggest_float("l_tier1_target2_pct", 20.0, 80.0, step=5.0)
    p["l_tier2_target1_pct"] = trial.suggest_float("l_tier2_target1_pct", 10.0, 30.0, step=2.0)
    p["l_tier2_target2_pct"] = trial.suggest_float("l_tier2_target2_pct", 20.0, 60.0, step=5.0)
    p["l_tier3_target1_pct"] = trial.suggest_float("l_tier3_target1_pct", 5.0, 25.0, step=2.0)
    p["l_tier3_target2_pct"] = trial.suggest_float("l_tier3_target2_pct", 10.0, 50.0, step=5.0)

    # L trail: tunable on/off + wider ranges
    l_use_trail = trial.suggest_categorical("l_use_trail", [True, False])
    if l_use_trail:
        p["l_trail_activate_pct"] = trial.suggest_float("l_trail_activate_pct", 1.0, 20.0, step=1.0)
        p["l_trail_pct"] = trial.suggest_float("l_trail_pct", 1.0, 15.0, step=1.0)
    else:
        p["l_trail_activate_pct"] = 9999.0  # never activates
        p["l_trail_pct"] = 1.0              # harmless when activate=9999

    # Force trailing off for all other strategies
    for prefix in "vhafdrwobkcsexijn":
        p[f"{prefix}_trail_pct"] = 0.0
        p[f"{prefix}_trail_activate_pct"] = 0.0

    # Only G and L enabled; disable everything else
    for s in "vhafdrwobkcsexijn":
        p[f"enable_{s}"] = False
    p["enable_g"] = True
    p["enable_l"] = True

    with _param_lock:
        _oc_set_params(p)
        configure_for_optimization()
        snapshot = _build_param_snapshot()

    result = run_backtest(dates, picks_by_date, snapshot)

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
    storage = optuna.storages.RDBStorage(
        url=args.db,
        engine_kwargs={"pool_size": 2, "max_overflow": 1, "pool_pre_ping": True, "pool_recycle": 300},
    )
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
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials.", flush=True)
        return

    best = study.best_trial
    ua = best.user_attrs

    print(f"\n{'='*60}", flush=True)
    print(f"BEST TRIAL #{best.number} (G+L TRAIL SEARCH)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Score:        ${best.value:,.0f}", flush=True)
    print(f"  PnL:          ${ua.get('total_pnl',0):,.0f}", flush=True)
    print(f"  PF:           {ua.get('pf',0):.3f}", flush=True)
    print(f"  WR:           {ua.get('wr',0):.1f}%", flush=True)
    print(f"  n:            {ua.get('n')}", flush=True)
    print(f"  Equity:       ${ua.get('equity',0):,.0f}", flush=True)
    print(f"\n  Params:", flush=True)
    for k, v in sorted(best.params.items()):
        print(f"    {k}: {v}", flush=True)

    if args.params_out:
        out = {
            "label": f"G+L trail search (optuna #{best.number})",
            "study": args.study,
            "trial": best.number,
            "score": best.value,
            "total_pnl": ua.get("total_pnl"),
            "pf": ua.get("pf"),
            "wr": ua.get("wr"),
            "n": ua.get("n"),
            "params": dict(best.params),
        }
        with open(args.params_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved to {args.params_out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="G+L trail optimization")
    parser.add_argument("--db", default="postgresql://postgres@127.0.0.1:5432/optuna_gl_trail",
                        help="PostgreSQL storage URL")
    parser.add_argument("--study", default="gl_trail", help="Study name")
    parser.add_argument("--trials", type=int, default=800, help="Target completed trials")
    parser.add_argument("--startup-trials", type=int, default=50,
                        help="Random trials before TPE kicks in")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--params-out", default="config/trial_gl_trail_best.json",
                        help="Path to write best trial params JSON")
    args = parser.parse_args()

    configure_simulator()
    print("Loading training data...", flush=True)
    dates, picks_by_date = load_data()

    # Load baseline so all required params (for disabled strategies) are present
    base_path = "config/trial_w21b_511_deploy.json"
    with open(base_path) as f:
        base_data = json.load(f)
    base_params = dict(base_data.get("params", {}))
    print(f"  Base config: {base_path} ({len(base_params)} params)", flush=True)

    data = {"dates": dates, "picks_by_date": picks_by_date, "base_params": base_params}

    if args.workers > 1:
        import multiprocessing
        procs = []
        for _ in range(args.workers):
            p = multiprocessing.Process(target=worker_loop, args=(data, args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        worker_loop(data, args)

    # Print final best
    storage = optuna.storages.RDBStorage(url=args.db)
    study = optuna.load_study(study_name=args.study, storage=storage)
    dump_best(study, args)
    print("\nG+L Trail Search complete.", flush=True)


if __name__ == "__main__":
    main()
