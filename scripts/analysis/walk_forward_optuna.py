"""Walk-forward Optuna with EXPANDING training window.

W1: train 2021                test 2022
W2: train 2021..2022          test 2023
W3: train 2021..2023          test 2024
W4: train 2021..2024          test 2025
W5: train 2021..2025          test 2026

For each window:
  1. Run Optuna n_trials times on the train years' picks
  2. Extract best trial, save params to JSON
  3. Backtest those params on the test year
  4. Record forward PnL

At the end, build a "walk-forward equity curve" that uses each window's
best params on its test year, glued end-to-end starting at $25K.

Compare against:
  - Static #818 deployed across all 5 test years
  - #6 generalist deployed across all 5 test years

Resumeable: each window's Optuna study uses its own SQLite DB
(optuna_walkforward_W{N}.db). Forward results saved to JSON. If a window
already has its forward JSON, it's skipped on rerun.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import multiprocessing
import os
import time
import numpy as np
import optuna
from optuna.samplers import TPESampler

import test_green_candle_combined as tgc
import optimize_combined as opt

# CLI override for Optuna's n_jobs — set in main() from args, read by run_window_optuna()
_CLI_N_JOBS = None
from optimize_combined import objective, set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

YEAR_DIRS = {
    "2021": ["stored_data_2021"],
    "2022": ["stored_data_2022"],
    "2023": ["stored_data_2023"],
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"],
}

WINDOWS = [
    (["2021"],                                  "2022"),
    (["2021", "2022"],                          "2023"),
    (["2021", "2022", "2023"],                  "2024"),
    (["2021", "2022", "2023", "2024"],          "2025"),
    # W5 (2026-06-18): drops 2021 (overfit bonanza year), keeps 2022-2025
    # to maintain 4-year diversity while cutting per-trial compute by ~25%.
    (["2022", "2023", "2024", "2025"],          "2026"),
]

BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000


def _dirs_for_years(years):
    out = []
    for y in years:
        out.extend(YEAR_DIRS[y])
    return out


def _restrict_to_years(daily_picks, all_dates, years):
    """Keep only days whose ISO date starts with one of the given years."""
    keep = set()
    for y in years:
        keep.add(y)
    out_dates = [d for d in all_dates if d[:4] in keep]
    out_picks = {d: daily_picks[d] for d in out_dates if d in daily_picks}
    return out_dates, out_picks


def _merged_params(best_params):
    """Merge trial params into the baseline (matches the OOS scripts)."""
    with open(BASELINE) as f:
        b = json.load(f)
    m = dict(b)
    m.update(best_params)
    return m


def run_window_optuna(train_years, n_trials, n_startup, db_path, study_name,
                       regime_filter=None, shape_filter=None, storage_url=None):
    print(f"\n  Training window {train_years} -> Optuna ({n_trials} trials, {n_startup} startup)")
    dirs = _dirs_for_years(train_years)
    dirs = [d for d in dirs if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    train_dates, train_picks = _restrict_to_years(picks, all_dates, train_years)
    print(f"  Train picks loaded: {len(train_dates)} days "
          f"({train_dates[0]} -> {train_dates[-1]})")

    # Regime filter: keep only days matching the requested regime
    if regime_filter:
        from strategies.regime_gate import classify_regime
        keep = [d for d in train_dates
                if classify_regime(train_picks.get(d, [])) == regime_filter]
        print(f"  Regime filter '{regime_filter}': {len(keep)} of {len(train_dates)} days kept "
              f"({100*len(keep)/max(1,len(train_dates)):.1f}%)")
        train_dates = keep
        train_picks = {d: train_picks[d] for d in keep if d in train_picks}

    # Shape filter: finer-grained than regime, uses 7-bucket taxonomy
    # (broad-squeeze, microcap-thin, thin-microcap, liquid-normal, mega-cap, corp-action, dead)
    # Accepts comma-separated list, e.g. "microcap-thin,thin-microcap" = "microcap-pump merged".
    if shape_filter:
        from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
        from strategies.regime_gate import classify_regime
        shapes_wanted = {s.strip() for s in shape_filter.split(",")}
        keep = []
        for d in train_dates:
            picks = train_picks.get(d, [])
            sig = day_signature(picks)
            regime = classify_regime(picks) if picks else "dead"
            shape = classify_shape(sig, regime)
            if shape in shapes_wanted:
                keep.append(d)
        print(f"  Shape filter '{shape_filter}': {len(keep)} of {len(train_dates)} days kept "
              f"({100*len(keep)/max(1,len(train_dates)):.1f}%)")
        train_dates = keep
        train_picks = {d: train_picks[d] for d in keep if d in train_picks}

    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.VOL_CAP_PCT = 5.0

    # Backend selection: Postgres (multi-worker concurrent) or SQLite (legacy).
    if storage_url and storage_url.startswith("postgresql"):
        rdb = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"pool_size": 4, "max_overflow": 8,
                            "pool_pre_ping": True, "pool_recycle": 3600},
        )
        print(f"  Storage: Postgres ({storage_url.rsplit('@',1)[-1]})")
    else:
        # Pre-bootstrap WAL mode for SQLite
        import sqlite3
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=NORMAL")
            conn.commit()
        finally:
            conn.close()
        from sqlalchemy import event
        rdb = optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path}",
            engine_kwargs={"connect_args": {"timeout": 120}},
        )
        @event.listens_for(rdb.engine, "connect")
        def _set_busy_timeout(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA busy_timeout=120000"); cur.close()
        print(f"  Storage: SQLite ({db_path})")

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=rdb,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=n_startup),
    )

    n_already = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_remaining = max(0, n_trials - n_already)
    print(f"  Existing COMPLETE trials: {n_already}  |  running {n_remaining} more")
    if n_remaining == 0:
        print(f"  (already at target n_trials; using existing best)")
    else:
        # n_jobs comes from CLI when set, else default cpu//2.
        # Pass --n-jobs 1 when running multiple processes for multi-process parallelism.
        n_jobs = _CLI_N_JOBS if _CLI_N_JOBS is not None else max(1, multiprocessing.cpu_count() // 2)
        t_start = time.time()
        study.optimize(
            lambda t: objective(t, train_picks, train_dates),
            n_trials=n_remaining,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )
        print(f"  Window train time: {(time.time()-t_start)/60:.1f} min")

    b = study.best_trial
    print(f"  Best trial #{b.number}  score=${b.value:,.0f}  "
          f"PF={b.user_attrs.get('pf','-')}  pnl=${b.user_attrs.get('total_pnl',0):,.0f}")
    return b


def forward_test(best_trial, test_year):
    """Run best_trial params on test_year. Returns equity curve + per-day pnl."""
    dirs = [d for d in YEAR_DIRS[test_year] if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    test_dates, test_picks = _restrict_to_years(picks, all_dates, [test_year])
    print(f"  Forward test {test_year}: {len(test_dates)} days")

    set_strategy_params(_merged_params(best_trial.params))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.VOL_CAP_PCT = 5.0

    cash = STARTING_CASH
    eq_curve = [cash]
    daily = []
    for d in test_dates:
        day_picks = test_picks.get(d, [])
        # PHASE NEWS: same PIT news filter the optimizer uses, so forward-test
        # honors the trial's filter choice.
        if day_picks and tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as _news_filter
            day_picks = _news_filter(
                day_picks, d,
                min_articles=tgc.NEWS_MIN_ARTICLES,
                require_catalyst=tgc.NEWS_REQUIRE_CATALYST,
            )
        if not day_picks:
            eq_curve.append(cash)
            daily.append({"date": d, "pnl": 0, "equity": cash})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            _, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
        except Exception as e:
            eq_curve.append(cash)
            daily.append({"date": d, "pnl": 0, "equity": cash, "error": str(e)[:60]})
            continue
        pnl = end_c - cash
        cash = end_c
        if is_cash:
            cash += unset
        eq_curve.append(cash)
        daily.append({"date": d, "pnl": pnl, "equity": cash})

    pnls = np.array([r["pnl"] for r in daily])
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0.0
    wins = int((pnls > 0).sum())
    losses = int((pnls < 0).sum())
    return {
        "test_year": test_year,
        "n_days": len(test_dates),
        "final_equity": cash,
        "total_pnl": cash - STARTING_CASH,
        "sharpe": sharpe,
        "wins": wins,
        "losses": losses,
        "daily": daily,
        "equity_curve": eq_curve,
        "first_date": test_dates[0],
        "last_date": test_dates[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--n-startup", type=int, default=10)
    ap.add_argument("--outdir", default="results/walk_forward")
    ap.add_argument("--only-window", type=int, default=None,
                    help="Run only this window (1-indexed). Otherwise runs all 5 in sequence.")
    ap.add_argument("--use-2min-slippage", action="store_true",
                    help="Activate USE_2MIN_SLIPPAGE in the simulator (single-window).")
    ap.add_argument("--use-multiwindow-slippage", action="store_true",
                    help="Activate USE_MULTIWINDOW_SLIPPAGE (V_eff = max(v_2min, "
                         "0.5*v_6min, 0.25*v_10min) + volatility adjustment). "
                         "Supersedes --use-2min-slippage when both are set.")
    ap.add_argument("--sharpe-objective", action="store_true",
                    help="Use total_pnl * min(sharpe_pct, 4.0) as objective instead of legacy PF.")
    ap.add_argument("--cv-objective", action="store_true",
                    help="Per-year-reset CV objective: score = min(year_pnl * min(year_pf, 3.0)) "
                         "across all training years. Each year backtested with fresh $25K. "
                         "Empirically (W7) min_year_pf correlates 0.64 with forward PnL vs "
                         "the legacy objective's 0.28. Use this to prevent regime overfit.")
    ap.add_argument("--regime-filter", default=None,
                    choices=["squeeze", "normal", "dead"],
                    help="Train only on days where classify_regime() == this regime.")
    ap.add_argument("--shape-filter", default=None,
                    help="Train only on days matching the requested shape(s) from "
                         "the 7-bucket taxonomy. Comma-separated. "
                         "Choices: broad-squeeze, microcap-thin, thin-microcap, "
                         "liquid-normal, mega-cap, corp-action. "
                         "Example: --shape-filter microcap-thin,thin-microcap "
                         "for the microcap-pump merged specialist.")
    ap.add_argument("--no-x", action="store_true",
                    help="Force-disable strategy X (Range Reversion) on all trials. "
                         "Useful when X's structural high participation drags PnL.")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="Optuna study.optimize n_jobs (intra-process threads). "
                         "Default = cpu//2. Set to 1 when running multiple Python "
                         "processes against the same DB to bypass GIL contention.")
    ap.add_argument("--storage-url", default=None,
                    help="Optuna RDB storage URL. Pass "
                         "'postgresql://postgres@127.0.0.1:5432/optuna' for the "
                         "Postgres backend (no SQLite write contention). "
                         "Default: per-window SQLite under --outdir.")
    args = ap.parse_args()

    # Module-level so run_window_optuna() can see it without threading the arg through.
    global _CLI_N_JOBS
    _CLI_N_JOBS = args.n_jobs

    # Wire flags into the optimize_combined objective + simulator.
    # Multiwindow supersedes 2-min when both are set.
    if args.use_multiwindow_slippage:
        tgc.USE_MULTIWINDOW_SLIPPAGE = True
        tgc.USE_VOLATILITY_ADJUSTMENT = True
        tgc.USE_DYNAMIC_SLIPPAGE = True
        print(f"  [flag] USE_MULTIWINDOW_SLIPPAGE = True  (V_eff multiwindow + vol-adj)")
        print(f"           Caps: cumulative 5% / regime {100*tgc.MAX_REGIME_PARTICIPATION:.0f}% of v_10min / "
              f"execution {100*tgc.MAX_2MIN_PARTICIPATION:.0f}% of V_eff_adj")
    elif args.use_2min_slippage:
        opt.ENABLE_2MIN_SLIPPAGE = True
        tgc.USE_2MIN_SLIPPAGE = True
        tgc.USE_DYNAMIC_SLIPPAGE = True
        print(f"  [flag] USE_2MIN_SLIPPAGE = True (single-window; MAX_2MIN_PARTICIPATION = {tgc.MAX_2MIN_PARTICIPATION})")
    if args.sharpe_objective:
        opt.USE_SHARPE_OBJECTIVE = True
        print(f"  [flag] USE_SHARPE_OBJECTIVE = True (score = total_pnl * min(sharpe_pct, 4.0))")
    if args.cv_objective:
        opt.USE_CV_OBJECTIVE = True
        print(f"  [flag] USE_CV_OBJECTIVE = True (per-year reset; score = min(year_pnl * min(year_pf, 3.0)))")
    if args.no_x:
        # Kill switch: X requires first_leg_gain >= X_MIN_FIRST_LEG_GAIN_PCT;
        # set this to 9999 so no day can ever qualify -> X never fires.
        # This is the same kill-switch mechanism used by enable_* flags.
        tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0
        print(f"  [flag] X (Range Reversion) DISABLED (X_MIN_FIRST_LEG_GAIN_PCT=9999)")
    if args.regime_filter:
        print(f"  [flag] Regime filter = {args.regime_filter} (per-window day-filter)")
    if args.shape_filter:
        print(f"  [flag] Shape filter  = {args.shape_filter} (per-window day-filter, 7-bucket taxonomy)")

    # --- PHASE 1A wiring ---
    # When PHASE_1A_ENABLED=1, the 34 new tunables enter the search space
    # (via suggest_all_params in optimize_combined.py). We also need to load
    # macro data so the day-favorability gate has VIX/DXY/BTC/IWM context.
    if os.environ.get("PHASE_1A_ENABLED", "0") == "1":
        macro_csv = "results/regime_trends/predictive_features.csv"
        if os.path.exists(macro_csv):
            n_rows = tgc.load_macro_data(macro_csv)
            print(f"  [PHASE 1A] Loaded {n_rows} macro rows from {macro_csv}")
            print(f"             VIX tercile bounds: {tgc._VIX_TERCILE_BOUNDS}")
            print(f"             IWM tercile bounds: {tgc._IWM_TERCILE_BOUNDS}")
        else:
            print(f"  [PHASE 1A] WARN: {macro_csv} not found — favorability gate inert.")
        print(f"  [PHASE 1A] ENABLED — 34 new tunables in search space")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Walk-forward Optuna: {len(WINDOWS)} windows | {args.n_trials} trials/window")

    for i, (train_years, test_year) in enumerate(WINDOWS, 1):
        if args.only_window and args.only_window != i:
            continue
        tag = f"W{i}_train_{'_'.join(train_years)}_test_{test_year}"
        forward_json = os.path.join(args.outdir, f"{tag}_forward.json")
        if os.path.exists(forward_json):
            print(f"\n=== Window {i} ({tag}) === SKIP (forward.json exists)")
            continue

        print(f"\n{'='*92}\n=== Window {i}/{len(WINDOWS)}: TRAIN {train_years} -> TEST {test_year} ===\n{'='*92}")
        t0 = time.time()
        db_path = os.path.join(args.outdir, f"{tag}.db")
        study_name = tag

        best = run_window_optuna(train_years, args.n_trials, args.n_startup,
                                  db_path, study_name,
                                  regime_filter=args.regime_filter,
                                  shape_filter=args.shape_filter,
                                  storage_url=args.storage_url)
        # Save best params + meta
        best_path = os.path.join(args.outdir, f"{tag}_best.json")
        with open(best_path, "w") as f:
            json.dump({
                "trial_number": best.number,
                "score": best.value,
                "user_attrs": dict(best.user_attrs),
                "params": dict(best.params),
                "train_years": train_years,
                "test_year": test_year,
            }, f, indent=2)
        print(f"  Saved best params -> {best_path}")

        # Forward test
        fwd = forward_test(best, test_year)
        with open(forward_json, "w") as f:
            json.dump(fwd, f, indent=2)
        print(f"  Forward {test_year}: final ${fwd['final_equity']:,.0f}  "
              f"pnl ${fwd['total_pnl']:+,.0f}  Sharpe {fwd['sharpe']:.2f}  "
              f"wins {fwd['wins']}/{fwd['n_days']}")
        print(f"  Window total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
