"""
Optuna G+V3 Joint Optimization — Multi-Process Workers
======================================================
Jointly optimizes G strategy params (8 dims) + V3 overlay params (5 dims)
using ASK/TELL with PostgreSQL so N workers can run in separate processes.

Usage:
  # First terminal — creates study + runs worker loop
  python scripts/optimize/optuna_g_v3_joint.py

  # Additional terminals — workers only
  python scripts/optimize/optuna_g_v3_joint.py --worker

  # Override totals / config
  python scripts/optimize/optuna_g_v3_joint.py --trials 600 --startup-trials 200

Architecture
  Phase 1: Pre-compute V3 candidates from #511 baseline G (once, saves cache)
  Phase 2: Each worker connects to PostgreSQL and calls ask -> objective -> tell
           in a loop until all trials are complete.

  V3 candidates are pre-computed because "v3 candidate set barely changes
  across G param variants" (optuna_g_l_cv.py approximation).

  13 total dimensions (8 G + 5 V3).
  n_startup_trials = 200  (~15x dims).
  Score floor: G trades >= 50, combined_pnl > 0, combined_pf >= 0.5.
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
POSITION_PCT = 0.30
BASELINE_PATH = "config/trial_g511_l626_v3_overlay.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
SHARED_CACHE = ".gv3_shared_data.pkl"

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

# ── Imports (lazy — after modifying sys.path) ───────────────────────────────
import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock


# ── Phase 1: Pre-compute V3 candidates (run once) ────────────────────────────

def build_shared_data():
    """Run Phase 1: load data, compute baseline G holds, pre-compute V3 candidates.
    Returns a dict with all data a worker needs."""
    print("=" * 60, flush=True)
    print("Phase 1: Pre-computing V3 candidates from baseline G", flush=True)
    print("=" * 60, flush=True)

    # Configure simulator
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

    # Load data
    print("Loading data...", flush=True)
    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates_dict, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates_dict if DATE_LO <= d <= DATE_HI])
    print(f"  Training window: {DATE_LO} to {DATE_HI} ({len(dates)} days)", flush=True)
    print(f"  Data dirs: {[d for d in DATA_DIRS if os.path.exists(d)]}", flush=True)

    if len(dates) == 0:
        print("ERROR: no trading days in range!", flush=True)
        sys.exit(1)

    # Load baseline config
    with open(BASELINE_PATH) as f:
        baseline_full = json.load(f)
    bl_params = dict(baseline_full["params"])

    # Override: only G enabled (disable L + everything else)
    for s in ALL_STRATS:
        bl_params[f"enable_{s}"] = (s == "g")
    bl_params["enable_x"] = False

    # Run baseline G to capture hold windows
    print("  Running baseline G-only sim...", flush=True)
    with _param_lock:
        _oc_set_params(bl_params)
        _disable_adaptive_controls()
        bl_snapshot = _build_param_snapshot()

    g_holds = {}
    cash = float(STARTING_CASH)
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp:
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                dp, cash, cash_account=is_cash, params=bl_snapshot
            )
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
                key = (st.get("ticker"), d)
                et = st.get("entry_time")
                xt = st.get("exit_time")
                if et and xt:
                    g_holds.setdefault(key, []).append((et, xt))
        cash = end_c + (unset if is_cash else 0)
    n_holds = sum(len(v) for v in g_holds.values())
    print(f"  G hold windows: {n_holds} across {len(g_holds)} (ticker,date) pairs", flush=True)

    # Compute V3 candidates
    counters = {"no_data": 0, "g_exited_after_eod": 0, "no_hit": 0, "fired": 0}
    v3_candidates = []
    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2:
                counters["no_data"] += 1
                continue
            day_open = float(mh.iloc[0]["Open"])
            bar0_close = float(mh.iloc[0]["Close"])
            bar0_red = bar0_close <= day_open
            holds = g_holds.get((p["ticker"], d), [])

            scan_start = 1
            if not bar0_red and holds:
                g_exit_ts = max(x for _, x in holds)
                new_start = None
                for i in range(1, len(mh)):
                    if mh.index[i] > g_exit_ts:
                        new_start = i
                        break
                if new_start is None:
                    counters["g_exited_after_eod"] += 1
                    continue
                scan_start = new_start

            entry_idx = None
            entry_price = None
            entry_ts = None
            for i in range(scan_start, len(mh)):
                c = float(mh.iloc[i]["Close"])
                if c > day_open:
                    entry_idx = i
                    entry_price = c
                    entry_ts = mh.index[i]
                    break
            if entry_idx is None or entry_price is None or entry_price <= 0:
                counters["no_hit"] += 1
                continue

            bars_after = mh.iloc[entry_idx + 1:]
            if len(bars_after) == 0:
                continue
            counters["fired"] += 1
            v3_candidates.append({
                "ticker": p["ticker"],
                "date": d,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "mh": mh,
                "bars_after": bars_after,
            })

    print(f"  V3 candidates: {len(v3_candidates)}", flush=True)
    for k, v in counters.items():
        print(f"    {k}: {v}", flush=True)

    if len(v3_candidates) < 50:
        print("WARNING: very few V3 candidates — V3 optimization may be weak", flush=True)

    shared = {
        "dates": dates,
        "picks_by_date": picks_by_date,
        "v3_candidates": v3_candidates,
        "bl_params": bl_params,
    }
    return shared


def _disable_adaptive_controls():
    """Lock all Phase 1A / adaptive controls to 0 (disabled)."""
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    for s in ALL_STRATS:
        setattr(tgc, f"{s.upper()}_PARTICIPATION_CAP", 0.0)
    tgc.NEWS_FILTER_ENABLED = False


# ── G-only backtest (parallel-safe via snapshot) ────────────────────────────

def run_g_backtest(daily_picks, all_dates, snapshot):
    """Run G-only backtest with a snapshot dict. Returns detailed metrics."""
    cash = float(STARTING_CASH)
    unsettled = 0.0
    all_pnls = []

    for d in all_dates:
        cash += unsettled
        unsettled = 0.0
        picks = daily_picks.get(d, [])
        if not picks:
            continue
        cash_account = cash < MARGIN_THRESHOLD
        try:
            states, cash, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account, params=snapshot
            )
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0)
                if pnl is not None:
                    all_pnls.append(pnl)

    n = len(all_pnls)
    if n == 0:
        return {"n": 0, "total_pnl": -9999, "pf": 0, "equity": cash,
                "gross_win": 0, "gross_loss": 1e-9, "wr": 0}

    total_pnl = sum(all_pnls)
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 99
    wr = len(wins) / n * 100 if n > 0 else 0

    return {
        "n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr,
        "equity": cash, "gross_win": gross_win, "gross_loss": gross_loss,
    }


# ── V3 overlay simulation ────────────────────────────────────────────────────

def _apply_caps(mh, ts, fill_price, requested):
    """Apply volume-based position caps."""
    pre = mh.loc[mh.index <= ts]
    vol_shares = float(pre["Volume"].sum()) if len(pre) > 0 else 0
    dollar_vol = fill_price * vol_shares
    if dollar_vol <= 0:
        return 0, 0
    vol_limit = dollar_vol * (tgc.VOL_CAP_PCT / 100)
    v_eff_adj, _, _, v_regime = tgc._multi_window_effective_volume(mh, ts, fill_price)
    if tgc.MAX_REGIME_PARTICIPATION > 0 and v_regime > 0:
        vol_limit = min(vol_limit, v_regime * tgc.MAX_REGIME_PARTICIPATION)
    if tgc.MAX_2MIN_PARTICIPATION > 0 and v_eff_adj > 0:
        vol_limit = min(vol_limit, v_eff_adj * tgc.MAX_2MIN_PARTICIPATION)
    return min(requested, vol_limit), v_eff_adj


def _v3_sim(v3_params, candidate):
    """Simulate one V3 trade. Returns PnL or 0."""
    entry_ts = candidate["entry_ts"]
    fill_price = candidate["entry_price"]
    mh = candidate["mh"]
    bars_after = candidate["bars_after"]

    pos_dollars = STARTING_CASH * POSITION_PCT
    capped_size, v_eff_adj = _apply_caps(mh, entry_ts, fill_price, pos_dollars)
    if capped_size < 50:
        return 0.0

    slip_in = tgc._entry_slip_pct(fill_price, capped_size, v_eff_adj)
    actual_entry = fill_price * (1 + slip_in / 100)
    shares = capped_size / actual_entry

    target = actual_entry * (1 + v3_params["v3_target_pct"] / 100)
    stop = actual_entry * (1 - v3_params["v3_stop_pct"] / 100)
    peak = actual_entry
    trail_stop = None
    max_bars = max(1, v3_params["v3_time_limit_min"] // 2)

    exit_price = None
    exit_ts = None
    for i, (ts, row) in enumerate(bars_after.iterrows()):
        if i >= max_bars:
            exit_price = float(row["Close"])
            exit_ts = ts
            break
        c_high = float(row["High"])
        c_low = float(row["Low"])
        if c_high >= target:
            exit_price = target
            exit_ts = ts
            break
        if c_low <= stop:
            exit_price = stop
            exit_ts = ts
            break
        if c_high > peak:
            peak = c_high
        unrealized = (peak / actual_entry - 1) * 100
        if unrealized >= v3_params["v3_trail_activate_pct"]:
            new_trail = peak * (1 - v3_params["v3_trail_pct"] / 100)
            if trail_stop is None or new_trail > trail_stop:
                trail_stop = new_trail
        if trail_stop is not None and c_low <= trail_stop:
            exit_price = trail_stop
            exit_ts = ts
            break

    if exit_price is None and len(bars_after) > 0:
        exit_price = float(bars_after.iloc[-1]["Close"])
        exit_ts = bars_after.index[-1]
    if exit_price is None or exit_ts is None:
        return 0.0

    fake_st = {"mh": mh}
    slip_out = tgc._exit_slip_pct(exit_price, shares, fake_st, exit_ts)
    actual_exit = exit_price * (1 - slip_out / 100)
    return shares * (actual_exit - actual_entry)


def compute_v3_pnl(v3_params, v3_candidates):
    """Compute V3 aggregate PnL across candidates.
    Returns (total_pnl, pf, n, gross_win, gross_loss)."""
    total = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    n = 0
    for c in v3_candidates:
        pnl = _v3_sim(v3_params, c)
        if pnl == 0.0:
            continue
        total += pnl
        if pnl > 0:
            gross_win += pnl
        else:
            gross_loss += -pnl
        n += 1
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    return total, pf, n, gross_win, gross_loss


# ── Optuna param suggesters ──────────────────────────────────────────────────

def suggest_g_params(trial):
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


def suggest_v3_params(trial):
    return {
        "v3_target_pct": trial.suggest_float("v3_target_pct", 10.0, 100.0, step=1.0),
        "v3_stop_pct": trial.suggest_float("v3_stop_pct", 5.0, 35.0, step=1.0),
        "v3_time_limit_min": trial.suggest_int("v3_time_limit_min", 3, 90, step=3),
        "v3_trail_pct": trial.suggest_float("v3_trail_pct", 0.5, 10.0, step=0.5),
        "v3_trail_activate_pct": trial.suggest_float("v3_trail_activate_pct", 0.0, 15.0, step=0.5),
    }


# ── Objective ────────────────────────────────────────────────────────────────

def objective(trial, data):
    """Optuna objective: suggest G+V3 params, run G sim, add V3 overlay."""
    picks_by_date = data["picks_by_date"]
    dates = data["dates"]
    v3_candidates = data["v3_candidates"]
    bl_params = data["bl_params"]

    g_params = suggest_g_params(trial)
    v3_params = suggest_v3_params(trial)

    # Build full param dict: baseline + disable all non-G + G trial overrides
    p = dict(bl_params)
    for s in ALL_STRATS:
        p[f"enable_{s}"] = (s == "g")
    p["enable_x"] = False
    p.update(g_params)

    # Set params, build snapshot
    with _param_lock:
        _oc_set_params(p)
        _disable_adaptive_controls()
        snapshot = _build_param_snapshot()

    # G only
    g_result = run_g_backtest(picks_by_date, dates, snapshot)

    # V3 overlay
    v3_total, v3_pf, v3_n, v3_gw, v3_gl = compute_v3_pnl(v3_params, v3_candidates)

    # Combined
    combined_pnl = g_result["total_pnl"] + v3_total
    combined_gw = g_result["gross_win"] + v3_gw
    combined_gl = g_result["gross_loss"] + v3_gl
    combined_pf = combined_gw / combined_gl if combined_gl > 0 else 99.0

    if g_result["n"] < 50 or combined_pnl <= 0 or combined_pf < 0.5:
        return -9999

    score = combined_pnl * min(combined_pf, 3.0)
    if math.isnan(score) or math.isinf(score):
        return -9999
    score = max(-9.9e12, min(9.9e12, float(score)))

    trial.set_user_attr("g_n", g_result["n"])
    trial.set_user_attr("g_pnl", round(g_result["total_pnl"], 2))
    trial.set_user_attr("g_pf", round(g_result["pf"], 3))
    trial.set_user_attr("g_wr", round(g_result["wr"], 1))
    trial.set_user_attr("v3_n", v3_n)
    trial.set_user_attr("v3_pnl", round(v3_total, 2))
    trial.set_user_attr("v3_pf", round(v3_pf, 3))
    trial.set_user_attr("combined_pnl", round(combined_pnl, 2))
    trial.set_user_attr("combined_pf", round(combined_pf, 3))

    return score


# ── Worker loop ──────────────────────────────────────────────────────────────

def worker_loop(data, args):
    """Connect to study and run ask → objective → tell until all trials done."""
    storage = optuna.storages.RDBStorage(url=args.db)

    # Prevent worker from creating a new study if DB doesn't exist yet;
    # if the study doesn't exist in the storage, create_study creates it.
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=args.startup_trials),
    )

    # Count existing complete trials
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

    # Register clean exit for Ctrl+C
    exiting = False
    def _handle_sigint(sig, frame):
        nonlocal exiting
        exiting = True
        print(f"\n[worker] SIGINT received — exiting after {my_count} trials...", flush=True)

    if os.name != "nt":
        signal.signal(signal.SIGINT, _handle_sigint)
    # On Windows, Ctrl+C raises KeyboardInterrupt which the except block catches

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
            # Mark the claimed trial as FAIL so it doesn't block the study
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

        # Progress report
        if my_count % 20 == 0 or my_count == last_report + 1:
            completed_now = len([t for t in study.trials
                                 if t.state == optuna.trial.TrialState.COMPLETE])
            elapsed = time.time() - t_start
            try:
                bt = study.best_trial
                print(f"  [{completed_now}/{args.trials}] {elapsed/60:.1f}m | "
                      f"best ${bt.value:,.0f} (#{bt.number}) "
                      f"G=${bt.user_attrs.get('g_pnl',0):,.0f} "
                      f"V3=${bt.user_attrs.get('v3_pnl',0):,.0f} "
                      f"PF={bt.user_attrs.get('g_pf',0):.2f}",
                      flush=True)
            except Exception:
                print(f"  [{completed_now}/{args.trials}] {elapsed/60:.1f}m | "
                      f"no completed trials yet", flush=True)
            last_report = my_count

    total_elapsed = time.time() - t_start
    print(f"\n[worker] Done: {my_count} trials in {total_elapsed/60:.1f}m", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G+V3 joint Optuna optimization (multi-process)")
    parser.add_argument("--trials", type=int, default=600, help="Total trials across all workers")
    parser.add_argument("--startup-trials", type=int, default=200,
                        help="TPE n_startup_trials")
    parser.add_argument("--db", default="postgresql://postgres@127.0.0.1:5432/optuna_g_v3",
                        help="PostgreSQL storage URL (shared across workers)")
    parser.add_argument("--study", default="g_v3_joint",
                        help="Optuna study name")
    parser.add_argument("--params-out", default="config/trial_g_v3_joint_best.json",
                        help="Best params output path")
    parser.add_argument("--worker", action="store_true",
                        help="Run as worker only (skips Phase 1 cache rebuild)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild the Phase 1 cache")
    parser.add_argument("--dump-best", action="store_true",
                        help="Dump best params from an existing study and exit")
    args = parser.parse_args()

    # ── Dump best mode ────────────────────────────────────────────────────
    if args.dump_best:
        use_postgres = args.db.startswith("postgresql://") or args.db.startswith("postgres://")
        if use_postgres:
            storage = optuna.storages.RDBStorage(url=args.db)
        else:
            storage = f"sqlite:///{args.db}"
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study,
            storage=storage,
            load_if_exists=True,
        )
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
            print("No completed trials in study.", flush=True)
            sys.exit(1)
        _dump_best(study, args)
        return

    # ── Phase 1: Load or compute shared data ──────────────────────────────
    if args.worker and not args.rebuild_cache:
        # Worker mode: load from shared cache
        if not os.path.exists(SHARED_CACHE):
            print(f"ERROR: {SHARED_CACHE} not found. Run main process (no --worker) first.", flush=True)
            sys.exit(1)
        print(f"Loading shared data from {SHARED_CACHE}...", flush=True)
        with open(SHARED_CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"  {len(data['dates'])} days, {len(data['v3_candidates'])} V3 candidates", flush=True)
    else:
        data = build_shared_data()
        print(f"\nSaving shared data to {SHARED_CACHE}...", flush=True)
        with open(SHARED_CACHE, "wb") as f:
            pickle.dump(data, f, protocol=4)
        print("  Done.", flush=True)

    # ── Phase 2: Worker loop ──────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 2: Worker Loop", flush=True)
    print(f"  Study: {args.study}", flush=True)
    print(f"  DB:    {args.db}", flush=True)
    print(f"  Dims:  8 G + 5 V3 = 13", flush=True)
    print(f"  n_startup_trials: {args.startup_trials} ({args.startup_trials/13:.0f}x dims)", flush=True)
    print(f"  Target completed trials: {args.trials}", flush=True)
    print(f"{'='*60}", flush=True)

    worker_loop(data, args)

    # ── Report best (only if we're the main process or the study is done) ──
    try:
        use_postgres = args.db.startswith("postgresql://") or args.db.startswith("postgres://")
        if use_postgres:
            storage = optuna.storages.RDBStorage(url=args.db)
        else:
            storage = f"sqlite:///{args.db}"
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study,
            storage=storage,
            load_if_exists=True,
        )
        _dump_best(study, args)
    except Exception as e:
        print(f"\nCould not report best: {e}", flush=True)


def _dump_best(study, args):
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
    print(f"  V3:  n={ua.get('v3_n')}, PnL=${ua.get('v3_pnl',0):,.0f},  "
          f"PF={ua.get('v3_pf',0):.2f}", flush=True)
    print(f"  Combo:        PnL=${ua.get('combined_pnl',0):,.0f},  "
          f"PF={ua.get('combined_pf',0):.2f}", flush=True)

    # Save best params
    v3_overlay = {"enabled": True}
    for k, v in best.params.items():
        if k.startswith("v3_"):
            v3_overlay[k.replace("v3_", "")] = v

    best_params = {
        "label": f"G+V3 joint optuna #{best.number}",
        "params": {k: v for k, v in best.params.items() if k.startswith("g_")},
        "v3_overlay": v3_overlay,
        "source": f"optuna_g_v3_joint.py trial #{best.number}",
        "score": best.value,
        "g_pnl": ua.get("g_pnl", 0),
        "v3_pnl": ua.get("v3_pnl", 0),
        "g_pf": ua.get("g_pf", 0),
        "combined_pf": ua.get("combined_pf", 0),
        "g_n": ua.get("g_n", 0),
        "v3_n": ua.get("v3_n", 0),
    }
    with open(args.params_out, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {args.params_out}", flush=True)


if __name__ == "__main__":
    main()
