"""
G+L Split-Param Optuna Worker
=============================
Tunes G and L with SEPARATE exit params for 1st trade vs 2nd+ trades per day.
Entry conditions shared; only exit params split (G1/G2, L1/L2).

The split-param mechanism works via entry_params on each state:
  - At entry, st["entry_params"] snapshots g1_*/g2_*/l1_*/l2_* keys
  - Exit logic reads from st["entry_params"] instead of module globals
  - The objective maps split params to standard names for set_strategy_params,
    and stores the full split keys in the snapshot for the simulator.

Usage:
  python optimize_gl_split.py --study gl_split_v1 --db postgresql://... --trials 500
  python optimize_gl_split.py --dump-best --db postgresql://... --study gl_split_v1
"""

import os
import sys
import io
import time
import json
import argparse
import multiprocessing
import numpy as np

if __name__ == "__main__" and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

import optuna
from optuna.samplers import TPESampler

import test_green_candle_combined as tgc
from test_full import load_all_picks, STARTING_CASH, MARGIN_THRESHOLD

# Reuse infrastructure from optimize_combined
from optimize_combined import (
    DATA_DIRS, ALL_STRATS, STRAT_KEYS, _param_lock,
    set_strategy_params, run_combined_backtest,
    make_callback,
    dump_best_params as _dump_best_params_combined,
    merge_enable_from_user_attrs,
)

# ── Defaults ────────────────────────────────────────────────────────────
DEFAULT_DB = "postgresql://postgres@127.0.0.1:5432/optuna_gl_split"
DEFAULT_STUDY = "gl_split_v2"
DEFAULT_PARAMS_OUT = "config/trial_gl_split_v2_best.json"


# ═══════════════════════════════════════════════════════════════════════
#  Suggest split params
# ═══════════════════════════════════════════════════════════════════════
def suggest_gl_split_params(trial):
    """Suggest params for G+L with G1/G2/L1/L2 exit splits.

    Returns a dict with:
      - Shared G entry: g_min_gap_pct, g_require_2nd_green, g_require_2nd_new_high
      - G1 exit: g1_target_pct, g1_time_limit_min, g1_stop_pct, g1_trail_pct, g1_trail_activate_pct
      - G2 exit: g2_target_pct, g2_time_limit_min, g2_stop_pct, g2_trail_pct, g2_trail_activate_pct
      - Shared L entry: l_min_gap, l_max_float, l_earliest_candle, l_latest_candle,
        l_vol_surge_mult, l_min_price_accel_pct, l_tier1_float, l_tier2_float
      - L1 exit: l1_tier1_target1_pct, l1_tier1_target2_pct, ..., l1_time_limit_min
      - L2 exit: l2_tier1_target1_pct, l2_tier1_target2_pct, ..., l2_time_limit_min
      - enable_g, enable_l, priority_g, priority_l (fixed ON)
    """
    params = {}

    # ── G shared entry (3 params) ──
    params["g_min_gap_pct"] = trial.suggest_float("g_min_gap_pct", 15.0, 80.0, step=5.0)
    params["g_require_2nd_green"] = trial.suggest_categorical("g_require_2nd_green", [True, False])
    params["g_require_2nd_new_high"] = trial.suggest_categorical("g_require_2nd_new_high", [True, False])

    # ── G1 exit (5 params) — 1st G trade of the day ──
    params["g1_target_pct"] = trial.suggest_float("g1_target_pct", 4.0, 20.0, step=1.0)
    params["g1_time_limit_min"] = trial.suggest_int("g1_time_limit_min", 3, 30, step=3)
    params["g1_stop_pct"] = trial.suggest_float("g1_stop_pct", 0.0, 12.0, step=2.0)
    params["g1_trail_pct"] = trial.suggest_float("g1_trail_pct", 0.0, 5.0, step=1.0)
    params["g1_trail_activate_pct"] = trial.suggest_float("g1_trail_activate_pct", 0.0, 8.0, step=2.0)

    # ── G2 exit (5 params) — 2nd+ G trades of the day ──
    params["g2_target_pct"] = trial.suggest_float("g2_target_pct", 4.0, 20.0, step=1.0)
    params["g2_time_limit_min"] = trial.suggest_int("g2_time_limit_min", 3, 30, step=3)
    params["g2_stop_pct"] = trial.suggest_float("g2_stop_pct", 0.0, 12.0, step=2.0)
    params["g2_trail_pct"] = trial.suggest_float("g2_trail_pct", 0.0, 5.0, step=1.0)
    params["g2_trail_activate_pct"] = trial.suggest_float("g2_trail_activate_pct", 0.0, 8.0, step=2.0)

    # ── L shared entry (8 params) ──
    params["l_min_gap"] = trial.suggest_int("l_min_gap", 15, 80, step=5)
    params["l_max_float"] = trial.suggest_int("l_max_float", 5_000_000, 20_000_000, step=5_000_000)
    params["l_earliest_candle"] = trial.suggest_int("l_earliest_candle", 3, 15, step=3)
    params["l_latest_candle"] = trial.suggest_int("l_latest_candle", 60, 150, step=15)
    params["l_vol_surge_mult"] = trial.suggest_float("l_vol_surge_mult", 1.0, 3.0, step=0.5)
    params["l_min_price_accel_pct"] = trial.suggest_float("l_min_price_accel_pct", 0.5, 3.0, step=0.5)
    params["l_tier1_float"] = trial.suggest_int("l_tier1_float", 500_000, 2_000_000, step=500_000)
    params["l_tier2_float"] = trial.suggest_int("l_tier2_float", 3_000_000, 7_000_000, step=1_000_000)

    # ── L1 exit (11 params) — 1st L trade of the day ──
    params["l1_tier1_target1_pct"] = trial.suggest_float("l1_tier1_target1_pct", 15.0, 50.0, step=5.0)
    params["l1_tier1_target2_pct"] = trial.suggest_float("l1_tier1_target2_pct", 30.0, 60.0, step=5.0)
    params["l1_tier2_target1_pct"] = trial.suggest_float("l1_tier2_target1_pct", 8.0, 30.0, step=2.0)
    params["l1_tier2_target2_pct"] = trial.suggest_float("l1_tier2_target2_pct", 20.0, 50.0, step=5.0)
    params["l1_tier3_target1_pct"] = trial.suggest_float("l1_tier3_target1_pct", 5.0, 20.0, step=1.0)
    params["l1_tier3_target2_pct"] = trial.suggest_float("l1_tier3_target2_pct", 15.0, 40.0, step=5.0)
    params["l1_stop_pct"] = trial.suggest_float("l1_stop_pct", 5.0, 20.0, step=1.0)
    params["l1_partial_sell_pct"] = trial.suggest_float("l1_partial_sell_pct", 0.0, 50.0, step=25.0)
    params["l1_trail_pct"] = trial.suggest_float("l1_trail_pct", 1.0, 6.0, step=1.0)
    params["l1_trail_activate_pct"] = trial.suggest_float("l1_trail_activate_pct", 1.0, 8.0, step=1.0)
    params["l1_time_limit_min"] = trial.suggest_int("l1_time_limit_min", 30, 120, step=10)

    # ── L2 exit (11 params) — 2nd+ L trades of the day ──
    params["l2_tier1_target1_pct"] = trial.suggest_float("l2_tier1_target1_pct", 15.0, 50.0, step=5.0)
    params["l2_tier1_target2_pct"] = trial.suggest_float("l2_tier1_target2_pct", 30.0, 60.0, step=5.0)
    params["l2_tier2_target1_pct"] = trial.suggest_float("l2_tier2_target1_pct", 8.0, 30.0, step=2.0)
    params["l2_tier2_target2_pct"] = trial.suggest_float("l2_tier2_target2_pct", 20.0, 50.0, step=5.0)
    params["l2_tier3_target1_pct"] = trial.suggest_float("l2_tier3_target1_pct", 5.0, 20.0, step=1.0)
    params["l2_tier3_target2_pct"] = trial.suggest_float("l2_tier3_target2_pct", 15.0, 40.0, step=5.0)
    params["l2_stop_pct"] = trial.suggest_float("l2_stop_pct", 5.0, 20.0, step=1.0)
    params["l2_partial_sell_pct"] = trial.suggest_float("l2_partial_sell_pct", 0.0, 50.0, step=25.0)
    params["l2_trail_pct"] = trial.suggest_float("l2_trail_pct", 1.0, 6.0, step=1.0)
    params["l2_trail_activate_pct"] = trial.suggest_float("l2_trail_activate_pct", 1.0, 8.0, step=1.0)
    params["l2_time_limit_min"] = trial.suggest_int("l2_time_limit_min", 30, 120, step=10)

    # ── Strategy enable/disable ──
    # Only G and L are active. All others must be explicitly disabled
    # because set_strategy_params defaults enable to True when key is missing.
    for s in ALL_STRATS:
        if s in ("g", "l"):
            params[f"enable_{s}"] = True
        else:
            params[f"enable_{s}"] = False
    params["priority_g"] = 1
    params["priority_l"] = 7

    return params


# ═══════════════════════════════════════════════════════════════════════
#  Map split params → standard names for set_strategy_params
# ═══════════════════════════════════════════════════════════════════════
def _map_split_to_standard(params):
    """Map G1/G2/L1/L2 split params to the standard g_*/l_* names that
    set_strategy_params() expects. Uses G1/L1 as the 'default' for
    module globals (the simulator reads per-trade params from entry_params).

    Returns a new dict with standard names + original split keys preserved.
    """
    std = dict(params)

    # G exit: use G1 as the default for module globals
    std["g_target_pct"] = params["g1_target_pct"]
    std["g_time_limit_min"] = params["g1_time_limit_min"]
    std["g_stop_pct"] = params["g1_stop_pct"]
    std["g_trail_pct"] = params["g1_trail_pct"]
    std["g_trail_activate_pct"] = params["g1_trail_activate_pct"]

    # L exit: use L1 as the default for module globals
    # Also map L1 tier targets to standard L_TIER* names for _get_tiered_targets
    std["l_tier1_target1_pct"] = params["l1_tier1_target1_pct"]
    std["l_tier1_target2_pct"] = params["l1_tier1_target2_pct"]
    std["l_tier2_target1_pct"] = params["l1_tier2_target1_pct"]
    std["l_tier2_target2_pct"] = params["l1_tier2_target2_pct"]
    std["l_tier3_target1_pct"] = params["l1_tier3_target1_pct"]
    std["l_tier3_target2_pct"] = params["l1_tier3_target2_pct"]
    std["l_stop_pct"] = params["l1_stop_pct"]
    std["l_partial_sell_pct"] = params["l1_partial_sell_pct"]
    std["l_trail_pct"] = params["l1_trail_pct"]
    std["l_trail_activate_pct"] = params["l1_trail_activate_pct"]
    std["l_time_limit_min"] = params["l1_time_limit_min"]

    return std


# ═══════════════════════════════════════════════════════════════════════
#  Build param snapshot with split keys
# ═══════════════════════════════════════════════════════════════════════
def _build_split_snapshot(std_params):
    """Build param snapshot: standard globals from tgc + split keys for simulator.

    Defense-in-depth: reads enable_<s> flags from std_params and forces gap=9999
    for ANY disabled strategy. This is independent of set_strategy_params — even if
    that function resets gaps to defaults, the snapshot enforces the correct state.
    """
    from optimize_combined import _build_param_snapshot, ALL_STRATS
    snapshot = _build_param_snapshot()

    # ── Gap override map (strategy letter → snapshot key) ──
    _gap_keys = {
        "h": "H_MIN_GAP_PCT", "a": "A_MIN_GAP_PCT", "f": "F_MIN_GAP_PCT",
        "d": "D_MIN_GAP_PCT", "v": "V_MIN_GAP_PCT", "p": "P_MIN_GAP_PCT",
        "m": "M_MIN_GAP_PCT", "o": "O_MIN_GAP_PCT", "b": "B_MIN_GAP_PCT",
        "k": "K_MIN_GAP_PCT", "c": "C_MIN_GAP_PCT", "s": "S_MIN_GAP_PCT",
        "e": "E_MIN_GAP_PCT", "i": "I_MIN_GAP_PCT", "j": "J_MIN_GAP_PCT",
        "n": "N_MIN_GAP_PCT", "w": "W_MIN_GAP_PCT",
    }

    # Force gap=9999 for every disabled strategy (reads enable_<s> from params)
    for s in ALL_STRATS:
        enabled = std_params.get(f"enable_{s}", False)
        if not enabled:
            if s == "r":
                snapshot["R_DAY1_MIN_GAP"] = 9999.0
            elif s in _gap_keys:
                snapshot[_gap_keys[s]] = 9999.0

    # Add all g1_*/g2_*/l1_*/l2_* keys to the snapshot
    # The simulator's entry_params will pick these up at trade entry
    for k, v in std_params.items():
        if k.startswith(("g1_", "g2_", "l1_", "l2_")):
            snapshot[k] = v

    return snapshot


# ═══════════════════════════════════════════════════════════════════════
#  Objective function
# ═══════════════════════════════════════════════════════════════════════
def objective_split(trial, daily_picks, all_dates):
    """Single full-backtest objective for split-param G+L study.

    Same structure as objective() in optimize_combined.py (non-CV path).
    Score: total_pnl × min(pf, 3.0)
    """
    assert tgc.USE_DYNAMIC_SLIPPAGE and tgc.USE_MULTIWINDOW_SLIPPAGE, \
        "Slippage parity violation: both USE_DYNAMIC_SLIPPAGE and USE_MULTIWINDOW_SLIPPAGE must be True"

    params = suggest_gl_split_params(trial)
    std_params = _map_split_to_standard(params)

    with _param_lock:
        set_strategy_params(std_params)
        tgc.USE_DYNAMIC_SLIPPAGE = True
        tgc.USE_MULTIWINDOW_SLIPPAGE = True
        snapshot = _build_split_snapshot(std_params)

    # ── Safety check: disabled strategies MUST have gap=9999 ──
    _gap_keys_check = {
        "h": "H_MIN_GAP_PCT", "a": "A_MIN_GAP_PCT", "f": "F_MIN_GAP_PCT",
        "d": "D_MIN_GAP_PCT", "v": "V_MIN_GAP_PCT", "p": "P_MIN_GAP_PCT",
        "m": "M_MIN_GAP_PCT", "o": "O_MIN_GAP_PCT", "b": "B_MIN_GAP_PCT",
        "k": "K_MIN_GAP_PCT", "c": "C_MIN_GAP_PCT", "s": "S_MIN_GAP_PCT",
        "e": "E_MIN_GAP_PCT", "i": "I_MIN_GAP_PCT", "j": "J_MIN_GAP_PCT",
        "n": "N_MIN_GAP_PCT", "w": "W_MIN_GAP_PCT", "r": "R_DAY1_MIN_GAP",
    }
    for s in ALL_STRATS:
        if not std_params.get(f"enable_{s}", False):
            gap_key = _gap_keys_check.get(s)
            if gap_key and snapshot.get(gap_key, 9999) < 9999:
                raise RuntimeError(
                    f"SNAPSHOT BUG: {s} is disabled but {gap_key}="
                    f"{snapshot[gap_key]} < 9999 in snapshot!"
                )

    def _safe(x, default=-9.9e12):
        import math
        try: x = float(x)
        except (TypeError, ValueError): return default
        if math.isnan(x) or math.isinf(x): return default
        return max(-9.9e12, min(9.9e12, x))

    result = run_combined_backtest(daily_picks, all_dates, params_snapshot=snapshot)

    n = result["n"]
    if n < 100:
        return -9999

    pf = result["pf"]
    if pf < 0.5:
        return -9999

    total_pnl = result["total_pnl"]
    score = _safe(total_pnl * min(pf, 3.0), -9999)

    # Per-strategy breakdown
    for s, v in result.get("strats", {}).items():
        sn = v.get("n", 0)
        trial.set_user_attr(f"{s}_n", sn)
        trial.set_user_attr(f"{s}_pnl", round(_safe(v.get("pnl", 0)), 2))
        wr_s = v.get("wins", 0) / sn * 100 if sn > 0 else 0
        trial.set_user_attr(f"{s}_wr", round(_safe(wr_s), 1))

    # G/L only — always enabled
    trial.set_user_attr("priority", "G>L")
    trial.set_user_attr("enabled", "G,L")
    trial.set_user_attr("n_strategies", 2)
    trial.set_user_attr("total_pnl", round(_safe(total_pnl), 2))
    trial.set_user_attr("pf", round(_safe(pf), 3))
    trial.set_user_attr("wr", round(_safe(result["wr"]), 1))
    trial.set_user_attr("n", n)
    trial.set_user_attr("equity", round(_safe(result["equity"]), 2))

    return score


# ═══════════════════════════════════════════════════════════════════════
#  Dump best params (split-param aware)
# ═══════════════════════════════════════════════════════════════════════
BEST_PARAMS_FILE = "config/trial_gl_split_v2_best.json"


def dump_best_split(trial, params_out=None):
    """Save best trial's params to JSON, preserving split keys."""
    out_path = params_out or BEST_PARAMS_FILE
    bp = dict(trial.params)
    # Ensure enable bits are correct
    bp["enable_g"] = True
    bp["enable_l"] = True

    ua = trial.user_attrs
    data = {
        "trial_number": trial.number,
        "score": trial.value,
        "total_pnl": ua.get("total_pnl", 0),
        "pf": ua.get("pf", 0),
        "wr": ua.get("wr", 0),
        "trades": ua.get("n", 0),
        "equity": ua.get("equity", 0),
        "enabled": "G,L",
        "n_strategies": 2,
        "priority": "G>L",
        "per_strategy": {},
        "params": bp,
    }

    for strat in ["G", "L"]:
        sn = ua.get(f"{strat}_n", 0)
        if sn > 0:
            data["per_strategy"][strat] = {
                "trades": sn,
                "wr": ua.get(f"{strat}_wr", 0),
                "pnl": ua.get(f"{strat}_pnl", 0),
            }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"      -> Params saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    global BEST_PARAMS_FILE

    parser = argparse.ArgumentParser(description="G+L Split-Param Optuna Worker")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--startup-trials", type=int, default=200)
    parser.add_argument("--dump-best", action="store_true",
                        help="Extract best params from existing DB and exit")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--params-out", default=DEFAULT_PARAMS_OUT)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dynamic-slip", action="store_true")
    parser.add_argument("--date-start", default="2024-01-01")
    parser.add_argument("--date-end", default="2026-02-28")
    args = parser.parse_args()

    BEST_PARAMS_FILE = args.params_out

    # Configure slippage
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True

    # PostgreSQL storage
    is_postgres = args.db.startswith("postgresql://") or args.db.startswith("postgres://")
    if is_postgres:
        rdb = optuna.storages.RDBStorage(url=args.db)
    else:
        import sqlite3
        from sqlalchemy import event
        # Strip sqlite:/// prefix if present
        db_path = args.db
        if db_path.startswith("sqlite:///"):
            db_path = db_path[len("sqlite:///"):]
        _bootstrap_conn = sqlite3.connect(db_path, timeout=60)
        try:
            _bootstrap_conn.execute("PRAGMA journal_mode=WAL")
            _bootstrap_conn.execute("PRAGMA synchronous=NORMAL")
            _bootstrap_conn.commit()
        finally:
            _bootstrap_conn.close()
        rdb = optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path}",
            engine_kwargs={"connect_args": {"timeout": 120}},
        )
        @event.listens_for(rdb.engine, "connect")
        def _set_busy_timeout(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA busy_timeout=120000")
            cur.close()

    # ── Dump best mode ──
    if args.dump_best:
        if not is_postgres and not os.path.exists(db_path if not is_postgres else args.db):
            print(f"ERROR: {args.db} not found.")
            sys.exit(1)
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study,
            storage=rdb,
            load_if_exists=True,
        )
        if len(study.trials) == 0:
            print("No completed trials in DB.")
            sys.exit(1)
        best = study.best_trial
        dump_best_split(best, args.params_out)
        print(f"\nBest trial {best.number}:")
        print(f"  Score: {best.value:,.0f}")
        ua = best.user_attrs
        print(f"  PnL: ${ua.get('total_pnl',0):,.0f} | PF: {ua.get('pf',0):.2f} | "
              f"WR: {ua.get('wr',0):.1f}% | Trades: {ua.get('n',0)}")
        sys.exit(0)

    # ── Load data ──
    print("Loading data...", flush=True)
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    all_dates = [d for d in all_dates if args.date_start <= d <= args.date_end]
    print(f"  Date range: {args.date_start} to {args.date_end} ({len(all_dates)} days)")

    # ── Create / resume study ──
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study,
        storage=rdb,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=args.startup_trials),
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"\n  Resuming: {n_existing} existing trials")
        try:
            best = study.best_trial
            print(f"  Current best: score={best.value:,.0f}, "
                  f"PnL=${best.user_attrs.get('total_pnl', 0):,.0f}")
        except Exception:
            print(f"  (No complete trials yet)")

    print(f"\n  Starting optimization ({args.trials} trials)...\n")
    start_time = time.time()

    # ── Run ──
    objective_fn = lambda trial: objective_split(
        trial, daily_picks, all_dates
    )

    study.optimize(
        objective_fn,
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        callbacks=[make_callback(start_time)],
    )

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Optimization complete: {total_time/60:.1f} min")
    print(f"{'='*70}")

    best = study.best_trial
    print(f"\n  Best trial: #{best.number}")
    print(f"  Score:      {best.value:,.0f}")
    ua = best.user_attrs
    print(f"  PnL: ${ua.get('total_pnl',0):,.0f} | PF: {ua.get('pf',0):.3f} | "
          f"WR: {ua.get('wr',0):.1f}% | Trades: {ua.get('n',0)}")

    # Show split params
    bp = best.params
    print(f"\n  --- G1 exit (1st trade) ---")
    for k in ["g1_target_pct", "g1_stop_pct", "g1_trail_pct", "g1_trail_activate_pct", "g1_time_limit_min"]:
        print(f"    {k}: {bp.get(k, '?')}")
    print(f"\n  --- G2 exit (2nd+ trades) ---")
    for k in ["g2_target_pct", "g2_stop_pct", "g2_trail_pct", "g2_trail_activate_pct", "g2_time_limit_min"]:
        print(f"    {k}: {bp.get(k, '?')}")
    print(f"\n  --- L1 exit (1st trade) ---")
    for k in ["l1_stop_pct", "l1_trail_pct", "l1_trail_activate_pct", "l1_time_limit_min",
              "l1_tier1_target1_pct", "l1_tier1_target2_pct"]:
        print(f"    {k}: {bp.get(k, '?')}")
    print(f"\n  --- L2 exit (2nd+ trades) ---")
    for k in ["l2_stop_pct", "l2_trail_pct", "l2_trail_activate_pct", "l2_time_limit_min",
              "l2_tier1_target1_pct", "l2_tier1_target2_pct"]:
        print(f"    {k}: {bp.get(k, '?')}")

    dump_best_split(best, args.params_out)
    print(f"\nFull params written to {args.params_out}")


if __name__ == "__main__":
    main()
