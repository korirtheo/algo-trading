"""W21: Narrow Optuna study tuning only G's exit-logic params.

Holds W13 #1202 base FIXED for entry conditions. Five tunable params:
  - g_target_pct         (4-40, step 1)
  - g_trail_pct          (0-10, step 0.5)
  - g_trail_activate_pct (0-15, step 0.5)
  - g_time_limit_min     (3-45, step 3)
  - g_stop_pct           (5-25, step 1) — FLOOR at 5 (no no-stop configs)

Objective: total_pnl x min(pf, 3) on 2024-01 to 2026-02 training window.
Blind OOS: Mar-Jun 2026 (verified after Optuna finishes).

Why: W13 #1202's 76% of Mar-Jun G trades exit via TRAIL at +1.4% avg —
trail activates immediately (g_trail_activate=0) and trails 1% from peak,
cutting winners early. With 65-78% WR we have risk budget to let winners
ride further.
"""
import json
import os
import sys
import numpy as np
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, '.')

STARTING_CASH = 25_000
ALL_STRATS = ["h", "g", "a", "f", "d", "v", "p", "m", "r", "w", "o", "b", "k", "c", "s", "e", "i", "j", "n", "l", "x"]
DATA_DIRS = ["stored_data_combined", "stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

with open("config/trial_w13_1202_deploy.json") as f:
    p_dep = json.load(f)["params"]
with open("config/trial_432_params.json") as f:
    baseline = json.load(f)

# Apply common simulator settings ONCE
tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0
tgc.NEWS_MODULATOR_ENABLED = False

# Load picks once
dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
train_dates = sorted([d for d in all_dates if "2024-01-01" <= d <= "2026-02-28"])
print(f"Training window: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")


def run_backtest(params):
    """Run W13 #1202 base + tunable target/trail on training window."""
    merged = {**baseline, **p_dep, **params}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)
    cash = STARTING_CASH
    all_trades = []
    for d in train_dates:
        dp = picks_by_date.get(d, [])
        if not dp:
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                all_trades.append({
                    "strategy": st.get("strategy"),
                    "pnl": st["pnl"], "position_cost": st["position_cost"],
                    "exit_reason": st.get("exit_reason"),
                })
        cash = end_c + (unset if is_cash else 0)

    n = len(all_trades)
    if n < 100:
        return None  # too few trades
    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
    losses = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0))
    pf = wins / losses if losses > 0 else 99.0
    wr = sum(1 for t in all_trades if t["pnl"] > 0) / n * 100
    return {"n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr, "equity": cash}


def objective(trial):
    g_target = trial.suggest_float("g_target_pct", 4.0, 40.0, step=1.0)
    g_trail = trial.suggest_float("g_trail_pct", 0.0, 10.0, step=0.5)
    g_trail_act = trial.suggest_float("g_trail_activate_pct", 0.0, 15.0, step=0.5)
    g_time = trial.suggest_int("g_time_limit_min", 3, 45, step=3)
    g_stop = trial.suggest_float("g_stop_pct", 5.0, 25.0, step=1.0)  # FLOOR at 5
    params = {
        "g_target_pct": g_target,
        "g_trail_pct": g_trail,
        "g_trail_activate_pct": g_trail_act,
        "g_time_limit_min": g_time,
        "g_stop_pct": g_stop,
    }
    result = run_backtest(params)
    if result is None:
        return -9999
    trial.set_user_attr("n", result["n"])
    trial.set_user_attr("pf", round(result["pf"], 3))
    trial.set_user_attr("wr", round(result["wr"], 1))
    trial.set_user_attr("total_pnl", round(result["total_pnl"], 0))
    trial.set_user_attr("equity", round(result["equity"], 0))
    if result["pf"] < 0.5:
        return -9999
    score = result["total_pnl"] * min(result["pf"], 3.0)
    if score != score:  # NaN check
        return -9999
    return max(-9.9e12, min(9.9e12, score))


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_w21")
    study = optuna.create_study(
        direction="maximize",
        study_name="w21_g_target_trail",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=50),
    )
    print(f"Starting W21 study (currently {len(study.trials)} trials)...")
    print(f"W13 #1202 BASELINE for reference:")
    print(f"  g_target_pct         = {p_dep.get('g_target_pct')}")
    print(f"  g_trail_pct          = {p_dep.get('g_trail_pct')}")
    print(f"  g_trail_activate_pct = {p_dep.get('g_trail_activate_pct')}")
    print(f"  g_time_limit_min     = {p_dep.get('g_time_limit_min')}")
    print(f"  g_stop_pct           = {p_dep.get('g_stop_pct')}")

    # Run W13 baseline first as reference
    baseline_result = run_backtest({
        "g_target_pct": p_dep.get("g_target_pct"),
        "g_trail_pct": p_dep.get("g_trail_pct"),
        "g_trail_activate_pct": p_dep.get("g_trail_activate_pct"),
        "g_time_limit_min": p_dep.get("g_time_limit_min"),
        "g_stop_pct": p_dep.get("g_stop_pct"),
    })
    if baseline_result:
        baseline_score = baseline_result["total_pnl"] * min(baseline_result["pf"], 3.0)
        print(f"\nW13 baseline (this objective): score=${baseline_score:,.0f}")
        print(f"  total_pnl=${baseline_result['total_pnl']:,.0f} pf={baseline_result['pf']:.2f} "
              f"wr={baseline_result['wr']:.1f}% n={baseline_result['n']}")

    # Optimize using ask/tell pattern (study.optimize has Win+Postgres compatibility issues)
    # 5 dims, 100 startup, 400 total trials
    study.sampler = TPESampler(n_startup_trials=100)
    import time as _time
    t_start = _time.time()
    for i in range(400):
        trial = study.ask()
        try:
            value = objective(trial)
            study.tell(trial, value)
        except Exception as e:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"  Trial {trial.number} FAILED: {e}")
            continue
        if (i + 1) % 20 == 0:
            elapsed = _time.time() - t_start
            try:
                best_val = study.best_value
            except Exception:
                best_val = 0
            print(f"  [{i+1}/400] elapsed {elapsed/60:.1f}m | best so far: {best_val:,.0f}")

    best = study.best_trial
    print(f"\n*** BEST TRIAL #{best.number} ***")
    print(f"  Score: {best.value:,.0f}")
    print(f"  g_target_pct         = {best.params['g_target_pct']}")
    print(f"  g_trail_pct          = {best.params['g_trail_pct']}")
    print(f"  g_trail_activate_pct = {best.params['g_trail_activate_pct']}")
    print(f"  g_time_limit_min     = {best.params['g_time_limit_min']}")
    print(f"  g_stop_pct           = {best.params['g_stop_pct']}")
    print(f"  n={best.user_attrs.get('n')} pf={best.user_attrs.get('pf')} "
          f"wr={best.user_attrs.get('wr')}% total_pnl=${best.user_attrs.get('total_pnl'):,.0f}")
