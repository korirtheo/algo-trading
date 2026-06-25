"""W21b v2: same 5-param exit tune as original W21b but with FULL training data.

The original W21b's DATA_DIRS only had ~41 days of 2024 (Jan-Feb from stored_data_combined).
~210 days of Mar-Dec 2024 were missing from training — those quarter dirs were
never added to DATA_DIRS. Discovered 2026-06-24 during W21c v2 work.

This re-run uses the FULL 2024 + 2025 data (matching the apples-to-apples comparison
that confirmed #511 wins). Search space identical to W21b: g_target_pct 4-100,
g_trail 0-10, g_trail_activate 0-15, g_time 3-45, g_stop 5-25.

Training window: 2024-01-01 to 2026-02-28 (same as W21b, single window).
Held out (true OOS): 2026-03-01 to 2026-06-24.

Objective: total_pnl * min(pf, 3) — same legacy PF objective as W21b.

Expected: TPE may surface a different basin than W21b's #511 once it sees the
full 9 missing months of 2024 data. If #511 still emerges or its basin still
wins, that's strong confirmation. If something else, comparison time.
"""
import json
import os
import sys
import time as _time
import numpy as np
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, '.')

STARTING_CASH = 25_000
ALL_STRATS = ["h", "g", "a", "f", "d", "v", "p", "m", "r", "w", "o", "b", "k", "c", "s", "e", "i", "j", "n", "l", "x"]

# FIX: include all 2024 + 2025 quarter dirs (W21b v1 only had stored_data_combined which has ~41 days of 2024)
DATA_DIRS = [
    "stored_data_combined",          # 2024-01-02 -> 2026-02-27 (mostly 2025 + Jan-Feb 2024 + Jan-Feb 2026)
    "stored_data",                   # Jan-Feb 2026
    "stored_data_jan_mar_2024",      # March 2024
    "stored_data_apr_jun_2024",      # Apr-Jun 2024
    "stored_data_jul_sep_2024",      # Jul-Sep 2024
    "stored_data_oct_dec_2024",      # Oct-Dec 2024
    "stored_data_jan_mar_2025",      # Q1 2025
    "stored_data_apr_jun_2025",      # Q2 2025
    "stored_data_jul_2025",          # Jul 2025
    "stored_data_oos",               # Aug-Dec 2025
]

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

with open("config/trial_w13_1202_deploy.json") as f:
    p_dep = json.load(f)["params"]
with open("config/trial_432_params.json") as f:
    baseline = json.load(f)

# Engine config (same as W21b)
tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0
tgc.NEWS_MODULATOR_ENABLED = False

# Load picks ONCE at startup (cached pickles, fast)
print("Loading training data...")
dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
train_dates = sorted([d for d in all_dates if "2024-01-01" <= d <= "2026-02-28"])
print(f"Training window: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")


def run_backtest(params):
    """Run W13 #1202 base + tunable target/trail on FULL training window."""
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
        return None
    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
    losses = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0))
    pf = wins / losses if losses > 0 else 99.0
    wr = sum(1 for t in all_trades if t["pnl"] > 0) / n * 100
    return {"n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr, "equity": cash}


def objective(trial):
    g_target = trial.suggest_float("g_target_pct", 4.0, 100.0, step=1.0)
    g_trail = trial.suggest_float("g_trail_pct", 0.0, 10.0, step=0.5)
    g_trail_act = trial.suggest_float("g_trail_activate_pct", 0.0, 15.0, step=0.5)
    g_time = trial.suggest_int("g_time_limit_min", 3, 45, step=3)
    g_stop = trial.suggest_float("g_stop_pct", 5.0, 25.0, step=1.0)
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
    if score != score:
        return -9999
    return max(-9.9e12, min(9.9e12, score))


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_w21b_v2")
    study = optuna.create_study(
        direction="maximize",
        study_name="w21b_v2_full_2024",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=80),
    )
    print(f"\nW21b v2 starting (currently {len(study.trials)} trials)...")

    # Baseline #511 reference
    p511 = {"g_target_pct": 62.0, "g_stop_pct": 25.0, "g_time_limit_min": 12,
            "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}
    print(f"\nW21b #511 reference (on FULL 2024+2025 training):")
    base_result = run_backtest(p511)
    if base_result:
        base_score = base_result["total_pnl"] * min(base_result["pf"], 3.0)
        print(f"  total_pnl=${base_result['total_pnl']:,.0f}  pf={base_result['pf']:.2f}  "
              f"wr={base_result['wr']:.1f}%  n={base_result['n']}  score=${base_score:,.0f}")

    # Ask/tell loop
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
        if (i + 1) % 10 == 0:
            elapsed = _time.time() - t_start
            try:
                best_val = study.best_value
                best_t = study.best_trial
                best_params = {k: round(v, 1) if isinstance(v, float) else v for k, v in best_t.params.items()}
                print(f"  [{i+1}/400] elapsed {elapsed/60:.1f}m | best ${best_val:,.0f} (#{best_t.number}) {best_params}")
            except Exception:
                print(f"  [{i+1}/400] elapsed {elapsed/60:.1f}m")

    best = study.best_trial
    print(f"\n*** W21b v2 BEST TRIAL #{best.number} ***")
    print(f"  Score: ${best.value:,.0f}")
    for k, v in best.params.items():
        print(f"  {k} = {v}")
    print(f"  n={best.user_attrs.get('n')}  pf={best.user_attrs.get('pf')}  "
          f"wr={best.user_attrs.get('wr')}%  total_pnl=${best.user_attrs.get('total_pnl'):,.0f}")
