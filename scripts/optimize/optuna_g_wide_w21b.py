"""G-only Optuna with W21b-style WIDE param ranges.

Fixes the prior G-only Optuna's narrow bounds (target 4-20, stop 5-12) which
excluded the deploy #511's actual params (target=62, stop=25). Uses W21b's
extended ranges so the optimizer can find #511-style basins.

Entry params FIXED at #511's confirmed best (2nd_green=True, 2nd_new_high=False, min_gap=15)
to focus the search on exits — matches W21b's "exit logic tuning on W13 entry conditions" approach.

Tunable (5 dims, wide ranges):
  g_target_pct       4 - 100 step 1
  g_stop_pct         5 - 30  step 1
  g_time_limit_min   3 - 90  step 3
  g_trail_pct        0.5 - 10 step 0.5
  g_trail_activate_pct 0 - 15 step 0.5

n_startup_trials=150 per project rule (>=4×dims, plus comfort margin).
"""
import json, os, sys, time as _time
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, '.')

STARTING_CASH = 25_000
BASELINE_PATH = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = [
    "stored_data_combined", "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024", "stored_data_jan_mar_2025",
    "stored_data_apr_jun_2025", "stored_data_jul_2025", "stored_data_oos", "stored_data",
]
DATE_LO = "2024-01-01"
DATE_HI = "2026-02-28"

print("Step 1: configure simulator + load data...")
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

with open(BASELINE_PATH) as f: baseline_p = json.load(f)
with open(W21B_DEPLOY) as f: p511 = json.load(f)["params"]

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
DATES = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
print(f"  Trading days: {len(DATES)}")


def _setup_for_trial(g_target, g_stop, g_time, g_trail, g_trail_act):
    """Build params with #511 G entry conditions + trial's exit values; G only enabled."""
    p = {**baseline_p, **p511}
    # Fix entry conditions to #511 confirmed-best
    p["g_min_gap_pct"] = 15.0
    p["g_require_2nd_green"] = True
    p["g_require_2nd_new_high"] = False
    # Override exit params from trial
    p["g_target_pct"] = g_target
    p["g_stop_pct"] = g_stop
    p["g_time_limit_min"] = int(g_time)
    p["g_trail_pct"] = g_trail
    p["g_trail_activate_pct"] = g_trail_act
    # G only — no L, no others
    for s in ALL_STRATS:
        p[f"enable_{s}"] = (s == "g")
    set_strategy_params(p)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False
    tgc.MARGIN_MULTIPLIER = 1.0


def objective(trial):
    g_target = trial.suggest_float("g_target_pct", 4.0, 100.0, step=1.0)
    g_stop = trial.suggest_float("g_stop_pct", 5.0, 30.0, step=1.0)
    g_time = trial.suggest_int("g_time_limit_min", 3, 90, step=3)
    g_trail = trial.suggest_float("g_trail_pct", 0.5, 10.0, step=0.5)
    g_trail_act = trial.suggest_float("g_trail_activate_pct", 0.0, 15.0, step=0.5)

    _setup_for_trial(g_target, g_stop, g_time, g_trail, g_trail_act)
    cash = STARTING_CASH
    n = 0; total_pnl = 0.0; wins = 0.0; losses = 0.0
    for d in DATES:
        dp = picks_by_date.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception: continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
                n += 1
                p = float(st.get("pnl") or 0); total_pnl += p
                if p > 0: wins += p
                else: losses += -p
        cash = end_c + (unset if is_cash else 0)
    trial.set_user_attr("n", n)
    trial.set_user_attr("total_pnl", round(total_pnl, 0))
    if n < 50 or total_pnl <= 0: return -9999
    pf = wins / losses if losses > 0 else 99.0
    if pf < 0.5: return -9999
    trial.set_user_attr("pf", round(pf, 3))
    trial.set_user_attr("final_cash", round(cash, 0))
    score = total_pnl * min(pf, 3.0)
    if not (score == score and score < 1e15): return -9999
    return max(-9.9e12, min(9.9e12, score))


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_g_wide")
    study = optuna.create_study(
        direction="maximize",
        study_name="g_wide_w21b",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=150),
    )
    print(f"\nStudy starting (currently {len(study.trials)} trials)")
    t_start = _time.time()
    for i in range(500):
        trial = study.ask()
        try:
            value = objective(trial)
            study.tell(trial, value)
        except Exception:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        if (i + 1) % 25 == 0:
            elapsed = _time.time() - t_start
            try:
                bv = study.best_value
                bt = study.best_trial
                bp = {k: round(v,1) if isinstance(v,float) else v for k,v in bt.params.items()}
                print(f"  [{i+1}/500] elapsed {elapsed/60:.1f}m | best ${bv:,.0f} (#{bt.number}) {bp}")
            except: print(f"  [{i+1}/500] elapsed {elapsed/60:.1f}m")

    best = study.best_trial
    print(f"\n*** BEST TRIAL #{best.number} ***")
    print(f"  Score: ${best.value:,.0f}")
    for k, v in best.params.items(): print(f"  {k} = {v}")
    for k, v in best.user_attrs.items(): print(f"  {k} = {v}")
