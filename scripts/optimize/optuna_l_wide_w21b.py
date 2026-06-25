"""L-only Optuna with EXTENDED param ranges (beyond the legacy defaults).

The prior L-only Optuna (which found #626) had several params at edges of search:
  - l_max_float at upper bound (20M)
  - l_vol_surge_mult at lower bound (1.0)
  - l_min_price_accel_pct at lower bound (0.5)
  - l_tier1_target1_pct at lower bound (15)
  - l_trail_pct at lower bound (1.0)
This re-run extends each by 1.5-3× to see if a better basin exists outside.

Tunable: 19 L params, n_startup_trials=200 (>=4×dims plus safety margin).
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


def _setup_for_trial(p_trial):
    p = {**baseline_p, **p511}
    p.update(p_trial)
    for s in ALL_STRATS:
        p[f"enable_{s}"] = (s == "l")  # L only
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
    # Extended ranges (vs the legacy in optimize_combined.py L block)
    p_trial = {
        "l_min_gap": trial.suggest_int("l_min_gap", 5, 100, step=5),
        "l_max_float": trial.suggest_int("l_max_float", 5_000_000, 50_000_000, step=5_000_000),
        "l_earliest_candle": trial.suggest_int("l_earliest_candle", 1, 30, step=3),
        "l_latest_candle": trial.suggest_int("l_latest_candle", 30, 240, step=15),
        "l_vol_surge_mult": trial.suggest_float("l_vol_surge_mult", 1.0, 5.0, step=0.5),
        "l_min_price_accel_pct": trial.suggest_float("l_min_price_accel_pct", 0.0, 5.0, step=0.5),
        "l_tier1_float": trial.suggest_int("l_tier1_float", 250_000, 3_000_000, step=250_000),
        "l_tier2_float": trial.suggest_int("l_tier2_float", 2_500_000, 10_000_000, step=500_000),
        "l_tier1_target1_pct": trial.suggest_float("l_tier1_target1_pct", 5.0, 80.0, step=5.0),
        "l_tier1_target2_pct": trial.suggest_float("l_tier1_target2_pct", 20.0, 100.0, step=5.0),
        "l_tier2_target1_pct": trial.suggest_float("l_tier2_target1_pct", 5.0, 50.0, step=2.0),
        "l_tier2_target2_pct": trial.suggest_float("l_tier2_target2_pct", 15.0, 80.0, step=5.0),
        "l_tier3_target1_pct": trial.suggest_float("l_tier3_target1_pct", 3.0, 30.0, step=1.0),
        "l_tier3_target2_pct": trial.suggest_float("l_tier3_target2_pct", 10.0, 50.0, step=5.0),
        "l_stop_pct": trial.suggest_float("l_stop_pct", 5.0, 30.0, step=1.0),
        "l_partial_sell_pct": trial.suggest_float("l_partial_sell_pct", 0.0, 75.0, step=25.0),
        "l_trail_pct": trial.suggest_float("l_trail_pct", 0.5, 10.0, step=0.5),
        "l_trail_activate_pct": trial.suggest_float("l_trail_activate_pct", 0.0, 15.0, step=1.0),
        "l_time_limit_min": trial.suggest_int("l_time_limit_min", 10, 240, step=10),
    }

    _setup_for_trial(p_trial)
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
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "L":
                n += 1
                p = float(st.get("pnl") or 0); total_pnl += p
                if p > 0: wins += p
                else: losses += -p
        cash = end_c + (unset if is_cash else 0)
    trial.set_user_attr("n", n)
    trial.set_user_attr("total_pnl", round(total_pnl, 0))
    if n < 30 or total_pnl <= 0: return -9999
    pf = wins / losses if losses > 0 else 99.0
    if pf < 0.5: return -9999
    trial.set_user_attr("pf", round(pf, 3))
    trial.set_user_attr("final_cash", round(cash, 0))
    score = total_pnl * min(pf, 3.0)
    if not (score == score and score < 1e15): return -9999
    return max(-9.9e12, min(9.9e12, score))


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_l_wide")
    study = optuna.create_study(
        direction="maximize",
        study_name="l_wide_w21b",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=200),
    )
    print(f"\nStudy starting (currently {len(study.trials)} trials)")
    t_start = _time.time()
    for i in range(800):
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
                print(f"  [{i+1}/800] elapsed {elapsed/60:.1f}m | best ${bv:,.0f} (#{bt.number})")
            except: print(f"  [{i+1}/800] elapsed {elapsed/60:.1f}m")

    best = study.best_trial
    print(f"\n*** BEST TRIAL #{best.number} ***  Score: ${best.value:,.0f}")
    for k, v in best.params.items(): print(f"  {k} = {v}")
    for k, v in best.user_attrs.items(): print(f"  {k} = {v}")
