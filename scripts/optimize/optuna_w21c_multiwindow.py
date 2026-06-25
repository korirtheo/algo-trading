"""W21c: Multi-window val. Same 5-param exit tune as W21b but score = min(year_pnl * pf) across 3 regimes.

Forces TPE to find exit params that work in BOTH:
  - 2022 (cool small-cap regime, 251 days, 6.9 picks/day)
  - 2024 (hot pump start, ~250 days)
  - 2025 (continued hot, ~250 days)

Held out (true OOS, NEVER touched during W21c):
  - 2026 Mar-Jun (74 days)

Objective per trial:
  per_year_score = total_pnl * min(pf, 3)   (PF cap to prevent gamed PF inflation)
  composite      = min(per_year_score)      (worst-year wins; rewards regime robustness)

If any year loses money -> return -9999.

Search space (same 5 params as W21b):
  - g_target_pct         (4-100, step 1)
  - g_trail_pct          (0-10, step 0.5)
  - g_trail_activate_pct (0-15, step 0.5)
  - g_time_limit_min     (3-45, step 3)
  - g_stop_pct           (5-25, step 1) — FLOOR at 5

Why: W21b found local optimum per regime but couldn't see cross-regime robustness.
W21b #511 wins 2022 by +46% but loses 2026 Mar-Jun by 5%. #561 wins 2026 Mar-Jun
by 8% but loses 2022 by 31%. Neither is robust. W21c objective penalizes worst-case.

Expected outcome: TPE converges on time≈10-11 (between 9 and 12) or finds a
regime-invariant variation in the other params.
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

# Per-year training windows
# 2024 + 2025 data is SPREAD across multiple per-quarter dirs. stored_data_combined
# only has ~41 days of 2024 (Jan-Feb). Must include all quarter dirs.
TRAIN_WINDOWS = [
    {"name": "2022", "data_dirs": ["stored_data_2022"],
     "date_lo": "2022-01-01", "date_hi": "2022-12-31"},
    {"name": "2024", "data_dirs": [
        "stored_data_combined",         # ~Jan-Feb 2024
        "stored_data_jan_mar_2024",     # March 2024
        "stored_data_apr_jun_2024",     # Apr-Jun 2024
        "stored_data_jul_sep_2024",     # Jul-Sep 2024
        "stored_data_oct_dec_2024",     # Oct-Dec 2024
     ], "date_lo": "2024-01-01", "date_hi": "2024-12-31"},
    {"name": "2025", "data_dirs": [
        "stored_data_combined",         # Jan-Feb 2025
        "stored_data_jan_mar_2025",     # Q1 2025
        "stored_data_apr_jun_2025",     # Q2 2025
        "stored_data_jul_2025",         # Jul 2025
        "stored_data_oos",              # Aug-Dec 2025
     ], "date_lo": "2025-01-01", "date_hi": "2025-12-31"},
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

# Pre-load all train windows ONCE (avoids re-loading per trial)
print("Pre-loading train window data...")
WINDOW_PICKS = {}
for w in TRAIN_WINDOWS:
    dirs = [d for d in w["data_dirs"] if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if w["date_lo"] <= d <= w["date_hi"]])
    WINDOW_PICKS[w["name"]] = {"dates": dates, "picks_by_date": picks_by_date}
    print(f"  {w['name']}: {len(dates)} days [{dates[0] if dates else 'NONE'} -> {dates[-1] if dates else 'NONE'}]")


def run_one_window(window_name, params):
    """Run one year-window with fresh $25K cash. Return year-end metrics."""
    wp = WINDOW_PICKS[window_name]
    dates = wp["dates"]
    picks_by_date = wp["picks_by_date"]
    merged = {**baseline, **p_dep, **params}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)
    cash = STARTING_CASH
    all_trades = []
    for d in dates:
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
                all_trades.append({"pnl": st["pnl"], "exit_reason": st.get("exit_reason"), "strategy": st.get("strategy")})
        cash = end_c + (unset if is_cash else 0)
    n = len(all_trades)
    if n == 0:
        return {"n": 0, "total_pnl": 0, "pf": 0, "wr": 0, "equity": cash}
    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
    losses = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0))
    pf = wins / losses if losses > 0 else 99.0
    wr = sum(1 for t in all_trades if t["pnl"] > 0) / n * 100
    return {"n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr, "equity": cash}


def objective(trial):
    params = {
        "g_target_pct": trial.suggest_float("g_target_pct", 4.0, 100.0, step=1.0),
        "g_trail_pct": trial.suggest_float("g_trail_pct", 0.0, 10.0, step=0.5),
        "g_trail_activate_pct": trial.suggest_float("g_trail_activate_pct", 0.0, 15.0, step=0.5),
        "g_time_limit_min": trial.suggest_int("g_time_limit_min", 3, 45, step=3),
        "g_stop_pct": trial.suggest_float("g_stop_pct", 5.0, 25.0, step=1.0),  # FLOOR
    }
    year_scores = []
    for w in TRAIN_WINDOWS:
        r = run_one_window(w["name"], params)
        trial.set_user_attr(f"n_{w['name']}", r["n"])
        trial.set_user_attr(f"pnl_{w['name']}", round(r["total_pnl"], 0))
        trial.set_user_attr(f"pf_{w['name']}", round(r["pf"], 3))
        trial.set_user_attr(f"wr_{w['name']}", round(r["wr"], 1))
        if r["pf"] < 0.5 or r["total_pnl"] < 0:
            return -9999  # hard floor: every year must be profitable with pf >= 0.5
        score = r["total_pnl"] * min(r["pf"], 3.0)
        year_scores.append(score)
    composite = min(year_scores)  # worst-year wins -> rewards robustness
    trial.set_user_attr("min_year_score", round(composite, 0))
    trial.set_user_attr("year_scores", [round(s, 0) for s in year_scores])
    if composite != composite:  # NaN
        return -9999
    return max(-9.9e12, min(9.9e12, composite))


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_w21c")
    study = optuna.create_study(
        direction="maximize",
        study_name="w21c_multiwindow_v2",   # v1 had bad 2024 dirs (Jan-Feb only)
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=80),
    )
    print(f"\nW21c starting (currently {len(study.trials)} trials)...")
    print(f"Objective: min(total_pnl * min(pf, 3)) across {[w['name'] for w in TRAIN_WINDOWS]}")
    print(f"Hard floor: every year pf>=0.5 AND total_pnl>0, else -9999")

    # Baseline reference: W13 #1202 deploy params
    baseline_params = {k: p_dep.get(k) for k in [
        "g_target_pct", "g_stop_pct", "g_time_limit_min", "g_trail_pct", "g_trail_activate_pct"]}
    print(f"\nW13 BASELINE (target=11, stop=12, time=24, trail=1, trail_act=0):")
    base_scores = []
    for w in TRAIN_WINDOWS:
        r = run_one_window(w["name"], baseline_params)
        s = r["total_pnl"] * min(r["pf"], 3.0) if r["pf"] >= 0.5 else -9999
        base_scores.append(s)
        print(f"  {w['name']}: pnl=${r['total_pnl']:>11,.0f}  pf={r['pf']:.2f}  wr={r['wr']:.1f}%  n={r['n']}  score=${s:,.0f}")
    print(f"  composite (min): ${min(base_scores):,.0f}")

    # W21b #511 reference
    p511 = {"g_target_pct": 62.0, "g_stop_pct": 25.0, "g_time_limit_min": 12,
            "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}
    print(f"\nW21b #511 REFERENCE (target=62, stop=25, time=12):")
    p511_scores = []
    for w in TRAIN_WINDOWS:
        r = run_one_window(w["name"], p511)
        s = r["total_pnl"] * min(r["pf"], 3.0) if r["pf"] >= 0.5 else -9999
        p511_scores.append(s)
        print(f"  {w['name']}: pnl=${r['total_pnl']:>11,.0f}  pf={r['pf']:.2f}  wr={r['wr']:.1f}%  n={r['n']}  score=${s:,.0f}")
    print(f"  composite (min): ${min(p511_scores):,.0f}  <- W21c needs to beat this for swap")

    # Ask/tell pattern (Windows+Postgres compat)
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
    print(f"\n*** W21c BEST TRIAL #{best.number} ***")
    print(f"  Composite score: ${best.value:,.0f}")
    for k, v in best.params.items():
        print(f"  {k} = {v}")
    print(f"  Per-year:")
    for w in TRAIN_WINDOWS:
        n = best.user_attrs.get(f"n_{w['name']}", 0)
        pnl = best.user_attrs.get(f"pnl_{w['name']}", 0)
        pf = best.user_attrs.get(f"pf_{w['name']}", 0)
        wr = best.user_attrs.get(f"wr_{w['name']}", 0)
        print(f"    {w['name']}: pnl=${pnl:>11,.0f}  pf={pf:.2f}  wr={wr:.1f}%  n={n}")
