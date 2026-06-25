"""Optuna search: news + 1st green alt-entry on #511's no-trade days.

Tunable params:
  - min_news_articles: 1-10   (PIT articles before 9:30, has_catalyst always required)
  - g_target_pct:      4-100, step 1
  - g_stop_pct:        5-30,  step 1
  - g_time_limit_min:  3-60,  step 3
  - g_trail_pct:       0.5-10, step 0.5
  - g_trail_activate_pct: 0-15, step 0.5

Objective: total_pnl * min(pf, 3) across all qualifying trades on no-trade days.
Hard floor: pf < 0.5 or total_pnl <= 0 -> -9999.

Pre-computes the no-trade-day candidate set ONCE per worker startup (slow, ~3 min)
then each trial just iterates the candidates and runs the exit logic.
"""
import json
import os
import sys
import time as _time
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, '.')

STARTING_CASH = 25_000
POSITION_PCT = 0.30
BASELINE_PATH = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h", "g", "a", "f", "d", "v", "p", "m", "r", "w", "o", "b", "k", "c", "s", "e", "i", "j", "n", "l", "x"]
DATA_DIRS = [
    "stored_data_2022",
    "stored_data_combined",
    "stored_data_jan_mar_2024",
    "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024",
    "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025",
    "stored_data_apr_jun_2025",
    "stored_data_jul_2025",
    "stored_data_oos",
    "stored_data",
    "stored_data_mar_may_2026",
    "stored_data_jun_2026",
    "stored_data_2026_gap_fill",
]
DATE_LO = "2022-01-01"
DATE_HI = "2026-06-24"


def _simulate_trade(entry_price, bars_after, target_pct, stop_pct, time_min, trail_pct, trail_act_pct):
    if entry_price <= 0:
        return None
    target = entry_price * (1 + target_pct / 100)
    stop   = entry_price * (1 - stop_pct / 100)
    peak   = entry_price
    trail_stop = None
    max_bars = max(1, time_min // 2)
    for i, (ts, row) in enumerate(bars_after.iterrows()):
        if i >= max_bars:
            return float(row["Close"])
        c_high = float(row["High"]); c_low = float(row["Low"])
        if c_high >= target:
            return target
        if c_low <= stop:
            return stop
        if c_high > peak:
            peak = c_high
        # Arm trail when unrealized >= trail_act_pct
        unrealized = (peak / entry_price - 1) * 100
        if unrealized >= trail_act_pct:
            new_trail = peak * (1 - trail_pct / 100)
            if trail_stop is None or new_trail > trail_stop:
                trail_stop = new_trail
        if trail_stop is not None and c_low <= trail_stop:
            return trail_stop
    return float(bars_after.iloc[-1]["Close"]) if len(bars_after) else None


# ---- Pre-compute candidates once ----
print("Pre-computing no-trade-day candidates...")
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from news_filter import count_pit

with open(BASELINE_PATH) as f:
    baseline_p = json.load(f)
with open(W21B_DEPLOY) as f:
    p511 = json.load(f)["params"]
merged = {**baseline_p, **p511}
for s in ALL_STRATS:
    merged[f"enable_{s}"] = (s in {"g", "l"})
set_strategy_params(merged)
tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.NEWS_MODULATOR_ENABLED = False

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])

cash = STARTING_CASH
days_with_trades = set()
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
            days_with_trades.add(d); break
    cash = end_c + (unset if is_cash else 0)
no_trade_days = [d for d in dates if d not in days_with_trades]
print(f"  No-trade days: {len(no_trade_days)} / {len(dates)}")

# Build candidate list: for each no-trade day, for each catalyst ticker, capture (n_articles, entry_idx, entry_price, bars_after)
CANDIDATES = []  # list of (n_articles, entry_price, bars_after_df)
for d in no_trade_days:
    for p in picks_by_date.get(d, []):
        n, has_cat = count_pit(p["ticker"], d)
        if not has_cat:
            continue
        mh = p.get("market_hour_candles")
        if mh is None or len(mh) < 2:
            continue
        # 1st green entry
        entry_idx = None
        entry_price = None
        for i, (ts, row) in enumerate(mh.iterrows()):
            if float(row["Close"]) > float(row["Open"]):
                entry_idx = i
                entry_price = float(row["Close"])
                break
        if entry_idx is None or entry_price <= 0:
            continue
        bars_after = mh.iloc[entry_idx + 1:]
        if len(bars_after) == 0:
            continue
        CANDIDATES.append((n, entry_price, bars_after))
print(f"  Pre-computed candidates: {len(CANDIDATES)}")


def objective(trial):
    min_articles = trial.suggest_int("min_news_articles", 1, 10)
    target = trial.suggest_float("g_target_pct", 4.0, 100.0, step=1.0)
    stop = trial.suggest_float("g_stop_pct", 5.0, 30.0, step=1.0)
    time_min = trial.suggest_int("g_time_limit_min", 3, 60, step=3)
    trail = trial.suggest_float("g_trail_pct", 0.5, 10.0, step=0.5)
    trail_act = trial.suggest_float("g_trail_activate_pct", 0.0, 15.0, step=0.5)

    total_pnl = 0.0; wins = 0.0; losses = 0.0; n_trades = 0
    pos_cost = STARTING_CASH * POSITION_PCT
    for n_articles, entry, bars_after in CANDIDATES:
        if n_articles < min_articles:
            continue
        exit_p = _simulate_trade(entry, bars_after, target, stop, time_min, trail, trail_act)
        if exit_p is None:
            continue
        shares = pos_cost / entry
        pnl = shares * (exit_p - entry)
        total_pnl += pnl
        if pnl > 0:
            wins += pnl
        else:
            losses += -pnl
        n_trades += 1

    trial.set_user_attr("n", n_trades)
    trial.set_user_attr("total_pnl", round(total_pnl, 0))
    if n_trades < 20 or total_pnl <= 0:
        return -9999
    pf = wins / losses if losses > 0 else 99.0
    if pf < 0.5:
        return -9999
    trial.set_user_attr("pf", round(pf, 3))
    score = total_pnl * min(pf, 3.0)
    if score != score:
        return -9999
    return max(-9.9e12, min(9.9e12, score))


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_news_alt")
    study = optuna.create_study(
        direction="maximize",
        study_name="news_alt_1st_green",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=50),
    )
    print(f"\nW news-alt study starting (currently {len(study.trials)} trials)")
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
        if (i + 1) % 25 == 0:
            elapsed = _time.time() - t_start
            try:
                best_val = study.best_value
                best_t = study.best_trial
                bp = {k: round(v, 1) if isinstance(v, float) else v for k, v in best_t.params.items()}
                print(f"  [{i+1}/400] elapsed {elapsed/60:.1f}m | best ${best_val:,.0f} (#{best_t.number}) {bp}")
            except Exception:
                print(f"  [{i+1}/400] elapsed {elapsed/60:.1f}m")

    best = study.best_trial
    print(f"\n*** BEST TRIAL #{best.number} ***")
    print(f"  Score: ${best.value:,.0f}")
    for k, v in best.params.items():
        print(f"  {k} = {v}")
    print(f"  n_trades = {best.user_attrs.get('n')}, total_pnl = ${best.user_attrs.get('total_pnl'):,.0f}, pf = {best.user_attrs.get('pf')}")
