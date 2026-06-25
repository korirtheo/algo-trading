"""Optuna R-O 'any_green_above_open' — option B (apples-to-apples).

Differences from prior R-O Optuna:
  - VOL CAPS APPLIED: imports simulator's `_last_2min_dollar_vol`,
    `_multi_window_effective_volume`, applies VOL_CAP_PCT / MAX_REGIME_PARTICIPATION
    / MAX_2MIN_PARTICIPATION exactly like simulate_day_combined does.
  - DYNAMIC SLIPPAGE: uses `_entry_slip_pct` and `_exit_slip_pct` for fills.
  - EXCLUSION MODE (b): skip R-O entries that fall WITHIN G/L's [entry_bar,
    exit_bar] window on the same (ticker, date). Entries BEFORE or AFTER G/L's
    holding window are allowed (planned engine change will permit concurrent
    positions on the same ticker at different bars).

Tunable 5-param exits:
  g_target_pct 4-100, g_stop_pct 5-30, g_time_limit_min 3-60,
  g_trail_pct 0.5-10, g_trail_activate_pct 0-15

Pre-computes candidates once per worker (~3 min), then each trial iterates
in pure Python (~1-3s/trial).
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
DATE_LO = "2024-01-01"
DATE_HI = "2026-02-28"


# ---- Set up simulator and load data ----
print("Step 1: configure simulator + load data...")
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

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
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.NEWS_MODULATOR_ENABLED = False

# Capture cap values after set_strategy_params (which may have overridden them)
VOL_CAP_PCT = tgc.VOL_CAP_PCT
MAX_REGIME = tgc.MAX_REGIME_PARTICIPATION
MAX_2MIN = tgc.MAX_2MIN_PARTICIPATION

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
print(f"  Trading days: {len(dates)}")

# Run baseline to identify G/L holding windows: (ticker, date) -> list of (entry_ts, exit_ts)
print("Step 2: run #511 baseline to map G/L holding windows...")
gl_holds = {}  # (ticker, date) -> list of (entry_ts, exit_ts)
cash = STARTING_CASH
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
        if st.get("exit_reason") and st.get("position_cost", 0) > 0:
            key = (st.get("ticker"), d)
            etime = st.get("entry_time")
            xtime = st.get("exit_time")
            if etime is not None:
                gl_holds.setdefault(key, []).append((etime, xtime))
    cash = end_c + (unset if is_cash else 0)
print(f"  G/L trade-holds across {len(gl_holds)} (ticker,date) pairs")

# Pre-compute R-O candidates (any_green_above_open) with mode (b) exclusion
print("Step 3: pre-compute R-O candidates with mode-b exclusion...")
CANDIDATES = []  # list of (ticker, date, entry_ts, fill_price, mh, bars_after)
skipped_within_gl = 0
for d in dates:
    for p in picks_by_date.get(d, []):
        mh = p.get("market_hour_candles")
        if mh is None or len(mh) < 2:
            continue
        day_open = float(mh.iloc[0]["Open"])
        # reclaim_after_dip entry: first bar where C > day_open AFTER any earlier bar C <= day_open
        entry_idx = None; entry_price = None; entry_ts = None
        ever_below = False
        for i, (ts, row) in enumerate(mh.iterrows()):
            c = float(row["Close"])
            if ever_below and c > day_open:
                entry_idx = i
                entry_price = c
                entry_ts = ts
                break
            if c <= day_open:
                ever_below = True
        if entry_idx is None or entry_price <= 0:
            continue
        # Mode (b): skip if entry_ts is WITHIN any G/L holding window on this (ticker,date)
        holds = gl_holds.get((p["ticker"], d), [])
        if any(et <= entry_ts <= (xt if xt is not None else et) for et, xt in holds):
            skipped_within_gl += 1
            continue
        bars_after = mh.iloc[entry_idx + 1:]
        if len(bars_after) == 0:
            continue
        CANDIDATES.append((p["ticker"], d, entry_ts, entry_price, mh, bars_after))
print(f"  Pre-computed {len(CANDIDATES)} candidates, skipped {skipped_within_gl} within G/L holds")


def _apply_caps(mh, ts, fill_price, requested):
    """Replicate simulator's vol-cap logic. Returns (capped_trade_size, v_eff_adj_for_slippage)."""
    pre = mh.loc[mh.index <= ts]
    vol_shares = float(pre["Volume"].sum()) if len(pre) > 0 else 0
    dollar_vol = fill_price * vol_shares
    if dollar_vol <= 0:
        return 0, 0
    # Cap 1: 5% of cum $-vol
    vol_limit = dollar_vol * (VOL_CAP_PCT / 100)
    v_eff_adj, _, _, v_regime = tgc._multi_window_effective_volume(mh, ts, fill_price)
    # Cap 2: regime (8% of v_regime)
    if MAX_REGIME > 0 and v_regime > 0:
        vol_limit = min(vol_limit, v_regime * MAX_REGIME)
    # Cap 3: 2-min execution (15% of v_eff_adj)
    if MAX_2MIN > 0 and v_eff_adj > 0:
        vol_limit = min(vol_limit, v_eff_adj * MAX_2MIN)
    return min(requested, vol_limit), v_eff_adj


def _simulate_one_trade(entry_ts, fill_price, mh, bars_after,
                        target_pct, stop_pct, time_min, trail_pct, trail_act_pct,
                        position_dollars):
    """Apply vol cap + entry slippage + exit logic + exit slippage. Returns pnl."""
    capped_size, v_eff_adj = _apply_caps(mh, entry_ts, fill_price, position_dollars)
    if capped_size < 50:
        return 0.0  # too thin — skip silently
    # Entry slippage (buy is hit upward)
    slip_in = tgc._entry_slip_pct(fill_price, capped_size, v_eff_adj)
    actual_entry = fill_price * (1 + slip_in / 100)
    shares = capped_size / actual_entry
    # Walk exits
    target = actual_entry * (1 + target_pct / 100)
    stop = actual_entry * (1 - stop_pct / 100)
    peak = actual_entry
    trail_stop = None
    max_bars = max(1, time_min // 2)
    exit_price = None
    exit_ts = None
    for i, (ts, row) in enumerate(bars_after.iterrows()):
        if i >= max_bars:
            exit_price = float(row["Close"])
            exit_ts = ts
            break
        c_high = float(row["High"]); c_low = float(row["Low"]); c_close = float(row["Close"])
        if c_high >= target:
            exit_price = target; exit_ts = ts; break
        if c_low <= stop:
            exit_price = stop; exit_ts = ts; break
        if c_high > peak:
            peak = c_high
        unrealized = (peak / actual_entry - 1) * 100
        if unrealized >= trail_act_pct:
            new_trail = peak * (1 - trail_pct / 100)
            if trail_stop is None or new_trail > trail_stop:
                trail_stop = new_trail
        if trail_stop is not None and c_low <= trail_stop:
            exit_price = trail_stop; exit_ts = ts; break
    if exit_price is None and len(bars_after) > 0:
        exit_price = float(bars_after.iloc[-1]["Close"])
        exit_ts = bars_after.index[-1]
    if exit_price is None or exit_ts is None:
        return 0.0
    # Exit slippage (sell is hit downward)
    fake_st = {"mh": mh}
    slip_out = tgc._exit_slip_pct(exit_price, shares, fake_st, exit_ts)
    actual_exit = exit_price * (1 - slip_out / 100)
    return shares * (actual_exit - actual_entry)


def objective(trial):
    target = trial.suggest_float("g_target_pct", 4.0, 100.0, step=1.0)
    stop = trial.suggest_float("g_stop_pct", 5.0, 30.0, step=1.0)
    time_min = trial.suggest_int("g_time_limit_min", 3, 90, step=3)
    trail = trial.suggest_float("g_trail_pct", 0.5, 10.0, step=0.5)
    trail_act = trial.suggest_float("g_trail_activate_pct", 0.0, 15.0, step=0.5)

    total_pnl = 0.0; wins = 0.0; losses = 0.0; n = 0
    pos_dollars = STARTING_CASH * POSITION_PCT
    for ticker, date, entry_ts, fill_price, mh, bars_after in CANDIDATES:
        pnl = _simulate_one_trade(entry_ts, fill_price, mh, bars_after,
                                  target, stop, time_min, trail, trail_act, pos_dollars)
        if pnl == 0.0:
            continue
        total_pnl += pnl
        if pnl > 0: wins += pnl
        else: losses += -pnl
        n += 1

    trial.set_user_attr("n", n)
    trial.set_user_attr("total_pnl", round(total_pnl, 0))
    if n < 50 or total_pnl <= 0:
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
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_ro_any_green")
    study = optuna.create_study(
        direction="maximize",
        study_name="ro_reclaim_train_w21b",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=50),
    )
    print(f"\nR-O option-B study starting (currently {len(study.trials)} trials)")
    t_start = _time.time()
    for i in range(400):
        trial = study.ask()
        try:
            value = objective(trial)
            study.tell(trial, value)
        except Exception as e:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
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
    print(f"  n = {best.user_attrs.get('n')}  total_pnl = ${best.user_attrs.get('total_pnl'):,.0f}  pf = {best.user_attrs.get('pf')}")
