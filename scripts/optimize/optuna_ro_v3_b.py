"""Optuna R-O v3 ('defer-bar-1, color-aware') — option B.

Entry rule (vs any_green which fires at first close > day_open):
  - Bar 0: NEVER fire (we don't yet know if this is a G-day).
  - At each bar i >= 1:
      * If bar 0 was GREEN (close > open of bar 0) AND G has a known hold
        window on this (ticker, date):
          - Wait until AFTER G exits. Scan resumes at G.exit_ts + 1.
      * If bar 0 was RED (close <= open of bar 0):
          - G structurally can't fire (needs 2nd_green pattern, broken
            when bar 0 is red). Fire on first bar i >= 1 where
            close > day_open.
      * If bar 0 was GREEN AND G did not fire (no hold window):
          - In live we'd know at bar 1 that G's pattern didn't trigger,
            so resume scan from bar 1 for first close > day_open.

This rule:
  - Kills the bar 0 'before_g' leakage (we no longer fire R-O at bar 0
    on G-eligible days speculating that G will fire later).
  - Naturally avoids overlap with G's hold window (we wait it out).
  - Captures post-G-exit R-O opportunities on G-firing days.
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
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = [
    "stored_data_2022", "stored_data_combined",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos", "stored_data",
    "stored_data_mar_may_2026", "stored_data_jun_2026",
    "stored_data_2026_gap_fill",
]
DATE_LO = "2024-01-01"
DATE_HI = "2026-02-28"

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

VOL_CAP_PCT = tgc.VOL_CAP_PCT
MAX_REGIME = tgc.MAX_REGIME_PARTICIPATION
MAX_2MIN = tgc.MAX_2MIN_PARTICIPATION

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
print(f"  Trading days: {len(dates)}")

print("Step 2: run #511 baseline to map G-only holding windows...")
g_holds = {}  # (ticker, date) -> list of (entry_ts, exit_ts) for G strategy only
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
        if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
            key = (st.get("ticker"), d)
            etime = st.get("entry_time")
            xtime = st.get("exit_time")
            if etime is not None and xtime is not None:
                g_holds.setdefault(key, []).append((etime, xtime))
    cash = end_c + (unset if is_cash else 0)
print(f"  G hold windows captured: {sum(len(v) for v in g_holds.values())} across {len(g_holds)} (ticker,date) pairs")

print("Step 3: pre-compute v3 R-O candidates (defer-bar-1, color-aware)...")
CANDIDATES = []
counters = {"no_data": 0, "bar0_green_no_g_no_hit": 0, "bar0_red_no_hit": 0,
            "fired_bar0_red": 0, "fired_bar0_green_no_g": 0, "fired_after_g": 0,
            "g_exited_after_eod": 0}
for d in dates:
    for p in picks_by_date.get(d, []):
        mh = p.get("market_hour_candles")
        if mh is None or len(mh) < 2:
            counters["no_data"] += 1
            continue
        day_open = float(mh.iloc[0]["Open"])
        bar0_open = float(mh.iloc[0]["Open"])
        bar0_close = float(mh.iloc[0]["Close"])
        bar0_red = bar0_close <= bar0_open
        holds = g_holds.get((p["ticker"], d), [])

        scan_start_idx = 1  # always skip bar 0
        if not bar0_red and holds:
            # Bar 0 green AND G has hold window — wait until after G's latest exit
            g_exit_ts = max(x for _, x in holds)
            new_start = None
            for i in range(1, len(mh)):
                if mh.index[i] > g_exit_ts:
                    new_start = i
                    break
            if new_start is None:
                counters["g_exited_after_eod"] += 1
                continue
            scan_start_idx = new_start

        # Now scan from scan_start_idx for first close > day_open
        entry_idx = None; entry_price = None; entry_ts = None
        for i in range(scan_start_idx, len(mh)):
            row = mh.iloc[i]
            c = float(row["Close"])
            if c > day_open:
                entry_idx = i
                entry_price = c
                entry_ts = row.name
                break
        if entry_idx is None or entry_price is None or entry_price <= 0:
            if bar0_red:
                counters["bar0_red_no_hit"] += 1
            else:
                counters["bar0_green_no_g_no_hit"] += 1
            continue

        bars_after = mh.iloc[entry_idx + 1:]
        if len(bars_after) == 0:
            continue

        if bar0_red:
            counters["fired_bar0_red"] += 1
        elif holds:
            counters["fired_after_g"] += 1
        else:
            counters["fired_bar0_green_no_g"] += 1

        CANDIDATES.append((p["ticker"], d, entry_ts, entry_price, mh, bars_after))

print(f"  Pre-computed {len(CANDIDATES)} v3 candidates")
for k, v in counters.items():
    print(f"    {k}: {v}")


def _apply_caps(mh, ts, fill_price, requested):
    pre = mh.loc[mh.index <= ts]
    vol_shares = float(pre["Volume"].sum()) if len(pre) > 0 else 0
    dollar_vol = fill_price * vol_shares
    if dollar_vol <= 0:
        return 0, 0
    vol_limit = dollar_vol * (VOL_CAP_PCT / 100)
    v_eff_adj, _, _, v_regime = tgc._multi_window_effective_volume(mh, ts, fill_price)
    if MAX_REGIME > 0 and v_regime > 0:
        vol_limit = min(vol_limit, v_regime * MAX_REGIME)
    if MAX_2MIN > 0 and v_eff_adj > 0:
        vol_limit = min(vol_limit, v_eff_adj * MAX_2MIN)
    return min(requested, vol_limit), v_eff_adj


def _simulate_one_trade(entry_ts, fill_price, mh, bars_after,
                        target_pct, stop_pct, time_min, trail_pct, trail_act_pct,
                        position_dollars):
    capped_size, v_eff_adj = _apply_caps(mh, entry_ts, fill_price, position_dollars)
    if capped_size < 50:
        return 0.0
    slip_in = tgc._entry_slip_pct(fill_price, capped_size, v_eff_adj)
    actual_entry = fill_price * (1 + slip_in / 100)
    shares = capped_size / actual_entry
    target = actual_entry * (1 + target_pct / 100)
    stop = actual_entry * (1 - stop_pct / 100)
    peak = actual_entry
    trail_stop = None
    max_bars = max(1, time_min // 2)
    exit_price = None
    exit_ts = None
    for i, (ts, row) in enumerate(bars_after.iterrows()):
        if i >= max_bars:
            exit_price = float(row["Close"]); exit_ts = ts; break
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
        study_name="ro_v3_train_w21b",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=50),
    )
    print(f"\nv3 study starting (currently {len(study.trials)} trials)")
    t_start = _time.time()
    for i in range(400):
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
