"""Multi-window CV Optuna for G+L with v3 R-O layered in the SAME compounding account.

Training windows (CV across both):
  - 2022 (cool regime — where current G+L breaks)
  - W21b 2024-01 to 2026-02-28 (hot regime — where the current Optuna trained)

Per trial:
  - Trial proposes G+L params
  - For each window: run joint compounded sim
      - G+L from simulate_day_combined
      - v3 with FIXED #576 params, layered on top of G+L cash daily
  - Score = geomean(window_pnls) * min(window_pfs, 3)  (rewards balanced performance)
  - Hard floor: any window negative or PF<0.5 -> -9999

v3 candidate detection uses pre-computed G holds from the #511 BASELINE
(approximation: avoids re-running v3 candidate logic per trial; v3 candidate
set barely changes across G param variants).

Tunable: 8 G params + 16 L params = 24 dims. n_startup_trials = 100.
"""
import json, os, sys, time as _time, math
import numpy as np
import optuna
from optuna.samplers import TPESampler
from collections import defaultdict

sys.path.insert(0, '.')

STARTING_CASH = 25_000
POSITION_PCT = 0.30
BASELINE_PATH = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

V3_PARAMS = {"g_target_pct": 57.0, "g_stop_pct": 30.0, "g_time_limit_min": 27,
             "g_trail_pct": 0.5, "g_trail_activate_pct": 0.0}

WINDOWS = {
    "2022": {"dirs": ["stored_data_2022"], "lo": "2022-01-01", "hi": "2022-12-31"},
    "W21b": {"dirs": ["stored_data_combined", "stored_data_jan_mar_2024",
                       "stored_data_apr_jun_2024", "stored_data_jul_sep_2024",
                       "stored_data_oct_dec_2024", "stored_data_jan_mar_2025",
                       "stored_data_apr_jun_2025", "stored_data_jul_2025",
                       "stored_data_oos", "stored_data"],
             "lo": "2024-01-01", "hi": "2026-02-28"},
}


print("Step 1: configure simulator + load data...")
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

with open(BASELINE_PATH) as f: baseline_p = json.load(f)
with open(W21B_DEPLOY) as f: p511 = json.load(f)["params"]

# Pre-load picks per window
WINDOW_DATA = {}
for wname, w in WINDOWS.items():
    print(f"  Loading {wname}...")
    dirs = [d for d in w["dirs"] if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if w["lo"] <= d <= w["hi"]])
    WINDOW_DATA[wname] = {"dates": dates, "picks": picks_by_date}
    print(f"    {wname}: {len(dates)} days")


def _setup_sim(params):
    set_strategy_params(params)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False


# Pre-compute v3 candidates per window using #511 baseline G holds (fixed approximation)
print("Step 2: pre-compute v3 candidates per window using #511 baseline G holds...")
V3_CANDS_PER_WINDOW = {}
for wname, wd in WINDOW_DATA.items():
    base = {**baseline_p, **p511}
    for s in ALL_STRATS:
        base[f"enable_{s}"] = (s in {"g","l"})
    _setup_sim(base)
    g_holds = defaultdict(list)
    cash = STARTING_CASH
    for d in wd["dates"]:
        dp = wd["picks"].get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except: continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
                et, xt = st.get("entry_time"), st.get("exit_time")
                if et and xt: g_holds[(st["ticker"], d)].append((et, xt))
        cash = end_c + (unset if is_cash else 0)

    cands_by_date = defaultdict(list)
    for d in wd["dates"]:
        for p in wd["picks"].get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2: continue
            day_open = float(mh.iloc[0]["Open"])
            bar0_red = float(mh.iloc[0]["Close"]) <= day_open
            ghol = g_holds.get((p["ticker"], d), [])
            scan = 1
            if not bar0_red and ghol:
                ge = max(x for _, x in ghol)
                ns = None
                for i in range(1, len(mh)):
                    if mh.index[i] > ge: ns = i; break
                if ns is None: continue
                scan = ns
            for i in range(scan, len(mh)):
                if float(mh.iloc[i]["Close"]) > day_open:
                    ba = mh.iloc[i+1:]
                    if len(ba) == 0: break
                    cands_by_date[d].append((p["ticker"], mh.index[i], float(mh.iloc[i]["Close"]), mh, ba))
                    break
    V3_CANDS_PER_WINDOW[wname] = cands_by_date
    print(f"  {wname}: v3 candidates = {sum(len(v) for v in cands_by_date.values())}")


def _v3_trade(mh, ets, fp, ba, pos_dollars):
    pre = mh.loc[mh.index <= ets]
    vs = float(pre["Volume"].sum()) if len(pre) > 0 else 0
    dv = fp * vs
    if dv <= 0: return 0.0
    lim = dv * (tgc.VOL_CAP_PCT/100)
    ve, _, _, vr = tgc._multi_window_effective_volume(mh, ets, fp)
    if tgc.MAX_REGIME_PARTICIPATION > 0 and vr > 0: lim = min(lim, vr * tgc.MAX_REGIME_PARTICIPATION)
    if tgc.MAX_2MIN_PARTICIPATION > 0 and ve > 0: lim = min(lim, ve * tgc.MAX_2MIN_PARTICIPATION)
    cs = min(pos_dollars, lim)
    if cs < 50: return 0.0
    si = tgc._entry_slip_pct(fp, cs, ve)
    ae = fp * (1 + si/100)
    sh = cs / ae
    tp = V3_PARAMS["g_target_pct"]; sp = V3_PARAMS["g_stop_pct"]
    tm = int(V3_PARAMS["g_time_limit_min"]); trp = V3_PARAMS["g_trail_pct"]; tap = V3_PARAMS["g_trail_activate_pct"]
    tgt = ae*(1+tp/100); stp = ae*(1-sp/100); peak = ae; ts_ = None
    ep = None; et2 = None
    mb = max(1, tm//2)
    for i, (ts, row) in enumerate(ba.iterrows()):
        if i >= mb: ep = float(row["Close"]); et2 = ts; break
        h = float(row["High"]); lo_ = float(row["Low"])
        if h >= tgt: ep = tgt; et2 = ts; break
        if lo_ <= stp: ep = stp; et2 = ts; break
        if h > peak: peak = h
        if (peak/ae-1)*100 >= tap:
            nt = peak*(1-trp/100)
            if ts_ is None or nt > ts_: ts_ = nt
        if ts_ is not None and lo_ <= ts_: ep = ts_; et2 = ts; break
    if ep is None and len(ba) > 0: ep = float(ba.iloc[-1]["Close"]); et2 = ba.index[-1]
    if ep is None or et2 is None: return 0.0
    so = tgc._exit_slip_pct(ep, sh, {"mh": mh}, et2)
    return sh * (ep*(1-so/100) - ae)


def run_window_with_v3(params, wname):
    """Run G+L joint compounded sim on `wname` with v3 layered. Returns (pnl, pf)."""
    _setup_sim(params)
    wd = WINDOW_DATA[wname]
    v3_cands = V3_CANDS_PER_WINDOW[wname]
    cash = STARTING_CASH
    gl_wins = gl_losses = v3_wins = v3_losses = 0.0
    n_g = n_l = n_v3 = 0
    for d in wd["dates"]:
        dp = wd["picks"].get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            continue
        # tally G+L PnLs
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                p = float(st.get("pnl") or 0)
                if s in {"G","L"}:
                    if s == "G": n_g += 1
                    else: n_l += 1
                    if p > 0: gl_wins += p
                    else: gl_losses += -p
        gl_end_cash = end_c + (unset if is_cash else 0)
        # Layer v3 trades using start-of-day cash
        v3_added = 0.0
        pos = cash * POSITION_PCT
        for (tkr, ets, fp, mh, ba) in v3_cands.get(d, []):
            pnl = _v3_trade(mh, ets, fp, ba, pos)
            if pnl != 0.0:
                n_v3 += 1
                if pnl > 0: v3_wins += pnl
                else: v3_losses += -pnl
                v3_added += pnl
        cash = gl_end_cash + v3_added
    pnl = cash - STARTING_CASH
    wins = gl_wins + v3_wins
    losses = gl_losses + v3_losses
    pf = wins / losses if losses > 0 else 99.0
    return pnl, pf, n_g, n_l, n_v3


def objective(trial):
    # Tunable: 8 G params + 16 L params
    params = dict(baseline_p)

    # G
    params["g_min_gap_pct"] = trial.suggest_float("g_min_gap_pct", 5.0, 50.0, step=5.0)
    params["g_require_2nd_green"] = trial.suggest_categorical("g_require_2nd_green", [True, False])
    params["g_require_2nd_new_high"] = trial.suggest_categorical("g_require_2nd_new_high", [True, False])
    params["g_stop_pct"] = trial.suggest_float("g_stop_pct", 5.0, 30.0, step=1.0)
    params["g_target_pct"] = trial.suggest_float("g_target_pct", 10.0, 80.0, step=2.0)
    params["g_time_limit_min"] = trial.suggest_int("g_time_limit_min", 6, 30, step=3)
    params["g_trail_pct"] = trial.suggest_float("g_trail_pct", 0.5, 3.0, step=0.5)
    params["g_trail_activate_pct"] = trial.suggest_float("g_trail_activate_pct", 0.0, 8.0, step=1.0)

    # L
    params["l_earliest_candle"] = trial.suggest_int("l_earliest_candle", 3, 30, step=3)
    params["l_latest_candle"] = trial.suggest_int("l_latest_candle", 30, 180, step=15)
    params["l_max_float"] = trial.suggest_int("l_max_float", 5_000_000, 25_000_000, step=5_000_000)
    params["l_min_gap"] = trial.suggest_int("l_min_gap", 10, 80, step=5)
    params["l_min_price_accel_pct"] = trial.suggest_float("l_min_price_accel_pct", 0.5, 3.0, step=0.5)
    params["l_partial_sell_pct"] = trial.suggest_float("l_partial_sell_pct", 0.0, 50.0, step=25.0)
    params["l_stop_pct"] = trial.suggest_float("l_stop_pct", 10.0, 25.0, step=1.0)
    params["l_tier1_target1_pct"] = trial.suggest_float("l_tier1_target1_pct", 15.0, 50.0, step=5.0)
    params["l_tier1_target2_pct"] = trial.suggest_float("l_tier1_target2_pct", 20.0, 60.0, step=5.0)
    params["l_tier2_target1_pct"] = trial.suggest_float("l_tier2_target1_pct", 10.0, 30.0, step=2.0)
    params["l_tier2_target2_pct"] = trial.suggest_float("l_tier2_target2_pct", 20.0, 50.0, step=5.0)
    params["l_tier3_target1_pct"] = trial.suggest_float("l_tier3_target1_pct", 5.0, 25.0, step=2.0)
    params["l_tier3_target2_pct"] = trial.suggest_float("l_tier3_target2_pct", 15.0, 40.0, step=5.0)
    params["l_time_limit_min"] = trial.suggest_int("l_time_limit_min", 30, 180, step=15)
    params["l_trail_activate_pct"] = trial.suggest_float("l_trail_activate_pct", 0.0, 8.0, step=1.0)
    params["l_trail_pct"] = trial.suggest_float("l_trail_pct", 0.5, 5.0, step=0.5)

    for s in ALL_STRATS:
        params[f"enable_{s}"] = (s in {"g","l"})

    pnls = {}; pfs = {}; ns = {}
    for wname in WINDOWS:
        try:
            pnl, pf, n_g, n_l, n_v3 = run_window_with_v3(params, wname)
        except Exception:
            return -9999
        pnls[wname] = pnl
        pfs[wname] = pf
        ns[wname] = (n_g, n_l, n_v3)

    trial.set_user_attr("pnl_2022", round(pnls["2022"], 0))
    trial.set_user_attr("pnl_w21b", round(pnls["W21b"], 0))
    trial.set_user_attr("pf_2022", round(pfs["2022"], 3))
    trial.set_user_attr("pf_w21b", round(pfs["W21b"], 3))
    trial.set_user_attr("n_2022", ns["2022"])
    trial.set_user_attr("n_w21b", ns["W21b"])

    # Hard floor: any window negative or PF too low
    if any(p <= 0 for p in pnls.values()):
        return -9999
    if any(p < 0.5 for p in pfs.values()):
        return -9999

    # Score: geomean(pnl) × min(pf, 3) — rewards balanced performance
    geomean_pnl = math.exp(sum(math.log(max(1, p)) for p in pnls.values()) / len(pnls))
    pf_factor = min(min(pfs.values()), 3.0)
    score = geomean_pnl * pf_factor
    if not math.isfinite(score): return -9999
    return max(-9.9e12, min(9.9e12, score))


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_g_l_cv_v3")
    study = optuna.create_study(
        direction="maximize",
        study_name="g_l_cv_v3_w21b_2022",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=200),
    )
    print(f"\nStudy starting (currently {len(study.trials)} trials)")
    t_start = _time.time()
    for i in range(400):
        trial = study.ask()
        try:
            value = objective(trial)
            study.tell(trial, value)
        except Exception:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        if (i + 1) % 10 == 0:
            elapsed = _time.time() - t_start
            try:
                best_val = study.best_value
                bt = study.best_trial
                p22 = bt.user_attrs.get("pnl_2022", 0)
                pw = bt.user_attrs.get("pnl_w21b", 0)
                print(f"  [{i+1}/400] elapsed {elapsed/60:.1f}m | best ${best_val:,.0f} (#{bt.number})  2022=${p22:,.0f}  W21b=${pw:,.0f}")
            except: print(f"  [{i+1}/400] elapsed {elapsed/60:.1f}m")

    best = study.best_trial
    print(f"\n*** BEST TRIAL #{best.number} ***")
    print(f"  Score: ${best.value:,.0f}")
    for k, v in best.params.items():
        print(f"  {k} = {v}")
    for k, v in best.user_attrs.items():
        print(f"  {k} = {v}")
