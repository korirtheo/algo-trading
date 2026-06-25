"""Wide-forward of ro_reclaim_train_w21b on OOS windows.

Pulls top-50 + random-50 (from rank 50-500) trials from the postgres study,
evaluates each on:
  - 2022 OOS (stored_data_2022)
  - 2026 Mar-Jun OOS (stored_data_mar_may_2026 + stored_data_jun_2026)

Uses the same reclaim_after_dip + mode (b) + vol-cap + dynamic-slippage
simulation as the Optuna training (just with different params per trial).

Reports per-trial forward PnL + WR + PF + max DD for each OOS window,
plus train-vs-forward Pearson and the top-10 forward-PnL trials.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import random
import statistics
import numpy as np
import psycopg2

STARTING_CASH = 25_000
POSITION_PCT = 0.30
BASELINE_PATH = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DB = "optuna_ro_any_green"
STUDY = "ro_reclaim_train_w21b"
TOP_N = 50
RANDOM_N = 50
RANDOM_RANGE = (50, 500)
SEED = 42

OOS_WINDOWS = {
    "2022":          {"dirs": ["stored_data_2022"], "lo": "2022-01-01", "hi": "2022-12-31"},
    "2026_mar_jun":  {"dirs": ["stored_data_mar_may_2026", "stored_data_jun_2026"],
                      "lo": "2026-03-01", "hi": "2026-06-30"},
}


def fetch_trials(db, study_name, top_n, random_n, random_range, seed):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value
          FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > 0
           AND t.study_id=(SELECT study_id FROM studies WHERE study_name=%s)
         ORDER BY tv.value DESC
    """, (study_name,))
    all_sorted = cur.fetchall()
    selected = list(all_sorted[:top_n])
    lo, hi = random_range
    hi = min(hi, len(all_sorted))
    middle = all_sorted[lo:hi] if hi > lo else []
    rng = random.Random(seed)
    rand_sample = rng.sample(middle, min(random_n, len(middle))) if middle else []
    selected.extend(rand_sample)
    seen, unique = set(), []
    for r in selected:
        if r[0] in seen: continue
        seen.add(r[0]); unique.append(r)

    trials = []
    for tid, num, score in unique:
        cur.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s", (tid,))
        params = {}
        for name, val, dist in cur.fetchall():
            try:
                d = json.loads(dist) if dist else {}
                kind = d.get("name", "")
                if "Int" in kind:
                    params[name] = int(val)
                else:
                    params[name] = float(val)
            except Exception:
                params[name] = float(val)
        cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
        ua = dict(cur.fetchall())
        train_rank = next(i+1 for i, r in enumerate(all_sorted) if r[0] == tid)
        trials.append({"trial_id": tid, "number": num, "score": float(score),
                       "user_attrs": ua, "params": params, "train_rank": train_rank})
    c.close()
    return trials, len(all_sorted)


def setup_sim():
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    with open(BASELINE_PATH) as f: baseline = json.load(f)
    with open(W21B_DEPLOY) as f: p511 = json.load(f)["params"]
    merged = {**baseline, **p511}
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
    return tgc


def precompute_candidates(tgc, dirs, date_lo, date_hi):
    """Run baseline #511 to map G/L holds, then pre-compute reclaim candidates with mode (b)."""
    from test_full import load_all_picks, MARGIN_THRESHOLD

    dirs = [d for d in dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])

    cash = STARTING_CASH
    gl_holds = {}
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                key = (st.get("ticker"), d)
                etime = st.get("entry_time"); xtime = st.get("exit_time")
                if etime is not None:
                    gl_holds.setdefault(key, []).append((etime, xtime))
        cash = end_c + (unset if is_cash else 0)

    candidates = []
    skipped = 0
    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2: continue
            day_open = float(mh.iloc[0]["Open"])
            entry_idx = None; entry_price = None; entry_ts = None
            ever_below = False
            for i, (ts, row) in enumerate(mh.iterrows()):
                c = float(row["Close"])
                if ever_below and c > day_open:
                    entry_idx = i; entry_price = c; entry_ts = ts; break
                if c <= day_open: ever_below = True
            if entry_idx is None or entry_price <= 0: continue
            holds = gl_holds.get((p["ticker"], d), [])
            if any(et <= entry_ts <= (xt if xt is not None else et) for et, xt in holds):
                skipped += 1; continue
            bars_after = mh.iloc[entry_idx + 1:]
            if len(bars_after) == 0: continue
            candidates.append((p["ticker"], d, entry_ts, entry_price, mh, bars_after))
    return candidates, len(dates), skipped


def make_simulator(tgc):
    VOL_CAP_PCT = tgc.VOL_CAP_PCT
    MAX_REGIME = tgc.MAX_REGIME_PARTICIPATION
    MAX_2MIN = tgc.MAX_2MIN_PARTICIPATION

    def _apply_caps(mh, ts, fill_price, requested):
        pre = mh.loc[mh.index <= ts]
        vol_shares = float(pre["Volume"].sum()) if len(pre) > 0 else 0
        dollar_vol = fill_price * vol_shares
        if dollar_vol <= 0: return 0, 0
        vol_limit = dollar_vol * (VOL_CAP_PCT / 100)
        v_eff_adj, _, _, v_regime = tgc._multi_window_effective_volume(mh, ts, fill_price)
        if MAX_REGIME > 0 and v_regime > 0:
            vol_limit = min(vol_limit, v_regime * MAX_REGIME)
        if MAX_2MIN > 0 and v_eff_adj > 0:
            vol_limit = min(vol_limit, v_eff_adj * MAX_2MIN)
        return min(requested, vol_limit), v_eff_adj

    def simulate_one(entry_ts, fill_price, mh, bars_after,
                     target_pct, stop_pct, time_min, trail_pct, trail_act_pct,
                     position_dollars):
        capped_size, v_eff_adj = _apply_caps(mh, entry_ts, fill_price, position_dollars)
        if capped_size < 50: return 0.0
        slip_in = tgc._entry_slip_pct(fill_price, capped_size, v_eff_adj)
        actual_entry = fill_price * (1 + slip_in / 100)
        shares = capped_size / actual_entry
        target = actual_entry * (1 + target_pct / 100)
        stop = actual_entry * (1 - stop_pct / 100)
        peak = actual_entry
        trail_stop = None
        max_bars = max(1, time_min // 2)
        exit_price = None; exit_ts = None
        for i, (ts, row) in enumerate(bars_after.iterrows()):
            if i >= max_bars:
                exit_price = float(row["Close"]); exit_ts = ts; break
            h, lo, c = float(row["High"]), float(row["Low"]), float(row["Close"])
            if h >= target: exit_price = target; exit_ts = ts; break
            if lo <= stop:  exit_price = stop; exit_ts = ts; break
            if h > peak: peak = h
            unrealized = (peak / actual_entry - 1) * 100
            if unrealized >= trail_act_pct:
                nt = peak * (1 - trail_pct / 100)
                if trail_stop is None or nt > trail_stop: trail_stop = nt
            if trail_stop is not None and lo <= trail_stop:
                exit_price = trail_stop; exit_ts = ts; break
        if exit_price is None and len(bars_after) > 0:
            exit_price = float(bars_after.iloc[-1]["Close"])
            exit_ts = bars_after.index[-1]
        if exit_price is None or exit_ts is None: return 0.0
        fake_st = {"mh": mh}
        slip_out = tgc._exit_slip_pct(exit_price, shares, fake_st, exit_ts)
        actual_exit = exit_price * (1 - slip_out / 100)
        return shares * (actual_exit - actual_entry)

    return simulate_one


def evaluate_trial(simulate_one, candidates, params):
    target = params["g_target_pct"]
    stop = params["g_stop_pct"]
    time_min = int(params["g_time_limit_min"])
    trail = params["g_trail_pct"]
    trail_act = params["g_trail_activate_pct"]
    pos_dollars = STARTING_CASH * POSITION_PCT
    pnls = []
    for tkr, d, ets, ep, mh, ba in candidates:
        pnl = simulate_one(ets, ep, mh, ba, target, stop, time_min, trail, trail_act, pos_dollars)
        if pnl == 0.0: continue
        pnls.append(pnl)
    n = len(pnls)
    if n == 0:
        return {"n": 0, "total_pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "mean_pnl": 0.0, "median_pnl": 0.0}
    tot = sum(pnls)
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p <= 0))
    wr = sum(1 for p in pnls if p > 0) / n * 100
    pf = wins / losses if losses > 0 else 99.0
    return {"n": n, "total_pnl": float(tot), "wr": float(wr), "pf": float(pf),
            "mean_pnl": float(tot/n), "median_pnl": float(statistics.median(pnls))}


def correlate(rows, x_key, y_key="forward_pnl"):
    xs, ys = [], []
    for r in rows:
        v, t = r.get(x_key), r.get(y_key)
        if v is None or t is None: continue
        try: v = float(v); t = float(t)
        except (TypeError, ValueError): continue
        if not (np.isfinite(v) and np.isfinite(t)): continue
        xs.append(v); ys.append(t)
    if len(xs) < 5: return None
    xs, ys = np.array(xs), np.array(ys)
    return float(np.corrcoef(xs, ys)[0,1])


def main():
    print(f"Fetching trials from {DB}.{STUDY}...")
    trials, total = fetch_trials(DB, STUDY, TOP_N, RANDOM_N, RANDOM_RANGE, SEED)
    print(f"  Selected {len(trials)} (top {TOP_N} + random {RANDOM_N} of {total})")

    print("\nSetup simulator + load OOS candidates per window...")
    tgc = setup_sim()
    simulate_one = make_simulator(tgc)

    window_candidates = {}
    for wname, w in OOS_WINDOWS.items():
        print(f"\n  Pre-computing {wname} candidates ({w['lo']} .. {w['hi']})...")
        cands, ndays, skipped = precompute_candidates(tgc, w["dirs"], w["lo"], w["hi"])
        print(f"    {ndays} days, {len(cands)} candidates, {skipped} skipped G/L overlap")
        window_candidates[wname] = cands

    print("\nEvaluating trials on each OOS window...")
    results = []
    for i, t in enumerate(trials):
        row = {"trial": t["number"], "train_rank": t["train_rank"],
               "train_score": t["score"], "params": t["params"]}
        for wname, cands in window_candidates.items():
            r = evaluate_trial(simulate_one, cands, t["params"])
            row[f"{wname}_n"] = r["n"]
            row[f"{wname}_total_pnl"] = r["total_pnl"]
            row[f"{wname}_wr"] = r["wr"]
            row[f"{wname}_pf"] = r["pf"]
            row[f"{wname}_mean_pnl"] = r["mean_pnl"]
        row["forward_pnl"] = row["2022_total_pnl"] + row["2026_mar_jun_total_pnl"]
        results.append(row)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(trials)}] done")

    results.sort(key=lambda r: -r["forward_pnl"])

    out = "results/wide_forward_ro_reclaim_w21b.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"study": STUDY, "n_trials": len(results), "results": results}, f, indent=2, default=str)

    print(f"\n{'='*100}")
    print(f"  TOP 15 by forward_pnl (sum of 2022 + 2026 Mar-Jun)")
    print(f"{'='*100}")
    print(f"  {'rk':<3} {'trial':<5} {'train_rk':<8} {'train$':>9} "
          f"{'2022$':>10} {'2022_wr':>7} {'26mj$':>10} {'26mj_wr':>7} {'fwd_tot$':>10}")
    for i, r in enumerate(results[:15]):
        print(f"  {i+1:<3} #{r['trial']:<4} {r['train_rank']:<8} "
              f"${r['train_score']/1000:>6.0f}K "
              f"${r['2022_total_pnl']:>+8,.0f} {r['2022_wr']:>5.1f}% "
              f"${r['2026_mar_jun_total_pnl']:>+8,.0f} {r['2026_mar_jun_wr']:>5.1f}% "
              f"${r['forward_pnl']:>+8,.0f}")

    print(f"\nCorrelations with forward_pnl:")
    feats = ["train_score", "train_rank", "2022_total_pnl", "2026_mar_jun_total_pnl",
             "2022_wr", "2026_mar_jun_wr", "2022_pf", "2026_mar_jun_pf"]
    for f in feats:
        c = correlate(results, f, "forward_pnl")
        if c is not None:
            print(f"  {f:<28} Pearson = {c:>+.3f}")

    # Top-N beat baseline?
    top_count = sum(1 for r in results[:TOP_N] if r["forward_pnl"] > 0)
    print(f"\n  Top {TOP_N} (by train) with positive forward PnL: {top_count}/{TOP_N} ({100*top_count/TOP_N:.0f}%)")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
