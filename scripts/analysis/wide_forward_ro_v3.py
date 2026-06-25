"""Wide-forward of ro_v3_train_w21b on OOS windows.

Uses the v3 entry rule (defer-bar-1, color-aware, post-G-exit re-scan).
Pulls top-50 + random-50 trials, evaluates on 2022 OOS + 2026 Mar-Jun OOS.

Also computes time-of-day breakdown for the top v3 trial:
  - bar 1 (9:32 ET), bar 2-5, bar 6-15, bar 16-30, bar 31+
  - WR / PF / mean PnL per bucket
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
STUDY = "ro_v3_train_w21b"
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
                if "Int" in kind: params[name] = int(val)
                else: params[name] = float(val)
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


def precompute_v3_candidates(tgc, dirs, date_lo, date_hi):
    """v3 rule: skip bar 0; if bar 0 green AND G fires → wait until after G exits; else fire on first close > day_open from bar 1+."""
    from test_full import load_all_picks, MARGIN_THRESHOLD

    dirs = [d for d in dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])

    cash = STARTING_CASH
    g_holds = {}  # (ticker, date) -> list of (entry_ts, exit_ts) for G strategy only
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
                key = (st.get("ticker"), d)
                etime, xtime = st.get("entry_time"), st.get("exit_time")
                if etime is not None and xtime is not None:
                    g_holds.setdefault(key, []).append((etime, xtime))
        cash = end_c + (unset if is_cash else 0)

    candidates = []
    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2: continue
            day_open = float(mh.iloc[0]["Open"])
            bar0_open = float(mh.iloc[0]["Open"])
            bar0_close = float(mh.iloc[0]["Close"])
            bar0_red = bar0_close <= bar0_open
            holds = g_holds.get((p["ticker"], d), [])

            scan_start_idx = 1
            if not bar0_red and holds:
                g_exit_ts = max(x for _, x in holds)
                new_start = None
                for i in range(1, len(mh)):
                    if mh.index[i] > g_exit_ts:
                        new_start = i; break
                if new_start is None: continue
                scan_start_idx = new_start

            entry_idx = None; entry_price = None; entry_ts = None
            for i in range(scan_start_idx, len(mh)):
                row = mh.iloc[i]
                c = float(row["Close"])
                if c > day_open:
                    entry_idx = i; entry_price = c; entry_ts = row.name; break
            if entry_idx is None or entry_price is None or entry_price <= 0: continue
            bars_after = mh.iloc[entry_idx + 1:]
            if len(bars_after) == 0: continue
            candidates.append((p["ticker"], d, entry_ts, entry_price, mh, bars_after, entry_idx))
    return candidates, len(dates)


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


def evaluate_trial(simulate_one, candidates, params, track_by_bar=False):
    target = params["g_target_pct"]
    stop = params["g_stop_pct"]
    time_min = int(params["g_time_limit_min"])
    trail = params["g_trail_pct"]
    trail_act = params["g_trail_activate_pct"]
    pos_dollars = STARTING_CASH * POSITION_PCT
    pnls = []
    by_bar = {}  # bar_idx -> list of pnls
    for c in candidates:
        tkr, d, ets, ep, mh, ba, bar_idx = c
        pnl = simulate_one(ets, ep, mh, ba, target, stop, time_min, trail, trail_act, pos_dollars)
        if pnl == 0.0: continue
        pnls.append(pnl)
        if track_by_bar:
            by_bar.setdefault(bar_idx, []).append(pnl)
    n = len(pnls)
    if n == 0:
        return {"n": 0, "total_pnl": 0.0, "wr": 0.0, "pf": 0.0, "mean_pnl": 0.0, "by_bar": {}}
    tot = sum(pnls)
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p <= 0))
    wr = sum(1 for p in pnls if p > 0) / n * 100
    pf = wins / losses if losses > 0 else 99.0
    result = {"n": n, "total_pnl": float(tot), "wr": float(wr), "pf": float(pf),
              "mean_pnl": float(tot/n)}
    if track_by_bar:
        bar_stats = {}
        for bi, ps in by_bar.items():
            ntot = sum(ps); nwin = sum(p for p in ps if p > 0); nloss = abs(sum(p for p in ps if p <= 0))
            bar_stats[bi] = {"n": len(ps), "total_pnl": ntot,
                             "wr": sum(1 for p in ps if p > 0)/len(ps)*100,
                             "pf": nwin/nloss if nloss > 0 else 99.0,
                             "mean": ntot/len(ps)}
        result["by_bar"] = bar_stats
    return result


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

    print("\nSetup simulator + load OOS v3 candidates per window...")
    tgc = setup_sim()
    simulate_one = make_simulator(tgc)

    window_candidates = {}
    for wname, w in OOS_WINDOWS.items():
        print(f"\n  Pre-computing {wname} v3 candidates ({w['lo']} .. {w['hi']})...")
        cands, ndays = precompute_v3_candidates(tgc, w["dirs"], w["lo"], w["hi"])
        print(f"    {ndays} days, {len(cands)} candidates")
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
    for f in ["train_score", "train_rank", "2022_total_pnl", "2026_mar_jun_total_pnl",
              "2022_wr", "2026_mar_jun_wr", "2022_pf", "2026_mar_jun_pf"]:
        c = correlate(results, f, "forward_pnl")
        if c is not None: print(f"  {f:<28} Pearson = {c:>+.3f}")

    top_count = sum(1 for r in results[:TOP_N] if r["forward_pnl"] > 0)
    print(f"\n  Top {TOP_N} (by train) with positive forward PnL: {top_count}/{TOP_N} ({100*top_count/TOP_N:.0f}%)")

    # Time-of-day breakdown for top trial
    top_trial = results[0]
    print(f"\n{'='*100}")
    print(f"  TIME-OF-DAY breakdown for top forward trial #{top_trial['trial']}")
    print(f"  Params: {top_trial['params']}")
    print(f"{'='*100}")
    for wname, cands in window_candidates.items():
        print(f"\n  Window: {wname}")
        r = evaluate_trial(simulate_one, cands, top_trial["params"], track_by_bar=True)
        by_bar = r["by_bar"]
        # Bucket bar_idx into time buckets
        buckets = {"bar 1 (9:32)": (1,1), "bar 2-5 (9:34-9:40)": (2,5),
                   "bar 6-15 (9:42-10:00)": (6,15), "bar 16-30 (10:00-10:30)": (16,30),
                   "bar 31+ (post-10:30)": (31, 999)}
        print(f"    {'bucket':<28} {'n':>5} {'total_pnl':>12} {'mean':>7} {'WR%':>6} {'PF':>6}")
        for bname, (b_lo, b_hi) in buckets.items():
            n_b = 0; tot_b = 0.0; wins_b = 0.0; losses_b = 0.0; w_count = 0
            for bi, s in by_bar.items():
                if b_lo <= bi <= b_hi:
                    n_b += s["n"]; tot_b += s["total_pnl"]
                    wins_b += s["total_pnl"] if s["total_pnl"] > 0 else 0
                    w_count += int(s["wr"] * s["n"] / 100)
            if n_b == 0: print(f"    {bname:<28} {0:>5}"); continue
            mean_b = tot_b / n_b
            wr_b = 100 * w_count / n_b
            # Approximate PF from aggregated stats
            print(f"    {bname:<28} {n_b:>5} ${tot_b:>+9,.0f} ${mean_b:>+5,.0f} {wr_b:>5.1f}%   --")

    out = "results/wide_forward_ro_v3_w21b.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"study": STUDY, "n_trials": len(results), "results": results}, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
