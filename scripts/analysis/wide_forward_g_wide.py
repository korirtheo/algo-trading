"""Wide-forward of g_wide_w21b that applies the SAME entry locks the Optuna used.

Critical: g_wide Optuna fixed g_min_gap_pct=15, g_require_2nd_green=True,
g_require_2nd_new_high=False. The generic wide_forward_g_l.py uses baseline
defaults which differ — must apply the locks here for valid OOS comparison.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse, json, os, random, statistics
import numpy as np
import psycopg2
from concurrent.futures import ProcessPoolExecutor, as_completed

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
TOP_N = 50
RANDOM_N = 50
RANDOM_RANGE = (50, 500)
SEED = 42

OOS_WINDOWS = {
    "2022":          {"dirs": ["stored_data_2022"], "lo": "2022-01-01", "hi": "2022-12-31"},
    "2026_mar_jun":  {"dirs": ["stored_data_mar_may_2026", "stored_data_jun_2026"],
                      "lo": "2026-03-01", "hi": "2026-06-30"},
}

# G_WIDE ENTRY LOCKS (same as in optuna_g_wide_w21b.py)
G_ENTRY_LOCKS = {
    "g_min_gap_pct": 15.0,
    "g_require_2nd_green": True,
    "g_require_2nd_new_high": False,
}


def fetch_trials(db, study_name):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value FROM trials t
        JOIN trial_values tv ON tv.trial_id=t.trial_id
        WHERE t.state='COMPLETE' AND tv.value > 0
          AND t.study_id=(SELECT study_id FROM studies WHERE study_name=%s)
        ORDER BY tv.value DESC
    """, (study_name,))
    all_sorted = cur.fetchall()
    selected = list(all_sorted[:TOP_N])
    lo, hi = RANDOM_RANGE
    hi = min(hi, len(all_sorted))
    middle = all_sorted[lo:hi] if hi > lo else []
    rng = random.Random(SEED)
    rand_sample = rng.sample(middle, min(RANDOM_N, len(middle))) if middle else []
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
                if kind == "CategoricalDistribution":
                    params[name] = d["attributes"]["choices"][int(val)]
                elif "Int" in kind: params[name] = int(val)
                else: params[name] = float(val)
            except: params[name] = float(val)
        train_rank = next(i+1 for i, r in enumerate(all_sorted) if r[0] == tid)
        trials.append({"trial_id": tid, "number": num, "score": float(score),
                       "params": params, "train_rank": train_rank})
    c.close()
    return trials, len(all_sorted)


def _forward(args):
    trial, wname, wdirs, wlo, whi = args
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    with open(W21B_DEPLOY) as f: p511 = json.load(f)["params"]
    merged = {**baseline, **p511}
    # Apply g_wide's entry locks (critical for fair OOS measurement)
    for k, v in G_ENTRY_LOCKS.items(): merged[k] = v
    # Override with trial's exit params
    for k, v in trial["params"].items(): merged[k] = v
    # G only
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s == "g")
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False
    tgc.MARGIN_MULTIPLIER = 1.0

    dirs = [d for d in wdirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if wlo <= d <= whi])
    cash = STARTING_CASH
    n = 0; wins = 0; losses_v = 0.0; wins_v = 0.0; daily_eq = [cash]
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try: states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except: daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
                n += 1
                p = float(st.get("pnl") or 0)
                if p > 0: wins += 1; wins_v += p
                else: losses_v += -p
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq); peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0
    return {"trial": trial["number"], "train_rank": trial["train_rank"],
            "train_score": trial["score"], "window": wname,
            "n_trades": n, "forward_pnl": float(cash - STARTING_CASH),
            "wr": (wins/n*100) if n else 0,
            "pf": (wins_v/losses_v) if losses_v > 0 else 99,
            "max_dd_pct": dd}


def correlate(rows, x_key, y_key="forward_pnl"):
    xs, ys = [], []
    for r in rows:
        v, t = r.get(x_key), r.get(y_key)
        if v is None or t is None: continue
        try: v = float(v); t = float(t)
        except: continue
        if not (np.isfinite(v) and np.isfinite(t)): continue
        xs.append(v); ys.append(t)
    if len(xs) < 5: return None
    return float(np.corrcoef(np.array(xs), np.array(ys))[0,1])


def main():
    print("Fetching trials from g_wide_w21b...")
    trials, total = fetch_trials("optuna_g_wide", "g_wide_w21b")
    print(f"  Selected {len(trials)} (top {TOP_N} + random {RANDOM_N} of {total})")

    print(f"\nForward-testing on {list(OOS_WINDOWS.keys())} with 6 workers...")
    tasks = []
    for t in trials:
        for wname, w in OOS_WINDOWS.items():
            tasks.append((t, wname, w["dirs"], w["lo"], w["hi"]))

    by_trial = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_forward, task): task[0]["number"] for task in tasks}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                by_trial.setdefault(r["trial"], {})[r["window"]] = r
                done += 1
                if done % 20 == 0: print(f"  [{done}/{len(tasks)}] done")
            except Exception as e: print(f"  failed: {e}")

    results = []
    for tnum, ws in by_trial.items():
        if "2022" not in ws or "2026_mar_jun" not in ws: continue
        r22 = ws["2022"]; rmj = ws["2026_mar_jun"]
        results.append({
            "trial": tnum, "train_rank": r22["train_rank"], "train_score": r22["train_score"],
            "2022_pnl": r22["forward_pnl"], "2022_n": r22["n_trades"], "2022_wr": r22["wr"], "2022_pf": r22["pf"],
            "26mj_pnl": rmj["forward_pnl"], "26mj_n": rmj["n_trades"], "26mj_wr": rmj["wr"], "26mj_pf": rmj["pf"],
            "forward_pnl": r22["forward_pnl"] + rmj["forward_pnl"],
        })
    results.sort(key=lambda r: -r["forward_pnl"])

    print(f"\n{'='*112}")
    print(f"  TOP 15 by forward_pnl (sum of 2022 + 2026 Mar-Jun)")
    print(f"{'='*112}")
    print(f"  {'rk':<3} {'trial':<5} {'train_rk':<8} {'train$':>10} "
          f"{'2022$':>11} {'2022_n':>7} {'2022_PF':>7} {'26mj$':>11} {'26mj_n':>6} {'26mj_PF':>7} {'fwd$':>12}")
    for i, r in enumerate(results[:15]):
        print(f"  {i+1:<3} #{r['trial']:<4} {r['train_rank']:<8} "
              f"${r['train_score']/1_000_000:>7.1f}M "
              f"${r['2022_pnl']:>+9,.0f} {r['2022_n']:>7} {r['2022_pf']:>6.2f} "
              f"${r['26mj_pnl']:>+9,.0f} {r['26mj_n']:>6} {r['26mj_pf']:>6.2f} "
              f"${r['forward_pnl']:>+10,.0f}")

    print(f"\nCorrelations with forward_pnl:")
    for f in ["train_score", "train_rank", "2022_pnl", "26mj_pnl", "2022_pf", "26mj_pf"]:
        c = correlate(results, f, "forward_pnl")
        if c is not None: print(f"  {f:<18} Pearson = {c:>+.3f}")

    pos = sum(1 for r in results[:TOP_N] if r["forward_pnl"] > 0)
    print(f"\n  Top {TOP_N} (by train) with positive forward: {pos}/{TOP_N} ({100*pos/TOP_N:.0f}%)")

    # Compare to #511 G baseline on same windows (apples-to-apples)
    print(f"\n{'='*112}")
    print(f"  Reference: #511 G alone on same OOS windows")
    print(f"{'='*112}")
    p511_full = {**json.load(open(BASELINE)), **json.load(open(W21B_DEPLOY))["params"]}
    for s in ALL_STRATS: p511_full[f"enable_{s}"] = (s == "g")
    ref_trial = {"number": 511, "train_rank": 0, "score": 47497984.0, "params": {}}
    from test_full import load_all_picks, MARGIN_THRESHOLD
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params

    # Hack: build a "trial" with empty params (no override) so the worker uses p511 + G_ENTRY_LOCKS
    # Actually for #511 reference, use ITS OWN entry params (don't apply g_wide locks)
    def run_ref(dirs_arg, lo, hi):
        set_strategy_params(p511_full)
        tgc.USE_DYNAMIC_SLIPPAGE = True; tgc.USE_MULTIWINDOW_SLIPPAGE = True
        tgc.USE_VOLATILITY_ADJUSTMENT = True; tgc.SLIP_IMPACT_K = 3.0
        tgc.VOL_CAP_PCT = 5.0; tgc.MAX_2MIN_PARTICIPATION = 0.15; tgc.MAX_REGIME_PARTICIPATION = 0.08
        tgc.NEWS_MODULATOR_ENABLED = False; tgc.MARGIN_MULTIPLIER = 1.0
        dirs = [d for d in dirs_arg if os.path.exists(d)]
        all_dates, picks_by_date = load_all_picks(dirs)
        dates = sorted([d for d in all_dates if lo <= d <= hi])
        cash = STARTING_CASH; n = 0
        for d in dates:
            dp = picks_by_date.get(d, []);
            if not dp: continue
            is_cash = cash < MARGIN_THRESHOLD
            try: states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
            except: continue
            for st in states:
                if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
                    n += 1
            cash = end_c + (unset if is_cash else 0)
        return cash - STARTING_CASH, n

    p22, n22 = run_ref(OOS_WINDOWS["2022"]["dirs"], "2022-01-01", "2022-12-31")
    pmj, nmj = run_ref(OOS_WINDOWS["2026_mar_jun"]["dirs"], "2026-03-01", "2026-06-30")
    total_511 = p22 + pmj
    print(f"  #511 G   2022: ${p22:>+12,.0f} ({n22} trades)   2026MJ: ${pmj:>+12,.0f} ({nmj} trades)   TOTAL: ${total_511:>+,.0f}")

    if results:
        top_fwd = results[0]
        print(f"  g_wide best fwd: #{top_fwd['trial']}   2022: ${top_fwd['2022_pnl']:>+12,.0f}   2026MJ: ${top_fwd['26mj_pnl']:>+12,.0f}   TOTAL: ${top_fwd['forward_pnl']:>+,.0f}")
        delta = top_fwd['forward_pnl'] - total_511
        print(f"\n  Δ best g_wide vs #511 G: ${delta:>+,.0f}  ({delta/total_511*100 if total_511 else 0:+.1f}%)")


if __name__ == "__main__":
    main()
