"""Basin analysis — what differentiates W9's forward-winning region from
its train-winning region?

ChatGPT observation: W9's top-5 by forward have train ranks [25, 22, 29, 26, 27]
— all neighbors. That suggests a discrete basin TPE found through random sampling
but DOESN'T optimize toward. Compare param distributions vs train-top-10.

Also light analysis of W7 vs W8: what structural property lets W7 #254 hit
$200K when W8 caps at $60K?
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import statistics
from collections import defaultdict

import numpy as np
import psycopg2


def fetch_trials_with_full_params(db, trial_numbers=None, top_n=None):
    """Pull selected trials with all params + user_attrs."""
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    if trial_numbers is not None:
        placeholders = ",".join(["%s"] * len(trial_numbers))
        cur.execute(f"""SELECT t.trial_id, t.number, tv.value
                       FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
                       WHERE t.state='COMPLETE' AND t.number IN ({placeholders})""",
                    list(trial_numbers))
    elif top_n is not None:
        cur.execute("""SELECT t.trial_id, t.number, tv.value
                       FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
                       WHERE t.state='COMPLETE' AND tv.value > 0
                       ORDER BY tv.value DESC LIMIT %s""", (top_n,))
    rows = cur.fetchall()
    out = []
    for tid, num, score in rows:
        cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
        ua = dict(cur.fetchall())
        cur.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s", (tid,))
        params = {}
        for name, val, dist in cur.fetchall():
            try:
                d = json.loads(dist) if dist else {}
                kind = d.get("name", "")
                if kind == "CategoricalDistribution":
                    params[name] = d["attributes"]["choices"][int(val)]
                elif "Int" in kind:
                    params[name] = int(val)
                else:
                    params[name] = float(val)
            except Exception:
                params[name] = val
        out.append({"trial_id": tid, "number": num, "score": float(score),
                    "user_attrs": ua, "params": params})
    c.close()
    return out


def get_top_by_train(db, k):
    """Get top k trials by training score (rank 1..k)."""
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    cur.execute("""SELECT t.number FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
                   WHERE t.state='COMPLETE' AND tv.value > 0
                   ORDER BY tv.value DESC LIMIT %s""", (k,))
    out = [r[0] for r in cur.fetchall()]
    c.close()
    return out


def get_rank_range(db, start, end):
    """Get trial numbers with training rank in [start, end] (1-indexed)."""
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    cur.execute("""SELECT t.number FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
                   WHERE t.state='COMPLETE' AND tv.value > 0
                   ORDER BY tv.value DESC OFFSET %s LIMIT %s""", (start - 1, end - start + 1))
    out = [r[0] for r in cur.fetchall()]
    c.close()
    return out


def compare_param_distributions(basin_a, basin_b, label_a, label_b, top_k=20):
    """For each numeric param, compute (mean_b - mean_a) / pooled_std.
    Sorted by abs(z-score). Reveals which params shift the most between basins.
    """
    # Collect params present in both basins
    all_params = set()
    for t in basin_a + basin_b:
        for k, v in t["params"].items():
            if isinstance(v, (int, float)):
                all_params.add(k)

    results = []
    for p in sorted(all_params):
        a_vals = [t["params"].get(p) for t in basin_a
                   if isinstance(t["params"].get(p), (int, float))]
        b_vals = [t["params"].get(p) for t in basin_b
                   if isinstance(t["params"].get(p), (int, float))]
        if len(a_vals) < 3 or len(b_vals) < 3: continue
        a_mean = statistics.mean(a_vals); b_mean = statistics.mean(b_vals)
        try:
            a_std = statistics.stdev(a_vals) if len(a_vals) > 1 else 0
            b_std = statistics.stdev(b_vals) if len(b_vals) > 1 else 0
        except statistics.StatisticsError:
            continue
        pooled = ((a_std ** 2 + b_std ** 2) / 2) ** 0.5
        if pooled < 1e-9:
            z = 0
        else:
            z = (b_mean - a_mean) / pooled
        results.append({
            "param": p, "a_mean": a_mean, "b_mean": b_mean,
            "a_std": a_std, "b_std": b_std, "z": z,
        })
    results.sort(key=lambda r: -abs(r["z"]))

    print(f"\n  Top {top_k} params by separation (|z-score|):")
    print(f"  {'param':<32} {label_a:>14} {label_b:>14} {'shift':>9}")
    print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*9}")
    for r in results[:top_k]:
        amean = f"{r['a_mean']:.3f}"; bmean = f"{r['b_mean']:.3f}"
        sign = "+" if r["z"] > 0 else "-"
        print(f"  {r['param']:<32} {amean:>14} {bmean:>14} {sign}{abs(r['z']):>5.2f}sd")
    return results


def compare_enabled_strategies(basin_a, basin_b, label_a, label_b):
    """Compare which strategies are enabled in each basin."""
    strats = ['h','g','a','f','d','v','p','m','r','w','o','b','k','c','s','e','i','j','n','l','x']
    print(f"\n  Strategy enable rate by basin:")
    print(f"  {'strat':<6} {label_a:>10} {label_b:>10} {'delta':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for s in strats:
        key = f"enable_{s}"
        a_rate = sum(1 for t in basin_a if t["params"].get(key) is True) / max(len(basin_a), 1) * 100
        b_rate = sum(1 for t in basin_b if t["params"].get(key) is True) / max(len(basin_b), 1) * 100
        delta = b_rate - a_rate
        marker = " ***" if abs(delta) > 30 else ""
        print(f"  {s.upper():<6} {a_rate:>9.0f}% {b_rate:>9.0f}% {delta:>+9.0f}pp{marker}")


def compare_user_attrs(basin_a, basin_b, label_a, label_b):
    """Compare key user_attrs (per-year breakdown) between basins."""
    keys = ["pnl_2022", "pnl_2023", "pnl_2024", "pnl_2025",
            "pf_2022", "pf_2023", "pf_2024", "pf_2025",
            "n_2022", "n_2023", "n_2024", "n_2025",
            "min_pf", "min_pnl", "sum_pnl", "n"]
    print(f"\n  User_attrs comparison (mean values):")
    print(f"  {'attr':<14} {label_a:>14} {label_b:>14} {'delta':>14}")
    print(f"  {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    for k in keys:
        a_vals, b_vals = [], []
        for t in basin_a:
            try: a_vals.append(float(t["user_attrs"].get(k)))
            except: pass
        for t in basin_b:
            try: b_vals.append(float(t["user_attrs"].get(k)))
            except: pass
        if not a_vals or not b_vals: continue
        a_mean = statistics.mean(a_vals); b_mean = statistics.mean(b_vals)
        delta = b_mean - a_mean
        a_str = f"{a_mean:.2f}"; b_str = f"{b_mean:.2f}"; d_str = f"{delta:+.2f}"
        print(f"  {k:<14} {a_str:>14} {b_str:>14} {d_str:>14}")


def main():
    print(f"\n{'='*92}")
    print(f"  W9 BASIN ANALYSIS")
    print(f"  Basin A = train ranks 1-10  (high train score, low fwd)")
    print(f"  Basin B = train ranks 20-35 (medium train score, high fwd)")
    print(f"{'='*92}")

    # Get trials at each rank range
    basin_a_nums = get_rank_range("optuna_w9", 1, 10)
    basin_b_nums = get_rank_range("optuna_w9", 20, 35)
    print(f"  Basin A (W9 ranks 1-10):  trial numbers {basin_a_nums}")
    print(f"  Basin B (W9 ranks 20-35): trial numbers {basin_b_nums}")

    basin_a = fetch_trials_with_full_params("optuna_w9", trial_numbers=basin_a_nums)
    basin_b = fetch_trials_with_full_params("optuna_w9", trial_numbers=basin_b_nums)
    print(f"  Fetched {len(basin_a)} Basin A trials, {len(basin_b)} Basin B trials")

    compare_user_attrs(basin_a, basin_b, "BasinA (1-10)", "BasinB (20-35)")
    compare_enabled_strategies(basin_a, basin_b, "BasinA", "BasinB")
    compare_param_distributions(basin_a, basin_b, "BasinA", "BasinB", top_k=25)

    # Now W7 vs W8 ceiling question
    print(f"\n\n{'='*92}")
    print(f"  W7 #254 vs W8 #1433 — single-trial comparison (the ceiling question)")
    print(f"{'='*92}")
    # W7 #254 train rank 3 (from our analysis), W8 winner #1433
    w7_254 = fetch_trials_with_full_params("optuna", trial_numbers=[254])
    w8_winner = fetch_trials_with_full_params("optuna_w8", trial_numbers=[1433])
    if w7_254 and w8_winner:
        print(f"\n  Enabled strategies:")
        strats = ['h','g','a','f','d','v','p','m','r','w','o','b','k','c','s','e','i','j','n','l','x']
        w7_en = [s for s in strats if w7_254[0]["params"].get(f"enable_{s}") is True]
        w8_en = [s for s in strats if w8_winner[0]["params"].get(f"enable_{s}") is True]
        print(f"    W7 #254:  {','.join(s.upper() for s in w7_en)}  ({len(w7_en)} strategies)")
        print(f"    W8 #1433: {','.join(s.upper() for s in w8_en)}  ({len(w8_en)} strategies)")

        print(f"\n  Shared & differing strategies:")
        only_w7 = set(w7_en) - set(w8_en)
        only_w8 = set(w8_en) - set(w7_en)
        shared = set(w7_en) & set(w8_en)
        print(f"    Both:    {','.join(s.upper() for s in shared)}")
        print(f"    Only W7: {','.join(s.upper() for s in only_w7)}")
        print(f"    Only W8: {','.join(s.upper() for s in only_w8)}")

        # Show key stop/target/trail params for shared strategies
        print(f"\n  For shared strategies — stop/target/trail comparison:")
        for s in sorted(shared):
            sU = s.upper()
            for param_suffix in ["target_pct", "stop_pct", "trail_pct", "time_limit_min"]:
                key = f"{s}_{param_suffix}"
                w7v = w7_254[0]["params"].get(key)
                w8v = w8_winner[0]["params"].get(key)
                if w7v is not None or w8v is not None:
                    print(f"    {key:<28}  W7: {w7v}  W8: {w8v}")

    print(f"\n{'='*92}")
    print(f"  KEY QUESTIONS")
    print(f"{'='*92}")
    print(f"  1. What distinguishes W9 Basin B (forward winners) from Basin A?")
    print(f"     -> See top z-score params above")
    print(f"  2. What does W7 #254 enable that W8 #1433 doesn't?")
    print(f"     -> See strategy diff above")
    print(f"  3. Could W10 use Basin B's signature as a regularizer?")


if __name__ == "__main__":
    main()
