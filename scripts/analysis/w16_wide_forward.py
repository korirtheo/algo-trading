"""Wide-forward W16: top 50 + random 50 from middle, forward on Mar-Jun BLIND OOS.

Tests whether top-by-validation-score correlates with blind-OOS performance
(Pearson), and whether the broader basin produces ANY trials that beat W13 #1202.
"""
import json, os, sys
import numpy as np
import psycopg2
import random
sys.path.insert(0, '.')

STARTING_CASH = 25_000
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

with open("config/trial_w13_1202_deploy.json") as f:
    p_dep = json.load(f)["params"]
with open("config/trial_432_params.json") as f:
    baseline = json.load(f)

c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w16')
cur = c.cursor()
cur.execute("""SELECT t.trial_id, t.number, tv.value FROM trials t
               JOIN trial_values tv ON tv.trial_id=t.trial_id
               WHERE t.state='COMPLETE'
               AND t.study_id=(SELECT study_id FROM studies WHERE study_name='w16_gl_val_obj')
               ORDER BY tv.value DESC""")
all_trials = cur.fetchall()
print(f"Total W16 complete trials with positive score: {sum(1 for _,_,v in all_trials if v > 0)}")
print(f"Total complete: {len(all_trials)}")


def extract(tid):
    cur.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s", (tid,))
    p = {}
    for n, v, d in cur.fetchall():
        try:
            d_ = json.loads(d) if d else {}
            if d_.get("name") == "CategoricalDistribution":
                ch = d_["attributes"]["choices"]
                p[n] = ch[int(v)] if int(v) < len(ch) else v
            elif "Int" in d_.get("name", ""):
                p[n] = int(v)
            else:
                p[n] = float(v)
        except Exception:
            p[n] = v
    return p


random.seed(42)
positive_trials = [t for t in all_trials if t[2] > 0]
top_50 = positive_trials[:50]
middle_pool = positive_trials[100:min(800, len(positive_trials))]
random_50 = random.sample(middle_pool, min(50, len(middle_pool)))
sample = top_50 + random_50
print(f"Sample: {len(top_50)} top + {len(random_50)} random = {len(sample)}")

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
mar_jun = sorted([d for d in all_dates if "2026-03-01" <= d <= "2026-12-31"])
print(f"Blind OOS window: {len(mar_jun)} days\n")

tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0
tgc.NEWS_MODULATOR_ENABLED = False


def fwd(p):
    merged = {**baseline, **p}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)
    cash = STARTING_CASH
    eq = [cash]
    for d in mar_jun:
        dp = picks_by_date.get(d, [])
        if not dp:
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            continue
        cash = end_c + (unset if is_cash else 0)
        eq.append(cash)
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = float((eq_arr - peak).min() / peak[(eq_arr - peak).argmin()] * 100) if peak.max() > 0 else 0
    return cash, dd


ref_final, ref_dd = fwd(p_dep)
print(f"REFERENCE W13 #1202 deployed on Mar-Jun OOS: ${ref_final:,.0f}  DD {ref_dd:.1f}%\n")

results = []
for i, (tid, num, score) in enumerate(sample):
    p = extract(tid)
    f, dd = fwd(p)
    is_top = i < len(top_50)
    results.append({
        "num": num, "val_score": float(score), "forward": f, "dd": dd, "is_top": is_top,
        "g_stop": p.get("g_stop_pct"), "g_2g": p.get("g_require_2nd_green"),
        "g_nh": p.get("g_require_2nd_new_high"), "g_gap": p.get("g_min_gap_pct"),
        "l_gap": p.get("l_min_gap"),
    })

val_arr = np.array([r["val_score"] for r in results])
fwd_arr = np.array([r["forward"] for r in results])
pearson = float(np.corrcoef(val_arr, fwd_arr)[0, 1]) if len(set(val_arr.tolist())) > 1 else 0.0
hit_rate = sum(1 for r in results if r["forward"] > ref_final) / len(results) * 100

print(f"=== WIDE-FORWARD STATS ===")
print(f"Pearson val_score -> forward:  {pearson:.3f}")
print(f"% trials beating W13 deployed: {hit_rate:.1f}%  ({sum(1 for r in results if r['forward']>ref_final)}/{len(results)})")
print(f"Forward median: ${np.median(fwd_arr):,.0f}")
print(f"Forward max:    ${fwd_arr.max():,.0f}")
print(f"Forward min:    ${fwd_arr.min():,.0f}")
print(f"Forward mean:   ${np.mean(fwd_arr):,.0f}")

results.sort(key=lambda r: -r["forward"])
print(f"\n=== TOP 15 BY FORWARD ===")
print(f"{'#':<6} {'src':<5} {'val_score':>10s} {'forward':>10s} {'DD':>6s} | "
      f"{'g_gap':>5s} {'g_stop':>6s} {'l_gap':>5s} {'2g':>4s} {'NH':>4s}")
for r in results[:15]:
    src = "TOP" if r["is_top"] else "RAND"
    print(f"#{r['num']:<5} {src:<5} ${r['val_score']/1e6:>7.1f}M ${r['forward']:>8,.0f} {r['dd']:>4.1f}% | "
          f"{r['g_gap']:>4}% {r['g_stop']:>5}% {r['l_gap']:>4}% "
          f"{'T' if r['g_2g'] else 'F':>4s} {'T' if r['g_nh'] else 'F':>4s}")

# Distribution check: how often does TPE's basin look like W16 #1078 vs different?
print(f"\n=== Top-50 by val: g_stop=0 / 2g=False frequency ===")
top50_results = [r for r in results if r["is_top"]]
n_no_stop = sum(1 for r in top50_results if (r["g_stop"] or 0) <= 1)
n_no_2g = sum(1 for r in top50_results if r["g_2g"] is False)
print(f"  no-stop (g_stop<=1): {n_no_stop}/{len(top50_results)}")
print(f"  2nd_green=False:     {n_no_2g}/{len(top50_results)}")

# Best forward — save it
best = results[0]
best_num = best["num"]
print(f"\n>>> Best by forward: #{best_num} = ${best['forward']:,.0f}")
print(f"    Beats W13 deployed by {(best['forward']-ref_final)/ref_final*100:+.1f}%" if ref_final else "")

# Save
out = {
    "ref_w13_deployed_forward": ref_final, "ref_w13_dd": ref_dd,
    "pearson_val_fwd": pearson, "hit_rate_pct": hit_rate,
    "forward_median": float(np.median(fwd_arr)), "forward_max": float(fwd_arr.max()),
    "forward_min": float(fwd_arr.min()),
    "top_by_fwd": [{
        "num": r["num"], "val_score": r["val_score"], "forward": r["forward"], "dd": r["dd"],
        "g_gap": r["g_gap"], "g_stop": r["g_stop"], "l_gap": r["l_gap"],
        "g_2g": r["g_2g"], "g_nh": r["g_nh"],
    } for r in results[:20]],
}
os.makedirs("results", exist_ok=True)
with open("results/w16_wide_forward.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved -> results/w16_wide_forward.json")
c.close()
