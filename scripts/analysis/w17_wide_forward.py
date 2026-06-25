"""Wide-forward W17 (top 50 + random 50) on blind Mar-Jun OOS.

Critical: tests whether multi-window val produces a basin that beats W13 #1202
on the held-out window, not just one outlier trial.
"""
import json
import os
import sys
import numpy as np
import psycopg2
import random

sys.path.insert(0, '.')

STARTING_CASH = 25_000
ALL_STRATS = ["h", "g", "a", "f", "d", "v", "p", "m", "r", "w", "o", "b", "k", "c", "s", "e", "i", "j", "n", "l", "x"]
DATA_DIRS = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

with open("config/trial_w13_1202_deploy.json") as f:
    p_dep = json.load(f)["params"]
with open("config/trial_432_params.json") as f:
    baseline = json.load(f)

c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w17')
cur = c.cursor()
cur.execute("""SELECT t.trial_id, t.number, tv.value FROM trials t
               JOIN trial_values tv ON tv.trial_id=t.trial_id
               WHERE t.state='COMPLETE' AND tv.value > 0
               AND t.study_id=(SELECT study_id FROM studies WHERE study_name='w17_gl_multi_val')
               ORDER BY tv.value DESC""")
all_trials = cur.fetchall()
print(f"Total positive trials: {len(all_trials)}")


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
top_50 = all_trials[:50]
middle_pool = all_trials[100:min(800, len(all_trials))]
random_50 = random.sample(middle_pool, min(50, len(middle_pool)))
sample = top_50 + random_50
print(f"Sample: {len(top_50)} top + {len(random_50)} random = {len(sample)}")

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
mar_jun = sorted([d for d in all_dates if "2026-03-01" <= d <= "2026-12-31"])
print(f"Blind OOS: {len(mar_jun)} days\n")

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
print(f"REFERENCE W13 #1202 on blind Mar-Jun OOS: ${ref_final:,.0f}  DD {ref_dd:.1f}%\n")

results = []
for i, (tid, num, score) in enumerate(sample):
    p = extract(tid)
    f, dd = fwd(p)
    results.append({
        "num": num, "val_score": float(score), "forward": f, "dd": dd,
        "is_top": i < len(top_50),
        "g_stop": p.get("g_stop_pct"), "g_gap": p.get("g_min_gap_pct"),
        "l_gap": p.get("l_min_gap"), "g_2g": p.get("g_require_2nd_green"),
        "g_nh": p.get("g_require_2nd_new_high"),
    })

val_arr = np.array([r["val_score"] for r in results])
fwd_arr = np.array([r["forward"] for r in results])
pearson = float(np.corrcoef(val_arr, fwd_arr)[0, 1]) if len(set(val_arr.tolist())) > 1 else 0
hit_rate = sum(1 for r in results if r["forward"] > ref_final) / len(results) * 100

print(f"=== WIDE-FORWARD STATS ===")
print(f"Pearson val_score -> forward:  {pearson:+.3f}")
print(f"% trials beating W13:          {hit_rate:.1f}% ({sum(1 for r in results if r['forward']>ref_final)}/{len(results)})")
print(f"Forward median: ${np.median(fwd_arr):,.0f}")
print(f"Forward max:    ${fwd_arr.max():,.0f}")
print(f"Forward mean:   ${np.mean(fwd_arr):,.0f}")

results.sort(key=lambda r: -r["forward"])
print(f"\n=== TOP 15 BY FORWARD ===")
print(f"{'#':<6} {'src':<5} {'val_score':>10s} {'forward':>10s} {'DD':>6s} | "
      f"{'g_gap':>5s} {'g_stop':>6s} {'l_gap':>5s} {'2g':>4s} {'NH':>4s}")
for r in results[:15]:
    src = "TOP" if r["is_top"] else "RAND"
    print(f"#{r['num']:<5} {src:<5} ${r['val_score']/1e6:>7.2f}M ${r['forward']:>8,.0f} {r['dd']:>4.1f}% | "
          f"{r['g_gap']:>4}% {r['g_stop']:>5}% {r['l_gap']:>4}% "
          f"{'T' if r['g_2g'] else 'F':>4s} {'T' if r['g_nh'] else 'F':>4s}")

best = results[0]
print(f"\n>>> Best by forward: #{best['num']} = ${best['forward']:,.0f}")
if best['forward'] > ref_final:
    print(f"    BEATS W13 by {(best['forward']-ref_final)/ref_final*100:+.1f}%")
else:
    print(f"    Loses to W13 by {(ref_final-best['forward'])/best['forward']*100:+.1f}%")

c.close()
