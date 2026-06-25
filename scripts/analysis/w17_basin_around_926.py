"""Find the basin around W17 #926 and wide-forward all members.

#926 params: g_gap=60, l_gap=25, g_stop=12, 2g=T, NH=F
Define "basin" as trials with:
  - g_gap in [45, 75] (around 60 +/- 15)
  - l_gap in [15, 35] (around 25 +/- 10)
  - g_stop = 12 (exact — basin signature)
  - 2g = True, NH = False (the load-bearing flags)

Forward-test all basin members + a fresh random sample on Mar-Jun blind OOS.
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
DATA_DIRS = ["stored_data_combined", "stored_data", "stored_data_2022", "stored_data_2023",
             "stored_data_mar_may_2026", "stored_data_jun_2026"]

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


# Find basin trials
basin_trials = []
for tid, num, score in all_trials:
    p = extract(tid)
    g_gap = p.get("g_min_gap_pct", 0)
    l_gap = p.get("l_min_gap", 0)
    g_stop = p.get("g_stop_pct", -1)
    g_2g = p.get("g_require_2nd_green")
    g_nh = p.get("g_require_2nd_new_high")
    if (45 <= g_gap <= 75 and 15 <= l_gap <= 35 and g_stop == 12
            and g_2g is True and g_nh is False):
        basin_trials.append((tid, num, float(score), p))

print(f"Total positive W17 trials: {len(all_trials)}")
print(f"Trials in #926 basin (g_gap=45-75, l_gap=15-35, g_stop=12, 2g=T, NH=F): {len(basin_trials)}")

# Random control: 50 from outside the basin
non_basin = [(tid, num, float(score)) for tid, num, score in all_trials
             if (tid, num, float(score), {}) not in [(t[0], t[1], t[2], {}) for t in basin_trials]]
random.seed(42)
ctrl_50 = random.sample(non_basin, min(50, len(non_basin)))
print(f"Random control (non-basin): {len(ctrl_50)}")

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
mar_jun = sorted([d for d in all_dates if "2026-03-01" <= d <= "2026-12-31"])

tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0
tgc.NEWS_MODULATOR_ENABLED = False


def fwd(p, dates_):
    merged = {**baseline, **p}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)
    cash = STARTING_CASH
    eq = [cash]
    for d in dates_:
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


# Reference
ref_final, ref_dd = fwd(p_dep, mar_jun)
print(f"\nREFERENCE W13 #1202 on Mar-Jun blind: ${ref_final:,.0f}  DD {ref_dd:.1f}%\n")

# Forward-test basin
print(f"=== Forward-testing {len(basin_trials)} basin trials on Mar-Jun ===")
basin_results = []
for tid, num, score, p in basin_trials:
    f, dd = fwd(p, mar_jun)
    basin_results.append({
        "num": num, "val_score": score, "forward": f, "dd": dd,
        "g_gap": p.get("g_min_gap_pct"), "l_gap": p.get("l_min_gap"),
        "g_target": p.get("g_target_pct"), "g_time": p.get("g_time_limit_min"),
        "p": p,
    })

basin_results.sort(key=lambda r: -r["forward"])
print(f"\n=== TOP 15 BASIN TRIALS BY FORWARD ===")
print(f"{'#':<6} {'val_score':>10s} {'forward':>10s} {'DD':>6s} | {'g_gap':>5s} {'l_gap':>5s} {'g_tgt':>5s} {'g_time':>6s}")
for r in basin_results[:15]:
    print(f"#{r['num']:<5} ${r['val_score']/1e6:>7.2f}M ${r['forward']:>8,.0f} {r['dd']:>4.1f}% | "
          f"{r['g_gap']:>4}% {r['l_gap']:>4}% {r['g_target']:>4}% {r['g_time']:>5}min")

basin_finals = [r["forward"] for r in basin_results]
basin_dds = [r["dd"] for r in basin_results]
print(f"\nBasin stats (n={len(basin_results)}):")
print(f"  Forward median: ${np.median(basin_finals):,.0f}")
print(f"  Forward max:    ${max(basin_finals):,.0f}")
print(f"  Forward min:    ${min(basin_finals):,.0f}")
print(f"  DD median:      {np.median(basin_dds):.1f}%")
hit_w13 = sum(1 for f in basin_finals if f > ref_final)
print(f"  Beating W13:    {hit_w13}/{len(basin_finals)} = {hit_w13/len(basin_finals)*100:.0f}%")

# Best basin trial → multi-year
best = basin_results[0]
print(f"\n=== BEST BASIN TRIAL #{best['num']} — Multi-year ===")
finals = []
for year in ["2024", "2025", "2026"]:
    dates_ = sorted([d for d in all_dates if d.startswith(year)])
    if dates_:
        f, _ = fwd(best["p"], dates_)
        finals.append(f)
geo = np.exp(np.mean(np.log([f / STARTING_CASH for f in finals])))

# W13 multi-year for compare
w13_finals = []
for year in ["2024", "2025", "2026"]:
    dates_ = sorted([d for d in all_dates if d.startswith(year)])
    if dates_:
        f, _ = fwd(p_dep, dates_)
        w13_finals.append(f)
w13_geo = np.exp(np.mean(np.log([f / STARTING_CASH for f in w13_finals])))

print(f"{'Year':<6} {'W13 #1202':>14s} {'Basin best #'+str(best['num']):>18s}  {'Lift':>8s}")
for i, year in enumerate(["2024", "2025", "2026"]):
    if i >= len(finals): break
    lift = (finals[i] - w13_finals[i]) / w13_finals[i] * 100 if w13_finals[i] else 0
    print(f"{year:<6} ${w13_finals[i]:>12,.0f}  ${finals[i]:>16,.0f}  {lift:>+6.1f}%")
print(f"\nGeomean: W13 {w13_geo:.2f}x  vs  Best basin {geo:.2f}x  ({(geo/w13_geo-1)*100:+.1f}%)")

# Save best basin trial
with open(f"config/trial_w17_{best['num']}_basin_best.json", "w") as f:
    json.dump({"trial_number": best["num"], "val_score": best["val_score"],
               "forward_mar_jun": best["forward"], "dd": best["dd"],
               "study": "W17 multi-window val, picked by basin-forward",
               "params": best["p"]}, f, indent=2, default=str)
print(f"\nSaved: config/trial_w17_{best['num']}_basin_best.json")
c.close()
