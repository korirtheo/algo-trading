"""Wide-forward W15 BV + multi-year test of W13 #1202 + B+V from #83."""
import json, os, sys
import numpy as np
import psycopg2
import random
sys.path.insert(0, '.')

STARTING_CASH = 25_000
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = ["stored_data_combined", "stored_data", "stored_data_2022", "stored_data_2023",
             "stored_data_mar_may_2026", "stored_data_jun_2026"]

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

with open("config/trial_w13_1202_deploy.json") as f:
    p_dep = json.load(f)["params"]
with open("config/trial_432_params.json") as f:
    baseline = json.load(f)

c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w14')
cur = c.cursor()
cur.execute("""SELECT t.trial_id, t.number, tv.value FROM trials t
               JOIN trial_values tv ON tv.trial_id=t.trial_id
               WHERE t.state='COMPLETE'
               AND t.study_id=(SELECT study_id FROM studies WHERE study_name='w15_bv_joint_3yr')
               ORDER BY tv.value DESC""")
all_trials = cur.fetchall()
print(f"Total W15 BV complete: {len(all_trials)}")


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
top_20 = all_trials[:20]
middle_pool = all_trials[30:min(80, len(all_trials))]
random_20 = random.sample(middle_pool, min(20, len(middle_pool)))
sample = top_20 + random_20
print(f"Sample: {len(top_20)} top + {len(random_20)} random = {len(sample)}")

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


def fwd_gl_only(dates_):
    merged = {**baseline, **p_dep}
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


def fwd_layered(p_w15, dates_):
    bv_params = {k: v for k, v in p_w15.items() if k.startswith(("b_", "v_", "priority_b", "priority_v"))}
    merged = {**baseline, **p_dep, **bv_params}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l", "b", "v"})
    set_strategy_params(merged)
    cash = STARTING_CASH
    per = {S: {"n": 0, "pnl": 0.0} for S in ["G", "L", "V", "B"]}
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
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                S = st.get("strategy", "?")
                if S in per:
                    per[S]["n"] += 1
                    per[S]["pnl"] += st["pnl"]
        cash = end_c + (unset if is_cash else 0)
        eq.append(cash)
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = float((eq_arr - peak).min() / peak[(eq_arr - peak).argmin()] * 100) if peak.max() > 0 else 0
    return cash, dd, per


ref_mj_dep, ref_dd_dep = fwd_gl_only(mar_jun)
print(f"\nREFERENCE W13 #1202 G+L only on Mar-Jun OOS: ${ref_mj_dep:,.0f} DD {ref_dd_dep:.1f}%\n")

print("=== WIDE-FORWARD: W13 #1202 + B+V from each W15 trial ===")
results = []
for i, (tid, num, score) in enumerate(sample):
    p = extract(tid)
    f, dd, per = fwd_layered(p, mar_jun)
    delta = f - ref_mj_dep
    results.append({
        "num": num, "train": float(score), "forward": f, "dd": dd,
        "is_top": i < len(top_20), "B_n": per["B"]["n"], "V_n": per["V"]["n"],
        "B_pnl": per["B"]["pnl"], "V_pnl": per["V"]["pnl"], "delta": delta,
    })

train_vals = np.array([r["train"] for r in results])
fwd_vals = np.array([r["forward"] for r in results])
if len(set(train_vals.tolist())) > 1:
    pearson = float(np.corrcoef(train_vals, fwd_vals)[0, 1])
else:
    pearson = 0
hit_rate = sum(1 for r in results if r["forward"] > ref_mj_dep) / len(results) * 100
print(f"Pearson train->forward: {pearson:.3f}")
print(f"% beating W13 #1202: {hit_rate:.0f}% ({sum(1 for r in results if r['delta']>0)}/{len(results)})")
print(f"Forward median: ${np.median(fwd_vals):,.0f}")
print(f"Forward max:    ${fwd_vals.max():,.0f}")

results.sort(key=lambda r: -r["forward"])
print("\n=== TOP 10 BY FORWARD (layered) ===")
print(f"{'#':<5} {'src':<5} {'train':>8s} {'forward':>10s} {'delta':>9s} {'DD':>6s} | {'B n':>3s} {'B $':>9s} {'V n':>3s} {'V $':>9s}")
for r in results[:10]:
    src = "TOP" if r["is_top"] else "RAND"
    print(f"#{r['num']:<4} {src:<5} ${r['train']/1e6:>5.1f}M ${r['forward']:>8,.0f} ${r['delta']:>+7,.0f} {r['dd']:>4.1f}% | "
          f"{r['B_n']:>3d} ${r['B_pnl']:>+7,.0f} {r['V_n']:>3d} ${r['V_pnl']:>+7,.0f}")

# Multi-year on #83
tid_83 = [s for s in all_trials if s[1] == 83][0][0]
p83 = extract(tid_83)
print("\n\n=== MULTI-YEAR: W13 #1202 + B+V from #83 vs W13 #1202 alone ===")
finals_gl = []
finals_bv = []
print(f"{'Year':<6} {'W13 G+L only':>22s} {'+ B+V from #83':>26s} {'Lift':>10s}")
for year in ["2024", "2025", "2026"]:
    dates_ = sorted([d for d in all_dates if d.startswith(year)])
    if not dates_:
        continue
    f_gl, dd_gl = fwd_gl_only(dates_)
    f_bv, dd_bv, per = fwd_layered(p83, dates_)
    finals_gl.append(f_gl)
    finals_bv.append(f_bv)
    lift = (f_bv - f_gl) / f_gl * 100 if f_gl else 0
    print(f"{year:<6} ${f_gl:>13,.0f} ({dd_gl:>4.1f}%) ${f_bv:>13,.0f} ({dd_bv:>4.1f}%)  {lift:>+7.2f}%")

geo_gl = np.exp(np.mean(np.log([f / STARTING_CASH for f in finals_gl])))
geo_bv = np.exp(np.mean(np.log([f / STARTING_CASH for f in finals_bv])))
print(f"\nGeomean year mult: G+L only {geo_gl:.2f}x  vs  + B+V {geo_bv:.2f}x  ({(geo_bv/geo_gl-1)*100:+.1f}%)")

c.close()
