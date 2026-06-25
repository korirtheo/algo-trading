"""Forward-test current W17 leader #980 on blind Mar-Jun OOS + multi-year.

Run before W17 finishes — gives early read on whether multi-window val produced
a config that beats W13 #1202 on blind OOS (a test the W16 design failed).
"""
import json
import os
import sys
import numpy as np
import psycopg2

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


def extract(num):
    cur.execute("""SELECT trial_id FROM trials WHERE number=%s
                   AND study_id=(SELECT study_id FROM studies WHERE study_name='w17_gl_multi_val')
                   AND state='COMPLETE'""", (num,))
    tid = cur.fetchone()[0]
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
    # Get user_attrs for diagnostic
    cur.execute("""SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s
                   AND key IN ('train_pnl','train_pf','train_wr','val1_pnl','val2_pnl','val3_pnl',
                               'val1_pf','val2_pf','val3_pf','val1_wr','val2_wr','val3_wr')""", (tid,))
    attrs = {k: json.loads(v) if v else None for k, v in cur.fetchall()}
    return p, attrs


p_980, attrs_980 = extract(980)
print("=== W17 #980 train/val metrics (from optuna) ===")
print(f"  train: pnl=${attrs_980.get('train_pnl', 0):>12,.0f}  pf={attrs_980.get('train_pf')}  wr={attrs_980.get('train_wr')}%")
print(f"  val1 (2024-Q4): pnl=${attrs_980.get('val1_pnl', 0):>10,.0f}  pf={attrs_980.get('val1_pf')}  wr={attrs_980.get('val1_wr')}%")
print(f"  val2 (2025-Q3): pnl=${attrs_980.get('val2_pnl', 0):>10,.0f}  pf={attrs_980.get('val2_pf')}  wr={attrs_980.get('val2_wr')}%")
print(f"  val3 (Jan-Feb 2026): pnl=${attrs_980.get('val3_pnl', 0):>10,.0f}  pf={attrs_980.get('val3_pf')}  wr={attrs_980.get('val3_wr')}%")

print(f"\n=== W17 #980 key params ===")
for k in ["g_min_gap_pct", "g_target_pct", "g_stop_pct", "g_time_limit_min",
          "g_trail_pct", "g_trail_activate_pct", "g_require_2nd_green",
          "g_require_2nd_new_high", "l_min_gap"]:
    w17_v = p_980.get(k)
    w13_v = p_dep.get(k)
    marker = " <-- SAME" if w17_v == w13_v else ""
    print(f"  {k}: W17={w17_v}  vs  W13={w13_v}{marker}")

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)

tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0
tgc.NEWS_MODULATOR_ENABLED = False


def run(p, dates_):
    merged = {**baseline, **p}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)
    cash = STARTING_CASH
    n_g = n_l = 0
    pnl_g = pnl_l = 0.0
    g_wins = l_wins = 0
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
                strat = st.get("strategy", "?")
                if strat == "G":
                    n_g += 1
                    pnl_g += st["pnl"]
                    g_wins += 1 if st["pnl"] > 0 else 0
                elif strat == "L":
                    n_l += 1
                    pnl_l += st["pnl"]
                    l_wins += 1 if st["pnl"] > 0 else 0
        cash = end_c + (unset if is_cash else 0)
        eq.append(cash)
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = float((eq_arr - peak).min() / peak[(eq_arr - peak).argmin()] * 100) if peak.max() > 0 else 0
    g_wr = g_wins / n_g * 100 if n_g else 0
    l_wr = l_wins / n_l * 100 if n_l else 0
    return cash, dd, n_g, pnl_g, g_wr, n_l, pnl_l, l_wr


# BLIND TEST: Mar-Jun 2026
mar_jun = sorted([d for d in all_dates if "2026-03-01" <= d <= "2026-12-31"])
print(f"\n=== BLIND OOS (Mar-Jun 2026, {len(mar_jun)} days) ===")
print(f"{'Config':<25s} {'final':>11s} {'DD':>6s} | {'G n':>4s} {'GWR':>5s} {'G$':>11s} | {'Ln':>3s} {'LWR':>5s} {'L$':>10s}")
print("-" * 130)
for label, p in [("W13 #1202 (deployed)", p_dep), ("W17 #980 (val-obj top)", p_980)]:
    f, dd, ng, pg, gwr, nl, pl, lwr = run(p, mar_jun)
    print(f"{label:<25s} ${f:>9,.0f} {dd:>4.1f}% | {ng:>4d} {gwr:>4.1f}% ${pg:>+9,.0f} | {nl:>3d} {lwr:>4.1f}% ${pl:>+8,.0f}")

# MULTI-YEAR
print(f"\n=== MULTI-YEAR ===")
print(f"{'Year':<6} {'W13 #1202':>15s} {'W17 #980':>15s} {'Lift':>9s}")
finals_w13 = []
finals_w17 = []
for year in ["2024", "2025", "2026"]:
    dates_ = sorted([d for d in all_dates if d.startswith(year)])
    if not dates_:
        continue
    fa, dda, _, _, _, _, _, _ = run(p_dep, dates_)
    fb, ddb, _, _, _, _, _, _ = run(p_980, dates_)
    finals_w13.append(fa)
    finals_w17.append(fb)
    lift = (fb - fa) / fa * 100 if fa else 0
    print(f"{year:<6} ${fa:>13,.0f}  ${fb:>13,.0f}  {lift:>+6.2f}%")
geo_w13 = np.exp(np.mean(np.log([f / STARTING_CASH for f in finals_w13])))
geo_w17 = np.exp(np.mean(np.log([f / STARTING_CASH for f in finals_w17])))
print(f"\nGeomean year mult: W13 {geo_w13:.2f}x  vs  W17 #980 {geo_w17:.2f}x  ({(geo_w17/geo_w13-1)*100:+.1f}%)")
c.close()
