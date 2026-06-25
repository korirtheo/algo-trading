"""3-way 2026 backtest chart: W13 (deployed) vs W21 #19 vs W21b #511."""
import json
import os
import sys
import numpy as np
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime

sys.path.insert(0, '.')

STARTING_CASH = 25_000
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data_2026_gap_fill"]

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

with open("config/trial_w13_1202_deploy.json") as f:
    p_dep = json.load(f)["params"]
with open("config/trial_w21_19_extracted.json") as f:
    w21_19 = json.load(f)["params"]
with open("config/trial_432_params.json") as f:
    baseline = json.load(f)

c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w21b')
cur = c.cursor()
cur.execute("SELECT trial_id FROM trials WHERE number=511 AND state='COMPLETE' AND study_id=(SELECT study_id FROM studies WHERE study_name='w21b_target100')")
tid = cur.fetchone()[0]
cur.execute("SELECT param_name, param_value FROM trial_params WHERE trial_id=%s", (tid,))
w21b_511 = {k: float(v) if k != 'g_time_limit_min' else int(float(v)) for k, v in cur.fetchall()}
cur.execute("SELECT trial_id FROM trials WHERE number=561 AND state='COMPLETE' AND study_id=(SELECT study_id FROM studies WHERE study_name='w21b_target100')")
tid = cur.fetchone()[0]
cur.execute("SELECT param_name, param_value FROM trial_params WHERE trial_id=%s", (tid,))
w21b_561 = {k: float(v) if k != 'g_time_limit_min' else int(float(v)) for k, v in cur.fetchall()}
c.close()
print(f"W21b #511 params: {w21b_511}")
print(f"W21b #561 params: {w21b_561}")

tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0
tgc.NEWS_MODULATOR_ENABLED = False

dirs = [d for d in DATA_DIRS if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
dates_2026 = sorted([d for d in all_dates if d.startswith("2026")])


def run_backtest(label, params_override):
    merged = {**baseline, **p_dep, **params_override}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)
    cash = STARTING_CASH
    dates_curve = []
    eq_curve = []
    daily_pnl = []
    trades = []
    for d in dates_2026:
        dp = picks_by_date.get(d, [])
        if not dp:
            dates_curve.append(datetime.strptime(d, "%Y-%m-%d"))
            eq_curve.append(cash)
            daily_pnl.append(0)
            continue
        is_cash = cash < MARGIN_THRESHOLD
        pre_cash = cash
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            dates_curve.append(datetime.strptime(d, "%Y-%m-%d"))
            eq_curve.append(cash)
            daily_pnl.append(0)
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                trades.append({"date": d, "strategy": st.get("strategy"), "pnl": st["pnl"]})
        cash = end_c + (unset if is_cash else 0)
        dates_curve.append(datetime.strptime(d, "%Y-%m-%d"))
        eq_curve.append(cash)
        daily_pnl.append(cash - pre_cash)
    eq_arr = np.array(eq_curve)
    peak = np.maximum.accumulate(eq_arr)
    dd_arr = (eq_arr - peak) / peak * 100
    max_dd = float(dd_arr.min())
    max_dd_idx = int(dd_arr.argmin())
    g_trades = [t for t in trades if t["strategy"] == "G"]
    l_trades = [t for t in trades if t["strategy"] == "L"]
    g_wins = sum(1 for t in g_trades if t["pnl"] > 0)
    l_wins = sum(1 for t in l_trades if t["pnl"] > 0)
    return {
        "label": label, "dates": dates_curve, "eq": eq_arr, "dd_arr": dd_arr, "daily_pnl": daily_pnl,
        "max_dd": max_dd, "max_dd_idx": max_dd_idx,
        "n_g": len(g_trades), "g_wr": g_wins / len(g_trades) * 100 if g_trades else 0,
        "n_l": len(l_trades), "l_wr": l_wins / len(l_trades) * 100 if l_trades else 0,
        "final": float(cash),
    }


print("Running W13 #1202...")
res_w13 = run_backtest("W13", {})
print(f"  Final: ${res_w13['final']:,.0f}")
print("Running W21 #19 (target=40 cap)...")
res_w21 = run_backtest("W21", w21_19)
print(f"  Final: ${res_w21['final']:,.0f}")
print("Running W21b #511 (target=62, time=12)...")
res_w21b = run_backtest("W21b", w21b_511)
print(f"  Final: ${res_w21b['final']:,.0f}")
print("Running W21b #561 (target=64, time=9)...")
res_561 = run_backtest("W21b_561", w21b_561)
print(f"  Final: ${res_561['final']:,.0f}")

fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [3, 1]})
ax1 = axes[0]
ax1.plot(res_w13["dates"], res_w13["eq"], linewidth=1.5, color='#1f77b4',
         label=f'W13 #1202 (deployed) - Final ${res_w13["final"]:,.0f}', alpha=0.85)
ax1.plot(res_w21["dates"], res_w21["eq"], linewidth=1.8, color='#d62728',
         label=f'W21 #19 (target=40 cap) - Final ${res_w21["final"]:,.0f}', alpha=0.85)
ax1.plot(res_w21b["dates"], res_w21b["eq"], linewidth=2.0, color='#2ca02c',
         label=f'W21b #511 (target=62, time=12, deployed) - Final ${res_w21b["final"]:,.0f}', alpha=0.85)
ax1.plot(res_561["dates"], res_561["eq"], linewidth=2.4, color='#9467bd',
         label=f'W21b #561 (target=64, time=9, micro-basin winner) - Final ${res_561["final"]:,.0f}', alpha=0.95)

ax1.fill_between(res_w13["dates"], STARTING_CASH, res_w13["eq"], alpha=0.06, color='#1f77b4')
ax1.fill_between(res_561["dates"], res_w21b["eq"], res_561["eq"],
                 where=(res_561["eq"] >= res_w21b["eq"]), alpha=0.12, color='#9467bd', interpolate=True)

ax1.axhline(STARTING_CASH, color='gray', linestyle='--', linewidth=0.8, label=f'Start ${STARTING_CASH:,}')
ax1.axvline(datetime(2026, 3, 1), color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
            label='Mar 1 (Blind OOS start)')

for res, color in [(res_w13, '#1f77b4'), (res_w21, '#d62728'), (res_w21b, '#2ca02c'), (res_561, '#9467bd')]:
    ax1.scatter([res["dates"][res["max_dd_idx"]]], [res["eq"][res["max_dd_idx"]]],
                color=color, s=70, zorder=5, edgecolor='black', linewidth=1)

ax1.set_yscale('log')
ax1.set_ylabel('Equity ($, log)', fontsize=11)
ax1.set_title('Full 2026 backtest: W13 deployed vs W21 (cap=40) vs W21b #511 (time=12) vs W21b #561 (time=9)  |  G+L, $25K start',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax1.grid(True, alpha=0.3, which='both')
ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

ax2 = axes[1]
diff = [res_561["daily_pnl"][i] - res_w21b["daily_pnl"][i] for i in range(len(res_w21b["dates"]))]
colors = ['#9467bd' if p >= 0 else '#2ca02c' for p in diff]
ax2.bar(res_561["dates"], diff, color=colors, alpha=0.7, width=1.0)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(datetime(2026, 3, 1), color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
ax2.set_ylabel('#561 - #511 daily PnL ($)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_title('Daily PnL: #561 (time=9) minus #511 (time=12). Purple = #561 beat. Green = #511 beat.', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

stats_text = (
    f"W13 #1202:    ${res_w13['final']:,.0f} ({res_w13['final']/STARTING_CASH:.1f}x)  "
    f"DD {res_w13['max_dd']:.1f}%  G WR {res_w13['g_wr']:.1f}%\n"
    f"W21 #19:      ${res_w21['final']:,.0f} ({res_w21['final']/STARTING_CASH:.1f}x)  "
    f"DD {res_w21['max_dd']:.1f}%  G WR {res_w21['g_wr']:.1f}%\n"
    f"W21b #511:    ${res_w21b['final']:,.0f} ({res_w21b['final']/STARTING_CASH:.1f}x)  "
    f"DD {res_w21b['max_dd']:.1f}%  G WR {res_w21b['g_wr']:.1f}%\n"
    f"W21b #561:    ${res_561['final']:,.0f} ({res_561['final']/STARTING_CASH:.1f}x)  "
    f"DD {res_561['max_dd']:.1f}%  G WR {res_561['g_wr']:.1f}%\n\n"
    f"W21b #511 lift over W13:  {(res_w21b['final']/res_w13['final']-1)*100:+.1f}%\n"
    f"W21b #561 lift over W13:  {(res_561['final']/res_w13['final']-1)*100:+.1f}%\n"
    f"W21b #561 lift over #511: {(res_561['final']/res_w21b['final']-1)*100:+.1f}%"
)
ax1.text(0.99, 0.02, stats_text, transform=ax1.transAxes,
         fontsize=10, family='monospace',
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.92, edgecolor='gray', boxstyle='round'))

fig.tight_layout()
out_path = "results/w13_vs_w21_vs_w21b511_vs_w21b561_2026_extended.png"
fig.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nChart saved: {out_path}")
