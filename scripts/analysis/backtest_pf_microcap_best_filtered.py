"""Backtest PF microcap-pump #124 ONLY on its target regime days in 2024.

Filters 2024 to days where classify_shape(picks) in {microcap-thin, thin-microcap}.
On other days, NO trades fire (cash unchanged, equity flat).
This shows the "specialist deployed only on its target regime" scenario.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape

DB = "results/wf_pf_microcap_pump_noX/W3_train_2021_2022_2023_test_2024.db"
BASELINE = "config/trial_432_params.json"
TEST_DIRS = ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
             "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"]
TARGET_SHAPES = {"microcap-thin", "thin-microcap"}
STARTING_CASH = 25_000
OUT_DIR = "results/wf_pf_microcap_pump_noX/charts_filtered_target_regime"


def _merged_params(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    s = optuna.load_study(study_name=optuna.get_all_study_names(f"sqlite:///{DB}")[0],
                          storage=f"sqlite:///{DB}")
    b = s.best_trial
    a = b.user_attrs
    print(f"Using trial #{b.number}  score=${b.value:,.0f}  IS PF={a.get('pf','-')}")

    set_strategy_params(_merged_params(dict(b.params)))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    dirs = [d for d in TEST_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    test_dates = [d for d in all_dates if d.startswith("2024")]

    cash = STARTING_CASH
    eq = [cash]
    daily = []
    trades = []
    n_traded = 0
    n_skipped = 0
    skip_by_shape = {}
    for d in test_dates:
        day_picks = picks.get(d, [])
        if not day_picks:
            eq.append(cash)
            daily.append({"date": d, "shape": "empty", "traded": False, "pnl": 0, "equity": cash})
            continue
        regime = classify_regime(day_picks)
        sig = day_signature(day_picks)
        shape = classify_shape(sig, regime) if sig else "empty"

        if shape not in TARGET_SHAPES:
            n_skipped += 1
            skip_by_shape[shape] = skip_by_shape.get(shape, 0) + 1
            eq.append(cash)
            daily.append({"date": d, "shape": shape, "traded": False, "pnl": 0, "equity": cash})
            continue

        n_traded += 1
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
        except Exception as e:
            eq.append(cash)
            daily.append({"date": d, "shape": shape, "traded": False, "pnl": 0, "equity": cash})
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                trades.append({
                    "date": d, "ticker": st["ticker"], "strategy": st.get("strategy"),
                    "pnl": st["pnl"], "reason": st.get("exit_reason"),
                })
        pnl = end_c - cash; cash = end_c
        if is_cash: cash += unset
        eq.append(cash)
        daily.append({"date": d, "shape": shape, "traded": True, "pnl": pnl, "equity": cash})

    eq = np.array(eq)
    final = eq[-1]; total_pnl = final - STARTING_CASH

    prev = eq[:-1]; curr = eq[1:]
    mask = prev > 0
    ret = np.where(mask, (curr - prev) / prev * 100, 0)
    nz_ret = ret[ret != 0]
    sharpe_full = ret.mean()/ret.std()*np.sqrt(252) if ret.std() > 0 else 0
    # Sharpe computed only over traded days (more meaningful)
    sharpe_traded = nz_ret.mean()/nz_ret.std()*np.sqrt(252) if len(nz_ret) > 1 and nz_ret.std() > 0 else 0

    print(f"\n{'='*72}")
    print(f"REGIME-FILTERED FORWARD 2024 (microcap-pump days only)")
    print(f"{'='*72}")
    print(f"  Days total:        {len(test_dates)}")
    print(f"  Days traded:       {n_traded}  (microcap-pump days)")
    print(f"  Days skipped:      {n_skipped}  (other regimes — flat)")
    print(f"  Skip breakdown:    {skip_by_shape}")
    print(f"\n  Start equity:      ${STARTING_CASH:,.0f}")
    print(f"  Final equity:      ${final:,.0f}  ({final/STARTING_CASH:.2f}× return)")
    print(f"  Total PnL:         ${total_pnl:+,.0f}")
    print(f"  PnL / traded day:  ${total_pnl/max(1,n_traded):+,.0f}")
    print(f"  Sharpe% (all):     {sharpe_full:.2f}")
    print(f"  Sharpe% (traded):  {sharpe_traded:.2f}")
    print(f"  Total trades:      {len(trades)}")
    wins_traded = sum(1 for r in daily if r['traded'] and r['pnl'] > 0)
    losses_traded = sum(1 for r in daily if r['traded'] and r['pnl'] < 0)
    print(f"  Win days/Traded:   {wins_traded}/{n_traded} ({100*wins_traded/max(1,n_traded):.1f}%)")

    # Per-strategy
    by_s = {}
    for t in trades:
        s = t['strategy']
        if s not in by_s: by_s[s] = {"n":0,"wins":0,"pnl":0.0}
        by_s[s]["n"]+=1; by_s[s]["pnl"]+=t["pnl"]
        if t["pnl"]>0: by_s[s]["wins"]+=1
    print(f"\nPer-strategy:")
    for s,v in sorted(by_s.items(), key=lambda x:-x[1]["pnl"]):
        wr = v["wins"]/v["n"]*100 if v["n"]>0 else 0
        print(f"  {s}  trades={v['n']:>3}  wins={v['wins']:>3} ({wr:>5.1f}%)  pnl=${v['pnl']:>+11,.0f}")

    # CHARTS
    # Equity (log) showing flat periods on skipped days
    fig, ax = plt.subplots(figsize=(13,6))
    ax.plot(range(len(eq)), eq, color="#d62728", linewidth=1.8)
    # Shade traded days
    for i,r in enumerate(daily):
        if r['traded']:
            ax.axvspan(i, i+1, color="#9ecae1", alpha=0.2)
    ax.axhline(STARTING_CASH, color="#aaa", linestyle="--", linewidth=0.8,
               label=f"start ${STARTING_CASH:,}")
    ax.set_yscale("log")
    ax.set_title(f"PF microcap-pump best #{b.number} — forward 2024 (REGIME-FILTERED, log)")
    ax.set_xlabel("Trading day index (blue shade = traded microcap-pump day)")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "equity_log_filtered.png")
    fig.savefig(p1, dpi=140); plt.close(fig)

    # Linear
    fig, ax = plt.subplots(figsize=(13,6))
    ax.plot(range(len(eq)), eq, color="#d62728", linewidth=1.8)
    for i,r in enumerate(daily):
        if r['traded']:
            ax.axvspan(i, i+1, color="#9ecae1", alpha=0.2)
    ax.axhline(STARTING_CASH, color="#aaa", linestyle="--", linewidth=0.8,
               label=f"start ${STARTING_CASH:,}")
    ax.set_title(f"PF microcap-pump best #{b.number} — forward 2024 (REGIME-FILTERED, linear)")
    ax.set_xlabel("Trading day index (blue shade = traded)")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "equity_linear_filtered.png")
    fig.savefig(p2, dpi=140); plt.close(fig)

    # Daily $ PnL on TRADED days only
    traded_pnls = [r['pnl'] for r in daily if r['traded']]
    fig, ax = plt.subplots(figsize=(14,5))
    colors = ["#2ca02c" if p>0 else ("#d62728" if p<0 else "#aaa") for p in traded_pnls]
    ax.bar(range(len(traded_pnls)), traded_pnls, color=colors, alpha=0.85)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_title(f"Daily PnL — TRADED microcap-pump days only ({n_traded} days)")
    ax.set_xlabel("Traded day index (microcap-pump days only)")
    ax.set_ylabel("Daily PnL ($)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "perday_pnl_traded_only.png")
    fig.savefig(p3, dpi=140); plt.close(fig)

    # % return on TRADED days only
    traded_pcts = []
    eq_running = STARTING_CASH
    for r in daily:
        if r['traded']:
            traded_pcts.append((r['pnl']/eq_running*100) if eq_running>0 else 0)
        eq_running = r['equity']
    fig, ax = plt.subplots(figsize=(14,5))
    colors2 = ["#2ca02c" if p>0 else ("#d62728" if p<0 else "#aaa") for p in traded_pcts]
    ax.bar(range(len(traded_pcts)), traded_pcts, color=colors2, alpha=0.85)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_title(f"Daily % return — TRADED days only")
    ax.set_xlabel("Traded day index")
    ax.set_ylabel("Daily % return")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p4 = os.path.join(OUT_DIR, "perday_pct_traded_only.png")
    fig.savefig(p4, dpi=140); plt.close(fig)

    print(f"\nCharts:")
    for p in (p1, p2, p3, p4):
        print(f"  {p}")


if __name__ == "__main__":
    main()
