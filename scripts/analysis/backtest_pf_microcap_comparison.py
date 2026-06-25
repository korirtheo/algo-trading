"""Side-by-side comparison: PF microcap-pump #124 deployment scenarios.

  Scenario A: Solo deploy on ALL 206 days of 2024 (specialist as only config)
  Scenario B: Deploy ONLY on target regime days (microcap-thin + thin-microcap)

Outputs:
  - Comparison equity curve (log + linear)
  - Per-day PnL distribution comparison
  - Summary table
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
OUT_DIR = "results/wf_pf_microcap_pump_noX/charts_comparison"


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def run_backtest(test_dates, picks, filter_to_target=False):
    """Returns (equity_array, daily_dicts)."""
    cash = STARTING_CASH
    eq = [cash]
    daily = []
    for d in test_dates:
        day_picks = picks.get(d, [])
        if not day_picks:
            eq.append(cash); daily.append({"date": d, "shape": "empty", "traded": False, "pnl": 0, "equity": cash})
            continue
        regime = classify_regime(day_picks)
        sig = day_signature(day_picks)
        shape = classify_shape(sig, regime) if sig else "empty"
        if filter_to_target and shape not in TARGET_SHAPES:
            eq.append(cash); daily.append({"date": d, "shape": shape, "traded": False, "pnl": 0, "equity": cash})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            _, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
        except Exception:
            eq.append(cash); daily.append({"date": d, "shape": shape, "traded": False, "pnl": 0, "equity": cash})
            continue
        pnl = end_c - cash; cash = end_c
        if is_cash: cash += unset
        eq.append(cash); daily.append({"date": d, "shape": shape, "traded": True, "pnl": pnl, "equity": cash})
    return np.array(eq), daily


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    s = optuna.load_study(study_name=optuna.get_all_study_names(f"sqlite:///{DB}")[0],
                          storage=f"sqlite:///{DB}")
    b = s.best_trial

    set_strategy_params(_merged(dict(b.params)))
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
    print(f"Loaded {len(test_dates)} 2024 days")

    print("\nRunning Scenario A (all 206 days)...")
    eqA, dailyA = run_backtest(test_dates, picks, filter_to_target=False)
    print("Running Scenario B (filtered to microcap-pump days)...")
    eqB, dailyB = run_backtest(test_dates, picks, filter_to_target=True)

    def _sharpe(eq):
        prev = eq[:-1]; curr = eq[1:]
        mask = prev > 0
        r = np.where(mask, (curr-prev)/prev*100, 0)
        return r.mean()/r.std()*np.sqrt(252) if r.std() > 0 else 0

    def _summary(eq, daily, label):
        final = eq[-1]; pnl = final - STARTING_CASH
        traded = sum(1 for d in daily if d['traded'])
        wins = sum(1 for d in daily if d['traded'] and d['pnl']>0)
        return {
            "label": label, "final": final, "pnl": pnl,
            "traded": traded, "wins": wins,
            "sharpe_pct": _sharpe(eq),
        }

    sA = _summary(eqA, dailyA, "All 206 days (solo deploy)")
    sB = _summary(eqB, dailyB, "Microcap-pump days only (121 days)")

    print(f"\n{'='*72}")
    print("COMPARISON")
    print(f"{'='*72}")
    for s in (sA, sB):
        print(f"\n{s['label']}:")
        print(f"  Final equity:   ${s['final']:,.0f}  ({s['final']/STARTING_CASH:.2f}x)")
        print(f"  Total PnL:      ${s['pnl']:+,.0f}")
        print(f"  Sharpe%:        {s['sharpe_pct']:.2f}")
        print(f"  Days traded:    {s['traded']}/206")
        print(f"  Win days:       {s['wins']}/{s['traded']}  ({100*s['wins']/max(1,s['traded']):.1f}%)")

    print(f"\nDifference: ${sA['pnl'] - sB['pnl']:+,.0f}  "
          f"(scenario A made {(sA['pnl']/sB['pnl']-1)*100:+.1f}% more)")

    # --- CHARTS ---
    # 1. Side-by-side equity curves (log scale)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(eqA)), eqA, color="#d62728", linewidth=2.0,
            label=f"All 206 days (final ${eqA[-1]:,.0f})")
    ax.plot(range(len(eqB)), eqB, color="#2ca02c", linewidth=2.0,
            label=f"Microcap-pump only (final ${eqB[-1]:,.0f})")
    ax.axhline(STARTING_CASH, color="#aaa", linestyle="--", linewidth=0.8,
               label=f"start ${STARTING_CASH:,}")
    ax.set_yscale("log")
    ax.set_title(f"PF microcap-pump #{b.number} forward 2024 — Solo deploy vs Specialist-regime-only")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Equity ($, log)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "equity_comparison_log.png")
    fig.savefig(p1, dpi=140); plt.close(fig)

    # 2. Linear scale
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(eqA)), eqA, color="#d62728", linewidth=2.0,
            label=f"All 206 days (final ${eqA[-1]:,.0f})")
    ax.plot(range(len(eqB)), eqB, color="#2ca02c", linewidth=2.0,
            label=f"Microcap-pump only (final ${eqB[-1]:,.0f})")
    ax.axhline(STARTING_CASH, color="#aaa", linestyle="--", linewidth=0.8,
               label=f"start ${STARTING_CASH:,}")
    ax.set_title(f"PF microcap-pump #{b.number} forward 2024 — Solo deploy vs Specialist-regime-only")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "equity_comparison_linear.png")
    fig.savefig(p2, dpi=140); plt.close(fig)

    # 3. Difference chart: equity_A - equity_B over time
    fig, ax = plt.subplots(figsize=(14, 5))
    diff = eqA - eqB
    ax.plot(range(len(diff)), diff, color="#9467bd", linewidth=1.8)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.fill_between(range(len(diff)), 0, diff,
                     where=(diff > 0), alpha=0.3, color="#d62728",
                     label="Scenario A ahead (off-regime contributing positively)")
    ax.fill_between(range(len(diff)), 0, diff,
                     where=(diff < 0), alpha=0.3, color="#2ca02c",
                     label="Scenario B ahead (off-regime days dragged)")
    ax.set_title(f"Equity difference: All-days minus Specialist-only "
                 f"(positive = off-regime helped, negative = off-regime hurt)")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Equity difference ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "equity_difference.png")
    fig.savefig(p3, dpi=140); plt.close(fig)

    print(f"\nCharts:")
    for p in (p1, p2, p3):
        print(f"  {p}")


if __name__ == "__main__":
    main()
