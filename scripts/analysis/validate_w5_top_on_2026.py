"""Forward-validate top W5 trials on 2026 (truly blind year).

Runs each top W5 candidate + #124 baseline on all 2026 days through the
data we have (Jan-Jun 17). Generates equity curves and a summary table.

Output:
  - results/w5_phase1a/blind_2026/equity_curves.png
  - results/w5_phase1a/blind_2026/normalized_curves.png
  - results/w5_phase1a/blind_2026/comparison_table.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

# Configs to validate
CANDIDATES = [
    ("#124 W3 (deployed)",  "config/trial_124_microcap_pump_extracted.json"),
    ("#312 W5 (new peak)",  "config/trial_312_w5_extracted.json"),
    ("#342 W5 (latest)",    "config/trial_342_w5_extracted.json"),
    ("#310 W5 (mid)",       "config/trial_310_w5_extracted.json"),
]

BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000
OUT_DIR = "results/w5_phase1a/blind_2026"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def run_year_2026(config_path, label):
    """Run a config on all 2026 days, return per-day equity and trades."""
    with open(config_path) as f:
        cfg = json.load(f)
    params_in = cfg.get("params", cfg)
    merged = _merged(params_in)
    set_strategy_params(merged)
    # Simulator settings (consistent across all candidates)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    dirs_present = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs_present)
    dates_2026 = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    eq = [cash]
    daily = []
    trades = []
    for d in dates_2026:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            eq.append(cash)
            daily.append({"date": d, "pnl": 0, "equity": cash, "n_trades": 0})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        starting = cash
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception as e:
            print(f"  [warn] {label} {d}: {e}")
            eq.append(cash)
            daily.append({"date": d, "pnl": 0, "equity": cash, "n_trades": 0})
            continue
        n_today = 0
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_today += 1
                trades.append({"date": d, "ticker": st["ticker"],
                               "strategy": st.get("strategy"),
                               "pnl": st["pnl"],
                               "pct": (st["pnl"]/st["position_cost"]*100) if st["position_cost"]>0 else 0,
                               "reason": st.get("exit_reason")})
        cash = end_c + (unset if is_cash else 0)
        day_pnl = cash - starting
        eq.append(cash)
        daily.append({"date": d, "pnl": day_pnl, "equity": cash, "n_trades": n_today})

    return {
        "label": label,
        "config": config_path,
        "dates": dates_2026,
        "daily": daily,
        "equity": np.array(eq),
        "trades": trades,
        "final": cash,
        "multiplier": cash / STARTING_CASH,
        "total_trades": len(trades),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for label, path in CANDIDATES:
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found")
            continue
        print(f"\n--- Validating {label} on 2026 ---")
        r = run_year_2026(path, label)
        results.append(r)
        win_days = sum(1 for d in r['daily'] if d['pnl'] > 0)
        loss_days = sum(1 for d in r['daily'] if d['pnl'] < 0)
        idle_days = sum(1 for d in r['daily'] if d['n_trades'] == 0)
        print(f"  Final: ${r['final']:,.0f} ({r['multiplier']:.2f}x)  "
              f"Trades: {r['total_trades']}  Win/Loss/Idle days: {win_days}/{loss_days}/{idle_days}")

    # --- Equity curve chart (log) ---
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"#124 W3 (deployed)":"#666", "#310 W5 (best)":"#d62728",
              "#307 W5":"#ff7f0e", "#268 W5":"#2ca02c"}
    for r in results:
        ax.plot(range(len(r['equity'])), r['equity'],
                color=colors.get(r['label'], '#999'),
                linewidth=2.5,
                label=f"{r['label']}: ${r['final']:,.0f} ({r['multiplier']:.2f}x)  [{r['total_trades']} trades]")
    ax.axhline(STARTING_CASH, color='gray', linestyle='--', alpha=0.5,
               label=f"start ${STARTING_CASH:,}")
    ax.set_yscale('log')
    ax.set_title("Blind 2026 forward validation — W5 top trials vs #124 (deployed)\n"
                 "Train: 2022-2025 (W5) or 2021-2023 (W3 #124); Test: 2026 (truly blind to both)",
                 fontsize=13)
    ax.set_xlabel("Trading day index (within 2026, ~Jan-Jun 17)")
    ax.set_ylabel("Equity ($, log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "equity_curves_log.png")
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"\nWrote {p1}")

    # --- Linear scale ---
    fig, ax = plt.subplots(figsize=(14, 7))
    for r in results:
        ax.plot(range(len(r['equity'])), r['equity'],
                color=colors.get(r['label'], '#999'),
                linewidth=2.5,
                label=f"{r['label']}: ${r['final']:,.0f} ({r['multiplier']:.2f}x)")
    ax.axhline(STARTING_CASH, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Blind 2026 forward validation (linear scale)")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "equity_curves_linear.png")
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"Wrote {p2}")

    # --- Bar chart of finals ---
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [r['label'] for r in results]
    finals = [r['final'] for r in results]
    bars = ax.bar(labels, finals, color=[colors.get(l,'#999') for l in labels])
    ax.axhline(STARTING_CASH, color='gray', linestyle='--', label=f"start ${STARTING_CASH:,}")
    for r, b in zip(results, bars):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                f"${r['final']:,.0f}\n{r['multiplier']:.2f}x", ha='center',
                fontsize=11, fontweight='bold')
    ax.set_title("Final 2026 equity (blind)", fontsize=13)
    ax.set_ylabel("Final equity ($)")
    ax.grid(True, axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "final_equity_bars.png")
    fig.savefig(p3, dpi=140); plt.close(fig)
    print(f"Wrote {p3}")

    # --- Summary table ---
    summary_path = os.path.join(OUT_DIR, "comparison_table.txt")
    with open(summary_path, "w") as f:
        f.write(f"{'='*100}\n")
        f.write(f"BLIND 2026 FORWARD VALIDATION SUMMARY\n")
        f.write(f"{'='*100}\n")
        f.write(f"  {'label':<24} {'final':>12} {'multi':>7} {'PnL':>12} {'trades':>7} {'win days':>10} {'WR':>6}\n")
        for r in results:
            win_days = sum(1 for d in r['daily'] if d['pnl'] > 0)
            tot_days = sum(1 for d in r['daily'] if d['n_trades'] > 0)
            wr = (win_days/tot_days*100) if tot_days else 0
            pnl = r['final'] - STARTING_CASH
            f.write(f"  {r['label']:<24} ${r['final']:>10,.0f} {r['multiplier']:>5.2f}x "
                    f"${pnl:>+10,.0f} {r['total_trades']:>7} {win_days:>5}/{tot_days:<3}  {wr:>5.1f}%\n")

    with open(summary_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
