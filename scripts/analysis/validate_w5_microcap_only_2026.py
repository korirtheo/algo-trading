"""Same as validate_w5_top_on_2026.py but filtered to microcap-pump days ONLY.

Tests whether W5 candidates win on their training distribution (microcap-thin
+ thin-microcap days), even if they lose on the unrestricted full-year test.

If W5 candidates win here -> a routing strategy might rescue them.
If they still lose -> the overfit is real, no regime can save them.
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
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
from strategies.regime_gate import classify_regime, compute_features

CANDIDATES = [
    ("#124 W3 (deployed)",  "config/trial_124_microcap_pump_extracted.json"),
    ("#312 W5 (new peak)",  "config/trial_312_w5_extracted.json"),
    ("#342 W5 (latest)",    "config/trial_342_w5_extracted.json"),
    ("#310 W5 (mid)",       "config/trial_310_w5_extracted.json"),
]
KEEP_SHAPES = {"microcap-thin", "thin-microcap"}
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000
OUT_DIR = "results/w5_phase1a/blind_2026_microcap_only"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def _classify(picks):
    try:
        regime = classify_regime(compute_features(picks))
    except Exception:
        regime = "NORMAL"
    sig = day_signature(picks)
    if sig is None: return "empty"
    return classify_shape(sig, regime)


def run_year_2026(config_path, label):
    with open(config_path) as f:
        cfg = json.load(f)
    params_in = cfg.get("params", cfg)
    merged = _merged(params_in)
    set_strategy_params(merged)
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
    trades = []
    n_traded = 0
    n_skipped_shape = 0

    for d in dates_2026:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            eq.append(cash); continue

        # SHAPE FILTER: skip non-microcap-pump days
        shape = _classify(day_picks)
        if shape not in KEEP_SHAPES:
            n_skipped_shape += 1
            eq.append(cash)
            continue

        n_traded += 1
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                trades.append({"date": d, "ticker": st["ticker"],
                               "strategy": st.get("strategy"),
                               "pnl": st["pnl"]})
        cash = end_c + (unset if is_cash else 0)
        eq.append(cash)

    return {
        "label": label,
        "equity": np.array(eq),
        "final": cash,
        "multiplier": cash / STARTING_CASH,
        "total_trades": len(trades),
        "n_traded_days": n_traded,
        "n_skipped_shape": n_skipped_shape,
        "trades": trades,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    print(f"=== Microcap-pump-only validation on 2026 ===")
    print(f"  Keeping shapes: {KEEP_SHAPES}\n")
    for label, path in CANDIDATES:
        if not os.path.exists(path):
            print(f"[skip] {label}: {path}"); continue
        print(f"--- {label} ---")
        r = run_year_2026(path, label)
        results.append(r)
        print(f"  Final: ${r['final']:,.0f} ({r['multiplier']:.2f}x)  "
              f"Trades: {r['total_trades']}  Traded days: {r['n_traded_days']}  "
              f"Skipped (wrong shape): {r['n_skipped_shape']}\n")

    # --- Chart ---
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"#124 W3 (deployed)":"#666", "#312 W5 (new peak)":"#d62728",
              "#342 W5 (latest)":"#ff7f0e", "#310 W5 (mid)":"#2ca02c"}
    for r in results:
        ax.plot(range(len(r['equity'])), r['equity'],
                color=colors.get(r['label'],'#999'), linewidth=2.5,
                label=f"{r['label']}: ${r['final']:,.0f} ({r['multiplier']:.2f}x)  "
                      f"[{r['total_trades']} trades, {r['n_traded_days']} days]")
    ax.axhline(STARTING_CASH, color='gray', linestyle='--', alpha=0.5,
               label=f"start ${STARTING_CASH:,}")
    ax.set_title("Blind 2026 forward validation - MICROCAP-PUMP DAYS ONLY (#124's training regime)\n"
                 f"Train: 2022-2025 (W5) or 2021-2023 (W3 #124); Test: 2026 microcap-pump only", fontsize=12)
    ax.set_xlabel("Trading day index in 2026 (all days; equity flat on skipped non-microcap-pump)")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "equity_curves_microcap_only.png")
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"Wrote {p1}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [r['label'] for r in results]
    finals = [r['final'] for r in results]
    bars = ax.bar(labels, finals, color=[colors.get(l,'#999') for l in labels])
    ax.axhline(STARTING_CASH, color='gray', linestyle='--', label=f"start ${STARTING_CASH:,}")
    for r, b in zip(results, bars):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                f"${r['final']:,.0f}\n{r['multiplier']:.2f}x", ha='center',
                fontsize=11, fontweight='bold')
    ax.set_title("Final 2026 equity - MICROCAP-PUMP DAYS ONLY", fontsize=13)
    ax.set_ylabel("Final equity ($)")
    ax.grid(True, axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "final_bars_microcap_only.png")
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"Wrote {p2}")

    # Summary
    print(f"\n{'='*100}")
    print(f"SUMMARY - microcap-pump-only blind 2026")
    print(f"{'='*100}")
    print(f"  {'label':<24} {'final':>12} {'multi':>7} {'PnL':>12} {'trades':>7} {'tradeday':>10}")
    for r in results:
        pnl = r['final'] - STARTING_CASH
        print(f"  {r['label']:<24} ${r['final']:>10,.0f} {r['multiplier']:>5.2f}x "
              f"${pnl:>+10,.0f} {r['total_trades']:>7} {r['n_traded_days']:>10}")


if __name__ == "__main__":
    main()
