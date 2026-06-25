"""Validate #124 on MICROCAP-PUMP DAYS ONLY across 2024, 2025, 2026.

Same engine as validate_124_multiyear.py but with a runtime shape gate
applied: only days classified as microcap-thin or thin-microcap are
traded. Non-microcap-pump days return zero PnL (treated as "bot skipped").

Compare directly to the all-days baseline:
    2024:  $209,548   (8.38x)   all-days
    2025:  $833,822  (33.35x)   all-days
    2026:  $ 47,534   (1.90x)   all-days

The question this answers: would a runtime shape gate IMPROVE risk-adjusted
returns, or would the lost compounding days hurt more than the gated wins?
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

# Reuse the same shape classifier we trained on
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
from strategies.regime_gate import classify_regime, compute_features

CONFIG_PATH = "config/trial_124_microcap_pump_extracted.json"
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000

YEAR_DIRS = {
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026"],
}

ALL_DAYS_BASELINE = {
    "2024": {"final": 209_548, "multiplier": 8.38},
    "2025": {"final": 833_822, "multiplier": 33.35},
    "2026": {"final":  47_534, "multiplier":  1.90},
}

KEEP_SHAPES = {"microcap-thin", "thin-microcap"}

OUT_DIR = "results/wf_pf_microcap_pump_noX/validation_microcap_pump_only"


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def _classify_day(picks):
    """Return shape label for this day's picks (e.g. 'microcap-thin')."""
    try:
        feats = compute_features(picks)
        regime = classify_regime(feats)
    except Exception:
        regime = "NORMAL"
    sig = day_signature(picks)
    if sig is None:
        return "empty"
    return classify_shape(sig, regime)


def run_year(year, dirs):
    dirs_present = [d for d in dirs if os.path.exists(d)]
    if not dirs_present:
        print(f"[skip] {year}: no data dirs"); return None
    all_dates, picks = load_all_picks(dirs_present)
    test_dates = [d for d in all_dates if d.startswith(year)]
    if not test_dates:
        return None

    cash = STARTING_CASH
    eq = [cash]
    trades = []
    n_traded = 0
    n_skipped = 0
    shape_counts = {}

    for d in test_dates:
        day_picks = picks.get(d, [])
        if not day_picks:
            eq.append(cash); continue

        shape = _classify_day(day_picks)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

        if shape not in KEEP_SHAPES:
            # Skip — bot would sit out
            eq.append(cash)
            n_skipped += 1
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
                               "strategy": st.get("strategy"), "pnl": st["pnl"],
                               "pct": (st["pnl"] / st["position_cost"] * 100) if st["position_cost"] > 0 else 0})
        cash = end_c
        if is_cash: cash += unset
        eq.append(cash)

    eq = np.array(eq)
    final = eq[-1]
    prev = eq[:-1]; curr = eq[1:]
    mask = prev > 0
    ret = np.where(mask, (curr - prev) / prev * 100, 0)
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0.0
    wins = int((ret > 0).sum())
    losses = int((ret < 0).sum())

    return {
        "year": year,
        "n_days_total": len(test_dates),
        "n_traded": n_traded,
        "n_skipped": n_skipped,
        "shape_counts": shape_counts,
        "final": final,
        "multiplier": final / STARTING_CASH,
        "sharpe_pct": sharpe,
        "win_days": wins,
        "loss_days": losses,
        "trades": trades,
        "eq": eq,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    print(f"Config: {CONFIG_PATH}")
    print(f"Filter: only {KEEP_SHAPES} days (all others SKIPPED)\n")

    set_strategy_params(_merged(cfg["params"]))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    results = []
    for year, dirs in YEAR_DIRS.items():
        print(f"--- Validating {year} (microcap-pump days only) ---")
        r = run_year(year, dirs)
        if r is None:
            continue
        results.append(r)
        print(f"  Total days: {r['n_days_total']}  Traded: {r['n_traded']}  Skipped: {r['n_skipped']}")
        print(f"  Shape distribution: {r['shape_counts']}")
        print(f"  Final: ${r['final']:,.0f} ({r['multiplier']:.2f}x)  "
              f"Sharpe% {r['sharpe_pct']:.2f}  "
              f"WinDays {r['win_days']}/{r['n_traded']}  Trades {len(r['trades'])}")
        print()

    print(f"\n{'='*88}")
    print(f"COMPARISON: gated (this run) vs ungated (all-days baseline)")
    print(f"{'='*88}")
    print(f"  {'year':<6} {'days_traded':<13} {'gated_final':<14} {'gated_mult':<11} "
          f"{'ungated_final':<14} {'ungated_mult':<11} {'verdict':<10}")
    for r in results:
        ung = ALL_DAYS_BASELINE.get(r['year'], {})
        verdict = ""
        if ung:
            if r['final'] > ung['final']:
                verdict = "GATED WINS"
            elif r['final'] < ung['final'] * 0.7:
                verdict = "GATED LOSES BAD"
            elif r['final'] < ung['final']:
                verdict = "gated worse"
            else:
                verdict = "roughly tied"
        print(f"  {r['year']:<6} {r['n_traded']:<13} ${r['final']:<13,.0f} "
              f"{r['multiplier']:<10.2f}x ${ung.get('final', 0):<13,.0f} "
              f"{ung.get('multiplier', 0):<10.2f}x {verdict:<10}")

    # Combined chart
    if results:
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = {"2024": "#d62728", "2025": "#2ca02c", "2026": "#1f77b4"}
        for r in results:
            n = len(r["eq"])
            ax.plot(range(n), r["eq"], color=colors.get(r["year"], "#888"),
                    linewidth=2.0,
                    label=f"{r['year']}: ${r['final']:,.0f} ({r['multiplier']:.2f}x, "
                          f"traded {r['n_traded']}/{r['n_days_total']} days)")
        ax.axhline(STARTING_CASH, color="#aaa", linestyle="--", linewidth=0.8,
                   label=f"start ${STARTING_CASH:,}")
        ax.set_yscale("log")
        ax.set_title("#124 — MICROCAP-PUMP DAYS ONLY (runtime shape gate)")
        ax.set_xlabel("Trading day index (within year)")
        ax.set_ylabel("Equity ($, log scale)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        fig.tight_layout()
        p = os.path.join(OUT_DIR, "equity_microcap_pump_only.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
