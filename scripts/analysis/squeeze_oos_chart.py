"""Visualize the squeeze-day OOS comparison on a given year.

Re-runs the 4 configs (#6, #326, #635, #541) on the regime-gated squeeze
days of {year} and saves:
  - results/squeeze_oos_{year}_equity.png : equity curves across squeeze days
  - results/squeeze_oos_{year}_perday.png : per-day PnL bars

Usage:
  python scripts/analysis/squeeze_oos_chart.py --year 2019
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime

TRIAL_6   = "config/trial_6_extracted.json"
TRIAL_326 = "config/trial_326_squeeze_extracted.json"
TRIAL_541 = "config/trial_541_squeeze_extracted.json"
TRIAL_587 = "config/trial_587_squeeze_extracted.json"
TRIAL_635 = "config/trial_635_extracted.json"
BASELINE  = "config/trial_432_params.json"
STARTING_CASH = 25_000

CONFIGS = [
    (TRIAL_6,   "#6   generalist",           "#1f77b4"),
    (TRIAL_326, "#326 old squeeze (2.09)",   "#ff7f0e"),
    (TRIAL_541, "#541 prev squeeze (2.73)",  "#2ca02c"),
    (TRIAL_587, "#587 NEW squeeze (2.91)",   "#d62728"),
]


def _merged(path):
    with open(path) as f:
        p = json.load(f)
    if isinstance(p, dict) and "params" in p:
        p = p["params"]
    with open(BASELINE) as f:
        base = json.load(f)
    m = dict(base); m.update(p)
    return m


def load_and_apply(path):
    set_strategy_params(_merged(path))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.VOL_CAP_PCT = 5.0


def run_on(daily_picks, dates, config_path):
    load_and_apply(config_path)
    cash = STARTING_CASH
    equity_curve = [cash]
    for d in dates:
        picks = daily_picks.get(d, [])
        if not picks:
            equity_curve.append(cash)
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            _, ending_cash, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account=is_cash
            )
        except Exception:
            equity_curve.append(cash)
            continue
        cash = ending_cash
        if is_cash:
            cash += unsettled
        equity_curve.append(cash)
    equity_curve = np.array(equity_curve)
    daily_pnl = np.diff(equity_curve)
    return equity_curve, daily_pnl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    data_dir = f"stored_data_{args.year}"
    if not os.path.exists(data_dir):
        raise SystemExit(f"{data_dir} not found")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading {data_dir}...")
    dates, daily_picks = load_all_picks([data_dir])
    print(f"  {len(dates)} trading days")

    squeeze_days = [d for d in dates if classify_regime(daily_picks.get(d, [])) == "squeeze"]
    print(f"  squeeze days: {len(squeeze_days)}")
    if not squeeze_days:
        raise SystemExit("No squeeze days — nothing to plot.")

    runs = []
    for cfg, label, color in CONFIGS:
        if not os.path.exists(cfg):
            print(f"  [skip] {cfg}")
            continue
        print(f"  running {label}...")
        eq, pnl = run_on(daily_picks, squeeze_days, cfg)
        runs.append((label, color, eq, pnl))
        print(f"    final ${eq[-1]:,.0f}  pnl_sum ${pnl.sum():+,.0f}")

    # --- Chart 1: equity curves ---
    x = np.arange(len(squeeze_days) + 1)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(STARTING_CASH, color="#aaaaaa", linestyle="--", linewidth=0.8,
               label=f"start ${STARTING_CASH:,}")
    for label, color, eq, _ in runs:
        ax.plot(x, eq, marker="o", markersize=4, linewidth=1.8,
                color=color, label=label)
    ax.set_title(f"{args.year} squeeze-day OOS equity curves "
                 f"({len(squeeze_days)} regime-gated days)")
    ax.set_xlabel("Squeeze day index (chronological)")
    ax.set_ylabel("Account equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    # mark date labels every Nth tick
    step = max(1, len(squeeze_days) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"day{i}" if i == 0 else squeeze_days[i-1] for i in x[::step]],
                       rotation=30, fontsize=8, ha="right")
    fig.tight_layout()
    eq_path = os.path.join(args.outdir, f"squeeze_oos_{args.year}_equity.png")
    fig.savefig(eq_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {eq_path}")

    # --- Chart 2: per-day PnL bars (grouped) ---
    n_runs = len(runs)
    width = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(14, 6))
    idx = np.arange(len(squeeze_days))
    for i, (label, color, _, pnl) in enumerate(runs):
        ax.bar(idx + i*width - 0.4 + width/2, pnl, width=width,
               color=color, label=label, alpha=0.85)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_title(f"{args.year} squeeze-day OOS per-day PnL "
                 f"({len(squeeze_days)} days)")
    ax.set_xlabel("Squeeze day")
    ax.set_ylabel("PnL ($)")
    ax.set_xticks(idx)
    ax.set_xticklabels(squeeze_days, rotation=45, fontsize=8, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout()
    pd_path = os.path.join(args.outdir, f"squeeze_oos_{args.year}_perday.png")
    fig.savefig(pd_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {pd_path}")

    # --- Summary ---
    print(f"\nSummary ({args.year}, {len(squeeze_days)} squeeze days):")
    print(f"  {'config':<32} {'final':>12} {'pnl':>12} {'pnl/day':>10}  win-days")
    for label, _, eq, pnl in runs:
        wins = int((pnl > 0).sum())
        print(f"  {label:<32} ${eq[-1]:>11,.0f} ${pnl.sum():>+11,.0f} ${pnl.sum()/len(squeeze_days):>+9,.0f}  {wins}/{len(squeeze_days)}")


if __name__ == "__main__":
    main()
