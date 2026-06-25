"""Plot W7 forward-test results on 2026 vs the deployed #124 baseline.

Reads results/walk_forward_v7_news/W5_train_2022..2025_test_2026_PARTIAL_trial_*.json
and produces:
  1. Equity curve over 98 days (each W7 candidate + #124 baseline)
  2. Daily PnL bars (win/loss colored)
  3. Cumulative PnL underwater chart
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

OUTDIR = "results/walk_forward_v7_news"
STARTING_CASH = 25_000
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
BASELINE_CONF = "config/trial_124_microcap_pump_extracted.json"
BASELINE_PARAMS = "config/trial_432_params.json"


def forward_2026(config_path, label):
    """Replay the saved forward result if it exists, else simulate."""
    with open(config_path) as f:
        cfg = json.load(f)
    params = cfg.get("params", cfg)
    with open(BASELINE_PARAMS) as f:
        baseline = json.load(f)
    merged = dict(baseline); merged.update(params)
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])
    cash = STARTING_CASH
    daily = []
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if day_picks and tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as nfp
            day_picks = nfp(day_picks, d, tgc.NEWS_MIN_ARTICLES, tgc.NEWS_REQUIRE_CATALYST)
        if not day_picks:
            daily.append({"date": d, "pnl": 0, "equity": cash}); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            _, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
        except Exception:
            daily.append({"date": d, "pnl": 0, "equity": cash}); continue
        pnl = end_c - cash
        cash = end_c + (unset if is_cash else 0)
        daily.append({"date": d, "pnl": pnl, "equity": cash})
    return daily, cash


def daily_to_arrays(daily):
    dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in daily]
    pnls = np.array([r["pnl"] for r in daily])
    equity = np.array([r["equity"] for r in daily])
    return dates, pnls, equity


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Run baseline (#124) on 2026
    print("=== #124 W3 baseline ===")
    base_daily, base_final = forward_2026(BASELINE_CONF, "#124")
    base_dates, base_pnls, base_equity = daily_to_arrays(base_daily)
    print(f"  #124 final: ${base_final:,.0f}  pnl ${base_final-STARTING_CASH:+,.0f}")

    # Each W7 candidate
    print("\n=== W7 candidates from saved JSON ===")
    candidates = {}
    for jf in sorted(glob.glob(f"{OUTDIR}/W5_train_2022..2025_test_2026_PARTIAL_trial_*.json")):
        with open(jf) as f: r = json.load(f)
        num = r["trial_number"]
        daily = r["daily"]
        dates, pnls, equity = daily_to_arrays(daily)
        # prepend starting cash for equity-curve continuity
        candidates[f"#{num}"] = (dates, pnls, equity, r["forward_final_equity"], r["train_score"])
        print(f"  #{num}  train ${r['train_score']:,.0f}  fwd ${r['forward_final_equity']:,.0f}  pnl ${r['forward_total_pnl']:+,.0f}")

    # ====== CHART 1: equity curves over 2026 days ======
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(base_dates, base_equity, color="#7f7f7f", linewidth=2.2,
            label=f"#124 W3 deployed: ${base_final:,.0f}  ({base_final/STARTING_CASH:.2f}x)")
    colors = {"#217": "#1f77b4", "#254": "#2ca02c", "#312": "#d62728"}
    for label, (dates, pnls, equity, final, train) in candidates.items():
        c = colors.get(label, None)
        ax.plot(dates, equity, color=c, linewidth=2.5, alpha=0.9,
                label=f"{label} W7: ${final:,.0f}  ({final/STARTING_CASH:.2f}x)  [train ${train/1e6:.1f}M]")
    ax.axhline(STARTING_CASH, color="black", linestyle=":", alpha=0.4,
               label=f"start ${STARTING_CASH:,}")
    ax.set_title("W7 forward test on 2026 (blind) — train 2022-2025, microcap-pump shape\n"
                 "vs. #124 deployed baseline", fontsize=12)
    ax.set_xlabel("2026 trading day"); ax.set_ylabel("Equity ($)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    p1 = f"{OUTDIR}/w7_forward_equity_2026.png"
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"\nWrote {p1}")

    # ====== CHART 2: daily PnL bars (top trial only — #254) ======
    if "#254" in candidates:
        dates, pnls, _, _, _ = candidates["#254"]
        colors_pnl = ["#2ca02c" if p > 0 else ("#d62728" if p < 0 else "#cccccc") for p in pnls]
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.bar(dates, pnls, color=colors_pnl, edgecolor="none", width=1.0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("#254 W7 daily PnL on 2026 (blind forward test)", fontsize=12)
        ax.set_xlabel("2026 trading day"); ax.set_ylabel("PnL ($)")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
        ax.grid(True, alpha=0.3, axis="y")
        wins = int((pnls > 0).sum()); losses = int((pnls < 0).sum()); flat = int((pnls == 0).sum())
        ax.text(0.02, 0.97, f"wins {wins}  losses {losses}  flat {flat}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        fig.tight_layout()
        p2 = f"{OUTDIR}/w7_254_daily_pnl.png"
        fig.savefig(p2, dpi=140); plt.close(fig)
        print(f"Wrote {p2}")

    # ====== CHART 3: rolling 10-day equity for each candidate (drawdown view) ======
    fig, ax = plt.subplots(figsize=(15, 7))
    # Compute drawdown = (equity - running_max) / running_max for each
    def drawdown_series(equity):
        peak = np.maximum.accumulate(equity)
        return (equity - peak) / peak * 100
    ax.plot(base_dates, drawdown_series(base_equity), color="#7f7f7f", linewidth=2,
            label=f"#124 deployed  (peak DD {drawdown_series(base_equity).min():.1f}%)")
    for label, (dates, _, equity, _, _) in candidates.items():
        c = colors.get(label, None)
        dd = drawdown_series(equity)
        ax.plot(dates, dd, color=c, linewidth=2,
                label=f"{label} W7  (peak DD {dd.min():.1f}%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Drawdown from running peak — W7 candidates vs #124 on 2026", fontsize=12)
    ax.set_xlabel("2026 trading day"); ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    p3 = f"{OUTDIR}/w7_drawdown_2026.png"
    fig.savefig(p3, dpi=140); plt.close(fig)
    print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
