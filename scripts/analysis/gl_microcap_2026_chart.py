"""Backtest G+L only on 2026 microcap-pump days only + generate charts.

Uses #254's tuned params, enables only G and L, applies the same
microcap-pump shape filter we use during training.

Outputs:
  results/gl_microcap_2026/equity_curve.png
  results/gl_microcap_2026/daily_pnl.png
  results/gl_microcap_2026/per_strategy.png
  results/gl_microcap_2026/summary.json
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
SOURCE_CONFIG = "config/trial_254_w7_extracted.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
OUTDIR = "results/gl_microcap_2026"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD
    from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
    from strategies.regime_gate import classify_regime

    # Build G+L only config from #254
    with open(SOURCE_CONFIG) as f: data = json.load(f)
    src_params = data.get("params", data)
    gl_params = dict(src_params)
    for s in ALL_STRATS:
        gl_params[f"enable_{s}"] = (s in {"g", "l"})

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(gl_params)
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0

    # Load 2026 picks
    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates_all = sorted([d for d in all_dates if d.startswith("2026")])
    print(f"  2026 total days: {len(dates_all)}")

    # Apply microcap-pump shape filter
    wanted_shapes = {"microcap-thin", "thin-microcap"}
    dates_microcap = []
    for d in dates_all:
        p = picks_by_date.get(d, [])
        if not p:
            continue
        sig = day_signature(p)
        regime = classify_regime(p)
        shape = classify_shape(sig, regime)
        if shape in wanted_shapes:
            dates_microcap.append(d)
    print(f"  Microcap-pump days (training-shape filter): {len(dates_microcap)}")

    # Backtest G+L only on microcap-pump 2026 days
    cash = STARTING_CASH
    daily_records = []
    all_trades = []
    daily_eq = [cash]
    for d in dates_microcap:
        day_picks = picks_by_date.get(d, [])
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception as e:
            print(f"  {d}: simulator error {str(e)[:50]}")
            daily_eq.append(cash)
            daily_records.append({"date": d, "pnl": 0, "equity": cash, "n_trades": 0})
            continue

        day_pnl = 0
        n_trades_today = 0
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                day_pnl += st["pnl"]
                n_trades_today += 1
                all_trades.append({
                    "date": d,
                    "ticker": st["ticker"],
                    "strategy": st.get("strategy", "?"),
                    "pnl": float(st["pnl"]),
                    "cost": float(st["position_cost"]),
                    "pct": float(st["pnl"]) / float(st["position_cost"]) * 100 if st["position_cost"] > 0 else 0,
                    "exit_reason": str(st.get("exit_reason")),
                })
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
        daily_records.append({"date": d, "pnl": day_pnl, "equity": cash, "n_trades": n_trades_today})

    final_pnl = cash - STARTING_CASH
    print(f"\n  Result: final equity ${cash:,.0f}  PnL ${final_pnl:+,.0f}  ({cash/STARTING_CASH:.2f}x)")
    print(f"  Trades: {len(all_trades)}")
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    print(f"  Win rate: {wins}/{len(all_trades)} = {wins/max(len(all_trades),1)*100:.1f}%")

    # Per-strategy attribution
    by_strat = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in all_trades:
        by_strat[t["strategy"]]["n"] += 1
        by_strat[t["strategy"]]["pnl"] += t["pnl"]
    print(f"\n  Per-strategy attribution:")
    for s, v in sorted(by_strat.items()):
        avg = v["pnl"] / v["n"] if v["n"] else 0
        print(f"    {s:<2}  {v['n']:>3} trades  ${v['pnl']:>+11,.0f}  (${avg:+,.0f}/trade)")

    eq_arr = np.array(daily_eq)
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (eq_arr - peak) / peak * 100 if peak.max() > 0 else np.zeros_like(eq_arr)

    summary = {
        "config": "G+L only from #254 params",
        "shape_filter": "microcap-thin + thin-microcap",
        "total_2026_days": len(dates_all),
        "microcap_days_traded": len(dates_microcap),
        "final_equity": float(cash),
        "total_pnl": float(final_pnl),
        "multiple": float(cash / STARTING_CASH),
        "n_trades": len(all_trades),
        "win_rate_pct": float(wins / max(len(all_trades), 1) * 100),
        "max_dd_pct": float(dd_pct.min()),
        "max_dd_dollar": float((eq_arr - peak).min()),
        "per_strategy": {k: {"n": v["n"], "pnl": v["pnl"]} for k, v in by_strat.items()},
        "daily": daily_records,
    }
    with open(f"{OUTDIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ====== CHART 1: equity curve ======
    dt_dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in daily_records]
    # include starting point
    dt_x = [dt_dates[0]] + dt_dates
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(dt_x, eq_arr, color="#2ca02c", linewidth=2.5,
              label=f"G+L equity (${cash:,.0f}, {cash/STARTING_CASH:.2f}×)")
    ax1.axhline(STARTING_CASH, color="black", linestyle=":", alpha=0.5,
                  label=f"start ${STARTING_CASH:,}")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"G+L only on 2026 microcap-pump days ({len(dates_microcap)} days, "
                    f"{len(all_trades)} trades)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    ax2.fill_between(dt_x, dd_pct, 0, color="#d62728", alpha=0.3)
    ax2.plot(dt_x, dd_pct, color="#d62728", linewidth=1.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("2026 trading day")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.text(0.02, 0.05, f"Max DD: {dd_pct.min():.1f}%",
              transform=ax2.transAxes, fontsize=10,
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    p1 = f"{OUTDIR}/equity_curve.png"
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"\n  Wrote {p1}")

    # ====== CHART 2: daily PnL bars ======
    pnls = np.array([r["pnl"] for r in daily_records])
    colors = ["#2ca02c" if p > 0 else "#d62728" if p < 0 else "#888888" for p in pnls]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(dt_dates, pnls, color=colors, width=1.0, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("G+L daily PnL on 2026 microcap-pump days")
    ax.set_xlabel("Date"); ax.set_ylabel("Daily PnL ($)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(True, alpha=0.3, axis="y")
    wins_d = int((pnls > 0).sum()); losses_d = int((pnls < 0).sum())
    flat_d = int((pnls == 0).sum())
    ax.text(0.02, 0.95,
              f"wins {wins_d}  losses {losses_d}  flat {flat_d}\n"
              f"best ${pnls.max():,.0f}  worst ${pnls.min():,.0f}",
              transform=ax.transAxes, fontsize=10, verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    p2 = f"{OUTDIR}/daily_pnl.png"
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"  Wrote {p2}")

    # ====== CHART 3: per-strategy bars ======
    fig, ax = plt.subplots(figsize=(8, 5))
    strats = sorted(by_strat.keys())
    pnls_s = [by_strat[s]["pnl"] for s in strats]
    ns = [by_strat[s]["n"] for s in strats]
    bar_colors = ["#1f77b4" if p > 0 else "#d62728" for p in pnls_s]
    bars = ax.bar(strats, pnls_s, color=bar_colors)
    for bar, n, pnl in zip(bars, ns, pnls_s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                  f"{n} trades\n${pnl:,.0f}",
                  ha="center", va="bottom" if pnl > 0 else "top", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("G+L on 2026 microcap-pump — per-strategy attribution")
    ax.set_xlabel("Strategy"); ax.set_ylabel("Total PnL ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p3 = f"{OUTDIR}/per_strategy.png"
    fig.savefig(p3, dpi=140); plt.close(fig)
    print(f"  Wrote {p3}")

    # Summary table to stdout
    print(f"\n  === Summary ===")
    print(f"  Microcap-pump days traded: {len(dates_microcap)} / {len(dates_all)}")
    print(f"  Final equity: ${cash:,.0f}  ({cash/STARTING_CASH:.2f}×)")
    print(f"  Total PnL:    ${final_pnl:+,.0f}")
    print(f"  Max DD:       {dd_pct.min():.1f}%  (${(eq_arr-peak).min():,.0f})")
    print(f"  Trades:       {len(all_trades)}  ({wins} wins / {len(all_trades)-wins} losses)")


if __name__ == "__main__":
    main()
