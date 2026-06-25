"""Reproduce the March 8, 2026 backtest (H+G+A+F+D+V+M+R+P) on Jan 2024 - Feb 2026
with TODAY'S realistic execution model (Almgren-Chriss + participation caps + vol-adj).

Comparison: original chart showed $124M from $25K. With current execution model,
the same config should land much lower — quantifies the "cost" of realistic fills.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
# Use #124 as the params source since it's the closest tuned config to a "default
# enabled" sweep. We override enables to match March's H+G+A+F+D+V+M+R+P.
SOURCE_CONFIG = "config/trial_124_microcap_pump_extracted.json"
OUTDIR = "results/march_repro"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

# March 8 backtest period
TRAINING_DIRS = [
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data",  # 2026 Jan-Feb
]

# Enabled strategies from the March 8 summary (from filename + chart title)
ENABLED = {"h", "g", "a", "f", "d", "v", "m", "r", "p"}  # 9 strategies


def run_scenario(label, slippage_off=False, caps_off=False):
    """Run the backtest under different execution model configurations."""
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(SOURCE_CONFIG) as f: data = json.load(f)
    src_params = data.get("params", data)

    # Force the March-era strategy enables
    params = dict(src_params)
    for s in ALL_STRATS:
        params[f"enable_{s}"] = (s in ENABLED)

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(params)
    set_strategy_params(merged)

    # Configure execution model based on scenario
    if slippage_off and caps_off:
        # March-era execution: no slippage modeling, no participation caps
        tgc.USE_DYNAMIC_SLIPPAGE = False
        tgc.USE_MULTIWINDOW_SLIPPAGE = False
        tgc.USE_VOLATILITY_ADJUSTMENT = False
        tgc.MAX_2MIN_PARTICIPATION = 0  # 0 = disabled
        tgc.MAX_REGIME_PARTICIPATION = 0
        tgc.VOL_CAP_PCT = 5.0  # the only check that existed in March
    elif slippage_off:
        # Just slippage off, caps on (isolate slippage effect)
        tgc.USE_DYNAMIC_SLIPPAGE = False
        tgc.USE_MULTIWINDOW_SLIPPAGE = False
        tgc.USE_VOLATILITY_ADJUSTMENT = False
        tgc.MAX_2MIN_PARTICIPATION = 0.15
        tgc.MAX_REGIME_PARTICIPATION = 0.08
        tgc.VOL_CAP_PCT = 5.0
    elif caps_off:
        # Slippage on, caps off (isolate caps effect)
        tgc.USE_DYNAMIC_SLIPPAGE = True
        tgc.USE_MULTIWINDOW_SLIPPAGE = True
        tgc.USE_VOLATILITY_ADJUSTMENT = True
        tgc.SLIP_IMPACT_K = 3.0
        tgc.MAX_2MIN_PARTICIPATION = 0
        tgc.MAX_REGIME_PARTICIPATION = 0
        tgc.VOL_CAP_PCT = 5.0
    else:
        # Current execution model — both on
        tgc.USE_DYNAMIC_SLIPPAGE = True
        tgc.USE_MULTIWINDOW_SLIPPAGE = True
        tgc.USE_VOLATILITY_ADJUSTMENT = True
        tgc.SLIP_IMPACT_K = 3.0
        tgc.MAX_2MIN_PARTICIPATION = 0.15
        tgc.MAX_REGIME_PARTICIPATION = 0.08
        tgc.VOL_CAP_PCT = 5.0

    tgc.MARGIN_MULTIPLIER = 1.0

    dirs = [d for d in TRAINING_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    # Match March's range: 2024-01-02 to 2026-02-27 (325 days)
    dates = sorted([d for d in all_dates if "2024-01-02" <= d <= "2026-02-27"])

    cash = STARTING_CASH
    daily_eq = [cash]
    all_trades = []
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception as e:
            daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                all_trades.append({
                    "date": d,
                    "ticker": st["ticker"],
                    "strategy": st.get("strategy", "?"),
                    "pnl": float(st["pnl"]),
                    "cost": float(st["position_cost"]),
                })
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)

    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = float((eq - peak).min() / peak[(eq - peak).argmin()] * 100) if peak.max() > 0 else 0
    final_pnl = float(eq[-1] - STARTING_CASH)
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    multi = float(eq[-1] / STARTING_CASH)

    print(f"\n  {label}")
    print(f"    Days simulated: {len(dates)}")
    print(f"    Final equity:   ${eq[-1]:,.0f}  ({multi:.1f}x)")
    print(f"    Total PnL:      ${final_pnl:+,.0f}")
    print(f"    Trades:         {len(all_trades)}  ({wins} wins, {wins/max(len(all_trades),1)*100:.1f}% WR)")
    print(f"    Max DD:         {dd_pct:.1f}%")

    return {
        "label": label,
        "final": float(eq[-1]),
        "pnl": final_pnl,
        "multiple": multi,
        "n_trades": len(all_trades),
        "win_rate": wins / max(len(all_trades), 1) * 100,
        "max_dd_pct": dd_pct,
        "daily_eq": eq.tolist(),
        "dates": dates,
        "trades": all_trades,
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Reproducing March 8 backtest (H+G+A+F+D+V+M+R+P on 2024-01-02 to 2026-02-27)")
    print("Under 4 different execution models to isolate each component's cost\n")

    results = []
    # Scenario A: March-era execution (no slippage, no caps) — should reproduce ~$124M
    results.append(run_scenario("A. March-era (no slippage model, no caps)",
                                 slippage_off=True, caps_off=True))
    # Scenario B: Slippage on, caps off (cost of slippage only)
    results.append(run_scenario("B. Slippage ON only (Almgren-Chriss, no participation caps)",
                                 slippage_off=False, caps_off=True))
    # Scenario C: Slippage off, caps on (cost of caps only)
    results.append(run_scenario("C. Participation caps ON only (no slippage model)",
                                 slippage_off=True, caps_off=False))
    # Scenario D: Current execution (both on) — the realistic estimate
    results.append(run_scenario("D. CURRENT realistic execution (slippage + caps)",
                                 slippage_off=False, caps_off=False))

    print(f"\n{'='*92}")
    print(f"  COMPARISON")
    print(f"{'='*92}")
    print(f"  {'Scenario':<55} {'Final $':>13} {'Multi':>9} {'Trades':>7}")
    march_baseline = 124_498_028
    for r in results:
        ratio_vs_march = r['final'] / march_baseline * 100
        print(f"  {r['label']:<55} ${r['final']:>11,.0f} {r['multiple']:>6.1f}x  {r['n_trades']:>6}")
    print(f"\n  March-8 chart baseline (from PNG):  $124,498,028  (4,980×)")
    if results:
        delta_a = results[0]['final'] / march_baseline
        delta_d = results[3]['final'] / march_baseline if len(results) > 3 else 0
        print(f"  Scenario A vs March:               {delta_a*100:.1f}%  (should be near 100% if repro is correct)")
        print(f"  Scenario D vs March:               {delta_d*100:.1f}%  (realistic execution cost)")
        if delta_d < 1:
            cost = results[0]['final'] - results[3]['final']
            print(f"  Cost of execution model:           ${cost:,.0f}  ({(1-delta_d)*100:.1f}% of A's notional)")

    # ====== Chart: equity curves overlay ======
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ["#888888", "#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, r in enumerate(results):
        dt = [datetime.strptime(d, "%Y-%m-%d") for d in r["dates"]]
        dt = [dt[0]] + dt  # include start
        ax.plot(dt, r["daily_eq"], color=colors[i], linewidth=2,
                  label=f"{r['label']}: ${r['final']:,.0f}")
    ax.axhline(STARTING_CASH, color="black", linestyle=":", alpha=0.5,
                  label=f"start ${STARTING_CASH:,}")
    ax.set_yscale("log")
    ax.set_title("H+G+A+F+D+V+M+R+P on 2024-01-02 to 2026-02-27 — execution model comparison\n"
                  "(log scale; reproduces March chart's $124M result vs current realistic execution)",
                  fontsize=11)
    ax.set_xlabel("Date"); ax.set_ylabel("Equity ($, log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout()
    chart_path = f"{OUTDIR}/march_repro_comparison.png"
    fig.savefig(chart_path, dpi=140); plt.close(fig)
    print(f"\n  Wrote {chart_path}")

    # Save full results
    out_json = f"{OUTDIR}/march_repro_results.json"
    with open(out_json, "w") as f:
        # Strip trades to keep file reasonable
        save = []
        for r in results:
            r2 = {k: v for k, v in r.items() if k != "trades" and k != "daily_eq" and k != "dates"}
            r2["n_trades"] = r["n_trades"]
            save.append(r2)
        json.dump(save, f, indent=2)
    print(f"  Wrote {out_json}")


if __name__ == "__main__":
    main()
