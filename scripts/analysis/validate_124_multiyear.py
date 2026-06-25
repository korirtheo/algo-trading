"""Forward-validate microcap-pump #124 on 2025 and 2026.

Same flags as training (multiwindow slippage + no X). Runs all-days (solo
specialist deploy) on each year and generates equity curve charts.

The 2024 result is already known ($25K -> $209K, 8.38x). This script tells
us whether the edge holds forward on 2025 and (critically) 2026 — the
current live deployment regime.
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

CONFIG_PATH = os.environ.get("VALIDATE_CONFIG_PATH", "config/trial_124_microcap_pump_extracted.json")
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000

YEAR_DIRS = {
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026"],
}

OUT_DIR = os.environ.get("VALIDATE_OUT_DIR", "results/wf_pf_microcap_pump_noX/validation_multiyear")


def _merged(params):
    with open(BASELINE) as f:
        b = json.load(f)
    m = dict(b)
    m.update(params)
    return m


def run_year(year, dirs):
    dirs_present = [d for d in dirs if os.path.exists(d)]
    if not dirs_present:
        print(f"[skip] {year}: no data dirs"); return None
    all_dates, picks = load_all_picks(dirs_present)
    test_dates = [d for d in all_dates if d.startswith(year)]
    if not test_dates:
        print(f"[skip] {year}: no test dates"); return None

    cash = STARTING_CASH
    eq = [cash]
    daily = []
    trades = []
    for d in test_dates:
        day_picks = picks.get(d, [])
        if not day_picks:
            eq.append(cash); daily.append({"date": d, "pnl": 0, "equity": cash}); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
        except Exception:
            eq.append(cash); daily.append({"date": d, "pnl": 0, "equity": cash}); continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                trades.append({"date": d, "ticker": st["ticker"],
                               "strategy": st.get("strategy"), "pnl": st["pnl"],
                               "pct": (st["pnl"] / st["position_cost"] * 100) if st["position_cost"] > 0 else 0,
                               "reason": st.get("exit_reason")})
        pnl = end_c - cash
        cash = end_c
        if is_cash: cash += unset
        eq.append(cash); daily.append({"date": d, "pnl": pnl, "equity": cash})

    eq = np.array(eq)
    final = eq[-1]; total_pnl = final - STARTING_CASH
    prev = eq[:-1]; curr = eq[1:]
    mask = prev > 0
    ret = np.where(mask, (curr - prev) / prev * 100, 0)
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0.0
    wins = int((ret > 0).sum())
    losses = int((ret < 0).sum())

    return {
        "year": year,
        "n_days": len(test_dates),
        "final": final, "pnl": total_pnl,
        "multiplier": final / STARTING_CASH,
        "sharpe_pct": sharpe,
        "win_days": wins, "loss_days": losses,
        "trades": trades,
        "eq": eq,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    print(f"Config: {CONFIG_PATH}")
    print(f"  Trial #{cfg.get('trial_number')}  score=${cfg.get('score', 0):,.0f}")

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
        print(f"\n--- Validating on {year} ---")
        r = run_year(year, dirs)
        if r is None: continue
        results.append(r)
        print(f"  {r['n_days']} days  Final ${r['final']:,.0f} ({r['multiplier']:.2f}x)  "
              f"PnL ${r['pnl']:+,.0f}  Sharpe% {r['sharpe_pct']:.2f}  "
              f"WinDays {r['win_days']}/{r['n_days']}  Trades {len(r['trades'])}")

    # --- Charts ---
    # 1. Combined equity curves (one normalized, one absolute)
    if results:
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = {"2024": "#d62728", "2025": "#2ca02c", "2026": "#1f77b4"}
        for r in results:
            n = len(r["eq"])
            ax.plot(range(n), r["eq"], color=colors.get(r["year"], "#888"),
                    linewidth=2.0,
                    label=f"{r['year']}: final ${r['eq'][-1]:,.0f} ({r['multiplier']:.2f}x, Sharpe% {r['sharpe_pct']:.2f})")
        ax.axhline(STARTING_CASH, color="#aaa", linestyle="--", linewidth=0.8,
                   label=f"start ${STARTING_CASH:,}")
        ax.set_yscale("log")
        ax.set_title("#124 microcap-pump specialist — multi-year forward validation (log)")
        ax.set_xlabel("Trading day index (within year)")
        ax.set_ylabel("Equity ($, log scale)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        fig.tight_layout()
        p1 = os.path.join(OUT_DIR, "equity_multiyear_log.png")
        fig.savefig(p1, dpi=140); plt.close(fig)
        print(f"\nWrote {p1}")

        # Linear
        fig, ax = plt.subplots(figsize=(13, 6))
        for r in results:
            n = len(r["eq"])
            ax.plot(range(n), r["eq"], color=colors.get(r["year"], "#888"),
                    linewidth=2.0,
                    label=f"{r['year']}: final ${r['eq'][-1]:,.0f} ({r['multiplier']:.2f}x)")
        ax.axhline(STARTING_CASH, color="#aaa", linestyle="--", linewidth=0.8,
                   label=f"start ${STARTING_CASH:,}")
        ax.set_title("#124 microcap-pump specialist — multi-year forward validation (linear)")
        ax.set_xlabel("Trading day index (within year)")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        fig.tight_layout()
        p2 = os.path.join(OUT_DIR, "equity_multiyear_linear.png")
        fig.savefig(p2, dpi=140); plt.close(fig)
        print(f"Wrote {p2}")

        # Bar chart of final equity by year
        fig, ax = plt.subplots(figsize=(10, 5))
        years = [r["year"] for r in results]
        finals = [r["final"] for r in results]
        bars = ax.bar(years, finals, color=[colors.get(y, "#888") for y in years])
        ax.axhline(STARTING_CASH, color="#444", linestyle="--", linewidth=0.8,
                   label=f"start ${STARTING_CASH:,}")
        for r, b in zip(results, bars):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                    f"${r['final']:,.0f}\n{r['multiplier']:.2f}x", ha="center",
                    fontsize=10, fontweight="bold")
        ax.set_title("#124 — Final equity by validation year")
        ax.set_ylabel("Final equity ($)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        fig.tight_layout()
        p3 = os.path.join(OUT_DIR, "final_equity_bars.png")
        fig.savefig(p3, dpi=140); plt.close(fig)
        print(f"Wrote {p3}")

    # Summary
    print(f"\n{'='*72}")
    print("MULTI-YEAR VALIDATION SUMMARY — #124 microcap-pump specialist")
    print(f"{'='*72}")
    print(f"  {'year':<6} {'days':<6} {'final':>11} {'multi':>7} {'pnl':>11} {'sharpe%':>8} {'wins':>10} {'trades':>7}")
    for r in results:
        print(f"  {r['year']:<6} {r['n_days']:<6} ${r['final']:>10,.0f} {r['multiplier']:>6.2f}x ${r['pnl']:>+10,.0f} "
              f"{r['sharpe_pct']:>8.2f} {r['win_days']:>3}/{r['n_days']:<5}  {len(r['trades']):>7}")


if __name__ == "__main__":
    main()
