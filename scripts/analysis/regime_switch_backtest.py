"""Run a year-long backtest with daily regime-routing.

For each trading day:
  - classify_regime(picks) → dead | normal | squeeze
  - corp-action filter (leader_gap > 500% AND leader_prev < $1) → skip
  - dead → skip
  - squeeze → switch params to #818 (sq2.81 NEW)
  - normal → switch params to #571 (norm2.08 NEW)

Compare against single-config baselines (no switching):
  - #6  generalist
  - #818 only
  - #571 only

Cash compounds across all days (each day starts where prior ended).
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

YEAR_DIRS = {
    "2019": ["stored_data_2019"],
    "2020": ["stored_data_2020"],
    "2026": ["stored_data", "stored_data_mar_may_2026"],
}

CFG_6   = "config/trial_6_extracted.json"
CFG_818 = "config/trial_818_squeeze_extracted.json"
CFG_571 = "config/trial_571_normal_extracted.json"
CFG_24  = "config/trial_24_combined_extracted.json"
CFG_635 = "config/trial_635_extracted.json"
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000


def _merged(p):
    with open(p) as f: x = json.load(f)
    if isinstance(x, dict) and "params" in x: x = x["params"]
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(x); return m


def _is_corp_action(picks):
    if not picks: return False
    leader = max(picks, key=lambda p: p.get("gap_pct") or 0)
    return (leader.get("gap_pct", 0) or 0) > 500 and (leader.get("prev_close", 0) or 0) < 1.0


def run_year(label, dates, dp, cfg_for_day):
    """Run one full year. cfg_for_day(date, regime) -> cfg_path or None (skip)."""
    cash = STARTING_CASH
    daily = []
    cur_cfg_path = None
    for d in dates:
        picks = dp.get(d, [])
        if _is_corp_action(picks):
            daily.append({"date": d, "regime": "corp-action", "cfg": None, "pnl": 0, "equity": cash, "n_picks": len(picks)})
            continue
        regime = classify_regime(picks) if picks else "dead"
        cfg = cfg_for_day(d, regime)
        if cfg is None or not picks:
            daily.append({"date": d, "regime": regime, "cfg": None, "pnl": 0, "equity": cash, "n_picks": len(picks)})
            continue
        if cfg != cur_cfg_path:
            set_strategy_params(_merged(cfg))
            tgc.USE_DYNAMIC_SLIPPAGE = True
            tgc.VOL_CAP_PCT = 5.0
            cur_cfg_path = cfg
        is_cash = cash < MARGIN_THRESHOLD
        try:
            _, end_c, unset, _ = tgc.simulate_day_combined(picks, cash, cash_account=is_cash)
        except Exception:
            daily.append({"date": d, "regime": regime, "cfg": cfg, "pnl": 0, "equity": cash, "n_picks": len(picks)})
            continue
        pnl = end_c - cash; cash = end_c
        if is_cash: cash += unset
        daily.append({"date": d, "regime": regime, "cfg": cfg, "pnl": pnl, "equity": cash, "n_picks": len(picks)})
    return cash, daily


def summarize(label, final_cash, daily):
    total_pnl = final_cash - STARTING_CASH
    n = len(daily)
    traded = [r for r in daily if r["cfg"] is not None]
    pnls = np.array([r["pnl"] for r in traded])
    wins = int((pnls > 0).sum()); losses = int((pnls < 0).sum())
    skip_dead = sum(1 for r in daily if r["regime"] == "dead" and r["cfg"] is None)
    skip_corp = sum(1 for r in daily if r["regime"] == "corp-action")
    skip_other = sum(1 for r in daily if r["cfg"] is None and r["regime"] not in ("dead", "corp-action"))
    print(f"\n{label}")
    print(f"  Final equity ${final_cash:,.0f}  PnL ${total_pnl:+,.0f}  ({100*total_pnl/STARTING_CASH:+.1f}%)")
    print(f"  Days: {n} total | traded={len(traded)} (wins={wins}, losses={losses}, flat={len(traded)-wins-losses}) | "
          f"skip_dead={skip_dead}, skip_corp={skip_corp}, skip_other={skip_other}")
    if len(pnls) > 1 and pnls.std() > 0:
        sharpe = pnls.mean() / pnls.std() * np.sqrt(252)
        print(f"  Sharpe (per-day) {sharpe:.2f}  avg ${pnls.mean():+,.0f}  std ${pnls.std():,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2019,2020")
    args = ap.parse_args()

    for y in args.years.split(","):
        y = y.strip()
        if y not in YEAR_DIRS: continue
        dirs = [d for d in YEAR_DIRS[y] if os.path.exists(d)]
        if not dirs: continue
        print("\n" + "=" * 92)
        print(f"YEAR {y}")
        print("=" * 92)
        dates, dp = load_all_picks(dirs)
        regime_counts = {"dead": 0, "normal": 0, "squeeze": 0, "corp-action": 0}
        for d in dates:
            picks = dp.get(d, [])
            if _is_corp_action(picks):
                regime_counts["corp-action"] += 1
            else:
                regime_counts[classify_regime(picks)] += 1
        print(f"\n{len(dates)} days | regimes: {regime_counts}")

        # 5 backtest variants
        scenarios = [
            ("ROUTED  (squeeze→#818, normal→#571)",
                lambda _d, r: CFG_818 if r == "squeeze" else (CFG_571 if r == "normal" else None)),
            ("#818 EVERY day (no switch)",
                lambda _d, r: CFG_818 if r in ("squeeze", "normal") else None),
            ("#635 PRE-SPLIT best (24-25 combined PnL $26.3M) EVERY day",
                lambda _d, r: CFG_635 if r in ("squeeze", "normal") else None),
            ("#571 EVERY day (no switch)",
                lambda _d, r: CFG_571 if r in ("squeeze", "normal") else None),
            ("#6 generalist EVERY day",
                lambda _d, r: CFG_6 if r in ("squeeze", "normal") else None),
        ]

        equity_curves = {}
        for label, fn in scenarios:
            final, daily = run_year(label, dates, dp, fn)
            summarize(label, final, daily)
            eq = [STARTING_CASH] + [r["equity"] for r in daily]
            equity_curves[label] = eq

        # Equity-curve chart
        out_path = f"results/regime_switch_{y}_equity.png"
        os.makedirs("results", exist_ok=True)
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = {"ROUTED  (squeeze→#818, normal→#571)": "#d62728",
                   "#818 EVERY day (no switch)": "#2ca02c",
                   "#635 PRE-SPLIT best (24-25 combined PnL $26.3M) EVERY day": "#9467bd",
                   "#571 EVERY day (no switch)": "#ff7f0e",
                   "#6 generalist EVERY day": "#1f77b4"}
        for label, eq in equity_curves.items():
            ax.plot(range(len(eq)), eq, label=label, linewidth=1.8, color=colors.get(label, "#444"))
        ax.axhline(STARTING_CASH, linestyle="--", color="#aaa", linewidth=0.8)
        ax.set_title(f"{y} — Regime-switching vs single-config equity curves")
        ax.set_xlabel("Trading day index")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
