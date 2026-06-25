"""Forward-test #124 and #254 on 2026 with MARGIN_MULTIPLIER=2.0.

Compares cash-only (1x) vs 2x intraday margin to show how Reg-T margin
would change PnL + drawdown profile. Slippage scales automatically because
the Almgren-Chriss model uses participation rate.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

STARTING_CASH = 25_000
DATA_DIRS = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
BASELINE = "config/trial_432_params.json"

CONFIGS = [
    ("#124 W3 deployed",  "config/trial_124_microcap_pump_extracted.json"),
    ("#254 W7 (current)", "config/trial_254_w7_extracted.json"),
]


def forward(config_path, margin_mult):
    with open(config_path) as f: cfg = json.load(f)
    params = cfg.get("params", cfg)
    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(params)
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = margin_mult

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])
    cash = STARTING_CASH
    daily_eq = [cash]
    n_trades = 0
    worst_trade = 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            daily_eq.append(cash); continue
        if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as nfp
            day_picks = nfp(day_picks, d, tgc.NEWS_MIN_ARTICLES, tgc.NEWS_REQUIRE_CATALYST)
        if not day_picks:
            daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_trades += 1
                worst_trade = min(worst_trade, st["pnl"])
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    return np.array(daily_eq), n_trades, worst_trade


def main():
    print(f"\n{'='*84}\n  2026 forward — cash-only (1x) vs 2x margin\n{'='*84}\n")
    print(f"  {'config':<22} {'leverage':>9} {'final':>11} {'PnL':>11} {'multi':>7} {'maxDD%':>7} {'maxDD$':>11} {'#tr':>5} {'worst':>10}")
    print(f"  {'-'*22} {'-'*9} {'-'*11} {'-'*11} {'-'*7} {'-'*7} {'-'*11} {'-'*5} {'-'*10}")
    rows = []
    for label, path in CONFIGS:
        if not os.path.exists(path):
            print(f"  [skip] {path}"); continue
        for mult in [1.0, 2.0]:
            eq, n, worst = forward(path, mult)
            peak = np.maximum.accumulate(eq)
            ddp = (eq - peak) / peak * 100
            ddd = eq - peak
            row = (label, mult, eq[-1], eq[-1]-STARTING_CASH, eq[-1]/STARTING_CASH, ddp.min(), ddd.min(), n, worst)
            rows.append(row)
            print(f"  {label:<22} {mult:>8.1f}x ${eq[-1]:>9,.0f} ${eq[-1]-STARTING_CASH:>+9,.0f} {eq[-1]/STARTING_CASH:>6.2f}x {ddp.min():>6.1f}% ${ddd.min():>+9,.0f} {n:>5} ${worst:>+8,.0f}")

    # Save
    out = "results/walk_forward_v7_news/margin_2x_comparison.json"
    with open(out, "w") as f:
        json.dump([
            {"config": r[0], "leverage": r[1], "final": r[2], "pnl": r[3],
             "multiple": r[4], "max_dd_pct": r[5], "max_dd_dollar": r[6],
             "n_trades": r[7], "worst_trade": r[8]}
            for r in rows
        ], f, indent=2)
    print(f"\n  Saved {out}")


if __name__ == "__main__":
    main()
