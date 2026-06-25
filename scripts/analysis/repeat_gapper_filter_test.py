"""Hardcoded filter test: 'past-year' repeat-gapper filter on 2026.

Compares baseline (no filter) vs:
  - Filter A: require ticker to have at least one prior appearance in past 365d
              (drops first-timers AND >365d stales)
  - Filter B: drop only if days_since_last > 365
              (keeps first-timers and all recents; only removes truly stale)

Picks are filtered BEFORE the simulator sees them, so cash freed up by
filtered tickers correctly funds remaining trades same day.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import glob
import csv
from collections import defaultdict
from datetime import datetime
from bisect import bisect_left

import numpy as np

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]


def build_appearance_index():
    index = defaultdict(set)
    for csv_path in sorted(glob.glob("stored_data*/daily_top_gainers.csv")):
        with open(csv_path) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) < 2: continue
                date, ticker = row[0], row[1]
                if not date or not ticker: continue
                try:
                    d = datetime.strptime(date[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
                except Exception:
                    continue
                index[ticker].add(d)
    return {t: sorted(dates) for t, dates in index.items()}


def days_since_last_prior(index, ticker, trade_date):
    """Returns (n_priors, days_since_last). days_since_last is None if no priors."""
    history = index.get(ticker, [])
    if not history:
        return 0, None
    i = bisect_left(history, trade_date)
    if i == 0:
        return 0, None
    last = history[i - 1]
    try:
        d1 = datetime.strptime(last, "%Y-%m-%d")
        d2 = datetime.strptime(trade_date, "%Y-%m-%d")
        return i, (d2 - d1).days
    except Exception:
        return i, None


def make_filter(index, mode):
    """mode: 'baseline' | 'A_past_year_required' | 'B_drop_stale_only'"""
    if mode == "baseline":
        return None
    def keep(pick, date_str):
        ticker = pick.get("ticker")
        if not ticker: return False
        n, days = days_since_last_prior(index, ticker, date_str)
        if mode == "A_past_year_required":
            # Must have ≥1 prior appearance within last 365 days
            return days is not None and days <= 365
        if mode == "B_drop_stale_only":
            # Drop only if has priors AND last was >365 days ago.
            # First-timers (days=None, n=0) are KEPT.
            if days is None:
                return True
            return days <= 365
        return True
    return keep


def forward_with_filter(config_path, index, mode):
    """Forward 2026 with optional picks-level filter."""
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

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
    tgc.MARGIN_MULTIPLIER = 1.0

    keep_fn = make_filter(index, mode)

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    daily_eq = [cash]
    n_trades = 0
    n_dropped = 0
    n_kept_picks = 0
    worst_trade = 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            daily_eq.append(cash); continue
        if keep_fn:
            kept = []
            for p in day_picks:
                if keep_fn(p, d): kept.append(p)
                else: n_dropped += 1
            day_picks = kept
            n_kept_picks += len(day_picks)
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
                if st["pnl"] < worst_trade: worst_trade = st["pnl"]
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = float((eq - peak).min() / peak[(eq - peak).argmin()] * 100) if len(peak) and peak[(eq - peak).argmin()] > 0 else 0
    dd_dollar = float((eq - peak).min())
    return {
        "final": float(eq[-1]),
        "pnl": float(eq[-1] - STARTING_CASH),
        "multiple": float(eq[-1] / STARTING_CASH),
        "max_dd_pct": dd_pct,
        "max_dd_dollar": dd_dollar,
        "n_trades": n_trades,
        "n_picks_dropped": n_dropped,
        "worst_trade": float(worst_trade),
    }


def main():
    print("Building ticker appearance index...")
    index = build_appearance_index()
    print(f"  {len(index):,} tickers, {sum(len(v) for v in index.values()):,} total appearances\n")

    configs = [
        ("#124 W3 deployed",  "config/trial_124_microcap_pump_extracted.json"),
        ("#254 W7 (current)", "config/trial_254_w7_extracted.json"),
    ]
    modes = [
        ("baseline",                 "no filter (current)"),
        ("A_past_year_required",     "Filter A: require ≥1 prior within 365d"),
        ("B_drop_stale_only",        "Filter B: drop only days_since_last > 365"),
    ]

    print(f"{'config':<22} {'filter':<28} {'final$':>11} {'PnL':>11} {'multi':>7} {'DD%':>7} {'#tr':>5} {'#drop':>7}")
    print(f"{'-'*22} {'-'*28} {'-'*11} {'-'*11} {'-'*7} {'-'*7} {'-'*5} {'-'*7}")

    results = {}
    for label, path in configs:
        if not os.path.exists(path):
            print(f"  [skip] {path}"); continue
        for mode_key, mode_label in modes:
            r = forward_with_filter(path, index, mode_key)
            results[f"{label} | {mode_key}"] = r
            print(f"{label:<22} {mode_label:<28} ${r['final']:>9,.0f} ${r['pnl']:>+9,.0f} {r['multiple']:>6.2f}x {r['max_dd_pct']:>6.1f}% {r['n_trades']:>5} {r['n_picks_dropped']:>7}")
        print()

    # Lift vs baseline
    print(f"{'='*92}\n  LIFT ANALYSIS — filter $ delta vs each config's baseline\n{'='*92}")
    for label, _ in configs:
        bk = f"{label} | baseline"
        if bk not in results: continue
        base = results[bk]
        for mode_key, mode_label in modes[1:]:
            k = f"{label} | {mode_key}"
            if k not in results: continue
            r = results[k]
            d_pnl = r["pnl"] - base["pnl"]
            d_dd = r["max_dd_pct"] - base["max_dd_pct"]
            sign = "✓ better" if d_pnl > 0 else "✗ worse"
            print(f"  {label:<22} {mode_label:<28}  PnL: ${d_pnl:>+11,.0f}  DD%: {d_dd:>+5.1f}pp  {sign}")

    out = "results/walk_forward_v7_news/repeat_gapper_filter_test.json"
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
