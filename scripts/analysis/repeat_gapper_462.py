"""Repeat-gapper signal + filter test for trial #462 (W7 top-train, weak forward).

Same methodology as repeat_gapper_signal.py + repeat_gapper_filter_test.py,
focused on the single trial that highlighted the train≠forward problem.
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
CONFIG_PATH = "config/trial_462_w7_extracted.json"
LABEL = "#462 W7 (overfit peak)"


def build_appearance_index():
    index = defaultdict(set)
    for csv_path in sorted(glob.glob("stored_data*/daily_top_gainers.csv")):
        with open(csv_path) as f:
            r = csv.reader(f); next(r, None)
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


def days_since_last(index, ticker, trade_date):
    history = index.get(ticker, [])
    if not history: return 0, None
    i = bisect_left(history, trade_date)
    if i == 0: return 0, None
    try:
        return i, (datetime.strptime(trade_date, "%Y-%m-%d")
                    - datetime.strptime(history[i-1], "%Y-%m-%d")).days
    except Exception:
        return i, None


def setup_tgc(params):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
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
    return tgc


def replay_with_trades(config_path):
    """Returns list of trades + per-day equity curve."""
    with open(config_path) as f: cfg = json.load(f)
    params = cfg.get("params", cfg)
    tgc = setup_tgc(params)
    from test_full import load_all_picks, MARGIN_THRESHOLD

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    trades = []
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                trades.append({
                    "date": d, "ticker": st["ticker"],
                    "strategy": st.get("strategy", "?"),
                    "pnl": float(st["pnl"]),
                    "cost": float(st["position_cost"]),
                    "pct": float(st["pnl"])/float(st["position_cost"])*100 if st["position_cost"] > 0 else 0,
                })
        cash = end_c + (unset if is_cash else 0)
    return trades


def forward_with_filter(config_path, index, mode):
    """Forward 2026 with optional picks filter (baseline | A | B)."""
    with open(config_path) as f: cfg = json.load(f)
    params = cfg.get("params", cfg)
    tgc = setup_tgc(params)
    from test_full import load_all_picks, MARGIN_THRESHOLD

    def keep(pick, date_str):
        if mode == "baseline": return True
        ticker = pick.get("ticker")
        if not ticker: return False
        _, days = days_since_last(index, ticker, date_str)
        if mode == "A_past_year_required":
            return days is not None and days <= 365
        if mode == "B_drop_stale_only":
            if days is None: return True   # keep first-timers
            return days <= 365
        return True

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    eq = [cash]
    n_trades = 0
    n_dropped = 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            eq.append(cash); continue
        if mode != "baseline":
            kept = []
            for p in day_picks:
                if keep(p, d): kept.append(p)
                else: n_dropped += 1
            day_picks = kept
        if not day_picks:
            eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_trades += 1
        cash = end_c + (unset if is_cash else 0)
        eq.append(cash)
    eqa = np.array(eq)
    peak = np.maximum.accumulate(eqa)
    dd_pct = float((eqa - peak).min() / peak[(eqa - peak).argmin()] * 100) if len(peak) and peak[(eqa - peak).argmin()] > 0 else 0
    return {
        "final": float(eqa[-1]),
        "pnl": float(eqa[-1] - STARTING_CASH),
        "multi": float(eqa[-1] / STARTING_CASH),
        "max_dd_pct": dd_pct,
        "n_trades": n_trades,
        "n_dropped": n_dropped,
    }


def bucket_report(label, trades, index):
    tagged = []
    for t in trades:
        n_prior, days_last = days_since_last(index, t["ticker"], t["date"])
        tagged.append({**t, "n_prior": n_prior, "days_last": days_last})

    print(f"\n{'='*92}\n  {label}  —  {len(trades)} trades on 2026\n{'='*92}")
    print(f"  {'n_prior':<20} {'n':>4} {'WR%':>6} {'avgPnL$':>10} {'avg%':>7} {'totalPnL$':>12}")
    print(f"  {'-'*20} {'-'*4} {'-'*6} {'-'*10} {'-'*7} {'-'*12}")
    total = sum(t["pnl"] for t in tagged)
    buckets = [
        ("0  (first-timer)", lambda t: t["n_prior"] == 0),
        ("1-2",              lambda t: 1 <= t["n_prior"] <= 2),
        ("3-5",              lambda t: 3 <= t["n_prior"] <= 5),
        ("6-10",             lambda t: 6 <= t["n_prior"] <= 10),
        ("11+",              lambda t: t["n_prior"] >= 11),
    ]
    for name, pred in buckets:
        sub = [t for t in tagged if pred(t)]
        if not sub:
            print(f"  {name:<20} {0:>4} {'-':>6} {'-':>10} {'-':>7} {'-':>12}")
            continue
        n = len(sub); wins = sum(1 for t in sub if t["pnl"] > 0)
        wr = wins/n*100; avg = sum(t["pnl"] for t in sub)/n
        avg_pct = sum(t["pct"] for t in sub)/n
        tot = sum(t["pnl"] for t in sub)
        share = tot/total*100 if total else 0
        print(f"  {name:<20} {n:>4} {wr:>5.1f}% ${avg:>+8,.0f} {avg_pct:>+6.2f}% ${tot:>+10,.0f}  ({share:+.0f}%)")

    print(f"\n  Days since LAST appearance:")
    print(f"  {'bucket':<20} {'n':>4} {'WR%':>6} {'avgPnL$':>10} {'avg%':>7}")
    print(f"  {'-'*20} {'-'*4} {'-'*6} {'-'*10} {'-'*7}")
    repeat = [t for t in tagged if t["n_prior"] > 0 and t["days_last"] is not None]
    rb = [
        ("<=7 days  (very recent)", lambda t: t["days_last"] <= 7),
        ("8-30",                    lambda t: 8 <= t["days_last"] <= 30),
        ("31-90",                   lambda t: 31 <= t["days_last"] <= 90),
        ("91-365",                  lambda t: 91 <= t["days_last"] <= 365),
        (">365     (stale)",        lambda t: t["days_last"] > 365),
    ]
    for name, pred in rb:
        sub = [t for t in repeat if pred(t)]
        if not sub:
            print(f"  {name:<20} {0:>4} {'-':>6} {'-':>10} {'-':>7}")
            continue
        n = len(sub); wins = sum(1 for t in sub if t["pnl"] > 0)
        wr = wins/n*100; avg = sum(t["pnl"] for t in sub)/n
        avg_pct = sum(t["pct"] for t in sub)/n
        print(f"  {name:<20} {n:>4} {wr:>5.1f}% ${avg:>+8,.0f} {avg_pct:>+6.2f}%")


def main():
    print("Building ticker appearance index...")
    index = build_appearance_index()
    print(f"  {len(index):,} tickers, {sum(len(v) for v in index.values()):,} appearances\n")

    if not os.path.exists(CONFIG_PATH):
        print(f"[skip] {CONFIG_PATH} not found"); return

    print(f"Replaying {LABEL} on 2026...")
    trades = replay_with_trades(CONFIG_PATH)
    bucket_report(LABEL, trades, index)

    print(f"\n{'='*92}\n  Filter test on {LABEL}\n{'='*92}")
    print(f"  {'filter':<30} {'final$':>11} {'PnL':>11} {'multi':>7} {'DD%':>7} {'#tr':>5} {'#drop':>7}")
    print(f"  {'-'*30} {'-'*11} {'-'*11} {'-'*7} {'-'*7} {'-'*5} {'-'*7}")
    modes = [
        ("baseline",              "no filter (current)"),
        ("A_past_year_required",  "Filter A: req >=1 prior <=365d"),
        ("B_drop_stale_only",     "Filter B: drop only >365d"),
    ]
    base = None
    results = {}
    for key, lbl in modes:
        r = forward_with_filter(CONFIG_PATH, index, key)
        results[key] = r
        if key == "baseline": base = r
        print(f"  {lbl:<30} ${r['final']:>9,.0f} ${r['pnl']:>+9,.0f} {r['multi']:>6.2f}x {r['max_dd_pct']:>6.1f}% {r['n_trades']:>5} {r['n_dropped']:>7}")

    print(f"\n  Lift vs baseline:")
    for key, lbl in modes[1:]:
        r = results[key]
        d_pnl = r["pnl"] - base["pnl"]
        d_dd = r["max_dd_pct"] - base["max_dd_pct"]
        sign = "better" if d_pnl > 0 else "worse"
        print(f"    {lbl:<30}  PnL: ${d_pnl:>+11,.0f}  DD%: {d_dd:>+5.1f}pp  ({sign})")

    out = "results/walk_forward_v7_news/repeat_gapper_trial_462.json"
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
