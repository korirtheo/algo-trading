"""Test the 'repeat premarket gapper' hypothesis: tickers that have ALREADY
appeared in past premarket-gainer lists perform better than first-timers.

Builds a point-in-time appearance index from all daily_top_gainers.csv files
(2019-2026), then runs a forward backtest on #124 and #254 and tags every
trade with N_prior_appearances + days_since_last. Reports WR / avg-PnL by
bucket so we can see if the signal is real.

If the signal IS real, the next step is to add `min_prior_appearances` as
an Optuna tunable in W8.
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


STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
OUTDIR = "results/walk_forward_v7_news"


def build_appearance_index():
    """Build {ticker: sorted_list_of_date_strings} from ALL gainer CSVs."""
    index = defaultdict(set)
    for csv_path in sorted(glob.glob("stored_data*/daily_top_gainers.csv")):
        with open(csv_path) as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) < 2:
                    continue
                date, ticker = row[0], row[1]
                if not date or not ticker:
                    continue
                # Normalize date to YYYY-MM-DD
                try:
                    d = datetime.strptime(date[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
                except Exception:
                    continue
                index[ticker].add(d)
    # Sort dates per ticker
    return {t: sorted(dates) for t, dates in index.items()}


def prior_appearances(index, ticker, trade_date):
    """Return (count_prior, days_since_last) — STRICTLY before trade_date.

    None for days_since_last if there are no priors.
    """
    history = index.get(ticker, [])
    if not history:
        return 0, None
    # Strict less-than: trade day itself doesn't count
    i = bisect_left(history, trade_date)
    if i == 0:
        return 0, None
    # i appearances strictly before trade_date
    last_date = history[i - 1]
    try:
        d1 = datetime.strptime(last_date, "%Y-%m-%d")
        d2 = datetime.strptime(trade_date, "%Y-%m-%d")
        days = (d2 - d1).days
    except Exception:
        days = None
    return i, days


def replay_with_trades(config_path):
    """Forward 2026 returning trade list with (date, ticker, strategy, pnl, cost)."""
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
                    "pct": float(st["pnl"]) / float(st["position_cost"]) * 100 if st["position_cost"] > 0 else 0,
                })
        cash = end_c + (unset if is_cash else 0)
    return trades


def analyze(label, trades, index):
    print(f"\n{'='*92}")
    print(f"  {label}  —  {len(trades)} trades on 2026")
    print(f"{'='*92}")

    # Tag each trade
    tagged = []
    for t in trades:
        n_prior, days_last = prior_appearances(index, t["ticker"], t["date"])
        t = {**t, "n_prior": n_prior, "days_last": days_last}
        tagged.append(t)

    # Bucket by n_prior
    buckets = [
        ("0  (first-timer)", lambda t: t["n_prior"] == 0),
        ("1-2",              lambda t: 1 <= t["n_prior"] <= 2),
        ("3-5",              lambda t: 3 <= t["n_prior"] <= 5),
        ("6-10",             lambda t: 6 <= t["n_prior"] <= 10),
        ("11+",              lambda t: t["n_prior"] >= 11),
    ]

    print(f"  {'n_prior bucket':<20} {'n':>4} {'WR%':>6} {'avgPnL$':>10} {'avg%':>7} {'totalPnL$':>12}")
    print(f"  {'-'*20} {'-'*4} {'-'*6} {'-'*10} {'-'*7} {'-'*12}")
    overall_pnl = sum(t["pnl"] for t in tagged)
    for name, pred in buckets:
        sub = [t for t in tagged if pred(t)]
        if not sub:
            print(f"  {name:<20} {0:>4} {'-':>6} {'-':>10} {'-':>7} {'-':>12}")
            continue
        n = len(sub)
        wins = sum(1 for t in sub if t["pnl"] > 0)
        wr = wins/n*100
        avg = sum(t["pnl"] for t in sub) / n
        avg_pct = sum(t["pct"] for t in sub) / n
        tot = sum(t["pnl"] for t in sub)
        share = tot/overall_pnl*100 if overall_pnl else 0
        print(f"  {name:<20} {n:>4} {wr:>5.1f}% ${avg:>+8,.0f} {avg_pct:>+6.2f}% ${tot:>+10,.0f}  ({share:+.0f}% of total)")

    # Days-since-last
    print(f"\n  Days since LAST appearance (only for repeat tickers):")
    print(f"  {'bucket':<20} {'n':>4} {'WR%':>6} {'avgPnL$':>10} {'avg%':>7}")
    print(f"  {'-'*20} {'-'*4} {'-'*6} {'-'*10} {'-'*7}")
    repeat = [t for t in tagged if t["n_prior"] > 0 and t["days_last"] is not None]
    rb = [
        ("≤7 days   (very recent)", lambda t: t["days_last"] <= 7),
        ("8-30      (recent)",      lambda t: 8 <= t["days_last"] <= 30),
        ("31-90     (medium)",      lambda t: 31 <= t["days_last"] <= 90),
        ("91-365    (old)",         lambda t: 91 <= t["days_last"] <= 365),
        (">365      (very old)",    lambda t: t["days_last"] > 365),
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

    # Save tagged trades
    out = f"{OUTDIR}/repeat_gapper_{label.replace(' ', '_').replace('#', 'trial_').replace('(', '').replace(')', '')}.json"
    with open(out, "w") as f:
        json.dump(tagged, f, indent=2, default=str)
    print(f"\n  Wrote {out}")
    return tagged


def main():
    print("Building ticker-appearance index from all gainer CSVs...")
    index = build_appearance_index()
    n_tickers = len(index)
    n_appearances = sum(len(v) for v in index.values())
    print(f"  {n_tickers:,} unique tickers, {n_appearances:,} total appearances")

    # Quick stats on the index itself
    pa_counts = [len(v) for v in index.values()]
    multi_appear = sum(1 for c in pa_counts if c > 1)
    print(f"  Tickers w/ >1 appearance: {multi_appear:,} ({100*multi_appear/n_tickers:.1f}% of universe)")
    avg_appearances_for_repeaters = sum(c for c in pa_counts if c > 1) / max(multi_appear, 1)
    print(f"  Avg appearances among repeaters: {avg_appearances_for_repeaters:.1f}")

    for label, path in [
        ("#124 W3 deployed",  "config/trial_124_microcap_pump_extracted.json"),
        ("#254 W7 (current)", "config/trial_254_w7_extracted.json"),
    ]:
        if not os.path.exists(path):
            print(f"[skip] {path}"); continue
        print(f"\nReplaying {label}...")
        trades = replay_with_trades(path)
        analyze(label, trades, index)


if __name__ == "__main__":
    main()
