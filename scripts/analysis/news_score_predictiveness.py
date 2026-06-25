"""Does morning news predict G/L trade outcome?

For each historical G/L trade under #511 deployed params, tag with the PIT
news bucket and compute WR + mean PnL per bucket. Answers: does a morning
news catalyst predict trade success enough to justify a filter or modulator?

Output: bucket stats table + JSON dump.

Methodology:
  - Train on FULL 2022-2025 (using all data dirs we have)
  - Run G+L with #511 deployed params
  - For each trade, look up news_cache for (ticker, trade_date)
  - Bucket by:
      0       = no news before 9:30 ET
      1-2     = 1-2 articles
      3+      = 3+ articles
      catalyst= has non-scanner article
  - Report: count, total PnL, mean PnL, WR per bucket
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict

import numpy as np

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

# Use ALL of 2022 + ALL of 2024 + ALL of 2025 (skip 2023 thin data)
DATA_DIRS = [
    "stored_data_2022",
    "stored_data_combined",
    "stored_data_jan_mar_2024",
    "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024",
    "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025",
    "stored_data_apr_jun_2025",
    "stored_data_jul_2025",
    "stored_data_oos",
]
DATE_LO = "2022-01-01"
DATE_HI = "2025-12-31"


def _bucket(n, has_catalyst):
    if has_catalyst:
        return "catalyst"
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    return "3+"


def main():
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD
    from news_filter import count_pit

    with open(BASELINE) as f:
        baseline = json.load(f)
    with open(W21B_DEPLOY) as f:
        p511 = json.load(f)["params"]
    merged = {**baseline, **p511}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)

    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0
    tgc.NEWS_MODULATOR_ENABLED = False

    print(f"Loading data from {len(DATA_DIRS)} dirs...")
    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
    print(f"Running on {len(dates)} trading days {dates[0]} -> {dates[-1]}")

    cash = STARTING_CASH
    trades = []
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp:
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                trades.append({
                    "date": d,
                    "ticker": st.get("ticker"),
                    "strategy": st.get("strategy"),
                    "pnl": st["pnl"],
                    "cost": st["position_cost"],
                    "exit_reason": st.get("exit_reason"),
                })
        cash = end_c + (unset if is_cash else 0)

    print(f"\nTotal trades: {len(trades)}")
    print(f"Total PnL: ${sum(t['pnl'] for t in trades):,.0f}")
    print(f"Final cash: ${cash:,.0f}")

    # Bucket by news
    buckets = defaultdict(list)
    bucket_meta = {}  # ticker|date -> (n, has_catalyst)
    for t in trades:
        n, has_cat = count_pit(t["ticker"], t["date"])
        bucket_meta[f"{t['ticker']}|{t['date']}"] = (n, has_cat)
        b = _bucket(n, has_cat)
        buckets[b].append(t)
        # Also add to "any-news" if n > 0
        if n > 0:
            buckets["any_news"].append(t)

    print()
    print("=" * 100)
    print(f"  NEWS-PREDICTIVENESS ANALYSIS — #511 params on {len(trades)} G+L trades 2022-2025")
    print("=" * 100)
    print(f"{'bucket':<12} {'n':>6} {'%':>5} {'total_pnl':>14} {'mean_pnl':>10} {'WR%':>6} {'pf':>6}")
    print("-" * 100)

    def _stats(rows):
        n = len(rows)
        if n == 0:
            return 0, 0, 0, 0, 0
        tot = sum(r["pnl"] for r in rows)
        wins = [r for r in rows if r["pnl"] > 0]
        losses = [r for r in rows if r["pnl"] <= 0]
        wr = len(wins) / n * 100 if n else 0
        wsum = sum(r["pnl"] for r in wins)
        lsum = abs(sum(r["pnl"] for r in losses))
        pf = (wsum / lsum) if lsum > 0 else 99.0
        return n, tot, tot / n, wr, pf

    bucket_order = ["0", "1-2", "3+", "catalyst", "any_news"]
    for b in bucket_order:
        n, tot, mean, wr, pf = _stats(buckets.get(b, []))
        if n == 0:
            continue
        pct = n / len(trades) * 100
        print(f"{b:<12} {n:>6} {pct:>4.1f}% ${tot:>+12,.0f} ${mean:>+8,.0f} {wr:>5.1f}% {pf:>5.2f}")

    # ALL
    n, tot, mean, wr, pf = _stats(trades)
    print("-" * 100)
    print(f"{'ALL':<12} {n:>6} {100:>4.1f}% ${tot:>+12,.0f} ${mean:>+8,.0f} {wr:>5.1f}% {pf:>5.2f}")

    # By strategy + bucket
    print()
    print("Per-strategy breakdown:")
    for strat in ["G", "L"]:
        srows = [t for t in trades if t.get("strategy") == strat]
        if not srows:
            continue
        print(f"\n  --- Strategy {strat} ({len(srows)} trades) ---")
        print(f"  {'bucket':<12} {'n':>5} {'mean_pnl':>10} {'WR%':>6} {'pf':>6}")
        s_buckets = defaultdict(list)
        for t in srows:
            n, has_cat = bucket_meta[f"{t['ticker']}|{t['date']}"]
            s_buckets[_bucket(n, has_cat)].append(t)
            if n > 0:
                s_buckets["any_news"].append(t)
        for b in bucket_order:
            n, tot, mean, wr, pf = _stats(s_buckets.get(b, []))
            if n == 0:
                continue
            print(f"  {b:<12} {n:>5} ${mean:>+8,.0f} {wr:>5.1f}% {pf:>5.2f}")

    # Save summary JSON
    summary = {
        "params": "W21b #511",
        "data_window": f"{DATE_LO} to {DATE_HI}",
        "n_trades": len(trades),
        "buckets": {},
    }
    for b in bucket_order:
        if buckets.get(b):
            n, tot, mean, wr, pf = _stats(buckets[b])
            summary["buckets"][b] = {
                "n": n, "total_pnl": float(tot), "mean_pnl": float(mean),
                "wr_pct": float(wr), "pf": float(pf),
            }
    n, tot, mean, wr, pf = _stats(trades)
    summary["buckets"]["ALL"] = {
        "n": n, "total_pnl": float(tot), "mean_pnl": float(mean),
        "wr_pct": float(wr), "pf": float(pf),
    }
    out_path = "results/news_predictiveness_2022_2025.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
