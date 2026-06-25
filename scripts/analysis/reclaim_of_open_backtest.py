"""Reclaim of Opening (R-O) signal backtest.

Pattern (simple version): gap-up ticker dips below day's open at some point,
then reclaims (closes above open). Enter at the reclaim bar's close.

Comparison modes:
  - reclaim_after_dip: enter first close > day_open AFTER any earlier bar
                       closed <= day_open (the proper dip-and-rip)
  - any_green_above_open: enter first close > day_open with no prior requirement
                          (almost always bar 1 of a healthy gap-up)

Uses #511's exit logic. Sized 30% of $25K per trade, no compounding.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict

STARTING_CASH = 25_000
POSITION_PCT = 0.30
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
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
    "stored_data",
    "stored_data_mar_may_2026",
    "stored_data_jun_2026",
    "stored_data_2026_gap_fill",
]
DATE_LO = "2022-01-01"
DATE_HI = "2026-06-24"

# #511 exits
G_TARGET_PCT = 62.0
G_STOP_PCT = 25.0
G_TIME_MIN = 12
G_TRAIL_PCT = 0.5
G_TRAIL_ACT_PCT = 0.0


def _simulate_trade(entry_price, bars_after):
    if entry_price <= 0:
        return None, None
    target = entry_price * (1 + G_TARGET_PCT / 100)
    stop   = entry_price * (1 - G_STOP_PCT / 100)
    peak   = entry_price
    trail_stop = None
    max_bars = G_TIME_MIN // 2
    for i, (ts, row) in enumerate(bars_after.iterrows()):
        if i >= max_bars:
            return float(row["Close"]), "TIME_STOP"
        c_high = float(row["High"]); c_low = float(row["Low"]); c_close = float(row["Close"])
        if c_high >= target:
            return target, "TARGET"
        if c_low <= stop:
            return stop, "STOP"
        if c_high > peak:
            peak = c_high
        new_trail = peak * (1 - G_TRAIL_PCT / 100)
        if trail_stop is None or new_trail > trail_stop:
            trail_stop = new_trail
        if c_low <= trail_stop:
            return trail_stop, "TRAIL"
    return float(bars_after.iloc[-1]["Close"]) if len(bars_after) else None, "EOD"


def find_reclaim_after_dip(mh, day_open):
    """First bar that closes > day_open AFTER any earlier bar closed <= day_open."""
    if day_open is None or day_open <= 0:
        return None, None
    ever_below = False
    for i, (ts, row) in enumerate(mh.iterrows()):
        c = float(row["Close"])
        if ever_below and c > day_open:
            return i, c
        if c <= day_open:
            ever_below = True
    return None, None


def find_any_green_above_open(mh, day_open):
    """First bar that closes > day_open — no prior requirement (control case)."""
    if day_open is None or day_open <= 0:
        return None, None
    for i, (ts, row) in enumerate(mh.iterrows()):
        c = float(row["Close"])
        if c > day_open:
            return i, c
    return None, None


def main():
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

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
    tgc.NEWS_MODULATOR_ENABLED = False

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    print(f"Loading {len(dirs)} dirs...")
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
    print(f"Trading days: {len(dates)}")

    # Pre-compute each (ticker,date) pick's bars + day_open
    candidates = []  # list of (ticker, date, mh, day_open)
    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 3:
                continue
            day_open = float(mh.iloc[0]["Open"]) if len(mh) > 0 else None
            if day_open is None or day_open <= 0:
                continue
            candidates.append((p["ticker"], d, mh, day_open))
    print(f"Candidate (ticker,day) pairs: {len(candidates)}")

    print()
    print("=" * 110)
    print("  RECLAIM OF OPENING (R-O) — two-mode comparison")
    print("  Exits: #511 (target=62, stop=25, time=12, trail=0.5/act=0)")
    print("  Position size: 30% of $25K per trade, no compounding")
    print("=" * 110)
    print(f"{'mode':<28} {'trades':>7} {'days':>6} {'total_pnl':>13} {'mean':>8} {'WR%':>6} {'pf':>6}  {'exit breakdown':<40}")
    print("-" * 110)

    summary = {}
    pos_cost = STARTING_CASH * POSITION_PCT
    modes = [
        ("reclaim_after_dip", find_reclaim_after_dip),
        ("any_green_above_open", find_any_green_above_open),
    ]
    for mode_name, fn in modes:
        trades = []
        for ticker, date, mh, day_open in candidates:
            idx, ep = fn(mh, day_open)
            if idx is None or ep <= 0:
                continue
            bars_after = mh.iloc[idx + 1:]
            if len(bars_after) == 0:
                continue
            exit_p, reason = _simulate_trade(ep, bars_after)
            if exit_p is None:
                continue
            shares = pos_cost / ep
            pnl = shares * (exit_p - ep)
            trades.append({"ticker": ticker, "date": date, "entry": ep, "exit": exit_p,
                          "pnl": pnl, "reason": reason, "bar_idx": idx})
        n = len(trades)
        if n == 0:
            print(f"{mode_name:<28} 0 — no signals")
            continue
        days = len({t["date"] for t in trades})
        tot = sum(t["pnl"] for t in trades)
        wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        losses = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        wr = sum(1 for t in trades if t["pnl"] > 0) / n * 100
        pf = wins / losses if losses > 0 else 99.0
        reasons = defaultdict(int)
        for t in trades:
            reasons[t["reason"]] += 1
        exit_str = " ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
        print(f"{mode_name:<28} {n:>7} {days:>6} ${tot:>+11,.0f} ${tot/n:>+6,.0f} {wr:>5.1f}% {pf:>5.2f}  {exit_str[:40]}")
        # Entry-bar distribution
        from collections import Counter
        bar_dist = Counter(t["bar_idx"] for t in trades)
        most_common_bars = bar_dist.most_common(5)
        print(f"{'  most-common entry bar idx:':<28} {most_common_bars}")
        summary[mode_name] = {"n": n, "days": days, "total_pnl": float(tot), "mean": float(tot/n),
                              "wr": float(wr), "pf": float(pf), "exit_breakdown": dict(reasons)}

    out_path = "results/reclaim_of_open_backtest.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
