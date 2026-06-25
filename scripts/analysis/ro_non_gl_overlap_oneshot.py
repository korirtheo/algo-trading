"""One-shot R-O backtest excluding (ticker, date) where G/L fires under #511.

Two modes:
  - any_green_above_open
  - reclaim_after_dip

Uses default #511 exits (target=62, stop=25, time=12, trail=0.5/act=0).
30% × $25K position, no compounding, no participation cap modeling.

Question answered: what is the additive R-O PnL on top of G/L?
(Excluding picks where G/L already bought the SAME ticker on the SAME day.
Other tickers on G/L-trade days are still R-O candidates.)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict, Counter

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


def find_any_green_above_open(mh, day_open):
    if day_open is None or day_open <= 0:
        return None, None
    for i, (ts, row) in enumerate(mh.iterrows()):
        c = float(row["Close"])
        if c > day_open:
            return i, c
    return None, None


def find_reclaim_after_dip(mh, day_open):
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

    print("Running #511 baseline to identify (ticker, date) where G/L fires...")
    cash = STARTING_CASH
    gl_fires = set()
    gl_trade_count = 0
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
                gl_fires.add((st.get("ticker"), d))
                gl_trade_count += 1
        cash = end_c + (unset if is_cash else 0)
    print(f"  G/L trades: {gl_trade_count}, unique (ticker,date) pairs: {len(gl_fires)}")

    # Run both R-O modes WITH and WITHOUT G/L exclusion
    print()
    print("=" * 110)
    print("  R-O additive PnL (excluding (ticker,date) where G/L fires under #511)")
    print("  Exits: #511 (target=62, stop=25, time=12, trail=0.5/act=0)  |  Size: 30% × $25K")
    print("=" * 110)

    pos_cost = STARTING_CASH * POSITION_PCT

    def _run_mode(fn, mode_name, exclude_gl):
        trades = []
        excluded = 0
        for d in dates:
            for p in picks_by_date.get(d, []):
                if exclude_gl and (p["ticker"], d) in gl_fires:
                    excluded += 1
                    continue
                mh = p.get("market_hour_candles")
                if mh is None or len(mh) < 2:
                    continue
                day_open = float(mh.iloc[0]["Open"])
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
                trades.append({"ticker": p["ticker"], "date": d, "entry": ep, "exit": exit_p,
                              "pnl": pnl, "reason": reason, "bar_idx": idx})
        n = len(trades)
        if n == 0:
            return {"label": mode_name, "n": 0, "excluded": excluded}
        tot = sum(t["pnl"] for t in trades)
        wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        losses = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        wr = sum(1 for t in trades if t["pnl"] > 0) / n * 100
        pf = wins / losses if losses > 0 else 99.0
        days = len({t["date"] for t in trades})
        bar_dist = Counter(t["bar_idx"] for t in trades)
        return {"label": mode_name, "n": n, "days": days, "total_pnl": tot,
                "mean": tot/n, "wr": wr, "pf": pf, "excluded": excluded,
                "bar_dist_top5": bar_dist.most_common(5)}

    print(f"{'mode':<40} {'trades':>7} {'days':>5} {'total_pnl':>13} {'mean':>7} {'WR%':>5} {'pf':>5}")
    print("-" * 110)
    results = []
    for mode_name, fn in [("any_green_above_open", find_any_green_above_open),
                          ("reclaim_after_dip", find_reclaim_after_dip)]:
        for exclude in [False, True]:
            label = f"{mode_name}{' [EXCL G/L]' if exclude else ''}"
            r = _run_mode(fn, label, exclude_gl=exclude)
            results.append(r)
            if r["n"] == 0:
                print(f"{label:<40} 0 trades")
                continue
            print(f"{label:<40} {r['n']:>7} {r['days']:>5} ${r['total_pnl']:>+11,.0f} ${r['mean']:>+5,.0f} {r['wr']:>4.1f}% {r['pf']:>4.2f}")
            print(f"{'  entry bar idx top5:':<40} {r['bar_dist_top5']}  (excluded: {r['excluded']} picks)")

    out_path = "results/ro_non_gl_overlap_oneshot.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"gl_trade_count": gl_trade_count, "gl_unique_pairs": len(gl_fires),
                   "results": results}, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
