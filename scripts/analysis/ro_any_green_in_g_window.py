"""How many any_green_above_open R-O entries fall INSIDE a G holding window?

For each (ticker, date), simulate #511 to capture G's [entry_time, exit_time] window.
Then walk any_green_above_open candidates and count entries whose entry_ts is
inside that window (G was actively holding when R-O tried to buy).

Splits the 6,230 any_green trades into:
  - clean        : G never fired on this (ticker, date)
  - inside_g     : G was holding (entry_ts in [G.entry, G.exit])
  - after_g_exit : G fired but had already exited before R-O entry
  - before_g     : R-O entry strictly before G entered (rare)
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
    "stored_data_2022", "stored_data_combined",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos", "stored_data",
    "stored_data_mar_may_2026", "stored_data_jun_2026",
    "stored_data_2026_gap_fill",
]
DATE_LO = "2022-01-01"
DATE_HI = "2026-06-24"

G_TARGET_PCT = 62.0
G_STOP_PCT = 25.0
G_TIME_MIN = 12
G_TRAIL_PCT = 0.5
G_TRAIL_ACT_PCT = 0.0


def _simulate_ro_trade(entry_price, bars_after):
    if entry_price <= 0:
        return None, None
    target = entry_price * (1 + G_TARGET_PCT / 100)
    stop = entry_price * (1 - G_STOP_PCT / 100)
    peak = entry_price
    trail_stop = None
    max_bars = G_TIME_MIN // 2
    for i, (ts, row) in enumerate(bars_after.iterrows()):
        if i >= max_bars:
            return float(row["Close"]), "TIME"
        h, lo, c = float(row["High"]), float(row["Low"]), float(row["Close"])
        if h >= target:
            return target, "TARGET"
        if lo <= stop:
            return stop, "STOP"
        if h > peak:
            peak = h
        nt = peak * (1 - G_TRAIL_PCT / 100)
        if trail_stop is None or nt > trail_stop:
            trail_stop = nt
        if lo <= trail_stop:
            return trail_stop, "TRAIL"
    return float(bars_after.iloc[-1]["Close"]) if len(bars_after) else None, "EOD"


def find_any_green(mh, day_open):
    for i, (ts, row) in enumerate(mh.iterrows()):
        if float(row["Close"]) > day_open:
            return i, ts, float(row["Close"])
    return None, None, None


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

    # Capture G's holding window per (ticker, date) by simulating #511.
    # (key, val) = (ticker, date) -> list of (strategy, entry_time, exit_time)
    print("Running #511 to capture per-trade [entry,exit] windows...")
    cash = STARTING_CASH
    gl_windows = defaultdict(list)
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
                strat = st.get("strategy")
                ent = st.get("entry_time")
                exi = st.get("exit_time")
                tkr = st.get("ticker")
                if strat is not None and ent is not None and exi is not None:
                    gl_windows[(tkr, d)].append((strat, ent, exi))
                    gl_trade_count += 1
        cash = end_c + (unset if is_cash else 0)
    print(f"  G/L trades: {gl_trade_count}, (ticker,date) pairs: {len(gl_windows)}")
    n_g_windows = sum(1 for v in gl_windows.values() for s, _, _ in v if s == "G")
    print(f"  of which G-strategy windows: {n_g_windows}")

    # Bucket any_green_above_open trades
    pos_cost = STARTING_CASH * POSITION_PCT
    buckets = {"clean": [], "inside_g": [], "after_g_exit": [], "before_g": [],
               "inside_l": [], "after_l_exit": [], "before_l": []}

    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2:
                continue
            day_open = float(mh.iloc[0]["Open"])
            if day_open <= 0:
                continue
            idx, ent_ts, ep = find_any_green(mh, day_open)
            if idx is None or ep is None or ep <= 0:
                continue
            bars_after = mh.iloc[idx + 1:]
            if len(bars_after) == 0:
                continue
            exit_p, reason = _simulate_ro_trade(ep, bars_after)
            if exit_p is None:
                continue
            shares = pos_cost / ep
            pnl = shares * (exit_p - ep)
            trade = {"ticker": p["ticker"], "date": d, "entry_ts": ent_ts,
                     "pnl": pnl, "bar_idx": idx}

            wins = gl_windows.get((p["ticker"], d), [])
            g_wins = [(e, x) for s, e, x in wins if s == "G"]
            l_wins = [(e, x) for s, e, x in wins if s == "L"]

            def classify(strat_wins, prefix):
                if not strat_wins:
                    return None
                for e, x in strat_wins:
                    if e <= ent_ts <= x:
                        return f"inside_{prefix}"
                if all(ent_ts > x for e, x in strat_wins):
                    return f"after_{prefix}_exit"
                if all(ent_ts < e for e, x in strat_wins):
                    return f"before_{prefix}"
                return f"after_{prefix}_exit"

            g_class = classify(g_wins, "g")
            l_class = classify(l_wins, "l")

            if g_class:
                buckets[g_class].append(trade)
            if l_class:
                buckets[l_class].append(trade)
            if not g_class and not l_class:
                buckets["clean"].append(trade)

    def summarize(label, ts):
        if not ts:
            return f"  {label:<22} n=0"
        tot = sum(t["pnl"] for t in ts)
        wins = sum(t["pnl"] for t in ts if t["pnl"] > 0)
        losses = abs(sum(t["pnl"] for t in ts if t["pnl"] <= 0))
        wr = sum(1 for t in ts if t["pnl"] > 0) / len(ts) * 100
        pf = wins / losses if losses > 0 else 99.0
        return f"  {label:<22} n={len(ts):<5} pnl=${tot:>+11,.0f}  mean=${tot/len(ts):>+5,.0f}  WR={wr:>4.1f}%  PF={pf:.2f}"

    print()
    print("=" * 95)
    print("  any_green_above_open trades, bucketed by overlap with G / L #511 holding window")
    print("=" * 95)
    for k in ["clean", "inside_g", "after_g_exit", "before_g",
              "inside_l", "after_l_exit", "before_l"]:
        print(summarize(k, buckets[k]))

    total_any = sum(len(buckets[k]) for k in ["clean", "inside_g", "after_g_exit", "before_g"])
    inside_g = len(buckets["inside_g"])
    inside_l = len(buckets["inside_l"])
    print()
    print(f"Total any_green trades (G-bucketed):  {total_any}")
    print(f"  Bought INSIDE a G holding window:   {inside_g}  ({100*inside_g/total_any:.1f}%)")
    print(f"  Bought INSIDE an L holding window:  {inside_l}")

    out = "results/ro_any_green_in_g_window.json"
    os.makedirs("results", exist_ok=True)
    summary = {k: {"n": len(v),
                   "total_pnl": sum(t["pnl"] for t in v),
                   "mean_pnl": (sum(t["pnl"] for t in v) / len(v)) if v else 0,
                   "wr": (sum(1 for t in v if t["pnl"] > 0) / len(v) * 100) if v else 0}
               for k, v in buckets.items()}
    with open(out, "w") as f:
        json.dump({"total_any_g_bucket": total_any, "summary": summary}, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
