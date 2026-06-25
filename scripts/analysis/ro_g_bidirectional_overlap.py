"""Bidirectional G <-> any_green overlap, plus 'before_g' sub-bucket profiling.

Computes:
  1. How many any_green entries fall inside an active G window  (already 0 — confirmed)
  2. How many G entries fall inside an active any_green window  (NEW)
  3. Per-bar-idx + per-year breakdown of the 'before_g' bucket (the 1,141 high-WR trades)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
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


def _simulate_ro(entry_price, bars_after_with_idx):
    """Returns (exit_price, exit_ts, reason). bars_after_with_idx is iterable of (ts, row)."""
    if entry_price <= 0:
        return None, None, None
    target = entry_price * (1 + G_TARGET_PCT / 100)
    stop = entry_price * (1 - G_STOP_PCT / 100)
    peak = entry_price
    trail = None
    max_bars = G_TIME_MIN // 2
    last_ts, last_close = None, None
    for i, (ts, row) in enumerate(bars_after_with_idx):
        last_ts, last_close = ts, float(row["Close"])
        if i >= max_bars:
            return last_close, ts, "TIME"
        h, lo, c = float(row["High"]), float(row["Low"]), last_close
        if h >= target:
            return target, ts, "TARGET"
        if lo <= stop:
            return stop, ts, "STOP"
        if h > peak:
            peak = h
        nt = peak * (1 - G_TRAIL_PCT / 100)
        if trail is None or nt > trail:
            trail = nt
        if lo <= trail:
            return trail, ts, "TRAIL"
    return last_close, last_ts, "EOD"


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

    # Capture G/L per-(ticker,date) windows
    print("Running #511 to capture G/L holding windows...")
    cash = STARTING_CASH
    gl = defaultdict(list)
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
                s, e, x, t = st.get("strategy"), st.get("entry_time"), st.get("exit_time"), st.get("ticker")
                if s and e and x:
                    gl[(t, d)].append((s, e, x))
        cash = end_c + (unset if is_cash else 0)

    n_g = sum(1 for v in gl.values() for s, _, _ in v if s == "G")
    n_l = sum(1 for v in gl.values() for s, _, _ in v if s == "L")
    print(f"  G trades: {n_g}, L trades: {n_l}")

    # Build any_green entry+exit windows
    print("Computing any_green windows...")
    pos_cost = STARTING_CASH * POSITION_PCT
    ag_windows = defaultdict(list)  # (ticker,date) -> [(entry_ts, exit_ts, pnl, bar_idx)]

    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2:
                continue
            day_open = float(mh.iloc[0]["Open"])
            if day_open <= 0:
                continue
            idx, ent_ts, ep = find_any_green(mh, day_open)
            if idx is None or ep <= 0:
                continue
            bars_after = list(mh.iloc[idx + 1:].iterrows())
            if not bars_after:
                continue
            exit_p, exit_ts, reason = _simulate_ro(ep, bars_after)
            if exit_p is None:
                continue
            shares = pos_cost / ep
            pnl = shares * (exit_p - ep)
            ag_windows[(p["ticker"], d)].append((ent_ts, exit_ts, pnl, idx))

    n_ag = sum(len(v) for v in ag_windows.values())
    print(f"  any_green trades: {n_ag}")

    # Q1: G entries inside any_green windows
    g_inside_ag = 0
    g_inside_ag_pairs = []
    for (t, d), ag_trades in ag_windows.items():
        for s, e_g, x_g in gl.get((t, d), []):
            if s != "G":
                continue
            for ag_e, ag_x, _, _ in ag_trades:
                if ag_e <= e_g <= ag_x:
                    g_inside_ag += 1
                    g_inside_ag_pairs.append((t, d, str(ag_e), str(e_g), str(ag_x)))
                    break

    print()
    print("=" * 80)
    print(f"  G entries that fire INSIDE an active any_green window: {g_inside_ag} / {n_g}")
    print(f"    ({100*g_inside_ag/n_g:.1f}% of G trades)")
    print("=" * 80)

    # Q2: before_g sub-bucket — break out by bar_idx + year + WR/PF
    print()
    print("=" * 80)
    print("  'before_g' sub-bucket: any_green that fired EARLIER than G on same (ticker,date)")
    print("=" * 80)

    bg_trades = []
    for (t, d), ag_trades in ag_windows.items():
        g_wins = [(e, x) for s, e, x in gl.get((t, d), []) if s == "G"]
        if not g_wins:
            continue
        for ag_e, ag_x, pnl, bar_idx in ag_trades:
            if all(ag_e < e for e, _ in g_wins):
                bg_trades.append({"ticker": t, "date": d, "year": d[:4],
                                  "bar_idx": bar_idx, "pnl": pnl})

    def stats(ts, label):
        if not ts:
            return f"  {label:<24} n=0"
        tot = sum(t["pnl"] for t in ts)
        wins_v = sum(t["pnl"] for t in ts if t["pnl"] > 0)
        losses = abs(sum(t["pnl"] for t in ts if t["pnl"] <= 0))
        wr = sum(1 for t in ts if t["pnl"] > 0) / len(ts) * 100
        pf = wins_v / losses if losses > 0 else 99.0
        return f"  {label:<24} n={len(ts):<5} pnl=${tot:>+10,.0f}  mean=${tot/len(ts):>+5,.0f}  WR={wr:>4.1f}%  PF={pf:.2f}"

    print(stats(bg_trades, "TOTAL before_g"))
    print()
    print("  By bar_idx (entry bar):")
    bg_by_bar = defaultdict(list)
    for t in bg_trades:
        bg_by_bar[t["bar_idx"]].append(t)
    for bi in sorted(bg_by_bar.keys())[:8]:
        print(stats(bg_by_bar[bi], f"bar_idx={bi}"))
    print()
    print("  By year:")
    bg_by_year = defaultdict(list)
    for t in bg_trades:
        bg_by_year[t["year"]].append(t)
    for yr in sorted(bg_by_year.keys()):
        print(stats(bg_by_year[yr], yr))

    out = "results/ro_g_bidirectional_overlap.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "n_g_trades": n_g,
            "n_l_trades": n_l,
            "n_any_green_trades": n_ag,
            "g_inside_any_green": g_inside_ag,
            "before_g_total": len(bg_trades),
            "before_g_by_bar": {str(k): len(v) for k, v in bg_by_bar.items()},
            "before_g_by_year": {k: len(v) for k, v in bg_by_year.items()},
            "g_inside_ag_examples": g_inside_ag_pairs[:30],
        }, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
