"""G-news-catalyst filter backtest.

Compares:
  - Baseline #511 (no news filter)
  - Universal catalyst filter (drops both G and L tickers without news_catalyst)
  - G-only catalyst filter (drops G entries without catalyst, keeps L unfiltered)

The G-only filter is implemented by monkey-patching _classify_candle2 to return
None for "G" classification when the day's pick has no catalyst — that way
fallback strategies (D, V, M, P) still get a chance to fire on the same ticker,
and L is unaffected.

Test window: 2022 + 2024 + 2025 (full, on each window with fresh $25K cash).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

WINDOWS = [
    {"name": "2022", "dirs": ["stored_data_2022"], "date_lo": "2022-01-01", "date_hi": "2022-12-31"},
    {"name": "2024", "dirs": ["stored_data_combined", "stored_data_jan_mar_2024",
                              "stored_data_apr_jun_2024", "stored_data_jul_sep_2024",
                              "stored_data_oct_dec_2024"],
     "date_lo": "2024-01-01", "date_hi": "2024-12-31"},
    {"name": "2025", "dirs": ["stored_data_combined", "stored_data_jan_mar_2025",
                              "stored_data_apr_jun_2025", "stored_data_jul_2025",
                              "stored_data_oos"],
     "date_lo": "2025-01-01", "date_hi": "2025-12-31"},
    {"name": "2026_Mar_Jun_OOS",
     "dirs": ["stored_data", "stored_data_mar_may_2026",
              "stored_data_jun_2026", "stored_data_2026_gap_fill"],
     "date_lo": "2026-03-01", "date_hi": "2026-06-24"},
]


def _setup_511():
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
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


def run_one_window(window_name, dirs, date_lo, date_hi, mode):
    """mode in {'baseline', 'all_catalyst', 'g_only_catalyst'}."""
    import test_green_candle_combined as tgc
    from test_full import load_all_picks, MARGIN_THRESHOLD
    from news_filter import count_pit

    dirs = [d for d in dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])

    # --- G-only filter: monkey-patch _classify_candle2 to skip G without catalyst ---
    orig_classify = tgc._classify_candle2
    catalyst_lookup = {}  # (ticker, date) -> bool

    if mode == "g_only_catalyst":
        # Pre-warm catalyst lookup for all picks
        for d in dates:
            for p in picks_by_date.get(d, []):
                key = (p["ticker"], d)
                if key not in catalyst_lookup:
                    n, has_cat = count_pit(p["ticker"], d)
                    catalyst_lookup[key] = has_cat

        # Build a context dict the patched function can consult
        _ctx = {"current_ticker": None, "current_date": None}

        def _patched_classify(gap_pct, body_pct, second_green, second_new_high, vol_confirm=False, params=None):
            strat = orig_classify(gap_pct, body_pct, second_green, second_new_high, vol_confirm, params)
            if strat == "G":
                key = (_ctx["current_ticker"], _ctx["current_date"])
                if not catalyst_lookup.get(key, False):
                    return None  # block G entry, allow fallback strats to take over
            return strat
        tgc._classify_candle2 = _patched_classify
    else:
        _ctx = None

    cash = STARTING_CASH
    trades = []
    try:
        for d in dates:
            day_picks = picks_by_date.get(d, [])
            if not day_picks:
                continue

            if mode == "all_catalyst":
                # Drop the entire pick if no catalyst
                day_picks = [p for p in day_picks if catalyst_lookup_or_compute(p["ticker"], d)]
            elif mode == "g_only_catalyst" and _ctx is not None:
                # Each pick needs to be processed with context — we can't pass it
                # in directly because the simulator iterates internally. Set context
                # per-pick by wrapping. Simplest: tag the pick's date and ticker on
                # entry, and have _patched_classify look it up. We thread context via
                # a small wrapper here that sets _ctx before each pick is processed.
                pass

            is_cash = cash < MARGIN_THRESHOLD
            if mode == "g_only_catalyst":
                # Run pick-by-pick so we can set _ctx before each strategy classifies
                day_states = []
                for p in day_picks:
                    _ctx["current_ticker"] = p["ticker"]
                    _ctx["current_date"] = d
                    try:
                        sub_states, end_c, unset, _ = tgc.simulate_day_combined(
                            [p], cash, cash_account=is_cash
                        )
                    except Exception:
                        continue
                    day_states.extend(sub_states)
                    cash = end_c + (unset if is_cash else 0)
                states = day_states
            else:
                try:
                    states, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
                except Exception:
                    continue
                cash = end_c + (unset if is_cash else 0)

            for st in states:
                if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                    trades.append({
                        "date": d, "ticker": st.get("ticker"),
                        "strategy": st.get("strategy"), "pnl": st["pnl"],
                        "exit_reason": st.get("exit_reason"),
                    })
    finally:
        tgc._classify_candle2 = orig_classify

    return {"trades": trades, "final_cash": cash}


def catalyst_lookup_or_compute(ticker, date_str):
    from news_filter import count_pit
    _, has_cat = count_pit(ticker, date_str)
    return has_cat


def summarize(trades, label):
    n = len(trades)
    if n == 0:
        return {"label": label, "n": 0}
    tot = sum(t["pnl"] for t in trades)
    g_trades = [t for t in trades if t["strategy"] == "G"]
    l_trades = [t for t in trades if t["strategy"] == "L"]
    other = [t for t in trades if t["strategy"] not in ("G", "L")]
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / n * 100
    return {
        "label": label, "n": n, "total_pnl": tot, "wr_pct": wr,
        "g_n": len(g_trades), "g_pnl": sum(t["pnl"] for t in g_trades),
        "g_wr": (sum(1 for t in g_trades if t["pnl"] > 0) / len(g_trades) * 100) if g_trades else 0,
        "l_n": len(l_trades), "l_pnl": sum(t["pnl"] for t in l_trades),
        "l_wr": (sum(1 for t in l_trades if t["pnl"] > 0) / len(l_trades) * 100) if l_trades else 0,
        "other_n": len(other), "other_pnl": sum(t["pnl"] for t in other),
    }


def main():
    _setup_511()

    rows = []  # (window, mode, result)
    for w in WINDOWS:
        for mode in ["baseline", "all_catalyst", "g_only_catalyst"]:
            print(f"Running {w['name']} / {mode}...")
            r = run_one_window(w["name"], w["dirs"], w["date_lo"], w["date_hi"], mode)
            s = summarize(r["trades"], f"{w['name']}/{mode}")
            s["final_cash"] = r["final_cash"]
            rows.append(s)

    print()
    print("=" * 110)
    print(f"  G-NEWS-CATALYST FILTER BACKTEST  (params: #511 deployed)")
    print("=" * 110)
    print(f"{'window':<14} {'mode':<22} {'n':>5} {'WR':>5} {'TOTAL':>13} {'G_n':>4} {'G_pnl':>11} {'G_WR':>5} {'L_n':>4} {'L_pnl':>11}")
    print("-" * 110)
    for r in rows:
        win, mode = r["label"].split("/")
        print(f"{win:<14} {mode:<22} {r['n']:>5} {r.get('wr_pct',0):>4.1f}% ${r.get('total_pnl',0):>+11,.0f} "
              f"{r.get('g_n',0):>4} ${r.get('g_pnl',0):>+9,.0f} {r.get('g_wr',0):>4.1f}% "
              f"{r.get('l_n',0):>4} ${r.get('l_pnl',0):>+9,.0f}")

    # Cross-window summary: per mode, total
    print()
    print("Cross-window total PnL by mode:")
    by_mode = defaultdict(float)
    for r in rows:
        win, mode = r["label"].split("/")
        by_mode[mode] += r.get("total_pnl", 0)
    for mode in ["baseline", "all_catalyst", "g_only_catalyst"]:
        delta = by_mode[mode] - by_mode["baseline"] if mode != "baseline" else 0
        print(f"  {mode:<22} ${by_mode[mode]:>+13,.0f}  ({delta:+,.0f} vs baseline)")

    out_path = "results/g_news_filter_backtest.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "by_mode_total": dict(by_mode)}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
