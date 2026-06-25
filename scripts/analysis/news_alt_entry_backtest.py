"""Test news-based alternative entries on G/L no-trade days.

On the 194 days when #511 produces zero trades but a PM watchlist ticker has a
fresh news catalyst, can a different entry trigger capture profitable runs?

Tests two alternative entry logics on those days:

  Test A — NEWS + 1ST GREEN:
    Enter at close of the FIRST green candle (vs G's 2nd-green requirement)
    when ticker has news catalyst.

  Test B — NEWS + HOD BREAK:
    Enter when price breaks the premarket high after 9:30 ET when ticker has
    news catalyst. Skip the gap/green criteria entirely.

Both tests use #511's exit logic: target=62, stop=25, time=12, trail=0.5/act=0.
Sized at TRADE_PCT of $25K per trade (sized independently per trade — not
compounded — so we get the raw edge per day, not the path-dependent compound).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import pandas as pd
from collections import defaultdict

STARTING_CASH = 25_000
POSITION_PCT = 0.30           # match LIVE_MAX_POSITION_PCT_OF_CASH
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

# #511 exit params
G_TARGET_PCT = 62.0
G_STOP_PCT = 25.0
G_TIME_MIN = 12               # 6 two-min bars
G_TRAIL_PCT = 0.5
G_TRAIL_ACT_PCT = 0.0         # arm immediately


def _simulate_trade(entry_price, bars_after, entry_idx):
    """Apply #511 exit logic from entry_price across bars_after (DataFrame).
    Returns (exit_price, exit_reason, bars_held)."""
    if entry_price <= 0:
        return None, None, 0
    target = entry_price * (1 + G_TARGET_PCT / 100)
    stop   = entry_price * (1 - G_STOP_PCT / 100)
    peak   = entry_price
    trail_stop = None
    max_bars = G_TIME_MIN // 2   # 12 min / 2 min per bar = 6 bars

    for i, (ts, row) in enumerate(bars_after.iterrows()):
        if i >= max_bars:
            # Time stop — exit at this bar's close
            return float(row["Close"]), "TIME_STOP", i + 1
        c_high = float(row["High"]); c_low = float(row["Low"]); c_close = float(row["Close"])
        # Target check first
        if c_high >= target:
            return target, "TARGET", i + 1
        # Stop check
        if c_low <= stop:
            return stop, "STOP", i + 1
        # Trail (act=0, so always armed)
        if c_high > peak:
            peak = c_high
        new_trail = peak * (1 - G_TRAIL_PCT / 100)
        if trail_stop is None or new_trail > trail_stop:
            trail_stop = new_trail
        if c_low <= trail_stop:
            return trail_stop, "TRAIL", i + 1
    # Bars ran out — exit at last close
    return float(bars_after.iloc[-1]["Close"]) if len(bars_after) else None, "EOD", len(bars_after)


def _entry_first_green(market_hour_candles):
    """Return (entry_idx, entry_price) of close of first green candle. None if no green."""
    for i, (ts, row) in enumerate(market_hour_candles.iterrows()):
        if float(row["Close"]) > float(row["Open"]):
            return i, float(row["Close"])
    return None, None


def _entry_hod_break(market_hour_candles, premarket_high):
    """Return (entry_idx, entry_price) of first bar where high breaks PM high.
    Entry price = PM high (assumes we get filled at the break point, conservative)."""
    if premarket_high is None or premarket_high <= 0:
        return None, None
    for i, (ts, row) in enumerate(market_hour_candles.iterrows()):
        if float(row["High"]) > premarket_high:
            return i, premarket_high
    return None, None


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
    tgc.NEWS_MODULATOR_ENABLED = False

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    print(f"Loading {len(dirs)} dirs...")
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if DATE_LO <= d <= DATE_HI])
    print(f"Trading days: {len(dates)}")

    # Find #511 no-trade days
    print("Running #511 baseline to find no-trade days...")
    cash = STARTING_CASH
    days_with_trades = set()
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
                days_with_trades.add(d)
                break
        cash = end_c + (unset if is_cash else 0)
    no_trade_days = [d for d in dates if d not in days_with_trades]
    print(f"No-trade days: {len(no_trade_days)} / {len(dates)}")

    # Helper: for one day + ticker, get the bars + news status
    def _get_pick(d, ticker):
        for p in picks_by_date.get(d, []):
            if p["ticker"] == ticker:
                return p
        return None

    # Run both tests on no-trade days
    trades_a = []  # NEWS + 1ST GREEN
    trades_b = []  # NEWS + HOD BREAK
    for d in no_trade_days:
        for p in picks_by_date.get(d, []):
            ticker = p["ticker"]
            n, has_cat = count_pit(ticker, d)
            if not has_cat:
                continue
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2:
                continue
            pm_high = p.get("premarket_high")

            # --- Test A: first green entry ---
            idx_a, ep_a = _entry_first_green(mh)
            if idx_a is not None and ep_a > 0:
                bars_after = mh.iloc[idx_a + 1:]
                if len(bars_after) > 0:
                    exit_p, reason, bars_held = _simulate_trade(ep_a, bars_after, idx_a)
                    if exit_p is not None:
                        pos_cost = STARTING_CASH * POSITION_PCT
                        shares = pos_cost / ep_a
                        pnl = shares * (exit_p - ep_a)
                        trades_a.append({
                            "date": d, "ticker": ticker, "entry": ep_a, "exit": exit_p,
                            "pnl": pnl, "reason": reason, "bars_held": bars_held, "news_n": n,
                        })

            # --- Test B: HOD break entry ---
            idx_b, ep_b = _entry_hod_break(mh, pm_high)
            if idx_b is not None and ep_b > 0:
                bars_after = mh.iloc[idx_b + 1:]
                if len(bars_after) > 0:
                    exit_p, reason, bars_held = _simulate_trade(ep_b, bars_after, idx_b)
                    if exit_p is not None:
                        pos_cost = STARTING_CASH * POSITION_PCT
                        shares = pos_cost / ep_b
                        pnl = shares * (exit_p - ep_b)
                        trades_b.append({
                            "date": d, "ticker": ticker, "entry": ep_b, "exit": exit_p,
                            "pnl": pnl, "reason": reason, "bars_held": bars_held, "news_n": n,
                        })

    def _stats(trades, label):
        n = len(trades)
        if n == 0:
            return {"label": label, "n": 0}
        tot = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr = wins / n * 100
        mean = tot / n
        wsum = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        lsum = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        pf = wsum / lsum if lsum > 0 else 99.0
        reasons = defaultdict(int)
        for t in trades:
            reasons[t["reason"]] += 1
        return {
            "label": label, "n": n, "total_pnl": float(tot), "mean_pnl": float(mean),
            "wr_pct": float(wr), "pf": float(pf),
            "exit_breakdown": dict(reasons),
            "n_unique_days": len({t["date"] for t in trades}),
        }

    sa = _stats(trades_a, "NEWS + 1st green")
    sb = _stats(trades_b, "NEWS + HOD break")

    print()
    print("=" * 100)
    print(f"  NEWS-CATALYST ALTERNATIVE ENTRY ON #511 NO-TRADE DAYS")
    print(f"  Universe: {len(no_trade_days)} no-trade days, only watchlist tickers with has_catalyst=True")
    print(f"  Sized: 30% of $25K per trade, no compounding (raw per-trade edge)")
    print("=" * 100)
    for s in (sa, sb):
        if s["n"] == 0:
            print(f"\n  {s['label']}: 0 trades")
            continue
        print(f"\n  {s['label']}:")
        print(f"    Trades:           {s['n']} ({s['n_unique_days']} days)")
        print(f"    Total PnL:        ${s['total_pnl']:>+12,.0f}")
        print(f"    Mean PnL:         ${s['mean_pnl']:>+10,.0f}")
        print(f"    WR:               {s['wr_pct']:.1f}%")
        print(f"    PF:               {s['pf']:.2f}")
        print(f"    Exit breakdown:   {s['exit_breakdown']}")

    out_path = "results/news_alt_entry_backtest.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "no_trade_days": len(no_trade_days),
            "test_a_1st_green": sa,
            "test_b_hod_break": sb,
            "sample_a_top10_pnl": sorted(trades_a, key=lambda t: -t["pnl"])[:10],
            "sample_b_top10_pnl": sorted(trades_b, key=lambda t: -t["pnl"])[:10],
        }, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
