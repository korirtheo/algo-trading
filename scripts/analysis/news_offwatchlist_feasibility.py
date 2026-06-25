"""Feasibility check: can news catalysts catch trades on no-PM-watchlist days?

Two questions:
  1. How many trading days had ZERO G/L trades under #511 from the PM watchlist?
  2. Of those days, how many had news catalysts on tickers NOT in the watchlist?
  3. Of those news-catalyst tickers, how many do we have intraday data for?

The 3rd question is the data-availability constraint. We only have intraday
bars for tickers our PM scanner picked at some point in history. News-only
candidates outside that universe can't be backtested without broader data.
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


def main():
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD
    from news_filter import _load, count_pit

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
    print(f"Total trading days in data: {len(dates)}")

    # Build set of all tickers we have ANY intraday data for (any day in our dirs)
    tickers_in_data = set()
    for d in dates:
        for p in picks_by_date.get(d, []):
            tickers_in_data.add(p["ticker"])
    print(f"Unique tickers in our intraday data universe: {len(tickers_in_data)}")

    # Per-day: what was the PM watchlist (the picks for that day)?
    picks_by_ticker_per_day = {d: {p["ticker"] for p in picks_by_date.get(d, [])} for d in dates}

    # Run #511 backtest to find NO-TRADE days
    print()
    print("Running #511 backtest to find no-trade days...")
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
        had_trade = False
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                days_with_trades.add(d)
                had_trade = True
                break
        cash = end_c + (unset if is_cash else 0)
    no_trade_days = [d for d in dates if d not in days_with_trades]
    print(f"Days with at least one #511 trade: {len(days_with_trades)} / {len(dates)} ({100*len(days_with_trades)/len(dates):.1f}%)")
    print(f"No-trade days under #511: {len(no_trade_days)} ({100*len(no_trade_days)/len(dates):.1f}%)")

    # Load news cache
    nc = _load()
    print()
    print(f"News cache entries: {len(nc)} TICKER|DATE keys")

    # On no-trade days, check news_cache for tickers with catalyst, broken down by:
    #   A. Was the ticker in the PM watchlist that day? (already filtered out by G/L not firing)
    #   B. Or was it OUTSIDE the watchlist?
    #   C. Of outside-watchlist tickers, how many do we have intraday data for? (testable)
    outside_with_data = defaultdict(set)   # day -> set of outside-PM tickers with catalyst AND data
    outside_no_data = defaultdict(set)     # day -> tickers with news + outside watchlist but no data
    in_watchlist_catalyst = defaultdict(set)  # day -> tickers IN watchlist with catalyst (these were already considered by G/L)

    for d in no_trade_days:
        watchlist = picks_by_ticker_per_day.get(d, set())
        for key in nc:
            if not key.endswith(f"|{d}"):
                continue
            ticker = key.split("|")[0]
            n, has_cat = count_pit(ticker, d)
            if not has_cat:
                continue
            if ticker in watchlist:
                in_watchlist_catalyst[d].add(ticker)
            elif ticker in tickers_in_data:
                outside_with_data[d].add(ticker)
            else:
                outside_no_data[d].add(ticker)

    days_with_outside_data = [d for d in no_trade_days if outside_with_data[d]]
    days_with_outside_nodata = [d for d in no_trade_days if outside_no_data[d]]

    print()
    print("=" * 90)
    print(f"  FEASIBILITY: NEWS-CATALYST DISCOVERY ON NO-TRADE DAYS")
    print("=" * 90)
    print(f"  No-trade days under #511:               {len(no_trade_days)}")
    print(f"  No-trade days with IN-watchlist catalyst (already considered): {sum(1 for d in no_trade_days if in_watchlist_catalyst[d])}")
    print(f"  No-trade days with OUTSIDE-watchlist catalyst:")
    print(f"     ... AND we have intraday data (TESTABLE):  {len(days_with_outside_data)}")
    print(f"     ... but no intraday data (would need download): {len(days_with_outside_nodata)}")

    # Distribution of testable days
    if days_with_outside_data:
        sizes = [len(outside_with_data[d]) for d in days_with_outside_data]
        print(f"\n  Tickers per testable day:")
        print(f"     median: {sorted(sizes)[len(sizes)//2]}")
        print(f"     max:    {max(sizes)}")
        print(f"     total unique-ticker-day pairs to test: {sum(sizes)}")

        # Sample
        print(f"\n  Sample testable days:")
        for d in days_with_outside_data[:10]:
            tickers = sorted(outside_with_data[d])
            print(f"     {d}: {len(tickers)} tickers — {tickers[:8]}")

    out_path = "results/news_offwatchlist_feasibility.json"
    summary = {
        "total_trading_days": len(dates),
        "no_trade_days": len(no_trade_days),
        "testable_no_trade_days": len(days_with_outside_data),
        "testable_ticker_day_pairs": sum(len(outside_with_data[d]) for d in days_with_outside_data),
        "days_blocked_on_data": len(days_with_outside_nodata),
        "testable_days_sample": {d: sorted(outside_with_data[d])[:20] for d in days_with_outside_data[:20]},
    }
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
