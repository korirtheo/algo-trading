"""
Strategy Overlap Analysis: A, B, G, F
======================================
For each day in stored_data_combined, check which stocks qualify for each
strategy based on candle-1 + candle-2 filters:

  A: gap>20%, body>=2%, 2nd green + new hi, +3% target
  B: gap>10%, body>=8%, 2nd green (no new hi needed), +7% target
  G: gap>30%, 2nd green + new hi, +10% target
  F: gap>10%, 2nd green (no new hi needed), +7% target

Key questions:
  - How many days does A+G fire vs B vs F?
  - Incremental days from B and F over A+G?
  - Stock-level multi-strategy overlap?
"""

import os
import sys
import io
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from collections import defaultdict

# ── Strategy definitions ─────────────────────────────────────────────────────

STRATEGIES = {
    "A": {
        "min_gap_pct": 20.0,
        "min_body_pct": 2.0,
        "require_2nd_green": True,
        "require_2nd_new_high": True,
        "target_pct": 3.0,
        "desc": "gap>20%, body>=2%, 2nd green+new hi, +3%",
    },
    "B": {
        "min_gap_pct": 10.0,
        "min_body_pct": 8.0,
        "require_2nd_green": True,
        "require_2nd_new_high": False,
        "target_pct": 7.0,
        "desc": "gap>10%, body>=8%, 2nd green, +7%",
    },
    "G": {
        "min_gap_pct": 30.0,
        "min_body_pct": 0.0,  # no body filter for G
        "require_2nd_green": True,
        "require_2nd_new_high": True,
        "target_pct": 10.0,
        "desc": "gap>30%, 2nd green+new hi, +10%",
    },
    "F": {
        "min_gap_pct": 10.0,
        "min_body_pct": 0.0,  # no body filter for F
        "require_2nd_green": True,
        "require_2nd_new_high": False,
        "target_pct": 7.0,
        "desc": "gap>10%, 2nd green, +7%",
    },
}


def qualifies_for(strat, gap_pct, body_pct, second_green, second_new_high):
    """Check if a stock qualifies for a given strategy."""
    s = STRATEGIES[strat]
    if gap_pct < s["min_gap_pct"]:
        return False
    if body_pct < s["min_body_pct"]:
        return False
    if s["require_2nd_green"] and not second_green:
        return False
    if s["require_2nd_new_high"] and not second_new_high:
        return False
    return True


def analyze_day(picks):
    """For one day's picks, compute candle-1 body and candle-2 green/new-high,
    then check which strategies each stock qualifies for.

    Returns list of dicts: {ticker, gap_pct, body_pct, second_green,
                            second_new_high, strategies: set()}
    """
    results = []
    for pick in picks:
        mh = pick.get("market_hour_candles")
        if mh is None or len(mh) < 2:
            continue

        ticker = pick["ticker"]
        gap_pct = pick["gap_pct"]

        # Candle 1
        c1 = mh.iloc[0]
        c1_open = float(c1["Open"])
        c1_close = float(c1["Close"])
        c1_high = float(c1["High"])

        if c1_open <= 0:
            continue
        body_pct = (c1_close / c1_open - 1) * 100
        if body_pct < 0:
            # Candle 1 must be green (close > open) for body filter to make sense
            # But some strategies have min_body_pct=0 so a red candle with body<0
            # would still fail body>=2% etc. We keep the raw value.
            pass

        # Candle 2
        c2 = mh.iloc[1]
        c2_open = float(c2["Open"])
        c2_close = float(c2["Close"])
        c2_high = float(c2["High"])

        second_green = c2_close > c2_open
        second_new_high = c2_high > c1_high

        # Check each strategy
        strats = set()
        for strat_name in STRATEGIES:
            if qualifies_for(strat_name, gap_pct, body_pct, second_green, second_new_high):
                strats.add(strat_name)

        results.append({
            "ticker": ticker,
            "gap_pct": gap_pct,
            "body_pct": body_pct,
            "second_green": second_green,
            "second_new_high": second_new_high,
            "strategies": strats,
        })

    return results


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cache_path = os.path.join("stored_data_combined", "fulltest_picks_v3.pkl")

    print("Strategy Overlap Analysis")
    print("=" * 70)
    for name, s in STRATEGIES.items():
        print(f"  {name}: {s['desc']}")
    print("=" * 70)
    print()

    print(f"Loading cached picks from {cache_path} ...")
    with open(cache_path, "rb") as f:
        daily_picks = pickle.load(f)
    all_dates = sorted(daily_picks.keys())
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")
    print()

    # ── Per-day analysis ─────────────────────────────────────────────────

    # day -> set of strategies that fired (at least 1 qualifying stock)
    day_strategies = {}
    # day -> list of (ticker, strategies) tuples
    day_details = {}
    # Global counters
    total_qualifying_stocks = 0
    strat_stock_counts = defaultdict(int)  # strategy -> total qualifying stocks
    multi_strat_stocks = defaultdict(int)  # combo -> count

    for d in all_dates:
        picks = daily_picks.get(d, [])
        results = analyze_day(picks)

        fired = set()
        details = []
        for r in results:
            if r["strategies"]:
                fired.update(r["strategies"])
                details.append((r["ticker"], r["strategies"], r["gap_pct"], r["body_pct"],
                                r["second_green"], r["second_new_high"]))
                total_qualifying_stocks += 1
                for s in r["strategies"]:
                    strat_stock_counts[s] += 1
                # Track multi-strategy combos
                if len(r["strategies"]) > 1:
                    combo = "+".join(sorted(r["strategies"]))
                    multi_strat_stocks[combo] += 1

        day_strategies[d] = fired
        day_details[d] = details

    # ── Compute metrics ──────────────────────────────────────────────────

    total_days = len(all_dates)

    # Days where at least one stock qualifies for a given strategy
    days_with = {}
    for strat_name in STRATEGIES:
        days_with[strat_name] = sum(1 for d in all_dates if strat_name in day_strategies[d])

    # Combined sets
    ag_days = sum(1 for d in all_dates if day_strategies[d] & {"A", "G"})
    b_only_incr = sum(1 for d in all_dates
                      if "B" in day_strategies[d] and not (day_strategies[d] & {"A", "G"}))
    f_only_incr = sum(1 for d in all_dates
                      if "F" in day_strategies[d] and not (day_strategies[d] & {"A", "G"}))
    abg_days = sum(1 for d in all_dates if day_strategies[d] & {"A", "B", "G"})
    any_strat_days = sum(1 for d in all_dates if day_strategies[d])
    all_four_days = sum(1 for d in all_dates
                        if day_strategies[d] & {"A", "B", "G", "F"})
    no_signal_days = sum(1 for d in all_dates if not day_strategies[d])

    # ── Print results ────────────────────────────────────────────────────

    print("=" * 70)
    print("  DAY-LEVEL STRATEGY COVERAGE")
    print("=" * 70)
    print(f"  Total trading days:          {total_days}")
    print(f"  Days with ANY signal:        {any_strat_days}  ({any_strat_days/total_days*100:.1f}%)")
    print(f"  Days with NO signal:         {no_signal_days}  ({no_signal_days/total_days*100:.1f}%)")
    print()

    for strat_name in ["A", "B", "G", "F"]:
        ct = days_with[strat_name]
        print(f"  Days {strat_name} fires:             {ct:>4}  ({ct/total_days*100:.1f}%)")
    print()

    print(f"  Days A+G fires (either):     {ag_days:>4}  ({ag_days/total_days*100:.1f}%)")
    print(f"  Days A+B+G fires (any):      {abg_days:>4}  ({abg_days/total_days*100:.1f}%)")
    print(f"  Days A+B+G+F fires (any):    {all_four_days:>4}  ({all_four_days/total_days*100:.1f}%)")
    print()

    print(f"  Incremental B days")
    print(f"    (B fires but A+G don't):   {b_only_incr:>4}  ({b_only_incr/total_days*100:.1f}%)")
    print(f"  Incremental F days")
    print(f"    (F fires but A+G don't):   {f_only_incr:>4}  ({f_only_incr/total_days*100:.1f}%)")
    print()

    # How many extra days does adding B give over A+G alone?
    abg_vs_ag = abg_days - ag_days
    print(f"  Extra days from adding B to A+G:  {abg_vs_ag}  (A+B+G={abg_days} vs A+G={ag_days})")
    # How many extra days does adding F give over A+G alone?
    agf_days = sum(1 for d in all_dates if day_strategies[d] & {"A", "G", "F"})
    agf_vs_ag = agf_days - ag_days
    print(f"  Extra days from adding F to A+G:  {agf_vs_ag}  (A+G+F={agf_days} vs A+G={ag_days})")

    print()
    print("=" * 70)
    print("  STOCK-LEVEL COUNTS (total qualifying stock-signals across all days)")
    print("=" * 70)
    print(f"  Total qualifying stock-signals:  {total_qualifying_stocks}")
    for strat_name in ["A", "B", "G", "F"]:
        ct = strat_stock_counts[strat_name]
        print(f"    {strat_name}: {ct:>5} stocks")
    print()

    print("  Multi-strategy overlap (stocks qualifying for 2+ strategies):")
    if multi_strat_stocks:
        for combo, count in sorted(multi_strat_stocks.items(), key=lambda x: -x[1]):
            print(f"    {combo:<15} {count:>5} stocks")
    else:
        print("    (none)")

    # Single-strategy-only counts
    print()
    print("  Single-strategy-only stocks (qualify for exactly one):")
    single_counts = defaultdict(int)
    for d in all_dates:
        for ticker, strats, *_ in day_details[d]:
            if len(strats) == 1:
                single_counts[list(strats)[0]] += 1
    for strat_name in ["A", "B", "G", "F"]:
        ct = single_counts[strat_name]
        print(f"    {strat_name} only: {ct:>5}")

    # ── Detailed daily breakdown ─────────────────────────────────────────

    print()
    print("=" * 70)
    print("  DAILY DETAIL (days with at least one signal)")
    print("=" * 70)
    print(f"  {'Date':<12} {'Strats':>8} {'#Stocks':>8} {'Tickers'}")
    print("  " + "-" * 66)

    for d in all_dates:
        details = day_details[d]
        if not details:
            continue
        fired = day_strategies[d]
        strat_label = "+".join(sorted(fired))
        n_stocks = len(details)
        ticker_parts = []
        for ticker, strats, gap, body, g2, nh in details:
            tag = "+".join(sorted(strats))
            ticker_parts.append(f"{ticker}[{tag}]")
        ticker_str = ", ".join(ticker_parts)
        if len(ticker_str) > 80:
            ticker_str = ticker_str[:77] + "..."
        print(f"  {d:<12} {strat_label:>8} {n_stocks:>8}  {ticker_str}")

    # ── A vs G overlap deep-dive ─────────────────────────────────────────

    print()
    print("=" * 70)
    print("  A vs G OVERLAP (stocks qualifying for both A and G)")
    print("=" * 70)
    ag_overlap_count = 0
    a_only_count = 0
    g_only_count = 0
    for d in all_dates:
        for ticker, strats, gap, body, g2, nh in day_details[d]:
            has_a = "A" in strats
            has_g = "G" in strats
            if has_a and has_g:
                ag_overlap_count += 1
            elif has_a and not has_g:
                a_only_count += 1
            elif has_g and not has_a:
                g_only_count += 1
    print(f"  A+G overlap:  {ag_overlap_count}")
    print(f"  A only:       {a_only_count}")
    print(f"  G only:       {g_only_count}")
    print(f"  (G is subset of A when gap>30% and body>=2%)")

    # ── B vs F overlap deep-dive ─────────────────────────────────────────

    print()
    print("=" * 70)
    print("  B vs F OVERLAP (stocks qualifying for both B and F)")
    print("=" * 70)
    bf_overlap_count = 0
    b_only_count = 0
    f_only_count = 0
    for d in all_dates:
        for ticker, strats, gap, body, g2, nh in day_details[d]:
            has_b = "B" in strats
            has_f = "F" in strats
            if has_b and has_f:
                bf_overlap_count += 1
            elif has_b and not has_f:
                b_only_count += 1
            elif has_f and not has_b:
                f_only_count += 1
    print(f"  B+F overlap:  {bf_overlap_count}")
    print(f"  B only:       {b_only_count}")
    print(f"  F only:       {f_only_count}")
    print(f"  (B is subset of F since F has no body filter; every B stock also qualifies for F)")

    # ── Summary table ────────────────────────────────────────────────────

    print()
    print("=" * 70)
    print("  SUMMARY: INCREMENTAL VALUE TABLE")
    print("=" * 70)
    print(f"  {'Combination':<25} {'Days':>6} {'vs A+G':>10} {'Stocks':>10}")
    print("  " + "-" * 55)

    combos = [
        ("A only", {"A"}),
        ("G only", {"G"}),
        ("A+G", {"A", "G"}),
        ("B only", {"B"}),
        ("F only", {"F"}),
        ("A+B+G", {"A", "B", "G"}),
        ("A+G+F", {"A", "G", "F"}),
        ("A+B+G+F", {"A", "B", "G", "F"}),
    ]
    for label, strat_set in combos:
        combo_days = sum(1 for d in all_dates if day_strategies[d] & strat_set)
        combo_stocks = 0
        for d in all_dates:
            for ticker, strats, *_ in day_details[d]:
                if strats & strat_set:
                    combo_stocks += 1
        vs_ag = combo_days - ag_days
        vs_ag_str = f"+{vs_ag}" if vs_ag >= 0 else str(vs_ag)
        print(f"  {label:<25} {combo_days:>6} {vs_ag_str:>10} {combo_stocks:>10}")

    print()
    print("=" * 70)
    print("  DONE")
    print("=" * 70)
