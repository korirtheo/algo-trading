"""News-vs-trade-outcome correlation analysis for Jan-Feb 2026.

For each ticker #124 traded in Jan-Feb 2026:
  1. Run the simulator to get the trades (entry date, PnL)
  2. Fetch Alpaca news for that ticker in a +/- 1 day window around the trade
  3. Classify news (scanner roundup vs specific catalyst, positive vs negative)
  4. Correlate news patterns with trade PnL

Output:
  - results/news_analysis/jan_feb_2026_trades_with_news.csv
  - Console summary of patterns
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import time
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

# Categorization keywords (rough but useful)
SCANNER_ROUNDUP_PATTERNS = [
    r"\d+ .*stocks moving",
    r"stocks moving",
    r"pre-market session",
    r"intraday session",
    r"after-hours session",
    r"top gainers",
    r"top losers",
    r"morning gainers",
    r"premarket movers",
]

POSITIVE_CATALYST_KEYWORDS = [
    "announces", "partnership", "contract", "approval",
    "fda", "acquisition", "merger", "upgrade", "raises",
    "exceeds", "beats", "strong", "buy", "bullish",
    "agreement", "launches", "expands", "earnings beat",
    "positive", "milestone", "wins", "awards",
]

NEGATIVE_CATALYST_KEYWORDS = [
    "dilution", "offering", "at-the-market", "atm program",
    "downgrade", "lowers", "misses", "weak", "bearish",
    "lawsuit", "fraud", "investigation", "delisting",
    "going concern", "bankruptcy", "warning", "sell",
    "reverse split", "fire-sale", "default",
]


def classify_headline(headline):
    h = headline.lower()
    # Scanner roundup?
    for pat in SCANNER_ROUNDUP_PATTERNS:
        if re.search(pat, h):
            return "scanner_roundup"
    pos = sum(1 for k in POSITIVE_CATALYST_KEYWORDS if k in h)
    neg = sum(1 for k in NEGATIVE_CATALYST_KEYWORDS if k in h)
    if neg > pos:
        return "catalyst_negative"
    elif pos > 0:
        return "catalyst_positive"
    return "catalyst_neutral"


def main():
    os.makedirs("results/news_analysis", exist_ok=True)

    # Step 1: get Jan-Feb 2026 #124 trades
    with open("config/trial_124_microcap_pump_extracted.json") as f: cfg = json.load(f)
    with open("config/trial_432_params.json") as f: base = json.load(f)
    merged = dict(base); merged.update(cfg["params"])
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    dirs = [d for d in ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"] if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    jan_feb_dates = sorted([d for d in all_dates if d.startswith("2026")])
    print(f"Backtesting ALL of 2026: {len(jan_feb_dates)} trading days")

    cash = 25_000
    trades = []
    for d in jan_feb_dates:
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
                pnl = st["pnl"]
                cost = st["position_cost"]
                trades.append({
                    "date": d, "ticker": st["ticker"],
                    "strategy": st.get("strategy"),
                    "pnl": pnl, "pct": pnl/cost*100 if cost > 0 else 0,
                    "cost": cost, "exit_reason": st.get("exit_reason"),
                })
        cash = end_c + (unset if is_cash else 0)

    print(f"Trades in Jan-Feb 2026: {len(trades)}")
    print(f"  Winners: {sum(1 for t in trades if t['pnl']>0)}  ({sum(1 for t in trades if t['pnl']>0)/len(trades)*100:.1f}%)")
    print(f"  Total PnL: ${sum(t['pnl'] for t in trades):+,.0f}")

    # Step 2: fetch news for each unique (ticker, date)
    unique_trades = set((t["date"], t["ticker"]) for t in trades)
    print(f"\nFetching news for {len(unique_trades)} unique (ticker, date) pairs...")

    c = NewsClient("PKIPXFIETM7H4BAGQ64FQV3IWJ", "25RY682kuN9EBcFr6SFxgbSpFjdZkn613PLKPy1TdYzG")
    news_cache = {}  # (date, ticker) -> list of articles
    for i, (d, ticker) in enumerate(sorted(unique_trades)):
        print(f"  [{i+1}/{len(unique_trades)}] {ticker} on {d}...", end="\r")
        start = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(days=1)
        end = start + timedelta(days=3)
        try:
            n = c.get_news(NewsRequest(symbols=ticker, start=start, end=end, limit=20))
            arts = n.data.get("news", [])
            news_cache[(d, ticker)] = [(a.created_at, a.headline, a.source) for a in arts]
        except Exception as e:
            news_cache[(d, ticker)] = []
        time.sleep(0.1)  # be polite
    print()

    # Step 3: classify and correlate
    for t in trades:
        key = (t["date"], t["ticker"])
        arts = news_cache.get(key, [])
        classes = [classify_headline(h) for _, h, _ in arts]
        t["n_articles"] = len(arts)
        t["n_scanner"] = sum(1 for c in classes if c == "scanner_roundup")
        t["n_positive"] = sum(1 for c in classes if c == "catalyst_positive")
        t["n_negative"] = sum(1 for c in classes if c == "catalyst_negative")
        t["n_neutral"] = sum(1 for c in classes if c == "catalyst_neutral")
        t["has_specific_catalyst"] = (t["n_positive"] + t["n_negative"] + t["n_neutral"]) > 0
        t["has_negative"] = t["n_negative"] > 0
        t["has_positive"] = t["n_positive"] > 0
        t["only_scanner"] = (t["n_articles"] > 0 and t["n_scanner"] == t["n_articles"])

    # Save CSV
    import csv
    csv_path = "results/news_analysis/jan_feb_2026_trades_with_news.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trades[0].keys())
        w.writeheader()
        w.writerows(trades)
    print(f"Wrote {csv_path}")

    # Step 4: correlate patterns
    print(f"\n{'='*72}")
    print("NEWS-VS-OUTCOME PATTERNS (Jan-Feb 2026 backtest)")
    print(f"{'='*72}")

    # Pattern 1: trades WITH any news vs WITHOUT
    with_news = [t for t in trades if t["n_articles"] > 0]
    without_news = [t for t in trades if t["n_articles"] == 0]
    print(f"\n1. Trades with ANY news vs no news:")
    for label, group in [("with news", with_news), ("no news", without_news)]:
        if not group: continue
        wr = sum(1 for t in group if t["pnl"]>0) / len(group) * 100
        avg = sum(t["pct"] for t in group) / len(group)
        print(f"  {label:<14} n={len(group):>4}  WR={wr:>5.1f}%  avg PnL%={avg:>+5.2f}")

    # Pattern 2: scanner-only vs specific catalyst
    only_scanner = [t for t in trades if t["only_scanner"]]
    has_catalyst = [t for t in trades if t["has_specific_catalyst"]]
    print(f"\n2. Only scanner-roundup news vs specific catalyst:")
    for label, group in [("only scanner", only_scanner), ("has catalyst", has_catalyst)]:
        if not group: continue
        wr = sum(1 for t in group if t["pnl"]>0) / len(group) * 100
        avg = sum(t["pct"] for t in group) / len(group)
        print(f"  {label:<14} n={len(group):>4}  WR={wr:>5.1f}%  avg PnL%={avg:>+5.2f}")

    # Pattern 3: positive vs negative news
    has_pos = [t for t in trades if t["has_positive"] and not t["has_negative"]]
    has_neg = [t for t in trades if t["has_negative"] and not t["has_positive"]]
    has_both = [t for t in trades if t["has_positive"] and t["has_negative"]]
    print(f"\n3. Positive-only / Negative-only / Mixed catalyst news:")
    for label, group in [("positive only", has_pos), ("negative only", has_neg), ("mixed", has_both)]:
        if not group: continue
        wr = sum(1 for t in group if t["pnl"]>0) / len(group) * 100
        avg = sum(t["pct"] for t in group) / len(group)
        print(f"  {label:<14} n={len(group):>4}  WR={wr:>5.1f}%  avg PnL%={avg:>+5.2f}")

    # Pattern 4: by # articles (news volume)
    print(f"\n4. By news volume (# articles in window):")
    buckets = [("0", 0, 0), ("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11+", 11, 999)]
    for label, lo, hi in buckets:
        group = [t for t in trades if lo <= t["n_articles"] <= hi]
        if len(group) < 5: continue
        wr = sum(1 for t in group if t["pnl"]>0) / len(group) * 100
        avg = sum(t["pct"] for t in group) / len(group)
        print(f"  {label:<6} n={len(group):>4}  WR={wr:>5.1f}%  avg PnL%={avg:>+5.2f}")

    # Pattern 5: most common winning vs losing headlines
    win_classes = Counter()
    lose_classes = Counter()
    for t in trades:
        if t["pnl"] > 0:
            win_classes["scanner"] += t["n_scanner"]
            win_classes["positive"] += t["n_positive"]
            win_classes["negative"] += t["n_negative"]
            win_classes["neutral"] += t["n_neutral"]
        elif t["pnl"] < 0:
            lose_classes["scanner"] += t["n_scanner"]
            lose_classes["positive"] += t["n_positive"]
            lose_classes["negative"] += t["n_negative"]
            lose_classes["neutral"] += t["n_neutral"]

    print(f"\n5. Headline class distribution across WIN vs LOSS trades:")
    print(f"  {'class':<12} {'win$':>8} {'loss$':>8}")
    for cls in ("scanner", "positive", "neutral", "negative"):
        print(f"  {cls:<12} {win_classes[cls]:>8} {lose_classes[cls]:>8}")

    # === DEEP DIVE: which news TYPES are actual signal ===
    print(f"\n{'='*72}")
    print("SIGNAL STRENGTH BY NEWS CATEGORY (keyword-tagged)")
    print(f"{'='*72}")

    # Build per-trade keyword tag matrix
    KEYWORD_TAGS = {
        "earnings":      ["earnings", "beats", "misses", "q1", "q2", "q3", "q4", "fiscal"],
        "fda":           ["fda", "phase", "trial", "approval", "clinical"],
        "merger_acq":    ["acquisition", "merger", "acquires", "buyout", "takeover"],
        "dilution":      ["offering", "at-the-market", "atm program", "public offering", "warrants", "reverse split"],
        "upgrade":       ["upgrade", "raises", "buy rating", "price target raise", "outperform"],
        "downgrade":     ["downgrade", "lowers", "sell rating", "underperform"],
        "partnership":   ["partnership", "agreement", "joint venture", "collaboration"],
        "contract_win":  ["contract", "awarded", "wins"],
        "insider":       ["13d", "13g", "insider", "stake", "purchase"],
        "lawsuit":       ["lawsuit", "settled", "settlement", "complaint", "fraud", "investigation"],
        "guidance":      ["guidance", "outlook", "forecast", "raises guidance"],
        "movers_premarket": ["pre-market session", "premarket movers", "morning gainers", "gappers"],
        "movers_intraday":  ["intraday session", "intraday movers", "midday movers"],
        "movers_after":     ["after-market session", "after-hours session", "afterhours"],
        "stock_surge":   ["surge", "surging", "soars", "skyrockets", "rallies"],
        "stock_crash":   ["plunge", "tumbles", "falls", "drops", "crashes"],
        "analyst_q":     ["why is", "what's behind", "what's driving"],
        "tech_growth":   ["artificial intelligence", "ai ", "blockchain", "crypto", "quantum"],
    }

    # Tag each trade with which keywords appeared
    for t in trades:
        key = (t["date"], t["ticker"])
        arts = news_cache.get(key, [])
        tags = set()
        for _, hl, _ in arts:
            h = hl.lower()
            for tag, kws in KEYWORD_TAGS.items():
                if any(k in h for k in kws):
                    tags.add(tag)
        t["tags"] = tags

    # For each tag, compute WR and avg PnL on trades that HAVE that tag vs not
    print(f"  {'tag':<20} {'present':>14} {'absent':>14}  {'lift':>6}")
    print(f"  {'':<20} {'n  WR%  avg%':<14} {'n  WR%  avg%':<14}  {'(WR pp)':>6}")
    results_by_tag = []
    for tag in KEYWORD_TAGS.keys():
        with_tag = [t for t in trades if tag in t["tags"]]
        without_tag = [t for t in trades if tag not in t["tags"]]
        if len(with_tag) < 5 or len(without_tag) < 5:
            continue
        w_wr = sum(1 for t in with_tag if t["pnl"]>0) / len(with_tag) * 100
        wo_wr = sum(1 for t in without_tag if t["pnl"]>0) / len(without_tag) * 100
        w_avg = sum(t["pct"] for t in with_tag) / len(with_tag)
        wo_avg = sum(t["pct"] for t in without_tag) / len(without_tag)
        lift = w_wr - wo_wr
        results_by_tag.append((tag, len(with_tag), w_wr, w_avg, len(without_tag), wo_wr, wo_avg, lift))
        print(f"  {tag:<20} {len(with_tag):>3}  {w_wr:>4.1f}%  {w_avg:>+5.2f}%   "
              f"{len(without_tag):>3}  {wo_wr:>4.1f}%  {wo_avg:>+5.2f}%   {lift:>+5.1f}pp")

    # Sort by lift to find the most predictive tags
    print(f"\n  TOP PREDICTIVE TAGS (by WR lift):")
    results_by_tag.sort(key=lambda r: -r[7])
    for r in results_by_tag[:7]:
        tag, n_w, w_wr, w_avg, n_wo, wo_wr, wo_avg, lift = r
        marker = "+" if lift > 0 else "-"
        print(f"    {marker} {tag:<20} present in {n_w:>3} trades -> WR {w_wr:.1f}% vs baseline {wo_wr:.1f}%  (+{lift:.1f}pp)")
    print(f"\n  BOTTOM TAGS (negative signal):")
    for r in results_by_tag[-5:]:
        tag, n_w, w_wr, w_avg, n_wo, wo_wr, wo_avg, lift = r
        print(f"    - {tag:<20} present in {n_w:>3} trades -> WR {w_wr:.1f}% vs baseline {wo_wr:.1f}%  ({lift:+.1f}pp)")

    # Show a few specific examples
    print(f"\n{'='*72}")
    print("EXAMPLES (3 biggest winners, 3 biggest losers)")
    print(f"{'='*72}")
    trades_sorted = sorted(trades, key=lambda t: t["pnl"])
    losers = trades_sorted[:3]
    winners = trades_sorted[-3:]

    print(f"\nWORST 3 LOSERS:")
    for t in losers:
        key = (t["date"], t["ticker"])
        arts = news_cache.get(key, [])
        print(f"\n  {t['date']} {t['ticker']} ({t['strategy']}) ${t['pnl']:+,.0f} ({t['pct']:+.1f}%) {t['exit_reason']}")
        for ts, hl, src in arts[:3]:
            print(f"    [{src}] {hl[:90]}")
        if not arts:
            print(f"    (no news)")

    print(f"\nBEST 3 WINNERS:")
    for t in winners:
        key = (t["date"], t["ticker"])
        arts = news_cache.get(key, [])
        print(f"\n  {t['date']} {t['ticker']} ({t['strategy']}) ${t['pnl']:+,.0f} ({t['pct']:+.1f}%) {t['exit_reason']}")
        for ts, hl, src in arts[:3]:
            print(f"    [{src}] {hl[:90]}")
        if not arts:
            print(f"    (no news)")


if __name__ == "__main__":
    main()
