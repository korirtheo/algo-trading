"""Test whether news filter rescues W5's losing 2026 performance.

Reuses the news classification approach. Tests #312 (W5 best) and #124
(baseline) on 2026 with the news filter applied.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import time
import re
from datetime import datetime, timezone, timedelta

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

CANDIDATES = [
    ("#124 W3 (deployed)",   "config/trial_124_microcap_pump_extracted.json"),
    ("#312 W5 (best train)", "config/trial_312_w5_extracted.json"),
]
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000
DATA_DIRS = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]

SCANNER_PATTERNS = [
    r"\d+ .*stocks moving", r"stocks moving",
    r"pre-market session", r"intraday session", r"after-market session",
    r"top gainers", r"top losers", r"morning gainers",
    r"premarket movers", r"midday movers"
]


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def get_trades(config_path, label):
    with open(config_path) as f: cfg = json.load(f)
    params_in = cfg.get("params", cfg)
    set_strategy_params(_merged(params_in))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates_2026 = sorted([d for d in all_dates if d.startswith("2026")])
    print(f"\n[{label}] Running backtest on {len(dates_2026)} days...")

    cash = STARTING_CASH
    trades = []
    for d in dates_2026:
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
                pnl = st["pnl"]; cost = st["position_cost"]
                trades.append({
                    "date": d, "ticker": st["ticker"], "pnl": pnl, "cost": cost,
                    "pct": pnl/cost*100 if cost > 0 else 0,
                    "strategy": st.get("strategy"),
                })
        cash = end_c + (unset if is_cash else 0)

    print(f"  Trades: {len(trades)}  Final equity: ${cash:,.0f}  PnL: ${cash-STARTING_CASH:+,.0f}")
    return trades, cash


def fetch_news_for_trades(client, trades, cache_path):
    """Fetch news for unique (ticker, date) pairs, with caching."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  Loaded cached news: {len(cache)} entries")
    else:
        cache = {}

    unique = set(f"{t['ticker']}|{t['date']}" for t in trades)
    fetched = 0
    for i, key in enumerate(sorted(unique)):
        if key in cache: continue
        ticker, d = key.split("|")
        start = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(days=1)
        end = start + timedelta(days=3)
        try:
            n = client.get_news(NewsRequest(symbols=ticker, start=start, end=end, limit=20))
            cache[key] = [h.headline for h in n.data.get("news", [])]
            fetched += 1
        except Exception as e:
            cache[key] = []
        time.sleep(0.05)
        if fetched > 0 and fetched % 10 == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"  Fetched {fetched} new entries, cache total: {len(cache)}")
    return cache


def classify_trade(trade, cache):
    """Return news features for a trade."""
    key = f"{trade['ticker']}|{trade['date']}"
    headlines = cache.get(key, [])
    n = len(headlines)
    n_scanner = sum(1 for h in headlines
                    if any(re.search(p, h.lower()) for p in SCANNER_PATTERNS))
    only_scanner = (n > 0 and n_scanner == n)
    has_catalyst = n > n_scanner
    return n, only_scanner, has_catalyst


def apply_filter(trades, cache, filter_name):
    """Apply a news filter and return new PnL."""
    cash = STARTING_CASH
    eq = [cash]
    kept_trades = []
    for t in trades:
        n, only_scanner, has_catalyst = classify_trade(t, cache)
        keep = True
        if filter_name == "skip_<2_articles_or_only_scanner":
            if n < 2 or only_scanner: keep = False
        elif filter_name == "skip_no_specific_catalyst":
            if not has_catalyst: keep = False
        elif filter_name == "skip_0_articles":
            if n == 0: keep = False
        elif filter_name == "skip_only_scanner":
            if only_scanner: keep = False
        if keep:
            cash += t["pnl"]
            kept_trades.append(t)
        eq.append(cash)
    return cash, kept_trades, eq


def main():
    os.makedirs("results/news_filter_w5", exist_ok=True)

    client = NewsClient(
        "PKIPXFIETM7H4BAGQ64FQV3IWJ",
        "25RY682kuN9EBcFr6SFxgbSpFjdZkn613PLKPy1TdYzG"
    )

    results = {}
    for label, path in CANDIDATES:
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        trades, baseline_equity = get_trades(path, label)

        # Cache news per candidate
        cache_path = f"results/news_filter_w5/news_cache_{os.path.basename(path).replace('.json','')}.json"
        print(f"\n[{label}] Fetching news...")
        cache = fetch_news_for_trades(client, trades, cache_path)

        # Apply filters
        results[label] = {
            "baseline_pnl": baseline_equity - STARTING_CASH,
            "baseline_equity": baseline_equity,
            "n_trades": len(trades),
        }
        print(f"\n[{label}] Filter results:")
        for fname in ["skip_0_articles", "skip_only_scanner",
                       "skip_<2_articles_or_only_scanner", "skip_no_specific_catalyst"]:
            new_eq, kept, _ = apply_filter(trades, cache, fname)
            cut = len(trades) - len(kept)
            wr = sum(1 for t in kept if t["pnl"]>0)/len(kept)*100 if kept else 0
            print(f"  {fname:<45} cut={cut:>3}  new equity=${new_eq:,.0f}  "
                  f"lift=${new_eq-baseline_equity:+,.0f}  WR={wr:.1f}%")
            results[label][fname] = new_eq

    # Final summary
    print(f"\n\n{'='*88}")
    print(f"SUMMARY — does news filter rescue W5?")
    print(f"{'='*88}")
    print(f"  {'config':<24} {'baseline':>12} {'best filter':>14} {'lift':>12} {'multiplier change':>20}")
    for label, r in results.items():
        best_key = max((k for k in r if k.startswith('skip')), key=lambda k: r[k])
        baseline = r['baseline_equity']
        best = r[best_key]
        b_mult = baseline / STARTING_CASH
        be_mult = best / STARTING_CASH
        print(f"  {label:<24} ${baseline:>11,.0f} ${best:>13,.0f} ${best-baseline:>+11,.0f} "
              f"{b_mult:.2f}x -> {be_mult:.2f}x")
        print(f"  {'':<24}    best filter: {best_key}")


if __name__ == "__main__":
    main()
