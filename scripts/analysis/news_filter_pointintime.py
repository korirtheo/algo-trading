"""News filter with POINT-IN-TIME correctness.

Fixes the look-ahead bias by only counting news articles published BEFORE
the trade entry time. For simplicity, we use 9:30 AM ET on the trade date
as the cutoff (most #124 trades fire 9:30-11 AM ET; this is a tight
conservative bound).

Compares:
  Filter approach #1 (BUGGY):  ±1 day window — counts post-trade articles
  Filter approach #2 (CORRECT): cutoff at 9:30 AM ET on trade date
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import time
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

ET = ZoneInfo("America/New_York")
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
    set_strategy_params(_merged(cfg.get("params", cfg)))
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

    cash = STARTING_CASH
    trades = []
    for d in dates_2026:
        day_picks = picks_by_date.get(d, [])
        if not day_picks: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception: continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                trades.append({
                    "date": d, "ticker": st["ticker"], "pnl": st["pnl"],
                    "cost": st["position_cost"],
                    "pct": st["pnl"]/st["position_cost"]*100 if st["position_cost"]>0 else 0,
                    "strategy": st.get("strategy"),
                })
        cash = end_c + (unset if is_cash else 0)
    return trades, cash


def fetch_news_with_timestamps(client, trades, cache_path):
    """Fetch news INCLUDING timestamps for point-in-time filtering."""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    unique = set(f"{t['ticker']}|{t['date']}" for t in trades)
    fetched = 0
    for key in sorted(unique):
        if key in cache: continue
        ticker, d = key.split("|")
        start = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(days=2)
        end = start + timedelta(days=4)
        try:
            n = client.get_news(NewsRequest(symbols=ticker, start=start, end=end, limit=30))
            arts = n.data.get("news", [])
            cache[key] = [
                {"ts": h.created_at.isoformat(), "headline": h.headline}
                for h in arts
            ]
            fetched += 1
        except Exception:
            cache[key] = []
        time.sleep(0.05)
        if fetched and fetched % 20 == 0:
            json.dump(cache, open(cache_path, "w"))
    json.dump(cache, open(cache_path, "w"))
    return cache


def is_scanner(headline):
    h = headline.lower()
    return any(re.search(p, h) for p in SCANNER_PATTERNS)


def count_news(trade, cache, cutoff_mode):
    """Count news using either look-ahead (BUGGY) or point-in-time (CORRECT)."""
    key = f"{trade['ticker']}|{trade['date']}"
    arts = cache.get(key, [])

    if cutoff_mode == "lookahead":
        # OLD: any article in ±1 day window (includes post-trade)
        valid = arts
    elif cutoff_mode == "pit":
        # POINT-IN-TIME: only articles published before 9:30 AM ET on trade date
        trade_date_930_et = ET.localize(
            datetime.strptime(trade['date'] + ' 09:30:00', '%Y-%m-%d %H:%M:%S')
        ) if hasattr(ET, 'localize') else datetime.strptime(
            trade['date'] + ' 09:30:00', '%Y-%m-%d %H:%M:%S'
        ).replace(tzinfo=ET)
        cutoff_utc = trade_date_930_et.astimezone(timezone.utc)
        valid = [a for a in arts if datetime.fromisoformat(a["ts"]) < cutoff_utc]
    else:
        raise ValueError(cutoff_mode)

    n = len(valid)
    n_scanner = sum(1 for a in valid if is_scanner(a["headline"]))
    only_scanner = (n > 0 and n_scanner == n)
    has_catalyst = n > n_scanner
    return n, only_scanner, has_catalyst


def apply_filter(trades, cache, cutoff_mode, filter_rule):
    """Apply filter and return final equity."""
    cash = STARTING_CASH
    kept = 0
    for t in trades:
        n, only_scanner, has_catalyst = count_news(t, cache, cutoff_mode)
        keep = True
        if filter_rule == "skip_<2_or_only_scanner":
            if n < 2 or only_scanner: keep = False
        elif filter_rule == "skip_no_catalyst":
            if not has_catalyst: keep = False
        elif filter_rule == "skip_0":
            if n == 0: keep = False
        elif filter_rule == "skip_only_scanner":
            if only_scanner: keep = False
        if keep:
            cash += t["pnl"]
            kept += 1
    return cash, kept


def main():
    os.makedirs("results/news_filter_pit", exist_ok=True)
    client = NewsClient(
        "PKIPXFIETM7H4BAGQ64FQV3IWJ",
        "25RY682kuN9EBcFr6SFxgbSpFjdZkn613PLKPy1TdYzG"
    )

    print(f"=== News filter: lookahead (BUG) vs point-in-time (CORRECT) ===\n")

    for label, path in CANDIDATES:
        if not os.path.exists(path):
            print(f"[skip] {path}"); continue
        print(f"--- {label} ---")
        trades, baseline = get_trades(path, label)
        n = len(trades)
        print(f"  {n} trades, baseline equity ${baseline:,.0f} ({baseline/STARTING_CASH:.2f}x)")

        cache_path = f"results/news_filter_pit/cache_{os.path.basename(path).replace('.json','')}.json"
        print(f"  Fetching news with timestamps...")
        cache = fetch_news_with_timestamps(client, trades, cache_path)

        # Compare lookahead vs point-in-time
        print(f"\n  {'filter':<32} {'mode':<14} {'kept':>5} {'equity':>12} {'lift':>12} {'multi':>8}")
        for rule in ["skip_0", "skip_only_scanner", "skip_<2_or_only_scanner", "skip_no_catalyst"]:
            for mode in ["lookahead", "pit"]:
                eq, kept = apply_filter(trades, cache, mode, rule)
                lift = eq - baseline
                multi = eq / STARTING_CASH
                marker = " (BUG)" if mode == "lookahead" else " (PIT)"
                print(f"  {rule:<32} {mode:<14} {kept:>5} ${eq:>11,.0f} ${lift:>+11,.0f} {multi:>7.2f}x")
        print()


if __name__ == "__main__":
    main()
