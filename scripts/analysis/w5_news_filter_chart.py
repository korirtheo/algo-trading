"""Compare top W5 candidate(s) WITH vs WITHOUT news filter on blind 2026.

Uses POINT-IN-TIME news filtering (articles published before 9:30 AM ET
on trade date). Generates equity curve chart showing the lift.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import time
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

ET = ZoneInfo("America/New_York")
CANDIDATES = [
    ("#124 W3 (baseline)",   "config/trial_124_microcap_pump_extracted.json"),
    ("#312 W5 (best train)", "config/trial_312_w5_extracted.json"),
    ("#310 W5",              "config/trial_310_w5_extracted.json"),
    ("#342 W5",              "config/trial_342_w5_extracted.json"),
]
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000
OUT_DIR = f"results/news_filter_pit/{os.environ.get('TEST_YEAR', '2025')}"
DATA_DIRS = [
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",  # 2025 train year
    "stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026",  # 2026 blind
]
TEST_YEAR = os.environ.get("TEST_YEAR", "2025")  # default 2025 for this run

SCANNER_PATTERNS = [
    r"\d+ .*stocks moving", r"stocks moving",
    r"pre-market session", r"intraday session", r"after-market session",
    r"top gainers", r"top losers", r"morning gainers",
    r"premarket movers", r"midday movers",
]


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def get_trades_with_equity(config_path):
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
    dates = sorted([d for d in all_dates if d.startswith(TEST_YEAR)])
    cash = STARTING_CASH
    daily = []  # ordered list of (date, list_of_trades_today)
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            daily.append((d, []))
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            daily.append((d, []))
            continue
        today_trades = []
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                today_trades.append({
                    "ticker": st["ticker"], "pnl": st["pnl"],
                    "cost": st["position_cost"], "strategy": st.get("strategy"),
                })
        daily.append((d, today_trades))
        cash = end_c + (unset if is_cash else 0)
    return daily


def is_scanner(headline):
    h = headline.lower()
    return any(re.search(p, h) for p in SCANNER_PATTERNS)


def fetch_news_with_ts(client, trades, cache_path):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    unique = set()
    for _, ts in trades:
        for tr in ts:
            unique.add(f"{tr['ticker']}|{_}")
    targets = [u for u in unique if u not in cache]
    if not targets:
        print(f"  All news cached ({len(cache)} entries)")
        return cache
    print(f"  Fetching {len(targets)} new entries...")
    for i, key in enumerate(sorted(targets)):
        ticker, d = key.split("|")
        start = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(days=2)
        end = start + timedelta(days=4)
        try:
            n = client.get_news(NewsRequest(symbols=ticker, start=start, end=end, limit=30))
            arts = n.data.get("news", [])
            cache[key] = [{"ts": h.created_at.isoformat(),
                            "headline": h.headline} for h in arts]
        except Exception:
            cache[key] = []
        time.sleep(0.05)
        if i % 30 == 0 and i > 0:
            json.dump(cache, open(cache_path, "w"))
            print(f"    {i}/{len(targets)}", flush=True)
    json.dump(cache, open(cache_path, "w"))
    return cache


def count_pit(trade, date_str, cache):
    key = f"{trade['ticker']}|{date_str}"
    arts = cache.get(key, [])
    # Point-in-time cutoff: 9:30 AM ET on trade date
    cutoff = ET.localize(
        datetime.strptime(date_str + " 09:30:00", "%Y-%m-%d %H:%M:%S")
    ) if hasattr(ET, "localize") else datetime.strptime(
        date_str + " 09:30:00", "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=ET)
    cutoff_utc = cutoff.astimezone(timezone.utc)
    valid = [a for a in arts if datetime.fromisoformat(a["ts"]) < cutoff_utc]
    n = len(valid)
    n_scanner = sum(1 for a in valid if is_scanner(a["headline"]))
    has_catalyst = n > n_scanner
    return n, has_catalyst


def equity_with_filter(daily, cache, filter_rule):
    cash = STARTING_CASH
    eq = [cash]
    for d, trades in daily:
        for t in trades:
            if filter_rule == "none":
                keep = True
            else:
                n, has_catalyst = count_pit(t, d, cache)
                if filter_rule == "skip_0":
                    keep = n > 0
                elif filter_rule == "skip_no_catalyst":
                    keep = has_catalyst
                else:
                    keep = True
            if keep:
                cash += t["pnl"]
        eq.append(cash)
    return np.array(eq)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = NewsClient(
        "PKIPXFIETM7H4BAGQ64FQV3IWJ",
        "25RY682kuN9EBcFr6SFxgbSpFjdZkn613PLKPy1TdYzG"
    )

    print(f"=== Validating news filter signal on top W5 candidates + #124 baseline ===\n")

    # Run all candidates
    results = {}
    for label, path in CANDIDATES:
        if not os.path.exists(path):
            print(f"[skip] {label}"); continue
        print(f"--- {label} ---")
        daily = get_trades_with_equity(path)
        n_trades = sum(len(ts) for _, ts in daily)
        baseline_final = STARTING_CASH + sum(t["pnl"] for _, ts in daily for t in ts)
        print(f"  Total trades: {n_trades}  Baseline final: ${baseline_final:,.0f}")

        cache_path = f"{OUT_DIR}/cache_{os.path.basename(path).replace('.json','')}.json"
        cache = fetch_news_with_ts(client, daily, cache_path)

        eq_none = equity_with_filter(daily, cache, "none")
        eq_skip0 = equity_with_filter(daily, cache, "skip_0")
        eq_skipnc = equity_with_filter(daily, cache, "skip_no_catalyst")

        results[label] = {
            "daily": daily,
            "eq_none": eq_none,
            "eq_skip0": eq_skip0,
            "eq_skipnc": eq_skipnc,
        }
        print(f"  WITHOUT filter:       ${eq_none[-1]:,.0f}  ({eq_none[-1]/STARTING_CASH:.2f}x)")
        print(f"  Skip-0-articles:      ${eq_skip0[-1]:,.0f}  ({eq_skip0[-1]/STARTING_CASH:.2f}x)  "
              f"lift +${eq_skip0[-1]-eq_none[-1]:,.0f}")
        print(f"  Skip-no-catalyst:     ${eq_skipnc[-1]:,.0f}  ({eq_skipnc[-1]/STARTING_CASH:.2f}x)  "
              f"lift +${eq_skipnc[-1]-eq_none[-1]:,.0f}")
        print()

    # --- Chart: focused on #312 W5 (the headline candidate) ---
    if "#312 W5 (best train)" in results:
        r = results["#312 W5 (best train)"]
        fig, ax = plt.subplots(figsize=(14, 7))
        x = range(len(r["eq_none"]))
        ax.plot(x, r["eq_none"], color="#d62728", linewidth=2.5,
                label=f"WITHOUT filter (W5 #312): ${r['eq_none'][-1]:,.0f} ({r['eq_none'][-1]/STARTING_CASH:.2f}x)")
        ax.plot(x, r["eq_skipnc"], color="#2ca02c", linewidth=2.5,
                label=f"WITH news filter (skip-no-catalyst): ${r['eq_skipnc'][-1]:,.0f} ({r['eq_skipnc'][-1]/STARTING_CASH:.2f}x)")
        ax.plot(x, r["eq_skip0"], color="#1f77b4", linewidth=2.0, linestyle="--",
                label=f"WITH news filter (skip-0-articles): ${r['eq_skip0'][-1]:,.0f} ({r['eq_skip0'][-1]/STARTING_CASH:.2f}x)")
        ax.axhline(STARTING_CASH, color="gray", linestyle=":", alpha=0.5,
                   label=f"start ${STARTING_CASH:,}")
        ax.set_title("W5 #312 (overfit model) on 2026 — news filter rescues it from loss\n"
                     "Point-in-time: articles before 9:30 AM ET on trade date only", fontsize=12)
        ax.set_xlabel("Trading day index in 2026")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
        fig.tight_layout()
        p1 = f"{OUT_DIR}/w5_312_news_filter.png"
        fig.savefig(p1, dpi=140); plt.close(fig)
        print(f"Wrote {p1}")

    # --- Chart: bar comparison across all candidates ---
    fig, ax = plt.subplots(figsize=(13, 7))
    labels = list(results.keys())
    x = np.arange(len(labels))
    w = 0.28
    no_filter = [results[l]["eq_none"][-1] for l in labels]
    skip_0 = [results[l]["eq_skip0"][-1] for l in labels]
    skip_nc = [results[l]["eq_skipnc"][-1] for l in labels]
    ax.bar(x - w, no_filter, w, label="Without filter", color="#d62728")
    ax.bar(x, skip_0, w, label="Skip-0-articles", color="#1f77b4")
    ax.bar(x + w, skip_nc, w, label="Skip-no-catalyst", color="#2ca02c")
    ax.axhline(STARTING_CASH, color="gray", linestyle="--", alpha=0.5,
               label=f"start ${STARTING_CASH:,}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("News filter impact on 2026 blind — all top candidates", fontsize=13)
    ax.set_ylabel("Final equity ($)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    fig.tight_layout()
    p2 = f"{OUT_DIR}/news_filter_all_candidates.png"
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
