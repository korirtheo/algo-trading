"""Intraday top-gainers discovery monitor (Phase 1 — no-trade).

Polls Alpaca's market-movers API every 60s during RTH and logs ticker discoveries
that weren't in the morning PM watchlist. Goal: build a real dataset of intraday
runners we currently miss, so Phase 2 backtest can use real data instead of
fantasy reconstructed signals.

Plan: results/intraday_top_gainers_plan.md (Phase 1)

Filters (microcap-pump-relevant):
  - percent_change >= +20% (from prev close — proxy for "running")
  - $1 <= price <= $50
  - exclude warrants / units / weird tickers

Output: logs/intraday_gainers/<YYYY-MM-DD>.jsonl, one JSON line per snapshot.
Each line includes the ticker's metadata + first-seen time + peak %.

Usage:
    python -m live.intraday_gainers_monitor
"""
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.enums import MarketType
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET
from live.event_logger import log_event

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

POLL_SECONDS = 60
MIN_PCT_CHANGE = 20.0       # %, vs prev close
MIN_PRICE = 1.0
MAX_PRICE = 50.0
TOP_N_PER_POLL = 50          # how many movers to pull each poll
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "intraday_gainers",
)


def _is_warrant_or_unit(ticker):
    if ".WS" in ticker or ".RT" in ticker:
        return True
    if re.match(r"^[A-Z]{3,}W$", ticker):
        return True
    if ticker.endswith("WW"):
        return True
    if re.match(r"^[A-Z]{3,}U$", ticker):
        return True
    if re.match(r"^[A-Z]{3,}R$", ticker):
        return True
    if "." in ticker or len(ticker) > 5:
        return True
    return False


def _within_rth():
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    close_t = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
    return open_t <= now < close_t


def _load_pm_watchlist(date_str):
    """Read the algotrader's watchlist for `date_str` from the AWS-style trade log if present.

    Returns set of tickers that were in the morning PM watchlist (so we can mark
    intraday discoveries as "NEW" vs. already-known PM gappers).
    """
    candidates = []
    log_dir = os.path.dirname(LOG_DIR.replace("intraday_gainers", ""))
    main_log = os.path.join(log_dir, f"{date_str}_live.log")
    if not os.path.exists(main_log):
        return set()
    with open(main_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # match: 'Watchlist locked for YYYY-MM-DD: N candidates' followed by ticker lines
            if "Subscribing to" in line and "symbols:" in line:
                m = re.search(r"\[([^\]]+)\]", line)
                if m:
                    candidates = [t.strip().strip("'\"") for t in m.group(1).split(",")]
                    break
    return set(candidates)


def _load_news_cache():
    """Load news_cache/master.json once at startup, return ticker -> list of headlines today."""
    try:
        from config.settings import ALPACA_API_KEY  # noqa — ensure config loaded
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "news_cache", "master.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        today = datetime.now(ET).date().isoformat()
        by_ticker = {}
        if isinstance(cache, dict):
            # cache may be {ticker: {date: [...]}}, or {date: {ticker: [...]}}, or flat list
            for k, v in cache.items():
                if isinstance(v, dict) and today in v:
                    by_ticker[k] = v[today]
                elif k == today and isinstance(v, dict):
                    for t, items in v.items():
                        by_ticker[t] = items
        return by_ticker
    except Exception as e:
        log.warning("news cache load failed: %s", e)
        log_event("data_error", "warning", f"News cache load failed: {e}")
        return {}


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    os.makedirs(LOG_DIR, exist_ok=True)

    client = ScreenerClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    today_str = datetime.now(ET).date().isoformat()
    out_path = os.path.join(LOG_DIR, f"{today_str}.jsonl")
    pm_watchlist = _load_pm_watchlist(today_str)
    news_today = _load_news_cache()

    log.info("Intraday gainers monitor starting. Output: %s", out_path)
    log.info("PM watchlist: %d tickers loaded", len(pm_watchlist))
    log.info("News cache: %d tickers with headlines today", len(news_today))
    log.info("Filter: %+.0f%%, $%.0f<=price<=$%.0f, top %d per poll, every %ds",
             MIN_PCT_CHANGE, MIN_PRICE, MAX_PRICE, TOP_N_PER_POLL, POLL_SECONDS)

    discovered = {}  # ticker -> {first_seen, peak_pct, last_seen, ...}
    stop_requested = False

    def _stop(*_):
        nonlocal stop_requested
        stop_requested = True
        log.info("Stop signal received, exiting cleanly...")

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    poll_count = 0
    while not stop_requested:
        if not _within_rth():
            now = datetime.now(ET)
            today_open = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
            # Find the next market open (today if before, else next weekday)
            next_open = today_open if now < today_open else today_open + timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
            wait_s = max(10, int((next_open - now).total_seconds()))
            label = "Before RTH" if now < today_open else "After RTH"
            log.info("%s. %d discoveries logged for %s. Sleeping until next open at %s (~%dm).",
                     label, len(discovered), today_str, next_open.isoformat(), wait_s // 60)
            # Sleep in chunks so SIGTERM is responsive
            slept = 0
            while slept < wait_s and not stop_requested:
                chunk = min(60, wait_s - slept)
                time.sleep(chunk)
                slept += chunk
            # Reset day-scoped state if the date rolled
            new_today = datetime.now(ET).date().isoformat()
            if new_today != today_str:
                log.info("Day rolled %s -> %s, resetting discoveries + reloading PM watchlist", today_str, new_today)
                today_str = new_today
                out_path = os.path.join(LOG_DIR, f"{today_str}.jsonl")
                pm_watchlist = _load_pm_watchlist(today_str)
                news_today = _load_news_cache()
                discovered = {}
                poll_count = 0
            continue

        try:
            req = MarketMoversRequest(top=TOP_N_PER_POLL, market_type=MarketType.STOCKS)
            movers = client.get_market_movers(req)
            now_iso = datetime.now(ET).isoformat()
            new_this_poll = []
            for g in (movers.gainers or []):
                tk = g.symbol
                if _is_warrant_or_unit(tk):
                    continue
                px = float(g.price)
                pct = float(g.percent_change)
                if pct < MIN_PCT_CHANGE or px < MIN_PRICE or px > MAX_PRICE:
                    continue
                in_pm = tk in pm_watchlist
                if tk not in discovered:
                    discovered[tk] = {
                        "ticker": tk,
                        "first_seen": now_iso,
                        "first_price": px,
                        "first_pct": pct,
                        "peak_pct": pct,
                        "in_pm_watchlist": in_pm,
                        "news_today_count": len(news_today.get(tk, [])),
                    }
                    new_this_poll.append(tk)
                d = discovered[tk]
                d["last_seen"] = now_iso
                d["last_price"] = px
                d["last_pct"] = pct
                d["peak_pct"] = max(d["peak_pct"], pct)

            # Write a single line per poll with the new entrants + current snapshot
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": now_iso,
                    "poll": poll_count,
                    "n_qualifying_now": sum(1 for d in discovered.values()
                                            if d.get("last_seen") == now_iso),
                    "new_this_poll": new_this_poll,
                    "active": [
                        {k: d[k] for k in ("ticker", "last_price", "last_pct", "peak_pct",
                                          "in_pm_watchlist", "news_today_count", "first_seen")}
                        for d in discovered.values() if d.get("last_seen") == now_iso
                    ],
                }) + "\n")

            if new_this_poll:
                tagged = [
                    f"{t}{'(PM)' if discovered[t]['in_pm_watchlist'] else ''}"
                    f"{'(NEWS)' if discovered[t]['news_today_count'] else ''}"
                    for t in new_this_poll
                ]
                log.info("Poll %d: NEW %s | active now=%d",
                         poll_count, ", ".join(tagged),
                         sum(1 for d in discovered.values() if d.get("last_seen") == now_iso))
            elif poll_count % 10 == 0:
                log.info("Poll %d: no new entrants. active=%d total_today=%d",
                         poll_count,
                         sum(1 for d in discovered.values() if d.get("last_seen") == now_iso),
                         len(discovered))
        except Exception as e:
            log.warning("Poll %d failed: %s", poll_count, e)
            log_event("intraday_scan_error", "warning", f"Intraday gainers poll {poll_count} failed: {e}")

        poll_count += 1
        time.sleep(POLL_SECONDS)

    log.info("Exiting. Total unique discoveries today: %d", len(discovered))


if __name__ == "__main__":
    run()
