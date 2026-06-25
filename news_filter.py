"""Point-in-time (PIT) news filter for trading picks.

Loads master news cache once per process and exposes a filter_picks() helper
that drops watchlist entries failing a configurable news-count + catalyst bar.

Cutoff is 9:30 AM ET on the trade date — only articles published BEFORE that
moment count, eliminating look-ahead bias.
"""
import os
import re
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
_CACHE = None
_CACHE_PATH = os.path.join("results", "news_cache", "master.json")

# Scanner-roundup headlines (low-signal) — same patterns used in the analysis
# scripts. Trades whose only "news" is a scanner roundup don't count as a
# catalyst.
SCANNER_PATTERNS = [
    r"\d+ .*stocks moving", r"stocks moving",
    r"pre-market session", r"intraday session", r"after-market session",
    r"top gainers", r"top losers", r"morning gainers",
    r"premarket movers", r"midday movers",
]
_SCANNER_RE = [re.compile(p) for p in SCANNER_PATTERNS]


def _load():
    global _CACHE
    if _CACHE is None:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH) as f:
                _CACHE = json.load(f)
        else:
            _CACHE = {}
    return _CACHE


def _is_scanner(headline):
    h = headline.lower()
    return any(p.search(h) for p in _SCANNER_RE)


def count_pit(ticker, date_str):
    """Return (n_articles, has_catalyst) for articles published before
    9:30 AM ET on date_str. has_catalyst means at least one non-scanner article.
    """
    cache = _load()
    arts = cache.get(f"{ticker}|{date_str}", [])
    if not arts:
        return 0, False
    cutoff = datetime.strptime(
        date_str + " 09:30:00", "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=ET).astimezone(timezone.utc)
    valid = [a for a in arts if datetime.fromisoformat(a["ts"]) < cutoff]
    n = len(valid)
    n_scanner = sum(1 for a in valid if _is_scanner(a["headline"]))
    return n, (n > n_scanner)


def filter_picks(picks, date_str, min_articles=0, require_catalyst=False):
    """Drop picks that don't pass the news bar. Returns filtered list.
    No-op when both thresholds are trivial."""
    if min_articles <= 0 and not require_catalyst:
        return picks
    out = []
    for p in picks:
        ticker = p.get("ticker")
        if not ticker:
            continue
        n, has_cat = count_pit(ticker, date_str)
        if n < min_articles:
            continue
        if require_catalyst and not has_cat:
            continue
        out.append(p)
    return out


def cache_stats():
    cache = _load()
    return {"entries": len(cache), "path": _CACHE_PATH}
