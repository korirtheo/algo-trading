"""Pre-market and runtime health checks for the live trading system.

Verifies all critical components are working before market opens and during
trading hours. Logs failures to system_events DB via log_event().

Usage:
    from live.health_check import run_health_check, check_streaming_health
    healthy, results = run_health_check(engine, executor, tradier_streamer, fill_stream)
"""

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from live.event_logger import log_event

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

TRADIER_QUOTES_URL = "https://api.tradier.com/v1/markets/quotes"
TRADIER_SESSION_URL = "https://api.tradier.com/v1/markets/events/session"


def _check(name, status, message):
    """Create a check result dict."""
    return {"component": name, "status": status, "message": message}


def check_alpaca_api(executor):
    """Verify Alpaca API is reachable and account is accessible."""
    try:
        acct = executor.get_account()
        equity = float(acct.equity) if acct else 0
        return _check("alpaca_api", "ok", f"Alpaca API reachable, equity=${equity:,.2f}")
    except Exception as e:
        return _check("alpaca_api", "error", f"Alpaca API unreachable: {e}")


def check_tradier_api():
    """Verify Tradier REST API is reachable (fetch SPY quote)."""
    from config.settings import TRADIER_API_KEY

    try:
        resp = requests.get(
            TRADIER_QUOTES_URL,
            params={"symbols": "SPY", "greeks": "false"},
            headers={
                "Authorization": f"Bearer {TRADIER_API_KEY}",
                "Accept": "application/json",
            },
            timeout=5,
        )
        resp.raise_for_status()
        quotes = resp.json().get("quotes", {}).get("quote", [])
        if quotes:
            return _check("tradier_api", "ok", f"Tradier API reachable (SPY quote OK)")
        return _check("tradier_api", "warning", "Tradier API reachable but no quote data")
    except Exception as e:
        return _check("tradier_api", "error", f"Tradier API unreachable: {e}")


def check_tradier_stream():
    """Verify Tradier streaming session can be created."""
    from config.settings import TRADIER_API_KEY

    try:
        resp = requests.post(
            TRADIER_SESSION_URL,
            headers={
                "Authorization": f"Bearer {TRADIER_API_KEY}",
                "Accept": "application/json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        session_id = resp.json().get("stream", {}).get("sessionid")
        if session_id:
            return _check("tradier_stream", "ok", f"Tradier streaming session created ({session_id[:8]}...)")
        return _check("tradier_stream", "error", "Tradier session creation returned no session ID")
    except Exception as e:
        return _check("tradier_stream", "error", f"Tradier streaming session failed: {e}")


def check_engine_db(engine):
    """Verify engine database is accessible."""
    try:
        with engine.db._conn() as conn:
            conn.execute("SELECT 1")
        return _check("engine_db", "ok", "Engine database accessible")
    except Exception as e:
        return _check("engine_db", "error", f"Engine database inaccessible: {e}")


def check_fill_stream(fill_stream):
    """Verify FillStream (Alpaca TradingStream) is alive."""
    if fill_stream is None:
        return _check("fill_stream", "warning", "FillStream not available (using legacy polling)")
    if getattr(fill_stream, "_started", False):
        return _check("fill_stream", "ok", "FillStream running")
    return _check("fill_stream", "error", "FillStream exists but not started")


def check_tradier_streaming_alive(tradier_streamer):
    """Check if Tradier streamer is actively connected (post-open check)."""
    if tradier_streamer is None:
        return _check("tradier_streaming", "error", "Tradier streamer not initialized")
    if not getattr(tradier_streamer, "_running", False):
        return _check("tradier_streaming", "error", "Tradier streamer is not running")
    if not getattr(tradier_streamer, "_session_id", None):
        return _check("tradier_streaming", "warning", "Tradier streamer running but no active session")
    return _check("tradier_streaming", "ok", f"Tradier streaming active (session {tradier_streamer._session_id[:8]}...)")


def check_bars_arriving(engine, max_stale_seconds=360, halt_monitor=None):
    """Check if bars are arriving within expected timeframe (post-open check).

    Args:
        max_stale_seconds: Max seconds since last bar before flagging as stale (default 6 min)
        halt_monitor: HaltMonitor instance — if provided, halted tickers are excluded from stale warnings
    """
    now = datetime.now(ET)

    # Build set of currently-halted tickers (halted but not yet resumed)
    halted_tickers = set()
    if halt_monitor is not None:
        for ev in halt_monitor.recent_events:
            if not ev.resumed:
                halted_tickers.add(ev.ticker)

    # Check each symbol's last bar time
    stale_symbols = []
    halted_stale = []
    for symbol, bars in engine.bar_data.items():
        if bars:
            last_bar = bars[-1]
            last_ts = last_bar.get("timestamp")
            if last_ts:
                if hasattr(last_ts, "astimezone"):
                    last_ts = last_ts.astimezone(ET)
                age = (now - last_ts).total_seconds()
                if age > max_stale_seconds:
                    if symbol in halted_tickers:
                        halted_stale.append(symbol)
                    else:
                        stale_symbols.append(f"{symbol} ({int(age)}s ago)")

    # Log halted tickers to system events for record-keeping
    for symbol in halted_stale:
        log_event(
            "ticker_halted",
            "info",
            f"{symbol} is halted (no bars — expected behavior)",
        )

    if not stale_symbols and not halted_stale:
        return _check("bar_feed", "ok", "All feeds delivering bars")

    parts = []
    if stale_symbols:
        parts.append(f"Stale bars: {', '.join(stale_symbols[:5])}")
    if halted_stale:
        parts.append(f"Halted (no bars expected): {', '.join(halted_stale[:5])}")
    msg = " | ".join(parts)

    severity = "warning" if len(stale_symbols) <= len(engine.bar_data) // 2 else "error"
    return _check("bar_feed", severity, msg)


def run_health_check(engine, executor, tradier_streamer=None, fill_stream=None):
    """Run all pre-market health checks.

    Returns:
        (healthy: bool, results: list[dict])
        Each result: {component, status: 'ok'|'warning'|'error', message}
    """
    results = [
        check_alpaca_api(executor),
        check_tradier_api(),
        check_tradier_stream(),
        check_engine_db(engine),
        check_fill_stream(fill_stream),
    ]

    # Log failures to system_events
    for r in results:
        if r["status"] == "error":
            log_event("health_check_failed", "error", f"[{r['component']}] {r['message']}")
            log.error(f"HEALTH CHECK FAILED: {r['component']}: {r['message']}")
        elif r["status"] == "warning":
            log_event("health_check_warning", "warning", f"[{r['component']}] {r['message']}")
            log.warning(f"HEALTH CHECK WARNING: {r['component']}: {r['message']}")
        else:
            log.info(f"HEALTH CHECK OK: {r['component']}: {r['message']}")

    healthy = all(r["status"] != "error" for r in results)
    return healthy, results


def check_streaming_health(engine, tradier_streamer=None, halt_monitor=None):
    """Run post-open streaming health checks (called periodically during trading).

    Returns:
        (healthy: bool, results: list[dict])
    """
    results = [
        check_tradier_streaming_alive(tradier_streamer),
        check_bars_arriving(engine, halt_monitor=halt_monitor),
    ]

    for r in results:
        if r["status"] == "error":
            log_event("streaming_health_failed", "error", f"[{r['component']}] {r['message']}")
            log.error(f"STREAMING HEALTH FAILED: {r['component']}: {r['message']}")
        elif r["status"] == "warning":
            log_event("streaming_health_warning", "warning", f"[{r['component']}] {r['message']}")
            log.warning(f"STREAMING HEALTH WARNING: {r['component']}: {r['message']}")

    healthy = all(r["status"] != "error" for r in results)
    return healthy, results
