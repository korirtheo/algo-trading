"""Centralized event logger — writes significant errors/events to system_events DB.

Usage:
    from live.event_logger import log_event, set_db

    # In exception handlers:
    log_event("buy_failed", "error", f"BUY {ticker} FAILED: {e}")

    # Optionally inject the engine's DB instance for connection reuse:
    set_db(engine.db)
"""

import logging

_log = logging.getLogger(__name__)

_db = None


def set_db(db):
    """Inject the engine's TradingDatabase instance (called from main.py)."""
    global _db
    _db = db


def log_event(event_type, severity, message, details=None):
    """Log an event to both Python logger and system_events DB.

    Args:
        event_type: Category (e.g. 'buy_failed', 'tradier_stream_error', 'session_crash')
        severity: 'info', 'warning', 'error', 'critical'
        message: Human-readable description
        details: Optional JSON string with extra context
    """
    global _db

    # Lazy-init DB if not injected
    if _db is None:
        try:
            from live.persistence_db import TradingDatabase

            _db = TradingDatabase()
        except Exception:
            _log.error(
                "event_logger: DB unavailable, cannot log %s: %s", event_type, message
            )
            return

    try:
        _db.log_system_event(event_type, severity, message, details=details)
    except Exception:
        _log.error(
            "event_logger: failed to write %s to DB: %s", event_type, message
        )
