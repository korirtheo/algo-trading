"""
Feed comparison logger — writes 2-min bars from both Tradier and Alpaca IEX
into the feed_comparison SQLite table for dashboard analysis.
"""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

SOURCE_TRADIER = "tradier"
SOURCE_ALPACA_IEX = "alpaca_iex"


class FeedLogger:
    """Logs bars from multiple sources to the feed_comparison table."""

    def __init__(self, db):
        self._db = db

    def log(self, source: str, symbol: str, bar: dict):
        """
        Record a 2-min bar from a data source.

        Args:
            source: 'tradier' or 'alpaca_iex'
            symbol: ticker symbol
            bar:    dict with Open, High, Low, Close, Volume, timestamp
        """
        try:
            ts = bar.get("timestamp")
            if ts is None:
                return

            if hasattr(ts, "astimezone"):
                ts_et = ts.astimezone(ET)
            else:
                ts_et = datetime.fromisoformat(str(ts)).astimezone(ET)

            date_str = ts_et.strftime("%Y-%m-%d")
            bar_time_str = ts_et.strftime("%H:%M")

            self._db.log_feed_bar(
                date=date_str,
                bar_time=bar_time_str,
                ticker=symbol,
                source=source,
                open_=float(bar.get("Open", 0)),
                high=float(bar.get("High", 0)),
                low=float(bar.get("Low", 0)),
                close=float(bar.get("Close", 0)),
                volume=int(bar.get("Volume", 0)),
            )
        except Exception as e:
            log.warning(f"FeedLogger.log error ({source}/{symbol}): {e}")
