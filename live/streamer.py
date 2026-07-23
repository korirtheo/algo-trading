"""
Real-Time Bar Streamer: Alpaca WebSocket -> 2-minute candle aggregation.

Subscribes to 1-minute bars for watchlist symbols via Alpaca's data stream.
Logs raw 1-min bars to logs/bars/raw-1min/<YYYY-MM-DD>/<symbol>.csv
Aggregates into 2-minute candles aligned to market open (9:30 ET).
Emits completed 2-min bars to callback.
"""
import logging
import threading
import os
import csv
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from collections import defaultdict

from alpaca.data.enums import DataFeed
from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED, ALPACA_FEED_DELAYED
from live.event_logger import log_event

_FEED_ENUM = DataFeed.IEX if ALPACA_FEED == "iex" else DataFeed.SIP

# When using delayed SIP (free tier), bars arrive 15 min late but have correct timestamps.
# We use the bar's timestamp (not wall clock) to align strategy timing.
FEED_DELAYED = ALPACA_FEED_DELAYED if ALPACA_FEED == "sip" else False

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
MARKET_OPEN = dt_time(9, 30)


class TwoMinBar:
    """Aggregates 1-min bars into a 2-min bar."""
    __slots__ = ("open", "high", "low", "close", "volume", "timestamp", "count")

    def __init__(self, bar):
        self.open = bar.open
        self.high = bar.high
        self.low = bar.low
        self.close = bar.close
        self.volume = bar.volume
        self.timestamp = bar.timestamp
        self.count = 1

    def merge(self, bar):
        self.high = max(self.high, bar.high)
        self.low = min(self.low, bar.low)
        self.close = bar.close
        self.volume += bar.volume
        self.count += 1

    def to_dict(self):
        return {
            "Open": self.open,
            "High": self.high,
            "Low": self.low,
            "Close": self.close,
            "Volume": self.volume,
            "timestamp": self.timestamp,
        }


def _bar_slot(ts):
    """Compute the 2-min slot index from a bar timestamp.
    Candle 1 = 9:30-9:31 -> slot 0
    Candle 2 = 9:32-9:33 -> slot 1
    """
    et = ts.astimezone(ET)
    minutes_since_open = (et.hour * 60 + et.minute) - (9 * 60 + 30)
    return minutes_since_open // 2


class BarStreamer:
    def __init__(self, on_2min_bar):
        """
        Args:
            on_2min_bar: callback(symbol: str, bar: dict) called when a 2-min bar completes
        """
        self.on_2min_bar = on_2min_bar
        from alpaca.data.live import StockDataStream
        self.stream = StockDataStream(ALPACA_API_KEY, ALPACA_API_SECRET, feed=_FEED_ENUM)
        self.pending = {}  # symbol -> (slot_index, TwoMinBar)
        self._symbols = []
        self._running = False
        self._lock = threading.Lock()

    async def _on_bar_async(self, bar):
        self._handle_bar(bar)

    def subscribe(self, symbols):
        """Subscribe to 1-min bars for given symbols."""
        with self._lock:
            self._symbols = list(symbols)
            log.info(f"Subscribing to {len(self._symbols)} symbols: {self._symbols}")
            if self._symbols:
                self.stream.subscribe_bars(self._on_bar_async, *self._symbols)

    def add_symbol(self, symbol):
        """Subscribe to bars for a ticker after the stream is running.

        Used by the halt-resume monitor to attach mid-day discoveries to the
        live bar feed. Safe to call from a background thread.
        """
        symbol = symbol.upper()
        with self._lock:
            if symbol in self._symbols:
                log.debug(f"add_symbol: {symbol} already subscribed")
                return False
            self._symbols.append(symbol)
        try:
            self.stream.subscribe_bars(self._on_bar_async, symbol)
            log.info(f"add_symbol: subscribed to {symbol} (total {len(self._symbols)})")
            return True
        except Exception as e:
            log.error(f"add_symbol: subscribe_bars({symbol}) failed: {e}")
            log_event("stream_subscribe_error", "error", f"subscribe_bars({symbol}) failed: {e}")
            with self._lock:
                if symbol in self._symbols:
                    self._symbols.remove(symbol)
            return False

    def remove_symbol(self, symbol):
        """Unsubscribe a ticker from the live feed."""
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self._symbols:
                return False
            self._symbols.remove(symbol)
        try:
            self.stream.unsubscribe_bars(symbol)
            log.info(f"remove_symbol: unsubscribed {symbol} (total {len(self._symbols)})")
            return True
        except Exception as e:
            log.warning(f"remove_symbol: unsubscribe_bars({symbol}) failed: {e}")
            log_event("stream_subscribe_error", "warning", f"unsubscribe_bars({symbol}) failed: {e}")
            return False

    def symbols(self):
        """Return a snapshot of currently subscribed symbols."""
        with self._lock:
            return list(self._symbols)

    def _log_1min_bar_to_csv(self, symbol, bar):
        """Log raw 1-min bar to CSV.

        Output path: logs/bars/raw-1min/<YYYY-MM-DD>/<symbol>.csv
        Useful for detailed audit trails and alternative analysis.
        """
        try:
            today = datetime.now(ET).strftime("%Y-%m-%d")
            # Use /app/logs directly since we're in Docker
            bars_date_dir = os.path.join("/app/logs", "bars", "raw-1min", today)
            os.makedirs(bars_date_dir, exist_ok=True)

            path = os.path.join(bars_date_dir, f"{symbol}.csv")
            new_file = not os.path.exists(path)

            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["timestamp", "Open", "High", "Low", "Close", "Volume"])
                w.writerow([
                    bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume,
                ])
        except Exception as e:
            log.warning(f"Failed to log 1-min bar for {symbol}: {e}")
            log_event("data_error", "warning", f"Failed to log 1-min bar for {symbol}: {e}")

    def _handle_bar(self, bar):
        """Process incoming 1-min bar, aggregate to 2-min."""
        symbol = bar.symbol
        log.info(f"1min bar: {symbol} close={bar.close:.2f} vol={bar.volume:,} t={bar.timestamp}")

        # Log raw 1-min bar
        self._log_1min_bar_to_csv(symbol, bar)

        slot = _bar_slot(bar.timestamp)

        if symbol in self.pending:
            prev_slot, prev_bar = self.pending[symbol]
            if slot == prev_slot:
                # Same 2-min window — merge
                prev_bar.merge(bar)
                if prev_bar.count >= 2:
                    # 2-min bar complete
                    log.info(f"EMIT 2min: {symbol} c={prev_bar.close:.2f} count={prev_bar.count}"); self.on_2min_bar(symbol, prev_bar.to_dict())
                    del self.pending[symbol]
                return
            else:
                # New slot — emit previous bar (even if only 1 min) and start new
                log.info(f"EMIT 2min: {symbol} c={prev_bar.close:.2f} count={prev_bar.count}"); self.on_2min_bar(symbol, prev_bar.to_dict())

        # Start new pending bar
        self.pending[symbol] = (slot, TwoMinBar(bar))

    def flush_pending(self):
        """Emit all pending partial bars (e.g., at EOD)."""
        for symbol, (slot, bar) in list(self.pending.items()):
            self.on_2min_bar(symbol, bar.to_dict())
        self.pending.clear()

    def start(self):
        """Start the WebSocket stream (blocking)."""
        self._running = True
        log.info("Starting bar stream...")
        self.stream.run()

    def start_async(self):
        """Start the WebSocket stream in a background thread."""
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        log.info("Bar stream started in background thread")
        return t

    def stop(self):
        """Stop the stream."""
        self._running = False
        try:
            self.stream.stop()
        except Exception:
            pass
        log.info("Bar stream stopped")
