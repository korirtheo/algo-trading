"""
Real-Time Bar Streamer: Alpaca WebSocket -> 2-minute candle aggregation.

Subscribes to 1-minute bars for watchlist symbols via Alpaca's data stream.
Aggregates into 2-minute candles aligned to market open (9:30 ET).
Emits completed 2-min bars to callback.
"""
import logging
import threading
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from collections import defaultdict

from alpaca.data.enums import DataFeed
from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED

_FEED_ENUM = DataFeed.IEX if ALPACA_FEED == "iex" else DataFeed.SIP

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

    def subscribe(self, symbols):
        """Subscribe to 1-min bars for given symbols."""
        self._symbols = list(symbols)
        log.info(f"Subscribing to {len(self._symbols)} symbols: {self._symbols}")

        async def _on_bar(bar):
            self._handle_bar(bar)

        self.stream.subscribe_bars(_on_bar, *self._symbols)

    def _handle_bar(self, bar):
        """Process incoming 1-min bar, aggregate to 2-min."""
        symbol = bar.symbol
        log.info(f"1min bar: {symbol} close={bar.close:.2f} vol={bar.volume:,} t={bar.timestamp}")
        slot = _bar_slot(bar.timestamp)

        if symbol in self.pending:
            prev_slot, prev_bar = self.pending[symbol]
            if slot == prev_slot:
                # Same 2-min window — merge
                prev_bar.merge(bar)
                if prev_bar.count >= 2:
                    # 2-min bar complete
                    self.on_2min_bar(symbol, prev_bar.to_dict())
                    del self.pending[symbol]
                return
            else:
                # New slot — emit previous bar (even if only 1 min) and start new
                self.on_2min_bar(symbol, prev_bar.to_dict())

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
