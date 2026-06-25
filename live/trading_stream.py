"""Alpaca TradingStream wrapper for real-time fill notifications.

Replaces the polling-based fill detection (which times out at 15s on thin
microcaps causing the engine to miss real fills and fire duplicate signals).

Usage:
    fs = FillStream(api_key, secret, paper=True)
    fs.register(order_id, callback)  # callback(event_type, order_data)
    fs.start_async()
"""
import asyncio
import logging
import threading

from alpaca.trading.stream import TradingStream

log = logging.getLogger(__name__)


class FillStream:
    """Wraps Alpaca's TradingStream and dispatches events to per-order callbacks."""

    # Events that indicate the order journey is complete (won't fire again for same id)
    TERMINAL_EVENTS = {"fill", "canceled", "rejected", "expired", "done_for_day"}

    def __init__(self, api_key, secret_key, paper=True):
        self.stream = TradingStream(api_key, secret_key, paper=paper)
        self._handlers = {}   # order_id (str) -> callback
        self._lock = threading.Lock()
        self.stream.subscribe_trade_updates(self._on_update)
        self._thread = None
        self._started = False

    def register(self, order_id, callback):
        """Register a callback for events on this order_id.

        Callback signature: callback(event_type: str, order_data) -> None
        Order_data has .symbol, .filled_qty, .filled_avg_price, .status, etc.
        """
        oid = str(order_id)
        with self._lock:
            self._handlers[oid] = callback

    def unregister(self, order_id):
        """Remove the callback for this order_id."""
        oid = str(order_id)
        with self._lock:
            self._handlers.pop(oid, None)

    async def _on_update(self, data):
        """Called by Alpaca TradingStream for every trade event."""
        try:
            event = getattr(data, "event", None)
            order = getattr(data, "order", None)
            if event is None or order is None:
                return
            event_str = str(event).lower().replace("tradeevent.", "")
            order_id = str(getattr(order, "id", ""))
            log.debug(f"FillStream: event={event_str} order={order_id} "
                      f"symbol={getattr(order, 'symbol', '?')} "
                      f"filled_qty={getattr(order, 'filled_qty', '?')} "
                      f"filled_avg_price={getattr(order, 'filled_avg_price', '?')}")
            with self._lock:
                cb = self._handlers.get(order_id)
                # Pop on terminal events so registry doesn't leak
                if cb is not None and event_str in self.TERMINAL_EVENTS:
                    self._handlers.pop(order_id, None)
            if cb is not None:
                try:
                    cb(event_str, order)
                except Exception as e:
                    log.error(f"FillStream callback failed for {order_id}: {e}", exc_info=True)
        except Exception as e:
            log.error(f"FillStream._on_update fatal: {e}", exc_info=True)

    def start_async(self):
        """Start the stream in a background daemon thread."""
        if self._started:
            log.warning("FillStream already started")
            return
        self._started = True

        def _run():
            log.info("FillStream: starting Alpaca TradingStream...")
            try:
                self.stream.run()
            except Exception as e:
                log.error(f"FillStream.run died: {e}", exc_info=True)

        self._thread = threading.Thread(target=_run, daemon=True, name="alpaca-trading-stream")
        self._thread.start()
        log.info("FillStream: background thread started")
        return self._thread

    def stop(self):
        try:
            self.stream.stop()
        except Exception:
            pass
        log.info("FillStream: stopped")

    def pending_count(self):
        """Number of orders currently being tracked."""
        with self._lock:
            return len(self._handlers)
