"""
Tradier SSE streaming feed — aggregates real-time trade ticks into 2-min bars.

Flow:
  1. POST /v1/markets/events/session  → get a streaming session token
  2. GET  stream.tradier.com/v1/markets/events?sessionid=...&symbols=...&filter=trade
  3. Parse SSE `trade` events  → accumulate ticks into TwoMinBar buckets
  4. Emit completed 2-min bar via on_2min_bar(symbol, bar_dict) — same
     signature as BarStreamer in streamer.py so wiring is a drop-in swap.

Thread model:
  start_async() → background daemon thread runs _stream_loop() which
  reconnects automatically on disconnect.
"""
import json
import logging
import threading
import time
import traceback
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from live.event_logger import log_event

import requests

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
MARKET_OPEN = dt_time(9, 30)

TRADIER_API_BASE = "https://api.tradier.com"
TRADIER_STREAM_BASE = "https://stream.tradier.com"
SESSION_URL = f"{TRADIER_API_BASE}/v1/markets/events/session"
EVENTS_URL = f"{TRADIER_STREAM_BASE}/v1/markets/events"


def _bar_slot(ts: datetime) -> int:
    """Return 2-min slot index from an ET datetime (same logic as streamer.py)."""
    et = ts.astimezone(ET)
    minutes_since_open = (et.hour * 60 + et.minute) - (9 * 60 + 30)
    return minutes_since_open // 2


class _TwoMinBar:
    """Accumulates ticks for a single 2-min slot."""
    __slots__ = ("open", "high", "low", "close", "volume", "timestamp", "tick_count")

    def __init__(self, price: float, size: int, ts: datetime):
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = size
        self.timestamp = ts
        self.tick_count = 1

    def add_tick(self, price: float, size: int):
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.tick_count += 1

    def to_dict(self) -> dict:
        return {
            "Open": self.open,
            "High": self.high,
            "Low": self.low,
            "Close": self.close,
            "Volume": self.volume,
            "timestamp": self.timestamp,
        }


class TradierStreamer:
    """
    Streams real-time trade ticks from Tradier and emits 2-min bars.

    Identical on_2min_bar callback signature to BarStreamer (streamer.py).
    """

    def __init__(self, api_key: str, on_2min_bar):
        self.api_key = api_key
        self.on_2min_bar = on_2min_bar
        self._symbols: list[str] = []
        self._pending: dict[str, tuple[int, _TwoMinBar]] = {}
        self._session_id: str | None = None
        self._running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------ #
    # Public API (mirrors BarStreamer)
    # ------------------------------------------------------------------ #

    def subscribe(self, symbols: list[str]):
        """Set initial symbol list (call before start_async)."""
        with self._lock:
            self._symbols = [s.upper() for s in symbols]
        log.info(f"TradierStreamer: queued {len(self._symbols)} symbols")

    def add_symbol(self, symbol: str) -> bool:
        """Add a symbol mid-stream (reconnects the SSE session)."""
        symbol = symbol.upper()
        with self._lock:
            if symbol in self._symbols:
                return False
            self._symbols.append(symbol)
        log.info(f"TradierStreamer: added {symbol}, will include on next SSE session")
        # Force reconnect so new symbol appears in the stream
        self._session_id = None
        return True

    def remove_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self._symbols:
                return False
            self._symbols.remove(symbol)
        self._session_id = None
        return True

    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._symbols)

    def flush_pending(self):
        """Emit any partially-accumulated bars (call at EOD)."""
        with self._lock:
            pending = dict(self._pending)
            self._pending.clear()
        for symbol, (_, bar) in pending.items():
            try:
                self.on_2min_bar(symbol, bar.to_dict())
            except Exception as e:
                log.warning(f"TradierStreamer flush error for {symbol}: {e}")
                log_event("tradier_stream_error", "warning", f"TradierStreamer flush error for {symbol}: {e}")

    def start_async(self) -> threading.Thread:
        self._running = True
        t = threading.Thread(target=self._stream_loop, daemon=True, name="tradier-stream")
        self._thread = t
        t.start()
        log.info("TradierStreamer: background thread started")
        return t

    def stop(self):
        self._running = False
        log.info("TradierStreamer: stop requested")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _get_session(self) -> str | None:
        """Obtain a streaming session token from Tradier."""
        try:
            resp = requests.post(SESSION_URL, headers=self._headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            session_id = data["stream"]["sessionid"]
            log.info(f"TradierStreamer: got session {session_id[:8]}...")
            return session_id
        except Exception as e:
            log.error(f"TradierStreamer: failed to get session: {e}")
            log_event("tradier_session_error", "error", f"TradierStreamer: failed to get session: {e}")
            return None

    def _stream_loop(self):
        """Main reconnect loop — runs forever until stop() is called."""
        backoff = 5
        while self._running:
            with self._lock:
                symbols = list(self._symbols)

            if not symbols:
                time.sleep(2)
                continue

            session_id = self._get_session()
            if not session_id:
                log.warning(f"TradierStreamer: session failed, retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            self._session_id = session_id
            backoff = 5  # reset on successful session

            params = {
                "sessionid": session_id,
                "symbols": ",".join(symbols),
                "filter": "trade",
                "linebreak": "true",
            }

            try:
                log.info(f"TradierStreamer: opening SSE for {len(symbols)} symbols")
                with requests.get(
                    EVENTS_URL,
                    params=params,
                    headers={**self._headers, "Accept": "application/json"},
                    stream=True,
                    timeout=(10, 60),
                ) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not self._running:
                            break
                        if not raw_line:
                            continue
                        try:
                            self._handle_line(raw_line)
                        except Exception as e:
                            log.warning(f"TradierStreamer: line parse error: {e}")

            except requests.exceptions.ReadTimeout:
                log.warning("TradierStreamer: SSE read timeout, reconnecting")
                log_event("tradier_stream_error", "warning", "TradierStreamer: SSE read timeout, reconnecting")
            except Exception as e:
                if self._running:
                    log.error(f"TradierStreamer: SSE error: {e}, reconnecting in {backoff}s")
                    log_event("tradier_stream_error", "warning", f"TradierStreamer: SSE error: {e}, reconnecting in {backoff}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)

    def _handle_line(self, raw_line: bytes | str):
        """Parse one SSE JSON line and accumulate into 2-min bar."""
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8").strip()
        if not raw_line:
            return

        event = json.loads(raw_line)
        if event.get("type") != "trade":
            return

        symbol = event.get("symbol", "").upper()
        if not symbol:
            return

        try:
            price = float(event["price"])
            size = int(event.get("size", 0))
            # Tradier uses "date" field (epoch milliseconds), not "timestamp"
            ts_ms = int(event["date"])
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=ET)
        except (KeyError, ValueError, TypeError) as e:
            log.debug(f"TradierStreamer: bad tick fields: {e} raw={raw_line[:120]}")
            return

        # Only process market-hours ticks
        et_time = ts.time()
        if et_time < MARKET_OPEN or et_time >= dt_time(16, 0):
            return

        slot = _bar_slot(ts)
        log.debug(f"TradierStreamer tick: {symbol} ${price:.2f} x{size} slot={slot}")

        with self._lock:
            if symbol in self._pending:
                prev_slot, bar = self._pending[symbol]
                if slot == prev_slot:
                    bar.add_tick(price, size)
                    return
                else:
                    # Slot rolled — emit previous bar
                    completed = bar
                    self._pending[symbol] = (slot, _TwoMinBar(price, size, ts))
            else:
                self._pending[symbol] = (slot, _TwoMinBar(price, size, ts))
                return

        # Emit outside the lock
        try:
            self.on_2min_bar(symbol, completed.to_dict())
        except Exception as e:
            log.warning(f"TradierStreamer on_2min_bar error for {symbol}: {e}\n{traceback.format_exc()}")
            log_event("tradier_stream_error", "warning", f"TradierStreamer on_2min_bar error for {symbol}: {e}")
