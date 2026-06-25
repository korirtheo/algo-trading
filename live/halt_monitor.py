"""
Halt-Resume Monitor: polls the NASDAQ public halt log for resume events.

Source: https://www.nasdaqtrader.com/dynamic/symdir/tradehalts.txt  (free, no key)
The file is appended throughout the trading day with halt + resume entries.
We poll every ~POLL_INTERVAL seconds, parse new rows, and fire a callback for
each newly-resumed ticker that passes the eligibility filter.

The same parser is reused by scripts/download/download_halt_history.py so live
and backtest see byte-identical data.
"""
import csv
import io
import logging
import os
import threading
import time as time_mod
import urllib.request
from dataclasses import dataclass
from datetime import datetime, date, time as dt_time
from typing import Callable, Optional
from zoneinfo import ZoneInfo

# 2026-06-24: persist every newly-seen halt to data/halts.csv so we accumulate
# history going forward. The .txt source went dead; rebuilding via daily scrape.
_HALTS_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "halts.csv",
)
_HALTS_CSV_HEADER = [
    "halt_date", "halt_time", "ticker", "reason",
    "resume_time", "resume_quote_time", "resume_trade_time",
    "halt_price", "resume_price",
]
_HALTS_CSV_LOCK = threading.Lock()


def _persist_halt_event(ev: "HaltEvent") -> None:
    """Append a HaltEvent row to data/halts.csv (idempotent on `key`)."""
    try:
        os.makedirs(os.path.dirname(_HALTS_CSV_PATH), exist_ok=True)
        seen = set()
        if os.path.exists(_HALTS_CSV_PATH):
            with open(_HALTS_CSV_PATH, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    seen.add(f"{row['halt_date']}|{row['ticker']}|{row['halt_time']}")
        row_key = f"{ev.halt_date.isoformat()}|{ev.ticker}|{ev.halt_time.isoformat()}"
        if row_key in seen:
            return
        with _HALTS_CSV_LOCK:
            need_header = not os.path.exists(_HALTS_CSV_PATH) or os.path.getsize(_HALTS_CSV_PATH) == 0
            with open(_HALTS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if need_header:
                    w.writerow(_HALTS_CSV_HEADER)
                w.writerow([
                    ev.halt_date.isoformat() if ev.halt_date else "",
                    ev.halt_time.isoformat() if ev.halt_time else "",
                    ev.ticker or "",
                    ev.reason or "",
                    ev.resume_time.isoformat() if ev.resume_time else "",
                    ev.resume_quote_time.isoformat() if ev.resume_quote_time else "",
                    ev.resume_trade_time.isoformat() if ev.resume_trade_time else "",
                    ev.halt_price if ev.halt_price is not None else "",
                    ev.resume_price if ev.resume_price is not None else "",
                ])
    except Exception as e:
        log.warning("Failed to persist halt event %s: %s", ev.ticker, e)

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# NASDAQ swapped the .txt endpoint for an HTML "Page Not Available" placeholder
# sometime before 2026-06-24 — the live monitor had been silently failing for
# weeks. The RSS feed is the only currently-working public source.
NASDAQ_HALT_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
POLL_INTERVAL_SECS = 30           # how often to refetch the halt log
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)


@dataclass
class HaltEvent:
    """One row from the NASDAQ halt log."""
    halt_date: date
    halt_time: dt_time
    ticker: str
    reason: str
    resume_time: Optional[dt_time]    # None until NASDAQ posts the resume
    resume_quote_time: Optional[dt_time]
    resume_trade_time: Optional[dt_time]
    halt_price: Optional[float]
    resume_price: Optional[float]

    @property
    def resumed(self) -> bool:
        return self.resume_trade_time is not None or self.resume_time is not None

    @property
    def resume_dt(self) -> Optional[datetime]:
        t = self.resume_trade_time or self.resume_time
        if t is None:
            return None
        return datetime.combine(self.halt_date, t, tzinfo=ET)

    @property
    def key(self) -> str:
        """Stable identity per halt — ticker + halt time on the day."""
        return f"{self.halt_date}|{self.ticker}|{self.halt_time}"


def _parse_time(s: str) -> Optional[dt_time]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    return None


def _parse_price(s: str) -> Optional[float]:
    s = (s or "").strip().replace("$", "").replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_halt_log(text: str) -> list[HaltEvent]:
    """Parse the NASDAQ trade-halt RSS feed into HaltEvent rows.

    Switched 2026-06-24 from pipe-delimited .txt (dead — returns HTML) to the
    RSS feed at /rss.aspx?feed=tradehalts. Fields live under the ndaq: namespace.
    """
    events: list[HaltEvent] = []
    if not text or "<rss" not in text.lower():
        return events
    import xml.etree.ElementTree as _ETree
    ns = {"ndaq": "http://www.nasdaqtrader.com/"}
    try:
        root = _ETree.fromstring(text)
    except _ETree.ParseError as e:
        log.warning("halt-log RSS parse error: %s", e)
        return events
    for item in root.findall(".//item"):
        def _get(tag):
            el = item.find(f"ndaq:{tag}", ns)
            return (el.text or "").strip() if el is not None and el.text else ""
        d = _parse_date(_get("HaltDate"))
        ht = _parse_time(_get("HaltTime"))
        sym = _get("IssueSymbol")
        if d is None or ht is None or not sym:
            continue
        # Resume date — usually same as halt date; if RSS gives a different one,
        # we still store the resume_time only (HaltEvent's resume_dt builds from
        # halt_date + resume_time, which is fine for same-day resumes).
        events.append(HaltEvent(
            halt_date=d,
            halt_time=ht,
            ticker=sym.upper(),
            reason=_get("ReasonCode"),
            resume_time=_parse_time(_get("ResumptionTradeTime")) or _parse_time(_get("ResumptionQuoteTime")),
            resume_quote_time=_parse_time(_get("ResumptionQuoteTime")),
            resume_trade_time=_parse_time(_get("ResumptionTradeTime")),
            halt_price=_parse_price(_get("PauseThresholdPrice")),
            resume_price=None,  # not present in RSS
        ))
    return events


def fetch_halt_log(timeout: float = 10.0) -> str:
    """Fetch the NASDAQ halt log over HTTPS.

    On Windows the system CA store is often empty, so we fall back to
    certifi's bundle when the default context fails the cert check.
    """
    import ssl
    req = urllib.request.Request(NASDAQ_HALT_URL,
                                  headers={"User-Agent": "halt-monitor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read().decode("utf-8", errors="replace")


class HaltMonitor:
    """Background poller that fires `on_resume(event)` for each newly-resumed
    halt that passes `eligibility_fn`.

    The callback runs on the monitor's poller thread. Wire it to a thread-safe
    handler in main.py.
    """

    def __init__(
        self,
        on_resume: Callable[[HaltEvent], None],
        eligibility_fn: Optional[Callable[[HaltEvent], bool]] = None,
        poll_interval_secs: int = POLL_INTERVAL_SECS,
    ):
        self.on_resume = on_resume
        self.eligibility_fn = eligibility_fn or (lambda e: True)
        self.poll_interval_secs = poll_interval_secs
        self._seen_resumed: set[str] = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.last_poll_ts: Optional[datetime] = None
        self.last_event_count: int = 0
        self.last_error: Optional[str] = None
        self.recent_events: list[HaltEvent] = []   # capped at 200 for dashboard

    def start(self) -> threading.Thread:
        if self._running:
            return self._thread  # type: ignore[return-value]
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="halt-monitor", daemon=True)
        self._thread.start()
        log.info("HaltMonitor started (poll every %ds)", self.poll_interval_secs)
        return self._thread

    def stop(self) -> None:
        self._running = False
        log.info("HaltMonitor stopping")

    def _loop(self) -> None:
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                self.last_error = str(e)
                log.warning("HaltMonitor poll error: %s", e)
            # Sleep in short chunks so stop() reacts quickly
            for _ in range(self.poll_interval_secs):
                if not self._running:
                    return
                time_mod.sleep(1)

    def _poll_once(self) -> None:
        now_et = datetime.now(ET)
        self.last_poll_ts = now_et

        # Skip polling outside market hours — log is only meaningful intraday.
        # Allow a small window before/after to catch early/late updates.
        t = now_et.time()
        if t < dt_time(9, 0) or t > dt_time(16, 30):
            return

        raw = fetch_halt_log()
        events = parse_halt_log(raw)
        self.last_event_count = len(events)
        today = now_et.date()

        # 2026-06-24: persist EVERY event in the feed (not just today's resumed
        # ones) so we accumulate history for backtest. The RSS feed includes
        # multi-year history per poll, so this catches all events we've ever
        # seen across restarts (dedup keyed by halt_date|ticker|halt_time).
        for ev in events:
            _persist_halt_event(ev)

        for ev in events:
            if ev.halt_date != today:
                continue
            if not ev.resumed:
                continue
            if ev.key in self._seen_resumed:
                continue
            # Mark seen BEFORE invoking callback — prevents double-firing
            # if the callback is slow and the next poll fires first.
            self._seen_resumed.add(ev.key)

            # Cache for dashboard
            self.recent_events.append(ev)
            if len(self.recent_events) > 200:
                self.recent_events = self.recent_events[-200:]

            if not self.eligibility_fn(ev):
                log.info("Halt-resume IGNORED (filter): %s reason=%s resume=%s",
                         ev.ticker, ev.reason, ev.resume_dt)
                continue

            log.info("Halt-resume FIRE: %s reason=%s halt_price=%s resume_price=%s @ %s",
                     ev.ticker, ev.reason, ev.halt_price, ev.resume_price, ev.resume_dt)
            try:
                self.on_resume(ev)
            except Exception as e:
                log.exception("on_resume callback raised for %s: %s", ev.ticker, e)

    # ----- helpers exposed for dashboard / introspection -----

    def get_today_events(self) -> list[HaltEvent]:
        today = datetime.now(ET).date()
        return [e for e in self.recent_events if e.halt_date == today]
