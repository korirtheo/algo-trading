"""
Intraday top-gainers discovery scanner.
Polls Alpaca's market-movers API periodically and injects new discoveries
into the engine's watchlist.
"""

import logging
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.data.enums import MarketType
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MarketMoversRequest

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET
from live.event_logger import log_event

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

MIN_PCT_CHANGE = 20.0
MIN_PRICE = 1.0
MAX_PRICE = 50.0
TOP_N_PER_POLL = 50


def _is_warrant_or_unit(ticker):
    """Returns True if the ticker is a warrant or unit, which should be excluded."""
    return (
        ".WS" in ticker
        or ".RT" in ticker
        or re.match(r"^[A-Z]{3,}W$", ticker)
        or ticker.endswith("WW")
        or re.match(r"^[A-Z]{3,}U$", ticker)
        or re.match(r"^[A-Z]{3,}R$", ticker)
        or "." in ticker
        or len(ticker) > 5
    )


class IntradayScanner:
    """
    Scans for and identifies new intraday top-gaining stocks.
    """

    def __init__(self, engine, poll_seconds=60):
        self.client = ScreenerClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        self.engine = engine
        self.poll_seconds = poll_seconds
        self.last_poll_time = 0
        self.discovered_today = set()

    def poll(self):
        """
        Polls the market movers API and notifies the engine of new discoveries.
        """
        now = time.time()
        if now - self.last_poll_time < self.poll_seconds:
            return

        self.last_poll_time = now
        now_dt = datetime.fromtimestamp(now, ET)

        # Reset discovered list each day
        if now_dt.date() != datetime.fromtimestamp(self.last_poll_time, ET).date():
            self.discovered_today = set()

        try:
            req = MarketMoversRequest(top=TOP_N_PER_POLL, market_type=MarketType.STOCKS)
            movers = self.client.get_market_movers(req)

            for g in movers.gainers or []:
                ticker = g.symbol
                if (
                    _is_warrant_or_unit(ticker)
                    or ticker in self.discovered_today
                    or ticker in self.engine.active_positions
                ):
                    continue

                price = float(g.price)
                pct_change = float(g.percent_change)

                if pct_change >= MIN_PCT_CHANGE and MIN_PRICE <= price <= MAX_PRICE:
                    log.info(
                        f"Intraday Gainer Discovery: {ticker} at {price} (+{pct_change}%)"
                    )
                    self.discovered_today.add(ticker)
                    self.engine.on_intraday_addition(
                        ticker,
                        {
                            "price": price,
                            "percent_change": pct_change,
                            "timestamp": now_dt.isoformat(),
                        },
                        source="intraday_gainer",
                    )
        except Exception as e:
            log.error(f"Failed to get intraday market movers: {e}")
            log_event("intraday_scan_error", "error", f"Failed to get intraday market movers: {e}")
