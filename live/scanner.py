"""
Pre-Market Scanner: Identifies gap-up candidates before market open.

Runs from ~8:00 AM ET to 9:25 AM ET.
Uses Alpaca REST API to fetch pre-market bars and compute:
  - Gap % (current price vs previous close)
  - Pre-market volume
  - Pre-market high

Produces a watchlist for the strategy engine.
"""
import re
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from config.settings import (
    ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED,
    MIN_GAP_PCT, TOP_N, MIN_PM_VOLUME, MAX_PRICE, FLOAT_DATA,
)

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def _is_warrant_or_unit(ticker):
    if ".WS" in ticker or ".RT" in ticker:
        return True
    if re.match(r"^[A-Z]{3,}W$", ticker):
        return True
    if ticker.endswith("WW"):
        return True
    if re.match(r"^[A-Z]{3,}U$", ticker):
        return True
    if re.match(r"^[A-Z]{3,}R$", ticker):
        return True
    return False


class PreMarketScanner:
    def __init__(self, min_gap_pct=None, max_float=None):
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
        self.min_gap_pct = min_gap_pct or MIN_GAP_PCT
        self.max_float = max_float or 15_000_000

    def get_tradeable_tickers(self):
        """Get all active US equity tickers."""
        assets = self.trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        tickers = [
            a.symbol for a in assets
            if a.tradable
            and not _is_warrant_or_unit(a.symbol)
            and "." not in a.symbol
            and len(a.symbol) <= 5
        ]
        log.info(f"Found {len(tickers)} tradeable US equities")
        return tickers

    def get_previous_closes(self, tickers, date=None):
        """Get previous day's close for a list of tickers."""
        if date is None:
            date = datetime.now(ET).date()

        prev_closes = {}
        batch_size = 500
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=datetime.combine(date - timedelta(days=5), datetime.min.time()),
                    end=datetime.combine(date, datetime.min.time()),
                    adjustment="raw",
                    feed=ALPACA_FEED,
                    limit=5,
                )
                bars = self.data_client.get_stock_bars(req)
                if not bars.df.empty:
                    df = bars.df.reset_index()
                    for ticker in df["symbol"].unique():
                        tdf = df[df["symbol"] == ticker].sort_values("timestamp")
                        if len(tdf) > 0:
                            prev_closes[ticker] = float(tdf.iloc[-1]["close"])
            except Exception as e:
                log.warning(f"Error fetching daily bars batch {i}: {e}")
            time.sleep(0.35)

        log.info(f"Got previous closes for {len(prev_closes)} tickers")
        return prev_closes

    def scan_premarket(self, tickers=None, prev_closes=None):
        """Scan for gap-up candidates with pre-market activity.

        Returns list of dicts:
        [{"ticker", "gap_pct", "pm_volume", "premarket_high", "prev_close", "float_shares"}, ...]
        """
        now = datetime.now(ET)
        today = now.date()

        if tickers is None:
            tickers = self.get_tradeable_tickers()
        if prev_closes is None:
            prev_closes = self.get_previous_closes(tickers, today)

        # Filter tickers by price range (float filtering done by strategy engine)
        candidates_to_check = []
        for t in tickers:
            if t not in prev_closes:
                continue
            pc = prev_closes[t]
            if pc <= 0 or pc > MAX_PRICE:
                continue
            candidates_to_check.append(t)

        log.info(f"Checking {len(candidates_to_check)} tickers for pre-market gaps")

        # Fetch pre-market bars for candidates
        results = []
        batch_size = 100
        market_open = datetime.combine(today, datetime.min.time().replace(hour=4))  # 4 AM ET

        for i in range(0, len(candidates_to_check), batch_size):
            batch = candidates_to_check[i:i + batch_size]
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Minute,
                    start=market_open,
                    end=now,
                    adjustment="raw",
                    feed=ALPACA_FEED,
                )
                bars = self.data_client.get_stock_bars(req)
                if bars.df.empty:
                    continue

                df = bars.df.reset_index()
                for ticker in df["symbol"].unique():
                    tdf = df[df["symbol"] == ticker]
                    if len(tdf) == 0:
                        continue

                    pc = prev_closes[ticker]
                    current_price = float(tdf.iloc[-1]["close"])
                    gap_pct = (current_price / pc - 1) * 100

                    if gap_pct < self.min_gap_pct:
                        continue

                    pm_volume = int(tdf["volume"].sum())
                    pm_high = float(tdf["high"].max())
                    float_shares = FLOAT_DATA.get(ticker)

                    results.append({
                        "ticker": ticker,
                        "gap_pct": gap_pct,
                        "pm_volume": pm_volume,
                        "premarket_high": pm_high,
                        "prev_close": pc,
                        "float_shares": float_shares,
                        "current_price": current_price,
                    })

            except Exception as e:
                log.warning(f"Error scanning batch {i}: {e}")
            time.sleep(0.35)

        # Filter by PM volume and sort by gap %
        results = [r for r in results if r["pm_volume"] >= MIN_PM_VOLUME]
        results.sort(key=lambda x: x["gap_pct"], reverse=True)
        results = results[:TOP_N]

        log.info(f"Found {len(results)} gap-up candidates")
        for r in results:
            log.info(f"  {r['ticker']}: gap={r['gap_pct']:.1f}%, PM vol={r['pm_volume']:,}, "
                     f"float={r['float_shares']/1e6:.1f}M, PM high=${r['premarket_high']:.2f}")

        return results

    def get_final_watchlist(self):
        """Full scan workflow: get tickers -> prev closes -> scan premarket."""
        log.info("Starting pre-market scan...")
        tickers = self.get_tradeable_tickers()
        prev_closes = self.get_previous_closes(tickers)
        candidates = self.scan_premarket(tickers, prev_closes)
        return candidates
