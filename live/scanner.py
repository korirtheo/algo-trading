"""
Pre-Market Scanner: Identifies gap-up candidates before market open.

Uses Alpaca's screener/movers endpoint to get top % gainers instantly,
then fetches PM bars for volume and premarket high.

Produces a watchlist for the strategy engine.
"""
import re
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from config.settings import (
    ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED,
    MIN_GAP_PCT, TOP_N, MIN_PM_VOLUME, MAX_PRICE, FLOAT_DATA,
)

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

DATA_BASE_URL = "https://data.alpaca.markets"


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


def _alpaca_headers():
    return {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}


class PreMarketScanner:
    def __init__(self, min_gap_pct=None, max_float=None):
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
        self.min_gap_pct = min_gap_pct or MIN_GAP_PCT
        self.max_float = max_float or 15_000_000

    def get_movers(self, top=100):
        """Get top % gainers from Alpaca screener endpoint (instant, no scanning)."""
        url = f"{DATA_BASE_URL}/v1beta1/screener/stocks/movers"
        params = {}
        try:
            resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            gainers = data.get("gainers", [])
            log.info(f"Alpaca movers: {len(gainers)} gainers returned")
            return gainers
        except Exception as e:
            log.warning(f"Movers endpoint failed: {e}")
            return []

    def scan_premarket(self, movers):
        """Fetch PM bars for movers to get volume and premarket high.

        Returns list of candidate dicts for the strategy engine.
        """
        now = datetime.now(ET)
        today = now.date()
        market_open_4am = datetime.combine(today, datetime.min.time().replace(hour=4)).replace(tzinfo=ET)

        # Build ticker -> mover info map, filtering by gap % and price
        ticker_map = {}
        for m in movers:
            ticker = m.get("symbol", "")
            if not ticker or "." in ticker or len(ticker) > 5 or _is_warrant_or_unit(ticker):
                continue
            price = m.get("price", 0)
            change_pct = m.get("percent_change", 0)
            prev_close = price / (1 + change_pct / 100) if change_pct != -100 else 0
            if price <= 0 or price > MAX_PRICE:
                continue
            if change_pct < self.min_gap_pct:
                continue
            ticker_map[ticker] = {
                "gap_pct": change_pct,
                "current_price": price,
                "prev_close": prev_close,
            }

        if not ticker_map:
            log.info("No tickers passed gap/price filters from movers")
            return []

        log.info(f"Fetching PM bars for {len(ticker_map)} candidates...")

        # Fetch PM bars in one batch
        results = []
        tickers = list(ticker_map.keys())
        try:
            req = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=TimeFrame.Minute,
                start=market_open_4am,
                end=now,
                adjustment="raw",
                feed=ALPACA_FEED,
            )
            bars = self.data_client.get_stock_bars(req)
            if not bars.df.empty:
                df = bars.df.reset_index()
                for ticker in df["symbol"].unique():
                    tdf = df[df["symbol"] == ticker]
                    pm_volume = int(tdf["volume"].sum())
                    pm_high = float(tdf["high"].max())
                    info = ticker_map[ticker]
                    float_shares = FLOAT_DATA.get(ticker)
                    results.append({
                        "ticker": ticker,
                        "gap_pct": info["gap_pct"],
                        "pm_volume": pm_volume,
                        "premarket_high": pm_high,
                        "prev_close": info["prev_close"],
                        "float_shares": float_shares,
                        "current_price": info["current_price"],
                    })
        except Exception as e:
            log.warning(f"Error fetching PM bars: {e}")
            # Fall back: return movers without PM bar data
            for ticker, info in ticker_map.items():
                float_shares = FLOAT_DATA.get(ticker)
                results.append({
                    "ticker": ticker,
                    "gap_pct": info["gap_pct"],
                    "pm_volume": MIN_PM_VOLUME,  # assume threshold met
                    "premarket_high": info["current_price"],
                    "prev_close": info["prev_close"],
                    "float_shares": float_shares,
                    "current_price": info["current_price"],
                })

        # Filter by PM volume and sort by gap %
        results = [r for r in results if r["pm_volume"] >= MIN_PM_VOLUME]
        results.sort(key=lambda x: x["gap_pct"], reverse=True)
        results = results[:TOP_N]

        log.info(f"Found {len(results)} gap-up candidates")
        for r in results:
            float_str = f"{r['float_shares']/1e6:.1f}M" if r.get("float_shares") else "N/A"
            log.info(f"  {r['ticker']}: gap={r['gap_pct']:.1f}%, PM vol={r['pm_volume']:,}, "
                     f"float={float_str}, PM high=${r['premarket_high']:.2f}")

        return results

    def get_final_watchlist(self):
        """Get top gainers via Alpaca movers endpoint, then enrich with PM bars."""
        log.info("Starting pre-market scan (Alpaca movers)...")
        movers = self.get_movers(top=100)
        if not movers:
            log.warning("No movers returned — market may be closed or subscription issue")
            return []
        candidates = self.scan_premarket(movers)
        return candidates
