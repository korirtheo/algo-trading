"""
Pre-Market Scanner: Identifies gap-up candidates before market open.

Uses Alpaca's screener/movers endpoint + Finviz pre-market screener to get
top % gainers, then fetches PM bars for premarket high.

Produces a watchlist for the strategy engine.
"""
import re
import logging
import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from config.settings import (
    ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED,
    MIN_GAP_PCT, TOP_N, MAX_PRICE, FLOAT_DATA,
)

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

DATA_BASE_URL = "https://data.alpaca.markets"

# IEX feed covers ~2.5% of real volume — PM volume filter is not meaningful
# We rely on gap% and the movers source for quality filtering
LIVE_MIN_PM_VOLUME = 1_000


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

    def get_movers(self):
        """Get top % gainers from Alpaca screener endpoint."""
        url = f"{DATA_BASE_URL}/v1beta1/screener/stocks/movers"
        try:
            resp = requests.get(url, headers=_alpaca_headers(), params={}, timeout=10)
            resp.raise_for_status()
            gainers = resp.json().get("gainers", [])
            log.info(f"Alpaca movers: {len(gainers)} gainers returned")
            for g in gainers:
                log.debug(f"  mover: {g.get('symbol')} price=${g.get('price')} chg={g.get('percent_change')}%")
            return gainers
        except Exception as e:
            log.warning(f"Alpaca movers failed: {e}")
            return []

    def get_finviz_premarket(self):
        """Scrape Finviz screener for pre-market gap-up candidates."""
        try:
            url = "https://finviz.com/screener.ashx"
            params = {
                "v": "111",
                "f": f"sh_price_u{int(MAX_PRICE)}",
                "o": "-prechange",
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()

            tables = pd.read_html(StringIO(resp.text))
            df = None
            for t in tables:
                cols = [str(c).strip() for c in t.columns]
                if "Ticker" in cols:
                    df = t
                    df.columns = cols
                    break

            if df is None:
                log.debug("Finviz: screener table not found in response")
                return []

            movers = []
            for _, row in df.iterrows():
                try:
                    ticker = str(row.get("Ticker", "")).strip()
                    if not ticker or ticker.lower() == "nan" or len(ticker) > 5 or "." in ticker:
                        continue
                    if _is_warrant_or_unit(ticker):
                        continue

                    price = float(str(row.get("Price", 0)).replace(",", ""))
                    if price <= 0 or price > MAX_PRICE:
                        continue

                    change_str = str(row.get("Change", "0%")).replace("%", "").replace("+", "").strip()
                    change_pct = float(change_str)

                    if change_pct < self.min_gap_pct:
                        break  # sorted descending — can stop here

                    movers.append({
                        "symbol": ticker,
                        "price": price,
                        "percent_change": change_pct,
                    })
                except (ValueError, TypeError):
                    continue

            log.info(f"Finviz pre-market: {len(movers)} candidates found")
            return movers

        except Exception as e:
            log.warning(f"Finviz scrape failed: {e}")
            return []

    def scan_premarket(self, movers):
        """Fetch PM bars for movers to get premarket high.

        Returns list of candidate dicts for the strategy engine.
        Note: IEX feed volumes are ~2.5% of real volume, so PM volume
        filter uses a very low threshold (LIVE_MIN_PM_VOLUME).
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
            bars_by_ticker = {}
            if not bars.df.empty:
                df = bars.df.reset_index()
                for ticker in df["symbol"].unique():
                    tdf = df[df["symbol"] == ticker]
                    bars_by_ticker[ticker] = {
                        "pm_volume": int(tdf["volume"].sum()),
                        "pm_high": float(tdf["high"].max()),
                    }
        except Exception as e:
            log.warning(f"Error fetching PM bars: {e}")
            bars_by_ticker = {}

        for ticker, info in ticker_map.items():
            bar_info = bars_by_ticker.get(ticker, {})
            pm_volume = bar_info.get("pm_volume", 0)
            pm_high = bar_info.get("pm_high", info["current_price"])

            if pm_volume < LIVE_MIN_PM_VOLUME:
                log.debug(f"  {ticker} filtered out: IEX PM vol={pm_volume:,} < {LIVE_MIN_PM_VOLUME:,}")
                continue

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

        results.sort(key=lambda x: x["gap_pct"], reverse=True)
        results = results[:TOP_N]

        log.info(f"Found {len(results)} gap-up candidates")
        for r in results:
            float_str = f"{r['float_shares']/1e6:.1f}M" if r.get("float_shares") else "N/A"
            log.info(f"  {r['ticker']}: gap={r['gap_pct']:.1f}%, IEX PM vol={r['pm_volume']:,}, "
                     f"float={float_str}, PM high=${r['premarket_high']:.2f}")

        return results

    def get_final_watchlist(self):
        """Get pre-market gap-up candidates from Alpaca movers + Finviz screener."""
        log.info("Starting pre-market scan (Alpaca + Finviz)...")

        # Gather from both sources
        alpaca_movers = self.get_movers()
        finviz_movers = self.get_finviz_premarket()

        # Merge, deduplicating by symbol (Alpaca takes priority for gap_pct accuracy)
        seen = {m["symbol"] for m in alpaca_movers}
        combined = list(alpaca_movers)
        for m in finviz_movers:
            if m["symbol"] not in seen:
                combined.append(m)
                seen.add(m["symbol"])

        if not combined:
            log.warning("No movers returned from Alpaca or Finviz")
            return []

        log.info(f"Combined universe: {len(combined)} tickers (Alpaca + Finviz)")
        return self.scan_premarket(combined)
