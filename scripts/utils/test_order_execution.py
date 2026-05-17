"""
Order Execution Test
====================
1. Wait 1 minute after launch
2. Buy 10 shares of the current top Webull gainer
3. Hold 5 minutes
4. Sell entire position
5. Report result

Usage:
  python test_order_execution.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import time
import requests
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SHARES            = 10
HOLD_SECONDS      = 5 * 60   # 5 minutes
WAIT_BEFORE_START = 60       # 1 minute after launch

ET = ZoneInfo("America/New_York")


def get_top_gainer():
    """Get the #1 pre-market/intraday gainer from Webull."""
    url = "https://quotes-gw.webullfintech.com/api/wlas/ranking/topGainers"
    params = {"regionId": 6, "rankType": "preMarket", "pageIndex": 1, "pageSize": 10}
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        for item in resp.json().get("data", []):
            t = item.get("ticker", {})
            v = item.get("values", {})
            symbol = t.get("symbol", "")
            if not symbol or len(symbol) > 5 or "." in symbol:
                continue
            price = float(v.get("price", 0))
            change_pct = float(v.get("changeRatio", 0)) * 100
            if price <= 0 or price > 50:
                continue
            return symbol, price, change_pct
    except Exception as e:
        log.error("Webull fetch failed: %s", e)
    return None, None, None


def get_current_price(client, symbol):
    """Get current price from Alpaca position or quote."""
    try:
        positions = client.get_all_positions()
        for p in positions:
            if p.symbol == symbol:
                return float(p.current_price)
    except Exception:
        pass
    return None


def main():
    client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

    # Verify account
    acct = client.get_account()
    log.info("Account: cash=$%.2f  buying_power=$%.2f", float(acct.cash), float(acct.buying_power))

    # Step 1: Wait 1 minute
    log.info("Waiting %d seconds before placing order...", WAIT_BEFORE_START)
    time.sleep(WAIT_BEFORE_START)

    # Step 2: Get top gainer
    symbol, price, change_pct = get_top_gainer()
    if not symbol:
        log.error("Could not get top gainer from Webull. Aborting.")
        return

    log.info("Top gainer: %s  price=$%.3f  change=+%.1f%%", symbol, price, change_pct)

    # Step 3: Buy 10 shares
    log.info("Placing BUY order: %d shares of %s", SHARES, symbol)
    try:
        buy_order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=SHARES,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        log.info("BUY ORDER SUBMITTED: id=%s  status=%s", buy_order.id, buy_order.status)
    except Exception as e:
        log.error("BUY ORDER FAILED: %s", e)
        return

    # Step 4: Wait for fill confirmation (up to 30s)
    log.info("Waiting for fill...")
    filled_price = None
    for _ in range(15):
        time.sleep(2)
        try:
            order = client.get_order_by_id(buy_order.id)
            log.info("  order status: %s  filled_qty=%s  filled_avg=%s",
                     order.status, order.filled_qty, order.filled_avg_price)
            if str(order.status) in ("filled", "partially_filled"):
                filled_price = float(order.filled_avg_price) if order.filled_avg_price else price
                log.info("FILLED: %s shares @ $%.3f", order.filled_qty, filled_price)
                break
        except Exception as e:
            log.warning("Order status check failed: %s", e)

    if filled_price is None:
        log.warning("Order not filled after 30s — proceeding anyway")
        filled_price = price

    # Step 5: Hold 5 minutes
    log.info("Holding for %d minutes...", HOLD_SECONDS // 60)
    time.sleep(HOLD_SECONDS)

    # Step 6: Get current price
    current = get_current_price(client, symbol)
    if current:
        pnl = (current - filled_price) * SHARES
        pct = (current / filled_price - 1) * 100
        log.info("Current price: $%.3f  (entry=$%.3f  unrealized PnL=$%+.2f  %+.2f%%)",
                 current, filled_price, pnl, pct)

    # Step 7: Sell
    log.info("Placing SELL order: closing %s position", symbol)
    try:
        sell_order = client.close_position(symbol)
        log.info("SELL ORDER SUBMITTED: id=%s  status=%s", sell_order.id, sell_order.status)
    except Exception as e:
        log.error("SELL ORDER FAILED: %s", e)
        return

    # Step 8: Wait for sell fill
    for _ in range(15):
        time.sleep(2)
        try:
            order = client.get_order_by_id(sell_order.id)
            log.info("  sell status: %s  filled_qty=%s  filled_avg=%s",
                     order.status, order.filled_qty, order.filled_avg_price)
            if str(order.status) in ("filled", "partially_filled"):
                sell_price = float(order.filled_avg_price) if order.filled_avg_price else (current or price)
                pnl = (sell_price - filled_price) * SHARES
                pct = (sell_price / filled_price - 1) * 100
                log.info("=" * 50)
                log.info("RESULT: %s  buy=$%.3f  sell=$%.3f", symbol, filled_price, sell_price)
                log.info("PnL: $%+.2f  (%+.2f%%)  on %d shares", pnl, pct, SHARES)
                log.info("=" * 50)
                break
        except Exception as e:
            log.warning("Sell status check failed: %s", e)


if __name__ == "__main__":
    main()
