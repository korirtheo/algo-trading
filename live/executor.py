"""
Order Executor: Places and manages orders via Alpaca Trading API.

Handles:
  - Market buy orders for entries
  - Stop/limit orders for exits
  - Position tracking
  - EOD forced close
"""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER, VOL_CAP_PCT

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class OrderExecutor:
    def __init__(self):
        self.client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
        self.positions = {}  # ticker -> position info
        self.pending_orders = {}  # ticker -> order info

    def get_account(self):
        """Get current account info."""
        return self.client.get_account()

    def get_buying_power(self):
        """Get available buying power."""
        acct = self.get_account()
        return float(acct.cash)

    def get_positions(self):
        """Get all open positions from Alpaca."""
        return self.client.get_all_positions()

    def buy(self, ticker, dollar_amount, current_price, cumulative_volume=0):
        """Place a market buy order.

        Args:
            ticker: stock symbol
            dollar_amount: dollar amount to invest
            current_price: approximate current price (for share calculation)
            cumulative_volume: total volume traded so far (for vol cap)

        Returns:
            order object or None if rejected
        """
        # Volume cap check
        if VOL_CAP_PCT > 0 and cumulative_volume > 0:
            dollar_vol = current_price * cumulative_volume
            vol_limit = dollar_vol * (VOL_CAP_PCT / 100)
            if vol_limit > 0 and dollar_amount > vol_limit:
                log.info(f"Vol cap: {ticker} limited from ${dollar_amount:,.0f} to ${vol_limit:,.0f}")
                dollar_amount = vol_limit
            if dollar_amount < 50:
                log.info(f"Skip {ticker}: vol-capped amount ${dollar_amount:.0f} too small")
                return None

        # Calculate shares (Alpaca supports fractional)
        shares = round(dollar_amount / current_price, 4)
        if shares < 0.01:
            log.info(f"Skip {ticker}: {shares} shares too small")
            return None

        try:
            order = self.client.submit_order(
                MarketOrderRequest(
                    symbol=ticker,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log.info(f"BUY {ticker}: {shares} shares @ ~${current_price:.2f} "
                     f"(${dollar_amount:,.0f}) | order_id={order.id}")
            self.positions[ticker] = {
                "order_id": str(order.id),
                "shares": shares,
                "entry_price": current_price,
                "entry_time": datetime.now(ET),
                "dollar_amount": dollar_amount,
            }
            return order
        except Exception as e:
            log.error(f"BUY {ticker} FAILED: {e}")
            return None

    def sell(self, ticker, shares=None, reason="MANUAL"):
        """Sell a position (full or partial).

        Args:
            ticker: stock symbol
            shares: number of shares to sell (None = close entire position)
            reason: exit reason for logging
        """
        try:
            if shares is None:
                # Close entire position
                order = self.client.close_position(ticker)
                log.info(f"SELL ALL {ticker} ({reason}) | order_id={order.id}")
            else:
                order = self.client.submit_order(
                    MarketOrderRequest(
                        symbol=ticker,
                        qty=round(shares, 4),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                log.info(f"SELL {ticker}: {shares:.4f} shares ({reason}) | order_id={order.id}")

            if ticker in self.positions:
                if shares is None or shares >= self.positions[ticker]["shares"]:
                    del self.positions[ticker]
                else:
                    self.positions[ticker]["shares"] -= shares

            return order
        except Exception as e:
            log.error(f"SELL {ticker} FAILED: {e}")
            return None

    def close_all_positions(self, reason="EOD_CLOSE"):
        """Close all open positions."""
        positions = self.get_positions()
        for pos in positions:
            self.sell(pos.symbol, reason=reason)
        log.info(f"Closed {len(positions)} positions ({reason})")

    def get_open_orders(self):
        """Get all open/pending orders."""
        return self.client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.client.cancel_orders()
        log.info("Cancelled all open orders")
