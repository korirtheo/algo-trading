"""
Order Executor: Places and manages orders via Alpaca Trading API.

Handles:
  - Market buy orders for entries
  - Stop/limit orders for exits
  - Position tracking
  - EOD forced close
  - Post-fill reconciliation to fills_calibration.csv (slippage Stage 1)
"""
import csv
import logging
import os
import threading
import time
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

from config.settings import (
    ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER, VOL_CAP_PCT,
    PDT_EQUITY_FLOOR, PDT_EQUITY_BUFFER, PDT_DAYTRADES_MAX,
)

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# Slippage calibration log — one row per terminal-status order.
_FILL_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "fills_calibration.csv",
)
_FILL_LOG_FIELDS = [
    "ts_signal", "ts_fill", "ticker", "side", "strategy",
    "signal_price", "fill_price", "slip_bp",
    "qty", "dollar_amount", "cum_dollar_vol", "participation_rate",
    "order_id", "status",
]
_FILL_LOG_LOCK = threading.Lock()
_TERMINAL_STATUSES = {"filled", "partially_filled", "canceled", "expired", "rejected", "done_for_day"}


def _ensure_fill_log_header():
    os.makedirs(os.path.dirname(_FILL_LOG_PATH), exist_ok=True)
    if not os.path.exists(_FILL_LOG_PATH):
        with open(_FILL_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_FILL_LOG_FIELDS).writeheader()


class OrderExecutor:
    def __init__(self):
        self.client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
        self.positions = {}  # ticker -> position info
        self.pending_orders = {}  # ticker -> order info
        _ensure_fill_log_header()

    def _reconcile_fill_async(self, order_id, ticker, side, signal_price,
                              cum_dollar_vol, strategy, ts_signal,
                              poll_interval=1.0, timeout_secs=30.0):
        """Background poll for the order's terminal status, then append one
        row to fills_calibration.csv with the realized slippage. This is the
        Stage-1 calibration instrumentation — it never changes live behavior,
        only collects data for the offline regression that will later fit
        SLIP_BASE_SPREAD / SLIP_PRICE_COEFF / SLIP_IMPACT_K to actual fills.
        """
        def _worker():
            deadline = time.time() + timeout_secs
            status = None
            fill_price = None
            filled_qty = None
            try:
                while time.time() < deadline:
                    o = self.client.get_order_by_id(order_id)
                    status = str(o.status).lower().replace("orderstatus.", "")
                    if status in _TERMINAL_STATUSES:
                        if o.filled_avg_price is not None:
                            fill_price = float(o.filled_avg_price)
                        if o.filled_qty is not None:
                            filled_qty = float(o.filled_qty)
                        break
                    time.sleep(poll_interval)
                else:
                    status = status or "timeout"
            except Exception as e:
                log.warning(f"reconcile {ticker} {order_id}: {e}")
                status = "poll_error"

            ts_fill = datetime.now(ET).isoformat()
            # Compute slippage in basis points. Convention: positive = adverse
            # fill (paid more on buy, received less on sell). Skip if either
            # leg is missing.
            slip_bp = None
            if fill_price is not None and signal_price and signal_price > 0:
                if side.lower() == "buy":
                    slip_bp = (fill_price / signal_price - 1.0) * 10_000
                else:
                    slip_bp = (signal_price / fill_price - 1.0) * 10_000

            dollar_amount = None
            participation = None
            if fill_price is not None and filled_qty is not None:
                dollar_amount = fill_price * filled_qty
                if cum_dollar_vol and cum_dollar_vol > 0:
                    participation = dollar_amount / cum_dollar_vol

            row = {
                "ts_signal": ts_signal,
                "ts_fill": ts_fill,
                "ticker": ticker,
                "side": side,
                "strategy": strategy or "",
                "signal_price": f"{signal_price:.4f}" if signal_price else "",
                "fill_price": f"{fill_price:.4f}" if fill_price is not None else "",
                "slip_bp": f"{slip_bp:.2f}" if slip_bp is not None else "",
                "qty": f"{filled_qty:.0f}" if filled_qty is not None else "",
                "dollar_amount": f"{dollar_amount:.2f}" if dollar_amount is not None else "",
                "cum_dollar_vol": f"{cum_dollar_vol:.0f}" if cum_dollar_vol else "",
                "participation_rate": f"{participation:.6f}" if participation is not None else "",
                "order_id": str(order_id),
                "status": status,
            }
            with _FILL_LOG_LOCK:
                with open(_FILL_LOG_PATH, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=_FILL_LOG_FIELDS).writerow(row)
            # Stash the real fill on the executor's position record so future
            # consumers (dashboard, EOD report) can pick it up.
            if ticker in self.positions and fill_price is not None:
                self.positions[ticker]["actual_fill_price"] = fill_price
                self.positions[ticker]["actual_status"] = status
            log.info(f"FILL-RECONCILE {ticker} {side}: "
                     f"signal=${signal_price:.4f} fill=${fill_price:.4f} "
                     f"slip={slip_bp:+.1f}bp status={status}"
                     if fill_price else
                     f"FILL-RECONCILE {ticker} {side}: status={status} (no fill price)")

        t = threading.Thread(target=_worker, name=f"reconcile-{order_id}", daemon=True)
        t.start()

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

    def _pdt_blocked(self, ticker):
        """Return True if opening a new position would risk a PDT violation.

        SEC rule: at equity < $25K an account is limited to 3 day-trades in
        any rolling 5-business-day window. The 4th flips the account to
        PDT-flagged → all trading blocked until equity is funded back above
        $25K (90-day lockout otherwise). Since trial #6 strategies typically
        round-trip same-day, every new entry under the floor risks the
        next-exit being a day-trade.
        """
        try:
            acct = self.get_account()
            equity = float(acct.equity)
            daytrades = int(getattr(acct, "daytrade_count", 0) or 0)
            already_flagged = bool(getattr(acct, "pattern_day_trader", False))
            threshold = PDT_EQUITY_FLOOR + PDT_EQUITY_BUFFER

            # Above the buffer: free to trade, PDT doesn't apply.
            if equity >= threshold:
                return False

            if already_flagged:
                log.warning(
                    f"PDT BLOCK {ticker}: account is PDT-flagged and equity "
                    f"${equity:,.2f} < ${PDT_EQUITY_FLOOR:,.0f} — Alpaca will "
                    f"reject any new entry"
                )
                return True

            if daytrades >= PDT_DAYTRADES_MAX:
                log.warning(
                    f"PDT BLOCK {ticker}: daytrade_count={daytrades}/"
                    f"{PDT_DAYTRADES_MAX} and equity ${equity:,.2f} below "
                    f"buffer ${threshold:,.0f} — entering now would risk the "
                    f"4th day-trade triggering a PDT flag on exit"
                )
                return True
            return False
        except Exception as e:
            log.debug(f"PDT check {ticker} failed ({e}); allowing entry")
            return False

    def buy(self, ticker, dollar_amount, current_price,
            cumulative_dollar_volume=0, strategy=None):
        """Place a market buy order.

        Args:
            ticker: stock symbol
            dollar_amount: dollar amount to invest
            current_price: approximate current price (for share calculation)
            cumulative_dollar_volume: cumulative $-volume since 9:30 ET today,
                used to cap position size at VOL_CAP_PCT of the day's dollar
                turnover.  Pre-this-fix the executor multiplied a share-count
                by `current_price`, which understated $-volume because not
                every bar trades at the latest price.

        Returns:
            order object or None if rejected
        """
        # PDT guard — must come before any submission attempt.
        if self._pdt_blocked(ticker):
            return None

        # Volume cap check
        if VOL_CAP_PCT > 0 and cumulative_dollar_volume > 0:
            vol_limit = cumulative_dollar_volume * (VOL_CAP_PCT / 100)
            if vol_limit > 0 and dollar_amount > vol_limit:
                log.info(f"Vol cap: {ticker} limited from ${dollar_amount:,.0f} "
                         f"to ${vol_limit:,.0f} (5% of ${cumulative_dollar_volume:,.0f} day $-vol)")
                dollar_amount = vol_limit
            if dollar_amount < 50:
                log.info(f"Skip {ticker}: vol-capped amount ${dollar_amount:.0f} too small")
                return None

        # Whole shares only. Most small-caps aren't fractionable on Alpaca,
        # and our $1-$50 universe always yields dozens+ of whole shares —
        # fractional precision was never needed and was the silent reason
        # the live engine kept getting BUY rejections.
        shares = int(dollar_amount // current_price)
        if shares < 1:
            log.info(f"Skip {ticker}: dollar_amount=${dollar_amount:.0f} "
                     f"at ${current_price:.2f} -> {shares} shares")
            return None

        ts_signal = datetime.now(ET).isoformat()
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
            # Stage-1 calibration: background poll for the fill, append row
            # to logs/fills_calibration.csv. Doesn't change live behavior.
            self._reconcile_fill_async(
                order_id=order.id, ticker=ticker, side="buy",
                signal_price=current_price,
                cum_dollar_vol=cumulative_dollar_volume,
                strategy=strategy, ts_signal=ts_signal,
            )
            return order
        except Exception as e:
            log.error(f"BUY {ticker} FAILED: {e}")
            return None

    def sell(self, ticker, shares=None, reason="MANUAL",
             signal_price=None, cumulative_dollar_volume=0, strategy=None):
        """Sell a position (full or partial).

        Args:
            ticker: stock symbol
            shares: number of shares to sell (None = close entire position)
            reason: exit reason for logging
            signal_price: the price the strategy *expected* to sell at; used
                for slippage calibration (see _reconcile_fill_async).
            cumulative_dollar_volume / strategy: calibration context.
        """
        ts_signal = datetime.now(ET).isoformat()
        try:
            if shares is None:
                # Close entire position
                order = self.client.close_position(ticker)
                log.info(f"SELL ALL {ticker} ({reason}) | order_id={order.id}")
            else:
                # Mirror the BUY-side: whole shares only (most small-caps
                # are non-fractionable on Alpaca).
                sell_qty = max(1, int(shares))
                order = self.client.submit_order(
                    MarketOrderRequest(
                        symbol=ticker,
                        qty=sell_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                log.info(f"SELL {ticker}: {sell_qty} shares ({reason}) | order_id={order.id}")

            # Stage-1 calibration row for the sell leg.
            if signal_price is not None:
                self._reconcile_fill_async(
                    order_id=order.id, ticker=ticker, side="sell",
                    signal_price=signal_price,
                    cum_dollar_vol=cumulative_dollar_volume,
                    strategy=strategy or reason, ts_signal=ts_signal,
                )

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
