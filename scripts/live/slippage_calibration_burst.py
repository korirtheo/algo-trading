"""Manual slippage calibration: round-trip small positions on tickers spanning
different liquidity tiers, append rows to logs/fills_calibration.csv.

Designed to run alongside the live engine (won't interfere — uses tickers
NOT on today's watchlist, small position sizes).

Each round-trip:
  1. Submit market BUY of N shares
  2. Wait for terminal status (filled / canceled / rejected), record slippage
  3. Submit market SELL of filled qty
  4. Wait for terminal status, record slippage

Tickers chosen for diverse participation rates:
  SPY  - ultra liquid (anchor for ~0 impact)
  AAPL - large liquid
  F    - moderate
  BAC  - moderate
  AMC  - microcap-ish thin

All small positions ($100-$500) to keep slippage cost negligible.
"""
import csv
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# Add project root for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import timedelta, timezone

# NEW account credentials
KEY = "PK2ESU3PR3BBJH2LT2MC6NBTAQ"
SECRET = "HCWTWV3JBKthRgYuD6avUV8gj6GFVpFek6mDKdV2oBTq"

ET = ZoneInfo("America/New_York")
# Absolute path — works whether run from /app or /tmp
CSV_PATH = os.environ.get("CALIB_CSV", "/app/logs/fills_calibration.csv")
TERMINAL = {"filled", "partially_filled", "canceled", "expired", "rejected", "done_for_day"}

# (ticker, target_dollar_amount) — small enough that even adverse fills cost <$5
TICKERS = [
    ("SPY",  300),   # ~0.5 shares; rounded to 1
    ("AAPL", 300),   # ~1-2 shares
    ("F",    300),   # ~30 shares
    ("BAC",  300),   # ~6-7 shares
    ("AMC",  300),   # microcap-ish, ~80 shares
]


def submit_and_wait(client, ticker, qty, side, timeout=15):
    """Submit a market order, poll for terminal status."""
    ts_submit = datetime.now(ET).isoformat()
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=qty, side=side, time_in_force=TimeInForce.DAY))
    except Exception as e:
        print(f"    SUBMIT FAILED: {e}")
        return None, ts_submit
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            o = client.get_order_by_id(order.id)
            last = o
            status = str(o.status).lower().replace("orderstatus.", "")
            if status in TERMINAL:
                break
        except Exception:
            pass
        time.sleep(0.5)
    return last, ts_submit


def get_quote(data_client, ticker):
    """Get current NBBO quote."""
    try:
        q = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
        return q[ticker]
    except Exception as e:
        print(f"  quote err: {e}")
        return None


def get_cum_dvol(data_client, ticker):
    """Estimate cumulative dollar volume today (from 9:30 ET)."""
    try:
        start = datetime.now(timezone.utc).replace(hour=13, minute=30, second=0, microsecond=0)
        end = datetime.now(timezone.utc)
        bars = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Minute,
            start=start, end=end, feed="iex")).df
        if bars.empty: return 0
        cum = (bars["close"] * bars["volume"]).sum()
        return float(cum)
    except Exception as e:
        print(f"  bars err: {e}")
        return 0


def write_row(side, ticker, signal_price, order, ts_submit, cum_dvol):
    """Append a row to fills_calibration.csv matching executor.py's format."""
    status = "unknown"
    fill_price = None
    filled_qty = None
    if order:
        status = str(order.status).lower().replace("orderstatus.", "")
        if order.filled_avg_price:
            fill_price = float(order.filled_avg_price)
        if order.filled_qty:
            filled_qty = float(order.filled_qty)

    slip_bp = None
    if fill_price is not None and signal_price and signal_price > 0:
        if side == "buy":
            slip_bp = (fill_price / signal_price - 1.0) * 10_000
        else:
            slip_bp = (signal_price / fill_price - 1.0) * 10_000

    dollar_amount = (fill_price * filled_qty) if (fill_price and filled_qty) else None
    participation = (dollar_amount / cum_dvol) if (dollar_amount and cum_dvol > 0) else None

    ts_fill = datetime.now(ET).isoformat()
    row = {
        "ts_signal": ts_submit, "ts_fill": ts_fill, "ticker": ticker,
        "side": side, "strategy": "CALIBRATION",
        "signal_price": f"{signal_price:.4f}" if signal_price else "",
        "fill_price": f"{fill_price:.4f}" if fill_price is not None else "",
        "slip_bp": f"{slip_bp:.2f}" if slip_bp is not None else "",
        "qty": f"{filled_qty:.0f}" if filled_qty is not None else "",
        "dollar_amount": f"{dollar_amount:.2f}" if dollar_amount is not None else "",
        "cum_dollar_vol": f"{cum_dvol:.0f}" if cum_dvol else "",
        "participation_rate": f"{participation:.6f}" if participation is not None else "",
        "order_id": str(order.id) if order else "",
        "status": status,
    }
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)
    return row


def main():
    print(f"Slippage calibration burst — writing to {CSV_PATH}")
    print(f"Tickers: {[t[0] for t in TICKERS]}\n")

    trader = TradingClient(KEY, SECRET, paper=True)
    data = StockHistoricalDataClient(KEY, SECRET)

    for ticker, target_usd in TICKERS:
        print(f"=== {ticker} ===")
        quote = get_quote(data, ticker)
        if quote is None:
            print(f"  no quote; skip"); continue
        mid = (float(quote.bid_price) + float(quote.ask_price)) / 2.0
        if mid <= 0:
            print(f"  zero mid; skip"); continue
        qty = max(1, int(target_usd / mid))
        cum_dvol = get_cum_dvol(data, ticker)
        spread_bp = (float(quote.ask_price) - float(quote.bid_price)) / mid * 10_000

        print(f"  bid=${quote.bid_price:.4f} ask=${quote.ask_price:.4f} mid=${mid:.4f} "
              f"spread={spread_bp:.1f}bp cum_dvol=${cum_dvol/1e6:.1f}M qty={qty}")

        # BUY
        print(f"  → BUY {qty} @ ~mid ${mid:.4f}")
        order, ts_submit = submit_and_wait(trader, ticker, qty, OrderSide.BUY, timeout=15)
        if order is None:
            continue
        row = write_row("buy", ticker, mid, order, ts_submit, cum_dvol)
        print(f"    fill={row['fill_price']} slip={row['slip_bp']}bp status={row['status']}")

        if not order.filled_qty or float(order.filled_qty) < 1:
            print(f"    no shares filled; skip sell")
            continue
        filled = float(order.filled_qty)
        time.sleep(2)

        # SELL to flatten
        sell_quote = get_quote(data, ticker)
        sell_mid = (float(sell_quote.bid_price) + float(sell_quote.ask_price)) / 2.0 if sell_quote else mid
        cum_dvol2 = get_cum_dvol(data, ticker)
        print(f"  → SELL {int(filled)} @ ~mid ${sell_mid:.4f}")
        order2, ts_submit2 = submit_and_wait(trader, ticker, int(filled), OrderSide.SELL, timeout=15)
        if order2 is None: continue
        row2 = write_row("sell", ticker, sell_mid, order2, ts_submit2, cum_dvol2)
        print(f"    fill={row2['fill_price']} slip={row2['slip_bp']}bp status={row2['status']}")

        time.sleep(2)
        print()

    print("Calibration burst complete.")


if __name__ == "__main__":
    main()
