"""Microcap slippage calibration — round-trips in our actual deployment regime
($1-$10 stocks with thin-to-moderate volume). NOT on today's live watchlist.

Filters candidates by recent volume so we only round-trip tickers that
actually have a liquid market right now. Skips dead tickers gracefully.
"""
import csv
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

KEY = "PK2ESU3PR3BBJH2LT2MC6NBTAQ"
SECRET = "HCWTWV3JBKthRgYuD6avUV8gj6GFVpFek6mDKdV2oBTq"

ET = ZoneInfo("America/New_York")
CSV_PATH = os.environ.get("CALIB_CSV", "/app/logs/fills_calibration.csv")
TERMINAL = {"filled", "partially_filled", "canceled", "expired", "rejected", "done_for_day"}

# Today's live watchlist — AVOID these to not conflict with the engine
LIVE_WATCHLIST = {
    "ADTX", "BWIN", "CANG", "CDT", "CHCI", "CRMT", "DFTX", "EHGO", "GDC", "GETY",
    "HQ", "ICCM", "NXTS", "POM", "SAGT", "SKYQ", "SSTK", "TNON", "VEEA", "WKSP",
}

# Microcap candidates — diverse mix of price/liquidity in our deployment regime.
# We'll pick the first N that have positive recent volume.
CANDIDATES = [
    "TLRY",   # cannabis, often $1-3
    "PLUG",   # hydrogen, $2-4
    "WKHS",   # delivery EV, $1-3
    "MULN",   # EV, often <$1
    "BBIG",   # small cap retail
    "HKD",    # AMTD Digital, volatile
    "AMTD",   # AMTD
    "MARA",   # mining
    "RIOT",   # mining
    "SOFI",   # finance, $5-8
    "F",      # already done — repeat for second sample
    "BFLY",   # Butterfly Network, $5-9
    "CRVO",   # CervoMed, $4-8
    "AMC",    # repeat for $2.82 range second sample
    "NVDA",   # high price anchor
]

TARGET_USD = 200  # per leg


def submit_and_wait(client, ticker, qty, side, timeout=15):
    ts_submit = datetime.now(ET).isoformat()
    try:
        order = client.submit_order(MarketOrderRequest(
            symbol=ticker, qty=qty, side=side, time_in_force=TimeInForce.DAY))
    except Exception as e:
        return None, ts_submit, str(e)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            o = client.get_order_by_id(order.id)
            status = str(o.status).lower().replace("orderstatus.", "")
            if status in TERMINAL:
                return o, ts_submit, None
        except Exception:
            pass
        time.sleep(0.4)
    try:
        return client.get_order_by_id(order.id), ts_submit, "timeout"
    except Exception:
        return None, ts_submit, "timeout"


def get_quote_and_vol(data_client, ticker):
    """Return (bid, ask, mid, spread_bp, cum_dvol_today) or None."""
    try:
        q = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
        quote = q[ticker]
        bid = float(quote.bid_price); ask = float(quote.ask_price)
        if bid <= 0 or ask <= 0: return None
        mid = (bid + ask) / 2
        spread_bp = (ask - bid) / mid * 10_000
    except Exception:
        return None
    try:
        start = datetime.now(timezone.utc).replace(hour=13, minute=30, second=0, microsecond=0)
        end = datetime.now(timezone.utc)
        bars = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Minute,
            start=start, end=end, feed="iex")).df
        cum_dvol = float((bars["close"] * bars["volume"]).sum()) if not bars.empty else 0
    except Exception:
        cum_dvol = 0
    return bid, ask, mid, spread_bp, cum_dvol


def write_row(side, ticker, signal_price, order, ts_submit, cum_dvol):
    status = "unknown"; fill_price = None; filled_qty = None
    if order:
        status = str(order.status).lower().replace("orderstatus.", "")
        if order.filled_avg_price: fill_price = float(order.filled_avg_price)
        if order.filled_qty: filled_qty = float(order.filled_qty)
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
    print(f"Microcap calibration — writing to {CSV_PATH}\n")
    trader = TradingClient(KEY, SECRET, paper=True)
    data = StockHistoricalDataClient(KEY, SECRET)

    done = 0
    target_n = 10  # round-trip ~10 microcaps

    for ticker in CANDIDATES:
        if ticker in LIVE_WATCHLIST:
            print(f"=== {ticker} ===  SKIP (on live watchlist)")
            continue
        if done >= target_n:
            break

        q = get_quote_and_vol(data, ticker)
        if q is None:
            print(f"=== {ticker} ===  SKIP (no quote)")
            continue
        bid, ask, mid, spread_bp, cum_dvol = q

        if cum_dvol < 100_000:  # require some real action
            print(f"=== {ticker} ===  SKIP (cum_dvol=${cum_dvol/1e3:.0f}K — too thin)")
            continue

        qty = max(1, int(TARGET_USD / mid))
        print(f"=== {ticker} ===")
        print(f"  bid=${bid:.4f} ask=${ask:.4f} mid=${mid:.4f} spread={spread_bp:.1f}bp "
              f"cum_dvol=${cum_dvol/1e6:.2f}M qty={qty}")

        order, ts_submit, err = submit_and_wait(trader, ticker, qty, OrderSide.BUY, timeout=15)
        if order is None:
            print(f"  BUY FAILED: {err}")
            continue
        row = write_row("buy", ticker, mid, order, ts_submit, cum_dvol)
        print(f"  → BUY qty={row['qty']} fill={row['fill_price']} slip={row['slip_bp']}bp status={row['status']}")

        if not order.filled_qty or float(order.filled_qty) < 1:
            print(f"  no fill on buy; skip")
            continue
        filled = float(order.filled_qty)
        time.sleep(2)

        q2 = get_quote_and_vol(data, ticker)
        sell_mid = q2[2] if q2 else mid
        sell_cum = q2[4] if q2 else cum_dvol

        order2, ts_submit2, err2 = submit_and_wait(trader, ticker, int(filled), OrderSide.SELL, timeout=15)
        if order2 is None:
            print(f"  SELL FAILED: {err2}")
            continue
        row2 = write_row("sell", ticker, sell_mid, order2, ts_submit2, sell_cum)
        print(f"  → SELL qty={row2['qty']} fill={row2['fill_price']} slip={row2['slip_bp']}bp status={row2['status']}")

        done += 1
        time.sleep(2)
        print()

    print(f"\nCompleted {done} round-trips")


if __name__ == "__main__":
    main()
