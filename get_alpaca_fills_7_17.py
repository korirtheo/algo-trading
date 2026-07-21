#!/usr/bin/env python3
"""Get actual Alpaca fills for 7/17/2026."""
import os
import sys
from datetime import date
from zoneinfo import ZoneInfo

try:
    from alpaca.trading.client import TradingClient
except ImportError:
    print("alpaca module not found")
    sys.exit(1)

api_key = os.environ.get("ALPACA_API_KEY")
api_secret = os.environ.get("ALPACA_API_SECRET")

if not api_key or not api_secret:
    print("ERROR: ALPACA_API_KEY or ALPACA_API_SECRET not set")
    sys.exit(1)

print("Connecting to Alpaca API...")
client = TradingClient(api_key, api_secret)

print("Fetching closed orders...")
orders = client.get_orders(status="closed", limit=500)

ET = ZoneInfo("America/New_York")
target_date = date(2026, 7, 17)

print("\n" + "="*80)
print("ALPACA FILLS ON 2026-07-17:")
print("="*80)

sdot_buy = None
sdot_sell = None
veee_buy = None
veee_sell = None

for order in orders:
    if not order.filled_at:
        continue

    order_date = order.filled_at.astimezone(ET).date()
    if order_date != target_date:
        continue

    if order.symbol == "SDOT":
        print(f"SDOT {order.side.upper()}: {order.filled_qty} @ ${order.filled_avg_price:.2f}")
        if order.side == "buy":
            sdot_buy = (order.filled_qty, order.filled_avg_price)
        else:
            sdot_sell = (order.filled_qty, order.filled_avg_price)

    elif order.symbol == "VEEE":
        print(f"VEEE {order.side.upper()}: {order.filled_qty} @ ${order.filled_avg_price:.2f}")
        if order.side == "buy":
            veee_buy = (order.filled_qty, order.filled_avg_price)
        else:
            veee_sell = (order.filled_qty, order.filled_avg_price)

print("\n" + "="*80)
print("CALCULATED P&L:")
print("="*80)

total_alpaca = 0

if sdot_buy and sdot_sell:
    buy_qty, buy_px = sdot_buy
    sell_qty, sell_px = sdot_sell
    sdot_pnl = (sell_px - buy_px) * buy_qty
    print(f"SDOT: {buy_qty} @ ${buy_px:.2f} -> {sell_qty} @ ${sell_px:.2f} = ${sdot_pnl:,.2f}")
    total_alpaca += sdot_pnl
else:
    print(f"SDOT: incomplete (buy={sdot_buy}, sell={sdot_sell})")

if veee_buy and veee_sell:
    buy_qty, buy_px = veee_buy
    sell_qty, sell_px = veee_sell
    veee_pnl = (sell_px - buy_px) * buy_qty
    print(f"VEEE: {buy_qty} @ ${buy_px:.2f} -> {sell_qty} @ ${sell_px:.2f} = ${veee_pnl:,.2f}")
    total_alpaca += veee_pnl
else:
    print(f"VEEE: incomplete (buy={veee_buy}, sell={veee_sell})")

print(f"\nAlpaca Total: ${total_alpaca:,.2f}")
print(f"Engine Reported: $200.57")
print(f"Difference: ${total_alpaca - 200.57:,.2f}")
