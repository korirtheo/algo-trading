"""Pull all Alpaca paper trades since account creation and analyze."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import QueryOrderStatus

from config import settings

c = TradingClient(settings.ALPACA_API_KEY, settings.ALPACA_API_SECRET, paper=True)

# Account snapshot
acc = c.get_account()
print(f"=== Account ===")
print(f"  Equity:           ${float(acc.equity):,.2f}")
print(f"  Cash:             ${float(acc.cash):,.2f}")
print(f"  Buying power:     ${float(acc.buying_power):,.2f}")
print(f"  Initial deposit:  ${float(acc.last_equity):,.2f} (last equity)")
print(f"  Daytrade count:   {acc.daytrade_count}")
print(f"  PDT flag:         {acc.pattern_day_trader}")
print(f"  Status:           {acc.status}")
print(f"  Created:          {acc.created_at}")

# Calculate days since creation
created = acc.created_at
if created.tzinfo is None:
    created = created.replace(tzinfo=timezone.utc)
days_open = (datetime.now(timezone.utc) - created).days
print(f"  Days open:        {days_open}")

# Portfolio history
print(f"\n=== Portfolio history (last {min(days_open, 60)} days) ===")
ph_req = GetPortfolioHistoryRequest(
    period=f"{min(days_open, 60)}D" if days_open > 1 else "1D",
    timeframe="1D"
)
try:
    ph = c.get_portfolio_history(filter=ph_req)
    if ph.equity and ph.timestamp:
        start_eq = ph.equity[0]
        end_eq = ph.equity[-1]
        pct = (end_eq - start_eq) / start_eq * 100 if start_eq else 0
        print(f"  Start equity:     ${start_eq:,.2f}")
        print(f"  End equity:       ${end_eq:,.2f}")
        print(f"  Total return:     {pct:+.2f}%")
        # Daily breakdown
        print(f"\n  Daily equity:")
        for ts, eq in zip(ph.timestamp[-30:], ph.equity[-30:]):
            d = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            print(f"    {d}  ${eq:,.2f}")
except Exception as e:
    print(f"  Could not fetch portfolio history: {e}")

# All orders (fills)
print(f"\n=== Recent orders (all statuses) ===")
all_orders = []
# Walk back through pages
end_dt = datetime.now(timezone.utc)
window_days = min(days_open + 5, 60)
req = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    after=created,
    until=end_dt,
    limit=500,
)
try:
    orders = c.get_orders(filter=req)
    print(f"  Pulled {len(orders)} orders since {created.strftime('%Y-%m-%d')}")
    all_orders = orders
except Exception as e:
    print(f"  Order fetch error: {e}")
    all_orders = []

if not all_orders:
    print("  No orders found.")
    return_path = None
else:
    # Group fills by symbol-day to reconstruct round trips
    by_symbol_day = defaultdict(list)
    for o in all_orders:
        if o.filled_qty and float(o.filled_qty) > 0:
            day = o.submitted_at.strftime("%Y-%m-%d") if o.submitted_at else "?"
            by_symbol_day[(o.symbol, day)].append(o)

    print(f"\n=== Round trips by symbol/day ({len(by_symbol_day)} symbol-days with fills) ===")
    print(f"  {'date':<11} {'symbol':<8} {'side':<5} {'qty':>7} {'avg_fill':>10} {'notional':>12}")
    print("-" * 80)
    trips = []
    for (sym, day), orders in sorted(by_symbol_day.items()):
        # Aggregate by side
        sides = defaultdict(lambda: {"qty": 0.0, "notional": 0.0})
        for o in orders:
            qty = float(o.filled_qty or 0)
            price = float(o.filled_avg_price or 0)
            sides[o.side.value]["qty"] += qty
            sides[o.side.value]["notional"] += qty * price
        buy = sides.get("buy", {"qty": 0, "notional": 0})
        sell = sides.get("sell", {"qty": 0, "notional": 0})
        if buy["qty"] > 0:
            avg_buy = buy["notional"] / buy["qty"]
            print(f"  {day:<11} {sym:<8} BUY   {buy['qty']:>7.0f} ${avg_buy:>9.2f} ${buy['notional']:>11,.0f}")
        if sell["qty"] > 0:
            avg_sell = sell["notional"] / sell["qty"]
            print(f"  {day:<11} {sym:<8} SELL  {sell['qty']:>7.0f} ${avg_sell:>9.2f} ${sell['notional']:>11,.0f}")
        if buy["qty"] > 0 and sell["qty"] > 0:
            qty_matched = min(buy["qty"], sell["qty"])
            pnl = (sell["notional"] / sell["qty"] - buy["notional"] / buy["qty"]) * qty_matched
            print(f"             round-trip PnL: ${pnl:+,.2f}  ({(pnl / (qty_matched * avg_buy) * 100):+.2f}%)")
            trips.append({
                "date": day, "symbol": sym, "qty": qty_matched,
                "avg_buy": avg_buy, "avg_sell": avg_sell, "pnl": pnl,
                "pct": (pnl / (qty_matched * avg_buy) * 100) if avg_buy > 0 else 0,
            })
        print()

    # Aggregate stats
    if trips:
        print(f"\n=== Round-trip stats ===")
        wins = [t for t in trips if t["pnl"] > 0]
        losses = [t for t in trips if t["pnl"] < 0]
        total_pnl = sum(t["pnl"] for t in trips)
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        pf = gross_win / gross_loss if gross_loss > 0 else 99
        print(f"  Round trips:     {len(trips)}")
        print(f"  Wins/Losses:     {len(wins)}/{len(losses)}  ({100*len(wins)/len(trips):.1f}% WR)")
        print(f"  Total PnL:       ${total_pnl:+,.2f}")
        print(f"  PF:              {pf:.2f}")
        print(f"  Avg trade %:     {sum(t['pct'] for t in trips)/len(trips):+.2f}%")
        print(f"  Best trade:      {max(trips, key=lambda t: t['pnl'])['symbol']} on {max(trips, key=lambda t: t['pnl'])['date']}: ${max(trips, key=lambda t: t['pnl'])['pnl']:+,.2f}")
        print(f"  Worst trade:     {min(trips, key=lambda t: t['pnl'])['symbol']} on {min(trips, key=lambda t: t['pnl'])['date']}: ${min(trips, key=lambda t: t['pnl'])['pnl']:+,.2f}")
