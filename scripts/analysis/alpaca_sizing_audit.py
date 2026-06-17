"""Audit live position sizing vs equity / cash / buying-power.

For every BUY fill on Alpaca paper, computes:
  - notional ($)
  - notional as % of that day's starting equity (from portfolio history)
  - whether sizes look consistent with FULL_BALANCE (~100% equity),
    EQUITY_CAP_30 (~30%), or BUYING_POWER (>100% equity, up to 4x)

This is the empirical confirmation step: does the live bot match the
backtest's full-balance sizing, the new 30% cash-cap, or is it using
margin buying power?
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from datetime import datetime, timezone
from collections import defaultdict
import statistics

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import QueryOrderStatus

from config import settings

c = TradingClient(settings.ALPACA_API_KEY, settings.ALPACA_API_SECRET, paper=True)

acc = c.get_account()
print("=== Account snapshot ===")
print(f"  Equity:        ${float(acc.equity):,.2f}")
print(f"  Cash:          ${float(acc.cash):,.2f}")
print(f"  Buying power:  ${float(acc.buying_power):,.2f}")
print(f"  Multiplier:    {float(acc.multiplier):.0f}x")
print(f"  PDT:           {acc.pattern_day_trader}  (daytrades={acc.daytrade_count})")
print(f"  Created:       {acc.created_at}")
print(f"  Last equity:   ${float(acc.last_equity):,.2f}")

created = acc.created_at
if created.tzinfo is None:
    created = created.replace(tzinfo=timezone.utc)
days_open = (datetime.now(timezone.utc) - created).days

# Pull portfolio history (daily) since creation
ph = c.get_portfolio_history(history_filter=GetPortfolioHistoryRequest(
    period=f"{min(days_open + 1, 90)}D", timeframe="1D"
))

day_equity = {}
if ph.timestamp and ph.equity:
    for ts, eq in zip(ph.timestamp, ph.equity):
        d = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        day_equity[d] = eq

print(f"\n=== Portfolio history ({len(day_equity)} days) ===")
if day_equity:
    days = sorted(day_equity.keys())
    print(f"  {days[0]}: ${day_equity[days[0]]:,.2f}")
    print(f"  {days[-1]}: ${day_equity[days[-1]]:,.2f}")
    delta = day_equity[days[-1]] - day_equity[days[0]]
    print(f"  Delta:  ${delta:+,.2f} ({delta/day_equity[days[0]]*100:+.2f}%)")

# Pull all orders
orders = c.get_orders(filter=GetOrdersRequest(
    status=QueryOrderStatus.ALL, after=created,
    until=datetime.now(timezone.utc), limit=500,
))
print(f"\n=== Fills ({len(orders)} orders pulled) ===")

# Walk each BUY fill (parent + bracket children) — group by parent ID if present
parent_buys = []  # (date, ticker, qty, fill_price, notional, order_class, child legs)
for o in orders:
    if not o.filled_qty or float(o.filled_qty) <= 0:
        continue
    if o.side.value != "buy":
        continue
    # Skip stop_loss/take_profit child legs that came back as fills (those are SELLS though)
    qty = float(o.filled_qty)
    price = float(o.filled_avg_price or 0)
    if price <= 0:
        continue
    day = o.submitted_at.strftime("%Y-%m-%d") if o.submitted_at else "?"
    parent_buys.append({
        "date": day, "ticker": o.symbol, "qty": qty,
        "fill_price": price, "notional": qty * price,
        "order_class": (o.order_class.value if o.order_class else "simple"),
        "submitted_at": o.submitted_at,
        "stop_loss": getattr(o, "stop_loss", None),
        "take_profit": getattr(o, "take_profit", None),
    })

if not parent_buys:
    print("  No buy fills found.")
    raise SystemExit(0)

print(f"  {'date':<11} {'symbol':<8} {'class':<8} {'qty':>7} {'fill':>8} {'notional':>11} {'eq_at_open':>11} {'%eq':>6} {'bracket':<10}")
print("-" * 110)

# Stats trackers
pct_of_equity = []
hit_cash_cap_at_30 = 0
exceeded_equity = 0
had_bracket = 0
had_stop = 0

for b in sorted(parent_buys, key=lambda x: (x["date"], x["ticker"])):
    eq = day_equity.get(b["date"])
    # If exact day missing (weekend submit?), find nearest prior day
    if eq is None:
        prior = [d for d in day_equity if d < b["date"]]
        if prior:
            eq = day_equity[max(prior)]
    pct_eq = (b["notional"] / eq * 100) if eq else None
    eq_str = f"${eq:,.0f}" if eq else "?"
    pct_str = f"{pct_eq:>5.1f}%" if pct_eq is not None else "    ?"

    bracket_marker = "no"
    if b["order_class"] == "bracket":
        bracket_marker = "YES"
        had_bracket += 1
    if b["stop_loss"]:
        had_stop += 1

    if pct_eq is not None:
        pct_of_equity.append(pct_eq)
        if pct_eq > 30:
            hit_cash_cap_at_30 += 1
        if pct_eq > 100:
            exceeded_equity += 1

    print(f"  {b['date']:<11} {b['ticker']:<8} {b['order_class']:<8} "
          f"{b['qty']:>7.0f} ${b['fill_price']:>7.2f} ${b['notional']:>10,.0f} "
          f"{eq_str:>11} {pct_str} {bracket_marker:<10}")

print(f"\n=== Sizing distribution (buy notional / day-start equity) ===")
if pct_of_equity:
    pct_of_equity.sort()
    print(f"  Count:             {len(pct_of_equity)}")
    print(f"  Median:            {statistics.median(pct_of_equity):.1f}% of equity")
    print(f"  Mean:              {statistics.mean(pct_of_equity):.1f}% of equity")
    print(f"  25th pct:          {pct_of_equity[len(pct_of_equity)//4]:.1f}%")
    print(f"  75th pct:          {pct_of_equity[3*len(pct_of_equity)//4]:.1f}%")
    print(f"  Max:               {max(pct_of_equity):.1f}%")
    print(f"  >30% (would hit cash-cap): {hit_cash_cap_at_30}/{len(pct_of_equity)} ({100*hit_cash_cap_at_30/len(pct_of_equity):.0f}%)")
    print(f"  >100% (used margin/BP):    {exceeded_equity}/{len(pct_of_equity)} ({100*exceeded_equity/len(pct_of_equity):.0f}%)")
    print(f"\n=== Safety rails status ===")
    print(f"  Bracket orders:    {had_bracket}/{len(parent_buys)} ({100*had_bracket/len(parent_buys):.0f}%)")
    print(f"  With stop_loss:    {had_stop}/{len(parent_buys)} ({100*had_stop/len(parent_buys):.0f}%)")

    print(f"\n=== Verdict ===")
    if exceeded_equity > 0:
        print(f"  CONFIRMED: Bot has been using MARGIN BUYING POWER")
        print(f"    {exceeded_equity} trades exceeded 100% of day-start equity")
        print(f"    Engine is sizing from buying_power, NOT clamped by cash cap.")
    elif hit_cash_cap_at_30 > 0:
        print(f"  No margin usage. {hit_cash_cap_at_30} trades >30% of equity —")
        print(f"  these WOULD have been clamped by the new 30% cash cap.")
        print(f"  (Cash cap not yet deployed — see deploy todo.)")
    else:
        print(f"  All trades under 30% of equity. Whatever was sizing them was")
        print(f"  ALREADY conservative. Cap is redundant on this data slice.")
