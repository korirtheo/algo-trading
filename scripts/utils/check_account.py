import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER

client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
acct = client.get_account()
print(f"Cash: ${float(acct.cash):,.2f}  BP: ${float(acct.buying_power):,.2f}")

positions = client.get_all_positions()
print(f"Open positions: {len(positions)}")
for p in positions:
    print(f"  {p.symbol}: {p.qty} shares @ ${float(p.avg_entry_price):.3f} | current=${float(p.current_price):.3f} | PnL=${float(p.unrealized_pl):+.2f}")

orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=10))
print(f"Last {len(orders)} orders:")
for o in orders:
    print(f"  {o.created_at.strftime('%H:%M:%S')} {o.symbol} {o.side} {o.qty}sh | status={o.status} | filled_avg=${o.filled_avg_price}")
