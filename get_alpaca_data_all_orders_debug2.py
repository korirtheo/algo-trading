from alpaca.trading.client import TradingClient
import os

# Credentials are already set in the container environment
api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_API_SECRET")

client = TradingClient(api_key, api_secret, paper=True)

# Get account info
account = client.get_account()
print(f"Account Equity: ${account.equity}")
print(f"Cash: ${account.cash}")

# Get all orders
all_orders = client.get_orders()

print(f"\nTotal orders: {len(all_orders)}\n")
print(all_orders)

for order in all_orders:
    print(f"  Symbol: {order.symbol}")
    print(f"  Created at: {order.created_at}")
    print(f"  Status: {order.status}")
    print(f"  Side: {order.side.upper()}")
    print(f"  Qty: {order.qty}")
    print(f"  Filled Qty: {order.filled_qty}")
    print(f"  Filled Avg Price: {order.filled_avg_price}")
    print("---")
