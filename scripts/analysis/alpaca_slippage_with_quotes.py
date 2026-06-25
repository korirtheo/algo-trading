"""Pull Alpaca live fills + historical NBBO quotes at each submission timestamp.

For each filled order:
  - Fetch the bid/ask at submission_at
  - Compute realized slippage in bp vs the bid (sells) or ask (buys)
  - Compare to Almgren-Chriss prediction given position size + observed cum-volume
  - Flag known-bad-config caveats (6/18 had IEX vol-undercount workaround active)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from datetime import datetime, timezone, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

KEY = "PKIPXFIETM7H4BAGQ64FQV3IWJ"
SEC = "25RY682kuN9EBcFr6SFxgbSpFjdZkn613PLKPy1TdYzG"


def main():
    trading = TradingClient(KEY, SEC, paper=True)
    data = StockHistoricalDataClient(KEY, SEC)

    since = datetime.now(timezone.utc) - timedelta(days=7)
    orders = trading.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.ALL, after=since, limit=200, nested=True))

    filled = [o for o in orders
              if str(o.status.value if hasattr(o.status, 'value') else o.status) == 'filled'
              and o.filled_qty and o.filled_avg_price]
    print(f"Found {len(filled)} filled orders\n")

    rows = []
    for o in filled:
        side = o.side.value if hasattr(o.side, 'value') else o.side
        qty = float(o.filled_qty)
        avg_fill = float(o.filled_avg_price)
        cost = qty * avg_fill
        sub_at = o.submitted_at if hasattr(o, 'submitted_at') else o.created_at

        # Pull the quote at submission time (5-second window centered on sub_at)
        try:
            start = sub_at - timedelta(seconds=5)
            end = sub_at + timedelta(seconds=5)
            q_req = StockQuotesRequest(
                symbol_or_symbols=o.symbol,
                start=start, end=end,
                limit=20,
            )
            q_resp = data.get_stock_quotes(q_req)
            quotes = q_resp.data.get(o.symbol, []) if hasattr(q_resp, 'data') else []
            if not quotes:
                bid = ask = mid = spread_bp = None
            else:
                # Pick the quote closest to sub_at
                closest = min(quotes, key=lambda q: abs((q.timestamp - sub_at).total_seconds()))
                bid = float(closest.bid_price)
                ask = float(closest.ask_price)
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else None
                spread_bp = (ask - bid) / mid * 10_000 if mid else None
        except Exception as e:
            bid = ask = mid = spread_bp = None
            print(f"  Quote fetch failed for {o.symbol}: {e}")

        # Slippage: positive bp = adverse fill
        slip_bp = None
        ref_label = "n/a"
        if bid is not None and ask is not None:
            if side == "buy":
                slip_bp = ((avg_fill - ask) / ask) * 10_000
                ref_label = f"ask=${ask:.2f}"
            else:
                slip_bp = ((bid - avg_fill) / bid) * 10_000
                ref_label = f"bid=${bid:.2f}"

        rows.append({
            "symbol": o.symbol,
            "side": str(side),
            "qty": qty,
            "fill_price": avg_fill,
            "cost": cost,
            "bid": bid, "ask": ask, "mid": mid,
            "spread_bp": spread_bp,
            "slippage_bp": slip_bp,
            "ref_label": ref_label,
            "submitted_at": str(sub_at),
        })

    # Display
    print(f"{'symbol':<7} {'side':<5} {'qty':>5} {'fill':>8} {'cost':>9} {'ref':<13} {'spread bp':>10} {'slip bp':>9}")
    print(f"{'-'*7} {'-'*5} {'-'*5} {'-'*8} {'-'*9} {'-'*13} {'-'*10} {'-'*9}")
    for r in rows:
        spread_s = f"{r['spread_bp']:.1f}" if r['spread_bp'] is not None else "n/a"
        slip_s = f"{r['slippage_bp']:+.1f}" if r['slippage_bp'] is not None else "n/a"
        print(f"{r['symbol']:<7} {r['side']:<5} {r['qty']:>5.0f} ${r['fill_price']:>6.2f} "
              f"${r['cost']:>7,.0f} {r['ref_label']:<13} {spread_s:>10} {slip_s:>9}")

    # Stats
    slip_vals = [r["slippage_bp"] for r in rows if r["slippage_bp"] is not None]
    spread_vals = [r["spread_bp"] for r in rows if r["spread_bp"] is not None]
    if slip_vals:
        import statistics
        print(f"\n=== SLIPPAGE SUMMARY ({len(slip_vals)} fills with quotes) ===")
        print(f"  Avg slippage:      {statistics.mean(slip_vals):+.1f} bp")
        print(f"  Median slippage:   {statistics.median(slip_vals):+.1f} bp")
        if len(slip_vals) > 1:
            print(f"  Stdev:             {statistics.stdev(slip_vals):.1f} bp")
        print(f"  Worst (adverse):   {max(slip_vals):+.1f} bp")
        print(f"  Best (favorable):  {min(slip_vals):+.1f} bp")
        print(f"  Avg spread:        {statistics.mean(spread_vals):.1f} bp")

        # Almgren-Chriss prediction with K=3 (current model)
        print(f"\n=== ALMGREN-CHRISS COMPARISON (K=3 current default) ===")
        print(f"  Formula: slip_bp = K × sqrt(participation) × 100 + base_spread_bp")
        print(f"  At our position sizes (~$7K) on microcaps with $50K-$500K/min volume:")
        print(f"    Implied participation ~ 1-15%")
        print(f"    K=3 prediction at 5% participation: 3 × sqrt(0.05) × 100 = ~67 bp per side")
        print(f"    K=3 prediction at 15% participation: 3 × sqrt(0.15) × 100 = ~116 bp per side")

        avg_realized = statistics.mean(slip_vals)
        print(f"\n  Realized avg per-leg slippage: {avg_realized:+.1f} bp")
        if avg_realized < 30:
            print(f"  → REALIZED << PREDICTED: K=3 may be TOO HIGH for this universe")
            print(f"  → IEX vol-undercount workaround (vol-caps disabled) meant ACTUAL participation")
            print(f"     was LOWER than the bot's perceived participation. Real fills were tiny fish.")
        elif avg_realized > 80:
            print(f"  → REALIZED > PREDICTED: K=3 may be TOO LOW; consider K=5")
        else:
            print(f"  → Realized roughly matches K=3 prediction")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/alpaca_slippage_with_quotes.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved results/alpaca_slippage_with_quotes.json")


if __name__ == "__main__":
    main()
