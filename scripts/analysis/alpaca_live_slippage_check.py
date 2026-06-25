"""Pull all Alpaca live fills since deployment and compute realized slippage.

For each filled order:
  - Limit price (if any) or market trigger
  - Actual fill price
  - Computed slippage (bp = ((fill - reference) / reference) × 10,000)
  - Order side (buy/sell) — sign-corrected so positive bp = adverse fill

Outputs to console + saves results/alpaca_live_slippage.json
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from datetime import datetime, timezone, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

KEY = "PKIPXFIETM7H4BAGQ64FQV3IWJ"
SEC = "25RY682kuN9EBcFr6SFxgbSpFjdZkn613PLKPy1TdYzG"
OUTDIR = "results"


def main():
    client = TradingClient(KEY, SEC, paper=True)

    # Get all orders since 5 days ago (covers the deployment)
    since = datetime.now(timezone.utc) - timedelta(days=5)
    orders = client.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.ALL, after=since, limit=200, nested=True))

    print(f"Found {len(orders)} orders in last 5 days\n")

    filled = []
    for o in orders:
        status = o.status.value if hasattr(o.status, 'value') else o.status
        if str(status) != 'filled': continue
        side = o.side.value if hasattr(o.side, 'value') else o.side
        order_type = o.order_type.value if hasattr(o.order_type, 'value') else o.order_type

        qty = float(o.filled_qty or 0)
        avg_fill = float(o.filled_avg_price or 0) if o.filled_avg_price else None
        limit = float(o.limit_price) if o.limit_price else None
        stop = float(o.stop_price) if o.stop_price else None

        if avg_fill is None or qty == 0: continue

        # Reference price for slippage calculation:
        #   Limit order: the limit price (what we asked for)
        #   Market/stop: the stop/trigger price (closest proxy)
        #   No reference price available: we'll mark it as N/A
        ref = limit if limit else stop
        if ref is None or ref == 0:
            # Market order — no reference. Skip slippage calc, but record it.
            filled.append({
                "symbol": o.symbol,
                "side": str(side),
                "type": str(order_type),
                "qty": qty,
                "filled_avg_price": avg_fill,
                "ref_price": None,
                "slippage_bp": None,
                "submitted_at": str(o.submitted_at),
                "filled_at": str(o.filled_at),
                "is_bracket_child": bool(o.legs) if hasattr(o, 'legs') else False,
            })
            continue

        # Sign convention: positive bp = adverse fill
        # Buy at higher than reference = adverse, so (fill - ref) / ref > 0 → positive bp adverse
        # Sell at lower than reference = adverse, so (ref - fill) / ref > 0 → positive bp adverse
        if side == "buy":
            slip_bp = ((avg_fill - ref) / ref) * 10_000
        else:
            slip_bp = ((ref - avg_fill) / ref) * 10_000

        filled.append({
            "symbol": o.symbol,
            "side": str(side),
            "type": str(order_type),
            "qty": qty,
            "filled_avg_price": avg_fill,
            "ref_price": ref,
            "slippage_bp": slip_bp,
            "submitted_at": str(o.submitted_at),
            "filled_at": str(o.filled_at),
            "is_bracket_child": bool(o.legs) if hasattr(o, 'legs') else False,
        })

    if not filled:
        print("No filled orders to analyze. (Live deployment may not have any fills yet.)")
        return

    print(f"=== {len(filled)} FILLED orders ===\n")
    print(f"  {'symbol':<7} {'side':<5} {'type':<11} {'qty':>5} {'ref':>8} {'fill':>8} {'slip_bp':>10} {'submitted_at':<25}")
    print(f"  {'-'*7} {'-'*5} {'-'*11} {'-'*5} {'-'*8} {'-'*8} {'-'*10} {'-'*25}")
    for f in filled:
        ref_s = f"${f['ref_price']:.2f}" if f['ref_price'] else "MARKET"
        fill_s = f"${f['filled_avg_price']:.2f}"
        slip_s = f"{f['slippage_bp']:+.1f} bp" if f['slippage_bp'] is not None else "(N/A)"
        sub = str(f['submitted_at'])[:19]
        print(f"  {f['symbol']:<7} {f['side']:<5} {str(f['type'])[:11]:<11} {f['qty']:>5.0f} {ref_s:>8} {fill_s:>8} {slip_s:>10} {sub}")

    # Stats on slippage
    slip_vals = [f['slippage_bp'] for f in filled if f['slippage_bp'] is not None]
    if slip_vals:
        import statistics
        print(f"\n  === Slippage stats (limit/stop orders with reference price) ===")
        print(f"    n: {len(slip_vals)}")
        print(f"    mean:    {statistics.mean(slip_vals):+.1f} bp")
        print(f"    median:  {statistics.median(slip_vals):+.1f} bp")
        if len(slip_vals) > 1:
            print(f"    stdev:   {statistics.stdev(slip_vals):.1f} bp")
        print(f"    min:     {min(slip_vals):+.1f} bp (best fill)")
        print(f"    max:     {max(slip_vals):+.1f} bp (worst fill = most adverse)")

        # Compare to Almgren-Chriss prediction
        # At 0.05% nominal slippage (March model): expect ~5 bp
        # At Almgren-Chriss K=3 with low participation (paper trades are tiny):
        # participation ~ position_size / cum_vol. For small fills like APWC,
        # the participation might be tiny, so expected slippage near minimum.
        print(f"\n  === Comparison to model assumptions ===")
        print(f"    Old (March) flat-bp:        5 bp (constant)")
        print(f"    Almgren-Chriss base spread: ~5 + 50/price bp (1% at $5 stock)")
        print(f"    Almgren-Chriss impact:      K × sqrt(participation) bp/100, K=3")
        print(f"    Actual realized:            {statistics.mean(slip_vals):.1f} bp avg")

    out_path = f"{OUTDIR}/alpaca_live_slippage.json"
    os.makedirs(OUTDIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(filled, f, indent=2, default=str)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
