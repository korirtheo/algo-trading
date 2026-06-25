"""Alternative slippage estimate using 2-min bars (IEX-feed compatible).

Compares fill_price to the bar's typical price (OHLC midpoint or VWAP).
Less precise than NBBO comparison but matches what's accessible on free tier.
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
from alpaca.data.requests import StockBarsRequest
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

        # Fetch 1-min bars around the fill
        try:
            start = sub_at - timedelta(minutes=2)
            end = sub_at + timedelta(minutes=2)
            b_req = StockBarsRequest(
                symbol_or_symbols=o.symbol,
                start=start, end=end,
                timeframe=TimeFrame.Minute,
                limit=10,
            )
            b_resp = data.get_stock_bars(b_req)
            bars = b_resp.data.get(o.symbol, []) if hasattr(b_resp, 'data') else []
            if not bars:
                bar_high = bar_low = bar_close = bar_vol = bar_vwap = None
            else:
                # Find the bar that contains sub_at (or the nearest)
                closest = min(bars, key=lambda b: abs((b.timestamp - sub_at).total_seconds()))
                bar_high = float(closest.high)
                bar_low = float(closest.low)
                bar_close = float(closest.close)
                bar_open = float(closest.open)
                bar_vol = float(closest.volume) if hasattr(closest, 'volume') else 0
                bar_vwap = float(closest.vwap) if hasattr(closest, 'vwap') and closest.vwap else (bar_high + bar_low + bar_close) / 3
        except Exception as e:
            print(f"  Bar fetch failed for {o.symbol}: {e}")
            bar_high = bar_low = bar_close = bar_vwap = bar_vol = None

        # Slippage relative to bar VWAP (typical fill reference)
        if bar_vwap and bar_vwap > 0:
            if side == "buy":
                slip_vs_vwap_bp = ((avg_fill - bar_vwap) / bar_vwap) * 10_000
                slip_vs_low_bp = ((avg_fill - bar_low) / bar_low) * 10_000  # vs best possible
            else:
                slip_vs_vwap_bp = ((bar_vwap - avg_fill) / bar_vwap) * 10_000
                slip_vs_low_bp = ((bar_high - avg_fill) / bar_high) * 10_000
            bar_range_bp = ((bar_high - bar_low) / bar_vwap) * 10_000
        else:
            slip_vs_vwap_bp = slip_vs_low_bp = bar_range_bp = None

        # Estimate participation: position_dollars / bar_dollar_volume
        bar_dvol = bar_vwap * bar_vol if (bar_vwap and bar_vol) else None
        participation = (cost / bar_dvol) if bar_dvol and bar_dvol > 0 else None

        rows.append({
            "symbol": o.symbol,
            "side": str(side),
            "qty": qty,
            "fill_price": avg_fill,
            "cost": cost,
            "bar_vwap": bar_vwap,
            "bar_low": bar_low,
            "bar_high": bar_high,
            "bar_volume": bar_vol,
            "bar_dollar_volume": bar_dvol,
            "participation_pct": participation * 100 if participation else None,
            "slip_vs_vwap_bp": slip_vs_vwap_bp,
            "slip_vs_best_bp": slip_vs_low_bp,
            "bar_range_bp": bar_range_bp,
            "submitted_at": str(sub_at),
        })

    # Display
    print(f"{'symbol':<7} {'side':<5} {'qty':>6} {'fill$':>7} {'cost$':>8} {'vwap$':>8} {'bar_range_bp':>13} {'partic%':>9} {'slip_vs_vwap':>13}")
    print(f"{'-'*7} {'-'*5} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*13} {'-'*9} {'-'*13}")
    for r in rows:
        vwap = f"${r['bar_vwap']:.2f}" if r['bar_vwap'] else "n/a"
        rng = f"{r['bar_range_bp']:.0f}" if r['bar_range_bp'] else "n/a"
        part = f"{r['participation_pct']:.1f}%" if r['participation_pct'] is not None else "n/a"
        slip = f"{r['slip_vs_vwap_bp']:+.0f} bp" if r['slip_vs_vwap_bp'] is not None else "n/a"
        print(f"{r['symbol']:<7} {r['side']:<5} {r['qty']:>6.0f} ${r['fill_price']:>5.2f} "
              f"${r['cost']:>6,.0f} {vwap:>8} {rng:>13} {part:>9} {slip:>13}")

    # Summary
    slips = [r["slip_vs_vwap_bp"] for r in rows if r["slip_vs_vwap_bp"] is not None]
    parts = [r["participation_pct"] for r in rows if r["participation_pct"] is not None]
    if slips:
        import statistics
        print(f"\n=== REALIZED SLIPPAGE (vs bar VWAP) ===")
        print(f"  Avg:    {statistics.mean(slips):+.1f} bp")
        print(f"  Median: {statistics.median(slips):+.1f} bp")
        print(f"  Range:  {min(slips):+.0f} to {max(slips):+.0f} bp")
        print(f"\n=== ESTIMATED PARTICIPATION (cost / bar_dvol) ===")
        if parts:
            print(f"  Avg:    {statistics.mean(parts):.1f}%")
            print(f"  Range:  {min(parts):.1f}% to {max(parts):.1f}%")

        # Almgren-Chriss back-calibration
        print(f"\n=== ALMGREN-CHRISS BACK-CALIBRATION ===")
        print(f"  Model: slip_bp = K × sqrt(participation) × 100")
        print(f"  Solving for K from each fill:")
        Ks = []
        for r in rows:
            if r["slip_vs_vwap_bp"] is None or r["participation_pct"] is None:
                continue
            p = r["participation_pct"] / 100
            if p <= 0: continue
            implied_K = r["slip_vs_vwap_bp"] / (p ** 0.5 * 100)
            Ks.append(implied_K)
            print(f"    {r['symbol']:<6} {r['side']:<5} part={p*100:>5.1f}% slip={r['slip_vs_vwap_bp']:>+6.0f}bp  → implied K = {implied_K:>6.2f}")

        if Ks:
            print(f"\n  Mean implied K: {statistics.mean(Ks):.2f}")
            print(f"  Median implied K: {statistics.median(Ks):.2f}")
            print(f"  Current model K: 3.0")
            if statistics.mean(Ks) < 1:
                print(f"  → K=3 is WILDLY too high. Real K appears closer to {statistics.mean(Ks):.1f}.")
                print(f"     But this is on TINY positions during the IEX-vol-undercount bad config.")
                print(f"     Need more fills under normal config before recalibrating.")

    os.makedirs("results", exist_ok=True)
    with open("results/alpaca_slippage_vs_bars.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved results/alpaca_slippage_vs_bars.json")


if __name__ == "__main__":
    main()
