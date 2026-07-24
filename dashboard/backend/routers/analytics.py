"""Analytics API - Database query endpoints."""

from fastapi import APIRouter
from datetime import datetime
import sys
import os

# Add project root to path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
)

from live.persistence_db import TradingDatabase

router = APIRouter()
db = TradingDatabase()


@router.get("/analytics/signals/{date}")
async def get_signals(date: str):
    """Get all signals for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, date, timestamp, ticker, strategy, signal_price, action, reason, order_id, gap_pct
            FROM signals
            WHERE date = ?
            ORDER BY timestamp DESC
        """,
            (date,),
        )
        rows = cursor.fetchall()

    signals = []
    for row in rows:
        signals.append(
            {
                "id": row[0],
                "date": row[1],
                "timestamp": row[2],
                "ticker": row[3],
                "strategy": row[4],
                "signal_price": row[5],
                "action": row[6],
                "reason": row[7],
                "order_id": row[8],
                "gap_pct": row[9],
            }
        )

    return signals


@router.get("/analytics/orders/{date}")
async def get_orders(date: str):
    """Get all order events for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, order_id, ticker, strategy, side, event_type,
                   signal_price, fill_price, filled_qty, slip_bp, status
            FROM order_events
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC
        """,
            (date,),
        )
        rows = cursor.fetchall()

    orders = []
    for row in rows:
        orders.append(
            {
                "id": row[0],
                "timestamp": row[1],
                "order_id": row[2],
                "ticker": row[3],
                "strategy": row[4],
                "side": row[5],
                "event_type": row[6],
                "signal_price": row[7],
                "fill_price": row[8],
                "filled_qty": row[9],
                "slip_bp": row[10],
                "status": row[11],
            }
        )

    return orders


@router.get("/analytics/snapshots/{date}")
async def get_snapshots(date: str):
    """Get all account snapshots for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, snapshot_type, cash, equity, buying_power,
                   portfolio_value, daily_pnl, trades_count, positions_count
            FROM account_snapshots
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp ASC
        """,
            (date,),
        )
        rows = cursor.fetchall()

    snapshots = []
    for row in rows:
        snapshots.append(
            {
                "id": row[0],
                "timestamp": row[1],
                "snapshot_type": row[2],
                "cash": row[3],
                "equity": row[4],
                "buying_power": row[5],
                "portfolio_value": row[6],
                "daily_pnl": row[7],
                "trades_count": row[8],
                "positions_count": row[9],
            }
        )

    return snapshots


@router.get("/analytics/events/{date}")
async def get_events(date: str):
    """Get all system events for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, event_type, severity, message, details
            FROM system_events
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC
        """,
            (date,),
        )
        rows = cursor.fetchall()

    events = []
    for row in rows:
        events.append(
            {
                "id": row[0],
                "timestamp": row[1],
                "event_type": row[2],
                "severity": row[3],
                "message": row[4],
                "details": row[5],
            }
        )

    return events


@router.get("/analytics/bars/{date}")
async def get_bars(date: str):
    """Get all bar summaries for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, date, ticker, open, high, low, close, volume, vwap, bar_count
            FROM bar_summaries
            WHERE date = ?
            ORDER BY ticker ASC
        """,
            (date,),
        )
        rows = cursor.fetchall()

    bars = []
    for row in rows:
        bars.append(
            {
                "id": row[0],
                "date": row[1],
                "ticker": row[2],
                "open": row[3],
                "high": row[4],
                "low": row[5],
                "close": row[6],
                "volume": row[7],
                "vwap": row[8],
                "bar_count": row[9],
            }
        )

    return bars


@router.get("/analytics/watchlist/{date}")
async def get_watchlist(date: str):
    """Get watchlist for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, date, ticker, gap_pct, pm_volume, float_shares, scan_time
            FROM watchlist
            WHERE date = ?
            ORDER BY gap_pct DESC NULLS LAST
        """,
            (date,),
        )
        rows = cursor.fetchall()

    watchlist = []
    for row in rows:
        watchlist.append(
            {
                "id": row[0],
                "date": row[1],
                "ticker": row[2],
                "gap_pct": row[3],
                "pm_volume": row[4],
                "float_shares": row[5],
                "scan_time": row[6],
            }
        )

    return watchlist


@router.get("/analytics/slippage/{date}")
async def get_slippage_by_date(date: str):
    """Get slippage data for a specific date."""
    from statistics import median
    from collections import defaultdict

    try:
        with db._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, order_id, ticker, strategy, side, event_type,
                       signal_price, fill_price, filled_qty, slip_bp, status,
                       cum_dollar_vol
                FROM order_events
                WHERE DATE(timestamp) = ? AND event_type IN ('fill', 'partial_fill')
                      AND slip_bp IS NOT NULL
                ORDER BY timestamp DESC
            """,
                (date,),
            )
            rows = cursor.fetchall()

        # Build result rows
        result_rows = []
        for row in rows:
            fill_price = row[8]
            filled_qty = row[9]
            dollar_amount = (fill_price or 0) * (filled_qty or 0)
            # Participation rate: our fill size / 2-min cumulative dollar volume
            participation_rate = None
            # cum_dollar_vol is column index 12 in the query
            cum_dollar_vol = row[12] if len(row) > 12 else None
            if dollar_amount > 0 and cum_dollar_vol and cum_dollar_vol > 0:
                participation_rate = dollar_amount / cum_dollar_vol
            result_rows.append(
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "order_id": row[2],
                    "ticker": row[3],
                    "strategy": row[4],
                    "side": row[5],
                    "event_type": row[6],
                    "signal_price": row[7],
                    "fill_price": fill_price,
                    "filled_qty": filled_qty,
                    "slip_bp": row[10],
                    "status": row[11],
                    "dollar_amount": dollar_amount,
                    "participation_rate": participation_rate,
                }
            )

        # Aggregate stats
        slips = [r["slip_bp"] for r in result_rows if r["slip_bp"] is not None]
        dollar_total = sum(r["dollar_amount"] for r in result_rows)
        dollar_slip_cost = sum(
            (r["slip_bp"] or 0) / 10_000 * r["dollar_amount"] for r in result_rows
        )

        stats = {
            "date": date,
            "n_fills": len(result_rows),
            "avg_slip_bp": (sum(slips) / len(slips)) if slips else None,
            "median_slip_bp": median(slips) if slips else None,
            "max_slip_bp": max(slips) if slips else None,
            "min_slip_bp": min(slips) if slips else None,
            "p95_slip_bp": (
                sorted(slips)[int(len(slips) * 0.95)] if len(slips) >= 20 else None
            ),
            "dollar_volume": dollar_total,
            "realized_cost": dollar_slip_cost,
        }

        # Per-strategy breakdown
        groups = defaultdict(list)
        for r in result_rows:
            groups[r["strategy"]].append(r)

        by_strategy = []
        for strat, items in groups.items():
            s = [it["slip_bp"] for it in items if it["slip_bp"] is not None]
            buys = [it for it in items if it["side"] == "buy"]
            sells = [it for it in items if it["side"] == "sell"]
            dvol = sum(it["dollar_amount"] for it in items)
            cost = sum(
                (it["slip_bp"] or 0) / 10_000 * it["dollar_amount"] for it in items
            )
            by_strategy.append(
                {
                    "strategy": strat,
                    "n": len(items),
                    "n_buys": len(buys),
                    "n_sells": len(sells),
                    "avg_slip_bp": sum(s) / len(s) if s else None,
                    "median_slip_bp": median(s) if s else None,
                    "min_slip_bp": min(s) if s else None,
                    "max_slip_bp": max(s) if s else None,
                    "avg_buy_slip_bp": (sum(it["slip_bp"] for it in buys) / len(buys))
                    if buys
                    else None,
                    "avg_sell_slip_bp": (
                        sum(it["slip_bp"] for it in sells) / len(sells)
                    )
                    if sells
                    else None,
                    "dollar_volume": dvol,
                    "realized_cost": cost,
                }
            )

        by_strategy.sort(key=lambda x: -x["dollar_volume"])

        return {
            "stats": stats,
            "by_strategy": by_strategy,
            "rows": result_rows,
        }
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"Failed to get slippage: {e}")
        return {"stats": {}, "by_strategy": [], "rows": []}


@router.get("/analytics/feed_comparison/{date}")
async def get_feed_comparison(date: str):
    """Get Tradier vs Alpaca IEX bar comparison for a date."""
    return db.get_feed_comparison(date)


@router.get("/analytics/trades/details/{date}")
async def get_trade_details(date: str):
    """Get detailed trade information for a specific date."""
    return db.get_trades_by_date(date)


@router.get("/analytics/intraday_discoveries/{date}")
async def get_intraday_discoveries(date: str):
    """Get intraday top gainers for a specific date, sorted by % change."""
    return db.get_intraday_discoveries_by_date(date)


@router.get("/analytics/halts/{date}")
async def get_halt_status(date: str):
    """Get halt status for tickers on a specific date.

    Returns a dict mapping ticker -> {halted: bool, reason: str, resumed: bool}.
    """
    import csv
    import os

    halts_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "halts.csv",
    )
    result = {}
    if not os.path.exists(halts_csv):
        return result
    try:
        with open(halts_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("halt_date") == date:
                    ticker = row.get("ticker", "")
                    if ticker:
                        result[ticker] = {
                            "halted": not bool(row.get("resume_time")),
                            "reason": row.get("reason", ""),
                            "resumed": bool(row.get("resume_time")),
                        }
    except Exception:
        pass
    return result
