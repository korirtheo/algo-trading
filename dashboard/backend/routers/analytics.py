"""Analytics API - Database query endpoints."""
from fastapi import APIRouter
from datetime import datetime
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "..")))

from live.persistence_db import TradingDatabase

router = APIRouter()
db = TradingDatabase()


@router.get("/analytics/signals/{date}")
async def get_signals(date: str):
    """Get all signals for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, date, ticker, strategy, signal_price, action, reason, order_id, gap_pct
            FROM signals
            WHERE date = ?
            ORDER BY date DESC
        """, (date,))
        rows = cursor.fetchall()

    signals = []
    for row in rows:
        signals.append({
            "id": row[0],
            "date": row[1],
            "ticker": row[2],
            "strategy": row[3],
            "signal_price": row[4],
            "action": row[5],
            "reason": row[6],
            "order_id": row[7],
            "gap_pct": row[8]
        })

    return signals


@router.get("/analytics/orders/{date}")
async def get_orders(date: str):
    """Get all order events for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, order_id, ticker, strategy, side, event_type,
                   signal_price, fill_price, filled_qty, slip_bp, status
            FROM order_events
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC
        """, (date,))
        rows = cursor.fetchall()

    orders = []
    for row in rows:
        orders.append({
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
            "status": row[11]
        })

    return orders


@router.get("/analytics/snapshots/{date}")
async def get_snapshots(date: str):
    """Get all account snapshots for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, snapshot_type, cash, equity, buying_power,
                   portfolio_value, daily_pnl, trades_count, positions_count
            FROM account_snapshots
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp ASC
        """, (date,))
        rows = cursor.fetchall()

    snapshots = []
    for row in rows:
        snapshots.append({
            "id": row[0],
            "timestamp": row[1],
            "snapshot_type": row[2],
            "cash": row[3],
            "equity": row[4],
            "buying_power": row[5],
            "portfolio_value": row[6],
            "daily_pnl": row[7],
            "trades_count": row[8],
            "positions_count": row[9]
        })

    return snapshots


@router.get("/analytics/events/{date}")
async def get_events(date: str):
    """Get all system events for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, event_type, severity, message, details
            FROM system_events
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC
        """, (date,))
        rows = cursor.fetchall()

    events = []
    for row in rows:
        events.append({
            "id": row[0],
            "timestamp": row[1],
            "event_type": row[2],
            "severity": row[3],
            "message": row[4],
            "details": row[5]
        })

    return events


@router.get("/analytics/bars/{date}")
async def get_bars(date: str):
    """Get all bar summaries for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, date, ticker, open, high, low, close, volume, vwap, bar_count
            FROM bar_summaries
            WHERE date = ?
            ORDER BY ticker ASC
        """, (date,))
        rows = cursor.fetchall()

    bars = []
    for row in rows:
        bars.append({
            "id": row[0],
            "date": row[1],
            "ticker": row[2],
            "open": row[3],
            "high": row[4],
            "low": row[5],
            "close": row[6],
            "volume": row[7],
            "vwap": row[8],
            "bar_count": row[9]
        })

    return bars


@router.get("/analytics/watchlist/{date}")
async def get_watchlist(date: str):
    """Get watchlist for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, date, ticker, gap_pct, pm_volume, float_shares, scan_time
            FROM watchlist
            WHERE date = ?
            ORDER BY gap_pct DESC NULLS LAST
        """, (date,))
        rows = cursor.fetchall()

    watchlist = []
    for row in rows:
        watchlist.append({
            "id": row[0],
            "date": row[1],
            "ticker": row[2],
            "gap_pct": row[3],
            "pm_volume": row[4],
            "float_shares": row[5],
            "scan_time": row[6]
        })

    return watchlist


@router.get("/analytics/trades/details/{date}")
async def get_trade_details(date: str):
    """Get trades with execution details for a specific date."""
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, ticker, strategy, entry_price, exit_price, shares, pnl, pnl_pct,
                   reason, entry_time, exit_time, deployed_amount, stop_price, target_price,
                   peak_price, trail_pct, time_limit_min, hold_time_min
            FROM trades
            WHERE date = ?
            ORDER BY entry_time DESC
        """, (date,))
        rows = cursor.fetchall()

    trades = []
    for row in rows:
        trades.append({
            "id": row[0],
            "ticker": row[1],
            "strategy": row[2],
            "entry_price": row[3],
            "exit_price": row[4],
            "shares": row[5],
            "pnl": row[6],
            "pnl_pct": row[7],
            "reason": row[8],
            "entry_time": row[9],
            "exit_time": row[10],
            "deployed_amount": row[11],
            "stop_price": row[12],
            "target_price": row[13],
            "peak_price": row[14],
            "trail_pct": row[15],
            "time_limit_min": row[16],
            "hold_time_min": row[17]
        })

    return trades
