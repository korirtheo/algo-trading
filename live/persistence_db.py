"""
SQLite-based persistence for trades, daily state, and position tracking.

Survives crashes, restarts, and container recreation.
All timestamps stored in ET timezone.
Database located at logs/trading.db (mounted volume, persists across restarts)
"""

import sqlite3
import json
import os
from datetime import datetime, date
from pytz import timezone
from contextlib import contextmanager
import logging

log = logging.getLogger(__name__)
ET = timezone("US/Eastern")


class TradingDatabase:
    """Persistent storage for all trading state."""

    def __init__(self, db_path="logs/trading.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()
        log.info(f"TradingDatabase initialized: {db_path}")

    @contextmanager
    def _conn(self):
        """Thread-safe connection context."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_schema(self):
        """Create tables if they don't exist."""
        with self._conn() as conn:
            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    reason TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    deployed_amount REAL,
                    stop_price REAL,
                    target_price REAL,
                    peak_price REAL,
                    trail_pct REAL,
                    time_limit_min INTEGER,
                    hold_time_min REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)"
            )

            # Daily state table (exit prices, done strategies, signal times)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    exit_price REAL,
                    is_done INTEGER DEFAULT 0,
                    signal_time TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker, strategy)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_daily_state_date ON daily_state(date)"
            )

            # Position state table (open positions with metadata)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    strategy TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    peak_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    trail_pct REAL NOT NULL,
                    time_limit_min INTEGER NOT NULL,
                    entry_time TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Watchlist/candidates table - what was available each day
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    gap_pct REAL NOT NULL,
                    pm_volume INTEGER NOT NULL,
                    premarket_high REAL NOT NULL,
                    prev_close REAL NOT NULL,
                    float_shares INTEGER,
                    scan_time TEXT NOT NULL,
                    UNIQUE(date, ticker)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_watchlist_date ON watchlist(date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_watchlist_ticker ON watchlist(ticker)"
            )

            # Signals table - all entry signals (taken and rejected)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_price REAL NOT NULL,
                    gap_pct REAL,
                    action TEXT NOT NULL,
                    reason TEXT,
                    order_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker, date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_action ON signals(action)"
            )

            # Order events table - full order lifecycle
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    side TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    signal_price REAL,
                    fill_price REAL,
                    qty INTEGER,
                    filled_qty INTEGER,
                    status TEXT,
                    slip_bp REAL,
                    cum_dollar_vol REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_events_date ON order_events(date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_events_order ON order_events(order_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_events_ticker ON order_events(ticker, date)"
            )

            # Slippage-calibration columns (added 2026-08-12). Added via ALTER
            # TABLE so existing DBs migrate in place; fresh DBs get them via the
            # CREATE above plus this no-op guard.
            _oe_cols = {r["name"] for r in conn.execute("PRAGMA table_info(order_events)")}
            for _col, _dtype in (
                ("veff_adj", "REAL"),
                ("participation_eff", "REAL"),
                ("modeled_slip_bp", "REAL"),
                ("pm_gate", "REAL"),
                ("pm_volume", "REAL"),
                ("bar_range_pct", "REAL"),
            ):
                if _col not in _oe_cols:
                    conn.execute(f"ALTER TABLE order_events ADD COLUMN {_col} {_dtype}")

            # Account snapshots table - equity curve
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    buying_power REAL NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_pnl REAL,
                    trades_count INTEGER,
                    positions_count INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_account_snapshots_date ON account_snapshots(date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_account_snapshots_type ON account_snapshots(snapshot_type)"
            )

            # System events table - crashes, restarts, recoveries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type)"
            )

            # Bar summaries table - daily OHLCV per ticker
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bar_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    vwap REAL,
                    bar_count INTEGER NOT NULL,
                    first_bar_time TEXT NOT NULL,
                    last_bar_time TEXT NOT NULL,
                    UNIQUE(date, ticker)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bar_summaries_date ON bar_summaries(date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bar_summaries_ticker ON bar_summaries(ticker)"
            )

            # Feed comparison table - Tradier vs Alpaca IEX 2-min bars
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feed_comparison (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    bar_time TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, bar_time, ticker, source)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feed_comp_date ON feed_comparison(date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feed_comp_ticker ON feed_comparison(ticker, date)"
            )

            # Intraday discoveries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS intraday_discoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    price REAL NOT NULL,
                    percent_change REAL NOT NULL,
                    source TEXT NOT NULL,
                    gap_pct REAL,
                    cumulative_volume REAL,
                    volume REAL,
                    float_shares REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_intraday_discoveries_date ON intraday_discoveries(date)"
            )

            # Migration: add new columns to existing intraday_discoveries table
            for col, typ in [
                ("gap_pct", "REAL"),
                ("cumulative_volume", "REAL"),
                ("volume", "REAL"),
                ("float_shares", "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE intraday_discoveries ADD COLUMN {col} {typ}")
                except Exception:
                    pass  # column already exists

            # Ticker stream data table — stores raw Tradier stream values
            # for later strategy development (not shown on analytics page)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_stream_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    price REAL,
                    volume REAL,
                    float_shares REAL,
                    gap_pct REAL,
                    prev_close REAL,                    day_high REAL,
                    day_low REAL,
                    avg_volume REAL,
                    source TEXT,
                    raw_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ticker_stream_date ON ticker_stream_data(date)"
            )

            # Daily post-close reconcile table — compares the live day's watchlist,
            # signals, and trades against a SIP-feed replay of the same day through
            # the deployed engine. Populated by scripts/reconcile/post_close_reconcile.py
            # (cron, after extended hours). Drives the "Daily Reconcile" dashboard view.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_reconcile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    run_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    watchlist_count INTEGER,
                    sip_fetched INTEGER,
                    sip_missing TEXT,
                    bt_trades INTEGER,
                    bt_pnl REAL,
                    live_trades INTEGER,
                    live_pnl REAL,
                    live_signals INTEGER,
                    match_count INTEGER,
                    live_only_count INTEGER,
                    bt_only_count INTEGER,
                    match_tickers TEXT,
                    live_only_tickers TEXT,
                    bt_only_tickers TEXT,
                    summary TEXT,
                    details TEXT,
                    status TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_daily_reconcile_date ON daily_reconcile(date)"
            )

    # ===== TRADES =====

    def save_trade(
        self,
        trade_date,
        ticker,
        strategy,
        entry_price,
        exit_price,
        shares,
        pnl,
        pnl_pct,
        reason,
        entry_time,
        exit_time,
        deployed_amount=None,
        stop_price=None,
        target_price=None,
        peak_price=None,
        trail_pct=None,
        time_limit_min=None,
    ):
        """Save a completed trade with execution details."""
        # Ensure times are in ET
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time.replace("+00:00", ""))
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time.replace("+00:00", ""))

        if entry_time.tzinfo is None:
            entry_time = ET.localize(entry_time)
        else:
            entry_time = entry_time.astimezone(ET)

        if exit_time.tzinfo is None:
            exit_time = ET.localize(exit_time)
        else:
            exit_time = exit_time.astimezone(ET)

        # Calculate hold time in minutes
        hold_time_min = (exit_time - entry_time).total_seconds() / 60.0

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    date, ticker, strategy, entry_price, exit_price, shares,
                    pnl, pnl_pct, reason, entry_time, exit_time, deployed_amount,
                    stop_price, target_price, peak_price, trail_pct, time_limit_min, hold_time_min
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade_date,
                    ticker,
                    strategy,
                    entry_price,
                    exit_price,
                    shares,
                    pnl,
                    pnl_pct,
                    reason,
                    entry_time.isoformat(),
                    exit_time.isoformat(),
                    deployed_amount,
                    stop_price,
                    target_price,
                    peak_price,
                    trail_pct,
                    time_limit_min,
                    hold_time_min,
                ),
            )
        log.info(
            f"Trade saved: {ticker} ({strategy}) {reason} P&L=${pnl:.2f} hold={hold_time_min:.1f}min"
        )

    def get_trades_by_date(self, trade_date):
        """Get all trades for a specific date."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM trades WHERE date = ? ORDER BY exit_time DESC
            """,
                (trade_date,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_trades_today(self):
        """Get today's trades."""
        today = datetime.now(ET).date().isoformat()
        return self.get_trades_by_date(today)

    # ===== DAILY RECONCILE (post-close SIP replay vs live) =====

    def save_daily_reconcile(self, data: dict) -> int:
        """Insert or update a daily reconcile record (one per date, newest wins)."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO daily_reconcile (
                    date, watchlist_count, sip_fetched, sip_missing,
                    bt_trades, bt_pnl, live_trades, live_pnl, live_signals,
                    match_count, live_only_count, bt_only_count,
                    match_tickers, live_only_tickers, bt_only_tickers,
                    summary, details, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO NOTHING
            """, (
                data.get("date"),
                data.get("watchlist_count"),
                data.get("sip_fetched"),
                data.get("sip_missing"),
                data.get("bt_trades"),
                data.get("bt_pnl"),
                data.get("live_trades"),
                data.get("live_pnl"),
                data.get("live_signals"),
                data.get("match_count"),
                data.get("live_only_count"),
                data.get("bt_only_count"),
                data.get("match_tickers"),
                data.get("live_only_tickers"),
                data.get("bt_only_tickers"),
                data.get("summary"),
                data.get("details"),
                data.get("status"),
            ))
            # keep one row per date — delete older runs for the same date
            conn.execute("""
                DELETE FROM daily_reconcile WHERE date=? AND id NOT IN (
                    SELECT id FROM daily_reconcile WHERE date=?
                    ORDER BY id DESC LIMIT 1
                )
            """, (data.get("date"), data.get("date")))
            row = conn.execute(
                "SELECT id FROM daily_reconcile WHERE date=? ORDER BY id DESC LIMIT 1",
                (data.get("date"),),
            ).fetchone()
            return row["id"] if row else None

    def get_daily_reconcile(self, reconcile_date=None, limit=30):
        """Get reconcile records, newest first. date=None -> latest across all days."""
        with self._conn() as conn:
            if reconcile_date:
                rows = conn.execute(
                    "SELECT * FROM daily_reconcile WHERE date=? ORDER BY id DESC LIMIT 1",
                    (reconcile_date,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM daily_reconcile ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    # ===== DAILY STATE =====

    def record_exit(self, ticker, strategy, exit_price):
        """Record exit price for re-entry floor."""
        today = datetime.now(ET).date().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO daily_state (date, ticker, strategy, exit_price)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(date, ticker, strategy)
                DO UPDATE SET exit_price=?, updated_at=CURRENT_TIMESTAMP
            """,
                (today, ticker, strategy, exit_price, exit_price),
            )

    def mark_done(self, ticker, strategy):
        """Mark a strategy as done for the day."""
        today = datetime.now(ET).date().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO daily_state (date, ticker, strategy, is_done)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(date, ticker, strategy)
                DO UPDATE SET is_done=1, updated_at=CURRENT_TIMESTAMP
            """,
                (today, ticker, strategy),
            )

    def is_done(self, ticker, strategy):
        """Check if strategy is done for the day."""
        today = datetime.now(ET).date().isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT is_done FROM daily_state
                WHERE date=? AND ticker=? AND strategy=?
            """,
                (today, ticker, strategy),
            )
            row = cursor.fetchone()
            return row and row["is_done"] == 1

    def record_signal_time(self, ticker, strategy, timestamp):
        """Record first signal time for time limit enforcement."""
        today = datetime.now(ET).date().isoformat()

        # Convert to ET
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("+00:00", ""))
        if timestamp.tzinfo is None:
            timestamp = ET.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(ET)

        with self._conn() as conn:
            # Only insert if not exists (first signal only)
            conn.execute(
                """
                INSERT INTO daily_state (date, ticker, strategy, signal_time)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(date, ticker, strategy)
                DO UPDATE SET signal_time = CASE
                    WHEN signal_time IS NULL THEN ?
                    ELSE signal_time
                END
            """,
                (today, ticker, strategy, timestamp.isoformat(), timestamp.isoformat()),
            )

    def get_signal_time(self, ticker, strategy):
        """Get first signal time for time limit check."""
        today = datetime.now(ET).date().isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT signal_time FROM daily_state
                WHERE date=? AND ticker=? AND strategy=?
            """,
                (today, ticker, strategy),
            )
            row = cursor.fetchone()
            if row and row["signal_time"]:
                return datetime.fromisoformat(row["signal_time"])
            return None

    def get_exit_price(self, ticker, strategy):
        """Get exit price for re-entry floor check."""
        today = datetime.now(ET).date().isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT exit_price FROM daily_state
                WHERE date=? AND ticker=? AND strategy=?
            """,
                (today, ticker, strategy),
            )
            row = cursor.fetchone()
            return row["exit_price"] if row and row["exit_price"] else None

    def get_daily_state_summary(self):
        """Get summary of today's state."""
        today = datetime.now(ET).date().isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN exit_price IS NOT NULL THEN 1 ELSE 0 END) as exits,
                    SUM(CASE WHEN is_done = 1 THEN 1 ELSE 0 END) as done,
                    SUM(CASE WHEN signal_time IS NOT NULL THEN 1 ELSE 0 END) as signals
                FROM daily_state WHERE date=?
            """,
                (today,),
            )
            row = cursor.fetchone()
            return (
                dict(row) if row else {"total": 0, "exits": 0, "done": 0, "signals": 0}
            )

    # ===== POSITION STATE =====

    def save_position(
        self,
        ticker,
        strategy,
        entry_price,
        shares,
        cost,
        peak_price,
        stop_price,
        target_price,
        trail_pct,
        time_limit_min,
        entry_time,
    ):
        """Save open position state."""
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time.replace("+00:00", ""))
        if entry_time.tzinfo is None:
            entry_time = ET.localize(entry_time)
        else:
            entry_time = entry_time.astimezone(ET)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO position_state (
                    ticker, strategy, entry_price, shares, cost, peak_price,
                    stop_price, target_price, trail_pct, time_limit_min, entry_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    strategy=?, shares=?, peak_price=?, updated_at=CURRENT_TIMESTAMP
            """,
                (
                    ticker,
                    strategy,
                    entry_price,
                    shares,
                    cost,
                    peak_price,
                    stop_price,
                    target_price,
                    trail_pct,
                    time_limit_min,
                    entry_time.isoformat(),
                    strategy,
                    shares,
                    peak_price,
                ),
            )

    def update_position_peak(self, ticker, new_peak):
        """Update peak price for trailing stop."""
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE position_state SET peak_price=?, updated_at=CURRENT_TIMESTAMP
                WHERE ticker=?
            """,
                (new_peak, ticker),
            )

    def get_position(self, ticker):
        """Get position state."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM position_state WHERE ticker=?", (ticker,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def remove_position(self, ticker):
        """Remove position after exit."""
        with self._conn() as conn:
            conn.execute("DELETE FROM position_state WHERE ticker=?", (ticker,))

    def get_all_positions(self):
        """Get all open positions."""
        with self._conn() as conn:
            cursor = conn.execute("SELECT * FROM position_state")
            return [dict(row) for row in cursor.fetchall()]

    # ===== WATCHLIST =====

    def save_watchlist(self, candidates, scan_time=None):
        """Save today's watchlist/candidates."""
        today = datetime.now(ET).date().isoformat()
        if scan_time is None:
            scan_time = datetime.now(ET)
        elif isinstance(scan_time, str):
            scan_time = datetime.fromisoformat(scan_time)
        if scan_time.tzinfo is None:
            scan_time = ET.localize(scan_time)
        else:
            scan_time = scan_time.astimezone(ET)

        with self._conn() as conn:
            for cand in candidates:
                conn.execute(
                    """
                    INSERT INTO watchlist (date, ticker, gap_pct, pm_volume, premarket_high,
                                         prev_close, float_shares, scan_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date, ticker) DO UPDATE SET
                        gap_pct=?, pm_volume=?, premarket_high=?, prev_close=?, float_shares=?
                """,
                    (
                        today,
                        cand["ticker"],
                        cand["gap_pct"],
                        cand["pm_volume"],
                        cand["premarket_high"],
                        cand["prev_close"],
                        cand.get("float_shares"),
                        scan_time.isoformat(),
                        cand["gap_pct"],
                        cand["pm_volume"],
                        cand["premarket_high"],
                        cand["prev_close"],
                        cand.get("float_shares"),
                    ),
                )
        log.info(f"Watchlist saved: {len(candidates)} candidates for {today}")

    def get_watchlist_by_date(self, date):
        """Get watchlist for a specific date."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM watchlist WHERE date=? ORDER BY gap_pct DESC", (date,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ===== SIGNALS =====

    def log_signal(
        self,
        ticker,
        strategy,
        signal_price,
        action,
        reason=None,
        order_id=None,
        gap_pct=None,
        timestamp=None,
    ):
        """Log an entry signal (taken or rejected).

        Args:
            action: 'TAKEN', 'REJECTED', 'SKIPPED'
            reason: Why taken/rejected (e.g., 'already_in_position', 'done', 'reentry_floor', 'vol_cap')
        """
        today = datetime.now(ET).date().isoformat()
        if timestamp is None:
            timestamp = datetime.now(ET)
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("+00:00", ""))
        if timestamp.tzinfo is None:
            timestamp = ET.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(ET)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO signals (date, timestamp, ticker, strategy, signal_price,
                                   gap_pct, action, reason, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    today,
                    timestamp.isoformat(),
                    ticker,
                    strategy,
                    signal_price,
                    gap_pct,
                    action,
                    reason,
                    order_id,
                ),
            )

    def get_signals_by_date(self, date, action=None):
        """Get all signals for a date, optionally filtered by action."""
        with self._conn() as conn:
            if action:
                cursor = conn.execute(
                    "SELECT * FROM signals WHERE date=? AND action=? ORDER BY timestamp",
                    (date, action),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM signals WHERE date=? ORDER BY timestamp", (date,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_signal_stats(self, date):
        """Get signal statistics for a date."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT
                    action,
                    COUNT(*) as count,
                    COUNT(DISTINCT ticker) as unique_tickers
                FROM signals
                WHERE date=?
                GROUP BY action
            """,
                (date,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # ===== ORDER EVENTS =====

    def log_order_event(
        self,
        order_id,
        ticker,
        strategy,
        side,
        event_type,
        signal_price=None,
        fill_price=None,
        qty=None,
        filled_qty=None,
        status=None,
        slip_bp=None,
        cum_dollar_vol=None,
        timestamp=None,
        veff_adj=None,
        participation_eff=None,
        modeled_slip_bp=None,
        pm_gate=None,
        pm_volume=None,
        bar_range_pct=None,
    ):
        """Log an order lifecycle event.

        Args:
            event_type: 'placed', 'partial_fill', 'fill', 'canceled', 'rejected', 'expired'
            slip_bp: Slippage in basis points (positive = paid more, negative = got better)
            veff_adj / participation_eff / modeled_slip_bp / pm_gate /
            pm_volume / bar_range_pct: slippage-calibration context (see
            executor._reconcile_fill_async). Optional — never affects trading.
        """
        today = datetime.now(ET).date().isoformat()
        if timestamp is None:
            timestamp = datetime.now(ET)
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("+00:00", ""))
        if timestamp.tzinfo is None:
            timestamp = ET.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(ET)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO order_events (date, timestamp, order_id, ticker, strategy, side,
                                        event_type, signal_price, fill_price, qty, filled_qty,
                                        status, slip_bp, cum_dollar_vol,
                                        veff_adj, participation_eff, modeled_slip_bp,
                                        pm_gate, pm_volume, bar_range_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    today,
                    timestamp.isoformat(),
                    order_id,
                    ticker,
                    strategy,
                    side,
                    event_type,
                    signal_price,
                    fill_price,
                    qty,
                    filled_qty,
                    status,
                    slip_bp,
                    cum_dollar_vol,
                    veff_adj,
                    participation_eff,
                    modeled_slip_bp,
                    pm_gate,
                    pm_volume,
                    bar_range_pct,
                ),
            )

    def get_order_events(self, order_id):
        """Get all events for a specific order."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM order_events WHERE order_id=? ORDER BY timestamp",
                (order_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_order_events_by_date(self, date, side=None):
        """Get all order events for a date."""
        with self._conn() as conn:
            if side:
                cursor = conn.execute(
                    "SELECT * FROM order_events WHERE date=? AND side=? ORDER BY timestamp",
                    (date, side),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM order_events WHERE date=? ORDER BY timestamp",
                    (date,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_slippage_stats(self, date):
        """Get slippage statistics for a date."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT
                    side,
                    COUNT(*) as fills,
                    AVG(slip_bp) as avg_slip_bp,
                    MIN(slip_bp) as min_slip_bp,
                    MAX(slip_bp) as max_slip_bp,
                    SUM(CASE WHEN fill_price IS NOT NULL AND signal_price IS NOT NULL
                        THEN ABS(fill_price - signal_price) * filled_qty
                        ELSE 0 END) as total_slip_cost
                FROM order_events
                WHERE date=? AND event_type IN ('fill', 'partial_fill') AND slip_bp IS NOT NULL
                GROUP BY side
            """,
                (date,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # ===== ACCOUNT SNAPSHOTS =====

    def save_account_snapshot(
        self,
        snapshot_type,
        cash,
        equity,
        buying_power,
        portfolio_value,
        daily_pnl=None,
        trades_count=None,
        positions_count=None,
        timestamp=None,
    ):
        """Save an account snapshot.

        Args:
            snapshot_type: 'market_open', 'market_close', 'intraday', 'recovery'
        """
        today = datetime.now(ET).date().isoformat()
        if timestamp is None:
            timestamp = datetime.now(ET)
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("+00:00", ""))
        if timestamp.tzinfo is None:
            timestamp = ET.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(ET)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO account_snapshots (date, timestamp, snapshot_type, cash, equity,
                                              buying_power, portfolio_value, daily_pnl,
                                              trades_count, positions_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    today,
                    timestamp.isoformat(),
                    snapshot_type,
                    cash,
                    equity,
                    buying_power,
                    portfolio_value,
                    daily_pnl,
                    trades_count,
                    positions_count,
                ),
            )

    def get_account_snapshots_by_date(self, date):
        """Get all account snapshots for a date."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM account_snapshots WHERE date=? ORDER BY timestamp",
                (date,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_equity_curve(self, start_date=None, end_date=None):
        """Get equity curve data."""
        with self._conn() as conn:
            if start_date and end_date:
                cursor = conn.execute(
                    """
                    SELECT date, snapshot_type, equity, daily_pnl
                    FROM account_snapshots
                    WHERE date >= ? AND date <= ?
                    ORDER BY timestamp
                """,
                    (start_date, end_date),
                )
            elif start_date:
                cursor = conn.execute(
                    """
                    SELECT date, snapshot_type, equity, daily_pnl
                    FROM account_snapshots
                    WHERE date >= ?
                    ORDER BY timestamp
                """,
                    (start_date,),
                )
            else:
                cursor = conn.execute("""
                    SELECT date, snapshot_type, equity, daily_pnl
                    FROM account_snapshots
                    ORDER BY timestamp
                """)
            return [dict(row) for row in cursor.fetchall()]

    # ===== SYSTEM EVENTS =====

    def log_system_event(
        self, event_type, severity, message, details=None, timestamp=None
    ):
        """Log a system event.

        Args:
            event_type: 'startup', 'shutdown', 'crash', 'recovery', 'reconnect', 'error'
            severity: 'info', 'warning', 'error', 'critical'
        """
        if timestamp is None:
            timestamp = datetime.now(ET)
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("+00:00", ""))
        if timestamp.tzinfo is None:
            timestamp = ET.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(ET)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO system_events (timestamp, event_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?)
            """,
                (timestamp.isoformat(), event_type, severity, message, details),
            )

    def get_system_events(
        self, start_time=None, end_time=None, event_type=None, severity=None
    ):
        """Get system events with optional filters."""
        with self._conn() as conn:
            query = "SELECT * FROM system_events WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(
                    start_time
                    if isinstance(start_time, str)
                    else start_time.isoformat()
                )
            if end_time:
                query += " AND timestamp <= ?"
                params.append(
                    end_time if isinstance(end_time, str) else end_time.isoformat()
                )
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            if severity:
                query += " AND severity = ?"
                params.append(severity)

            query += " ORDER BY timestamp DESC"
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # ===== BAR SUMMARIES =====

    def save_bar_summary(
        self,
        ticker,
        open_price,
        high,
        low,
        close,
        volume,
        vwap,
        bar_count,
        first_bar_time,
        last_bar_time,
        date=None,
    ):
        """Save daily bar summary for a ticker."""
        if date is None:
            date = datetime.now(ET).date().isoformat()

        if isinstance(first_bar_time, str):
            first_bar_time = datetime.fromisoformat(
                first_bar_time.replace("+00:00", "")
            )
        if isinstance(last_bar_time, str):
            last_bar_time = datetime.fromisoformat(last_bar_time.replace("+00:00", ""))

        if first_bar_time.tzinfo is None:
            first_bar_time = ET.localize(first_bar_time)
        else:
            first_bar_time = first_bar_time.astimezone(ET)

        if last_bar_time.tzinfo is None:
            last_bar_time = ET.localize(last_bar_time)
        else:
            last_bar_time = last_bar_time.astimezone(ET)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO bar_summaries (date, ticker, open, high, low, close, volume,
                                          vwap, bar_count, first_bar_time, last_bar_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, ticker) DO UPDATE SET
                    open=?, high=?, low=?, close=?, volume=?, vwap=?, bar_count=?,
                    first_bar_time=?, last_bar_time=?
            """,
                (
                    date,
                    ticker,
                    open_price,
                    high,
                    low,
                    close,
                    volume,
                    vwap,
                    bar_count,
                    first_bar_time.isoformat(),
                    last_bar_time.isoformat(),
                    open_price,
                    high,
                    low,
                    close,
                    volume,
                    vwap,
                    bar_count,
                    first_bar_time.isoformat(),
                    last_bar_time.isoformat(),
                ),
            )

    def get_bar_summary(self, ticker, date):
        """Get bar summary for a specific ticker and date."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM bar_summaries WHERE ticker=? AND date=?", (ticker, date)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_bar_summaries_by_date(self, date):
        """Get all bar summaries for a date."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM bar_summaries WHERE date=? ORDER BY ticker", (date,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ===== UTILITIES =====

    def vacuum(self):
        """Optimize database file size."""
        with self._conn() as conn:
            conn.execute("VACUUM")

    def get_stats(self):
        """Get database statistics."""
        with self._conn() as conn:
            trades_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            daily_state_count = conn.execute(
                "SELECT COUNT(*) FROM daily_state"
            ).fetchone()[0]
            positions_count = conn.execute(
                "SELECT COUNT(*) FROM position_state"
            ).fetchone()[0]
            signals_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            order_events_count = conn.execute(
                "SELECT COUNT(*) FROM order_events"
            ).fetchone()[0]
            snapshots_count = conn.execute(
                "SELECT COUNT(*) FROM account_snapshots"
            ).fetchone()[0]
            system_events_count = conn.execute(
                "SELECT COUNT(*) FROM system_events"
            ).fetchone()[0]
            watchlist_count = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[
                0
            ]
            bar_summaries_count = conn.execute(
                "SELECT COUNT(*) FROM bar_summaries"
            ).fetchone()[0]

            return {
                "db_path": self.db_path,
                "trades": trades_count,
                "daily_state_entries": daily_state_count,
                "open_positions": positions_count,
                "signals": signals_count,
                "order_events": order_events_count,
                "account_snapshots": snapshots_count,
                "system_events": system_events_count,
                "watchlist_entries": watchlist_count,
                "bar_summaries": bar_summaries_count,
            }

    # ===== FEED COMPARISON =====

    def log_feed_bar(
        self,
        date: str,
        bar_time: str,
        ticker: str,
        source: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ):
        """Insert or replace a 2-min bar row from one data source."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO feed_comparison
                    (date, bar_time, ticker, source, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, bar_time, ticker, source)
                DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume,
                    created_at=CURRENT_TIMESTAMP
            """,
                (date, bar_time, ticker, source, open_, high, low, close, volume),
            )

    def get_feed_comparison(self, date: str) -> list[dict]:
        """
        Return all bars for a date from both sources, joined so each row
        has both tradier and alpaca_iex values side-by-side where available.
        """
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT
                    bar_time, ticker,
                    MAX(CASE WHEN source='tradier'    THEN open  END) AS t_open,
                    MAX(CASE WHEN source='tradier'    THEN high  END) AS t_high,
                    MAX(CASE WHEN source='tradier'    THEN low   END) AS t_low,
                    MAX(CASE WHEN source='tradier'    THEN close END) AS t_close,
                    MAX(CASE WHEN source='tradier'    THEN volume END) AS t_volume,
                    MAX(CASE WHEN source='alpaca_iex' THEN open  END) AS a_open,
                    MAX(CASE WHEN source='alpaca_iex' THEN high  END) AS a_high,
                    MAX(CASE WHEN source='alpaca_iex' THEN low   END) AS a_low,
                    MAX(CASE WHEN source='alpaca_iex' THEN close END) AS a_close,
                    MAX(CASE WHEN source='alpaca_iex' THEN volume END) AS a_volume
                FROM feed_comparison
                WHERE date = ?
                GROUP BY bar_time, ticker
                ORDER BY bar_time ASC, ticker ASC
            """,
                (date,),
            )
            rows = cursor.fetchall()

        result = []
        for row in rows:
            (
                bar_time,
                ticker,
                t_open,
                t_high,
                t_low,
                t_close,
                t_vol,
                a_open,
                a_high,
                a_low,
                a_close,
                a_vol,
            ) = row

            close_diff_bp = None
            if t_close and a_close and a_close != 0:
                close_diff_bp = round((t_close - a_close) / a_close * 10_000, 1)

            vol_diff_pct = None
            if t_vol and a_vol and a_vol != 0:
                vol_diff_pct = round((t_vol - a_vol) / a_vol * 100, 1)

            result.append(
                {
                    "bar_time": bar_time,
                    "ticker": ticker,
                    "tradier_open": t_open,
                    "tradier_high": t_high,
                    "tradier_low": t_low,
                    "tradier_close": t_close,
                    "tradier_volume": t_vol,
                    "alpaca_open": a_open,
                    "alpaca_high": a_high,
                    "alpaca_low": a_low,
                    "alpaca_close": a_close,
                    "alpaca_volume": a_vol,
                    "close_diff_bp": close_diff_bp,
                    "vol_diff_pct": vol_diff_pct,
                }
            )
        return result

    def log_intraday_discovery(self, ticker, price, percent_change, timestamp, source,
                                gap_pct=None, cumulative_volume=None, volume=None,
                                float_shares=None):
        """Log an intraday discovery."""
        today = datetime.now(ET).date().isoformat()
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("+00:00", ""))
        if timestamp.tzinfo is None:
            timestamp = ET.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(ET)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO intraday_discoveries (date, timestamp, ticker, price, percent_change, source,
                                                  gap_pct, cumulative_volume, volume, float_shares)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, ticker) DO NOTHING
            """,
                (today, timestamp.isoformat(), ticker, price, percent_change, source,
                 gap_pct, cumulative_volume, volume, float_shares),
            )

    def get_intraday_discoveries_by_date(self, date):
        """Get intraday top gainers for a date, sorted by % change descending."""
        with self._conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM intraday_discoveries WHERE date=? ORDER BY percent_change DESC",
                (date,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def log_ticker_stream_data(self, ticker, price=None, volume=None, float_shares=None,
                                gap_pct=None, prev_close=None, day_high=None, day_low=None,
                                avg_volume=None, source=None, raw_json=None):
        """Store Tradier ticker stream values for later strategy development.

        This captures the full snapshot of a ticker's data at discovery time,
        including values not shown on the analytics page.
        """
        today = datetime.now(ET).date().isoformat()
        timestamp = datetime.now(ET).isoformat()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ticker_stream_data
                    (date, timestamp, ticker, price, volume, float_shares,
                     gap_pct, prev_close, day_high, day_low, avg_volume, source, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, ticker) DO UPDATE SET
                    price=excluded.price, volume=excluded.volume,
                    float_shares=excluded.float_shares, gap_pct=excluded.gap_pct,
                    prev_close=excluded.prev_close, day_high=excluded.day_high,
                    day_low=excluded.day_low, avg_volume=excluded.avg_volume,
                    raw_json=excluded.raw_json, timestamp=excluded.timestamp
            """,
                (today, timestamp, ticker, price, volume, float_shares,
                 gap_pct, prev_close, day_high, day_low, avg_volume, source, raw_json),
            )
