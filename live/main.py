"""
Live Paper Trading: Combined Strategy Engine (12 strategies from Optuna trial 432)
===================================================================================
Entry point for live paper trading on Alpaca using all enabled strategies.
Includes embedded dashboard server on port 8000.

Schedule:
  7:00 AM ET  - First pre-market scan
  7:30 AM ET  - Second pre-market scan
  9:00 AM ET  - Third scan (more PM volume data)
  9:25 AM ET  - Rescan
  9:27 AM ET  - Final scan, lock watchlist
  9:30 AM ET  - Market open scan + strategy engine starts
  3:45 PM ET  - EOD close all positions
  4:00 PM ET  - Daily summary, dashboard stays alive

Data Feed:
  SIP (delayed 15-min on free tier): Bars arrive at wall_clock = bar_timestamp + 15min.
  Strategy uses the bar's timestamp as "market time", not wall clock. So a 9:30 AM bar
  arrives at 9:45 AM wall time, but is processed as "9:30 AM market time."

  This gives us:
  - Complete market coverage (SIP = all exchanges consolidated)
  - Consistent bar delivery (no IEX feed gaps)
  - G/L strategies work correctly (see first bars at "market open")
  - Paper trading validation matches backtest (same data quality)

Usage:
  python -m live.main              # run full day + dashboard
  python -m live.main --scan-only  # only run scanner, print watchlist
  python -m live.main --dry-run    # process bars but don't place orders
  python -m live.main --no-dash    # run without dashboard
"""

import sys
import time
import logging
import argparse
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from live.scanner import PreMarketScanner
from live.streamer import BarStreamer
from live.engine_combined import CombinedEngine
from live.executor import OrderExecutor
from live.halt_monitor import HaltMonitor
from live.intraday_scanner import IntradayScanner
from live.event_logger import log_event, set_db

ET = ZoneInfo("America/New_York")


def recover_open_positions(engine, executor, candidates, log):
    """
    On restart: detect any open Alpaca positions and restore engine state.
    - Fetches today's 2-min bars for each open position
    - Immediately sells if stop (-15%) or target (+25%) already breached
    - Otherwise restores engine state and adds to candidates for stream
    Returns updated candidates list.
    """
    try:
        open_positions = executor.get_positions()
    except Exception as e:
        log.warning(f"Recovery: could not fetch positions: {e}")
        log_event("recovery_error", "error", f"Recovery: could not fetch positions: {e}")
        return candidates

    if not open_positions:
        log.info("RECOVERY: No open Alpaca positions")
        return candidates

    # Reconcile with persistent state file
    log.info(
        f"RECOVERY: Found {len(open_positions)} open Alpaca positions — reconciling with state file"
    )
    recovered = engine.position_state.reconcile_with_alpaca(open_positions)

    if not recovered:
        log.info("RECOVERY: All Alpaca positions already tracked (no action needed)")
        return candidates

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED
    from datetime import time as dt_time, date as dt_date

    hist_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    candidate_tickers = {c["ticker"] for c in candidates}
    today = datetime.now(ET).date()
    market_open_dt = datetime.combine(today, dt_time(9, 30), tzinfo=ET)

    for ticker, state in recovered.items():
      try:
        entry_price = state["entry_price"]
        current_price = state["current_price"]
        strategy = state["strategy"]
        has_metadata = state["has_metadata"]
        change_pct = (current_price / entry_price - 1) * 100

        log.info(
            f"RECOVERY: {ticker} ({strategy}) | "
            f"entry=${entry_price:.2f} current=${current_price:.2f} ({change_pct:+.1f}%) | "
            f"metadata={'YES' if has_metadata else 'NO (orphan)'}"
        )

        # Check if stop/target already breached during downtime
        stop_price = state.get("stop_price", entry_price * 0.95)
        target_price = state.get("target_price", entry_price * 1.10)

        if current_price <= stop_price:
            log.warning(
                f"RECOVERY: {ticker} STOP BREACHED (${current_price:.2f} <= ${stop_price:.2f}) — selling immediately"
            )
            executor.sell(ticker, reason="RECOVERY_STOP")
            engine.position_state.remove_position(ticker)
            continue

        if current_price >= target_price:
            log.info(
                f"RECOVERY: {ticker} TARGET REACHED (${current_price:.2f} >= ${target_price:.2f}) — taking profit"
            )
            executor.sell(ticker, reason="RECOVERY_TARGET")
            engine.position_state.remove_position(ticker)
            continue

        # Fetch today's 2-min bars
        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame(2, TimeFrameUnit.Minute),
                start=market_open_dt,
                end=datetime.now(ET),
                feed=ALPACA_FEED,
            )
            bars_resp = hist_client.get_stock_bars(req)
            df = bars_resp.df.reset_index() if not bars_resp.df.empty else None
        except Exception as e:
            log.warning(f"RECOVERY: Could not fetch bars for {ticker}: {e}")
            log_event("recovery_error", "warning", f"RECOVERY: Could not fetch bars for {ticker}: {e}")
            df = None

        # Inject bars into engine
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                t = row["timestamp"]
                if hasattr(t, "to_pydatetime"):
                    t = t.to_pydatetime()
                engine.bar_data.setdefault(ticker, []).append(
                    {
                        "timestamp": t,
                        "Open": float(row["open"]),
                        "High": float(row["high"]),
                        "Low": float(row["low"]),
                        "Close": float(row["close"]),
                        "Volume": float(row["volume"]),
                    }
                )

        # Restore engine runtime state
        engine.active_positions.add(ticker)
        engine.position_entry[ticker] = {
            "entry_price": entry_price,
            "shares": state["shares"],
            "cost": state["cost"],
            "strategy": strategy,
            "entry_time": state.get("entry_time"),
        }
        # Pre-populate last_states so engine won't re-enter
        engine.last_states[ticker] = {
            "ticker": ticker,
            "entry_price": entry_price,
            "shares": state["shares"],
            "strategy": strategy,
            "exit_price": None,
        }

        # Add to candidates if not already there
        if ticker not in candidate_tickers:
            candidates = list(candidates) + [
                {
                    "ticker": ticker,
                    "gap_pct": 0,
                    "pm_volume": 0,
                    "premarket_high": entry_price,
                    "prev_close": entry_price,
                    "float_shares": None,
                }
            ]
            engine.picks.append(
                {
                    "ticker": ticker,
                    "gap_pct": 0,
                    "market_open": entry_price,
                    "premarket_high": entry_price,
                    "prev_close": entry_price,
                    "pm_volume": 0,
                    "market_hour_candles": None,
                }
            )
            candidate_tickers.add(ticker)

        if has_metadata:
            log.info(
                f"RECOVERY: {ticker} restored with full monitoring | "
                f"strategy={strategy} stop=${stop_price:.2f} target=${target_price:.2f} "
                f"trail={state.get('trail_pct', 0):.1f}% peak=${state.get('peak_price', 0):.2f}"
            )
        else:
            log.warning(
                f"RECOVERY: {ticker} orphan — defensive monitoring | "
                f"stop=${stop_price:.2f} (-5%) target=${target_price:.2f} (+10%) trail=1%"
            )
      except Exception as e:
        log.error(f"RECOVERY: Failed to recover {ticker}: {e}", exc_info=True)
        log_event("recovery_failed", "error", f"RECOVERY: Failed to recover {ticker}: {e}")
        try:
            if ticker in executor.get_positions():
                log.warning(f"RECOVERY: Selling {ticker} due to recovery failure")
                executor.sell(ticker, reason="RECOVERY_FAILED")
                engine.position_state.remove_position(ticker)
        except Exception as sell_err:
            log.error(f"RECOVERY: Emergency sell failed for {ticker}: {sell_err}")
            log_event("emergency_sell_failed", "critical", f"RECOVERY: Emergency sell failed for {ticker}: {sell_err}")
        continue

    # Save account snapshot after recovery
    try:
        account = executor.get_account()
        engine.db.save_account_snapshot(
            "recovery",
            cash=account.cash,
            equity=account.equity,
            buying_power=account.buying_power,
            portfolio_value=account.portfolio_value,
            positions_count=len(recovered),
        )
        # Log recovery event
        if recovered:
            details = ", ".join(
                f"{t} ({s.get('strategy', '?')})" for t, s in recovered.items()
            )
            engine.db.log_system_event(
                "recovery",
                "info",
                f"Recovery completed: {len(recovered)} positions restored and being tracked: {details}",
            )
    except Exception as e:
        log.warning(f"Failed to save recovery snapshot: {e}")
        log_event("snapshot_error", "warning", f"Failed to save recovery snapshot: {e}")

    return candidates


class _ETFormatter(logging.Formatter):
    """Logging formatter that stamps times in US/Eastern (ET)."""

    def formatTime(self, record, datefmt=None):
        from datetime import datetime as _dt

        ct = _dt.fromtimestamp(record.created, tz=ET)
        return ct.strftime(datefmt or "%H:%M:%S")


_FILE_HANDLER = None  # module-level so _rotate_log_for_new_day can swap it


def setup_logging():
    global _FILE_HANDLER
    today = datetime.now(ET).strftime("%Y-%m-%d")
    log_dir = "logs"
    import os

    os.makedirs(log_dir, exist_ok=True)

    fmt = "%(asctime)s ET [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_ETFormatter(fmt, datefmt=datefmt))

    fh = logging.FileHandler(f"{log_dir}/{today}_live.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_ETFormatter(fmt, datefmt=datefmt))
    _FILE_HANDLER = fh

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(fh)
    return logging.getLogger("live.main")


def _rotate_log_for_new_day():
    """Swap the FileHandler to today's log file. Fixes 2026-06-25 rollover bug
    where all today's traffic was written to yesterday's named log file."""
    global _FILE_HANDLER
    if _FILE_HANDLER is None:
        return
    today = datetime.now(ET).strftime("%Y-%m-%d")
    log_dir = "logs"
    new_path = f"{log_dir}/{today}_live.log"
    if getattr(_FILE_HANDLER, "baseFilename", "").endswith(f"{today}_live.log"):
        return  # already rotated (idempotent on same-day calls)
    fmt = "%(asctime)s ET [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    new_fh = logging.FileHandler(new_path, encoding="utf-8")
    new_fh.setLevel(logging.DEBUG)
    new_fh.setFormatter(_ETFormatter(fmt, datefmt=datefmt))
    root = logging.getLogger()
    root.removeHandler(_FILE_HANDLER)
    try:
        _FILE_HANDLER.close()
    except Exception:
        pass
    root.addHandler(new_fh)
    _FILE_HANDLER = new_fh

    return logging.getLogger("live.main")


def wait_until(target_time, log):
    """Wait until a specific ET time today."""
    now = datetime.now(ET)
    target = datetime.combine(now.date(), target_time, tzinfo=ET)
    if target <= now:
        return
    wait_secs = (target - now).total_seconds()
    log.info(f"Waiting {wait_secs / 60:.1f} minutes until {target_time}...")
    time.sleep(max(0, wait_secs))


def _next_trading_day_prep_time(executor, log):
    """Return ET datetime to wake at for the next trading day's pre-market.
    Uses Alpaca's get_clock() so weekends and exchange holidays are handled
    automatically. We wake 2.5 hours before next_open, which lands near 7:00 ET
    on regular days (9:30 - 2:30) and gives the scan loop time to do its
    7:00 / 7:30 / 8:30 / 9:00 / 9:25 / 9:27 / 9:30 sweep.
    """
    from datetime import time as dt_time

    try:
        clock = executor.client.get_clock()
        next_open = clock.next_open
        if next_open.tzinfo is None:
            next_open = next_open.replace(tzinfo=ET)
        next_open_et = next_open.astimezone(ET)
        wake_at = next_open_et - timedelta(hours=2, minutes=30)
        log.info(f"Next session opens at {next_open_et}; waking at {wake_at}")
        return wake_at
    except Exception as e:
        log.warning(
            f"Alpaca clock fetch failed ({e}); falling back to next weekday 7:00 ET"
        )
        log_event("api_error", "warning", f"Alpaca clock fetch failed: {e}")
        now = datetime.now(ET)
        nxt = now.date() + timedelta(days=1)
        while nxt.weekday() >= 5:  # 5=Sat, 6=Sun
            nxt = nxt + timedelta(days=1)
        return datetime.combine(nxt, dt_time(7, 0), tzinfo=ET)


def _sleep_until(target_dt, log):
    """Sleep in 5-minute chunks until target_dt (ET-aware). Allows KeyboardInterrupt."""
    while True:
        now = datetime.now(ET)
        remaining = (target_dt - now).total_seconds()
        if remaining <= 0:
            return
        chunk = min(remaining, 300.0)
        time.sleep(chunk)


def start_dashboard(engine, executor, candidates, port=8000):
    """Start the FastAPI dashboard server in a background thread."""
    import uvicorn
    from dashboard.backend.app import app, bridge

    # Wire up the bridge to the live engine
    bridge.engine = engine
    bridge.executor = executor
    bridge.scanner_candidates = candidates

    log = logging.getLogger("dashboard")
    log.info(f"Starting dashboard on http://localhost:{port}")

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    # Disable signal handlers — they only work in the main thread on Linux
    server.install_signal_handlers = lambda: None

    def _run():
        try:
            server.run()
        except Exception as e:
            log.error(f"Dashboard server crashed: {e}", exc_info=True)
            log_event("dashboard_error", "error", f"Dashboard server crashed: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _run_one_day(executor, args, log):
    """Run a single trading session: scan -> engine -> stream -> EOD.
    Returns a summary dict (or None when no trading happened, e.g. empty
    watchlist or --scan-only).
    """
    from datetime import time as dt_time

    # Phase 1: Pre-market scan loop (9:00 → 9:25 → 9:27 → lock)
    scanner = PreMarketScanner()

    def do_scan(label):
        log.info(f"Phase 1: Pre-market scan ({label})")
        try:
            c = scanner.get_final_watchlist()
        except Exception as e:
            log.error(f"Scanner error during {label}: {e}", exc_info=True)
            log_event("scanner_error", "error", f"Scanner error during {label}: {e}")
            return []
        if c:
            for x in c:
                float_str = (
                    f"{x['float_shares'] / 1e6:.1f}M"
                    if x.get("float_shares")
                    else "N/A"
                )
                log.info(
                    f"  {x['ticker']}: gap={x['gap_pct']:.1f}%, "
                    f"PM vol={x['pm_volume']:,}, float={float_str}"
                )
            # Update dashboard with latest scan results immediately
            if not args.no_dash:
                try:
                    from dashboard.backend.app import bridge

                    bridge.scanner_candidates = c
                except Exception:
                    pass
        else:
            log.info("  No candidates found")
        return c

    # Scan schedule: 7:00, 7:30, 9:00, 9:25, 9:27, 9:30
    scan_times = [
        (dt_time(7, 0), "7:00"),
        (dt_time(7, 30), "7:30"),
        (dt_time(8, 30), "8:30"),
        (dt_time(9, 0), "9:00"),
        (dt_time(9, 25), "9:25"),
        (dt_time(9, 27), "9:27 FINAL"),
        (dt_time(9, 30), "9:30"),
    ]

    now = datetime.now(ET)
    if now >= datetime.combine(now.date(), dt_time(9, 30), tzinfo=ET):
        # Restarted during market hours — scan immediately
        candidates = do_scan("RESTART")
    else:
        # Wait for the first scan time we haven't passed yet
        candidates = []
        for i, (scan_t, label) in enumerate(scan_times):
            now = datetime.now(ET)
            target = datetime.combine(now.date(), scan_t, tzinfo=ET)
            if now < target:
                wait_until(scan_t, log)
            # Skip scan times that are already past
            if datetime.now(ET) >= target:
                new_candidates = do_scan(label)
                if new_candidates:
                    candidates = new_candidates

            if args.scan_only and i == 0:
                log.info("--scan-only mode. Exiting.")
                return None

    if args.scan_only:
        log.info("--scan-only mode. Exiting.")
        return None

    if not candidates:
        log.info(
            "No candidates after all scans for %s — will retry next trading day.",
            datetime.now(ET).strftime("%Y-%m-%d"),
        )
        return None

    trading_day = datetime.now(ET).date()
    log.info(f"Watchlist locked for {trading_day}: {len(candidates)} candidates")

    # Phase 2: Initialize combined engine
    # 2026-06-23: instantiate TradingStream for real-time fill notifications
    # (replaces the 15s polling that lost 85 untracked fills on GITS today).
    fill_stream = None
    try:
        from live.trading_stream import FillStream
        from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER

        fill_stream = FillStream(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
        fill_stream.start_async()
        log.info("FillStream: started — fills will be tracked via Alpaca TradingStream")
    except Exception as e:
        log.error(
            f"FillStream init failed, falling back to legacy 30s polling: {e}",
            exc_info=True,
        )
        log_event("stream_init_error", "error", f"FillStream init failed: {e}")
        fill_stream = None

    engine = CombinedEngine(executor, fill_stream=fill_stream)
    set_db(engine.db)
    engine.initialize_watchlist(candidates)

    # Save watchlist to database
    engine.db.save_watchlist(candidates, scan_time=datetime.now(ET))

    # Log system startup event
    engine.db.log_system_event(
        "startup",
        "info",
        f"Engine started with {len(candidates)} candidates for {trading_day}",
    )

    # Recover any open positions from a previous session/crash
    candidates = recover_open_positions(engine, executor, candidates, log)

    if args.dry_run:
        log.info("--dry-run mode: will process bars but not place orders")

    # Phase 3: Wire engine into dashboard bridge
    if not args.no_dash:
        from dashboard.backend.app import bridge

        bridge.engine = engine
        bridge.scanner_candidates = candidates
        bridge.scan_date = trading_day

    # Phase 4: Wait for market open (stream starts at 9:30)
    now = datetime.now(ET)
    if now < datetime.combine(now.date(), dt_time(9, 30), tzinfo=ET):
        wait_until(dt_time(9, 30), log)

    # Phase 5: Start bar streams (Tradier primary + Alpaca IEX for comparison)
    log.info("Starting bar streams")
    symbols = [c["ticker"] for c in candidates]

    # Feed comparison logger — writes to feed_comparison DB table
    from live.feed_logger import FeedLogger, SOURCE_TRADIER, SOURCE_ALPACA_IEX

    feed_logger = FeedLogger(engine.db)

    def on_bar_with_ws(symbol, bar):
        """Process bar in engine AND broadcast to dashboard."""
        engine.on_bar(symbol, bar)
        try:
            from dashboard.backend.services.ws_manager import ws_manager

            ws_manager.broadcast_sync(
                {
                    "type": "bar",
                    "symbol": symbol,
                    "data": {
                        k: (str(v) if hasattr(v, "isoformat") else v)
                        for k, v in bar.items()
                    },
                }
            )
        except Exception:
            pass

    _last_bar_time = {}

    def on_tradier_bar(symbol, bar):
        """Tradier bar: drives the engine + log for comparison."""
        _last_bar_time[symbol] = bar["timestamp"]
        feed_logger.log(SOURCE_TRADIER, symbol, bar)
        on_bar_with_ws(symbol, bar) if not args.no_dash else engine.on_bar(symbol, bar)

    def on_alpaca_bar(symbol, bar):
        """Alpaca IEX bar: comparison logging only (no engine calls)."""
        feed_logger.log(SOURCE_ALPACA_IEX, symbol, bar)

    # Primary feed: Tradier (SIP-level real-time)
    tradier_streamer = None
    from config.settings import TRADIER_API_KEY

    try:
        from live.tradier_streamer import TradierStreamer

        tradier_streamer = TradierStreamer(
            api_key=TRADIER_API_KEY, on_2min_bar=on_tradier_bar
        )
        tradier_streamer.subscribe(symbols)
        tradier_stream_thread = tradier_streamer.start_async()
        log.info("TradierStreamer: started as primary data feed")
    except Exception as e:
        log.error(
            f"TradierStreamer failed to start, falling back to Alpaca IEX as primary: {e}"
        )
        log_event("tradier_init_error", "error", f"TradierStreamer failed to start: {e}")
        tradier_streamer = None

    # Secondary feed: Alpaca IEX — comparison logging only when Tradier is up,
    # or primary engine feed when Tradier is unavailable.
    if tradier_streamer is not None:
        alpaca_streamer = BarStreamer(on_2min_bar=on_alpaca_bar)
    else:
        alpaca_streamer = BarStreamer(
            on_2min_bar=on_bar_with_ws if not args.no_dash else engine.on_bar
        )
        log.warning(
            "Tradier unavailable — Alpaca IEX is driving the engine (volume caps unreliable)"
        )

    # Keep `streamer` pointing at whichever streamer owns the engine feed
    # (halt-resume monitor calls streamer.add_symbol)
    streamer = tradier_streamer if tradier_streamer is not None else alpaca_streamer

    alpaca_streamer.subscribe(symbols)
    stream_thread = alpaca_streamer.start_async()

    # Tag the engine with which feed is driving it
    if tradier_streamer is not None:
        engine.active_feed = "tradier"
        engine.feed_fallback = False
    else:
        engine.active_feed = "alpaca_iex"
        engine.feed_fallback = True

    log.info("Streaming... waiting for signals (primary feed: %s)", engine.active_feed)

    # ── Pre-market health check ──────────────────────────────────────────
    from live.health_check import run_health_check, check_streaming_health

    healthy, health_results = run_health_check(engine, executor, tradier_streamer, fill_stream)
    if not args.no_dash:
        from dashboard.backend.app import bridge as _bridge
        _bridge.system_health = health_results
    if not healthy:
        log.error("PRE-MARKET HEALTH CHECK FAILED — see system events for details")
    else:
        log.info("Pre-market health check: ALL PASSED")

    # Phase 5b: Halt-resume monitor (intraday discovery channel)
    halt_monitor = None
    try:
        from config.settings import (
            HALT_MONITOR_ENABLED,
            HALT_MIN_PRICE,
            HALT_MAX_PRICE,
            HALT_MAX_FLOAT,
            HALT_REASONS_TRADED,
            HALT_POLL_INTERVAL_SECS,
            FLOAT_DATA,
        )
    except ImportError:
        HALT_MONITOR_ENABLED = False

    if HALT_MONITOR_ENABLED:
        _whitelist = {r.upper() for r in HALT_REASONS_TRADED}

        def _halt_eligible(ev):
            reason = (ev.reason or "").strip().upper()
            if reason not in _whitelist:
                return False
            rp = ev.resume_price
            if rp is None or not (HALT_MIN_PRICE <= rp <= HALT_MAX_PRICE):
                return False
            fl = FLOAT_DATA.get(ev.ticker)
            if fl is not None and fl > HALT_MAX_FLOAT:
                return False
            return True

        def _on_resume(ev):
            # Runs on the halt-monitor poller thread.
            ticker = ev.ticker
            try:
                added = engine.on_intraday_addition(ticker, ev, source="halt_resume")
                if added:
                    streamer.add_symbol(ticker)
                    # Also subscribe Alpaca IEX side for comparison logging
                    if tradier_streamer is not None:
                        alpaca_streamer.add_symbol(ticker)
                    if not args.no_dash:
                        try:
                            from dashboard.backend.app import bridge

                            evs = getattr(bridge, "halt_events", [])
                            evs.append(
                                {
                                    "ticker": ticker,
                                    "reason": ev.reason,
                                    "halt_price": ev.halt_price,
                                    "resume_price": ev.resume_price,
                                    "resume_ts": str(ev.resume_dt)
                                    if ev.resume_dt
                                    else None,
                                }
                            )
                            bridge.halt_events = evs[-200:]
                        except Exception:
                            pass
            except Exception as e:
                log.exception("halt-monitor on_resume(%s) failed: %s", ticker, e)
                log_event("halt_monitor_error", "error", f"Halt-monitor on_resume({ticker}) failed: {e}")

        halt_monitor = HaltMonitor(
            on_resume=_on_resume,
            eligibility_fn=_halt_eligible,
            poll_interval_secs=HALT_POLL_INTERVAL_SECS,
        )
        halt_monitor.start()
        if not args.no_dash:
            try:
                from dashboard.backend.app import bridge

                bridge.halt_monitor = halt_monitor
            except Exception:
                pass
        log.info(
            "Halt-resume monitor enabled (reasons=%s, $%g-$%g, max_float=%dM)",
            sorted(_whitelist),
            HALT_MIN_PRICE,
            HALT_MAX_PRICE,
            HALT_MAX_FLOAT // 1_000_000,
        )
    else:
        log.info("Halt-resume monitor DISABLED (HALT_MONITOR_ENABLED=false)")

    intraday_scanner = IntradayScanner(engine)

    # REST polling fallback: fetch 2-min bars for tickers that the IEX stream misses
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED

    _last_warn_time = 0

    def check_feed_health():
        """Check if Tradier WebSocket feed has missed recently expected bars."""
        nonlocal _last_warn_time
        now_et = datetime.now(ET)

        # Only check during market hours (9:30 to 16:00 ET)
        from datetime import time as dt_time

        if not (dt_time(9, 30) <= now_et.time() <= dt_time(16, 0)):
            return

        stale_symbols = []
        for sym in symbols:
            last_seen = _last_bar_time.get(sym)
            if last_seen is None:
                stale_symbols.append(sym)
                continue
            try:
                last_seen_et = last_seen.astimezone(ET)
            except Exception:
                last_seen_et = last_seen
            if now_et - last_seen_et > timedelta(minutes=6):
                stale_symbols.append(sym)

        if len(symbols) > 0 and (len(stale_symbols) / len(symbols)) > 0.5:
            now_ts = time.time()
            if now_ts - _last_warn_time > 120:  # throttle warning to every 2 minutes
                _last_warn_time = now_ts
                msg = f"Tradier feed warning: {len(stale_symbols)}/{len(symbols)} symbols are stale (>6 min without bars)"
                log.warning(msg)
                import json

                engine.db.log_system_event(
                    "feed_warning",
                    "warning",
                    msg,
                    details=json.dumps({"stale_symbols": stale_symbols}),
                )

    # Phase 6: Main loop - monitor until EOD
    stop_keys = [
        v for k, v in engine.params.items() if k.endswith("_stop_pct") and v > 0
    ]
    target_keys = [
        v
        for k, v in engine.params.items()
        if k.endswith("_target_pct")
        or k.endswith("_target1_pct")
        or k.endswith("_target2_pct")
    ]
    RECOVERY_STOP = -(max(stop_keys) if stop_keys else 12.0)
    RECOVERY_TARGET = max(target_keys) if target_keys else 23.0
    _loop_count = 0
    try:
        while True:
            now = datetime.now(ET)

            # Sanity: trading day rolled over while we were still in the loop —
            # something is very wrong; abort so the outer rollover can re-init.
            if now.date() != trading_day:
                log.error(
                    "DATE MISMATCH: trading_day=%s but now=%s — exiting loop",
                    trading_day,
                    now.date(),
                )
                break

            # EOD close at 3:45 PM
            if now.hour == 15 and now.minute >= 45:
                log.info("EOD: Closing all positions")
                engine.eod_close()

                # Save EOD account snapshot
                try:
                    account = executor.get_account()
                    engine.db.save_account_snapshot(
                        "market_close",
                        cash=account.cash,
                        equity=account.equity,
                        buying_power=account.buying_power,
                        portfolio_value=account.portfolio_value,
                        daily_pnl=engine.daily_pnl,
                        trades_count=len(engine.trades_today),
                        positions_count=len(engine.active_positions),
                    )
                    log.info(
                        f"EOD snapshot saved: equity=${account.equity:.2f} P&L=${engine.daily_pnl:.2f}"
                    )
                except Exception as e:
                    log.warning(f"Failed to save EOD snapshot: {e}")
                    log_event("snapshot_error", "warning", f"Failed to save EOD snapshot: {e}")

                # Save bar summaries
                try:
                    engine.save_bar_summaries()
                except Exception as e:
                    log.warning(f"Failed to save bar summaries: {e}")
                    log_event("db_error", "warning", f"Failed to save bar summaries: {e}")

                # Log shutdown event
                engine.db.log_system_event(
                    "shutdown",
                    "info",
                    f"Market close: {len(engine.trades_today)} trades, P&L=${engine.daily_pnl:.2f}",
                )

                break

            # After market close
            if now.hour >= 16:
                break

            # Check Tradier feed health every 2 minutes
            _loop_count += 1
            if _loop_count % 4 == 0:  # every 4 × 30s = 2 min
                check_feed_health()

            # Monitor recovered positions with hard stop/target
            for active in engine.active_positions:
                if engine.position_entry.get(active, {}).get("strategy") == "RECOVERED":
                    try:
                        positions = executor.get_positions()
                        for pos in positions:
                            if pos.symbol == active:
                                entry = engine.position_entry[active]["entry_price"]
                                current = float(pos.current_price)
                                chg = (current / entry - 1) * 100
                                if chg <= RECOVERY_STOP:
                                    log.warning(
                                        f"RECOVERY STOP HIT: {active} {chg:+.1f}% — selling"
                                    )
                                    executor.sell(active, reason="RECOVERY_STOP")
                                    engine.active_positions.discard(active)
                                elif chg >= RECOVERY_TARGET:
                                    log.info(
                                        f"RECOVERY TARGET HIT: {active} {chg:+.1f}% — selling"
                                    )
                                    executor.sell(active, reason="RECOVERY_TARGET")
                                    engine.active_positions.discard(active)
                    except Exception:
                        pass

            intraday_scanner.poll()

            # ── Periodic streaming health check (every ~5 min = 10 × 30s) ──
            _loop_count += 1
            if _loop_count % 10 == 0:
                try:
                    s_healthy, s_results = check_streaming_health(engine, tradier_streamer, halt_monitor)
                    if not args.no_dash:
                        from dashboard.backend.app import bridge as _bridge2
                        _bridge2.system_health = s_results
                    if not s_healthy:
                        log.error("STREAMING HEALTH CHECK FAILED — see system events")
                except Exception:
                    pass

            time.sleep(30)

    except KeyboardInterrupt:
        log.info("Interrupted by user")

    # Stop streamer and print daily summary
    streamer.flush_pending()
    streamer.stop()
    if halt_monitor is not None:
        halt_monitor.stop()

    summary = engine.get_summary()
    log.info("=" * 60)
    log.info(f"DAILY SUMMARY ({trading_day})")
    log.info(f"  Trades: {summary['trades']}")
    log.info(f"  Wins:   {summary['wins']}")
    log.info(f"  Losses: {summary['losses']}")
    log.info(f"  PnL:    ${summary['daily_pnl']:+,.2f}")
    for t in summary["trade_details"]:
        log.info(
            f"    {t['ticker']} ({t.get('strategy', '?')}): ${t['pnl']:+,.2f} "
            f"({t['reason']}) ${t['entry_price']:.2f} -> ${t['exit_price']:.2f}"
        )
    log.info("=" * 60)
    return summary


def run(args):
    """Main loop: start dashboard once, then run one trading session per
    weekday until interrupted. Sleeps between sessions until ~2.5h before
    next Alpaca open (i.e. ~7:00 ET on regular days).
    """
    log = setup_logging()
    log.info("=" * 60)
    log.info("Combined Strategy Live Paper Trading (Trial 1215 (G+L modified))")
    log.info("=" * 60)

    # Check account
    executor = OrderExecutor()
    acct = executor.get_account()
    log.info(
        f"Account: cash=${float(acct.cash):,.2f}, "
        f"buying_power=${float(acct.buying_power):,.2f}"
    )

    # Start dashboard once — it persists across trading days
    if not args.no_dash:
        start_dashboard(None, executor, [], port=args.port)
        log.info(f"Dashboard running at http://localhost:{args.port}")

    while True:
        try:
            _run_one_day(executor, args, log)
        except KeyboardInterrupt:
            log.info("Interrupted by user — shutting down.")
            return
        except Exception as e:
            log.error(f"Trading session crashed: {e}", exc_info=True)
            log_event("session_crash", "critical", f"Trading session crashed: {e}")

        if args.scan_only:
            return

        # Wait until ~2.5h before next Alpaca open (Mon–Fri, skips holidays)
        wake_at = _next_trading_day_prep_time(executor, log)
        log.info("=" * 60)
        log.info(f"Session done. Sleeping until {wake_at} ET.")
        log.info("=" * 60)
        try:
            _sleep_until(wake_at, log)
        except KeyboardInterrupt:
            log.info("Interrupted during overnight wait — shutting down.")
            return

        _rotate_log_for_new_day()
        log.info("=" * 60)
        log.info(f"New trading day: {datetime.now(ET).strftime('%Y-%m-%d')}")
        log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Combined Strategy Live Paper Trading")
    parser.add_argument("--scan-only", action="store_true", help="Only run scanner")
    parser.add_argument(
        "--dry-run", action="store_true", help="Process bars but don't trade"
    )
    parser.add_argument(
        "--no-dash", action="store_true", help="Disable dashboard server"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Dashboard port (default: 8000)"
    )
    args = parser.parse_args()
    try:
        run(args)
    except Exception as e:
        log = logging.getLogger("live.main")
        log.error(f"Fatal error: {e}", exc_info=True)
        log_event("fatal_error", "critical", f"Fatal error: {e}")
        # Keep dashboard alive even after a crash
        if not args.no_dash:
            log.info("Dashboard still running despite error...")
            try:
                while True:
                    time.sleep(300)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
