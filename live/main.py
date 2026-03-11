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
        return candidates

    if not open_positions:
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

    for pos in open_positions:
        ticker = pos.symbol
        entry_price = float(pos.avg_entry_price)
        qty = float(pos.qty)
        current_price = float(pos.current_price)
        change_pct = (current_price / entry_price - 1) * 100

        log.info(f"RECOVERY: Found open position {ticker} | "
                 f"entry=${entry_price:.2f} | current=${current_price:.2f} | "
                 f"change={change_pct:+.1f}%")

        # Immediate safety checks — derived from worst-case strategy params
        stop_keys = [v for k, v in engine.params.items() if k.endswith("_stop_pct") and v > 0]
        target_keys = [v for k, v in engine.params.items() if k.endswith("_target_pct") or k.endswith("_target1_pct") or k.endswith("_target2_pct")]
        HARD_STOP_PCT = -(max(stop_keys) if stop_keys else 12.0)
        HARD_TARGET_PCT = max(target_keys) if target_keys else 23.0

        if change_pct <= HARD_STOP_PCT:
            log.warning(f"RECOVERY STOP: {ticker} down {change_pct:.1f}% — selling immediately")
            executor.sell(ticker, reason="RECOVERY_STOP")
            continue

        if change_pct >= HARD_TARGET_PCT:
            log.info(f"RECOVERY TARGET: {ticker} up {change_pct:.1f}% — taking profit")
            executor.sell(ticker, reason="RECOVERY_TARGET")
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
            df = None

        # Inject bars into engine
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                t = row["timestamp"]
                if hasattr(t, "to_pydatetime"):
                    t = t.to_pydatetime()
                engine.bar_data.setdefault(ticker, []).append({
                    "timestamp": t,
                    "Open": float(row["open"]),
                    "High": float(row["high"]),
                    "Low": float(row["low"]),
                    "Close": float(row["close"]),
                    "Volume": float(row["volume"]),
                })

        # Restore engine position state
        engine.active_position = ticker
        engine.position_entry[ticker] = {
            "entry_price": entry_price,
            "shares": qty,
            "cost": entry_price * qty,
            "strategy": "RECOVERED",
            "entry_time": datetime.now(ET),
        }
        # Pre-populate last_states so engine won't re-enter
        engine.last_states[ticker] = {
            "ticker": ticker,
            "entry_price": entry_price,
            "shares": qty,
            "strategy": "RECOVERED",
            "exit_price": None,
        }

        # Add to candidates if not already there
        if ticker not in candidate_tickers:
            candidates = list(candidates) + [{
                "ticker": ticker,
                "gap_pct": 0,
                "pm_volume": 0,
                "premarket_high": entry_price,
                "prev_close": entry_price,
                "float_shares": None,
            }]
            # Also add to engine picks
            engine.picks.append({
                "ticker": ticker,
                "gap_pct": 0,
                "market_open": entry_price,
                "premarket_high": entry_price,
                "prev_close": entry_price,
                "pm_volume": 0,
                "market_hour_candles": None,
            })
            candidate_tickers.add(ticker)

        log.info(f"RECOVERY: {ticker} restored — will manage to EOD (stop={HARD_STOP_PCT}%, target={HARD_TARGET_PCT}%)")

    return candidates


class _ETFormatter(logging.Formatter):
    """Logging formatter that stamps times in US/Eastern (ET)."""
    def formatTime(self, record, datefmt=None):
        from datetime import datetime as _dt
        ct = _dt.fromtimestamp(record.created, tz=ET)
        return ct.strftime(datefmt or "%H:%M:%S")


def setup_logging():
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

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(fh)

    return logging.getLogger("live.main")


def wait_until(target_time, log):
    """Wait until a specific ET time today."""
    now = datetime.now(ET)
    target = datetime.combine(now.date(), target_time, tzinfo=ET)
    if target <= now:
        return
    wait_secs = (target - now).total_seconds()
    log.info(f"Waiting {wait_secs/60:.1f} minutes until {target_time}...")
    time.sleep(max(0, wait_secs))


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

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def run(args):
    log = setup_logging()
    log.info("=" * 60)
    log.info("Combined Strategy Live Paper Trading (Trial 432)")
    log.info("=" * 60)

    # Check account
    executor = OrderExecutor()
    acct = executor.get_account()
    log.info(f"Account: cash=${float(acct.cash):,.2f}, "
             f"buying_power=${float(acct.buying_power):,.2f}")

    # Start dashboard immediately so it's accessible before scan
    if not args.no_dash:
        start_dashboard(None, executor, [], port=args.port)
        log.info(f"Dashboard running at http://localhost:{args.port}")

    from datetime import time as dt_time

    # Phase 1: Pre-market scan loop (9:00 → 9:25 → 9:27 → lock)
    scanner = PreMarketScanner()

    def do_scan(label):
        log.info(f"Phase 1: Pre-market scan ({label})")
        try:
            c = scanner.get_final_watchlist()
        except Exception as e:
            log.error(f"Scanner error during {label}: {e}", exc_info=True)
            return []
        if c:
            for x in c:
                float_str = f"{x['float_shares']/1e6:.1f}M" if x.get('float_shares') else "N/A"
                log.info(f"  {x['ticker']}: gap={x['gap_pct']:.1f}%, "
                         f"PM vol={x['pm_volume']:,}, float={float_str}")
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
        (dt_time(7, 0),  "7:00"),
        (dt_time(7, 30), "7:30"),
        (dt_time(9, 0),  "9:00"),
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
                return

    if args.scan_only:
        log.info("--scan-only mode. Exiting.")
        return

    if not candidates:
        log.info("No candidates after all scans.")
        if not args.no_dash:
            log.info("Dashboard still running at http://localhost:%d", args.port)
            try:
                while True:
                    time.sleep(300)
            except KeyboardInterrupt:
                log.info("Shutting down.")
        return

    log.info(f"Watchlist locked: {len(candidates)} candidates")

    # Phase 2: Initialize combined engine
    engine = CombinedEngine(executor)
    engine.initialize_watchlist(candidates)

    # Recover any open positions from a previous session/crash
    candidates = recover_open_positions(engine, executor, candidates, log)

    if args.dry_run:
        log.info("--dry-run mode: will process bars but not place orders")

    # Phase 3: Wire engine into dashboard bridge
    if not args.no_dash:
        from dashboard.backend.app import bridge
        bridge.engine = engine
        bridge.scanner_candidates = candidates

    # Phase 4: Wait for market open (stream starts at 9:30)
    now = datetime.now(ET)
    if now < datetime.combine(now.date(), dt_time(9, 30), tzinfo=ET):
        wait_until(dt_time(9, 30), log)

    # Phase 5: Start bar stream
    log.info("Starting bar stream")
    symbols = [c["ticker"] for c in candidates]

    def on_bar_with_ws(symbol, bar):
        """Process bar in engine AND broadcast to dashboard."""
        engine.on_bar(symbol, bar)
        # Broadcast to WebSocket clients
        try:
            from dashboard.backend.services.ws_manager import ws_manager
            ws_manager.broadcast_sync({
                "type": "bar",
                "symbol": symbol,
                "data": {k: (str(v) if hasattr(v, 'isoformat') else v) for k, v in bar.items()},
            })
        except Exception:
            pass

    streamer = BarStreamer(on_2min_bar=on_bar_with_ws if not args.no_dash else engine.on_bar)
    streamer.subscribe(symbols)
    stream_thread = streamer.start_async()

    log.info("Streaming... waiting for signals")

    # Phase 6: Main loop - monitor until EOD
    stop_keys = [v for k, v in engine.params.items() if k.endswith("_stop_pct") and v > 0]
    target_keys = [v for k, v in engine.params.items() if k.endswith("_target_pct") or k.endswith("_target1_pct") or k.endswith("_target2_pct")]
    RECOVERY_STOP = -(max(stop_keys) if stop_keys else 12.0)
    RECOVERY_TARGET = max(target_keys) if target_keys else 23.0
    try:
        while True:
            now = datetime.now(ET)

            # EOD close at 3:45 PM
            if now.hour == 15 and now.minute >= 45:
                log.info("EOD: Closing all positions")
                engine.eod_close()
                break

            # After market close
            if now.hour >= 16:
                break

            # Monitor recovered positions with hard stop/target
            active = engine.active_position
            if active and engine.position_entry.get(active, {}).get("strategy") == "RECOVERED":
                try:
                    positions = executor.get_positions()
                    for pos in positions:
                        if pos.symbol == active:
                            entry = engine.position_entry[active]["entry_price"]
                            current = float(pos.current_price)
                            chg = (current / entry - 1) * 100
                            if chg <= RECOVERY_STOP:
                                log.warning(f"RECOVERY STOP HIT: {active} {chg:+.1f}% — selling")
                                executor.sell(active, reason="RECOVERY_STOP")
                                engine.active_position = None
                            elif chg >= RECOVERY_TARGET:
                                log.info(f"RECOVERY TARGET HIT: {active} {chg:+.1f}% — selling")
                                executor.sell(active, reason="RECOVERY_TARGET")
                                engine.active_position = None
                except Exception:
                    pass

            time.sleep(30)

    except KeyboardInterrupt:
        log.info("Interrupted by user")

    # Stop streamer and print daily summary
    streamer.flush_pending()
    streamer.stop()

    summary = engine.get_summary()
    log.info("=" * 60)
    log.info("DAILY SUMMARY")
    log.info(f"  Trades: {summary['trades']}")
    log.info(f"  Wins:   {summary['wins']}")
    log.info(f"  Losses: {summary['losses']}")
    log.info(f"  PnL:    ${summary['daily_pnl']:+,.2f}")
    for t in summary["trade_details"]:
        log.info(f"    {t['ticker']} ({t.get('strategy','?')}): ${t['pnl']:+,.2f} "
                 f"({t['reason']}) ${t['entry_price']:.2f} -> ${t['exit_price']:.2f}")
    log.info("=" * 60)

    # Keep process alive so dashboard stays accessible after hours
    if not args.no_dash:
        log.info("Market closed. Dashboard still running at http://localhost:%d", args.port)
        try:
            while True:
                time.sleep(300)
        except KeyboardInterrupt:
            log.info("Shutting down.")


def main():
    parser = argparse.ArgumentParser(description="Combined Strategy Live Paper Trading")
    parser.add_argument("--scan-only", action="store_true", help="Only run scanner")
    parser.add_argument("--dry-run", action="store_true", help="Process bars but don't trade")
    parser.add_argument("--no-dash", action="store_true", help="Disable dashboard server")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port (default: 8000)")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as e:
        log = logging.getLogger("live.main")
        log.error(f"Fatal error: {e}", exc_info=True)
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
