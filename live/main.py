"""
Live Paper Trading: Combined Strategy Engine (12 strategies from Optuna trial 432)
===================================================================================
Entry point for live paper trading on Alpaca using all enabled strategies.
Includes embedded dashboard server on port 8000.

Schedule:
  8:00 AM ET  - Start pre-market scan
  9:25 AM ET  - Finalize watchlist
  9:29 AM ET  - Subscribe to bar stream
  9:30 AM ET  - Strategy engine starts processing
  3:45 PM ET  - EOD close all positions
  4:00 PM ET  - Daily summary + shutdown

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


def setup_logging():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    log_dir = "logs"
    import os
    os.makedirs(log_dir, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    fh = logging.FileHandler(f"{log_dir}/{today}_live.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

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
    thread = threading.Thread(target=server.run, daemon=True)
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
    now = datetime.now(ET)
    if now < datetime.combine(now.date(), dt_time(9, 0), tzinfo=ET):
        wait_until(dt_time(9, 0), log)

    scanner = PreMarketScanner()

    def do_scan(label):
        log.info(f"Phase 1: Pre-market scan ({label})")
        c = scanner.get_final_watchlist()
        if c:
            for x in c:
                float_str = f"{x['float_shares']/1e6:.1f}M" if x.get('float_shares') else "N/A"
                log.info(f"  {x['ticker']}: gap={x['gap_pct']:.1f}%, "
                         f"PM vol={x['pm_volume']:,}, float={float_str}")
        else:
            log.info("  No candidates found")
        return c

    candidates = do_scan("9:00")

    if args.scan_only:
        log.info("--scan-only mode. Exiting.")
        return

    # Rescan at 9:25
    now = datetime.now(ET)
    if now < datetime.combine(now.date(), dt_time(9, 25), tzinfo=ET):
        wait_until(dt_time(9, 25), log)
        candidates = do_scan("9:25")

    # Final scan at 9:27
    now = datetime.now(ET)
    if now < datetime.combine(now.date(), dt_time(9, 27), tzinfo=ET):
        wait_until(dt_time(9, 27), log)
        candidates = do_scan("9:27 FINAL")

    if not candidates:
        log.info("No candidates after final scan. Exiting.")
        return

    log.info(f"Watchlist locked: {len(candidates)} candidates")

    # Phase 2: Initialize combined engine
    engine = CombinedEngine(executor)
    engine.initialize_watchlist(candidates)

    if args.dry_run:
        log.info("--dry-run mode: will process bars but not place orders")

    # Phase 3: Wire engine into dashboard bridge
    if not args.no_dash:
        from dashboard.backend.app import bridge
        bridge.engine = engine
        bridge.scanner_candidates = candidates

    # Phase 4: Wait for market open
    now = datetime.now(ET)
    if now < datetime.combine(now.date(), dt_time(9, 29), tzinfo=ET):
        wait_until(dt_time(9, 29), log)

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

            time.sleep(10)

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        streamer.flush_pending()
        streamer.stop()

        # Daily summary
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


def main():
    parser = argparse.ArgumentParser(description="Combined Strategy Live Paper Trading")
    parser.add_argument("--scan-only", action="store_true", help="Only run scanner")
    parser.add_argument("--dry-run", action="store_true", help="Process bars but don't trade")
    parser.add_argument("--no-dash", action="store_true", help="Disable dashboard server")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port (default: 8000)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
