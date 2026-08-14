"""
Post-Close Daily Reconcile: Live vs SIP-feed backtest replay.

Runs automatically via cron after extended hours (~8pm ET). For the most recent
completed trading day it:

  1. Reads the live day's watchlist, signals, and trades from logs/trading.db.
  2. Downloads 1-min SIP bars (feed='sip' — the SAME feed backtests use) for
     every watchlist ticker. NOTE: Alpaca's free SIP forbids querying the
     CURRENT in-progress day, but any COMPLETED trading day is fine — which is
     why this runs after close.
  3. Builds picks exactly like the live scanner (gap_pct, pm_volume,
     premarket_high, prev_close, float) and runs the deployed G config through
     the REAL simulate_day_combined (slippage + volcaps ON).
  4. Compares backtest trades vs the live trades on (date, ticker):
       match      = both traded
       live_only  = live traded but SIP replay did not (a divergence to flag)
       bt_only    = SIP replay traded but live did not (replay caught something
                    live missed — often a fill/signal timing difference)
  5. Writes a human-readable summary + comments into the daily_reconcile table,
     surfaced on the dashboard's "Daily Reconcile" view.

Data integrity: uses Alpaca SIP ONLY (never IEX — wrong volume; never Yahoo —
wrong candle structure). Mirrors the download_losers_alpaca.py fetch pattern.

Usage:
  python scripts/reconcile/post_close_reconcile.py            # default: latest completed day
  python scripts/reconcile/post_close_reconcile.py --date 2026-08-14
  python scripts/reconcile/post_close_reconcile.py --dry-run
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET
from live.persistence_db import TradingDatabase

ET = ZoneInfo("America/New_York")
DEPLOY = os.path.join("config", "trial_gl_1min_g2_1x_G_only_deploy.json")
RATE_LIMIT_DELAY = 0.6
MAX_WATCHLIST = 30  # the live scanner's TOP_N universe cap

import logging
log = logging.getLogger("post_close_reconcile")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def latest_completed_trading_day() -> str:
    """Most recent weekday before today (best-effort; holidays just mean fewer picks)."""
    d = datetime.now(ET).date() - timedelta(days=1)
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d.isoformat()


def fetch_minute_bars(client, ticker, day):
    """1-min SIP bars for ticker on day (market hours only). Returns DataFrame or None."""
    s = datetime.combine(day, datetime.min.time().replace(hour=9, minute=30)).replace(tzinfo=ET)
    e = datetime.combine(day, datetime.min.time().replace(hour=16, minute=0)).replace(tzinfo=ET)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=s,
        end=e,
        adjustment="raw",
        feed="sip",
    )
    resp = client.get_stock_bars(req)
    if resp.df is None or resp.df.empty:
        return None
    df = resp.df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == ticker].drop(columns=["symbol"])
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def run_reconcile(trade_date, dry_run=False):
    db = TradingDatabase()
    day = date.fromisoformat(trade_date)

    # 1. live data
    watchlist = db.get_watchlist_by_date(trade_date)
    live_trades = db.get_trades_by_date(trade_date)
    live_signals = db.get_signals_by_date(trade_date) if hasattr(db, "get_signals_by_date") else []
    # fallback: query signals directly
    if not live_signals:
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM signals WHERE date=? AND action='TAKEN'", (trade_date,)).fetchall()
            live_signals = [dict(r) for r in rows]
        except Exception:
            pass
        finally:
            conn.close()

    if not watchlist:
        log.warning("No watchlist for %s — nothing to replay (holiday/weekend?)", trade_date)
        db.save_daily_reconcile({
            "date": trade_date, "watchlist_count": 0, "sip_fetched": 0,
            "sip_missing": None, "bt_trades": 0, "bt_pnl": 0.0,
            "live_trades": len(live_trades), "live_pnl": 0.0,
            "live_signals": len(live_signals), "match_count": 0,
            "live_only_count": 0, "bt_only_count": 0,
            "match_tickers": None, "live_only_tickers": None, "bt_only_tickers": None,
            "summary": f"No watchlist for {trade_date} (no market data to replay).",
            "details": json.dumps({"live_trades": len(live_trades), "live_signals": len(live_signals)}),
            "status": "skipped",
        })
        return

    # 2. download SIP bars
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    bars_map = {}
    missing = []
    for w in watchlist[:MAX_WATCHLIST]:
        tk = w["ticker"]
        try:
            df = fetch_minute_bars(client, tk, day)
            if df is not None and len(df) >= 5:
                bars_map[tk] = df
            else:
                missing.append(tk)
        except Exception as e:
            # recent-data cap should not hit a completed day; log and continue
            log.warning("SIP %s: %s", tk, str(e)[:80])
            missing.append(tk)
        time.sleep(RATE_LIMIT_DELAY)

    log.info("SIP fetched %d/%d watchlist tickers for %s", len(bars_map), len(watchlist), trade_date)

    # 3. run real-engine backtest (deploy config, G-only)
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params, _build_param_snapshot

    with open(DEPLOY) as f:
        deploy = json.load(f)
    params = dict(deploy.get("params", {}))
    params["enable_g"] = True
    for s in "hafdvmrpwobkcsexijnl":
        params[f"enable_{s}"] = False
    tgc.G_FIRST_BAR_ONLY = bool(deploy.get("g_first_bar_only"))
    tgc.MARGIN_MULTIPLIER = float(deploy.get("margin_multiplier", 1.0))
    tgc.MAX_POSITION_PCT_OF_CASH = float(deploy.get("max_position_pct_of_cash", 100.0))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    set_strategy_params(params)
    snap = _build_param_snapshot()

    picks = []
    for w in watchlist:
        tk = w["ticker"]
        df = bars_map.get(tk)
        if df is None or len(df) < 5:
            continue
        prev_close = float(w.get("prev_close") or 0)
        if not prev_close:
            prev_close = float(df.iloc[0]["Open"]) / (1 + float(w["gap_pct"]) / 100)
        picks.append({
            "ticker": tk,
            "gap_pct": float(w["gap_pct"]),
            "market_open": float(df.iloc[0]["Open"]),
            "premarket_high": float(w.get("premarket_high") or df.iloc[0]["Open"]),
            "prev_close": prev_close,
            "pm_volume": float(w.get("pm_volume") or 0),
            "float_shares": w.get("float_shares"),
            "market_hour_candles": df,
        })

    bt_trades = []
    bt_pnl = 0.0
    if not dry_run:
        states, _, _, _ = tgc.simulate_day_combined(picks, 10000.0, False, params=snap)
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                bt_trades.append({
                    "ticker": st.get("ticker"),
                    "pnl": st.get("pnl", 0),
                    "reason": st.get("exit_reason"),
                    "entry_time": str(st.get("entry_time", ""))[11:16],
                })
                bt_pnl += st.get("pnl", 0)

    # 4. compare
    live_set = {t["ticker"] for t in live_trades if t.get("ticker")}
    live_sig_set = {s.get("ticker") for s in live_signals if s.get("ticker")}
    bt_set = {t["ticker"] for t in bt_trades}
    match = sorted(live_set & bt_set)
    live_only = sorted(live_set - bt_set)
    bt_only = sorted(bt_set - live_set)

    live_pnl = sum(t.get("pnl", 0) or 0 for t in live_trades)
    live_traded_tickers = sorted(live_set)

    # human-readable comment / findings
    comments = []
    if live_only:
        comments.append(
            f"LIVE-ONLY ({len(live_only)}): {', '.join(live_only)} traded live but the SIP "
            "replay did not — check fill timing, bar-candle-1 color, or a signal the "
            "replay couldn't reproduce (e.g. slow fill, halt, or stream-driven entry)."
        )
    if bt_only:
        comments.append(
            f"BT-ONLY ({len(bt_only)}): {', '.join(bt_only)} fired in the SIP replay but "
            "not live — the live engine may have skipped them (cash/BP, rejected order, "
            "already-in-position, halt, or a stream that missed the bar)."
        )
    if match:
        comments.append(
            f"MATCH ({len(match)}): {', '.join(match)} — live and SIP replay agree."
        )
    if not live_only and not bt_only and match:
        comments.append("PERFECT MATCH: live and SIP replay traded the same set.")
    if not match and not live_only and not bt_only:
        comments.append("NO TRADES in either live or SIP replay this day.")

    status = "match" if (not live_only and not bt_only and match) else (
        "divergence" if (live_only or bt_only) else "no_trades")
    if live_only and not match:
        status = "live_only_missed_by_replay"

    summary = (
        f"{trade_date}: live traded {len(live_set)} ({len(live_trades)} fills, "
        f"${live_pnl:,.0f}); SIP replay traded {len(bt_set)} (${bt_pnl:,.0f}); "
        f"match {len(match)}, live-only {len(live_only)}, replay-only {len(bt_only)}."
    )

    details = {
        "live_trades": [
            {"ticker": t.get("ticker"), "pnl": t.get("pnl"), "reason": t.get("exit_reason"),
             "entry_time": str(t.get("entry_time", ""))[11:16]}
            for t in live_trades
        ],
        "bt_trades": bt_trades,
        "live_signals": [{"ticker": s.get("ticker"), "price": s.get("signal_price"),
                          "gap": s.get("gap_pct")} for s in live_signals],
        "comments": comments,
    }

    record = {
        "date": trade_date,
        "watchlist_count": len(watchlist),
        "sip_fetched": len(bars_map),
        "sip_missing": ",".join(missing) if missing else None,
        "bt_trades": len(bt_trades),
        "bt_pnl": round(bt_pnl, 2),
        "live_trades": len(live_trades),
        "live_pnl": round(live_pnl, 2),
        "live_signals": len(live_signals),
        "match_count": len(match),
        "live_only_count": len(live_only),
        "bt_only_count": len(bt_only),
        "match_tickers": ",".join(match) if match else None,
        "live_only_tickers": ",".join(live_only) if live_only else None,
        "bt_only_tickers": ",".join(bt_only) if bt_only else None,
        "summary": summary,
        "details": json.dumps(details),
        "status": status,
    }

    if not dry_run:
        rid = db.save_daily_reconcile(record)
        log.info("Saved reconcile id=%s status=%s", rid, status)
    else:
        log.info("DRY-RUN (not saved): %s", summary)

    # console report
    print("\n" + "=" * 70)
    print(f"DAILY RECONCILE {trade_date}  (status: {status})")
    print("=" * 70)
    print(f"  watchlist: {len(watchlist)}  | SIP fetched: {len(bars_map)}  "
          f"| missing: {','.join(missing) or '-'}")
    print(f"  LIVE trades: {len(live_trades)} (${live_pnl:,.0f})  tickers: {live_traded_tickers or '-'}")
    print(f"  BT   trades: {len(bt_trades)} (${bt_pnl:,.0f})  tickers: {bt_set or '-'}")
    print(f"  MATCH: {match or '-'}")
    print(f"  LIVE-ONLY: {live_only or '-'}")
    print(f"  BT-ONLY: {bt_only or '-'}")
    print("\n  Comments:")
    for c in comments:
        print(f"    - {c}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest completed trading day)")
    ap.add_argument("--dry-run", action="store_true", help="Run the backtest + compare but don't save to DB")
    args = ap.parse_args()
    trade_date = args.date or latest_completed_trading_day()
    run_reconcile(trade_date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
