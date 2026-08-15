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


def get_day_starting_cash(db, trade_date, fallback=10000.0):
    """Exact starting cash for the trade date = the PRIOR trading day's market-close
    cash balance (the cash that was available at this day's open). The live engine
    seeds simulate_day_combined with acct.cash (not margin buying power), so the
    replay must use the same number for dollar figures to align.

    Falls back to fallback if no prior snapshot exists (fresh account / missing data).
    """
    from datetime import date as _date
    target = _date.fromisoformat(trade_date)
    d = target
    for _ in range(14):  # back up to ~2 weeks for weekends/holidays
        d = d - timedelta(days=1)
        try:
            snaps = db.get_account_snapshots_by_date(d.isoformat())
        except Exception:
            snaps = []
        # prefer the last market_close cash (fall back to any snapshot's cash)
        closes = [s for s in snaps if s.get("snapshot_type") == "market_close"]
        pool = closes if closes else snaps
        for s in pool:
            c = s.get("cash")
            if c is not None:
                try:
                    c = float(c)
                except (TypeError, ValueError):
                    continue
                if c >= 0:
                    log.info("Starting cash for %s: $%.2f (from %s close)", trade_date, c, d)
                    return c
    log.warning("No prior cash snapshot for %s — using fallback $%.0f", trade_date, fallback)
    return fallback


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


def scrape_and_persist_halts():
    """Scrape today's NASDAQ halt feed into data/halts.csv (accumulates history).

    2026-08-15: the halt RSS is the only live source (the .txt is dead); it's a
    rolling ~60-event window. Scraping it once daily after close preserves a
    permanent record for backtest. Uses the parser with the millisecond-time fix.
    """
    from live.halt_monitor import fetch_halt_log, parse_halt_log, _persist_halt_event
    try:
        raw = fetch_halt_log()
        evs = parse_halt_log(raw)
        n = 0
        for ev in evs:
            try:
                _persist_halt_event(ev)
                n += 1
            except Exception:
                pass
        log.info("halt scrape: parsed %d events, persisted %d", len(evs), n)
        return len(evs)
    except Exception as e:
        log.warning("halt scrape failed: %s", e)
        return 0


def run_reconcile(trade_date, dry_run=False):
    db = TradingDatabase()
    day = date.fromisoformat(trade_date)

    # Persist today's halts (accumulate history for backtest).
    scrape_and_persist_halts()

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
    start_cash = get_day_starting_cash(db, trade_date)
    if not dry_run:
        # Use the EXACT starting cash the live account had at this day's open
        # (= prior trading day's market-close cash; live seeds sim with acct.cash,
        # NOT margin buying power). Dollar figures then align with live.
        states, _, _, _ = tgc.simulate_day_combined(picks, start_cash, False, params=snap)
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                cost = st.get("position_cost", 0) or 0
                pnl = st.get("pnl", 0) or 0
                et = str(st.get("entry_time", ""))[11:16]
                xt = str(st.get("exit_time", ""))[11:16]
                ep = st.get("entry_price") or 0
                xp = st.get("exit_price") or 0

                # --- entry/exit candle detail from the sim's candle DataFrame ---
                mh = st.get("mh")
                entry_candles = None   # [ {bar_time, O,H,L,C,green,new_high} x 2 ]
                exit_candle = None
                if mh is not None and len(mh) > 0:
                    try:
                        mhf = mh.copy()
                        if mhf.index.tz is None:
                            mhf.index = mhf.index.tz_localize("UTC")
                        mhf.index = mhf.index.tz_convert("America/New_York")
                        times = [t.strftime("%H:%M") for t in mhf.index]
                        c1o = float(mhf.iloc[0]["Open"]); c1c = float(mhf.iloc[0]["Close"])
                        c1h = float(mhf.iloc[0]["High"])
                        entry_candles = [
                            {
                                "bar_time": times[0],
                                "open": round(c1o, 4), "high": round(c1h, 4),
                                "close": round(c1c, 4), "green": c1c > c1o,
                                "body_pct": round((c1c / c1o - 1) * 100, 2) if c1o > 0 else 0,
                            },
                        ]
                        if len(mhf) > 1:
                            c2o = float(mhf.iloc[1]["Open"]); c2c = float(mhf.iloc[1]["Close"])
                            c2h = float(mhf.iloc[1]["High"])
                            entry_candles.append({
                                "bar_time": times[1],
                                "open": round(c2o, 4), "high": round(c2h, 4),
                                "close": round(c2c, 4), "green": c2c > c2o,
                                "body_pct": round((c2c / c2o - 1) * 100, 2) if c2o > 0 else 0,
                                "new_high": c2h > c1h,
                            })
                        # exit bar = last bar in the trade's window
                        if len(mhf) > 0:
                            last = mhf.iloc[-1]
                            lo = float(last["Open"]); lc = float(last["Close"])
                            exit_candle = {
                                "bar_time": times[-1], "open": round(lo, 4),
                                "high": round(float(last["High"]), 4),
                                "close": round(lc, 4), "green": lc > lo,
                            }
                    except Exception:
                        entry_candles = None
                        exit_candle = None

                bt_trades.append({
                    "ticker": st.get("ticker"),
                    "strategy": st.get("strategy"),
                    "gap_pct": round(st.get("gap_pct", 0), 1),
                    "pnl": pnl,
                    "pnl_pct": round(pnl / cost * 100, 2) if cost > 0 else 0,
                    "reason": st.get("exit_reason"),
                    "entry_time": et,
                    "exit_time": xt,
                    "entry_price": round(float(ep), 4) if ep else None,
                    "exit_price": round(float(xp), 4) if xp else None,
                    "shares": st.get("shares", 0),
                    "hold_min": round(((st.get("exit_time") or st.get("entry_time")) - (st.get("entry_time") or 0)).total_seconds() / 60, 1)
                    if hasattr(st.get("exit_time"), "total_seconds") and hasattr(st.get("entry_time"), "total_seconds")
                    else None,
                    "entry_candles": entry_candles,
                    "exit_candle": exit_candle,
                })
                bt_pnl += pnl

    # 4. compare
    live_set = {t["ticker"] for t in live_trades if t.get("ticker")}
    live_sig_set = {s.get("ticker") for s in live_signals if s.get("ticker")}
    bt_set = {t["ticker"] for t in bt_trades}
    match = sorted(live_set & bt_set)
    live_only = sorted(live_set - bt_set)
    bt_only = sorted(bt_set - live_set)

    live_pnl = sum(t.get("pnl", 0) or 0 for t in live_trades)
    live_traded_tickers = sorted(live_set)

    # ---- per-ticker divergence diagnosis (halt gaps in the live Tradier feed) ----
    def _detect_halt_gaps(ticker):
        """Compare live Tradier bar coverage vs the 09:30-16:00 minute grid.
        Returns a dict describing missing-bar windows (halt indicator)."""
        bars = db.get_tradier_bars_for_ticker(trade_date, ticker)
        times = sorted(b.get("bar_time", "") for b in bars if b.get("bar_time"))
        # parse HH:MM -> minutes since 09:30
        try:
            mins = [int(t[:2]) * 60 + int(t[3:5]) for t in times]
        except Exception:
            return {"bars": len(bars), "note": "unparseable bar times"}
        session_min = [t for t in mins if 9 * 60 + 30 <= t <= 16 * 60]
        if not session_min:
            return {"bars": len(bars), "note": "no session bars in live feed"}
        gaps = []
        prev = None
        for m in session_min:
            if prev is not None and m - prev > 1:
                gaps.append((prev, m))  # (last bar minute, first missing)
            prev = m
        gap_str = "; ".join(
            f"{g0//60:02d}:{g0%60:02d}-{g1//60:02d}:{g1%60:02d} missing" for g0, g1 in gaps
        )
        return {"bars": len(bars), "first": times[0] if times else None,
                "last": times[-1] if times else None, "gaps": gap_str}

    findings = {}
    for tk in live_only + bt_only:
        findings[tk] = _detect_halt_gaps(tk)

    # human-readable comment / findings
    comments = []
    for tk in bt_only:
        f = findings.get(tk, {})
        if f.get("gaps"):
            comments.append(
                f"BT-ONLY {tk}: SIP replay traded it but LIVE did not — the live Tradier "
                f"feed had a bar gap [{f['gaps']}] (halt/resume). The live G path can't "
                f"confirm candle-2 through a halt, so it skipped the trade; the post-hoc "
                f"SIP replay sees the complete tape."
            )
        elif f.get("bars", 0) == 0:
            comments.append(
                f"BT-ONLY {tk}: SIP replay traded it but LIVE had NO bars for this ticker "
                f"(stream issue or halt) — live G never saw a signal."
            )
        else:
            comments.append(
                f"BT-ONLY {tk}: SIP replay traded it but LIVE did not (no halt gap detected; "
                f"likely cash/BP, rejected order, already-in-position, or a candle-color "
                f"difference between the live and SIP feeds)."
            )
    for tk in live_only:
        f = findings.get(tk, {})
        comments.append(
            f"LIVE-ONLY {tk}: traded live but the SIP replay did not — check fill timing, "
            f"candle-1 color, or a signal the replay couldn't reproduce. Live Tradier bars: "
            f"{f.get('bars', '?')}."
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

    # --- live candle detail (Tradier feed_comparison bars around entry/exit) ---
    def _live_candles(ticker, entry_min_str, exit_min_str=None):
        """First 3 live Tradier bars + the exit bar for a ticker (for the live trade log)."""
        bars = db.get_tradier_bars_for_ticker(trade_date, ticker)
        if not bars:
            return None, None
        try:
            entry_candles = []
            for b in bars[:3]:
                c = {
                    "bar_time": b.get("bar_time"),
                    "open": round(float(b.get("open") or 0), 4),
                    "high": round(float(b.get("high") or 0), 4),
                    "close": round(float(b.get("close") or 0), 4),
                    "green": float(b.get("close") or 0) > float(b.get("open") or 0),
                }
                if entry_candles:
                    c["new_high"] = float(b.get("high") or 0) > entry_candles[-1]["high"]
                entry_candles.append(c)
            exit_candle = None
            if exit_min_str:
                for b in bars:
                    if b.get("bar_time") == exit_min_str:
                        exit_candle = {
                            "bar_time": b.get("bar_time"),
                            "open": round(float(b.get("open") or 0), 4),
                            "high": round(float(b.get("high") or 0), 4),
                            "close": round(float(b.get("close") or 0), 4),
                            "green": float(b.get("close") or 0) > float(b.get("open") or 0),
                        }
                        break
            return entry_candles, exit_candle
        except Exception:
            return None, None

    live_trades_detail = []
    for t in live_trades:
        et = str(t.get("entry_time", ""))[11:16]
        xt = str(t.get("exit_time", ""))[11:16]
        ec, xc = _live_candles(t.get("ticker"), et, xt)
        live_trades_detail.append({
            "ticker": t.get("ticker"),
            "strategy": t.get("strategy"),
            "gap_pct": round(float(t.get("gap_pct") or 0), 1),
            "pnl": t.get("pnl"),
            "pnl_pct": round(t.get("pnl_pct", 0) or 0, 2),
            "reason": t.get("exit_reason"),
            "entry_time": et,
            "exit_time": xt,
            "entry_price": round(float(t.get("entry_price", 0) or 0), 4),
            "exit_price": round(float(t.get("exit_price", 0) or 0), 4),
            "shares": t.get("shares", 0),
            "entry_candles": ec,
            "exit_candle": xc,
        })

    details = {
        "live_trades": live_trades_detail,
        "bt_trades": bt_trades,
        "live_signals": [{"ticker": s.get("ticker"), "price": s.get("signal_price"),
                          "gap": s.get("gap_pct")} for s in live_signals],
        "comments": comments,
        "findings": findings,
    }

    record = {
        "date": trade_date,
        "watchlist_count": len(watchlist),
        "sip_fetched": len(bars_map),
        "sip_missing": ",".join(missing) if missing else None,
        "start_cash": round(start_cash, 2),
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
    if bt_trades:
        print("\n  --- BACKTEST TRADES (SIP replay through deploy G config) ---")
        print(f"  {'ticker':<7} {'entry':>6} {'exit':>6} {'entry$':>8} {'exit$':>8} {'reason':<10} {'P&L':>10} {'P&L%':>7}")
        for t in bt_trades:
            print(f"  {t['ticker']:<7} {t.get('entry_time') or '-':>6} {t.get('exit_time') or '-':>6} "
                  f"{t.get('entry_price') or 0:>8.3f} {t.get('exit_price') or 0:>8.3f} "
                  f"{t.get('reason') or '-':<10} {t.get('pnl', 0):>10,.0f} {t.get('pnl_pct', 0):>+6.1f}%")
            for c in (t.get('entry_candles') or []):
                hi = f" NEWHI" if c.get('new_high') else ""
                print(f"    entry-candle {c.get('bar_time')}: {c.get('open')}->{c.get('close')} "
                      f"({'GREEN' if c.get('green') else 'red'}{hi})")
            if t.get('exit_candle'):
                x = t['exit_candle']
                print(f"    exit-candle  {x.get('bar_time')}: {x.get('open')}->{x.get('close')} "
                      f"({'GREEN' if x.get('green') else 'red'})")
    if live_trades:
        print("\n  --- LIVE TRADES ---")
        for t in live_trades:
            print(f"  {t.get('ticker','?'):<7} entry={str(t.get('entry_time',''))[11:16]} "
                  f"${t.get('pnl',0):>+10,.0f}  ({t.get('exit_reason') or ''})")
    if findings:
        print("\n  --- DIVERGENCE DIAGNOSIS (live Tradier bar coverage) ---")
        for tk, f in findings.items():
            print(f"    {tk}: {f.get('bars','?')} live bars  gaps: {f.get('gaps') or 'none'}")
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
