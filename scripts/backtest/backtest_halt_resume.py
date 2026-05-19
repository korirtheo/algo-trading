"""
Standalone backtest of the halt-resume strategy.

Joins data/halts.csv with per-ticker intraday bars in stored_data_*/intraday/
and replays strategies.halt_resume.check_signal / check_exit against each
resume event in the given date range.

Usage:
    python -m scripts.backtest.backtest_halt_resume \
        --halts data/halts.csv \
        --data stored_data_combined stored_data_2023 \
        --start 2024-01-01 --end 2026-02-28 \
        --cash 25000
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, date, time as dt_time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from strategies import halt_resume as hr
from config.settings import FLOAT_DATA, SLIPPAGE_PCT, EOD_EXIT_MINUTES

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_CLOSE = dt_time(16, 0)


@dataclass
class HaltRow:
    halt_date: date
    halt_time: dt_time
    ticker: str
    reason: str
    resume_time: Optional[dt_time]
    resume_trade_time: Optional[dt_time]
    halt_price: Optional[float]
    resume_price: Optional[float]

    @property
    def resume_dt_et(self) -> Optional[datetime]:
        t = self.resume_trade_time or self.resume_time
        if t is None:
            return None
        return datetime.combine(self.halt_date, t, tzinfo=ET)

    # Strategy module reads .resume_dt (mirrors live.halt_monitor.HaltEvent).
    @property
    def resume_dt(self) -> Optional[datetime]:
        return self.resume_dt_et


def load_halts(path: str, start: Optional[date], end: Optional[date]) -> list[HaltRow]:
    rows: list[HaltRow] = []
    if not os.path.exists(path):
        print(f"ERROR: halts file not found: {path}")
        return rows
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            try:
                hd = datetime.fromisoformat(d["halt_date"]).date()
            except Exception:
                continue
            if start and hd < start:
                continue
            if end and hd > end:
                continue
            try:
                ht = datetime.fromisoformat(f"2000-01-01T{d['halt_time']}").time()
            except Exception:
                continue
            def _t(key):
                v = d.get(key) or ""
                if not v:
                    return None
                try:
                    return datetime.fromisoformat(f"2000-01-01T{v}").time()
                except Exception:
                    return None
            def _p(key):
                v = d.get(key) or ""
                if not v:
                    return None
                try:
                    return float(v)
                except Exception:
                    return None
            rows.append(HaltRow(
                halt_date=hd, halt_time=ht, ticker=d["ticker"].strip().upper(),
                reason=d.get("reason", "").strip(),
                resume_time=_t("resume_time"),
                resume_trade_time=_t("resume_trade_time"),
                halt_price=_p("halt_price"),
                resume_price=_p("resume_price"),
            ))
    return rows


def _find_intraday_csv(data_dirs: list[str], ticker: str) -> Optional[str]:
    for d in data_dirs:
        cand = os.path.join(d, "intraday", f"{ticker}.csv")
        if os.path.exists(cand):
            return cand
    return None


def _load_bars(path: str, halt_date: date) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, parse_dates=["Datetime"])
    except Exception as e:
        return None
    if df.empty:
        return None
    df = df.dropna(subset=["Datetime"])
    if df["Datetime"].dt.tz is None:
        df["Datetime"] = df["Datetime"].dt.tz_localize("UTC")
    df["Datetime_ET"] = df["Datetime"].dt.tz_convert(ET)
    mask = df["Datetime_ET"].dt.date == halt_date
    day_df = df.loc[mask].sort_values("Datetime").reset_index(drop=True)
    return day_df if not day_df.empty else None


def _aggregate_to_2min(df: pd.DataFrame) -> pd.DataFrame:
    """If bars are 1-minute, aggregate to 2-minute slots aligned to 9:30 ET.
    If already 2-min, returns unchanged."""
    if df.empty:
        return df
    # Detect cadence by median gap (seconds)
    if len(df) < 2:
        return df
    gaps = df["Datetime"].diff().dt.total_seconds().dropna()
    if gaps.empty:
        return df
    median = gaps.median()
    if median >= 110:   # already ~2 min
        return df
    df = df.set_index("Datetime")
    o = df["Open"].resample("2min", origin="start_day").first()
    h = df["High"].resample("2min", origin="start_day").max()
    l = df["Low"].resample("2min", origin="start_day").min()
    c = df["Close"].resample("2min", origin="start_day").last()
    v = df["Volume"].resample("2min", origin="start_day").sum()
    out = pd.concat([o, h, l, c, v], axis=1).dropna(how="all")
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = out.reset_index()
    out["Datetime_ET"] = out["Datetime"].dt.tz_convert(ET)
    return out


def backtest_one(halt: HaltRow, data_dirs: list[str], cash: float,
                 slippage_pct: float = SLIPPAGE_PCT,
                 params=None) -> Optional[dict]:
    """Run the halt-resume strategy on a single halt event. Returns trade dict
    or None if ineligible / no fill."""
    params = params or hr.DEFAULT_PARAMS
    if halt.resume_dt_et is None:
        return None

    # Eligibility (matches live filter)
    float_shares = FLOAT_DATA.get(halt.ticker)
    if not hr.is_eligible(halt, float_shares=float_shares, params=params):
        return None

    csv_path = _find_intraday_csv(data_dirs, halt.ticker)
    if csv_path is None:
        return {"ticker": halt.ticker, "halt_date": halt.halt_date,
                "skipped": "no_intraday_data"}

    day_df = _load_bars(csv_path, halt.halt_date)
    if day_df is None or day_df.empty:
        return {"ticker": halt.ticker, "halt_date": halt.halt_date,
                "skipped": "no_bars_on_date"}

    day_df = _aggregate_to_2min(day_df)
    resume_dt_utc = halt.resume_dt_et.astimezone(UTC)
    post = day_df[day_df["Datetime"] >= resume_dt_utc].reset_index(drop=True)
    if post.empty:
        return {"ticker": halt.ticker, "halt_date": halt.halt_date,
                "skipped": "no_post_resume_bars"}

    state = hr.create_state(halt.ticker, halt, float_shares=float_shares, params=params)
    entry_idx = None

    # ----- Entry -----
    for i, row in post.iterrows():
        fired = hr.check_signal(state,
                                 float(row["Open"]), float(row["High"]),
                                 float(row["Low"]), float(row["Close"]),
                                 float(row["Volume"]), params=params)
        if fired:
            entry_fill = row["Close"] * (1 + slippage_pct / 100.0)
            shares = max(1.0, cash // entry_fill) if entry_fill > 0 else 0
            state["entry_price"] = entry_fill
            state["entry_time"] = row["Datetime"]
            state["shares"] = shares
            state["position_cost"] = shares * entry_fill
            state["highest_since_entry"] = float(row["High"])
            entry_idx = i
            break
        if state["done"]:
            break

    if entry_idx is None:
        return {"ticker": halt.ticker, "halt_date": halt.halt_date,
                "skipped": "no_entry_signal", "reason": halt.reason}

    # ----- Exit -----
    entry_time = state["entry_time"]
    orig_shares = state["shares"]
    partial_proceeds = 0.0
    exit_row = None
    exit_price = None
    exit_reason = None

    for i in range(entry_idx + 1, len(post)):
        row = post.iloc[i]
        ts = row["Datetime"]
        ts_et = ts.tz_convert(ET) if hasattr(ts, "tz_convert") else ts.astimezone(ET)
        mins_in = max(0, int((ts - entry_time).total_seconds() // 60))
        close_dt = datetime.combine(halt.halt_date, MARKET_CLOSE, tzinfo=ET)
        mins_to_close = max(0, int((close_dt - ts_et).total_seconds() // 60))

        should_exit, ep, reason = hr.check_exit(
            state, float(row["High"]), float(row["Low"]), float(row["Close"]),
            mins_in, mins_to_close,
            slippage_pct=slippage_pct, eod_exit_minutes=EOD_EXIT_MINUTES,
            params=params,
        )
        if not should_exit:
            continue

        if reason == "PARTIAL":
            partial_share_count = orig_shares * (params["partial_sell_pct"] / 100.0)
            partial_fill = ep * (1 - slippage_pct / 100.0)
            partial_proceeds = partial_share_count * partial_fill
            state["shares"] = max(0.0, state["shares"] - partial_share_count)
            continue

        exit_fill = ep * (1 - slippage_pct / 100.0)
        exit_price = exit_fill
        exit_reason = reason
        exit_row = row
        break

    # Did not exit within the day → force close on last bar
    if exit_price is None:
        last = post.iloc[-1]
        exit_price = float(last["Close"]) * (1 - slippage_pct / 100.0)
        exit_reason = "EOD_FORCED"
        exit_row = last

    proceeds = partial_proceeds + state["shares"] * exit_price
    pnl = proceeds - (state["entry_price"] * orig_shares)
    pct = pnl / max(1e-9, state["entry_price"] * orig_shares) * 100

    return {
        "ticker": halt.ticker,
        "halt_date": halt.halt_date,
        "reason": halt.reason,
        "halt_price": halt.halt_price,
        "resume_price": halt.resume_price,
        "entry_price": state["entry_price"],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "shares": orig_shares,
        "pnl": pnl,
        "pct": pct,
    }


def summarize(trades: list[dict]):
    real = [t for t in trades if "pnl" in t]
    skipped = [t for t in trades if "skipped" in t]
    total_pnl = sum(t["pnl"] for t in real)
    wins = [t for t in real if t["pnl"] > 0]
    losses = [t for t in real if t["pnl"] <= 0]
    wr = len(wins) / max(1, len(real)) * 100

    by_reason = defaultdict(lambda: [0, 0, 0.0])  # count, wins, pnl
    for t in real:
        r = (t.get("reason") or "?").upper()
        by_reason[r][0] += 1
        by_reason[r][1] += 1 if t["pnl"] > 0 else 0
        by_reason[r][2] += t["pnl"]

    print("=" * 78)
    print(f"HALT-RESUME BACKTEST SUMMARY")
    print("=" * 78)
    print(f"  Trades attempted : {len(real)}")
    print(f"  Skipped events   : {len(skipped)}")
    print(f"  Winners          : {len(wins)} ({wr:.1f}%)")
    print(f"  Losers           : {len(losses)}")
    print(f"  Total PnL        : ${total_pnl:+,.2f}")
    if real:
        avg = total_pnl / len(real)
        print(f"  Avg / trade      : ${avg:+,.2f}")
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = -sum(t["pnl"] for t in losses) or 1e-9
        print(f"  Profit factor    : {gross_win/gross_loss:.2f}")
    print(f"  By halt reason:")
    for r, (n, w, p) in sorted(by_reason.items()):
        print(f"    {r:<6} {n:>4} trades  {w/max(1,n)*100:>5.1f}% WR  ${p:+,.2f}")
    print("=" * 78)

    # Skip-reason breakdown
    skip_counts = defaultdict(int)
    for t in skipped:
        skip_counts[t["skipped"]] += 1
    if skip_counts:
        print("Skip reasons:")
        for k, v in sorted(skip_counts.items(), key=lambda x: -x[1]):
            print(f"  {k:<25} {v}")
        print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halts", default="data/halts.csv",
                    help="Path to halts CSV (default: data/halts.csv)")
    ap.add_argument("--data", nargs="+", default=["stored_data_combined"],
                    help="Intraday data dirs to search (in order)")
    ap.add_argument("--start", help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", help="YYYY-MM-DD inclusive")
    ap.add_argument("--cash", type=float, default=25_000,
                    help="Cash per trade (default 25000)")
    ap.add_argument("--slippage", type=float, default=SLIPPAGE_PCT)
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).date() if args.start else None
    end = datetime.fromisoformat(args.end).date() if args.end else None
    halts = load_halts(args.halts, start, end)
    print(f"Loaded {len(halts)} halt events")

    trades: list[dict] = []
    for h in halts:
        out = backtest_one(h, args.data, args.cash, slippage_pct=args.slippage)
        if out is None:
            continue
        trades.append(out)

    # Per-trade printout for the realized trades
    real = [t for t in trades if "pnl" in t]
    for t in sorted(real, key=lambda x: (x["halt_date"], x["ticker"])):
        print(f"  {t['halt_date']} {t['ticker']:<6} {t['reason']:<5} "
              f"entry=${t['entry_price']:.3f}  exit=${t['exit_price']:.3f}  "
              f"({t['exit_reason']:<10})  PnL=${t['pnl']:+,.2f} ({t['pct']:+.1f}%)")

    summarize(trades)


if __name__ == "__main__":
    main()
