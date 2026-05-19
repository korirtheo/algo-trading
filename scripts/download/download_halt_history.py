"""
Download historical NASDAQ trade-halt data for backtest.

Source: https://www.nasdaqtrader.com/dynamic/symdir/tradehalts.txt
The public file holds the CURRENT trading day's halts only. Historical halt
data is available via the NASDAQ Trader site's archives, but a more reliable
approach is to incrementally append every weekday from a scheduled scrape.

This script supports both modes:
  --today           : fetch today's halt log and append to data/halts.csv
  --backfill PATH   : load a previously-saved historical CSV (any source) and
                       normalize into data/halts.csv

Output schema: data/halts.csv with columns
  halt_date, halt_time, ticker, reason, resume_time, resume_quote_time,
  resume_trade_time, halt_price, resume_price

The output is consumed by scripts/backtest/backtest_halt_resume.py.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import csv
import os
from datetime import datetime

from live.halt_monitor import fetch_halt_log, parse_halt_log, HaltEvent


def _project_root():
    return _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


OUT_DIR = _os.path.join(_project_root(), "data")
OUT_PATH = _os.path.join(OUT_DIR, "halts.csv")

CSV_HEADER = [
    "halt_date", "halt_time", "ticker", "reason",
    "resume_time", "resume_quote_time", "resume_trade_time",
    "halt_price", "resume_price",
]


def _row(ev: HaltEvent):
    return [
        ev.halt_date.isoformat() if ev.halt_date else "",
        ev.halt_time.isoformat() if ev.halt_time else "",
        ev.ticker or "",
        ev.reason or "",
        ev.resume_time.isoformat() if ev.resume_time else "",
        ev.resume_quote_time.isoformat() if ev.resume_quote_time else "",
        ev.resume_trade_time.isoformat() if ev.resume_trade_time else "",
        f"{ev.halt_price:.4f}" if ev.halt_price is not None else "",
        f"{ev.resume_price:.4f}" if ev.resume_price is not None else "",
    ]


def _load_existing_keys(path: str) -> set[str]:
    if not _os.path.exists(path):
        return set()
    keys = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(f"{row['halt_date']}|{row['ticker']}|{row['halt_time']}")
    return keys


def _ensure_outfile(path: str):
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    if not _os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def fetch_today(out_path: str = OUT_PATH) -> int:
    """Append today's NASDAQ halts to out_path. Returns rows written."""
    _ensure_outfile(out_path)
    existing = _load_existing_keys(out_path)

    print(f"Fetching: from NASDAQ tradehalts.txt")
    raw = fetch_halt_log()
    events = parse_halt_log(raw)
    print(f"Parsed: {len(events)} halt rows")

    written = 0
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        for ev in events:
            if ev.key in existing:
                continue
            w.writerow(_row(ev))
            written += 1
    print(f"Appended {written} new rows to {out_path}")
    return written


def backfill_from_csv(in_path: str, out_path: str = OUT_PATH) -> int:
    """Normalize a third-party halt CSV into our canonical schema."""
    _ensure_outfile(out_path)
    existing = _load_existing_keys(out_path)
    print(f"Backfilling from {in_path}")

    with open(in_path, newline="") as f:
        # Try pipe first, then comma
        sample = f.read(2048)
        f.seek(0)
        delim = "|" if "|" in sample else ","
        text = f.read()

    events = parse_halt_log(text) if delim == "|" else _parse_csv(text)
    print(f"Parsed: {len(events)} rows")

    written = 0
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        for ev in events:
            if ev.key in existing:
                continue
            w.writerow(_row(ev))
            written += 1
    print(f"Appended {written} new rows to {out_path}")
    return written


def _parse_csv(text: str) -> list[HaltEvent]:
    """Fallback parser for comma-delimited halt CSVs from other sources.
    Assumes the same column names as the NASDAQ feed (case-insensitive)."""
    from live.halt_monitor import parse_halt_log
    # parse_halt_log handles arbitrary delimiters via the leading-row pattern;
    # if you have a non-standard schema, normalize externally first.
    return parse_halt_log(text.replace(",", "|"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", action="store_true", help="Fetch today's halts and append.")
    ap.add_argument("--backfill", metavar="PATH", help="Backfill from a saved halt CSV/TSV.")
    ap.add_argument("--out", default=OUT_PATH, help=f"Output CSV (default: {OUT_PATH})")
    args = ap.parse_args()

    if not args.today and not args.backfill:
        ap.error("Pass --today or --backfill PATH")

    if args.today:
        fetch_today(args.out)
    if args.backfill:
        backfill_from_csv(args.backfill, args.out)


if __name__ == "__main__":
    main()
