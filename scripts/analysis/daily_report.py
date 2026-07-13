"""
Daily Trading Report — post-market pipeline.

Collects today's live data from AWS, computes trade/fill/anomaly stats,
and writes a Markdown report + memory entries to the project knowledge base.

Usage:
    python scripts/analysis/daily_report.py                          # today
    python scripts/analysis/daily_report.py --date 2026-06-26         # specific day
    python scripts/analysis/daily_report.py --no-fetch                # re-report from cached files
    python scripts/analysis/daily_report.py --date 2026-07-13 --aws   # fetch+cache only, no report
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

# ── paths ───────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Memory dir is in the .claude project memory directory
CLAUDEDIR = Path.home() / ".claude" / "projects"
_mem_candidates = [d for d in CLAUDEDIR.glob("c--Users-Theo-Korir-Documents-Python-algo-trading/memory")]
MEMORY_DIR = _mem_candidates[0] if _mem_candidates else PROJECT_ROOT / "memory"

AWS_HOST = "54.172.65.25"
AWS_PORT = "2222"
SSH_KEY = PROJECT_ROOT / "trading-key-v2.pem"
# Files are mounted on the host under /home/ubuntu/algo-trading/logs/
REMOTE_DATA_DIR = "/home/ubuntu/algo-trading/logs"


# ── helpers ──────────────────────────────────────────────────────────────────
def et_now() -> date:
    """Return today's date in US/Eastern (market calendar)."""
    # approximate: during EDT (Mar-Nov) UTC-4
    now_utc = datetime.now(timezone.utc)
    est = now_utc - timedelta(hours=4) if now_utc.astimezone().dst() else now_utc - timedelta(hours=5)
    return est.date()


def run(cmd: list[str], capture: bool = True, check: bool = True) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(cmd, capture_output=capture, text=True, check=check)
    return result.stdout.strip() if capture else ""


def scp_from(remote_path: str, local_path: str) -> bool:
    """SCP a single file from AWS. Returns True if successful. Falls back to docker exec path."""
    # Try host path
    try:
        run([
            "scp", "-i", str(SSH_KEY), "-P", AWS_PORT,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"ubuntu@{AWS_HOST}:{remote_path}", local_path,
        ])
        if Path(local_path).stat().st_size > 0:
            return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Fallback: try inside docker (use docker cp via SSH pipe)
    docker_path = remote_path.replace("/home/ubuntu/algo-trading/logs", "/app/logs")
    try:
        run([
            "ssh", "-i", str(SSH_KEY), "-p", AWS_PORT,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"ubuntu@{AWS_HOST}",
            f"docker exec algotrader cat {docker_path}",
        ], capture=False)
        # Write stdout to file
        result = subprocess.run([
            "ssh", "-i", str(SSH_KEY), "-p", AWS_PORT,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"ubuntu@{AWS_HOST}",
            f"docker exec algotrader cat {docker_path}",
        ], capture_output=True, text=True, check=False)
        if result.returncode == 0 and len(result.stdout) > 0:
            with open(local_path, "w") as f:
                f.write(result.stdout)
            return True
    except (subprocess.CalledProcessError, OSError):
        pass

    return False


# ── data fetchers ────────────────────────────────────────────────────────────
def fetch_data(target_date: str, cache_dir: Path) -> dict[str, Path | None]:
    """SCP today's data files from AWS into cache_dir. Returns dict of paths."""
    fetched: dict[str, Path | None] = {}

    files = {
        "trades_json": f"{REMOTE_DATA_DIR}/{target_date}_trades.json",
        "fills_csv": f"{REMOTE_DATA_DIR}/fills_calibration.csv",
        "gainers_jsonl": f"{REMOTE_DATA_DIR}/intraday_gainers/{target_date}.jsonl",
        "live_log": f"{REMOTE_DATA_DIR}/{target_date}_live.log",
    }

    for key, remote_path in files.items():
        local = cache_dir / f"{key}_{target_date}"
        ok = scp_from(remote_path, str(local))
        fetched[key] = local if ok and local.stat().st_size > 0 else None

    return fetched


# ── parsers ──────────────────────────────────────────────────────────────────
def parse_trades_json(path: Path | None) -> list[dict]:
    """Parse trades JSON: list of {ticker, strategy, pnl, entry/exit, reason}."""
    if not path:
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def parse_fills_csv(path: Path | None) -> list[dict]:
    """Parse fills CSV. Return list of dicts limited to target_date."""
    if not path:
        return []
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Fill rows don't have a date column directly — filter by ts_signal
                rows.append(row)
    except (FileNotFoundError, csv.Error):
        pass
    return rows


def filter_fills_for_date(fills: list[dict], target_date: str) -> list[dict]:
    """Return only fills whose ts_signal starts with target_date."""
    return [f for f in fills if f.get("ts_signal", "").startswith(target_date)]


def parse_gainers(path: Path | None) -> dict:
    """Parse intraday gainers JSONL. Return summary stats."""
    if not path:
        return {"polls": 0, "total_discovered": 0, "distinct_tickers": set()}
    tickers = set()
    polls = 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                polls += 1
                obj = json.loads(line)
                for a in obj.get("active", []):
                    tickers.add(a.get("ticker"))
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    return {"polls": polls, "total_discovered": len(tickers), "distinct_tickers": tickers}


def parse_log_summary(path: Path | None, target_date: str) -> dict:
    """Extract key summary info from the engine log for this day."""
    info: dict = {
        "emit_2min_count": 0,
        "first_bar_time": None,
        "last_bar_time": None,
        "errors": [],
        "v3_entries": 0,
        "g_entries": 0,
        "l_entries": 0,
    }
    if not path:
        return info
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if "EMIT 2min" in line:
                    info["emit_2min_count"] += 1
                    # Extract time from "[HH:MM:SS ET ...]"
                    if info["first_bar_time"] is None:
                        info["first_bar_time"] = line[:8]
                    info["last_bar_time"] = line[:8]
                elif "ERROR" in line or "CRITICAL" in line:
                    info["errors"].append(line.strip()[:120])
                elif "entry signal" in line.lower() and "v3" in line.lower():
                    info["v3_entries"] += 1
                elif "entry signal" in line.lower() and (" strategy=G " in line or " strategy='G'" in line):
                    info["g_entries"] += 1
                elif "entry signal" in line.lower() and (" strategy=L " in line or " strategy='L'" in line):
                    info["l_entries"] += 1
    except FileNotFoundError:
        pass
    return info


# ── analysis ─────────────────────────────────────────────────────────────────
def analyze_trades(trades: list[dict]) -> dict:
    """Compute trade-level statistics."""
    if not trades:
        return {"count": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "avg_pnl": 0.0, "max_win": 0.0, "max_loss": 0.0,
                "by_strategy": {}, "by_ticker": {}}

    pnls = [t.get("pnl", 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    by_strat: dict[str, list[float]] = defaultdict(list)
    by_ticker: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        strat = t.get("strategy", "?")
        ticker = t.get("ticker", "?")
        pnl = t.get("pnl", 0)
        by_strat[strat].append(pnl)
        by_ticker[ticker].append(pnl)

    def stats(plist: list[float]) -> dict:
        return {
            "count": len(plist),
            "total": sum(plist),
            "avg": sum(plist) / len(plist) if plist else 0.0,
            "max": max(plist) if plist else 0.0,
            "min": min(plist) if plist else 0.0,
        }

    return {
        "count": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "total_pnl": sum(pnls),
        "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
        "max_win": max(wins) if wins else 0.0,
        "max_loss": min(losses) if losses else 0.0,
        "med_pnl": sorted(pnls)[len(pnls) // 2] if pnls else 0.0,
        "by_strategy": {k: stats(v) for k, v in sorted(by_strat.items())},
        "by_ticker": {k: stats(v) for k, v in sorted(by_ticker.items())},
    }


def analyze_fills(fills: list[dict]) -> dict:
    """Compute slippage statistics from fills data."""
    if not fills:
        return {"count": 0, "buys": 0, "sells": 0, "by_strategy": {}}

    slip_bps: list[float] = []
    by_strat: dict[str, list[float]] = defaultdict(list)
    buys = sells = 0

    for f in fills:
        side = f.get("side", "")
        strat = f.get("strategy", "?")
        if side == "buy":
            buys += 1
        elif side == "sell":
            sells += 1

        try:
            sb = float(f["slip_bp"]) if f.get("slip_bp") else None
        except (ValueError, TypeError):
            sb = None
        if sb is not None:
            slip_bps.append(sb)
            by_strat[strat].append(sb)

    def slip_stats(bps: list[float]) -> dict:
        if not bps:
            return {"count": 0, "mean_bp": 0.0, "median_bp": 0.0}
        return {
            "count": len(bps),
            "mean_bp": sum(bps) / len(bps),
            "median_bp": sorted(bps)[len(bps) // 2],
            "p90_bp": sorted(bps)[int(len(bps) * 0.9)],
            "max_bp": max(bps),
        }

    return {
        "count": len(fills),
        "buys": buys,
        "sells": sells,
        "overall": slip_stats(slip_bps),
        "by_strategy": {k: slip_stats(v) for k, v in sorted(by_strat.items())},
    }


def detect_anomalies(trades: list[dict], trade_stats: dict, fills: list[dict],
                     log_info: dict, gainers: dict) -> list[dict]:
    """Detect anomalous conditions for the day. Returns list of findings."""
    findings = []
    pnl = trade_stats["total_pnl"]

    # Zero trades
    if trade_stats["count"] == 0:
        if log_info["emit_2min_count"] > 0:
            findings.append({
                "severity": "high",
                "title": "Zero trades despite bars flowing",
                "detail": f"Engine emitted {log_info['emit_2min_count']} bars but no trades executed.",
                "category": "execution",
            })
        elif log_info["emit_2min_count"] == 0:
            findings.append({
                "severity": "high",
                "title": "No bars emitted — engine possibly stalled",
                "detail": "Zero EMIT 2min events detected. Engine may have restarted or stalled.",
                "category": "engine",
            })
        else:
            findings.append({
                "severity": "info",
                "title": "No trades today",
                "detail": "No qualifying signals or positions entered.",
                "category": "info",
            })

    # Large loss day
    if trade_stats["count"] > 0 and pnl < -500:
        findings.append({
            "severity": "high",
            "title": f"Large loss day: ${pnl:,.0f}",
            "detail": f"Largest loss ${trade_stats['max_loss']:,.2f}. "
                      f"WR {trade_stats['win_rate']:.0f}% ({trade_stats['wins']}W/{trade_stats['losses']}L).",
            "category": "pnl",
        })

    # Consecutive losses by strategy
    for strat, s in trade_stats.get("by_strategy", {}).items():
        if s["count"] >= 3 and s["total"] < 0:
            findings.append({
                "severity": "medium",
                "title": f"Strategy {strat} underwater: ${s['total']:,.0f} across {s['count']} trades",
                "detail": f"WR not computable from aggregate, but total negative. "
                          f"Avg trade ${s['avg']:,.2f}.",
                "category": "strategy",
            })

    # High slippage
    for strat, ss in fills.get("by_strategy", {}).items():
        if ss["count"] >= 5 and abs(ss["median_bp"]) > 10:
            findings.append({
                "severity": "medium",
                "title": f"High slippage on {strat}: median {ss['median_bp']:.1f}bp",
                "detail": f"{ss['count']} fills, mean {ss['mean_bp']:.1f}bp, p90 {ss['p90_bp']:.1f}bp.",
                "category": "slippage",
            })

    # No scanner picks
    if gainers["total_discovered"] == 0:
        findings.append({
            "severity": "info",
            "title": "No intraday gainers detected",
            "detail": "Intraday gainers monitor found no qualifying tickers.",
            "category": "scanner",
        })

    # Engine errors
    if len(log_info["errors"]) >= 5:
        findings.append({
            "severity": "medium",
            "title": f"{len(log_info['errors'])} error lines in log",
            "detail": f"First error: {log_info['errors'][0][:100]}",
            "category": "engine",
        })

    return findings


# ── report writer ────────────────────────────────────────────────────────────
def write_report(target_date: str, trade_stats: dict, fill_stats: dict,
                 log_info: dict, gainers: dict, anomalies: list[dict]) -> str:
    """Write the daily report to a Markdown file. Returns the file path."""
    report_path = MEMORY_DIR / f"daily_{target_date}.md"

    lines = [
        f"---",
        f"date: {target_date}",
        f"type: daily-report",
        f"---",
        f"",
        f"# Daily Report — {target_date}",
        f"",
    ]

    # ── PnL Summary ──
    pnl = trade_stats["total_pnl"]
    pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
    lines += [
        f"## PnL Summary",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Daily PnL | **{pnl_str}** |",
        f"| Trades | {trade_stats['count']} |",
        f"| Wins/Losses | {trade_stats['wins']}W / {trade_stats['losses']}L |",
        f"| Win Rate | {trade_stats['win_rate']:.1f}% |",
        f"| Avg PnL | ${trade_stats['avg_pnl']:,.2f} |",
        f"| Median PnL | ${trade_stats['med_pnl']:,.2f} |",
        f"| Max Win | ${trade_stats['max_win']:,.2f} |",
        f"| Max Loss | ${trade_stats['max_loss']:,.2f} |",
        f"",
    ]

    # ── Per-Strategy ──
    if trade_stats["by_strategy"]:
        lines += [f"## By Strategy", f""]
        lines += [f"| Strategy | Trades | Total PnL | Avg PnL | Best | Worst |",
                  f"|---|---|---|---|---|---|"]
        for strat, s in trade_stats["by_strategy"].items():
            tp = f"+${s['total']:,.0f}" if s['total'] >= 0 else f"-${abs(s['total']):,.0f}"
            ap = f"+${s['avg']:,.2f}" if s['avg'] >= 0 else f"-${abs(s['avg']):,.2f}"
            lines.append(f"| {strat} | {s['count']} | {tp} | {ap} | ${s['max']:,.2f} | ${s['min']:,.2f} |")
        lines.append("")

    # ── Slippage ──
    if fill_stats["count"] > 0:
        os_ = fill_stats["overall"]
        lines += [f"## Slippage", f""]
        lines += [f"| Metric | Value |", f"|---|---|"]
        lines += [f"| Fills | {fill_stats['count']} ({fill_stats['buys']}B/{fill_stats['sells']}S) |"]
        lines += [f"| Mean Slip | {os_['mean_bp']:.2f} bp |"]
        lines += [f"| Median Slip | {os_['median_bp']:.2f} bp |"]
        lines += [f"| P90 Slip | {os_['p90_bp']:.2f} bp |"]
        lines += [f""]

        if fill_stats["by_strategy"]:
            lines += [f"### Slippage by Strategy", f""]
            lines += [f"| Strategy | Fills | Mean bp | Med bp | P90 bp |",
                      f"|---|---|---|---|---|"]
            for strat, ss in fill_stats["by_strategy"].items():
                lines.append(f"| {strat} | {ss['count']} | {ss['mean_bp']:.1f} | {ss['median_bp']:.1f} | {ss['p90_bp']:.1f} |")
            lines.append("")

    # ── Engine ──
    lines += [f"## Engine", f""]
    lines += [f"| Metric | Value |", f"|---|---|"]
    lines += [f"| 2-min Bars Emitted | {log_info['emit_2min_count']} |"]
    lines += [f"| First Bar | {log_info['first_bar_time'] or 'N/A'} |"]
    lines += [f"| Last Bar | {log_info['last_bar_time'] or 'N/A'} |"]
    lines += [f""]

    # ── Scanner ──
    lines += [f"## Scanner", f""]
    lines += [f"| Metric | Value |", f"|---|---|"]
    lines += [f"| Intraday Gainers Discovered | {gainers['total_discovered']} |"]
    lines += [f"| Scanner Polls | {gainers['polls']} |"]
    lines += [f""]

    # ── Anomalies ──
    if anomalies:
        lines += [f"## Anomalies", f""]
        for a in anomalies:
            icon = {"high": "🔴", "medium": "🟡", "info": "ℹ️"}.get(a["severity"], "•")
            lines.append(f"- {icon} **{a['title']}** — {a['detail']}")
        lines.append("")

    # ── Per-Ticker Detail ──
    if trade_stats["by_ticker"]:
        lines += [f"## Per-Ticker Detail", f""]
        lines += [f"| Ticker | Trades | PnL | Avg PnL |",
                  f"|---|---|---|---|"]
        for ticker, s in sorted(trade_stats["by_ticker"].items(), key=lambda x: abs(x[1]["total"]), reverse=True):
            tp = f"+${s['total']:,.0f}" if s['total'] >= 0 else f"-${abs(s['total']):,.0f}"
            lines.append(f"| {ticker} | {s['count']} | {tp} | ${s['avg']:,.2f} |")
        lines.append("")

    # ── Raw Trades ──
    # lines += ["## Raw Trades", ""]  # commented out for brevity; can add --verbose

    content = "\n".join(lines)

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  Report written: {report_path}")
    return str(report_path)


def add_to_memory_index(report_path: str, target_date: str):
    """Add the report link to MEMORY.md if not already present."""
    mem_path = MEMORY_DIR / "MEMORY.md"
    link = f"- [Daily Report {target_date}](daily_{target_date}.md)"
    if mem_path.exists():
        with open(mem_path, encoding="utf-8") as f:
            content = f.read()
        if f"daily_{target_date}.md" in content:
            return  # already linked
        # Append after the User Preferences section
        marker = "## User Preferences"
        if marker in content:
            idx = content.rfind(marker)
            insert = content[:idx] + link + "\n" + content[idx:]
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write(insert)
        else:
            # Append at end
            with open(mem_path, "a", encoding="utf-8") as f:
                f.write(f"\n{link}\n")


# ── write finding to memory (for anomalies) ──────────────────────────────────
def write_anomaly_memory(anomaly: dict, target_date: str):
    """Write a single anomaly as a standalone memory file for high-severity items."""
    if anomaly["severity"] not in ("high",):
        return
    slug = anomaly["title"].lower().replace(" ", "-").replace("--", "-")[:40]
    slug = "".join(c for c in slug if c.isalnum() or c in "-_")
    slug = f"anomaly-{target_date}-{slug}"[:60]
    fpath = MEMORY_DIR / f"{slug}.md"

    if fpath.exists():
        return  # don't overwrite

    content = f"""---
name: {slug}
description: "Anomaly detected {target_date}: {anomaly['title']}"
metadata:
  type: feedback
  date: {target_date}
  severity: {anomaly['severity']}
---

**Date:** {target_date}

## Anomaly: {anomaly['title']}

{anomaly['detail']}

**Category:** {anomaly['category']}

**Why this matters:** Flagged by the daily report pipeline for review.
"""
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Anomaly memory: {fpath}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Daily trading report pipeline")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today ET)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip AWS fetch, use cached files from ./logs/")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Fetch data to cache dir and exit (no report)")
    args = parser.parse_args()

    target_date = args.date or et_now().isoformat()
    print(f"Daily Report - {target_date}")
    print(f"{'='*50}")

    # ── 1. Fetch / load data ──
    cache_dir = LOGS_DIR / target_date
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.no_fetch:
        # Use cached files
        fetched = {
            "trades_json": cache_dir / f"trades_json_{target_date}",
            "fills_csv": cache_dir / f"fills_csv_{target_date}",
            "gainers_jsonl": cache_dir / f"gainers_jsonl_{target_date}",
            "live_log": cache_dir / f"live_log_{target_date}",
        }
        for k, p in list(fetched.items()):
            if not p or not p.exists():
                fetched[k] = None
    else:
        print(f"  Fetching from AWS ({AWS_HOST})...")
        fetched = fetch_data(target_date, cache_dir)

        # fills CSV is a single growing file — copy it to a dated snapshot
        if fetched["fills_csv"] and fetched["fills_csv"].exists():
            snapshot = cache_dir / f"fills_snapshot_{target_date}.csv"
            if snapshot.exists():
                snapshot.unlink()
            fetched["fills_csv"].rename(snapshot)
            fetched["fills_csv"] = snapshot

    if args.fetch_only:
        print("  Fetch-only mode: exiting.")
        for k, v in fetched.items():
            print(f"    {k}: {'✓' if v else '✗'}")
        return

    # ── 2. Parse ──
    print("  Parsing...")
    trades = parse_trades_json(fetched["trades_json"])
    fills_all = parse_fills_csv(fetched["fills_csv"])
    fills_today = filter_fills_for_date(fills_all, target_date)
    gainers = parse_gainers(fetched["gainers_jsonl"])
    log_info = parse_log_summary(fetched["live_log"], target_date)

    print(f"    Trades: {len(trades)} | Fills: {len(fills_today)} | "
          f"Gainers: {gainers['total_discovered']} | Bars: {log_info['emit_2min_count']}")

    # ── 3. Analyze ──
    print("  Analyzing...")
    trade_stats = analyze_trades(trades)
    fill_stats = analyze_fills(fills_today)
    anomalies = detect_anomalies(trades, trade_stats, fill_stats, log_info, gainers)

    # ── 4. Anomaly memory files ──
    for a in anomalies:
        write_anomaly_memory(a, target_date)

    # ── 5. Write report ──
    print("  Writing report...")
    report_path = write_report(target_date, trade_stats, fill_stats, log_info, gainers, anomalies)

    # ── 6. Index ──
    add_to_memory_index(report_path, target_date)

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"Done - {len(anomalies)} anomalies, report saved.")
    sev_count = sum(1 for a in anomalies if a["severity"] == "high")
    if sev_count:
        print(f"  ** {sev_count} high-severity anomalies - check memory files.")


if __name__ == "__main__":
    main()
