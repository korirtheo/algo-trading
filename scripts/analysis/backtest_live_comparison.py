"""
Backtest-vs-Live Comparison Tool

Loads bars logged by the live engine from logs/bars/<date>/<symbol>.csv
and replays them through the backtest simulator to compare simulated vs
actual trades.

Usage:
  python scripts/analysis/backtest_live_comparison.py [--date 2026-07-14] [--symbol FCEL]

If no date is given, defaults to today. If no symbol is given, backtests all
symbols in that day's logs.
"""
import sys
import os
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

import test_green_candle_combined as tgc
from test_full import MARGIN_THRESHOLD
from optimize_combined import set_strategy_params, _build_param_snapshot, _param_lock

# Deployed config
CONFIG_PATH = "config/trial_g511_l626_v3_overlay.json"
LOGS_DIR = "logs/bars"

def load_bars_from_csv(date_str, symbol, bars_type="intraday"):
    """Load a symbol's bars logged by the live engine.

    Args:
        date_str: YYYY-MM-DD format date
        symbol: ticker symbol
        bars_type: "intraday" for 2-min aggregated bars (for strategy backtest),
                  or "raw-1min" for raw 1-min bars (for detailed audit)

    Returns a list of dicts: [{"timestamp": ..., "Open": ..., "High": ..., ...}, ...]
    """
    bar_path = Path(LOGS_DIR) / bars_type / date_str / f"{symbol}.csv"
    if not bar_path.exists():
        return None

    try:
        df = pd.read_csv(bar_path)
        bars = []
        for _, row in df.iterrows():
            bars.append({
                "timestamp": row["timestamp"],
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": int(row["Volume"]),
            })
        return bars
    except Exception as e:
        print(f"ERROR loading {bar_path}: {e}")
        return None


def build_picks_from_logs(date_str, bars_type="intraday"):
    """Scan all bars in the log directory for that date and create picks.

    Args:
        date_str: YYYY-MM-DD format date
        bars_type: "intraday" for 2-min aggregated (strategy backtest) or "raw-1min"

    Returns a list of pick dicts suitable for simulate_day_combined().
    """
    bars_date_dir = Path(LOGS_DIR) / bars_type / date_str
    if not bars_date_dir.exists():
        print(f"ERROR: {bars_date_dir} not found")
        return []

    picks = []
    for csv_file in sorted(bars_date_dir.glob("*.csv")):
        symbol = csv_file.stem
        bars = load_bars_from_csv(date_str, symbol, bars_type=bars_type)

        if not bars or len(bars) == 0:
            continue

        # Get prev close (assume first bar open if we don't have daily data)
        # In a real scenario, we'd load this from daily logs or external source
        prev_close = bars[0]["Close"]  # Fallback; improve if daily logs available
        open_price = bars[0]["Open"]

        gap_pct = (open_price / prev_close - 1) * 100 if prev_close > 0 else 0

        if gap_pct < 2.0:  # Minimum gap filter
            continue

        # Compute premarket volume (bars before 9:30 AM ET)
        pm_volume = 0
        for bar in bars:
            ts = pd.Timestamp(bar["timestamp"])
            if ts.hour < 13 or (ts.hour == 13 and ts.minute < 30):  # Before 9:30 AM ET (13:30 UTC)
                pm_volume += bar["Volume"]
            else:
                break

        pm_dollar_vol = pm_volume * prev_close

        if pm_dollar_vol < 250_000:  # Minimum PM volume filter
            continue

        pick = {
            "ticker": symbol,
            "prev_close": prev_close,
            "open": open_price,
            "gap_pct": gap_pct,
            "pm_volume": pm_volume,
            "pm_dollar_vol": pm_dollar_vol,
        }
        picks.append(pick)

    picks.sort(key=lambda x: x["gap_pct"], reverse=True)
    return picks


def run_backtest(picks, config, date_str):
    """Run backtest on picked symbols using logged bars."""
    print(f"\nRunning backtest on {len(picks)} symbols from {date_str}...")

    # Configure simulator
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

    params = config["params"]
    with _param_lock:
        set_strategy_params(params)
        snapshot = _build_param_snapshot()

    cash = 25000.0
    cash_account = cash < MARGIN_THRESHOLD

    try:
        states, final_cash, unsettled, day_stats = tgc.simulate_day_combined(
            picks, cash, cash_account, params=snapshot
        )
    except Exception as e:
        print(f"ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

    return {
        "states": states,
        "final_cash": final_cash,
        "unsettled": unsettled,
        "day_stats": day_stats,
        "starting_cash": cash,
    }


def print_backtest_results(results, picks):
    """Print backtest summary."""
    if results is None:
        return

    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"Starting cash: ${results['starting_cash']:,.2f}")
    print(f"Final cash:    ${results['final_cash']:,.2f}")
    print(f"Unsettled:     ${results['unsettled']:,.2f}")
    print(f"Effective:     ${results['final_cash'] + results['unsettled']:,.2f}")
    print(f"PnL:           ${results['final_cash'] + results['unsettled'] - results['starting_cash']:+,.2f}")

    states = results["states"]
    trades = [s for s in states if s.get("exit_reason") is not None]
    print(f"\nTotal trades: {len(trades)}")

    if len(trades) == 0:
        print("⚠️ NO TRADES EXECUTED")
        print(f"\nTop 10 picks (by gap %):")
        for p in picks[:10]:
            print(f"  {p['ticker']:>6}  gap={p['gap_pct']:>6.2f}%  pm_$vol=${p['pm_dollar_vol']:>10,.0f}")
        return

    wins = [s for s in trades if s.get("pnl", 0) > 0]
    losses = [s for s in trades if s.get("pnl", 0) <= 0]
    total_pnl = sum(s.get("pnl", 0) for s in trades)

    print(f"  Wins:   {len(wins)}")
    print(f"  Losses: {len(losses)}")
    print(f"  WR:     {len(wins)/len(trades)*100:.1f}%")
    print(f"  Avg PnL: ${total_pnl/len(trades):,.2f}")

    print(f"\nTrade details (first 20):")
    for i, s in enumerate(trades[:20], 1):
        print(f"  [{i:2d}] {s['ticker']:>6}  {s['strategy']:>2}  "
              f"entry=${s.get('entry_price', 0):>8.2f}  exit=${s.get('exit_price', 0):>8.2f}  "
              f"pnl=${s.get('pnl', 0):>+8,.2f}  reason={s['exit_reason']}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest against live-logged bar data"
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date to backtest (YYYY-MM-DD, default=today)"
    )
    parser.add_argument(
        "--symbol",
        help="Single symbol to backtest (if omitted, tests all symbols in logs for that date)"
    )
    parser.add_argument(
        "--bars-type",
        choices=["intraday", "raw-1min"],
        default="intraday",
        help="Bar type to backtest: intraday (2-min aggregated, default) or raw-1min"
    )

    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"Backtest-vs-Live Comparison Tool")
    print(f"{'='*70}")
    print(f"Date: {args.date}")
    print(f"Config: {CONFIG_PATH}")

    # Load deployed config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    print(f"Label: {config['label']}")

    # Build picks from logged bars
    if args.symbol:
        bars = load_bars_from_csv(args.date, args.symbol, bars_type=args.bars_type)
        if bars is None:
            print(f"ERROR: No {args.bars_type} bars found for {args.symbol} on {args.date}")
            return

        # Single symbol: manually build pick
        prev_close = bars[0]["Close"]
        open_price = bars[0]["Open"]
        gap_pct = (open_price / prev_close - 1) * 100

        picks = [{
            "ticker": args.symbol,
            "prev_close": prev_close,
            "open": open_price,
            "gap_pct": gap_pct,
            "pm_volume": sum(b["Volume"] for b in bars),
            "pm_dollar_vol": sum(b["Volume"] for b in bars) * prev_close,
        }]

        print(f"Symbol: {args.symbol}")
    else:
        picks = build_picks_from_logs(args.date, bars_type=args.bars_type)

        if len(picks) == 0:
            print(f"ERROR: No picks found in {LOGS_DIR}/{args.bars_type}/{args.date}/")
            return

        print(f"Found {len(picks)} gap-ups in logs")
        print(f"Top 5:")
        for p in picks[:5]:
            print(f"  {p['ticker']:>6}  gap={p['gap_pct']:>6.2f}%")

    # Run backtest
    results = run_backtest(picks, config, args.date)

    # Print results
    print_backtest_results(results, picks)


if __name__ == "__main__":
    main()
