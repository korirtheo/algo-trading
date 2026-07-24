# -*- coding: utf-8 -*-
"""
OOS 2026 backtest: win rate by trade sequence per day, for G and L strategies.

For each day, sorts G trades by entry_time, then L trades by entry_time.
Computes win rate for the 1st, 2nd, 3rd, ... trade of the day.

Usage:
    python scripts/analysis/oos_trade_sequence.py                      # uses deployed config
    python scripts/analysis/oos_trade_sequence.py --config path.json   # custom config
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params, ALL_STRATS
from test_full import load_all_picks, MARGIN_THRESHOLD

DATA_DIRS = ["stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data_jul_2026"]
STARTING_CASH = 25_000
DEFAULT_CONFIG = "config/trial_oglhmafp_v5_best.json"


def load_config(path):
    """Load params from a config JSON file."""
    with open(path) as f:
        raw = json.load(f)
    # Configs may nest under "params" key
    if "params" in raw and isinstance(raw["params"], dict):
        return raw["params"]
    return raw


def run_backtest(params):
    """Run backtest, return list of (day, strategy, entry_time, pnl) for completed trades."""
    set_strategy_params(params)
    for s in ALL_STRATS:
        setattr(tgc, f"ENABLE_{s.upper()}", params.get(f"enable_{s}", False))

    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    oos_dates = [d for d in all_dates if d >= "2026-03-01"]

    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True

    cash = STARTING_CASH
    unsettled = 0.0
    all_trades = []  # (day, strategy, entry_time, pnl)

    for i, d in enumerate(oos_dates):
        picks = daily_picks.get(d, [])
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(picks, cash, is_live=False)
        except:
            end_c = cash
            unset = 0.0
            states = []
        pnl = end_c - cash
        cash = end_c
        if is_cash:
            cash += unset

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(oos_dates)}] {d}  cash=${cash:,.0f}  pnl=${pnl:+,.0f}", flush=True)

        for st in states:
            if st["exit_reason"] is not None:
                strat = st["strategy"]
                entry_time = st.get("entry_time", "")
                pnl_val = st.get("pnl", 0)
                all_trades.append((d, strat, entry_time, pnl_val))

    return all_trades, oos_dates


def analyze_sequence(trades, strategies=("G", "L")):
    """Group trades by day, sort by entry_time within each strategy, compute WR by sequence position."""
    results = {}

    for strat in strategies:
        strat_trades = [(d, et, pnl) for d, s, et, pnl in trades if s == strat]
        if not strat_trades:
            continue

        # Group by day
        by_day = {}
        for d, et, pnl in strat_trades:
            by_day.setdefault(d, []).append((et, pnl))

        # Sort each day by entry_time
        for d in by_day:
            by_day[d].sort(key=lambda x: x[0] if x[0] else "")

        # Collect trades by sequence position
        max_seq = max(len(v) for v in by_day.values())
        seq_data = {}  # seq_pos -> list of (day, pnl)
        for d, day_trades in by_day.items():
            for i, (et, pnl) in enumerate(day_trades):
                seq_data.setdefault(i + 1, []).append((d, pnl))

        # Compute stats for each sequence position
        seq_stats = {}
        for pos in range(1, max_seq + 1):
            if pos not in seq_data:
                continue
            day_pnls = seq_data[pos]
            n = len(day_pnls)
            wins = sum(1 for _, p in day_pnls if p > 0)
            wr = wins / n * 100 if n > 0 else 0
            avg_pnl = np.mean([p for _, p in day_pnls])
            total_pnl = sum(p for _, p in day_pnls)
            seq_stats[pos] = {
                "n_trades": n,
                "wins": wins,
                "losses": n - wins,
                "wr": wr,
                "avg_pnl": avg_pnl,
                "total_pnl": total_pnl,
            }

        results[strat] = {
            "total_trades": len(strat_trades),
            "total_days": len(by_day),
            "max_trades_per_day": max_seq,
            "sequence": seq_stats,
        }

    return results


def print_results(results, total_pnl, n_days):
    print(f"\nOverall: {n_days} OOS days, total PnL=${total_pnl:+,.0f}")

    for strat in ["G", "L"]:
        if strat not in results:
            continue
        r = results[strat]
        print(f"\n{'='*70}")
        print(f"  Strategy {strat}: {r['total_trades']} trades across {r['total_days']} days")
        print(f"  Max trades in a single day: {r['max_trades_per_day']}")
        print(f"{'='*70}")
        print(f"  {'Seq':>4}  {'Trades':>7}  {'Wins':>5}  {'Losses':>7}  {'WR':>7}  {'Avg PnL':>10}  {'Total PnL':>12}")
        print(f"  {'-'*4}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*12}")

        for pos, s in sorted(r["sequence"].items()):
            print(f"  {pos:>4}  {s['n_trades']:>7}  {s['wins']:>5}  {s['losses']:>7}  "
                  f"{s['wr']:>6.1f}%  ${s['avg_pnl']:>+9,.0f}  ${s['total_pnl']:>+11,.0f}")

        # Summary: 1st vs rest
        if 1 in r["sequence"] and len(r["sequence"]) > 1:
            first = r["sequence"][1]
            rest_trades = sum(s["n_trades"] for pos, s in r["sequence"].items() if pos > 1)
            rest_wins = sum(s["wins"] for pos, s in r["sequence"].items() if pos > 1)
            rest_wr = rest_wins / rest_trades * 100 if rest_trades > 0 else 0
            rest_avg = np.mean([p for pos, s in r["sequence"].items() if pos > 1
                                for _ in range(s["n_trades"]) for p in [s["avg_pnl"]]])
            print(f"\n  1st trade WR:     {first['wr']:.1f}%  (n={first['n_trades']}, avg PnL=${first['avg_pnl']:+,.0f})")
            print(f"  2nd+ trade WR:    {rest_wr:.1f}%  (n={rest_trades}, avg PnL=${rest_avg:+,.0f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOS trade sequence win rate analysis")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config JSON path")
    args = parser.parse_args()

    print(f"Config: {args.config}")
    params = load_config(args.config)

    enabled = sorted([k.replace("enable_", "").upper() for k, v in params.items()
                      if k.startswith("enable_") and v is True])
    print(f"Enabled strategies: {enabled}")

    print(f"\nRunning backtest (2026 OOS)...")
    trades, oos_dates = run_backtest(params)

    total_pnl = sum(pnl for _, _, _, pnl in trades)
    print(f"\nDone: {len(trades)} trades over {len(oos_dates)} days")

    results = analyze_sequence(trades)
    print_results(results, total_pnl, len(oos_dates))
