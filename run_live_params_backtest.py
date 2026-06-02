"""
Backtest using the LIVE engine's params (config/trial_432_params.json) on a
chosen data directory, with configurable starting cash.

This is the apples-to-apples comparison for the live system: same params,
same simulate_day_combined code path, just replayed against cached picks.

Usage:
  python run_live_params_backtest.py
  python run_live_params_backtest.py --data stored_data_mar_may_2026
  python run_live_params_backtest.py --data stored_data_mar_may_2026 --cash 10000
  python run_live_params_backtest.py --data stored_data_mar_may_2026 --cash 10000 --start 2026-03-15 --end 2026-05-16
"""
import argparse
import json
import os
import sys

import numpy as np

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD


def load_live_params(path):
    with open(path) as f:
        params = json.load(f)
    # Optimizer dumps wrap the flat param dict under "params" alongside metadata
    # (trial_number, score, pf, etc.). Unwrap if we see that shape.
    if isinstance(params, dict) and "params" in params and isinstance(params["params"], dict):
        params = params["params"]
    # Optimizer only suggests params for ENABLED strategies; disabled strats
    # have no entry in the dump. Backfill missing keys from the live config
    # so set_strategy_params doesn't KeyError on a disabled strategy.
    baseline_path = os.path.join(os.path.dirname(__file__),
                                  "config", "trial_432_params.json")
    if os.path.exists(baseline_path) and baseline_path != os.path.abspath(path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        merged = dict(baseline)
        merged.update(params)
        params = merged
    set_strategy_params(params)
    enabled = []
    for s in "HGAFDVPMRWOBKCEIJNL":
        gap_attr = f"{s}_MIN_GAP_PCT" if s != "R" else "R_DAY1_MIN_GAP"
        if getattr(tgc, gap_attr, 9999) < 9000:
            enabled.append(s)
    print(f"Loaded {len(params)} params from {os.path.basename(path)}")
    print(f"  Enabled: {', '.join(enabled)}")
    print(f"  Priority: {tgc.STRAT_PRIORITY}")
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="stored_data_mar_may_2026",
                        help="Data directory (default: stored_data_mar_may_2026)")
    parser.add_argument("--cash", type=float, default=10_000,
                        help="Starting cash (default: $10,000)")
    parser.add_argument("--params", default="config/trial_432_params.json",
                        help="Live params JSON (default: config/trial_432_params.json)")
    parser.add_argument("--start", default=None, help="Restrict to dates >= YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Restrict to dates <= YYYY-MM-DD")
    parser.add_argument("--slippage", type=float, default=None,
                        help="Override SLIPPAGE_PCT (e.g. 1.0 = 1%% per leg)")
    parser.add_argument("--vol-cap", type=float, default=None,
                        help="Override VOL_CAP_PCT (e.g. 1.0 = 1%% of cumulative dollar vol)")
    parser.add_argument("--cash-account-always", action="store_true",
                        help="Force T+1 settlement always (ignore $25K margin threshold)")
    parser.add_argument("--dynamic-slip", action="store_true",
                        help="Enable liquidity-aware slippage (base_spread + sqrt-participation impact)")
    parser.add_argument("--slip-impact-k", type=float, default=None,
                        help="Override SLIP_IMPACT_K (default 3.0; higher = more punitive)")
    args = parser.parse_args()

    # Apply slippage / vol-cap overrides to the simulator module globals
    if args.slippage is not None:
        tgc.SLIPPAGE_PCT = float(args.slippage)
    if args.vol_cap is not None:
        tgc.VOL_CAP_PCT = float(args.vol_cap)
    if args.dynamic_slip:
        tgc.USE_DYNAMIC_SLIPPAGE = True
    if args.slip_impact_k is not None:
        tgc.SLIP_IMPACT_K = float(args.slip_impact_k)
    margin_threshold = float("inf") if args.cash_account_always else MARGIN_THRESHOLD

    print("=" * 72)
    print(f"  LIVE-PARAMS BACKTEST")
    print(f"  Data:     {args.data}")
    print(f"  Cash:     ${args.cash:,.0f}")
    if getattr(tgc, "USE_DYNAMIC_SLIPPAGE", False):
        print(f"  Slippage: DYNAMIC (base={tgc.SLIP_BASE_SPREAD}+{tgc.SLIP_PRICE_COEFF}/price + {tgc.SLIP_IMPACT_K}*sqrt(participation))")
    else:
        print(f"  Slippage: {tgc.SLIPPAGE_PCT}% per leg (legacy constant)")
    print(f"  Vol cap:  {tgc.VOL_CAP_PCT}% of cumulative dollar vol")
    print(f"  Margin:   {'never (always cash-account T+1)' if margin_threshold == float('inf') else f'unlocked at ${MARGIN_THRESHOLD:,}'}")
    print(f"  Params:   {args.params}")
    if args.start or args.end:
        print(f"  Dates:    {args.start or 'start'} -> {args.end or 'end'}")
    print("=" * 72)

    load_live_params(args.params)
    print()

    all_dates, picks_by_date = load_all_picks([args.data])

    if args.start:
        all_dates = [d for d in all_dates if d >= args.start]
    if args.end:
        all_dates = [d for d in all_dates if d <= args.end]

    if not all_dates:
        print("ERROR: no trading days in range.")
        sys.exit(1)

    print(f"Trading days: {all_dates[0]} -> {all_dates[-1]} ({len(all_dates)} days)")
    print()

    cash = float(args.cash)
    unsettled = 0.0
    starting_cash = cash
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    daily_pnls = []
    exit_reasons = {}
    strat_stats = {}

    for i, d in enumerate(all_dates):
        cash += unsettled
        unsettled = 0.0

        picks = picks_by_date.get(d, [])
        cash_account = cash < margin_threshold

        if not picks:
            daily_pnls.append(0.0)
            continue

        states, cash, unsettled, _ = tgc.simulate_day_combined(picks, cash, cash_account=cash_account)

        day_pnl = 0.0
        day_trades = 0
        day_wins = 0
        day_detail = []
        for st in states:
            if st.get("exit_price") is None:
                continue
            pnl = st.get("pnl", 0)
            day_pnl += pnl
            day_trades += 1
            if pnl > 0:
                day_wins += 1
            reason = st.get("exit_reason", "UNKNOWN")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            strat = st.get("strategy", "?")
            s = strat_stats.setdefault(strat, {"n": 0, "wins": 0, "pnl": 0.0})
            s["n"] += 1
            s["pnl"] += pnl
            if pnl > 0:
                s["wins"] += 1
            day_detail.append(f"{st['ticker']}({strat}, ${pnl:+,.0f})")

        total_trades += day_trades
        total_wins += day_wins
        total_pnl += day_pnl
        daily_pnls.append(day_pnl)

        print(f"  [{i+1:>3}/{len(all_dates)}] {d}: {day_trades} trades, "
              f"PnL=${day_pnl:+,.0f}, cash=${cash + unsettled:,.0f}"
              + (f" | {', '.join(day_detail[:4])}" + ("..." if len(day_detail) > 4 else "") if day_detail else ""))

    print()
    print("=" * 72)
    print(f"  RESULTS — {args.data} (live trial 432 params)")
    print("=" * 72)
    final_equity = cash + unsettled
    ret_pct = (final_equity / starting_cash - 1) * 100
    wr = (total_wins / total_trades * 100) if total_trades else 0
    print(f"  Starting cash:  ${starting_cash:,.0f}")
    print(f"  Final equity:   ${final_equity:,.0f}  ({ret_pct:+.1f}%)")
    print(f"  Total P&L:      ${total_pnl:+,.0f}")
    print(f"  Trades:         {total_trades}  ({total_wins}W / {total_trades - total_wins}L, {wr:.1f}% WR)")
    print(f"  Trading days:   {len(all_dates)}")

    if daily_pnls:
        green = sum(1 for p in daily_pnls if p > 0)
        red = sum(1 for p in daily_pnls if p < 0)
        flat = sum(1 for p in daily_pnls if p == 0)
        print(f"  Day breakdown:  {green} green / {red} red / {flat} flat")
        arr = np.array(daily_pnls)
        if arr.std() > 0:
            sharpe = np.mean(arr) / np.std(arr) * np.sqrt(252)
            print(f"  Sharpe (daily): {sharpe:.2f}")
        print(f"  Best day:       ${arr.max():+,.0f}")
        print(f"  Worst day:      ${arr.min():+,.0f}")

    if strat_stats:
        print(f"\n  Per-strategy:")
        for strat in sorted(strat_stats.keys()):
            s = strat_stats[strat]
            swr = (s["wins"] / s["n"] * 100) if s["n"] else 0
            print(f"    {strat}: {s['n']:>3} trades, ${s['pnl']:+10,.0f}, {swr:>4.0f}% WR")

    if exit_reasons:
        print(f"\n  Exit reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
