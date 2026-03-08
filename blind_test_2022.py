"""
Blind backtest on 2022 out-of-sample data using Trial #432 params from v8 optimizer.
This data was NEVER seen during optimization (trained on 2024-2026).

Usage:
  python blind_test_2022.py                    # full run
  python blind_test_2022.py --trial 411        # use different trial
  python blind_test_2022.py --db optuna_combined_v8_top10_521trials.db
"""
import sys
import os
import json
import argparse
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import the backtest infrastructure
from test_full import load_all_picks, MARGIN_THRESHOLD
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params

STARTING_CASH = 25_000
DATA_DIRS = ["stored_data_2022"]
STRAT_KEYS = ["H","G","A","F","D","V","P","M","R","W","O","B","K","C","S","E","I","J","N","L"]


def load_trial_params(db_path, trial_number=None):
    """Load params from an Optuna trial."""
    storage = f"sqlite:///{db_path}"
    names = optuna.study.get_all_study_names(storage=storage)
    study = optuna.load_study(study_name=names[0], storage=storage)

    if trial_number is not None:
        trial = [t for t in study.trials if t.number == trial_number][0]
    else:
        trial = study.best_trial

    print(f"  Study: {names[0]}")
    print(f"  Trial: #{trial.number} (score: {trial.value:,.0f})")

    # Get enabled strategies
    enabled = [s.upper() for s in "hgafdvpmlrwobcseijn" if trial.params.get(f"enable_{s}", False)]
    print(f"  Enabled: {', '.join(enabled)} ({len(enabled)} strategies)")

    # Get priority
    priority = {s.upper(): trial.params.get(f"priority_{s}", 99) for s in "hgafdvpmlrwobcseijn"}
    priority_sorted = sorted(priority.items(), key=lambda x: x[1])
    priority_str = " > ".join(f"{k}" for k, v in priority_sorted if k in enabled)
    print(f"  Priority: {priority_str}")

    # Print key metrics from user_attrs
    for k in ['total_pnl', 'n', 'wr', 'pf']:
        if k in trial.user_attrs:
            v = trial.user_attrs[k]
            if isinstance(v, float):
                print(f"  {k}: {v:,.2f}")
            else:
                print(f"  {k}: {v}")

    return trial.params


def main():
    parser = argparse.ArgumentParser(description="Blind test on 2022 OOS data")
    parser.add_argument("--db", default="optuna_combined_v8.db", help="Optuna DB path")
    parser.add_argument("--trial", type=int, default=432, help="Trial number (default: 432)")
    args = parser.parse_args()

    print("=" * 70)
    print("  BLIND BACKTEST: 2022 Out-of-Sample Data")
    print("  Params optimized on: 2024-01-01 to 2026-02-28")
    print("  Testing on: 2022-01-01 to 2022-12-31 (NEVER SEEN)")
    print("=" * 70)

    print(f"\nLoading trial #{args.trial} from {args.db}...")
    params = load_trial_params(args.db, args.trial)

    print(f"\nApplying params to strategy engine...")
    set_strategy_params(params)

    print(f"\nLoading 2022 data from {DATA_DIRS}...")
    all_dates, daily_picks = load_all_picks(DATA_DIRS)
    print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")

    # Pre-scan for R (Multi-Day Runner) candidates
    r_day2_picks = {}
    for idx in range(len(all_dates) - 1):
        d1, d2 = all_dates[idx], all_dates[idx + 1]
        for pick in daily_picks.get(d1, []):
            if pick["gap_pct"] < tgc.R_DAY1_MIN_GAP:
                continue
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) < 10:
                continue
            day1_close = float(mh["Close"].values[-1])
            d2_data = tgc._load_r_intraday(pick["ticker"], d2, DATA_DIRS)
            if d2_data is not None and len(d2_data) >= 20:
                r_pick = {
                    "ticker": pick["ticker"],
                    "gap_pct": pick["gap_pct"],
                    "premarket_high": 0,
                    "pm_volume": 0,
                    "market_hour_candles": d2_data,
                    "is_r_candidate": True,
                    "r_day1_close": day1_close,
                }
                r_day2_picks.setdefault(d2, []).append(r_pick)
    r_total = sum(len(v) for v in r_day2_picks.values())
    print(f"  R candidates: {r_total} across {len(r_day2_picks)} days")

    # Run backtest
    cash = float(STARTING_CASH)
    unsettled = 0.0
    all_results = []

    print(f"\n  Starting Cash: ${cash:,.0f}")
    print()
    print(f"{'Date':<12} {'Strat':>5} {'Trades':>6} {'Win':>4} {'Loss':>5} "
          f"{'Day P&L':>12} {'Balance':>14}")
    print("-" * 72)

    for d in all_dates:
        cash += unsettled
        unsettled = 0.0

        picks = daily_picks.get(d, [])
        r_picks = r_day2_picks.get(d, [])
        all_picks = picks + r_picks
        cash_account = cash < MARGIN_THRESHOLD

        states, cash, unsettled, day_selection = tgc.simulate_day_combined(
            all_picks, cash, cash_account
        )

        day_pnl = 0.0
        day_trades = 0
        day_wins = 0
        day_losses = 0
        counts = {k: [0, 0] for k in STRAT_KEYS}

        for st in states:
            if st["exit_reason"] is not None:
                day_trades += 1
                day_pnl += st["pnl"]
                if st["pnl"] > 0:
                    day_wins += 1
                else:
                    day_losses += 1
                s = st["strategy"]
                if s in counts:
                    counts[s][0] += 1
                    if st["pnl"] > 0:
                        counts[s][1] += 1

        equity = cash + unsettled

        parts = []
        for key in STRAT_KEYS:
            if counts[key][0] > 0:
                parts.append(f"{key}{counts[key][0]}")
        strat_label = "".join(parts) if parts else ""

        print(f"{d:<12} {strat_label:>5} {day_trades:>6} {day_wins:>4} {day_losses:>5} "
              f"${day_pnl:>+11,.0f} ${equity:>13,.0f}")

        # Per-trade detail
        traded_states = [s for s in states if s["exit_reason"] is not None]
        for st in traded_states:
            pct = (st["pnl"] / st["position_cost"] * 100) if st["position_cost"] > 0 else 0
            reason_short = {"TARGET": "T", "TIME_STOP": "TS", "EOD_CLOSE": "EOD",
                            "STOP": "SL", "TRAIL": "TR"}.get(
                st["exit_reason"], st["exit_reason"][:3])
            vc_tag = " VC" if st.get("vol_capped") else ""
            print(f"  -> [{st['strategy']}] {st['ticker']:<6} {reason_short:<3}  "
                  f"${st['pnl']:>+10,.0f}  ({pct:>+6.2f}%){vc_tag}")

        all_results.append({
            "date": d, "picks": picks, "states": states,
            "day_pnl": day_pnl, "equity": equity,
            "trades": day_trades, "wins": day_wins, "losses": day_losses,
            **{f"{k.lower()}_trades": counts[k][0] for k in STRAT_KEYS},
            **{f"{k.lower()}_wins": counts[k][1] for k in STRAT_KEYS},
        })

    # --- Summary ---
    final_equity = cash + unsettled
    total_trades = sum(r["trades"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    strat_totals = {k: sum(r[f"{k.lower()}_trades"] for r in all_results) for k in STRAT_KEYS}
    strat_wins = {k: sum(r[f"{k.lower()}_wins"] for r in all_results) for k in STRAT_KEYS}
    daily_pnls = [r["day_pnl"] for r in all_results if r["trades"] > 0]
    green = sum(1 for p in daily_pnls if p > 0)
    red = sum(1 for p in daily_pnls if p <= 0)
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if daily_pnls and np.std(daily_pnls) > 0 else 0

    all_trade_pnls = []
    strat_pnls = {k: [] for k in STRAT_KEYS}
    for r in all_results:
        for st in r["states"]:
            if st["exit_reason"] is not None:
                all_trade_pnls.append(st["pnl"])
                s = st["strategy"]
                if s in strat_pnls:
                    strat_pnls[s].append(st["pnl"])

    avg_win = np.mean([p for p in all_trade_pnls if p > 0]) if total_wins > 0 else 0
    avg_loss = np.mean([p for p in all_trade_pnls if p <= 0]) if total_losses > 0 else 0
    pf = abs(sum(p for p in all_trade_pnls if p > 0) / sum(p for p in all_trade_pnls if p < 0)) if any(p < 0 for p in all_trade_pnls) else float('inf')

    # Max drawdown
    peak = STARTING_CASH
    max_dd = 0
    for r in all_results:
        peak = max(peak, r["equity"])
        dd = (peak - r["equity"]) / peak * 100
        max_dd = max(max_dd, dd)

    print(f"\n{'='*70}")
    print(f"  BLIND BACKTEST RESULTS: 2022 OUT-OF-SAMPLE")
    print(f"{'='*70}")
    print(f"  Starting Cash:    ${STARTING_CASH:,}")
    print(f"  Ending Equity:    ${final_equity:,.0f}  ({(final_equity/STARTING_CASH - 1)*100:+.1f}%)")
    print(f"  Total PnL:        ${final_equity - STARTING_CASH:+,.0f}")
    print(f"  Trading Days:     {len(all_dates)}")
    print(f"  Total Trades:     {total_trades}")
    print(f"    Winners:        {total_wins} ({total_wins/max(total_trades,1)*100:.1f}%)")
    print(f"    Losers:         {total_losses}")
    print(f"  Avg Win:          ${avg_win:+,.0f}")
    print(f"  Avg Loss:         ${avg_loss:+,.0f}")
    print(f"  Profit Factor:    {pf:.2f}")
    print(f"  Sharpe Ratio:     {sharpe:.2f}")
    print(f"  Max Drawdown:     {max_dd:.1f}%")
    print(f"  Green/Red Days:   {green}/{red}")

    print(f"\n  Per-Strategy Breakdown:")
    print(f"  {'Strat':<6} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PnL':>14} {'Avg':>10}")
    print(f"  {'-'*52}")
    for k in STRAT_KEYS:
        n = strat_totals[k]
        if n == 0:
            continue
        w = strat_wins[k]
        wr = w / n * 100
        pnl = sum(strat_pnls[k])
        avg = pnl / n
        print(f"  {k:<6} {n:>7} {w:>5} {wr:>5.1f}% ${pnl:>+13,.0f} ${avg:>+9,.0f}")

    print(f"{'='*70}")

    # Compare with in-sample
    print(f"\n  COMPARISON:")
    print(f"  In-sample (2024-2026):  $123.8M PnL, 1843 trades, 74.7% WR, PF 3.27")
    print(f"  Out-of-sample (2022):   ${final_equity - STARTING_CASH:+,.0f} PnL, {total_trades} trades, {total_wins/max(total_trades,1)*100:.1f}% WR, PF {pf:.2f}")
    print()


if __name__ == "__main__":
    main()
