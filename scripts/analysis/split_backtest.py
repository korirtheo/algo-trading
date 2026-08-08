"""
Split-Param Backtest & Forward Test
====================================
Runs backtests with G1/G2/L1/L2 split-param breakdown.
Forward tests top N trials + random N trials on OOS data.

Usage:
  python scripts/analysis/split_backtest.py --top 100 --random 100
  python scripts/analysis/split_backtest.py --top 1 --random 0  # quick test
  python scripts/analysis/split_backtest.py --top 10 --random 10 --oos-start 2026-01-01 --oos-end 2026-02-28
"""

import os
import sys
import argparse
import random as _random

import numpy as np
import optuna

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import test_green_candle_combined as tgc
from optimize_combined import (
    DATA_DIRS, _param_lock, set_strategy_params,
    _build_param_snapshot,
)
from optimize_gl_split import _map_split_to_standard, _build_split_snapshot
from test_full import load_all_picks, STARTING_CASH, MARGIN_THRESHOLD

# ── Defaults ────────────────────────────────────────────────────────────
DEFAULT_DB = "postgresql://postgres@127.0.0.1:5432/optuna_gl_split"
DEFAULT_STUDY = "gl_split_v1"


# ═══════════════════════════════════════════════════════════════════════
#  Backtest that captures entry_trade_seq for G1/G2/L1/L2 breakdown
# ═══════════════════════════════════════════════════════════════════════
def run_split_backtest(daily_picks, dates, params_snapshot=None):
    """Run backtest and capture full trade data including entry_trade_seq.

    Same logic as run_combined_backtest but records per-trade details
    needed for G1/G2/L1/L2 split analysis.
    """
    cash = float(STARTING_CASH)
    unsettled = 0.0
    all_trades = []
    daily_equity = [float(STARTING_CASH)]

    for d in dates:
        cash += unsettled
        unsettled = 0.0

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as _news_filter
            picks = _news_filter(
                picks, d,
                min_articles=tgc.NEWS_MIN_ARTICLES,
                require_catalyst=tgc.NEWS_REQUIRE_CATALYST,
            )

        if tgc.MIN_FAVORABILITY_THRESHOLD > 0 and tgc._MACRO_DATA:
            _score, _ = tgc.compute_day_favorability(d, picks)
            if _score is not None and _score < tgc.MIN_FAVORABILITY_THRESHOLD:
                daily_equity.append(cash + unsettled)
                continue

        states, cash, unsettled, _ = tgc.simulate_day_combined(
            picks, cash, cash_account, params=params_snapshot
        )

        for st in states:
            if st["exit_reason"] is not None:
                all_trades.append({
                    "strategy": st.get("strategy", "?"),
                    "pnl": st["pnl"],
                    "position_cost": st["position_cost"],
                    "entry_trade_seq": st.get("entry_trade_seq", 0),
                    "ticker": st.get("ticker", "?"),
                    "entry_price": st.get("entry_price"),
                    "exit_price": st.get("exit_price"),
                    "entry_time": st.get("entry_time"),
                    "exit_time": st.get("exit_time"),
                    "exit_reason": st.get("exit_reason"),
                })

        daily_equity.append(cash + unsettled)

    equity = cash + unsettled
    n = len(all_trades)

    # Max drawdown
    peak = STARTING_CASH
    max_dd_pct = 0.0
    for eq in daily_equity:
        if eq > peak:
            peak = eq
        dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    if n == 0:
        return {"n": 0, "pf": 0, "total_pnl": -9999, "equity": equity,
                "wr": 0, "max_dd_pct": 0, "trades": [], "splits": {}}

    total_pnl = sum(t["pnl"] for t in all_trades)
    wins = [t["pnl"] for t in all_trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in all_trades if t["pnl"] <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 99
    wr = len(wins) / n * 100

    # Per-split breakdown (G1/G2/L1/L2)
    splits = {}
    for t in all_trades:
        s = t["strategy"]
        seq = t["entry_trade_seq"]
        if s == "G":
            split = "G1" if seq == 0 else "G2"
        elif s == "L":
            split = "L1" if seq == 0 else "L2"
        else:
            split = s

        if split not in splits:
            splits[split] = {"trades": 0, "wins": 0, "pnl": 0.0, "gross_win": 0.0, "gross_loss": 0.0, "hold_mins": []}
        sp = splits[split]
        sp["trades"] += 1
        sp["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            sp["wins"] += 1
            sp["gross_win"] += t["pnl"]
        else:
            sp["gross_loss"] += abs(t["pnl"])

        # Hold time in minutes
        if t["entry_time"] and t["exit_time"]:
            try:
                et = t["entry_time"]
                xt = t["exit_time"]
                if hasattr(et, "timestamp") and hasattr(xt, "timestamp"):
                    hold = (xt - et).total_seconds() / 60.0
                    sp["hold_mins"].append(hold)
                elif isinstance(et, str) and isinstance(xt, str):
                    from datetime import datetime
                    fmt = "%Y-%m-%d %H:%M:%S"
                    et_dt = datetime.strptime(str(et)[:19], fmt)
                    xt_dt = datetime.strptime(str(xt)[:19], fmt)
                    hold = (xt_dt - et_dt).total_seconds() / 60.0
                    sp["hold_mins"].append(hold)
            except Exception:
                pass

    # Compute per-split PF and WR
    for split, sp in splits.items():
        sp["wr"] = sp["wins"] / sp["trades"] * 100 if sp["trades"] > 0 else 0
        sp["pf"] = sp["gross_win"] / sp["gross_loss"] if sp["gross_loss"] > 0 else 99
        sp["avg_pnl"] = sp["pnl"] / sp["trades"] if sp["trades"] > 0 else 0
        sp["avg_hold"] = float(np.mean(sp["hold_mins"])) if sp["hold_mins"] else 0

    return {
        "n": n, "pf": pf, "total_pnl": total_pnl, "equity": equity,
        "wr": wr, "max_dd_pct": max_dd_pct,
        "trades": all_trades, "splits": splits,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Build param snapshot for a trial
# ═══════════════════════════════════════════════════════════════════════
def _build_snapshot_for_trial(params):
    """Map split params to standard + build snapshot with split keys.

    Critical: injects enable_<s> flags because Optuna doesn't store
    manually-set params (only trial.suggest_* values are persisted).
    Without this, all strategies default to enabled.
    """
    from optimize_combined import ALL_STRATS
    std_params = _map_split_to_standard(params)
    # Inject enable flags — only G+L active, all others disabled
    for s in ALL_STRATS:
        std_params[f"enable_{s}"] = (s in ("g", "l"))
    with _param_lock:
        set_strategy_params(std_params)
        tgc.USE_DYNAMIC_SLIPPAGE = True
        tgc.USE_MULTIWINDOW_SLIPPAGE = True
        snapshot = _build_split_snapshot(std_params)
    return snapshot


# ═══════════════════════════════════════════════════════════════════════
#  Print helpers
# ═══════════════════════════════════════════════════════════════════════
def _print_split_table(label, results_list, split_names):
    """Print aggregated per-split stats across multiple trial results."""
    # Aggregate across all trials
    agg = {}
    for split in split_names:
        agg[split] = {"trades": 0, "wins": 0, "pnl": 0.0, "gross_win": 0.0, "gross_loss": 0.0, "hold_sum": 0.0, "hold_n": 0}

    for r in results_list:
        for split in split_names:
            sp = r.get("splits", {}).get(split, {})
            if sp:
                agg[split]["trades"] += sp.get("trades", 0)
                agg[split]["wins"] += sp.get("wins", 0)
                agg[split]["pnl"] += sp.get("pnl", 0)
                agg[split]["gross_win"] += sp.get("gross_win", 0)
                agg[split]["gross_loss"] += sp.get("gross_loss", 0)
                # Weighted hold time: avg_hold * trades for this result
                hold_n = sp.get("trades", 0)
                avg_h = sp.get("avg_hold", 0)
                agg[split]["hold_sum"] += avg_h * hold_n
                agg[split]["hold_n"] += hold_n

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  {'Split':<6} {'Trades':>7} {'WR%':>7} {'PnL':>12} {'Avg PnL':>10} {'Avg Hold':>10} {'PF':>7}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*12} {'-'*10} {'-'*10} {'-'*7}")

    for split in split_names:
        a = agg[split]
        wr = a["wins"] / a["trades"] * 100 if a["trades"] > 0 else 0
        pf = a["gross_win"] / a["gross_loss"] if a["gross_loss"] > 0 else 99
        avg = a["pnl"] / a["trades"] if a["trades"] > 0 else 0
        avg_hold = a["hold_sum"] / a["hold_n"] if a["hold_n"] > 0 else 0
        print(f"  {split:<6} {a['trades']:>7} {wr:>6.1f}% ${a['pnl']:>10,.0f} ${avg:>8,.0f} {avg_hold:>8.0f}m {pf:>6.2f}")


def _print_overall_table(label, results_list):
    """Print overall stats across multiple trial results."""
    n_trials = len(results_list)
    if n_trials == 0:
        return

    total_trades = sum(r["n"] for r in results_list)
    total_pnl = sum(r["total_pnl"] for r in results_list)
    avg_pnl = total_pnl / n_trials
    avg_wr = np.mean([r["wr"] for r in results_list if r["n"] > 0])
    avg_dd = np.mean([r["max_dd_pct"] for r in results_list if r["n"] > 0])
    avg_pf = np.mean([r["pf"] for r in results_list if r["n"] > 0])
    median_pnl = np.median([r["total_pnl"] for r in results_list if r["n"] > 0])
    profitable = sum(1 for r in results_list if r["total_pnl"] > 0 and r["n"] > 0)

    print(f"\n  --- {label} Summary ({n_trials} trials) ---")
    print(f"  Profitable:   {profitable}/{n_trials} ({profitable/n_trials*100:.0f}%)")
    print(f"  Avg PnL:      ${avg_pnl:,.0f}  |  Median: ${median_pnl:,.0f}")
    print(f"  Avg WR:       {avg_wr:.1f}%  |  Avg PF: {avg_pf:.2f}")
    print(f"  Avg Max DD:   {avg_dd:.1f}%")
    print(f"  Total Trades: {total_trades:,}")


# ═══════════════════════════════════════════════════════════════════════
#  Worker function for parallel execution (must be top-level for pickling)
# ═══════════════════════════════════════════════════════════════════════
_worker_cache = {}  # Process-level cache: persists across trials in same worker

def _worker_run_trial(args_tuple):
    """Run IS+OOS backtest for a single trial. Designed for ProcessPoolExecutor."""
    trial_number, trial_params, trial_score, is_dates, oos_dates = args_tuple
    try:
        snapshot = _build_snapshot_for_trial(trial_params)
        # Cache data per worker process (loads once, reuses for all trials)
        if "picks" not in _worker_cache:
            _, dp = load_all_picks(DATA_DIRS)
            _worker_cache["picks"] = dp
        dp = _worker_cache["picks"]
        is_result = run_split_backtest(dp, is_dates, params_snapshot=snapshot) if is_dates else None
        oos_result = run_split_backtest(dp, oos_dates, params_snapshot=snapshot) if oos_dates else None
        return {"number": trial_number, "is": is_result, "oos": oos_result, "score": trial_score}
    except Exception as e:
        return {"number": trial_number, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Split-Param Backtest & Forward Test")
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--top", type=int, default=100, help="Top N trials by score")
    parser.add_argument("--random", type=int, default=100, help="Random N trials")
    parser.add_argument("--workers", type=int, default=6, help="Parallel worker processes (default: 6)")
    parser.add_argument("--oos-only", action="store_true", help="Skip IS backtest, only run OOS")
    parser.add_argument("--is-start", default="2024-01-01", help="In-sample start")
    parser.add_argument("--is-end", default="2025-12-31", help="In-sample end")
    parser.add_argument("--oos-start", default="2026-01-01", help="Out-of-sample start")
    parser.add_argument("--oos-end", default="2026-02-28", help="Out-of-sample end")
    args = parser.parse_args()

    # Configure slippage
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True

    # Load study
    print(f"Loading study '{args.study}' from {args.db}...")
    storage = optuna.storages.RDBStorage(url=args.db)
    study = optuna.load_study(study_name=args.study, storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  {len(completed)} completed trials")

    # Select trials
    top_trials = sorted(completed, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:args.top]
    random_trials = _random.sample(completed, min(args.random, len(completed)))

    # Load data (for single-worker mode; parallel workers load their own)
    if args.workers <= 1:
        print("Loading data...")
        all_dates, daily_picks = load_all_picks(DATA_DIRS)
    else:
        print(f"Loading data (for date ranges only, workers load picks independently)...")
        all_dates, _ = load_all_picks(DATA_DIRS)

    is_dates = [] if args.oos_only else sorted([d for d in all_dates if args.is_start <= d <= args.is_end])
    oos_dates = sorted([d for d in all_dates if args.oos_start <= d <= args.oos_end])
    if not args.oos_only:
        print(f"  IS:  {args.is_start} to {args.is_end} ({len(is_dates)} days)")
    print(f"  OOS: {args.oos_start} to {args.oos_end} ({len(oos_dates)} days)")

    # Run backtests
    split_names = ["G1", "G2", "L1", "L2"]
    all_results = {}  # trial_number -> {"is": result, "oos": result}

    def _run_batch(trials, label):
        nonlocal all_results
        results = []

        if args.workers <= 1:
            # Sequential mode
            for i, t in enumerate(trials):
                print(f"\r  {label} {i+1}/{len(trials)} (trial #{t.number})...", end="", flush=True)
                try:
                    snapshot = _build_snapshot_for_trial(t.params)
                    is_result = run_split_backtest(daily_picks, is_dates, params_snapshot=snapshot) if is_dates else None
                    oos_result = run_split_backtest(daily_picks, oos_dates, params_snapshot=snapshot) if oos_dates else None
                    all_results[t.number] = {"is": is_result, "oos": oos_result, "score": t.value}
                    results.append(oos_result if oos_result else is_result)
                except Exception as e:
                    print(f"\n  ERROR trial #{t.number}: {e}")
            print()
        else:
            # Parallel mode
            from concurrent.futures import ProcessPoolExecutor, as_completed
            tasks = [
                (t.number, t.params, t.value, is_dates, oos_dates)
                for t in trials
            ]
            print(f"  Running {len(tasks)} trials with {args.workers} workers...")
            done = 0
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_worker_run_trial, task): task for task in tasks}
                for future in as_completed(futures):
                    done += 1
                    r = future.result()
                    if "error" in r:
                        print(f"\n  ERROR trial #{r['number']}: {r['error']}")
                    else:
                        all_results[r["number"]] = {"is": r["is"], "oos": r["oos"], "score": r["score"]}
                        results.append(r["oos"] if r["oos"] else r["is"])
                    print(f"\r  {label} {done}/{len(tasks)} done...", end="", flush=True)
            print()

        return results

    print(f"\nRunning {args.top} top trials...")
    top_oos = _run_batch(top_trials, "Top")

    print(f"\nRunning {args.random} random trials...")
    random_oos = _run_batch(random_trials, "Random")

    # ── Print results ──
    if is_dates:
        print("\n" + "=" * 80)
        print("  IN-SAMPLE RESULTS")
        print("=" * 80)

        top_is = [all_results[t.number]["is"] for t in top_trials if t.number in all_results and all_results[t.number]["is"]]
        random_is = [all_results[t.number]["is"] for t in random_trials if t.number in all_results and all_results[t.number]["is"]]

        _print_split_table("Top Trials — IS Per-Split", top_is, split_names)
        _print_overall_table("Top Trials — IS", top_is)

        _print_split_table("Random Trials — IS Per-Split", random_is, split_names)
        _print_overall_table("Random Trials — IS", random_is)

    if oos_dates:
        print("\n" + "=" * 80)
        print("  OUT-OF-SAMPLE RESULTS")
        print("=" * 80)

        top_oos_valid = [all_results[t.number]["oos"] for t in top_trials if t.number in all_results and all_results[t.number]["oos"]]
        random_oos_valid = [all_results[t.number]["oos"] for t in random_trials if t.number in all_results and all_results[t.number]["oos"]]

        _print_split_table("Top Trials — OOS Per-Split", top_oos_valid, split_names)
        _print_overall_table("Top Trials — OOS", top_oos_valid)

        _print_split_table("Random Trials — OOS Per-Split", random_oos_valid, split_names)
        _print_overall_table("Random Trials — OOS", random_oos_valid)

        # ── Top 10 individual trials detail ──
        print("\n" + "=" * 80)
        # ── Top 20 by OOS PnL (combined top + random, with source label) ──
        top_numbers = {t.number for t in top_trials}
        all_oos = []
        for t in list(top_trials) + list(random_trials):
            if t.number in all_results and all_results[t.number]["oos"]:
                source = "TOP" if t.number in top_numbers else "RND"
                all_oos.append((t, all_results[t.number], source))
        # Deduplicate (a trial could appear in both sets)
        seen = set()
        deduped = []
        for t, r, src in all_oos:
            if t.number not in seen:
                seen.add(t.number)
                deduped.append((t, r, src))
        deduped.sort(key=lambda x: x[1]["oos"]["total_pnl"], reverse=True)

        print("\n" + "=" * 80)
        print("  TOP 20 TRIALS BY OOS PnL (TOP = top-scored, RND = random)")
        print("=" * 80)

        for rank, (t, r, src) in enumerate(deduped[:20], 1):
            oos_r = r["oos"]
            print(f"\n  #{rank:<3} Trial #{t.number:<5} [{src}]  Score: {t.value:,.0f}  |  OOS PnL: ${oos_r['total_pnl']:,.0f}  |  WR: {oos_r['wr']:.1f}%  |  PF: {oos_r['pf']:.2f}  |  DD: {oos_r['max_dd_pct']:.1f}%  |  Trades: {oos_r['n']}")
            print(f"  {'Split':<6} {'Trades':>7} {'WR%':>7} {'PnL':>12} {'Avg PnL':>10} {'Avg Hold':>9} {'PF':>7}")
            print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*12} {'-'*10} {'-'*9} {'-'*7}")
            for split in ["G1", "G2", "L1", "L2"]:
                sp = oos_r["splits"].get(split, {})
                st = sp.get("trades", 0)
                sw = f"{sp.get('wr', 0):.1f}%" if st > 0 else "  0.0%"
                sp_pnl = sp.get("pnl", 0)
                sp_avg = sp.get("avg_pnl", 0) if st > 0 else 0
                sp_hold = f"{sp.get('avg_hold', 0):.0f}m" if st > 0 else "  0m"
                sp_pf = f"{sp.get('pf', 0):.2f}" if st > 0 else "  0.00"
                print(f"  {split:<6} {st:>7} {sw:>7} ${sp_pnl:>10,.0f} ${sp_avg:>8,.0f} {sp_hold:>9} {sp_pf:>7}")

    print("\nDone.")


if __name__ == "__main__":
    main()
