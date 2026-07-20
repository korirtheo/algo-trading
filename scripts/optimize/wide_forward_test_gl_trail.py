#!/usr/bin/env python3
"""
Wide Forward Test for G+L Trail Study
Tests top 50 + 100 random trials on 2026 OOS (blind data)
Uses proper set_strategy_params -> _build_param_snapshot -> simulate_day_combined pipeline.
"""

import os
import sys
import json
import argparse
import pandas as pd
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import optuna
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params, _build_param_snapshot, run_combined_backtest
from test_full import load_all_picks, STARTING_CASH


def load_2026_oos_data():
    """Load all 2026 OOS dates (2026-03-01 onwards)."""
    data_dirs = [
        "stored_data_mar_may_2026",
        "stored_data_jun_2026",
        "stored_data_oos",
    ]
    all_dates, daily_picks = load_all_picks(data_dirs)

    cutoff = '2026-03-01'
    oos_dates = sorted([d for d in all_dates if d >= cutoff])
    oos_picks = {d: daily_picks[d] for d in oos_dates if d in daily_picks}

    print(f"  Loaded {len(oos_dates)} OOS dates: {oos_dates[0] if oos_dates else 'N/A'} → {oos_dates[-1] if oos_dates else 'N/A'}")
    return oos_dates, oos_picks


def run_trial_oos(merged_params_lower, oos_dates, oos_picks):
    """Set params, build snapshot, run backtest. Returns stats dict."""
    set_strategy_params(merged_params_lower)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    snapshot = _build_param_snapshot()
    result = run_combined_backtest(oos_picks, oos_dates, params_snapshot=snapshot)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="postgresql://postgres@127.0.0.1:5432/optuna_gl_trail")
    parser.add_argument("--study", default="gl_trail")
    args = parser.parse_args()

    print("=" * 70)
    print("WIDE FORWARD TEST: G+L Trail Study — 2026 OOS (Mar–Jun blind)")
    print("=" * 70)

    study = optuna.load_study(study_name=args.study, storage=args.db)
    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    print(f"\nCompleted trials: {len(completed)}")

    # Top 50 by training score
    top_50 = sorted(completed, key=lambda t: t.value, reverse=True)[:50]
    top_50_nums = {t.number for t in top_50}

    # Random 100
    random.seed(42)
    random_100_nums = set(random.sample([t.number for t in completed], min(100, len(completed))))

    # Combine (deduplicate)
    all_test_nums = top_50_nums | random_100_nums
    test_trials = [t for t in completed if t.number in all_test_nums]
    test_trials.sort(key=lambda t: -t.value)  # run top ones first

    print(f"Top-50 + random-100 = {len(test_trials)} unique trials to test")

    # Load base config (lowercase keys)
    with open('config/trial_w21b_511_deploy.json') as f:
        base_cfg = json.load(f)
        base_params = base_cfg['params']  # already lowercase

    # Load 2026 OOS data
    print("\nLoading 2026 OOS data...")
    oos_dates, oos_picks = load_2026_oos_data()
    if not oos_dates:
        print("ERROR: no OOS dates loaded")
        return

    print(f"\nRunning {len(test_trials)} trials on {len(oos_dates)} OOS days...\n")

    results = []
    for i, trial in enumerate(test_trials, 1):
        # Merge: base has all strategies, trial overrides only G+L keys
        merged = base_params.copy()
        merged.update(trial.params)  # trial.params are already lowercase

        try:
            r = run_trial_oos(merged, oos_dates, oos_picks)

            total_pnl = r.get('total_pnl', 0)
            pf        = r.get('pf', 0)
            wr        = r.get('wr', 0)
            n_trades  = r.get('n', 0)

            results.append({
                'trial':       trial.number,
                'train_score': trial.value,
                'oos_pnl':     total_pnl,
                'oos_pf':      pf,
                'oos_wr':      wr,
                'oos_trades':  n_trades,
                'in_top50':    trial.number in top_50_nums,
            })

            if i <= 5 or i % 10 == 0:
                tag = 'TOP' if trial.number in top_50_nums else 'RND'
                print(f"  [{i:3d}/{len(test_trials)}] #{trial.number:3d} [{tag}]  "
                      f"train=${trial.value:12,.0f}  OOS=${total_pnl:10,.0f}  "
                      f"PF={pf:.2f}  WR={wr:.1f}%  n={n_trades}")

        except Exception as e:
            print(f"  [{i:3d}/{len(test_trials)}] #{trial.number:3d}: ERROR — {e}")
            continue

    if not results:
        print("ERROR: no results collected")
        return

    df = pd.DataFrame(results).sort_values('oos_pnl', ascending=False)

    print(f"\n{'='*70}")
    print(f"TOP 15 BY 2026 OOS PnL")
    print(f"{'='*70}")
    print(f"{'Rank':>4}  {'Trial':>5}  {'Tag':>3}  {'OOS PnL':>12}  {'PF':>5}  {'WR':>6}  {'n':>4}  {'Train Score':>14}")
    print("-" * 70)
    for rank, (_, row) in enumerate(df.head(15).iterrows(), 1):
        tag = 'TOP' if row['in_top50'] else 'RND'
        print(f"  {rank:2d}   #{int(row['trial']):3d}  [{tag}]  "
              f"${row['oos_pnl']:11,.0f}  {row['oos_pf']:5.2f}  "
              f"{row['oos_wr']:5.1f}%  {int(row['oos_trades']):4d}  "
              f"${row['train_score']:13,.0f}")

    # Correlation: train rank vs OOS rank
    df['train_rank'] = df['train_score'].rank(ascending=False)
    df['oos_rank']   = df['oos_pnl'].rank(ascending=False)
    corr = df[['train_rank', 'oos_rank']].corr().iloc[0, 1]
    print(f"\nTrain→OOS rank correlation: {corr:.3f}")

    pct_positive = 100 * (df['oos_pnl'] > 0).sum() / len(df)
    print(f"% trials with positive OOS PnL: {pct_positive:.1f}%")

    # Top-50 vs random hit rate
    top50_df = df[df['in_top50']]
    rnd_df   = df[~df['in_top50']]
    print(f"\nTop-50 median OOS PnL:  ${top50_df['oos_pnl'].median():,.0f}")
    print(f"Random-100 median OOS PnL: ${rnd_df['oos_pnl'].median():,.0f}")

    df.to_csv('config/trial_gl_trail_v2_oos_2026_results.csv', index=False)
    df.to_json('config/trial_gl_trail_v2_oos_2026_results.json', orient='records', indent=2)
    print(f"\nSaved → config/trial_gl_trail_v2_oos_2026_results.csv / .json")


if __name__ == '__main__':
    main()
