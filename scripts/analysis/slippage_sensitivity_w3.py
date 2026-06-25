"""Run W3 (train 2021-23, test 2024) best config under 4 slippage scenarios.

Tests the realism of W3's $4.38M headline against modeled real-world impact:
  1. BASELINE                   USE_2MIN_SLIPPAGE=False, K=3   (= the original W3 run)
  2. 2MIN K=3                   USE_2MIN_SLIPPAGE=True,  K=3   (= realistic impact)
  3. 2MIN K=6                   USE_2MIN_SLIPPAGE=True,  K=6   (= 2x stress test)
  4. 2MIN K=9                   USE_2MIN_SLIPPAGE=True,  K=9   (= 3x stress test)

Reports per scenario:
  - Final equity, total PnL
  - Sharpe on percentage daily returns
  - Win rate, vol-capped trade count
  - % of trades that hit the 2-min cap
  - $ x-ray: position size / 2-min vol distribution
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

W3_BEST = "results/walk_forward/W3_train_2021_2022_2023_test_2024_best.json"
BASELINE = "config/trial_432_params.json"
TEST_YEAR_DIRS = ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
                   "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"]
STARTING_CASH = 25_000


def _merged_params(best):
    with open(BASELINE) as f:
        b = json.load(f)
    m = dict(b)
    m.update(best["params"])
    return m


def run_year(daily_picks, dates):
    cash = STARTING_CASH
    eq = [cash]
    daily = []
    vol_capped_count = 0
    trades_made = 0
    for d in dates:
        picks = daily_picks.get(d, [])
        if not picks:
            eq.append(cash); daily.append({"date": d, "pnl": 0, "equity": cash})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(picks, cash, cash_account=is_cash)
        except Exception:
            eq.append(cash); daily.append({"date": d, "pnl": 0, "equity": cash})
            continue
        # Count vol-capped trades for this day
        for s in states:
            if s.get("entry_price") is not None:
                trades_made += 1
                if s.get("vol_capped"):
                    vol_capped_count += 1
        pnl = end_c - cash
        cash = end_c
        if is_cash:
            cash += unset
        eq.append(cash); daily.append({"date": d, "pnl": pnl, "equity": cash})
    return cash, np.array(eq), daily, trades_made, vol_capped_count


def metrics(label, final_cash, eq, daily, trades, vol_capped):
    total_pnl = final_cash - STARTING_CASH
    eq = np.array(eq)
    prev = eq[:-1]; curr = eq[1:]
    mask = prev > 0
    ret = np.where(mask, (curr - prev) / prev * 100, 0)
    sharpe_pct = ret.mean()/ret.std()*np.sqrt(252) if ret.std() > 0 else 0
    wins = int((ret > 0).sum())
    losses = int((ret < 0).sum())
    cap_pct = 100 * vol_capped / max(1, trades)
    return {
        "label": label,
        "final": final_cash,
        "pnl": total_pnl,
        "sharpe_pct": sharpe_pct,
        "wins": wins,
        "losses": losses,
        "n_days": len(ret),
        "trades": trades,
        "vol_capped": vol_capped,
        "cap_pct": cap_pct,
    }


def main():
    print("Loading W3 best params...")
    with open(W3_BEST) as f:
        best = json.load(f)
    print(f"  W3 best: train_years={best['train_years']}  test_year={best['test_year']}")
    print(f"  score=${best['score']:,.0f}  PF={best['user_attrs'].get('pf','?')}")

    print(f"\nLoading test-year ({best['test_year']}) picks...")
    dirs = [d for d in TEST_YEAR_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    test_dates = [d for d in all_dates if d.startswith(best["test_year"])]
    print(f"  {len(test_dates)} test days: {test_dates[0]} -> {test_dates[-1]}")

    scenarios = [
        # (label, use_2min, use_multiwin, use_vol_adj, K)
        ("BASELINE (cumulative, K=3)",       False, False, False, 3.0),
        ("2MIN K=3 (single-window)",         True,  False, False, 3.0),
        ("MULTIWIN K=3 (no vol-adj)",        False, True,  False, 3.0),
        ("MULTIWIN K=3 + vol-adj",           False, True,  True,  3.0),
        ("MULTIWIN K=6 + vol-adj (2x stress)", False, True, True,  6.0),
        ("MULTIWIN K=9 + vol-adj (3x stress)", False, True, True,  9.0),
    ]

    results = []
    for label, use_2min, use_multiwin, use_voladj, k in scenarios:
        # Reload + reset params each run (set_strategy_params sets module globals)
        set_strategy_params(_merged_params(best))
        tgc.USE_DYNAMIC_SLIPPAGE = True
        tgc.VOL_CAP_PCT = 5.0
        tgc.USE_2MIN_SLIPPAGE = use_2min
        tgc.USE_MULTIWINDOW_SLIPPAGE = use_multiwin
        tgc.USE_VOLATILITY_ADJUSTMENT = use_voladj
        tgc.SLIP_IMPACT_K = k
        tgc.MAX_2MIN_PARTICIPATION = 0.15 if (use_2min or use_multiwin) else 0.0
        tgc.MAX_REGIME_PARTICIPATION = 0.08 if use_multiwin else 0.0

        print(f"\n=== {label} ===")
        final, eq, daily, trades, capped = run_year(picks, test_dates)
        m = metrics(label, final, eq, daily, trades, capped)
        results.append((m, eq, daily))
        print(f"  Final ${m['final']:,.0f}  PnL ${m['pnl']:+,.0f}  "
              f"Sharpe(%ret) {m['sharpe_pct']:.2f}  "
              f"wins {m['wins']}/{m['n_days']}  "
              f"trades {m['trades']}  vol_capped {m['vol_capped']} ({m['cap_pct']:.1f}%)")

    # Summary table
    print("\n" + "=" * 92)
    print("SLIPPAGE SENSITIVITY — W3 forward 2024")
    print("=" * 92)
    print(f"{'Scenario':<28} {'Final':>13} {'PnL':>13} {'Sharpe%':>9} {'Wins':>9} {'Capped':>10}")
    print("-" * 92)
    base = results[0][0]
    for m, _, _ in results:
        pct_of_base = 100 * m["final"] / base["final"]
        print(f"{m['label']:<28} ${m['final']:>12,.0f} ${m['pnl']:>+12,.0f} "
              f"{m['sharpe_pct']:>9.2f} {m['wins']:>4}/{m['n_days']:<4} "
              f"{m['vol_capped']:>4}/{m['trades']:<4}  ({pct_of_base:.0f}% of base)")


if __name__ == "__main__":
    main()
