"""Standalone validation of the Range Reversion strategy (X).

Runs the simulator with X enabled and all other strategies DISABLED on 2024
data. Compares against:
  - The empirical scan expectation (~16% median upside per trade)
  - The first-leg-only baseline (existing 20 strategies, X disabled)

Reports:
  - X trade count
  - X win rate
  - X median trade PnL %
  - X total PnL
  - First-leg-only baseline for comparison

Uses default X constants (no Optuna tuning yet). This is a sanity check that
the empirical pattern translates to actual simulator trades.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

YEAR_DIRS = ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
             "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"]
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000


def _merged_params(extras=None):
    with open(BASELINE) as f:
        b = json.load(f)
    m = dict(b)
    if extras:
        m.update(extras)
    return m


def _enable_only(strats):
    """Apply baseline with all strategies disabled except `strats`."""
    extras = {}
    for s in tgc.STRAT_KEYS:
        extras[f"enable_{s.lower()}"] = (s in strats)
    return _merged_params(extras)


def run(dates, daily_picks, label, params):
    set_strategy_params(params)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True   # honest slippage
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08

    cash = STARTING_CASH
    daily_pnls = []
    all_x_trades = []
    all_trades = []
    for d in dates:
        picks = daily_picks.get(d, [])
        if not picks:
            daily_pnls.append(0); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_cash, unset, _ = tgc.simulate_day_combined(
                picks, cash, cash_account=is_cash)
        except Exception as e:
            print(f"  ERR {d}: {e}")
            daily_pnls.append(0); continue
        for s in states:
            if s.get("exit_reason") is not None and s.get("position_cost", 0) > 0:
                tr = {
                    "date": d, "ticker": s["ticker"],
                    "strategy": s["strategy"], "pnl": s["pnl"],
                    "exit": s.get("exit_price"),
                    "reason": s["exit_reason"],
                    "pct": (s["pnl"] / s["position_cost"] * 100) if s["position_cost"] > 0 else 0,
                }
                all_trades.append(tr)
                if s["strategy"] == "X":
                    all_x_trades.append(tr)
        daily_pnls.append(end_cash - cash)
        cash = end_cash
        if is_cash: cash += unset

    final_pnl = cash - STARTING_CASH
    print(f"\n=== {label} ===")
    print(f"  Final equity: ${cash:,.0f}  (PnL ${final_pnl:+,.0f})")
    print(f"  Total trades: {len(all_trades)}")
    if all_x_trades:
        pcts = np.array([t["pct"] for t in all_x_trades])
        pnls = np.array([t["pnl"] for t in all_x_trades])
        wins = (pcts > 0).sum()
        losses = (pcts < 0).sum()
        reasons = {}
        for t in all_x_trades:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        print(f"\n  X-trades only: {len(all_x_trades)}")
        print(f"    Wins/Losses: {wins}/{losses}  ({100*wins/len(all_x_trades):.1f}% WR)")
        print(f"    Median PnL %:  {np.median(pcts):+.2f}%")
        print(f"    Mean PnL %:    {pcts.mean():+.2f}%")
        print(f"    Total X PnL:   ${pnls.sum():+,.0f}")
        print(f"    Exit reasons:  {reasons}")
        # Show a few sample trades
        print(f"\n  Sample X trades:")
        for t in all_x_trades[:8]:
            exit_p = t.get('exit') or 0
            print(f"    {t['date']}  {t['ticker']:<6}  exit=${exit_p:.2f}  "
                  f"pct={t['pct']:+.1f}%  ({t['reason']})")
    return final_pnl, all_x_trades, all_trades


def main():
    print("Loading 2024 picks...")
    dirs = [d for d in YEAR_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    print(f"  {len(all_dates)} days, {all_dates[0]} -> {all_dates[-1]}")

    # Scenario 1: X-ONLY (all other strategies disabled)
    p_x = _enable_only({"X"})
    final_x, x_trades, _ = run(all_dates, picks, "X-ONLY (proof Range works)", p_x)

    # Scenario 2: ALL existing 20 strategies, X disabled (baseline)
    p_legacy = _enable_only(set(tgc.STRAT_KEYS) - {"X"})
    final_legacy, _, all_legacy = run(all_dates, picks, "LEGACY (all 20, no X)", p_legacy)

    # Scenario 3: ALL 21 strategies INCLUDING X
    p_combined = _enable_only(set(tgc.STRAT_KEYS))
    final_combined, x_in_combo, all_combined = run(all_dates, picks, "COMBINED (all 21, X enabled)", p_combined)

    print("\n" + "=" * 80)
    print("SUMMARY — 2024 Range strategy validation")
    print("=" * 80)
    print(f"  X-ONLY final equity:    ${STARTING_CASH + final_x:,.0f}  (PnL ${final_x:+,.0f}, {len(x_trades)} X trades)")
    print(f"  LEGACY final equity:    ${STARTING_CASH + final_legacy:,.0f}  (PnL ${final_legacy:+,.0f}, {len(all_legacy)} total trades)")
    print(f"  COMBINED final equity:  ${STARTING_CASH + final_combined:,.0f}  (PnL ${final_combined:+,.0f}, {len(all_combined)} total, {len(x_in_combo)} X)")
    delta = final_combined - final_legacy
    print(f"\n  Combined - Legacy = ${delta:+,.0f}  (X strategy net contribution)")
    if delta > 0:
        print(f"  Range strategy adds value: +${delta:,.0f} on 2024 ({100*delta/max(1,final_legacy):.1f}% over legacy)")
    else:
        print(f"  Range strategy NET NEGATIVE: ${delta:+,.0f}. X trades lose more than they add.")


if __name__ == "__main__":
    main()
