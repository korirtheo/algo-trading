"""
Sweep VOL_CAP_PCT to see impact on results.
Tests: 0% (disabled), 3%, 5% (current), 6%, 7%, 10%, 15%
"""
import sys
import numpy as np
from test_full import load_all_picks, SLIPPAGE_PCT, STARTING_CASH, MARGIN_THRESHOLD, VOL_CAP_PCT, ET_TZ
import test_green_candle_combined as tgcc

DATA_DIRS = ["stored_data_combined"]
DATE_RANGE = ("2025-01-01", "2026-02-28")

print("Loading data...")
all_dates, daily_picks = load_all_picks(DATA_DIRS)
all_dates = [d for d in all_dates if DATE_RANGE[0] <= d <= DATE_RANGE[1]]
print(f"  {len(all_dates)} trading days\n")

print(f"{'VolCap':>6} {'Equity':>14} {'Return':>10} {'Trades':>7} {'WR':>6} {'Sharpe':>7} {'PF':>6} {'VC_Trades':>9} {'MaxDD%':>7}")
print("-" * 80)

import test_full

for vc_pct in [0, 3, 5, 6, 7, 10, 15, 25]:
    # Patch vol cap in BOTH modules (tgcc imports by value)
    old_vc_tf = test_full.VOL_CAP_PCT
    old_vc_tgcc = tgcc.VOL_CAP_PCT
    test_full.VOL_CAP_PCT = float(vc_pct)
    tgcc.VOL_CAP_PCT = float(vc_pct)

    cash = STARTING_CASH
    unsettled = 0.0
    total_trades = 0
    total_wins = 0
    daily_pnls = []
    vc_count = 0

    for d in all_dates:
        if unsettled > 0:
            cash += unsettled
            unsettled = 0.0

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        states, new_cash, new_unsettled = tgcc.simulate_day_combined(picks, cash, cash_account)

        day_pnl = 0.0
        for st in states:
            if st["exit_reason"] is not None:
                total_trades += 1
                day_pnl += st["pnl"]
                if st["pnl"] > 0:
                    total_wins += 1
                if st.get("vol_capped"):
                    vc_count += 1

        daily_pnls.append(day_pnl)
        cash = new_cash
        unsettled = new_unsettled

    final_eq = cash + unsettled
    ret = (final_eq / STARTING_CASH - 1) * 100
    wr = total_wins / max(total_trades, 1) * 100
    sharpe = 0.0
    if daily_pnls and np.std(daily_pnls) > 0:
        sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252)

    win_pnls = [p for p in daily_pnls if p > 0]
    loss_pnls = [p for p in daily_pnls if p < 0]
    pf = sum(win_pnls) / abs(sum(loss_pnls)) if loss_pnls else 999

    # Max drawdown %
    equities = []
    eq = STARTING_CASH
    uns = 0.0
    for d in all_dates:
        if uns > 0:
            eq += uns
            uns = 0.0
        equities.append(eq)  # approximate
    # Use final equity curve from daily_pnls
    eq_curve = [STARTING_CASH]
    for p in daily_pnls:
        eq_curve.append(eq_curve[-1] + p)
    peak = eq_curve[0]
    max_dd_pct = 0.0
    for e in eq_curve:
        if e > peak:
            peak = e
        dd_pct = (peak - e) / peak if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd_pct)

    label = f"{vc_pct}%" if vc_pct > 0 else "OFF"
    marker = " <-- current" if vc_pct == 5 else ""
    print(f"{label:>6} ${final_eq:>13,.0f} {ret:>+9.0f}% {total_trades:>7} {wr:>5.1f}% {sharpe:>6.2f} {pf:>5.2f} {vc_count:>9} {max_dd_pct*100:>6.1f}%{marker}")

    test_full.VOL_CAP_PCT = old_vc_tf
    tgcc.VOL_CAP_PCT = old_vc_tgcc
