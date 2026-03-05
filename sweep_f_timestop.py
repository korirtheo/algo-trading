"""
Sweep F_TIME_LIMIT_MINUTES, A_TIME_LIMIT_MINUTES, G_TIME_LIMIT_MINUTES
across a grid and run the combined G+A+F backtest for each combination.

No SPY regime filter - trades every day, same as test_green_candle_combined.py.
"""

import io
import sys
import time
import itertools

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from test_full import load_all_picks, STARTING_CASH, MARGIN_THRESHOLD
import test_green_candle_combined as tgcc
from test_green_candle_combined import simulate_day_combined

# ---- Grid values ----
F_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
A_VALUES = [6, 8, 10, 12]
G_VALUES = [14, 16, 18, 20, 24]

# ---- Load data once ----
print("Loading data (stored_data_oos + stored_data) ...")
t0 = time.time()
all_dates, daily_picks = load_all_picks(["stored_data_oos", "stored_data"])
all_dates = [d for d in all_dates if "2025-10-01" <= d <= "2026-02-28"]
print(f"  {len(all_dates)} trading days: {all_dates[0]} to {all_dates[-1]}")
print(f"  Data loaded in {time.time() - t0:.1f}s\n")

total_combos = len(A_VALUES) * len(G_VALUES) * len(F_VALUES)
print(f"Sweeping {len(A_VALUES)} A x {len(G_VALUES)} G x {len(F_VALUES)} F = {total_combos} combinations\n")

results = []
combo_num = 0

for a_time, g_time, f_time in itertools.product(A_VALUES, G_VALUES, F_VALUES):
    combo_num += 1

    # Patch module-level variables
    tgcc.A_TIME_LIMIT_MINUTES = a_time
    tgcc.G_TIME_LIMIT_MINUTES = g_time
    tgcc.F_TIME_LIMIT_MINUTES = f_time

    cash = STARTING_CASH
    unsettled_cash = 0.0
    total_trades = 0
    total_wins = 0
    g_trades = 0
    a_trades = 0
    f_trades = 0

    for d in all_dates:
        if unsettled_cash > 0:
            cash += unsettled_cash
            unsettled_cash = 0.0

        picks = daily_picks.get(d, [])
        cash_account = cash < MARGIN_THRESHOLD

        states, new_cash, new_unsettled = simulate_day_combined(
            picks, cash, cash_account
        )

        for st in states:
            if st["exit_reason"] is not None:
                total_trades += 1
                if st["pnl"] > 0:
                    total_wins += 1
                s = st["strategy"]
                if s == "G":
                    g_trades += 1
                elif s == "A":
                    a_trades += 1
                elif s == "F":
                    f_trades += 1

        cash = new_cash
        unsettled_cash = new_unsettled

    final_equity = cash + unsettled_cash
    ret_pct = (final_equity / STARTING_CASH - 1) * 100
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

    results.append({
        "a_time": a_time,
        "g_time": g_time,
        "f_time": f_time,
        "final_equity": final_equity,
        "return_pct": ret_pct,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "g_trades": g_trades,
        "a_trades": a_trades,
        "f_trades": f_trades,
    })

    sys.stdout.write(f"\r  [{combo_num}/{total_combos}] A={a_time:>2} G={g_time:>2} F={f_time:>2}  "
                     f"-> ${final_equity:>10,.0f} ({ret_pct:+6.1f}%)  "
                     f"trades={total_trades} WR={win_rate:.1f}%")
    sys.stdout.flush()

print("\n")

# ---- Sort and display results ----
results.sort(key=lambda r: r["final_equity"], reverse=True)

header = (f"{'#':>4}  {'A_time':>6} {'G_time':>6} {'F_time':>6}  "
          f"{'Final_Equity':>14} {'Return%':>9}  "
          f"{'Trades':>7} {'WinRate':>8}  "
          f"{'G_tr':>5} {'A_tr':>5} {'F_tr':>5}")
print("=" * len(header))
print("  SWEEP RESULTS - Sorted by Final Equity (descending)")
print("=" * len(header))
print(header)
print("-" * len(header))

for i, r in enumerate(results):
    print(f"{i+1:>4}  {r['a_time']:>6} {r['g_time']:>6} {r['f_time']:>6}  "
          f"${r['final_equity']:>13,.0f} {r['return_pct']:>+8.1f}%  "
          f"{r['total_trades']:>7} {r['win_rate']:>7.1f}%  "
          f"{r['g_trades']:>5} {r['a_trades']:>5} {r['f_trades']:>5}")

print("-" * len(header))
print(f"\nTotal combinations: {len(results)}")
print(f"Best:  A={results[0]['a_time']} G={results[0]['g_time']} F={results[0]['f_time']}  "
      f"-> ${results[0]['final_equity']:,.0f} ({results[0]['return_pct']:+.1f}%)")
print(f"Worst: A={results[-1]['a_time']} G={results[-1]['g_time']} F={results[-1]['f_time']}  "
      f"-> ${results[-1]['final_equity']:,.0f} ({results[-1]['return_pct']:+.1f}%)")
