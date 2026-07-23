"""
Run OOS backtest with best Optuna trial params (2026-03 onwards).
Usage: python run_best_trial_backtest.py
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psycopg2
import numpy as np

# ── 1. Load best trial params (with correct enable_* from user_attrs) ──
conn = psycopg2.connect("postgresql://postgres@127.0.0.1:5432/optuna_oglhmafp")
cur = conn.cursor()
cur.execute("""
    SELECT t.trial_id, t.number, v.value
    FROM trials t JOIN trial_values v ON t.trial_id = v.trial_id
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = 'oglhmafp_v4' AND t.state = 'COMPLETE'
    ORDER BY v.value DESC LIMIT 1
""")
row = cur.fetchone()
trial_id, trial_num, trial_value = row
print(f"Best trial #{trial_num} (IS: ${trial_value:,.2f})")

# Load strategy params
cur.execute("SELECT param_name, param_value FROM trial_params WHERE trial_id = %s", (trial_id,))
params = {name: val for name, val in cur.fetchall()}

# Load enable_* from user_attrs (set_user_attr stores correct booleans)
cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id = %s", (trial_id,))
for key, val in cur.fetchall():
    if key.startswith("enable_"):
        params[key] = val == "true" if isinstance(val, str) else val
conn.close()
print(f"Loaded {len(params)} params from PostgreSQL\n")

# PostgreSQL stores ints as floats — cast whole-number floats back to int
for k, v in params.items():
    if isinstance(v, float) and v == int(v):
        params[k] = int(v)

# ── 2. Set slippage flags ─────────────────────────────────────────────
import test_green_candle_combined as tgc
tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True

# ── 3. Apply params via set_strategy_params ────────────────────────────
from optimize_combined import set_strategy_params
set_strategy_params(params)

# Apply enable_* bits from params (correct from user_attrs)
for s in "abcdefghijklmnoprsvwx":
    setattr(tgc, f"ENABLE_{s.upper()}", params.get(f"enable_{s}", False))

# ── 4. Load data & filter to 2026-03+ ─────────────────────────────────
from test_full import load_all_picks, _is_warrant_or_unit, SLIPPAGE_PCT, STARTING_CASH

data_dirs = [
    "stored_data_mar_may_2026",
    "stored_data_jun_2026",
    "stored_data_jul_2026",
]
all_dates, daily_picks = load_all_picks(data_dirs)
print(f"Total data: {len(all_dates)} days ({all_dates[0]} to {all_dates[-1]})")

# Filter to OOS: 2026-03-01 onwards (training ends Feb 2026)
DATE_START = "2026-03-01"
oos_dates = [d for d in all_dates if d >= DATE_START]
print(f"OOS period: {len(oos_dates)} days ({oos_dates[0]} to {oos_dates[-1]})\n")

if not oos_dates:
    print("ERROR: No OOS data found")
    sys.exit(1)

# ── 5. Run backtest (adapted from test_green_candle_combined.py main) ──
import pandas as pd
from collections import defaultdict

STRAT_KEYS = [
    "H", "G", "A", "F", "D", "V", "P", "M", "R", "W",
    "O", "B", "K", "C", "S", "E", "I", "J", "N", "L", "X",
]

# Import backtest functions
from test_green_candle_combined import (
    _load_r_intraday, simulate_day_combined, R_DAY1_MIN_GAP,
    STRAT_PRIORITY, TOP_N,
)
from optimize_combined import ALL_STRATS

# Pre-scan for R candidates (only within OOS range)
r_day2_picks = {}
for idx in range(len(all_dates) - 1):
    d1, d2 = all_dates[idx], all_dates[idx + 1]
    if d2 < DATE_START:
        continue
    for pick in daily_picks.get(d1, []):
        if pick["gap_pct"] < R_DAY1_MIN_GAP:
            continue
        mh = pick.get("market_hour_candles")
        if mh is None or len(mh) < 10:
            continue
        day1_close = float(mh["Close"].values[-1])
        d2_data = _load_r_intraday(pick["ticker"], d2, data_dirs)
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

print(f"R candidates: {sum(len(v) for v in r_day2_picks.values())} across {len(r_day2_picks)} days")

# Run day-by-day
cash = STARTING_CASH
unsettled = 0.0
all_results = []
all_selection_logs = []

for d in oos_dates:
    picks = daily_picks.get(d, [])
    r_extra = r_day2_picks.get(d, [])
    day_picks = picks + r_extra

    states, cash, unsettled, selection_log = simulate_day_combined(
        day_picks, cash, is_live=False
    )

    # Count trades, wins, losses per strategy
    day_trades = 0
    day_wins = 0
    day_losses = 0
    counts = {k: [0, 0] for k in STRAT_KEYS}  # [trades, wins]
    day_pnl = 0.0

    for st in states:
        if st["exit_reason"] is not None:
            day_trades += 1
            day_pnl += st["pnl"]
            s = st["strategy"]
            if s in counts:
                counts[s][0] += 1
                if st["pnl"] > 0:
                    counts[s][1] += 1
                    day_wins += 1
                else:
                    day_losses += 1

    equity = cash + unsettled

    all_results.append({
        "date": d,
        "day_pnl": day_pnl,
        "equity": equity,
        "trades": day_trades,
        "wins": day_wins,
        "losses": day_losses,
        "states": states,
        **{f"{k.lower()}_trades": counts[k][0] for k in STRAT_KEYS},
        **{f"{k.lower()}_wins": counts[k][1] for k in STRAT_KEYS},
    })
    if selection_log:
        all_selection_logs.append(selection_log)

# ── 6. Summary ─────────────────────────────────────────────────────────
final_equity = cash + unsettled
total_trades = sum(r["trades"] for r in all_results)
total_wins = sum(r["wins"] for r in all_results)
total_losses = sum(r["losses"] for r in all_results)

strat_totals = {k: sum(r[f"{k.lower()}_trades"] for r in all_results) for k in STRAT_KEYS}
strat_wins = {k: sum(r[f"{k.lower()}_wins"] for r in all_results) for k in STRAT_KEYS}

all_exits = {}
all_trade_pnls = []
strat_pnls = {k: [] for k in STRAT_KEYS}
for r in all_results:
    for st in r["states"]:
        if st["exit_reason"] is not None:
            reason_key = f"{st['strategy']}_{st['exit_reason']}"
            all_exits[reason_key] = all_exits.get(reason_key, 0) + 1
            all_trade_pnls.append(st["pnl"])
            s = st["strategy"]
            if s in strat_pnls:
                strat_pnls[s].append(st["pnl"])

daily_pnls = [r["day_pnl"] for r in all_results if r["trades"] > 0]
green = sum(1 for p in daily_pnls if p > 0) if daily_pnls else 0
red = sum(1 for p in daily_pnls if p <= 0) if daily_pnls else 0
sharpe = (
    (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252)
    if daily_pnls and np.std(daily_pnls) > 0 else 0
)

avg_win = np.mean([p for p in all_trade_pnls if p > 0]) if total_wins > 0 else 0
avg_loss = np.mean([p for p in all_trade_pnls if p <= 0]) if total_losses > 0 else 0

# Max drawdown
equities = [r["equity"] for r in all_results]
peak = equities[0]
max_dd_dollar = 0.0
max_dd_pct = 0.0
for eq in equities:
    if eq > peak:
        peak = eq
    dd_dollar = peak - eq
    dd_pct = dd_dollar / peak * 100 if peak > 0 else 0
    if dd_dollar > max_dd_dollar:
        max_dd_dollar = dd_dollar
        max_dd_pct = dd_pct

# Profit factor
gross_wins = sum(p for p in all_trade_pnls if p > 0)
gross_losses = abs(sum(p for p in all_trade_pnls if p <= 0))
pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

# Geometric mean
pct_returns = []
for i in range(1, len(equities)):
    if equities[i-1] > 0:
        pct_returns.append(equities[i] / equities[i-1])
geo_mean = (np.prod(pct_returns) ** (1/len(pct_returns)) - 1) * 100 if pct_returns else 0

print(f"\n{'=' * 70}")
print(f"  OOS BACKTEST: 2026-03-01 onwards | Best Trial #{trial_num}")
print(f"{'=' * 70}")
print(f"  Starting Cash:    ${STARTING_CASH:,}")
print(f"  Ending Equity:    ${final_equity:,.0f}  ({(final_equity / STARTING_CASH - 1) * 100:+.1f}%)")
if unsettled > 0:
    print(f"    (Cash: ${cash:,.0f} + Unsettled: ${unsettled:,.0f})")
print(f"  Trading Days:     {len(oos_dates)}")
print(f"  Total Trades:     {total_trades}")
print(f"    Winners:        {total_wins} ({total_wins / max(total_trades, 1) * 100:.1f}%)")
print(f"    Losers:         {total_losses}")
print(f"  Avg Win:          ${avg_win:+,.0f}")
print(f"  Avg Loss:         ${avg_loss:+,.0f}")
print(f"  Profit Factor:    {pf:.2f}")
print(f"  Geo Mean (daily): {geo_mean:+.4f}%")
print(f"  Sharpe (ann.):    {sharpe:.2f}")
print(f"  Max Drawdown:     ${max_dd_dollar:,.0f} ({max_dd_pct:.1f}%)")

# ── 7. Per-strategy breakdown ──────────────────────────────────────────
strat_info = [
    ("H (High Conviction)", "H"),
    ("G (Big Gap Runner)", "G"),
    ("A (Quick Scalp)", "A"),
    ("F (Catch-All)", "F"),
    ("D (Opening Dip)", "D"),
    ("V (VWAP Reclaim)", "V"),
    ("P (PM High Break)", "P"),
    ("M (Morning Spike)", "M"),
    ("R (Multi-Day)", "R"),
    ("W (Consolidation)", "W"),
    ("O (Range Breakout)", "O"),
    ("B (Red-to-Green)", "B"),
    ("K (First Pullback)", "K"),
    ("C (Micro Flag)", "C"),
    ("S (Stuff-and-Break)", "S"),
    ("E (Gap-and-Go)", "E"),
    ("I (PM High Imm)", "I"),
    ("J (VWAP+PMH)", "J"),
    ("N (HOD Reclaim)", "N"),
    ("L (Low Float)", "L"),
    ("X (Recovery)", "X"),
]

print(f"\n{'=' * 70}")
print(f"  PER-STRATEGY BREAKDOWN")
print(f"{'=' * 70}")
print(f"  {'Strategy':<25} {'Trades':>7} {'Wins':>6} {'WR%':>6} {'Total PnL':>12} {'Avg Win':>10} {'Avg Loss':>10}")
print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*6} {'-'*12} {'-'*10} {'-'*10}")

total_strat_pnl = 0
for label, key in strat_info:
    total_s = strat_totals[key]
    wins_s = strat_wins[key]
    pnls_s = strat_pnls[key]
    wr = wins_s / max(total_s, 1) * 100
    total_pnl = sum(pnls_s) if pnls_s else 0
    total_strat_pnl += total_pnl
    w = [p for p in pnls_s if p > 0]
    l = [p for p in pnls_s if p <= 0]
    avg_w = np.mean(w) if w else 0
    avg_l = np.mean(l) if l else 0
    if total_s > 0:
        print(f"  {label:<25} {total_s:>7} {wins_s:>6} {wr:>5.1f}% ${total_pnl:>+10,.0f} ${avg_w:>+9,.0f} ${avg_l:>+9,.0f}")

print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*6} {'-'*12} {'-'*10} {'-'*10}")
print(f"  {'TOTAL':<25} {total_trades:>7} {total_wins:>6} {total_wins/max(total_trades,1)*100:>5.1f}% ${total_strat_pnl:>+10,.0f}")

# ── 8. Exit reasons ────────────────────────────────────────────────────
print(f"\n  Exit Reasons:")
for reason, count in sorted(all_exits.items(), key=lambda x: -x[1]):
    print(f"    {reason:<25} {count:>4} ({count / max(total_trades, 1) * 100:.1f}%)")

# ── 9. Daily stats ─────────────────────────────────────────────────────
if daily_pnls:
    print(f"\n  Green Days:       {green}/{len(daily_pnls)} ({green / len(daily_pnls) * 100:.1f}%)")
    print(f"  Red Days:         {red}/{len(daily_pnls)}")
    print(f"  Best Day:         ${max(daily_pnls):+,.0f}")
    print(f"  Worst Day:        ${min(daily_pnls):+,.0f}")
    print(f"  Avg P&L/Day:      ${np.mean(daily_pnls):+,.0f}")

# ── 10. Monthly breakdown ──────────────────────────────────────────────
print(f"\n  Monthly Breakdown:")
monthly = {}
for r in all_results:
    month = r["date"][:7]
    if month not in monthly:
        monthly[month] = {"pnl": 0, "trades": 0, "wins": 0, "equity_start": 0, "equity_end": 0}
    monthly[month]["pnl"] += r["day_pnl"]
    monthly[month]["trades"] += r["trades"]
    monthly[month]["wins"] += r["wins"]
    monthly[month]["equity_end"] = r["equity"]
    if monthly[month]["equity_start"] == 0:
        monthly[month]["equity_start"] = r["equity"]

for month in sorted(monthly.keys()):
    m = monthly[month]
    wr = m["wins"] / max(m["trades"], 1) * 100
    ret = (m["equity_end"] / m["equity_start"] - 1) * 100 if m["equity_start"] > 0 else 0
    print(f"    {month}  PnL: ${m['pnl']:>+10,.0f}  Trades: {m['trades']:>4}  WR: {wr:>5.1f}%  Ret: {ret:>+6.1f}%")

print(f"\n{'=' * 70}")
