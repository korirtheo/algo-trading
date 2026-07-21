"""Analyze #511 on March 2026 OOS: daily P&L distribution, loss days."""
import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")
import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock

STARTING_CASH = 25_000
OOS_DATA_DIRS = ["stored_data_mar_may_2026"]

def configure_simulator():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

def load_data():
    dirs = [d for d in OOS_DATA_DIRS if Path(d).exists()]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if "2026-03" in d])
    return dates, picks_by_date

print("="*80)
print("MARCH 2026 OOS DAILY LOSS ANALYSIS: #511 (WITH TRAILING)")
print("="*80)

configure_simulator()
dates, picks_by_date = load_data()
print(f"\nAnalyzing {len(dates)} trading days in March 2026\n")

# Load #511 config
with open("config/trial_g511_l626_v3_overlay.json") as f:
    params = dict(json.load(f)["params"])

# Run day-by-day
daily_pnl = {}
with _param_lock:
    _oc_set_params(params)
    snapshot = _build_param_snapshot()

cash = float(STARTING_CASH)
for d in dates:
    picks = picks_by_date.get(d, [])
    if not picks:
        daily_pnl[d] = 0.0
        continue
    
    cash_account = cash < MARGIN_THRESHOLD
    try:
        states, cash, unsettled, _ = tgc.simulate_day_combined(picks, cash, cash_account, params=snapshot)
    except Exception as e:
        daily_pnl[d] = 0.0
        continue
    
    day_pnl = sum(st.get("pnl", 0) for st in states if st.get("exit_reason") and st.get("pnl"))
    daily_pnl[d] = day_pnl
    cash += (unsettled if cash_account else 0)

# Analysis
print("DAILY P&L (March 2026 OOS):")
print("-" * 60)
loss_days = [d for d, pnl in daily_pnl.items() if pnl < 0]
win_days = [d for d, pnl in daily_pnl.items() if pnl > 0]
flat_days = [d for d, pnl in daily_pnl.items() if pnl == 0]

for d in sorted(daily_pnl.keys()):
    pnl = daily_pnl[d]
    status = "LOSS" if pnl < 0 else "WIN " if pnl > 0 else "FLAT"
    print(f"{d}: {status} | ${pnl:+8,.0f}")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
total_pnl = sum(daily_pnl.values())
print(f"Total P&L:       ${total_pnl:,.2f}")
print(f"Win days:        {len(win_days)} ({len(win_days)/len(dates)*100:.1f}%)")
print(f"Loss days:       {len(loss_days)} ({len(loss_days)/len(dates)*100:.1f}%)")
print(f"Flat days:       {len(flat_days)} ({len(flat_days)/len(dates)*100:.1f}%)")
print(f"\nAvg win day:     ${sum(daily_pnl[d] for d in win_days)/max(len(win_days),1):,.0f}")
print(f"Avg loss day:    ${sum(daily_pnl[d] for d in loss_days)/max(len(loss_days),1):,.0f}")
print(f"Largest loss:    ${min(daily_pnl.values()):,.0f} ({min(daily_pnl, key=daily_pnl.get)})")
