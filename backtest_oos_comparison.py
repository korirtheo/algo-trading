"""
Backtest high-score trials on OOS window (Mar-Jun 2026) to compare with #511.

Trials to test:
  #818 (squeeze) - Score: $137.5M (unknown OOS)
  #587 (squeeze) - Score: $67.0M (unknown OOS)
  #564 (G+V3) - Score: $114.1M (unknown OOS)
  #511 (W21b) - Score: $47.5M, Mar-Jun OOS: $660,731 (baseline)
"""

import json
import os
import sys

sys.path.insert(0, ".")

import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params, _build_param_snapshot, _param_lock

STARTING_CASH = 25_000
OOS_DATE_LO = "2026-03-01"
OOS_DATE_HI = "2026-06-30"

# Configure simulator
tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0
tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15
tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.NEWS_MODULATOR_ENABLED = False

# Load OOS data (Mar-Jun 2026)
print("Loading OOS data (Mar-Jun 2026)...")
dirs = [d for d in ["stored_data_mar_may_2026", "stored_data_jun_2026"] if os.path.exists(d)]
all_dates, picks_by_date = load_all_picks(dirs)
oos_dates = sorted([d for d in all_dates if OOS_DATE_LO <= d <= OOS_DATE_HI])
print(f"OOS window: {len(oos_dates)} days ({oos_dates[0]} to {oos_dates[-1]})")

def run_backtest(params, dates, picks_by_date, label):
    """Run backtest on OOS window."""
    cash = float(STARTING_CASH)
    all_trades = []

    for d in dates:
        picks = picks_by_date.get(d, [])
        if not picks:
            continue
        cash_account = cash < MARGIN_THRESHOLD
        try:
            states, cash, unsettled, _ = tgc.simulate_day_combined(picks, cash, cash_account, params=None)
        except Exception as e:
            continue

        effective_cash = cash + (unsettled if cash_account else 0)
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0)
                if pnl is not None:
                    all_trades.append(pnl)
        cash = effective_cash

    n = len(all_trades)
    if n == 0:
        return {"n": 0, "total_pnl": 0, "pf": 0, "wr": 0, "equity": cash}

    total_pnl = sum(all_trades)
    wins = sum(1 for p in all_trades if p > 0)
    gross_win = sum(p for p in all_trades if p > 0)
    gross_loss = abs(sum(p for p in all_trades if p <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    wr = wins / n * 100

    return {"n": n, "total_pnl": total_pnl, "pf": pf, "wr": wr, "equity": cash}

# Test each trial
trials_to_test = [
    ("config/trial_818_squeeze_extracted.json", "#818 Squeeze"),
    ("config/trial_587_squeeze_extracted.json", "#587 Squeeze"),
    ("config/trial_g_v3_joint_best.json", "#564 G+V3"),
    ("config/trial_w21b_511_deploy.json", "#511 W21b (Current Live)"),
]

print("\n" + "="*120)
print("OOS BACKTEST RESULTS (Mar-Jun 2026)")
print("="*120)
print()

results = []

for config_path, label in trials_to_test:
    if not os.path.exists(config_path):
        print(f"{label}: Config not found")
        continue

    with open(config_path) as f:
        data = json.load(f)

    params = data.get("params", {})

    # Set params and build snapshot
    with _param_lock:
        set_strategy_params(params)
        snapshot = _build_param_snapshot()

    # Run backtest
    result = run_backtest(params, oos_dates, picks_by_date, label)
    results.append((label, result))

    print(f"{label}")
    print(f"  Trades:       {result['n']}")
    print(f"  Total PnL:    ${result['total_pnl']:,.0f}")
    print(f"  Profit Factor: {result['pf']:.2f}")
    print(f"  Win Rate:     {result['wr']:.1f}%")
    print(f"  Final Equity: ${result['equity']:,.0f}")
    print()

# Rank by PnL
print("\n" + "="*120)
print("RANKING BY OOS PnL (Mar-Jun 2026)")
print("="*120)
print()

ranked = sorted(results, key=lambda x: x[1]['total_pnl'], reverse=True)
for i, (label, result) in enumerate(ranked, 1):
    print(f"{i}. {label:>20}  ${result['total_pnl']:>12,.0f}  PF={result['pf']:.2f}  WR={result['wr']:.1f}%  n={result['n']}")

EOF
