"""
Compare current code behavior with old vs new configs.
Goal: identify what causes the $71M → $91K drop.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import test_full as tf
from regime_filters import RegimeFilter

DATA_DIRS = tf.DEFAULT_DATA_DIRS
ALL_DATES, daily_picks = tf.load_all_picks(DATA_DIRS)

regime = RegimeFilter(
    spy_ma_period=tf.SPY_SMA_PERIOD,
    enable_vix=False, enable_spy_trend=True, enable_adaptive=False,
)
regime.load_data(ALL_DATES[0], ALL_DATES[-1])

CONFIGS = {
    "A: OLD config (SI 19%, gate ON)": {
        "SCALE_IN": 1,
        "SCALE_IN_TRIGGER_PCT": 19.0,
        "SCALE_IN_GATE_PARTIAL": True,
        "BREAKEVEN_STOP_PCT": 0.0,
        "ENTRY_CUTOFF_MINUTES": 0,
    },
    "B: SI OFF (current)": {
        "SCALE_IN": 0,
        "SCALE_IN_TRIGGER_PCT": 19.0,
        "SCALE_IN_GATE_PARTIAL": True,
        "BREAKEVEN_STOP_PCT": 0.0,
        "ENTRY_CUTOFF_MINUTES": 0,
    },
    "C: SI 14%, gate OFF (current)": {
        "SCALE_IN": 1,
        "SCALE_IN_TRIGGER_PCT": 14.0,
        "SCALE_IN_GATE_PARTIAL": False,
        "BREAKEVEN_STOP_PCT": 0.0,
        "ENTRY_CUTOFF_MINUTES": 0,
    },
}

results = {}
for label, cfg in CONFIGS.items():
    # Set config
    saves = {}
    for k, v in cfg.items():
        saves[k] = getattr(tf, k)
        setattr(tf, k, v)

    rolling_cash = tf.STARTING_CASH
    wins = 0
    losses = 0
    total_trades = 0
    scale_ins = 0

    for date_str in ALL_DATES:
        should_trade, _, _ = regime.check(date_str)
        if not should_trade:
            continue
        picks = daily_picks.get(date_str, [])
        if not picks:
            continue
        is_cash = rolling_cash < tf.MARGIN_THRESHOLD
        states, ending_cash = tf.simulate_day(picks, rolling_cash, cash_account=is_cash)
        for s in states:
            if s["entry_price"] is not None:
                total_trades += 1
                if s["pnl"] > 0:
                    wins += 1
                else:
                    losses += 1
                if s.get("scaled_in") and s.get("scale_in_time") is not None:
                    scale_ins += 1
        rolling_cash = ending_cash

    wr = wins / total_trades * 100 if total_trades > 0 else 0
    results[label] = {
        "cash": rolling_cash,
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "scale_ins": scale_ins,
    }
    print(f"  {label}: ${rolling_cash:>15,.2f}  trades={total_trades}  WR={wr:.1f}%  SI={scale_ins}")

    # Restore config
    for k, v in saves.items():
        setattr(tf, k, v)

print(f"\n{'='*80}")
print(f"  COMPARISON SUMMARY")
print(f"{'='*80}")
print(f"  {'Config':<35} {'Final Cash':>15} {'Trades':>7} {'WR':>6} {'SI':>4}")
print(f"  {'-'*35} {'-'*15} {'-'*7} {'-'*6} {'-'*4}")
for label, r in results.items():
    print(f"  {label:<35} ${r['cash']:>14,.2f} {r['trades']:>7} {r['wr']:>5.1f}% {r['scale_ins']:>4}")
print(f"\n  OLD baseline was: $71,084,943.58 with 44 scale-ins")
print(f"  If Config A << $71M, the CODE changes (not config) caused the drop.")
print(f"  If Config A ≈ $71M, the CONFIG changes caused the drop.")
