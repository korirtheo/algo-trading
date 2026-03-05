"""Compare trades on 2025-01-22 between old code ($71M run) and current code."""
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

# Import old two-pass simulation
from verify_root_cause2 import simulate_day_old_twopass

TARGET_DATE = "2025-01-22"

# Run BOTH code paths up to Jan 22, then print details for that day
def run_to_date(sim_func, config_overrides, label):
    """Run simulation up to TARGET_DATE using given sim function and config."""
    saves = {}
    for k, v in config_overrides.items():
        saves[k] = getattr(tf, k)
        setattr(tf, k, v)

    rolling_cash = tf.STARTING_CASH
    for date_str in ALL_DATES:
        if date_str > TARGET_DATE:
            break
        should_trade, _, _ = regime.check(date_str)
        if not should_trade:
            if date_str == TARGET_DATE:
                print(f"\n  [{label}] {TARGET_DATE}: REGIME SKIP (SPY below SMA)")
            continue
        picks = daily_picks.get(date_str, [])
        if not picks:
            continue
        is_cash = rolling_cash < tf.MARGIN_THRESHOLD

        if date_str == TARGET_DATE:
            print(f"\n{'='*80}")
            print(f"  [{label}] {TARGET_DATE}  Starting Cash: ${rolling_cash:,.2f}  (cash_acct={is_cash})")
            print(f"  Config: SI_TRIGGER={tf.SCALE_IN_TRIGGER_PCT}%, GATE={tf.SCALE_IN_GATE_PARTIAL}")
            print(f"{'='*80}")
            tickers = [p['ticker'] for p in picks]
            gaps = [round(p['gap_pct'], 1) for p in picks]
            print(f"  Picks: {tickers}")
            print(f"  Gaps:  {gaps}")

        states, ending_cash = sim_func(picks, rolling_cash, cash_account=is_cash)

        if date_str == TARGET_DATE:
            for s in states:
                ticker = s["ticker"]
                entry = s["entry_price"]
                orig_entry = s.get("original_entry_price")
                exit_p = s.get("exit_price")
                reason = s["exit_reason"]
                shares = s.get("shares", 0)
                remaining = s.get("remaining_shares", 0)
                pnl = s.get("pnl", s["total_exit_value"] - s["total_cash_spent"] if s["total_cash_spent"] > 0 else 0)
                scaled = s.get("scaled_in", False)
                si_price = s.get("scale_in_price")
                partial = s.get("partial_sold", False)
                partial_p = s.get("partial_sell_price")
                cost = s.get("total_cash_spent", 0)

                if entry is None:
                    print(f"  {ticker:<8} NO_ENTRY ({reason})")
                else:
                    exit_str = f"${exit_p:.2f}" if exit_p else "N/A"
                    print(f"  {ticker:<8} entry=${entry:.2f} orig=${orig_entry:.2f} "
                          f"exit={exit_str} shares={shares:.1f} remain={remaining:.1f} "
                          f"reason={reason} pnl=${pnl:+,.2f} cost=${cost:.2f}")
                    if scaled:
                        si_str = f"${si_price:.2f}" if si_price else "N/A"
                        print(f"           SCALE-IN at {si_str}")
                    if partial:
                        pp_str = f"${partial_p:.2f}" if partial_p else "N/A"
                        print(f"           PARTIAL at {pp_str}")

            day_pnl = ending_cash - rolling_cash
            print(f"  Day P&L: ${day_pnl:+,.2f}  Ending Cash: ${ending_cash:,.2f}")

        rolling_cash = ending_cash

    for k, v in saves.items():
        setattr(tf, k, v)

# Config for the $71M run (old code: deferred scale-in, entry_price for exits)
OLD_CONFIG = {
    "SCALE_IN": 1,
    "SCALE_IN_TRIGGER_PCT": 19.0,
    "SCALE_IN_GATE_PARTIAL": True,
    "BREAKEVEN_STOP_PCT": 0.0,
    "ENTRY_CUTOFF_MINUTES": 0,
}

# Config for run_024941 (new code: inline scale-in, original_entry_price for exits)
NEW_CONFIG = {
    "SCALE_IN": 1,
    "SCALE_IN_TRIGGER_PCT": 13.0,
    "SCALE_IN_GATE_PARTIAL": False,
    "BREAKEVEN_STOP_PCT": 0.0,
    "ENTRY_CUTOFF_MINUTES": 0,
}

print("TEST 1: OLD code path (deferred scale-in, entry_price for exits, SI=19%, gate=True)")
run_to_date(simulate_day_old_twopass, OLD_CONFIG, "OLD/$71M")

print("\n\nTEST 2: CURRENT code path (inline scale-in, original_entry_price for exits, SI=13%, gate=False)")
run_to_date(tf.simulate_day, NEW_CONFIG, "NEW/$69K")
