"""
Backtest 2026-07-13 to understand why Alpaca live trading had 0 trades.
"""
import sys
import os
import pickle
from datetime import datetime

sys.path.insert(0, ".")

import pandas as pd
import test_green_candle_combined as tgc
from test_full import MARGIN_THRESHOLD
import json

DATA_DIR = "stored_data_jul_2026"
TARGET_DATE = "2026-07-13"
CONFIG_PATH = "config/trial_g511_l626_v3_overlay.json"
MIN_GAP_PCT = 2.0
MIN_VOLUME_USD = 250000

def load_picks():
    """Build picks list by scanning intraday CSVs and identifying gap-ups."""
    intraday_dir = os.path.join(DATA_DIR, "intraday")
    daily_dir = os.path.join(DATA_DIR, "daily")

    if not os.path.exists(intraday_dir):
        print(f"ERROR: {intraday_dir} not found")
        return []

    picks = []
    csv_files = [f for f in os.listdir(intraday_dir) if f.endswith('.csv')]
    print(f"Scanning {len(csv_files)} CSV files...")

    for i, csv_file in enumerate(csv_files, 1):
        ticker = csv_file.replace('.csv', '')
        intraday_path = os.path.join(intraday_dir, csv_file)
        daily_path = os.path.join(daily_dir, csv_file)

        if i % 50 == 0:
            print(f"  Processed {i}/{len(csv_files)}...")

        try:
            # Read intraday 1-min bars
            intraday_df = pd.read_csv(intraday_path, index_col=0, parse_dates=True)
            if len(intraday_df) == 0:
                continue

            # Read daily bars for prev_close
            if not os.path.exists(daily_path):
                continue
            daily_df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
            if len(daily_df) == 0:
                continue

            # Get prev close (last available daily close before target date)
            prev_close = float(daily_df.iloc[-1]['close'])

            # Get open price from first 1-min bar
            open_price = float(intraday_df.iloc[0]['open'])

            # Compute gap
            gap_pct = (open_price / prev_close - 1) * 100

            if gap_pct < MIN_GAP_PCT:
                continue

            # Compute pre-market volume (sum of volume before 9:30 AM ET)
            # Alpaca timestamps are in UTC, 9:30 AM ET = 13:30 UTC (or 14:30 during DST)
            # For 2026-07-13 (July = DST), 9:30 AM ET = 13:30 UTC
            market_open_utc = pd.Timestamp(f"{TARGET_DATE} 13:30:00", tz='UTC')
            premarket = intraday_df[intraday_df.index < market_open_utc]
            pm_volume = int(premarket['volume'].sum()) if len(premarket) > 0 else 0
            pm_dollar_vol = pm_volume * prev_close

            if pm_dollar_vol < MIN_VOLUME_USD:
                continue

            # Build pick dict
            pick = {
                "ticker": ticker,
                "prev_close": prev_close,
                "open": open_price,
                "gap_pct": gap_pct,
                "pm_volume": pm_volume,
                "pm_dollar_vol": pm_dollar_vol,
            }
            picks.append(pick)

        except Exception as e:
            continue

    picks.sort(key=lambda x: x["gap_pct"], reverse=True)
    print(f"\nFound {len(picks)} gap-ups >= {MIN_GAP_PCT}% with pm_$vol >= ${MIN_VOLUME_USD:,.0f}")
    print(f"Top 20:")
    for p in picks[:20]:
        print(f"  {p['ticker']:>6}  gap={p['gap_pct']:>+6.2f}%  pm_$vol=${p['pm_dollar_vol']:>10,.0f}")

    return picks


def run_backtest(picks, config):
    """Run backtest on 2026-07-13 with deployed strategy."""
    print(f"\nRunning backtest on {TARGET_DATE}...")

    # Configure simulator
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

    # Load config params
    params = config["params"]

    # Set strategy params
    from optimize_combined import set_strategy_params, _build_param_snapshot, _param_lock
    with _param_lock:
        set_strategy_params(params)
        snapshot = _build_param_snapshot()

    # Run simulation
    cash = 25000.0
    cash_account = cash < MARGIN_THRESHOLD

    try:
        states, final_cash, unsettled, day_stats = tgc.simulate_day_combined(
            picks, cash, cash_account, params=snapshot
        )
    except Exception as e:
        print(f"ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Analyze results
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS: {TARGET_DATE}")
    print(f"{'='*60}")
    print(f"Starting cash: ${cash:,.2f}")
    print(f"Final cash:    ${final_cash:,.2f}")
    print(f"Unsettled:     ${unsettled:,.2f}")
    print(f"Effective:     ${final_cash + unsettled:,.2f}")
    print(f"PnL:           ${final_cash + unsettled - cash:,.2f}")

    # Count trades
    trades = [s for s in states if s.get("exit_reason") is not None]
    print(f"\nTotal trades: {len(trades)}")

    if len(trades) == 0:
        print("\n⚠️  NO TRADES EXECUTED")
        print("\nPossible reasons:")
        print("  1. No picks met entry criteria (gap%, 2nd_green, 2nd_new_high)")
        print("  2. Slippage/vol caps rejected all position sizes")
        print("  3. Price action didn't trigger entry conditions")
        print(f"\nTop 10 picks to inspect:")
        for p in picks[:10]:
            print(f"  {p['ticker']:>6}  gap={p['gap_pct']:>6.2f}%  pm_$vol=${p['pm_dollar_vol']:>10,.0f}")
        return

    # Summarize trades
    wins = [s for s in trades if s.get("pnl", 0) > 0]
    losses = [s for s in trades if s.get("pnl", 0) <= 0]
    total_pnl = sum(s.get("pnl", 0) for s in trades)

    print(f"  Wins:   {len(wins)}")
    print(f"  Losses: {len(losses)}")
    print(f"  WR:     {len(wins)/len(trades)*100:.1f}%")
    print(f"  Avg PnL: ${total_pnl/len(trades):,.2f}")

    print(f"\nTrade details:")
    for i, s in enumerate(trades[:20], 1):
        print(f"  [{i}] {s['ticker']:>6}  {s['strategy']:>2}  "
              f"entry=${s.get('entry_price', 0):.2f}  exit=${s.get('exit_price', 0):.2f}  "
              f"pnl=${s.get('pnl', 0):>+8,.2f}  reason={s['exit_reason']}")


def main():
    print(f"{'='*60}")
    print(f"Backtest 2026-07-13 (Understanding Alpaca 0-Trade Day)")
    print(f"{'='*60}")

    # Load deployed config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    print(f"Loaded config: {config['label']}")

    # Build picks
    picks = load_picks()

    if len(picks) == 0:
        print("\n⚠️  NO PICKS FOUND")
        print("This explains why live trading had 0 trades:")
        print("  → No stocks met gap-up + volume criteria on 2026-07-13")
        return

    # Run backtest
    run_backtest(picks, config)


if __name__ == "__main__":
    main()
