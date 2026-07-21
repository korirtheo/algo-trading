"""Backtest 2026-07-15 to compare with live trading."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import json

# Load the live params
with open("optuna_best_params_dynamic_2021_2026.json") as f:
    params = json.load(f)

DATA_DIR = Path("stored_data_jul_2026")
ET = ZoneInfo("America/New_York")

# Watchlist from live
WATCHLIST = ['VIVS', 'ELVA', 'KUST', 'ERNA', 'VTAK', 'TGHL', 'TRT', 'IZM',
             'GNTA', 'VEEE', 'KOPN', 'NVVE', 'GCTK', 'NTHI', 'QNC', 'SHMD',
             'YJ', 'MTEX', 'GEVO', 'BCDA', 'SOBR']

def load_data(ticker, date_str="2026-07-15"):
    """Load 1-min bars and resample to 2-min aligned to 9:30."""
    path = DATA_DIR / f"{ticker}_{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.time >= dt_time(9, 30)]
    df = df[df['timestamp'].dt.time < dt_time(16, 0)]

    if df.empty:
        return df

    # Resample to 2-min aligned to 9:30
    df = df.set_index('timestamp')
    df = df.resample('2min', origin=datetime(2026, 7, 15, 9, 30, tzinfo=ET)).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }).dropna()

    return df.reset_index()


# V3 strategy simplified
def check_v3_entry(bars, params):
    """V3 green candle momentum strategy."""
    if len(bars) < 10:
        return False

    last = bars.iloc[-1]

    # Green candle
    if last['Close'] <= last['Open']:
        return False

    # Momentum check
    body_pct = (last['Close'] - last['Open']) / last['Open'] * 100
    if body_pct < params.get('v3_min_body_pct', 1.0):
        return False

    # Volume spike
    avg_vol = bars['Volume'].tail(10).mean()
    if last['Volume'] < avg_vol * params.get('v3_min_volume_ratio', 1.5):
        return False

    # Above recent highs
    recent_high = bars['High'].tail(5).max()
    if last['Close'] < recent_high * 0.98:
        return False

    return True


def run_backtest():
    """Run simple backtest for 7/15."""
    print("=" * 60)
    print("BACKTEST: 2026-07-15")
    print("=" * 60)

    trades = []

    for ticker in WATCHLIST:
        df = load_data(ticker)
        if df.empty:
            continue

        print(f"\n{ticker}: {len(df)} bars")

        # Simple scan for V3 entries
        for i in range(10, len(df)):
            window = df.iloc[:i+1]

            if check_v3_entry(window, params):
                entry_price = window.iloc[-1]['Close']
                entry_idx = i

                # Find exit (trail or stop)
                exit_price = None
                exit_reason = None
                peak = entry_price

                for j in range(i+1, len(df)):
                    bar = df.iloc[j]
                    peak = max(peak, bar['High'])
                    trail_stop = peak * (1 - params.get('v3_trail_pct', 10) / 100)
                    hard_stop = entry_price * (1 - params.get('v3_stop_pct', 12) / 100)

                    if bar['Low'] <= trail_stop:
                        exit_price = trail_stop
                        exit_reason = "V3_TRAIL"
                        break
                    elif bar['Low'] <= hard_stop:
                        exit_price = hard_stop
                        exit_reason = "V3_STOP"
                        break

                    # EOD
                    if j == len(df) - 1:
                        exit_price = bar['Close']
                        exit_reason = "EOD"
                        break

                if exit_price:
                    pnl = (exit_price - entry_price) / entry_price * 100 * 1000  # assume $1000 position
                    trades.append({
                        'ticker': ticker,
                        'strategy': 'V3',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'reason': exit_reason,
                    })
                    print(f"  TRADE: ${entry_price:.2f} -> ${exit_price:.2f} = ${pnl:+.2f} ({exit_reason})")
                    break  # One trade per ticker

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Total Trades: {len(trades)}")
    if trades:
        total_pnl = sum(t['pnl'] for t in trades)
        wins = [t for t in trades if t['pnl'] > 0]
        print(f"Total PnL: ${total_pnl:+,.2f}")
        print(f"Win Rate: {len(wins)}/{len(trades)} ({len(wins)/len(trades)*100:.1f}%)")
        print(f"\nBy Ticker:")
        for t in sorted(trades, key=lambda x: x['pnl'], reverse=True):
            print(f"  {t['ticker']}: ${t['pnl']:+,.2f} ({t['reason']})")

    # Compare with live
    print("\n" + "=" * 60)
    print("LIVE vs BACKTEST")
    print("=" * 60)
    print("Live traded: TGHL (PnL: -$330.59)")

    backtest_tickers = [t['ticker'] for t in trades]
    print(f"Backtest traded: {', '.join(backtest_tickers) if backtest_tickers else 'None'}")

    if 'NVVE' in backtest_tickers:
        nvve_trade = [t for t in trades if t['ticker'] == 'NVVE'][0]
        print(f"  NVVE backtest: ${nvve_trade['pnl']:+,.2f}")
    else:
        print("  NVVE: No signal (same as live)")

    if 'SOBR' in backtest_tickers:
        sobr_trade = [t for t in trades if t['ticker'] == 'SOBR'][0]
        print(f"  SOBR backtest: ${sobr_trade['pnl']:+,.2f}")
        print(f"  ** SOBR was NOT in live watchlist!")
    else:
        print("  SOBR: No signal")


if __name__ == "__main__":
    run_backtest()
