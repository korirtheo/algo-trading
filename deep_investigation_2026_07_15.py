"""Deep investigation: Why backtest differs from live and why no G/L strategies fired."""
import pandas as pd
import re
from pathlib import Path
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import json

ET = ZoneInfo("America/New_York")
DATA_DIR = Path("stored_data_jul_2026")
LIVE_LOG = Path("logs/2026-07-15/live_log_2026-07-15")

# Load live params
with open("optuna_best_params_dynamic_2021_2026.json") as f:
    params = json.load(f)

TICKERS = ['VIVS', 'ELVA', 'TGHL']


def load_alpaca_bars(ticker):
    """Load raw Alpaca data with proper timezone handling."""
    path = DATA_DIR / f"{ticker}_2026-07-15.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Convert to ET
    df['timestamp_et'] = df['timestamp'].dt.tz_convert(ET)
    df['time_et'] = df['timestamp_et'].dt.time

    return df


def extract_live_2min_bars(ticker):
    """Extract live 2-min bars with full details."""
    bars = []

    with open(LIVE_LOG, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if f'EMIT 2min: {ticker}' in line:
                time_match = re.match(r'(\d{2}:\d{2}:\d{2})', line)
                close_match = re.search(r'c=([\d.]+)', line)
                count_match = re.search(r'count=(\d+)', line)

                if time_match and close_match:
                    bars.append({
                        'time_et': time_match.group(1),
                        'close': float(close_match.group(1)),
                        'count': int(count_match.group(1)) if count_match else None,
                    })

    return pd.DataFrame(bars)


def check_strategy_signals(log_path):
    """Extract all strategy signals from log."""
    signals = {'V3': [], 'G': [], 'L': []}

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # V3 signals
            if 'V3 ENTRY' in line or 'V3 entry' in line:
                signals['V3'].append(line.strip())
            # G signals
            if 'G ENTRY' in line or 'G entry' in line:
                signals['G'].append(line.strip())
            # L signals
            if 'L ENTRY' in line or 'L entry' in line:
                signals['L'].append(line.strip())

    return signals


print("=" * 80)
print("DEEP INVESTIGATION: Backtest vs Live + Strategy Analysis")
print("=" * 80)

# Part 1: Timezone and bar timing issue
print("\n[1] TIMEZONE & BAR TIMING ANALYSIS")
print("-" * 80)

for ticker in TICKERS:
    print(f"\n{ticker}:")
    df_raw = load_alpaca_bars(ticker)

    if df_raw.empty:
        print("  No data")
        continue

    # Show timezone info
    print(f"  Total bars from Alpaca: {len(df_raw)}")
    print(f"  First bar UTC: {df_raw.iloc[0]['timestamp']}")
    print(f"  First bar ET:  {df_raw.iloc[0]['timestamp_et']} ({df_raw.iloc[0]['time_et']})")

    # Market hours only (9:30 - 16:00 ET)
    mh = df_raw[(df_raw['time_et'] >= dt_time(9, 30)) & (df_raw['time_et'] < dt_time(16, 0))]
    print(f"  Market hours (9:30-16:00 ET): {len(mh)} bars")

    if len(mh) > 0:
        print(f"    First: {mh.iloc[0]['timestamp_et']} - ${mh.iloc[0]['Close']:.2f}")
        print(f"    Last:  {mh.iloc[-1]['timestamp_et']} - ${mh.iloc[-1]['Close']:.2f}")

    # Live comparison
    live_bars = extract_live_2min_bars(ticker)
    print(f"  Live 2-min bars emitted: {len(live_bars)}")

    if len(mh) > 0:
        # Aggregate to 2-min
        df_2m = mh.set_index('timestamp_et')
        df_2m = df_2m.resample('2min', origin=datetime(2026, 7, 15, 9, 30, tzinfo=ET)).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        print(f"  Backtest 2-min aggregation: {len(df_2m)} bars")
        print(f"  Discrepancy: Backtest={len(df_2m)}, Live={len(live_bars)}, Diff={len(df_2m)-len(live_bars)}")

print("\n[2] STRATEGY SIGNAL ANALYSIS")
print("-" * 80)

signals = check_strategy_signals(LIVE_LOG)

print(f"\nV3 (Green candle momentum) signals: {len(signals['V3'])}")
for sig in signals['V3']:
    print(f"  {sig[:120]}")

print(f"\nG (Gap continuation) signals: {len(signals['G'])}")
if signals['G']:
    for sig in signals['G']:
        print(f"  {sig[:120]}")
else:
    print("  None - checking why...")

print(f"\nL (Long consolidation) signals: {len(signals['L'])}")
if signals['L']:
    for sig in signals['L']:
        print(f"  {sig[:120]}")
else:
    print("  None - checking why...")

# Check if G/L are even enabled
print("\n[3] STRATEGY CONFIGURATION")
print("-" * 80)

# Look for strategy filtering in log
with open(LIVE_LOG, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        if 'active_strategies' in line.lower() or 'enabled strategies' in line.lower():
            print(f"  {line.strip()}")
            break
    else:
        print("  No strategy configuration logged")

# Check params
print(f"\nStrategy parameters (from optuna_best_params_dynamic_2021_2026.json):")
g_params = {k: v for k, v in params.items() if k.startswith('g_')}
l_params = {k: v for k, v in params.items() if k.startswith('l_')}
v3_params = {k: v for k, v in params.items() if k.startswith('v3_')}

print(f"\nV3 params: {len(v3_params)} found")
for k in list(v3_params.keys())[:5]:
    print(f"  {k}: {v3_params[k]}")

print(f"\nG params: {len(g_params)} found")
if g_params:
    for k in list(g_params.keys())[:5]:
        print(f"  {k}: {g_params[k]}")
else:
    print("  ** NO G PARAMETERS! Strategy may not be enabled")

print(f"\nL params: {len(l_params)} found")
if l_params:
    for k in list(l_params.keys())[:5]:
        print(f"  {k}: {l_params[k]}")
else:
    print("  ** NO L PARAMETERS! Strategy may not be enabled")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("1. Check if backtest is using pre-market bars incorrectly")
print("2. Check why live received fewer bars than backtest")
print("3. Investigate if G/L strategies are disabled in live engine")
