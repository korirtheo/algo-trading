# Premarket Gap-Up Breakout Strategy

An automated backtesting system for trading small-cap stocks that gap up in premarket. The strategy identifies stocks gapping above their previous close, waits for a confirmed breakout above the premarket high, enters on a pullback, and exits through a multi-tranche profit-taking system with ATR-based trailing stops.

## Strategy Overview

### Entry Logic (3-Phase)
1. **Breakout Confirmation** -- Price closes above the premarket high for 2 out of 4 candles
2. **Pullback Wait** -- Wait for price to pull back toward the premarket high (within 4%)
3. **Bounce Entry** -- Enter when price closes back above premarket high after the pullback
4. If no pullback within 24 candles, enter on the next close above premarket high (timeout)

### Exit Logic (3-Tranche)
1. **Tranche 1** -- Sell 90% of position at +15% profit (uses original entry price)
2. **Tranche 2** -- Sell 35% of remaining at +25% profit
3. **Trail Stop** -- Trail the rest using ATR(8) x 4.25 multiplier
4. **Hard Stop** -- 16% stop loss on entire position (uses original entry price pre-T1, scale-in adjusted post-T1)
5. **EOD Exit** -- Sell remaining shares 30 minutes before close
6. **EOD Close** -- Any leftover position is closed at 3:59 PM

### Filters
- **Gap filter** -- Only trade stocks gapping >2% above previous close
- **Volume filter** -- Minimum 250,000 shares traded in premarket
- **Volume cap** -- Position size capped at 5% of the stock's dollar volume up to entry time (prevents unrealistic fills in thin stocks)
- **Regime filter** -- Skip trading when SPY is below its 40-day SMA
- **Warrant/unit filter** -- Exclude warrants, units, and rights

### Two-Pass Simulation Architecture

Each timestamp is processed in two passes:
1. **Pass 1 (Exits + Scale-in + Signals)** -- Process all open positions (stop loss, trailing stop, partial sells, scale-in execution) to free cash, then detect entry signals for non-entered stocks
2. **Pass 2 (Allocate)** -- Collect all entry candidates, rank by PM dollar volume (best liquidity first), and allocate capital

This structure ensures that cash freed by exits in Pass 1 is available for new entries in Pass 2 (subject to settlement rules). Scale-in executes inline in Pass 1 before partial sells, so a +14% scale-in fires before the +15% T1 sell on the same candle.

### Cash Account / Margin Account (T+1 Settlement)

The backtest models realistic PDT account rules:
- **Under $25,000 (cash account)** -- Sell proceeds settle T+1. Cash from today's sells is not available for new trades until the next trading day.
- **At/above $25,000 (margin account)** -- Sell proceeds are available instantly. Cash freed by partial sells or exits can be reused for new entries the same day.

### Position Sizing
- **50% flat** -- 50% of available cash per trade
- **Scale-in** -- Add 50% of original position size when price rises +14% above entry
- Scale-in uses correct remaining-cost accounting: `remaining_cost = remaining_shares * entry_price`
- Cash-constrained: never trades more than available cash

## Current Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| STOP_LOSS_PCT | 16.0% | Hard stop loss |
| PARTIAL_SELL_FRAC | 0.90 | Tranche 1: sell 90% of shares |
| PARTIAL_SELL_PCT | 15.0% | Tranche 1: at +15% profit |
| PARTIAL_SELL_FRAC_2 | 0.35 | Tranche 2: sell 35% of remaining |
| PARTIAL_SELL_PCT_2 | 25.0% | Tranche 2: at +25% profit |
| ATR_PERIOD | 8 | ATR lookback for trailing stop |
| ATR_MULTIPLIER | 4.25 | Trail distance = ATR x 4.25 |
| CONFIRM_ABOVE | 2 | Candles above PM high needed |
| CONFIRM_WINDOW | 4 | Window to check confirmation |
| PULLBACK_PCT | 4.0% | How close to PM high for pullback |
| PULLBACK_TIMEOUT | 24 | Candles before timeout entry |
| TRADE_PCT | 0.50 | 50% of account per trade |
| MIN_PM_VOLUME | 250,000 | Minimum premarket volume |
| SCALE_IN | 1 | Scale-in enabled |
| SCALE_IN_TRIGGER_PCT | 14.0% | Add to position at +14% |
| SCALE_IN_FRAC | 0.50 | Add 50% of original position |
| VOL_CAP_PCT | 5.0% | Max position = 5% of dollar volume |
| EOD_EXIT_MINUTES | 30 | Start selling 30 min before close |
| SPY_SMA_PERIOD | 40 | Regime filter: SPY > SMA(40) |
| MARGIN_THRESHOLD | $25,000 | Cash account below, margin above |

## Backtest Results (Jan 2025 - Feb 2026)

**Config: $25,000 starting capital, 50% flat sizing, SMA(40) regime filter**

| Metric | Value |
|--------|-------|
| Starting Capital | $25,000 |
| Ending Capital | $235,821 |
| Total Return | +843% |
| Trading Days | 284 |
| Total Trades | 420 |
| Win Rate | 54.5% |
| Green Days | 113/284 |
| Best Day | +$41,938 |
| Worst Day | -$55,669 |
| Scale-ins | 90 |
| Regime Skipped | 73 days |
| Cash Account Days | 47 (T+1) |
| Margin Unlocked | 2025-01-24 |

### Monthly Progression
| Month | End Balance | Notes |
|-------|------------|-------|
| Jan 2025 | $26,625 | Modest gain, margin unlocked Jan 24 |
| Feb 2025 | $17,569 | Drawdown (-30%), regime skips start |
| Mar 2025 | $17,569 | All regime skipped (SPY < SMA40) |
| Apr 2025 | $17,569 | All regime skipped |
| May 2025 | $23,301 | Recovery begins |
| Jun 2025 | $29,923 | Back above starting capital |
| Jul 2025 | $28,137 | Flat |
| Aug 2025 | $42,210 | Breakout month |
| Sep 2025 | $81,239 | Compounding accelerates |
| Oct 2025 | $154,798 | Strong momentum |
| Nov 2025 | $137,206 | Drawdown |
| Dec 2025 | $291,351 | Best month |
| Jan 2026 | $190,890 | Drawdown |
| Feb 2026 | $235,821 | Recovery |

### Stress Tests (flat $10K/day)
| Test | P&L | Sharpe | Verdict |
|------|-----|--------|---------|
| Baseline (0.05% slip) | +$10,521 | 1.36 | PASS |
| Remove top 10% days | -$13,690 | -2.46 | FAIL |
| 0.2% slippage | +$4,179 | 0.55 | PASS |
| 0.5% slippage (extreme) | -$3,138 | -0.41 | FAIL |
| Kelly > 0 | -15.6% | -- | FAIL |
| Monte Carlo (>80% prof.) | 100% | -- | PASS |

## Project Structure

### Core Scripts

| File | Purpose |
|------|---------|
| `test_full.py` | Full backtest with rolling cash, T+1 settlement, charts, and regime filter. The main script. Use `--no-charts` to skip chart generation. |
| `optimize.py` | Optuna parameter optimizer (15 params, 500 trials). Uses flat $10K daily cash. |
| `download_polygon.py` | Downloads intraday (2-min) and daily data from Polygon.io API. |
| `regime_filters.py` | Regime filter module (SPY SMA, VIX, adaptive sizing). Used by test_full.py and optimize.py. |

### Analysis & Testing Scripts

| File | Purpose |
|------|---------|
| `analyze_trades.py` | Per-trade analysis (exit types, MFE/MAE, volume buckets, time analysis) |
| `pattern_analysis.py` | ML feature analysis (volume profiles, VWAP, opening range) |
| `stress_test.py` | Robustness tests (remove top 10% days, slippage, Kelly criterion, Monte Carlo) |
| `stress_test_regime.py` | Regime filter comparison (VIX, SPY trend, sensitivity tests) |
| `volume_analysis.py` | Volume pattern analysis |

### Debugging & Verification Scripts

| File | Purpose |
|------|---------|
| `verify_root_cause2.py` | Old two-pass simulation recreation for comparison |
| `test_initial_commit.py` | Runs the exact initial commit logic for baseline comparison |
| `compare_jan22.py` | Compares Jan 22 trades between old and new code paths |

### Data Scripts

| File | Purpose |
|------|---------|
| `generate_daily_gainers.py` | Generates daily_top_gainers.csv from raw data |
| `validate_data.py` | Validates data completeness across all months |
| `resume_download.py` | Resumes interrupted Polygon downloads |
| `patch_aug_sep.py` | Patches missing Aug-Sep data |
| `patch_oct_nov.py` | Patches missing Oct-Nov data |

### Data Directories

| Directory | Period | Tickers |
|-----------|--------|---------|
| `stored_data_jan_mar_2025/` | Jan - Mar 2025 | 756 |
| `stored_data_apr_jun_2025/` | Apr - Jun 2025 | 786 |
| `stored_data_jul_2025/` | Jul 2025 | 306 |
| `stored_data_oos/` | Aug - Dec 2025 | 1,410 |
| `stored_data/` | Jan - Feb 2026 | 544 |
| `stored_data_combined/` | Jan 2025 - Feb 2026 (all merged) | 2,460 |
| `regime_data/` | Cached SPY and VIX daily data (yfinance) | -- |
| `charts/` | Generated backtest charts (timestamped folders) | -- |

Each data directory contains:
- `daily_top_gainers.csv` -- All gap-up candidates with dates and gap percentages
- `intraday/` -- 2-minute candle CSVs per ticker
- `daily/` -- Daily OHLCV CSVs per ticker
- Pickle caches (`fulltest_picks_v3.pkl`, `optimize_picks_novol.pkl`)

### Chart Versioning

Each backtest run saves charts to a timestamped folder: `charts/run_YYYYMMDD_HHMMSS/`. This allows side-by-side comparison of different configurations. Each summary chart includes a config details panel showing the exact settings used.

## Usage

### Run the full backtest
```bash
# All data directories (default: 5 dirs, Jan 2025 - Feb 2026)
python test_full.py

# Skip chart generation (faster, console output only)
python test_full.py --no-charts

# Specific directories
python test_full.py stored_data_combined
```

### Run the optimizer
```bash
# Optimize on combined data (default)
python optimize.py

# Optimize on specific data
python optimize.py stored_data_oos
```

### Download data from Polygon.io
```bash
# Download to stored_data_oos/ (default)
python download_polygon.py YOUR_API_KEY 2025-08-01 2025-12-31

# Download to a different directory
POLYGON_OUT_DIR=stored_data python download_polygon.py YOUR_API_KEY 2026-01-01 2026-02-28
```
Note: Free Polygon plan allows 5 API calls/minute. Downloading 760 candidates takes ~2.5 hours.

### Run stress tests
```bash
python stress_test.py
python stress_test_regime.py
```

## Known Limitations

- **Early month weakness**: Feb 2025 drawdown (-30%) due to 50% sizing hitting 16% stops on a small account. Strategy needs several months to build compounding base.
- **Regime filter dead zones**: SMA(40) blocked all of Mar-Apr 2025 (44 consecutive skip days). Correct behavior (tariff crash) but prevents any recovery during that period.
- **Stress test failures**: "Remove top 10% days" and "0.5% slippage" tests fail, indicating dependence on outlier wins and sensitivity to execution costs.
- **Negative Kelly**: Kelly criterion is negative, suggesting the edge per trade is thin. Profits come from compounding a slight edge over many trades.
- **Slippage underestimate**: 0.05% slippage is optimistic for illiquid small-cap stocks. Real spreads can be 1-5%.
- **No commission modeling**: Does not account for per-trade commissions.
- **Liquidity constraints**: As the account compounds, position sizes can exceed realistic fill capacity. The 5% volume cap helps but may not be sufficient for very large accounts.
- **Compounding amplification**: 50% position sizing produces extreme returns that would not replicate with proper risk management.
- **Single strategy risk**: No diversification across uncorrelated strategies.

## Dependencies

```
pandas
numpy
matplotlib
yfinance
optuna
requests
```
