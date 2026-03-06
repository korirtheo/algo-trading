# Combined Green Candle Breakout Strategy (H+G+A+F+P)

A multi-strategy backtesting system for day-trading small-cap gap-up stocks. Five complementary strategies share a single capital pool, each targeting different gap sizes and profit profiles. The system uses 100% balance sizing with a priority cascade (H > G > A > F > P) to allocate capital to the highest-conviction setup each day.

## Strategy Overview

### Strategies at a Glance

| Strategy | Gap Req | Entry Signal | Target | Time Limit | Exit Method |
|----------|---------|-------------|--------|------------|-------------|
| **H** (High Conviction) | >= 35% | 2nd green + new hi + vol confirm, body >= 4% | +16% | 15m | Target or time stop |
| **G** (Big Gap Runner) | >= 30% | 2nd green + new hi | +11% | 10m | Target or time stop |
| **A** (Quick Scalp) | >= 15% | 2nd green + new hi, body >= 4% | +6% | 12m | Target or time stop |
| **F** (Catch-All) | >= 10% | 2nd green | +8% | 3m | Target or time stop |
| **P** (PM High Breakout) | >= 10% | PM high breakout + pullback + bounce | +9%/+13% | 40m | Partial sell + trailing stop |

### How Strategies Interact

1. On candle 2, each stock is classified into H, G, A, or F based on gap size and candle pattern
2. If no H/G/A/F signal fires, the stock becomes P-eligible (PM high breakout tracking)
3. A stock can only run ONE strategy -- mutual exclusivity is enforced
4. Capital allocation: 100% of available balance goes to the highest-priority signal

### Strategy H (High Conviction)
- **Gap**: >= 35%
- **Entry**: Candle 2 must be green with body >= 4%, close > candle 1 high, volume > candle 1 volume
- **Exit**: +16% target or 15-minute time stop
- **Rationale**: Biggest gaps with strong momentum continuation have the highest upside

### Strategy G (Big Gap Runner)
- **Gap**: >= 30%
- **Entry**: Candle 2 must be green, close > candle 1 high
- **Exit**: +11% target or 10-minute time stop
- **Rationale**: Large gaps that show immediate follow-through

### Strategy A (Quick Scalp)
- **Gap**: >= 15%
- **Entry**: Candle 2 must be green with body >= 4%, close > candle 1 high
- **Exit**: +6% target or 12-minute time stop
- **Rationale**: Medium gaps with strong candle quality

### Strategy F (Catch-All)
- **Gap**: >= 10%
- **Entry**: Candle 2 must be green
- **Exit**: +8% target or 3-minute time stop
- **Rationale**: Catches broad momentum with tight time limit to control risk

### Strategy P (PM High Breakout) -- Optuna v2 Advanced
- **Gap**: >= 10%
- **Entry**: 3-phase process:
  1. **Breakout confirmation**: 4 out of 6 candles close above premarket high
  2. **Pullback**: Price pulls back to within 7% of PM high (or timeout after 30 candles)
  3. **Bounce**: Price closes back above PM high
- **Exit**: Multi-level with trailing stop:
  1. **Hard stop**: -12% from entry (active until trail activates)
  2. **Trail activation**: After +2% unrealized gain, trailing stop activates
  3. **Trailing stop**: Highest price since entry minus 2% (locks in gains)
  4. **Partial sell**: Sell 25% of position at +9% (cash returned to pool)
  5. **Runner target**: Sell remaining 75% at +13%
  6. **Time stop**: Close at 40 minutes if still open
- **Max entry candle**: 75 (no entries after ~2.5 hours)
- **Rationale**: Captures stocks that reject P for H/G/A/F and later break out above their PM high with sustained momentum

## Shared Filters & Rules

- **Volume filter**: Minimum 250,000 shares traded in premarket
- **Volume cap**: Position size capped at 5% of the stock's dollar volume up to entry time
- **Regime filter**: Skip trading when SPY is below its 40-day SMA
- **Warrant/unit filter**: Exclude warrants, units, and rights
- **Slippage**: 0.05% applied to all entries and exits
- **EOD close**: All positions closed at end of day

### Position Sizing
- **100% of available balance** per trade (FULL_BALANCE_SIZING)
- Only 1 position at a time (unless cash is freed by an exit and re-allocated)
- Cash-constrained: never trades more than available cash

### T+1 Settlement (Cash/Margin Account)
- **Under $25,000**: Cash account -- sell proceeds settle T+1 (not available until next day)
- **At/above $25,000**: Margin account -- sell proceeds available instantly

## Backtest Results (Jan 2024 - Feb 2026)

**Config: $25,000 starting capital, 100% balance sizing, SMA(40) regime filter**

| Metric | Value |
|--------|-------|
| Starting Capital | $25,000 |
| Ending Capital | $36,498,784 |
| Total Return | +145,895% |
| Trading Days | 530 |
| Total Trades | 1,003 |
| Win Rate | 65.7% |
| Profit Factor | 2.59 |
| Sharpe (ann.) | 4.61 |
| Green Days | 329/443 (74.3%) |
| Best Day | +$2,292,938 |
| Worst Day | -$1,423,424 |

### Per-Strategy Breakdown

| Strategy | Trades | Win Rate | Total PnL | Avg Win | Avg Loss |
|----------|--------|----------|-----------|---------|----------|
| H (High Conviction) | 141 | 69.5% | +$7,389,903 | $102,493 | -$61,731 |
| G (Big Gap Runner) | 198 | 65.2% | +$5,349,370 | $72,796 | -$58,570 |
| A (Quick Scalp) | 98 | 68.4% | +$810,794 | $33,604 | -$46,474 |
| F (Catch-All) | 332 | 54.2% | +$758,197 | $26,008 | -$25,811 |
| P (PM High Breakout) | 234 | 79.1% | +$22,165,519 | $178,772 | -$222,598 |

### Exit Reason Distribution

| Exit Reason | Count | % |
|-------------|-------|---|
| F_TIME_STOP | 266 | 26.5% |
| P_TRAIL | 168 | 16.7% |
| G_TIME_STOP | 116 | 11.6% |
| G_TARGET | 82 | 8.2% |
| H_TARGET | 77 | 7.7% |
| F_TARGET | 66 | 6.6% |
| H_TIME_STOP | 64 | 6.4% |
| A_TARGET | 59 | 5.9% |
| P_TIME_STOP | 39 | 3.9% |
| A_TIME_STOP | 39 | 3.9% |
| P_STOP | 16 | 1.6% |
| P_TARGET | 11 | 1.1% |

### Stress Tests

| Test | Result | Verdict |
|------|--------|---------|
| Max Drawdown | 25.5% | PASS |
| Max Losing Streak | 4 days | PASS |
| Remove Top 10% Days | +$6,558,980 (Sharpe 1.51) | PASS |
| Kelly Criterion | 51.6% | PASS |
| Monte Carlo (>80% prof.) | 100% | PASS |
| Vol-Capped Trades | 829 | -- |
| **OVERALL** | **ALL PASS** | |

## Project Structure

### Core Scripts

| File | Purpose |
|------|---------|
| `test_green_candle_combined.py` | **Main backtest** -- Combined H+G+A+F+P strategy with rolling cash, T+1 settlement, charts, and regime filter. Use `--no-charts` to skip chart generation. |
| `test_full.py` | Legacy single-strategy backtest (PM high breakout only). |
| `optimize.py` | Optuna parameter optimizer for test_full.py (15 params, 500 trials). |
| `optimize_p_advanced.py` | Optuna optimizer for P strategy with trailing stop, partial selling, and scale-in. |
| `download_polygon.py` | Downloads intraday (2-min) and daily data from Polygon.io API. |
| `regime_filters.py` | Regime filter module (SPY SMA, VIX, adaptive sizing). |

### Analysis & Testing Scripts

| File | Purpose |
|------|---------|
| `analyze_trades.py` | Per-trade analysis (exit types, MFE/MAE, volume buckets, time analysis) |
| `pattern_analysis.py` | ML feature analysis (volume profiles, VWAP, opening range) |
| `stress_test.py` | Robustness tests (remove top 10% days, slippage, Kelly criterion, Monte Carlo) |
| `stress_test_regime.py` | Regime filter comparison (VIX, SPY trend, sensitivity tests) |

### Data Directories

| Directory | Period | Tickers |
|-----------|--------|---------|
| `stored_data_jan_feb_2024/` | Jan - Feb 2024 | 2,026 (daily only, no intraday) |
| `stored_data_jan_mar_2024/` | Jan - Mar 2024 | 300 |
| `stored_data_apr_jun_2024/` | Apr - Jun 2024 | ~780 |
| `stored_data_jul_sep_2024/` | Jul - Sep 2024 | ~600 |
| `stored_data_oct_dec_2024/` | Oct - Dec 2024 | ~700 |
| `stored_data_combined/` | Jan 2025 - Feb 2026 (all merged) | 2,460 |
| `regime_data/` | Cached SPY and VIX daily data (yfinance) | -- |
| `charts/` | Generated backtest charts (timestamped folders) | -- |

Each data directory contains:
- `daily_top_gainers.csv` -- All gap-up candidates with dates and gap percentages
- `intraday/` -- 2-minute candle CSVs per ticker
- `daily/` -- Daily OHLCV CSVs per ticker
- Pickle caches for faster reload

### Chart Versioning

Each backtest run saves charts to a timestamped folder: `charts/run_YYYYMMDD_HHMMSS/`. Each summary chart includes a config details panel showing the exact settings used.

## Usage

### Run the combined backtest
```bash
# All data directories (Jan 2024 - Feb 2026)
python test_green_candle_combined.py stored_data_jan_mar_2024 stored_data_apr_jun_2024 stored_data_jul_sep_2024 stored_data_oct_dec_2024 stored_data_combined

# Skip chart generation (faster, console output only)
python test_green_candle_combined.py --no-charts stored_data_combined

# Single data directory
python test_green_candle_combined.py stored_data_combined
```

### Run the P optimizer
```bash
# Optimize P with trailing stop + partial selling
python optimize_p_advanced.py
```

### Download data from Polygon.io
```bash
# Download to stored_data_oos/ (default)
python download_polygon.py YOUR_API_KEY 2025-08-01 2025-12-31

# Download to a different directory
POLYGON_OUT_DIR=stored_data python download_polygon.py YOUR_API_KEY 2026-01-01 2026-02-28
```
Note: Free Polygon plan allows 5 API calls/minute. Downloading 760 candidates takes ~2.5 hours. Free plan has a 2-year rolling window for intraday data.

## Known Limitations

- **Compounding amplification**: 100% position sizing produces extreme returns ($25K to $36.5M) that would not replicate at scale due to liquidity constraints
- **Volume cap partially mitigates**: 829/1003 trades were volume-capped, but late-stage position sizes still exceed realistic fill capacity
- **Slippage underestimate**: 0.05% slippage is optimistic for illiquid small-cap stocks. Real spreads can be 1-5%
- **No commission modeling**: Does not account for per-trade commissions
- **P strategy dominance**: P generates 61% of total PnL ($22.2M of $36.5M). Strategy is concentrated in one pattern
- **Regime filter dead zones**: SMA(40) blocks trading during prolonged downturns (tariff crash Mar-Apr 2025). Correct behavior but prevents recovery
- **Survivorship bias risk**: Data may not include delisted or halted stocks
- **Jan-Feb 2024 intraday gap**: Polygon.io free plan 2-year rolling window prevents downloading intraday data before ~Mar 2024

## Dependencies

```
pandas
numpy
matplotlib
yfinance
optuna
requests
```
