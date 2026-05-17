# Premarket Gap-Up Breakout Strategy — Implementation Plan

## Strategy Summary
1. Each trading day, find the **top 5 premarket percentage gainers**
2. At market open, if price breaks **above the premarket high** → BUY
3. Take partial profits on the way up during the first hour
4. Sell all remaining at **10:30 AM ET** (1 hour after open)
5. Backtest over the past year, display results

## Data Reality & Approach

### Problem
- Free historical **premarket data** (4AM–9:30AM candles) doesn't exist for a full year
- yfinance 1-minute data only goes back ~60 days; hourly goes ~730 days

### Solution: Two-tier approximation using daily + hourly data

| Concept | Approximation |
|---|---|
| **Premarket gainer %** | Gap-up % = `(Open - PrevClose) / PrevClose` |
| **Premarket high** | `Open price` (the open IS the settlement of premarket; any move above it = breakout) |
| **First-hour price action** | Use `1h` interval data from yfinance (the 9:30–10:30 candle) |
| **Buy trigger** | First-hour High > Open (price broke above premarket high) |
| **Entry price** | Open + small buffer (0.1% above open, simulating breakout entry) |
| **Profit-taking** | Scale out: sell 33% at +1.5%, 33% at +3%, hold rest until 10:30 |
| **Final exit** | Close of the first hourly candle (≈10:30 AM) |

## File: `premarket_gap_strategy.py`

### Structure
1. **Config** — dates, thresholds, ticker universe
2. **get_sp500_tickers()** — pull S&P 500 list for broad universe
3. **download_daily_data()** — 1 year of daily OHLC for all tickers
4. **identify_gap_ups()** — for each day, compute gap %, rank, pick top 5
5. **download_hourly_data()** — fetch 1h candles for the top-gainer tickers
6. **simulate_trades()** — for each day's top 5:
   - Check if first-hour High > Open (breakout confirmed)
   - Entry at Open + 0.1%
   - Partial exits at +1.5% and +3%
   - Final exit at first-hour Close
7. **calculate_results()** — P&L, win rate, avg return, max drawdown
8. **display_results()** — print summary table + matplotlib charts:
   - Equity curve
   - Win/loss distribution
   - Monthly returns heatmap
   - Per-trade scatter plot

### Key Design Decisions
- **Universe**: S&P 500 (~500 tickers) — liquid enough, broad enough
- **Min gap threshold**: Only consider stocks gapping up ≥2% to filter noise
- **Position sizing**: Equal weight across the top 5 each day ($10k per trade, $50k total daily exposure)
- **Slippage**: 0.05% added to entries, subtracted from exits
- **Commission**: $0 (modern brokers)

### Output
- Console: summary stats table
- Charts: equity curve, monthly returns, trade distribution
- CSV: full trade log exported
