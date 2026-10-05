# Corrected G/A Resurrection — Findings

Run: 37239477312
Commit under test: 702a1517776fd6dbfbe6b41902488f16fa3e7109
Trials: 700 per strategy
Selection: 2025 only
Validation: untouched 2026 archive through 2026-02-27
Execution: candle-2 confirmation -> next-bar open; current dynamic slippage/liquidity caps

## G — Big Gap Runner
Best development-selected parameters:
- min gap 30%
- min first-candle body 0%
- candle-2 volume >= 0.75x candle-1 volume
- target 12%
- stop 10%
- trail 1%, activates +3%
- time exit 60 min

2025 development:
- 172 trades
- PnL +$5,697.87
- PF 1.1127
- WR 66.28%
- mean +0.5063%
- median +3.1604%
- max DD -22.77%

Untouched 2026:
- 27 trades
- PnL +$2,793.10
- PF 1.3996
- WR 70.37%
- mean +1.4390%
- median +3.1841%
- max DD -8.98%

Interpretation: corrected G retains a forward-positive edge, unlike many legacy strategies,
but its mean/median economics are below the desired high-payoff threshold. Keep as a
surviving research candidate, not a promoted/frozen strategy.

## A — Quick Scalp
Best development-selected parameters:
- min gap 15%
- min first-candle body 2%
- candle-2 volume >= 1.5x candle-1 volume
- target 16%
- stop 10%
- trail 1%, activates +3%
- time exit 40 min

2025 development:
- 98 trades
- PnL +$6,785.62
- PF 1.2937
- WR 68.37%
- mean +0.9122%
- median +2.9821%
- max DD -15.85%

Untouched 2026:
- 14 trades
- PnL -$1,978.76
- PF 0.5525
- WR 64.29%
- mean -1.8650%
- median +1.8089%
- max DD -11.66%

Interpretation: A does not survive forward validation. Reject as currently defined.

High-payoff validation gate count across top 30 development candidates:
- G: 0
- A: 0
