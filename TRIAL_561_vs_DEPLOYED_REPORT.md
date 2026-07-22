# G+L ONLY COMPARATIVE ANALYSIS: TRIAL #561 vs DEPLOYED (#538)
## March 2026+ Out-Of-Sample (OOS) Period (60 Trading Days)

### Summary Comparison Table
| Metric | Trial #561 | Deployed (#538) |
| :--- | :---: | :---: |
| **Total PnL** | $590,765.94 | $1,877,848.49 |
| **Profit Factor (PF)** | 5.591 | 6.763 |
| **Win Rate (WR)** | 59.5% | 73.2% |
| **Total Trades** | 116 | 198 |
| **Green / Red / Flat Days** | 37 / 14 / 9 | 46 / 10 / 4 |
| **Avg. Daily PnL** | $9,846.10 | $31,297.47 |

### Strategy Performance Breakdown
#### Strategy G (Big Gap Runner)
| Parameter | Trial #561 | Deployed (#538) |
| :--- | :---: | :---: |
| **Min Gap %** | 15.0% | 10.0% |
| **Time Limit** | 6 min | 12 min |
| **Stop %** | 29.0% | 26.0% |
| **Target %** | 5.0% | 15.0% |
| **Trail Stop %** | 1.0% (at +10.0%) | 0.5% (no activation req) |
| **OOS PnL** | $542,266.54 | $1,288,037.41 |
| **Trades** | 114 | 125 |
| **PF** | 5.341 | 42.748 |
| **WR** | 59.6% | 77.6% |

#### Strategy L (Low Float Squeeze)
| Parameter | Trial #561 | Deployed (#538) |
| :--- | :---: | :---: |
| **Min Gap %** | 80.0% | 25.0% |
| **Max Float** | 5M | 25M |
| **Earliest Candle** | 24 | 30 |
| **Latest Candle** | 45 | 150 |
| **Stop %** | 24.0% | 18.0% |
| **Tier 1 Targets** | 45% / 70% | 40% / 30% |
| **Tier 2 Targets** | 24% / 50% | 30% / 55% |
| **Tier 3 Targets** | 5% / 15% | 25% / 40% |
| **Trail Stop %** | 1.0% (at +131.0%) | 1.0% (at +4.0%) |
| **OOS PnL** | $48,499.40 | $589,811.08 |
| **Trades** | 2 | 73 |
| **PF** | 13.845 | 3.000 |
| **WR** | 50.0% | 65.8% |

### Key Takeaways
1. **OOS Performance**: Trial #561 achieved an OOS PnL of **$590,765.94** compared to Deployed (#538) **$1,877,848.49**.
2. **Strategy G (Gappers)**: Trial #561's G strategy parameters were more restrictive on gaps but tighter on time limits, leading to different trade capture compared to #538.
3. **Strategy L (Low Float)**: Trial #561 uses a very restrictive `l_min_gap` (80.0%) and `l_max_float` (5M), leading to fewer but highly focused trades.
4. **Daily Average**: Trial #561 average daily PnL of **$9,846.10** compared to Deployed's **$31,297.47** over the 60-day OOS period.
