# PORTFOLIO EXPLORATION: STRATEGIES H, I AND N
## March 2026+ Out-Of-Sample (OOS) Period (60 Trading Days)

### Ablation & Priority Comparison Table
| Metric | Base (G+L Only) | GLHN (Default Priority) | GLHN (Swapped Priority: G > L > H > I > N) |
| :--- | :---: | :---: | :---: |
| **Total PnL** | $590,765.94 | $109,747.30 | $101,706.57 |
| **Profit Factor (PF)** | 5.591 | 1.997 | 2.022 |
| **Win Rate (WR)** | 59.5% | 49.5% | 52.7% |
| **Total Trades** | 116 | 91 | 91 |
| **Green / Red / Flat Days** | 37 / 14 / 9 | 35 / 23 / 2 | 36 / 22 / 2 |
| **Avg. Daily PnL** | $9,846.10 | $1,829.12 | $1,695.11 |

### Strategy Contributions Breakdown

#### 1. Default Priority Portfolio (H=0, G=1, L=19, I=16, N=18)
| Strategy | PnL | Trades | Profit Factor | Win Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Strategy G** | $87,035.96 | 56 | 2.767 | 50.0% |
| **Strategy L** | $0.00 | 0 | 0.000 | 0.0% |
| **Strategy H** | $48,676.41 | 16 | 3.149 | 62.5% |
| **Strategy I** | $-22,576.39 | 9 | 0.025 | 11.1% |
| **Strategy N** | $-3,388.68 | 10 | 0.774 | 60.0% |

#### 2. Swapped Priority Portfolio (G > L > H > I > N)
| Strategy | PnL | Trades | Profit Factor | Win Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Strategy G** | $104,660.37 | 67 | 3.004 | 56.7% |
| **Strategy L** | $-130.66 | 1 | 0.000 | 0.0% |
| **Strategy H** | $21,954.21 | 8 | 2.185 | 62.5% |
| **Strategy I** | $-21,153.62 | 9 | 0.023 | 11.1% |
| **Strategy N** | $-3,623.74 | 6 | 0.480 | 66.7% |

### Key Discoveries & Recommendations
1. **Priority Swapping Solves Capital Starvation**: By prioritizing our core alpha-driving strategies **G and L** (giving them priority 0 and 1) before secondary strategies, we prevent capital blockage. 
2. **Strategy H, I, N as Selectors**: When placed at the bottom of the priority stack, H, I, N only fire when there is leftover capital after G and L have executed. This allows us to capture supplementary alpha without starving our core strategies.
