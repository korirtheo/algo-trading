# Regime detector forensic diagnosis — 29 Sep 2026

## Status

This document consolidates already-saved causal research outputs for the new regime detector. It is **not** a fresh raw-Parquet replay of REGIME-DETECTOR-V1: the current analysis runtime was unavailable while this document was prepared. No frozen strategy is modified.

Protected strategies:
- PM-AH3
- RTH-SHORT-FROZEN-2997

The detector remains an observation/diagnostic layer only.

## PM-AH3: strong evidence of a 2026 lifecycle/payoff shift

The frozen 2024/2025 PM-AH3 family averaged roughly 9% net per signal in development, while the exact preselected frozen policy produced +1.3669% mean and PF 1.2906 on the available 2026 OOS archive. The mechanism remained broadly positive across the preselected parameter basin, so this is compression rather than disappearance of the alpha.

Trajectory diagnostics:

| Period | Median MFE | Median MAE | Hit +18% | Target exit | Time exit | Mean net |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | +23.43% | -8.69% | 73.0% | 55.1% | 21.3% | +9.47% |
| 2025 | +27.75% | -9.06% | 81.8% | 59.6% | 11.8% | +8.92% |
| 2026 Jan-Apr | +27.14% | -11.40% | 65.9% | 36.6% | 24.4% | +1.24% |
| 2026 May-Aug | +7.94% | -6.60% | 27.8% | 14.8% | 68.5% | +1.47% |

Interpretation:
- Jan-Apr 2026 retained large upside excursion but experienced worse adverse excursion and more failures before realization.
- May-Aug 2026 is the clearer structural break: the second-wave magnitude compressed sharply.

The 2026 diagnostic basin shifted mostly in lifecycle parameters rather than the model score threshold:

| Parameter | 2024/25 basin median | 2026 basin median |
|---|---:|---:|
| score quantile | 0.868 | 0.874 |
| minutes since activation | 62 | 109 |
| reset depth | 19.63% | 26.02% |
| target | 18% | 28% |
| max hold | 165m | 190m |
| trail arm | 14% | 23% |
| trail | 2.5% | 6.75% |

A signal-state-only classifier distinguishing 2026 from 2024/25 had ROC AUC 0.604. Therefore pre-entry state alone is too weak to be the regime switch.

### PM causal strategy-health timeline

Historical 20-trade rolling PM-AH3 distributions:
- 10th-percentile rolling mean: +3.39%
- 5th-percentile rolling mean: +1.79%
- 10th-percentile target-hit rate: 40%

2026:
- by 18 Feb, the first 20 completed trades had rolling mean -1.60% and target-hit rate 35%;
- by 20 Feb, three consecutive 20-trade windows were below the historical 5th-percentile mean;
- a stricter five-window confirmation did not occur until roughly 14-15 May.

Detector interpretation:
- WATCH: around 18-20 Feb 2026.
- persistent/high-confidence strategy-health SHIFT: around 14-15 May 2026.
- the May shift lines up with the later compressed-MFE regime.

These dates are historical diagnostics for validating detector behavior. They must not be retuned into thresholds and then described as untouched 2026 OOS.

## RTH short: evidence points to a later/more mature backside state

2023 untouched replay of RTH-SHORT-FROZEN-2997 produced only 12 selected trades, mean +8.55%, PF 3.42. It is useful as a stress observation but too small to define the baseline distribution by itself.

The strongest powered baseline remains 2024/2025. The 10,000-trial 2024/2025 development study's best family had 549 trades (233 in 2024, 316 in 2025), PF 3.388.

Comparing that 2024/2025 development basin with the later 2026-refined #2997 configuration is **diagnostic only**, because 2026 was already exposed when #2997 was selected. The parameter movement nevertheless tells us what changed in the profitable fade geometry:

- earliest entry moved from about 09:41 ET to about 10:16 ET;
- CORE minimum day-fade moved from ~3.7% to ~12.1%;
- CORE minimum fade moved from ~1.5% to ~3.8%;
- CORE time since peak moved 5 -> 10 minutes;
- CORE minimum red-volume share moved ~0.206 -> ~0.292;
- SAT time since peak moved 5 -> 45 minutes;
- the eligible window extended much later, to roughly 14:31 ET.

#2997's authoritative 2026 refinement surface had 234 trades, +4.569% mean, PF 2.092, 67.5% win rate, and positive robustness across its weakest two-month block. A separate provisional local-search family also remained positive in Jan-Feb, Mar-Apr, May-Jun and Jul-Aug. This supports the hypothesis that the RTH fade mechanism persisted but the profitable state became more mature/confirmed rather than early top-picking.

That is not proof that these exact 2026-selected parameters will generalize. For the detector, the usable lesson is the direction of structural movement: peak age, realized drawdown from HOD, red-volume confirmation, VWAP state, and time-of-day should be monitored as regime variables.

## 2023 handling

2023 should receive low weight in regime calibration. The exact #2997 replay has only 12 trades, and earlier universe research also flagged unusually sparse/suspect coverage. It is useful as a sparse-regime stress test, not as a normal-regime anchor.

## Detector architecture implied by the evidence

Keep three independent evidence layers:

1. MARKET STATE — strategy-agnostic point-in-time PM/OPEN/RTH distribution shift.
2. STRATEGY HEALTH — completed-trade-only rolling return/PF/hit-rate/MFE/MAE drift, separately for PM-AH3, RTH CORE and RTH SAT.
3. PARAMETER-BASIN FORENSICS — research-only explanation of what kind of behavior changed. Never a live trigger and never allowed to leak future-selected parameters into earlier detector states.

A regime alert should be strongest when market-state drift and completed-trade health deterioration persist together. Market-state-only evidence should normally remain WATCH unless it is extreme and persistent.

## Current conclusion

- PM-AH3: 2026 contains a real lifecycle/payoff regime change; first causal health warning appears by late February, and a stricter persistent break is visible around mid-May. Alpha is compressed, not absent.
- RTH shorts: the fade alpha appears to persist, but profitable entry geometry moved toward later, more confirmed backside states. The 2026-refined parameters are explanatory evidence only.
- Frozen strategies remain untouched. The regime detector should initially warn and log; automatic parameter switching requires future untouched validation.
