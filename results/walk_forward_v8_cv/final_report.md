# Autonomous multi-round Optuna orchestrator — full report

Generated: 2026-06-20T04:05:56.953103


## What this run did and why

Background: W7's empirical analysis (top 30 trials × 5 years backtested with fresh $25K each year) showed the legacy objective `total_pnl × min(pf, 3)` correlates only 0.281 (Pearson) with 2026 forward PnL. Per-year `min_pf` correlates 0.638 — over twice as informative. Compounding across training years was masking 2023's regime-fragility signal from TPE.

The fix tested here: run the backtest in CV mode — reset cash to $25K at each year boundary so TPE sees per-year performance directly. Score by the WORST year's PnL × PF, forcing the optimizer to find configs that survive every regime, not just ace 2024-25.

The orchestrator runs up to 3 rounds with different objective variants. If the default CV objective produces a 'boring strategy' (small trade counts hugging the floor) or a 'minPF-overfit' trial that wins train but flops forward, it auto-launches the next variant designed to fix that specific failure mode.

Baseline to beat: **W7 #254** had forward 2026 PnL = $199,553 (8.98× starting cap). For context: the deployed **#124** baseline is +$22,534 (1.90×).

## TL;DR

- **Best round**: W8 (variant: `default`)
  - Winner trial: **#1550**
  - Forward 2026 PnL: **$+36,668** (2.47×)
  - Max DD: -63.3%   Trades: 362
  - vs W7 #254 baseline: $-162,885
  - vs #124 deployed:    $+14,134
  - Verdict: **OVERFIT_GOODHART**

## Comparison vs known baselines

| Config | 2026 forward PnL | Multi | Trades | Note |
|---|---|---|---|---|
| #124 W3 deployed | +$22,534 | 1.90× | 196 | Currently live on $26K PKIPX |
| #254 W7 (legacy obj) | +$199,553 | 8.98× | 215 | Forward winner of W7 — but no objective signal, just lucky |
| **W8 winner #1550** | $+36,668 | 2.47× | 362 | variant: default |
| **W9 winner #611** | $+2,882 | 1.12× | 353 | variant: geomean_year_score |

## Round W8 — variant `default`

**Objective formula**:
  score = min(year_pnl × min(year_pf, 3.0)) across [2022,2023,2024,2025]. Each year backtested independently with fresh $25K. Punishes the WORST training year, forcing TPE to optimize regime robustness. Per-year min activity floor: 20 trades/year.

**Trial counts**:
- COMPLETE: 1546
- RUNNING (at report time): 6
- PRUNED (constraint violations): 38

**Top 5 ranked by 2026 forward PnL** (not by training score!):

| Rank | Trial | Train score | Fwd PnL | Fwd multi | Fwd DD% | Fwd trades |
|---|---|---|---|---|---|---|
| 1 | #1550 | $29,286 | $+36,668 | 2.47× | -63.3% | 362 |
| 2 | #1573 | $29,435 | $+29,102 | 2.16× | -62.7% | 340 |
| 3 | #1566 | $29,286 | $+21,200 | 1.85× | -64.0% | 340 |
| 4 | #1564 | $29,286 | $+18,531 | 1.74× | -64.0% | 346 |
| 5 | #1556 | $29,286 | $+16,426 | 1.66× | -64.0% | 346 |

**Per-year breakdown — winner #1550** (each year backtested with fresh $25K during training):

| Year | PnL | PF | Trades |
|---|---|---|---|
| 2022 | $37334.0 | 1.122 | 385 |
| 2023 | $9762.0 | 3.332 | 20 |
| 2024 | $118316.0 | 1.325 | 364 |
| 2025 | $127247.0 | 1.164 | 569 |

- **min_pf** across years: **1.122** (the metric we optimized for)
- **min_pnl** (worst year $ PnL): 9762.0
- **sum_pnl** across years: 292659.0

**Verdict for W8: `OVERFIT_GOODHART`**

_Why_: Round winner forward < $80K AND min_pf > 0.7. Classic Goodhart: TPE found the cheapest way to satisfy the min-year constraint (trade frequently with tiny per-trade edge) — every year clears the threshold but no year has real alpha. Fix: switch from min() to geomean(year_pnl × min(year_pf, 3)). Geomean still punishes weak years on log-scale but REWARDS strong years, so high-edge configs can't be dominated by 'barely positive everywhere' trash.

_Action_: Next round will use variant `geomean_year_score` because the OVERFIT_GOODHART failure mode has a targeted fix.

## Round W9 — variant `geomean_year_score`

**Objective formula**:
  score = geomean(year_pnl × min(year_pf, 3.0)) across all years. Same per-year reset as default, but uses geometric mean instead of min. Still punishes weak years on log-scale, but REWARDS strong years instead of ignoring them. Fix for the W8 pattern where TPE found `barely-positive every year` strategies. Any year with <=0 score -> hard penalty (preserves positivity gate).

**Trial counts**:
- COMPLETE: 604
- RUNNING (at report time): 7
- PRUNED (constraint violations): 28

**Top 5 ranked by 2026 forward PnL** (not by training score!):

| Rank | Trial | Train score | Fwd PnL | Fwd multi | Fwd DD% | Fwd trades |
|---|---|---|---|---|---|---|
| 1 | #611 | $31,012 | $+2,882 | 1.12× | -62.7% | 353 |
| 2 | #614 | $29,096 | $-2,039 | 0.92× | -65.8% | 339 |
| 3 | #615 | $20,013 | $-2,077 | 0.92× | -58.9% | 273 |
| 4 | #606 | $19,919 | $-3,200 | 0.87× | -60.7% | 260 |
| 5 | #605 | $20,333 | $-4,847 | 0.81× | -58.8% | 262 |

**Per-year breakdown — winner #611** (each year backtested with fresh $25K during training):

| Year | PnL | PF | Trades |
|---|---|---|---|
| 2022 | $8206.0 | 1.039 | 341 |
| 2023 | $3110.0 | 1.518 | 20 |
| 2024 | $83678.0 | 1.32 | 353 |
| 2025 | $162640.0 | 1.278 | 546 |

- **min_pf** across years: **1.039** (the metric we optimized for)
- **min_pnl** (worst year $ PnL): 3110.0
- **sum_pnl** across years: 257635.0

**Verdict for W9: `OVERFIT_GOODHART`**

_Why_: Round winner forward < $80K AND min_pf > 0.7. Classic Goodhart: TPE found the cheapest way to satisfy the min-year constraint (trade frequently with tiny per-trade edge) — every year clears the threshold but no year has real alpha. Fix: switch from min() to geomean(year_pnl × min(year_pf, 3)). Geomean still punishes weak years on log-scale but REWARDS strong years, so high-edge configs can't be dominated by 'barely positive everywhere' trash.

_Action_: Next round will use variant `geomean_year_score` because the OVERFIT_GOODHART failure mode has a targeted fix.

## Did 2023 evolve across rounds? (the key validation signal)

Background: every one of W7's top 30 trials LOST money on 2023 (range −$2,608 to −$8,166). The hypothesis is that CV-style objectives should produce trials that don't lose on 2023.

| Round | Variant | Winner | 2023 PnL | min_pf | 2026 fwd PnL |
|---|---|---|---|---|---|
| W7 (reference) | legacy (compounded) | #254 | −$2,608 | 0.69 | +$199,553 |
| W8 | default | #1550 | 9762.0 | 1.122 | $+36,668 |
| W9 | geomean_year_score | #611 | 3110.0 | 1.039 | $+2,882 |

## Methodology — every decision the orchestrator made

**Decision tree:**

```
After each round:
  IF winner_forward >= $150K           -> STOP, success
  IF winner_forward < $80K
    AND median(year_n) < 30            -> next variant = activity_weighted
    AND min_pf > 0.7                   -> next variant = hybrid_sum_minpf2
  ELSE                                 -> STOP, inconclusive (no clear remedy)
Stop after 3 rounds or when all 2 alternate variants tried.
```

**Variants implemented:**

- `default`: score = min(year_pnl × min(year_pf, 3.0)) across [2022,2023,2024,2025]. Each year backtested independently with fresh $25K. Punishes the WORST training year, forcing TPE to optimize regime robustness. Per-year min activity floor: 20 trades/year.
- `activity_weighted`: score = min(year_pnl × min(year_pf, 3.0)) × geomean(year_n) / 100. Same as default but multiplied by activity bonus. Rewards trials that trade actively across all years, not just clear the floor.
- `hybrid_sum_minpf2`: score = sum_pnl × min_pf². Keeps total-PnL signal but weights minPF² hard. A trial with min_pf 0.3 scores 9× lower than min_pf 0.9 at same PnL. Per-year min activity floor relaxed to 15 trades/year.

## What to do next

- All variants tried produced sub-$80K forwards. The CV objective family may not be the right fix — consider:
  - Drop 2023 from training (it's only 67 microcap-pump days)
  - Add 2021 to training (more data variety)
  - Switch to true expanding walk-forward (train 22, test 23; train 22-23, test 24; etc.)

## Caveat — known data gap

- 2026 picks pkl has empty entries for 2026-05-21 through 2026-06-16 (daily_top_gainers.csv has gappers, but picks builder produced empty lists). All forward-2026 numbers above are effectively only **2026-01-05 through 2026-05-15**. Picks rebuild deferred until after this orchestrator finishes (would compete for CPU with the workers).
