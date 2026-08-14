# G Exit-Strategy Investigation - Complete Summary (2026-08-14)

Prepared for external AI review (Claude). Every exit-strategy idea we tested this
session, the exact numbers, and the conclusions. Context: G = gap-up runner
(gap >= 30%, 2nd green candle + new high, entry candle-2 close ~09:31/09:32),
deployed exit = target +18%, trail 1% (active from entry), stop -10%, time 12m,
no partial.

---

## 0. The starting facts (250 G trades, 2024-2026)

- Max favorable excursion (MFE) is fat-tailed: median +16%, p90 +83%, p95 +142%.
  20% of trades reach +50%, 9% reach +100%.
- BUT the current 1% trail banks a median +2.8% / mean +4.6% (gross) and
  captures 0% of the +50%/+100% moves. The +18% target fires 0 times in
  250 trades: exits are 94% TRAIL, 4% STOP, 1% TIME. The 30% target2 never fires.
- Names fade after the opening spike: forward return from 15-min mark to close is
  negative at every threshold (up 0% -> -5.5%, up 30% -> -4.3%, up 50% -> -16.8%).
- Hypothesis: adaptive trail *shape* (ATR / swing-low / staged-ratchet) +
  partial-sell runner tranche could capture the home runs without giving back the
  base win rate.

---

## 1. Fixed-trail width sweep (baseline for everything)

Same entry; only the fixed trail % changes (no target binds; trail dominates):

| exit | mean (gross) | win% |
|---|---|---|
| trail 1% (current) | +4.6% | 82% |
| trail 2% | +3.6% | 70% |
| trail 3% | +2.6% | 59% |
| trail 5% | +1.0% | 45% |

Monotonic degradation - the 1% trail is already near-optimal for fixed widths.
Holding longer (wider trail) gives back the early spike on the 80% that fade.

---

## 2. Adaptive trail shapes (the "fancy exits")

New machinery added to strategies/config.py + test_green_candle_combined.py
(exit-mode selector, all default-OFF so deployed behavior is unchanged):

- staged - tight 1% below a profit threshold, wide trail above it (ratchets)
- atr - Chandelier: trail = peak - k*ATR%
- swing - swing-low: trail = peak - k*(peak - recent N-bar low)
- partial-sell runner - sell X% at target1, ride remainder to target2 with
  an entry floor (never below entry)

Results on the same 250-trade set (gross %, no time limit):

| exit | mean | win% | max |
|---|---|---|---|
| staged 1/2.5/5 @ +8/+20 | +3.0% | 76% | +18% |
| ATR k=2.0 | +0.2% | 35% | +222% |
| swing k=1.5/2.0 | +0.1% | 28% | +900% |
| partial sell 30% @ +25, swing k2 runner | +6.5% | 82% | +637% |
| partial sell 30% @ +18, swing k1.5 | +6.2% | 81% | +635% |
| partial sell 50% @ +25, swing k2 | +5.9% | 82% | +462% |
| whole-position swing (no partial) | +4.0% | 12% | +900% |

Key findings:
- Adaptive trails capture the monsters (swing max +900%, ATR +222%) that fixed
  trails can't (max +60-78%), but the *mean* is flat-to-negative because 80% fade.
- The swing trail is specifically better than a fixed wide trail on the runner
  tranche (+6.5% vs +4.5% for fixed 3/5/8%) - it rides parabolic expansion.
- The partial-sell is what makes it work: whole-position swing (no partial)
  collapses to +4.0%/12% WR vs +6.5%/82% with the partial + entry floor.
- The entry floor barely mattered (+6.5 vs +6.4) - the swing stop naturally keeps
  runners above entry.
- Year split killed it: 2024 +3.6%, 2025 +3.8%, 2026 +13.8%. The entire
  edge is 2026 (runner-rich regime). In 2024/2025 the partial-sell was equal-or-
  slightly WORSE than baseline. Robust to top-5 trimming (still +6.5-6.9% vs base
  +4.9%), but the year dependence is the real story.

---

## 3. Optuna study A - full exit-shape (13 tunable dims)

gl_1min_g2_1x_exit_shape_v2_pf, Postgres optuna_gl_1min. Entry LOCKED from the
deploy config (G_FIXED_FROM_CONFIG + G_TUNE_EXITS_ONLY); only exit params tuned:
target, target2, partial%, stop, trail, trail-activate, time, exit_mode,
atr_mult, swing_k, swing_window, staged_thresh, staged_wide. Train 2024-2026-02
(488d), val 2026-01..02 (38d), objective min(val_pnl * min(val_pf, 3)). Startup
80, converged ~150, stopped at 149 complete.

- Train-best #126: exit_mode=atr, partial 25%, tgt +17, tgt2 +150, stop 12,
  time 9. val +$37k / PF 7.97.
- Wide-forward (100 trials on 2026 OOS, $10k):

| | TOP-50 | RANDOM-50 |
|---|---|---|
| positive-PnL | 50/50 | 47/50 |
| median PnL | $70,480 | $30,453 |
| median PF | 3.69 | 2.92 |
| median WR | 63.2% | 66.7% |
| best | $280,401 | $356,956 |
| worst | $3,895 | -$4,508 |

train-score -> OOS corr +0.24 (Spearman 0.20, p=0.045) - real but modest signal.
- Single-best-trial trap: train-best #126 (ATR) was WORSE than baseline OOS
  (+$78.9k, 52.8% WR, -37.8% DD). The best-OOS config was #101, which is just
  baseline + partial-sell.

---

## 4. Optuna study B - partial-only (7 tunable dims)

gl_1min_g2_1x_partial_v1. Same setup but G_PARTIAL_EXITS_ONLY=1: exit_mode
LOCKED to fixed, only target, target2, partial%, stop, trail, trail-activate,
time tuned. Startup 40, 171 complete.

- Best #54: partial 50% @ +9, tgt2 +70, stop 0, trail 1 (act +8), time 3.
- Degenerate basin: top-8 ALL have stop=0; across all 100 wide-forward trials,
  81/100 stop=0, 88/100 partial=50%. TPE found the "no-stop + bank-half-fast"
  valley (W16 trap - looks free in the benign 38-day val window, uncapped tail OOS).
- Wide-forward (2026 OOS, $10k):

| | TOP-50 | RANDOM-50 |
|---|---|---|
| positive-PnL | 50/50 | 50/50 |
| median PnL | $73,492 | $73,492 |
| median PF | 3.02 | 3.02 |
| median WR | 69.2% | 69.2% |
| max | $79,213 | $265,839 |

The TOP-50 are *nearly identical* (med $73,492, PF 3.02, WR 69.2% for every one)
because they're the same stop=0 config - same trades, same outcome.

---

## 5. Head-to-head vs deployed baseline (2026 OOS Mar-Aug, $10k and $25k)

| config | PnL ($10k) | WR | PnL ($25k) | WR |
|---|---|---|---|---|
| baseline (deployed) | +$288,792 | 100% | +$564,654 | 100% |
| best-OOS #101 (exit-shape study) | +$356,956 | 100% | +$677,626 | 100% |
| best-OOS #8 (partial-only study) | +$265,839 | 97.7% | - | - |
| wide-forward median (either study) | ~$73k | ~69% | - | - |

- #101 (baseline + 25% partial @ +16, runner +180, trail act +2, stop 12) beat
  baseline by +24% ($10k) / +20% ($25k) with identical 100% WR / 0 DD. This is
  the ONLY config that beat baseline cleanly. Its structural change vs baseline:
  partial-sell + delayed trail activation + wider stop (NOT the fancy trails).
- But 0/100 wide-forward trials beat baseline in study B, and study A's top
  distribution only *partially* clears it. The #101 result is a single config, not
  a robust generalizable edge across the search.

---

## 6. Conclusions

1. The 1% trail is the entire exit (94% of exits; targets essentially never
   fire) and is near-optimal. Widening fixed trails monotonically loses.
2. Adaptive trails (ATR/swing/staged) capture the home runs but lose OOS -
   they give back more on the 80% that fade. The "ride the monster" hypothesis is
   mechanically real (swing max +900%) but not profitable on average OOS.
3. Partial-sell + runner tranche is the only structural change with any edge,
   and only in a runner-rich regime (2026). It's upside optionality, not a
   guaranteed improvement - the year-split shows it flat-to-negative in 2024/2025.
4. No-stop is the W16 degenerate trap - TPE finds it because it maximizes the
   val-window score, but it has uncapped tail risk and 69% WR OOS. Any future
   study should enforce --g-stop-min 5.
5. Single-best-trial selection is dangerous (train-best #126 was worse than
   baseline OOS). Wide-forward + distribution-level checks are mandatory.
6. Recommendation: keep the deployed G config as-is. The most defensible
   improvement (validated but regime-dependent) would be #101's partial-sell exit,
   but it needs a second OOS window (2025) to confirm before deploying.

---

## 6b. REVISED VERDICT after Claude's methodology push + bootstrap (2026-08-14)

Claude flagged four concerns: (1) the 100% WR 2026 OOS is too small a sample /
benign regime, (2) "the edge is 2026" means #101 was tested on the data that
generated it, (3) corr 0.24 train->OOS means Optuna is near noise-fitting,
(4) the per-trade edge needs a bootstrap to rule out a few lucky monster trades.
We ran the checks:

**A. Genuinely held-out 2025 test (246 days, $25k, 108-112 trades):**

| metric | baseline | #101 |
|---|---|---|
| trades | 108 | 112 |
| win% | 90.7% | 92.9% |
| total PnL | +$846,553 | +$1,016,483 |
| PF | 13.18 | 14.18 |
| max DD | -10.9% | -9.3% |

#101 beat baseline +20% PnL on 2025 too, with higher WR and lower DD. AND 2025
shows REAL losses (baseline 10, #101 8; worst -$65k) - confirming Claude's point
that the 2026 100% WR was small-sample/benign-regime, not a real property.

**B. Bootstrap / permutation test (10,000 resamples, all 250 trades, faithful sim):**

| config | mean/trade | edge vs base | 95% CI | P(edge<=0) |
|---|---|---|---|---|
| baseline | +4.58% | - | - | - |
| A#126 (ATR train-best) | -0.73% | -3.34% | [-4.66%, -1.96%] | 100% |
| A#101 (best-OOS) | +1.32% | -1.29% | [-1.91%, -0.67%] | 100% |
| A#85 (top-50 best) | +1.28% | -1.33% | [-1.96%, -0.74%] | 100% |
| B#54 (no-stop) | +0.92% | -1.69% | [-2.46%, -0.91%] | 100% |
| B#8 (rnd best OOS) | +0.78% | -1.83% | [-2.83%, -0.89%] | 100% |

EVERY tuned exit config loses to baseline with 100% bootstrap confidence (CI never
straddles zero). Per-year: baseline is positive all 3 years (+2.40/+2.40/+3.15%);
every tuned config underperforms baseline in EVERY year. There is no hidden 2026
edge - the earlier "2026 +13.8%" was a sim-fidelity artifact.

**REVISED CONCLUSION:**
- The earlier "$ gain" OOS wins for #101 (2025 +20%, 2026 +24%) were COMPOUNDING
  VARIANCE, not per-trade edge. Full-balance sizing amplifies slightly-higher-median
  configs into big $ swings, but the per-trade edge is negative and significant.
- Both Optuna searches (13-dim and 7-dim) found configs that look good on a 38-day
  val window but have NEGATIVE per-trade edge vs baseline on the full sample =
  classic overfit (corr 0.24 confirmed this).
- **THE DEPLOYED BASELINE IS OPTIMAL.** Every alternative tested (fixed widths,
  ATR/swing/staged shapes, partial-sell, no-stop) loses on the per-trade level
  with 100% bootstrap confidence.
- Claude's methodology (held-out 2025 first, bootstrap before believing any $ PnL
  claim) should be standard practice going forward. The "keep deployed config"
  recommendation is now supported by rigorous evidence, not just OOS dollars.

---

## 6c. SECOND-ROUND BOOTSTRAP through the REAL ENGINE (2026-08-14)

Claude's second push: (a) my first bootstrap used a hand-rolled sim_exit, not the
deployed engine pipeline; (b) sections 1-2 (fixed-trail sweep, adaptive shapes,
partial-sell +6.5%) came from a GROSS/no-slippage sim and needed the same
bootstrap treatment; (c) the trade-count discrepancy (250 vs 180) needed explaining.

Ran all candidates through the REAL simulate_day_combined (the deployed engine's
exact code, dynamic + multiwindow slippage ON, volcaps ON, deploy entry gate
g_min_2nd_body_pct=2.0), $1M cash so per-trade returns aren't sizing-constrained.
Aligned to the common 177-trade set. 10,000 bootstrap resamples:

| config | mean% | edge% | 95% CI | P(edge<=0) |
|---|---|---|---|---|
| baseline | +8.09% | - | - | - |
| trail2 | +7.03% | -1.06% | [-1.09, -1.02] | 100% |
| trail3 | +5.99% | -2.10% | [-2.18, -2.02] | 100% |
| trail5 | +4.65% | -3.44% | [-3.78, -3.06] | 100% |
| staged | +6.07% | -2.02% | [-2.38, -1.64] | 100% |
| ATR k2 | +4.16% | -3.93% | [-5.28, -2.57] | 100% |
| swing k2 | +4.37% | -3.72% | [-5.19, -2.22] | 100% |
| partial_swing | +5.51% | -2.58% | [-4.66, -0.46] | 99% |
| A#126 (ATR) | +4.58% | -3.50% | [-5.21, -1.78] | 100% |
| A#101 | +8.38% | +0.29% | [-0.20, +0.89] | 14% |
| B#54 (no-stop) | +6.73% | -1.36% | [-2.23, -0.46] | 100% |
| B#8 | +7.26% | -0.83% | [-1.53, -0.07] | 98% |

FINDINGS:
- Sections 1-2 conclusions HOLD under the real engine: fixed-trail sweep
  (1% > 2% > 3% > 5%), staged, ATR, swing all lose to baseline with ~100%
  confidence. The earlier gross-sim direction was right, magnitudes differed.
- A#101 is the ONLY candidate that is not a clean null: +0.29%/trade edge,
  P(edge<=0)=14%, CI [-0.20, +0.89] — statistically INCONCLUSIVE (straddles zero),
  neither a clean win nor a clean loss. The previous hand-rolled-sim bootstrap
  (which showed #101 at -1.29%) was a pipeline artifact.
- Trade counts: the deploy gate g_min_2nd_body_pct=2.0 gives baseline 180 trades
  on the 2024-26 window; the Optuna trials (which re-suggested that gate) trade
  246-249. That 66-trade difference is the entry-gate effect, not exit behavior.
- NET: baseline stays recommended. #101 is the only candidate worth a future,
  longer-sample test (more trades) to resolve the inconclusive +0.29% — but it is
  NOT proven better and should not deploy as-is.

STANDING RULE (from Claude, adopted): never evaluate an exit-strategy candidate
on compounded $ PnL alone — always pair with per-trade mean/median AND a
bootstrap CI over the aligned common trade set before treating a $ result as
signal. Compounding math + fat right tails (p95 MFE +142%) can make a losing
per-trade config look like a winner on $ over a short window.

---

## 6d. GENUINELY-UNUSED ERA TEST: 2020-2023 (2026-08-14)

Claude's graduation gate demands a genuinely-unused period. Ran baseline vs A#101
through the REAL engine on 2020-23 1-min data (876 days), aligned-trade bootstrap
(5,000 resamples):

| year | baseline n | #101 n | aligned n | edge | 95% CI | P(edge<=0) |
|---|---|---|---|---|---|---|
| 2020 | 94 | 140 | 54 | +0.01% | [-0.13%, +0.16%] | 45% |
| 2021 | 53 | 78 | 35 | -0.10% | [-0.26%, +0.07%] | 88% |
| 2022 | 47 | 61 | 25 | -0.39% | [-1.09%, +0.18%] | 90% |
| 2023 | 3 | 4 | 2 | +0.46% | (n too small) | 26% |

Per-trade means: baseline wins every year (2020 +11.0%/96% WR vs +8.8%/92%;
2021 +8.5%/100% vs +6.1%/90%; 2022 +6.3%/81% vs +4.4%/75%).

VERDICT: A#101's edge does NOT generalize to 2020-23. Flat on 2020, negative-
leaning on 2021-22, inconclusive on 2024-26, and only "won" via compounding
variance on the specific 2025/2026 $ windows. CLOSED: baseline is optimal, #101
does not graduate to deploy. This is a clean null across every era tested.

---

## 7. Bonus - the real win of the day: reconcile bug fix (not exit-related)

STKH 2026-08-14 incident: a slow-filling market order (2125 sh filled 09:33-09:34
after 09:32 placement) was abandoned by the old 60s safety-poll, so the engine
cleared position state while Alpaca kept filling -> real position left unmanaged
(no trail/stop, trapped at -20%). Fixed + deployed live:
- Safety-poll now polls with exponential backoff (60/120/300s) + checks filled_qty
  and GET /positions before ever clearing.
- _reconcile_position never clears internal state on a lookup failure when
  internal shares > 0.
- New _reconcile_all_positions: 5-min full reconcile vs GET /positions that
  rehydrates orphaned positions and picks up external closes.
- Verified with a 4-scenario test (orphan rehydration, pending-buy preservation,
  external-close clearing, transient-failure state retention). 56 repo tests pass.
