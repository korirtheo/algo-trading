# Walk-Forward Optuna Runs — Tags

Convention: `<objective>_<regime>_<extras>` so future objective experiments
slot cleanly alongside.

## Completed / on-disk runs

| Directory | Tag | Objective | Slippage | Regime filter | X enabled | Status |
|---|---|---|---|---|---|---|
| `walk_forward/` | `wf_pf_full_x_v1` | `pnl × PF`  (legacy) | Cumulative (legacy) | none | yes | W1-W3 done; W4 partial |
| `walk_forward_v2/` | `wf_sliding_killed_v1` | `pnl × PF` (sliding 2yr) | Cumulative | none | yes | killed mid-run |
| `walk_forward_v3/` | `wf_sharpe_full_x_v1` | `pnl × Sharpe%` | Multiwindow + vol-adj | none | yes | W3 only — DONE |
| `walk_forward_v4_squeeze/` | `wf_sharpe_squeeze_noX_v1` | `pnl × Sharpe%` | Multiwindow + vol-adj | `squeeze` | no | W3 RUNNING |

## Future runs (planned)

| Tag | Objective formula | Notes |
|---|---|---|
| `wf_sortino_squeeze_noX_v1` | `pnl × Sortino` (downside vol only) | Sortino is Sharpe with downside-only std. Often better for asymmetric strategies. |
| `wf_calmar_squeeze_noX_v1`  | `pnl × Calmar = CAGR / MaxDD`     | Direct drawdown penalty. Quant-desk favorite. |
| `wf_composite_squeeze_noX_v1` | `Sharpe × √(trades/100) × (1 − MaxDD)` with hard floors (trades ≥ 100, PF > 1.2, MaxDD < 25%) | ChatGPT-style composite. Constraints prevent overfit-to-noise. |
| `wf_sharpe_normal_noX_v1`   | `pnl × Sharpe%` | Same as v4 but `--regime-filter normal` |
| `wf_sortino_normal_noX_v1`  | etc. | parallel sortino normal-day specialist |

## Headline forward-2024 results so far

| Tag | Forward 2024 final | Sharpe% | Notes |
|---|---|---|---|
| `wf_pf_full_x_v1` | $4,381,635 | 4.99 (post-hoc % returns) | Inflated by cumulative slippage; honest projection $1.18M |
| `wf_sharpe_full_x_v1` | $99,582 | 1.81 (forward) | Honest projection from Optuna trained against multiwindow |
| `wf_sharpe_squeeze_noX_v1` | _(running)_ | _(pending)_ | Smaller training set (squeeze days only); expect cleaner specialist |

## How to read this when comparing runs

Each new objective should be evaluated:
1. **In-sample** score (training)
2. **Forward** % return Sharpe on its test year
3. **Across windows** stability (W1-W5)
4. **Real-fill expectation** (once `fills_calibration.csv` calibrates K)

The right deployment choice is the one with the **best forward stability across
all 5 windows**, not the highest-score-on-one-window.

See [docs/aws_migration.md](../docs/aws_migration.md) for the live deployment
runbook — currently using `config/trial_6_extracted.json` (the
generalist baseline).
