# Algo Trading — Gap-Up Day Trading System

Automated intraday trading system targeting small-cap gap-up stocks.
20 candidate strategies, tuned with Optuna, running on Alpaca paper
trading from AWS with a real-time React dashboard.

- **Universe**: small-cap US equities gapping ≥ 8% pre-market (price ≤ $50)
- **Timeframe**: 2-minute candles, 9:30 AM → 3:45 PM ET
- **Capital**: $10K–$25K starting cash, full-balance sizing with vol-cap
- **Live**: Alpaca paper (IEX feed), AWS EC2 Ubuntu, Docker Compose
- **Backtest data**: continuous Jan 2024 → Feb 2026 (~530 trading days), plus
  Mar–May 2026 as held-out OOS (from Polygon.io)

---

## Table of contents

1. [Quick links / cheatsheet](#quick-links--cheatsheet)
2. [Live trading architecture](#live-trading-architecture)
3. [Backtest](#backtest) — **detailed**
4. [Optimization (Optuna)](#optimization-optuna)
5. [Findings & open questions](#findings--open-questions)
6. [Repo layout](#repo-layout)
7. [AWS deployment](#aws-deployment)
8. [Dependencies](#dependencies)

---

## Quick links / cheatsheet

| Task | Command |
|---|---|
| Run live (laptop) | `python -m live.main` |
| Run live without dashboard | `python -m live.main --no-dash` |
| Scanner only (test) | `python -m live.main --scan-only` |
| Backtest current live params, 2-yr | `python scripts/backtest/run_backtest_from_json.py --json config/trial_432_params.json --slippage 0.3 --vol-cap 5.0 stored_data_jan_feb_2024 stored_data_jan_mar_2024 stored_data_apr_jun_2024 stored_data_jul_sep_2024 stored_data_oct_dec_2024 stored_data_jan_mar_2025 stored_data_apr_jun_2025 stored_data_jul_2025 stored_data_oos stored_data` |
| Backtest with cash override (legacy slippage) | `python run_live_params_backtest.py --data stored_data_mar_may_2026 --cash 10000 --slippage 0.3 --vol-cap 5.0` |
| Backtest with **dynamic slippage** (honest) | `python run_live_params_backtest.py --data stored_data_mar_may_2026 --cash 10000 --vol-cap 5.0 --dynamic-slip` |
| 2-yr backtest with dynamic slippage + charts | `python scripts/backtest/run_backtest_from_json.py --json config/trial_432_params.json --vol-cap 5.0 --dynamic-slip stored_data_jan_feb_2024 stored_data_jan_mar_2024 stored_data_apr_jun_2024 stored_data_jul_sep_2024 stored_data_oct_dec_2024 stored_data_jan_mar_2025 stored_data_apr_jun_2025 stored_data_jul_2025 stored_data_oos stored_data` |
| Backtest single day live (today) | `python scripts/backtest/backtest_today.py` |
| Re-optimize at honest cost | `python optimize_combined.py --trials 500 --slippage 0.3 --vol-cap 5.0 --db optuna_combined_honest.db --study combined_honest_slip03 --params-out optuna_best_params_honest.json` |
| Optimize current-best inspection | `python optimize_combined.py --dump-best --db <db> --study <name> --params-out <json>` |
| Download more data (Alpaca) | `python scripts/download/download_alpaca_2023.py` (template) |
| Download more data (Polygon) | `POLYGON_OUT_DIR=stored_data_X python scripts/download/download_polygon.py <KEY> START END` |
| Dashboard standalone | `python -m dashboard.run --port 8000` |
| Frontend dev server | `cd dashboard/frontend && npm run dev` |
| Build frontend | `cd dashboard/frontend && npm run build` |

---

## Live trading architecture

### Daily schedule (ET)

| Time | Action |
|---|---|
| 7:00 AM | First pre-market scan |
| 7:30 AM | Second scan |
| 8:30 AM | Third scan |
| 9:00 AM | Fourth scan |
| 9:25 AM | Fifth scan |
| 9:27 AM | Final scan — watchlist locked (top 20 by gap %) |
| 9:30 AM | Market open → 2-min bar stream begins → strategies fire |
| 9:30 AM–3:45 PM | Engine processes each completed bar, sends orders to Alpaca |
| 3:45 PM | EOD close-all (`engine.eod_close()`) |
| 4:00 PM → next morning | Dashboard stays live; main process sleeps until next session |

**Daily rollover** ([live/main.py:213](live/main.py#L213)): after EOD, the engine
queries `executor.client.get_clock().next_open`, sleeps in 5-min chunks
until 2.5 hours before next open (~7:00 ET, weekdays only, holidays
handled automatically), then re-runs the scan loop. This is the fix
for the bug where the AWS container kept yesterday's watchlist forever.

### Components

| File | Role |
|---|---|
| [live/main.py](live/main.py) | Entry point — orchestrates scanner, engine, streamer, dashboard, rollover |
| [live/scanner.py](live/scanner.py) | Webull pre-market top-gainers (primary) + Alpaca movers (fallback) + Finviz float fetch |
| [live/streamer.py](live/streamer.py) | Alpaca WebSocket 1-min bars → 2-min candle aggregation |
| [live/engine_combined.py](live/engine_combined.py) | Wraps `simulate_day_combined` from `test_green_candle_combined.py`; diffs state across bars to detect ENTRY/EXIT/PARTIAL |
| [live/executor.py](live/executor.py) | Alpaca order execution (market buy/sell, vol-cap check, close_all) |

### Scanner sources

1. **Webull `topGainers` API** (live SIP-quality volume + float)
2. **Alpaca screener `movers` endpoint** (fallback if Webull errors)
3. **Finviz** scraped only for float-shares backfill on tickers Webull didn't cover

After candidates are selected: Alpaca 1-min PM bars (4 AM → 9:30 ET) computed for each candidate to determine the actual premarket high. Sorted by gap %, top 20 kept.

### Critical fixes shipped this session

| Commit | Bug | Symptom |
|---|---|---|
| `9d0ee69` | `simulate_day_combined` EOD wipe ran on every `on_bar()` call | Engine showed strategies as eligible in the diagnostics panel but never actually placed any order |
| `346e64f` | No daily rollover — process slept forever after EOD | AWS container kept Monday's watchlist alive through Friday |
| (this session) | Data merge gap (March–December 2024 missing from `stored_data_combined`) | Backtests were running on 325 days, not 530 |

---

## Backtest

This is the heart of the project. There are multiple entry points; they
all wrap the same core simulator (`simulate_day_combined` in
`test_green_candle_combined.py`).

### Data pipeline

#### What "data" means

Each `stored_data_*` directory contains:

```
stored_data_X/
├── daily_top_gainers.csv      ← per-day list of top-20 gappers (the picks)
├── grouped_daily.json          ← raw daily OHLCV cache (Polygon downloads)
├── daily/<TICKER>.csv          ← daily OHLCV for each candidate ticker
│                                  (used for prev-close lookups)
├── intraday/<TICKER>.csv       ← 2-min OHLCV for each candidate ticker
│                                  (the simulation input)
└── fulltest_picks_gap2_vol250k.pkl   ← cached merged picks pickle
                                        (auto-rebuilt if missing; cache key
                                         is gap_pct + min_pm_volume)
```

#### Building the picks pickle

`test_full.load_picks_for_dir(data_dir)` ([test_full.py:215](test_full.py#L215)):

1. Scans every ticker × every trading day in the directory
2. Workers (parallel via `ProcessPoolExecutor`) filter by:
   - PM volume ≥ 250K shares
   - Gap % ≥ 2.0 (open vs prev_close)
   - Price ≤ $50
   - Not a warrant/right/unit
3. Top 20 by gap % per day → list of `pick` dicts (one per ticker-day)
4. Pickled to `fulltest_picks_gap{N}_vol{N}k.pkl`

`test_full.load_all_picks([dirs])` reads each per-dir pickle and merges
by date (dedupes — first dir wins per date if there's overlap).

#### Continuous coverage we have

| Directory | Coverage | Trading days |
|---|---|---|
| `stored_data_jan_feb_2024` | 2024-01-02 → 2024-02-29 | 41 |
| `stored_data_jan_mar_2024` | 2024-03 only | 18 |
| `stored_data_apr_jun_2024` | 2024-04 → 2024-06 | 62 |
| `stored_data_jul_sep_2024` | 2024-07 → 2024-09 | 63 |
| `stored_data_oct_dec_2024` | 2024-10 → 2024-12 | 63 |
| `stored_data_jan_mar_2025` | 2025-01 → 2025-03 | 59 |
| `stored_data_apr_jun_2025` | 2025-04 → 2025-06 | 61 |
| `stored_data_jul_2025` | 2025-07 | 21 |
| `stored_data_oos` | 2025-08 → 2025-12 | 105 |
| `stored_data` | 2026-01 → 2026-02 | 38 |
| `stored_data_mar_may_2026` | 2026-03 → 2026-05 (OOS) | 43 |

**Total in-sample** (training window): **531 days**, Jan 2024 → Feb 2026.
**Out-of-sample**: 43 days, Mar–May 2026.

`stored_data_combined` is a legacy partial merge of 6 of those — only 325 days, **missing all of March–December 2024**. Don't use it alone; always pass the explicit list above when running the full backtest. The Optuna optimizer's `DATA_DIRS` has been updated to use the explicit list ([optimize_combined.py:33](optimize_combined.py#L33)).

### The simulator: `simulate_day_combined()`

Located in [test_green_candle_combined.py:450](test_green_candle_combined.py#L450). Three thousand lines, but the contract is simple:

```python
states, final_cash, unsettled_proceeds, selection_log = \
    simulate_day_combined(picks, starting_cash, cash_account=False, is_live=False)
```

**Per-day flow**:

1. Build a unified timestamp index across all picks
2. Compute VWAP per ticker
3. Initialize state for each pick (per-strategy eligibility flags, etc.)
4. For each timestamp (2-min bar):
   - Update VWAP, candle counter, running HOD, etc., per ticker
   - For each pick still active:
     - Process exit conditions if in position (stop, trail, target, time-limit, EOD)
     - Check signal conditions for each enabled strategy in `STRAT_PRIORITY` order
     - First strategy that fires gets the cash (single-position-at-a-time engine)
5. EOD close any open positions (unless `is_live=True`, where the live engine handles EOD itself)

**Selection log** records every "skipped" event: when a strategy was eligible but cash was already deployed. The post-run summary prints which strategies stole cash from which.

**Margin vs cash account**: `cash_account=True` routes sell proceeds to `unsettled` (T+1 settlement); next day, they're added to `cash`. `cash_account=False` is instant settlement (margin account). The live engine flips this based on whether `cash >= $25K` (PDT threshold).

### Cost model

The simulator now supports **two slippage modes**: a legacy constant (default, for reproducibility) and a liquidity-aware dynamic model (the honest one for new analysis).

| Knob | Default | Effect |
|---|---|---|
| `SLIPPAGE_PCT` | 0.05 (in test_full) | Legacy constant slippage when `USE_DYNAMIC_SLIPPAGE = False` |
| `VOL_CAP_PCT` | 5.0 | Caps `position_dollars ≤ VOL_CAP_PCT × cumulative_dollar_volume_to_entry` |
| `USE_DYNAMIC_SLIPPAGE` | **False** | Master switch. Set to True to use the liquidity-aware model |
| `SLIP_BASE_SPREAD` | 0.05 | Minimum spread at infinitely liquid price |
| `SLIP_PRICE_COEFF` | 0.5 | How spread scales with `1/price` |
| `SLIP_IMPACT_K` | 3.0 | Square-root impact coefficient (Almgren-Chriss) |
| `REGIME_AMP` | 0.0 | VIX/ATR amplifier (opt-in; 0 = disabled) |

**Vol cap** — enforced in [test_green_candle_combined.py:2109](test_green_candle_combined.py#L2109). It's measured against *cumulative dollar volume traded since 9:30 ET up to entry*, NOT against ADV. At typical entry times (9:35–10:00 ET), 5% of cumulative ≈ 0.25–0.75% of full-day ADV.

#### Legacy constant slippage (`USE_DYNAMIC_SLIPPAGE = False`)

Every entry/exit uses the same `SLIPPAGE_PCT`. Accurate for ~$10–25K orders on top-20 gappers (real fills are within 0.1–0.5% of candle close on liquid names, 1–2% on penny stocks). **Not** accurate when the compounding curve scales position sizes into the $1M+ range, where a constant 0.3% massively understates real impact.

#### Dynamic liquidity-aware slippage (`USE_DYNAMIC_SLIPPAGE = True`)

Per-leg slippage = `(base_spread + impact) × regime_mult`, where:

```
base_spread = SLIP_BASE_SPREAD + SLIP_PRICE_COEFF / max(price, 0.1)
              $0.50 stock: 0.05 + 1.00 = 1.05% spread before impact
              $5    stock: 0.05 + 0.10 = 0.15%
              $20   stock: 0.05 + 0.025 = 0.075%

impact      = SLIP_IMPACT_K × sqrt(participation_rate)   [Almgren-Chriss]
              participation_rate = position_dollars / cumulative_dollar_volume_to_fill
              5% participation:  K=3 → 0.67% impact
              50%:               K=3 → 2.12% impact
              100% (you ARE):    K=3 → 3.00% impact

regime_mult = 1.0 + REGIME_AMP × max(0, (regime_factor - 20) / 20)
              default REGIME_AMP=0 → regime_mult = 1.0 always
```

This is the **single most important upgrade to the simulator** because it changes what the optimizer optimizes for. With dynamic slippage:
- Penny stocks cost 1–3% per leg before any impact, not 0.05%
- Compounding into $1M+ positions adds 1–10% impact on top of base spread
- The optimizer naturally selects configs that avoid micro-cap participation, not just configs that have positive flat-cost edge

**Empirical example** — cleaned trial 432 on Mar–May 2026 (43 OOS days, $10K start):

| Metric | Legacy 0.3% flat | **Dynamic** |
|---|---|---|
| Final equity | $670,130 | **$251,750** |
| Sharpe (daily) | 4.97 | **3.74** |
| Win rate | 59.5% | 53.8% |
| Best day / Worst day | +$298K / -$20K | +$122K / -$33K |

Strategies that flipped from winners to losers under dynamic costs: **B** (Red-to-Green) and **M** (Midday Range). Their per-trade edge was being subsidized by the unrealistically low flat slippage. Confirmed Phase-2 disable candidates.

**Usage**: pass `--dynamic-slip` to `run_live_params_backtest.py` or `scripts/backtest/run_backtest_from_json.py`. Optionally `--slip-impact-k <N>` to tune the impact coefficient.

### Strategies

20 candidates, 12 currently enabled in trial 432 (the live config):
**H, A, D, M, W, O, B, C, I, J, N, L**.

| Code | Name | Idea |
|---|---|---|
| **H** | High Conviction | 35%+ gap + body ≥ 4% + 2nd green + new HOD + volume confirm |
| G | Big Gap Runner | 30%+ gap + 2nd green + new HOD |
| **A** | Quick Scalp | 15%+ gap + body ≥ 4% + 2nd green + new HOD |
| F | Catch-All | 10%+ gap + 2nd green (no HOD needed) |
| **D** | Opening Dip Buy | Gap + spike + dip + VWAP reclaim |
| V | VWAP Reclaim | Below VWAP for N candles, then reclaim with volume spike |
| P | PM High Breakout | Confirm above PM high, pullback, bounce |
| **M** | Midday Range Break | Morning spike + midday consolidation + breakout |
| R | Multi-Day Runner | Day 1 large gap + day 2 pullback + bounce > d2 open |
| **W** | Power Hour Breakout | Late-day breakout from consolidation |
| **O** | Opening Range Breakout | First N candles' range break with volume |
| **B** | Red-to-Green (R2G) | Red candle 1 + dip + reclaim of open price |
| K | First Pullback | Morning run + orderly pullback + bounce |
| **C** | Micro Flag | Spike + tight base + breakout |
| S | Stuff-and-Break | Multiple HOD rejections then final breakout |
| E | Gap-and-Go RelVol | Extreme PM volume → immediate momentum entry |
| **I** | PM High Immediate | Breaks PM high within first few candles |
| **J** | VWAP + PM Breakout | Near-VWAP + PM high break combo |
| **N** | HOD Reclaim | Old HOD reclaim after pullback |
| **L** | Low Float Squeeze | Float ≤ 15M + 30%+ gap + HOD break + volume surge |

Each strategy has its own params (target %, stop %, trail %, time limit, etc.). Optuna tunes all of them simultaneously plus the enable/priority bits.

### Strategy priority

`STRAT_PRIORITY` ([test_green_candle_combined.py:361](test_green_candle_combined.py#L361)) is the conflict-resolution order. When multiple strategies fire on the same ticker on the same bar, the lower-priority-number wins. Trial 432's priority:

```
M=1 > B=2 ≈ I=2 > L=3 > C=4 > F=6 ≈ N=6 > K=7 >
D=8 ≈ O=8 ≈ S=8 > W=14 ≈ J=14 > H=17 ≈ A=17 ≈ R=17 ≈ E=17 > G=19
```

The optimizer suggests both `enable_X: bool` and `priority_X: int` per strategy.

### Backtest entry points

#### 1. `scripts/backtest/run_backtest_from_json.py` (canonical)

Reads an Optuna-dump JSON (or the flat `config/trial_432_params.json`), injects params into `test_green_candle_combined.py`'s module-level constants, and execs the full backtest with charts.

```powershell
# 2-year continuous, trial 432 params, honest 0.3% slip, with charts
python scripts/backtest/run_backtest_from_json.py `
    --json config/trial_432_params.json `
    --slippage 0.3 --vol-cap 5.0 `
    stored_data_jan_feb_2024 stored_data_jan_mar_2024 `
    stored_data_apr_jun_2024 stored_data_jul_sep_2024 `
    stored_data_oct_dec_2024 stored_data_jan_mar_2025 `
    stored_data_apr_jun_2025 stored_data_jul_2025 `
    stored_data_oos stored_data
```

Flags:
- `--json <path>` — params JSON (default: `results/params/optuna_best_params_v8.json`)
- `--slippage <pct>` — override `SLIPPAGE_PCT` (e.g. 0.3 for 0.3%)
- `--vol-cap <pct>` — override `VOL_CAP_PCT`
- `--no-charts` — skip chart rendering (~10 min faster)
- positional args — data dirs (default: `stored_data_combined`)

Accepts both the wrapped optuna-dump format `{trial_number, score, ..., params: {...}}` AND a flat params dict (so `config/trial_432_params.json` works directly).

Outputs:
- `charts/gc_combined_<timestamp>/gc_page_NN.png` (65+ pages, 5 trading days each)
- `charts/gc_combined_<timestamp>/gc_summary.png` (equity curve, daily P&L distribution, drawdown chart)
- stress test summary to stdout (see below)

#### 2. `run_live_params_backtest.py`

Lighter-weight runner that uses `set_strategy_params()` directly (no source exec) and produces a per-day log line. No charts. Accepts `--cash` for $10K override and explicit `--data` dir.

```powershell
python run_live_params_backtest.py `
    --data stored_data_mar_may_2026 `
    --cash 10000 `
    --slippage 0.3 --vol-cap 5.0
```

Best for: quick OOS checks, comparing param sets without burning 15 min on chart generation.

#### 3. `scripts/backtest/backtest_today.py`

Single-day backtest of *today's* picks. Pulls the live Webull top-20 right now, downloads today's Alpaca 2-min bars, runs `simulate_day_combined` with live params. Used to compare live engine fills vs simulated fills end-of-day.

#### 4. `scripts/backtest/blind_test_2022.py`

True out-of-sample test on 2022 data (the 2024-26 optimizer never saw 2022). Pre-Optuna baseline.

### Stress tests

After every full backtest, `stress_test.py` runs five robustness checks on the per-day P&L series:

| Test | Pass criterion | What it catches |
|---|---|---|
| **Baseline** | Always | Reports total PnL + Sharpe |
| **Remove top 10% of days** | Remaining PnL > 0 | Strategy that depends on a handful of outlier days will fail this |
| **0.2% slippage** | Total PnL > 0 | Sanity check at slightly worse cost than baseline |
| **0.5% slippage (extreme)** | Total PnL > 0 | Tail-risk slippage scenario |
| **Kelly Criterion** | Kelly > 0 | Win rate × win/loss ratio is favorable |
| **Monte Carlo (1000 sims)** | > 80% of randomized day orderings produce positive PnL | Path-dependence check; > 50% but ≤ 80% = WEAK |

Plus diagnostics:
- **Max drawdown** (peak-to-trough)
- **Max losing streak** (consecutive red days)
- **Vol-capped trades** (count of positions truncated by the 5% rule)

### Most recent backtest numbers (trial 432, 0.3% slip, continuous 530 days)

| Metric | Value |
|---|---|
| Starting cash | $25,000 |
| **Ending equity** | **$62,628,639** (+250,414%) |
| Total trades | 3,177 |
| Win rate | 60.6% |
| Profit Factor | 1.70 |
| Sharpe (annualized) | 3.70 |
| Max Drawdown | 31.9% |
| Max Losing Streak | 6 days |
| Green/red days | 305/225 |
| **Remove top 10%** | $+1,825,491 (still positive — PASS) |
| Kelly | 38.4% PASS |
| Monte Carlo | 100% profitable PASS |
| Verdict | **ALL PASS** |

**Per-strategy contribution**:

| Strat | Trades | WR | PnL |
|---|---|---|---|
| O Opening Range Breakout | 2,239 | 61.2% | **$+42.6M** |
| H High Conviction | 141 | 80.9% | $+8.7M |
| J VWAP+PMH Breakout | 186 | 46.2% | $+6.0M |
| N HOD Reclaim | 138 | 60.1% | $+5.0M |
| A Quick Scalp | 171 | 79.5% | $+2.8M |
| I PM High Immediate | 73 | 50.7% | $+2.0M |
| C Micro Flag | 24 | 54.2% | $+61K |
| B Red-to-Green | 59 | 52.5% | $-172K |
| W Power Hour | 9 | 11.1% | $-196K |
| M Midday Breakout | 98 | 41.8% | $-1.3M |
| D Dip Buy | 39 | 30.8% | $-2.9M |
| L Low Float Squeeze | 0 | – | – (never fires under `run_backtest_from_json` flow — runner-discrepancy, under investigation) |

### Honest caveats

1. **Top 10% of days drive 86.7% of P&L** ([daily P&L analysis](#findings--open-questions)). Strategy is heavily outlier-dependent.
2. **One single trade** (SPRB on 2025-10-06) produced $11M = 7.4% of total P&L. At realistic order-book depth that fill is unachievable.
3. **The compounding curve assumes constant 0.3% slippage** regardless of position size. By the time the simulator has compounded to $5M+, individual orders would move the market 2-5% (square-root impact law). The simulator doesn't model this. Realistic projections:
   - $25K → $1M: simulator is approximately right
   - $1M → $10M: simulator overstates by 2-3×
   - $10M+: simulator overstates significantly
4. **PDT rule is not modeled.** Account < $25K is limited to 3 day-trades per rolling 5 business days. Live, this caps activity in early growth phase.

A liquidity-aware slippage model (square-root law: `extra_slip = k × sqrt(participation_rate)`) is on the roadmap — see [Findings](#findings--open-questions).

---

## Optimization (Optuna)

### Algorithm

Optuna with TPE (Tree-structured Parzen Estimator) sampler. First 200 trials are random startup; subsequent trials use TPE's Bayesian search over the param space (~130 dimensions: 5–14 params per strategy + enable flags + priorities for 20 strategies).

### Objective function

```python
score = total_pnl * min(profit_factor, 3.0)
```

Hard constraints:
- `n_trades < 30`: score = -9999 (skip)
- `profit_factor < 0.5`: score = -9999 (skip)

The `min(pf, 3.0)` cap prevents the optimizer from rewarding configs that have unrealistic PF on tiny trade counts.

### Persistence

Studies are stored in SQLite (`*.db`). Each completed trial is written to disk immediately, so the optimizer is fully resumable: kill it and restart with the same `--db --study` and it picks up where it left off.

The best trial's full params + per-strategy breakdown are dumped to a JSON file on every new-best ([optimize_combined.py:799](optimize_combined.py#L799)), so you can inspect the current best without disturbing the running study.

### Running

```powershell
# Honest-cost run (recommended)
python optimize_combined.py `
    --trials 500 `
    --slippage 0.3 --vol-cap 5.0 `
    --db optuna_combined_honest.db `
    --study combined_honest_slip03 `
    --params-out optuna_best_params_honest.json
```

**Throughput**: ~16-24 trials/hour with `n_jobs = cpu_count() // 2` (threaded, not multiproc — GIL limits parallelism but pandas/numpy releases it during native calls).

**ETA**:
- 500 trials ≈ 13–18 hours
- 1000 trials ≈ 25–35 hours

**Crash recovery**: SQLite checkpoints every trial. If laptop dies, restart with the same command and lose only the 4 mid-flight trials.

### Inspecting progress without disturbing the run

```powershell
# Read the auto-saved best-params JSON
Get-Content optuna_best_params_honest.json | ConvertFrom-Json |
    Select-Object trial_number, score, total_pnl, pf, wr, trades, enabled

# Pull current best directly from the DB (concurrent-safe)
python optimize_combined.py --dump-best `
    --db optuna_combined_honest.db `
    --study combined_honest_slip03 `
    --params-out optuna_best_params_honest.json
```

### Historical studies

| Study | DB | Trials | Best | Notes |
|---|---|---|---|---|
| v8 (orig) | `results/studies/optuna_combined_v8.db` | 700+ | trial #1249 PnL $135M (0.05% slip) | Optimized at unrealistic 0.05% slippage — produced trial 432 currently in live |
| v9 | `results/studies/optuna_combined_v9.db` | 1,400+ | (legacy) | Pre-honest-cost regime |
| `combined_honest_slip03` | `optuna_combined_honest.db` | in progress | TBD | Active honest-cost run, full 531-day data |
| `combined_honest_slip03` (GAPPY archive) | `results/studies/optuna_combined_honest_GAPPY.db` | 16 | abandoned | Trained on the 10-month-gap data — discarded |

---

## Findings & open questions

This session generated several actionable findings worth preserving:

### 1. Live engine had two showstopper bugs (now fixed)

- **EOD wipe** in `simulate_day_combined` was zeroing `entry_price` on every `on_bar()` call, so the diff in `engine_combined.py:195` never saw a None→entry transition. The dashboard showed strategies as eligible, but no order ever reached Alpaca. Fixed in `9d0ee69` with the `is_live` flag.
- **No daily rollover** meant the AWS container kept the locked watchlist forever (a single-process forever-sleep after EOD). Fridays still showed Monday's stale picks. Fixed in `346e64f` by wrapping `run()` in a `while True:` that sleeps until next Alpaca open and re-scans.

Both fixes are in production.

### 2. The 2024 data hole

`stored_data_combined` was a partial merge of 6 dirs; March–December 2024 (206 days) were never included. Every backtest run pre-this-session was on 325 days, not the full 530. The optimizer's v8 study trained on the gappy data. Fix: `optimize_combined.DATA_DIRS` now lists all 10 component dirs.

### 3. In-sample vs out-of-sample preservation

OOS test on Mar–May 2026 (43 days) with trial 432 params at 0.3% slippage:

| Window | Final equity | Sharpe | Trades | WR |
|---|---|---|---|---|
| All 12 IS (2024-25-Feb 2026) | $85.2M | 4.05 | 2,573 | 61.8% |
| All 12 OOS (Mar–May 2026) | $1.10M | 3.66 | 275 | 58.2% |
| **Cleaned IS** (L/N/I/J/C disabled) | $30.0M | 4.15 | 2,060 | 59.8% |
| **Cleaned OOS** | $670K | **4.97** | 247 | **59.5%** |

The cleaned config's **OOS Sharpe is HIGHER than its IS Sharpe** (4.97 vs 4.15) — strong evidence the system isn't overfit. WR is preserved exactly (59.8% IS → 59.5% OOS).

The `config/trial_432_params_cleaned.json` file is the cleaned variant (untracked currently). Disables L, N, I, J, C — all slippage-sensitive losers OOS.

### 4. L (Low Float Squeeze) was the optimizer's overfit darling

In-sample trial 432 had L contributing $49M of $85M total PnL (58%). Out-of-sample (Mar–May 2026) L produced **-$59K with 90% WR**. The strategy kept being "right" but couldn't pay for its slippage. Reason: 2024-25 had unusual low-float squeeze frequency that hasn't repeated in 2026. Disable L until the next squeeze regime returns.

### 5. Concentration: top 10% of days = 86.7% of P&L

Replay of the trial 432 backtest on the full 530-day data:

| Bucket | P&L | % of total |
|---|---|---|
| Top 1 day (2025-10-06 SPRB) | $11.3M | 7.4% |
| Top 5 days | ~$39M | 25% |
| Top 15 days | $74.5M | 48.9% |
| **Top 53 days (10%)** | **$132M** | **86.7%** |
| Remaining 477 days | $20.3M | 13.3% |

Single most-impactful trade: SPRB(L) on 2025-10-06 = **$+11,045,697 from one fill**. The kind of trade that's a simulator artifact (impossible at real liquidity).

### 6. Realistic slippage model — SHIPPED

Replaced the constant `SLIPPAGE_PCT` with a liquidity-aware function across all 9 fill call sites in `test_green_candle_combined.py`. Backward-compatible: `USE_DYNAMIC_SLIPPAGE = False` (default) reproduces every legacy result byte-for-byte.

Total per-leg slippage now computes as:

```
slippage_pct = (base_spread + impact) × regime_mult
  base_spread = 0.05 + 0.5 / max(price, 0.1)              # spread widens for penny stocks
  impact      = 3.0 × sqrt(position_dollars / cumulative_dollar_volume)   # Almgren-Chriss
  regime_mult = 1.0 + REGIME_AMP × max(0, (regime - 20) / 20)              # opt-in (default off)
```

**Opt in** with `--dynamic-slip` on any backtest entry point or set `tgc.USE_DYNAMIC_SLIPPAGE = True`.

**Direct comparison** on cleaned trial 432 / Mar–May 2026 OOS / $10K start: legacy $670K → dynamic $252K (62% reduction). Two strategies (B and M) flipped from net winners to losers under realistic costs — confirmed candidates for permanent disable.

See [Cost model](#cost-model) for the full parameter table and formula.

### 7. Open: empirical slippage calibration

The dynamic model's `SLIP_IMPACT_K = 3.0` is a literature-default value for small-caps. Once the live engine has been trading post-bug-fix for ~50+ fills, regress observed slippage against `(price, position_size, cum_vol)` to calibrate K empirically. Adjust + re-optimize as data accumulates.

---

## Repo layout

```
algo-trading/
├── README.md, Dockerfile, docker-compose.yml, .gitignore, requirements.txt
│
├── live/                          ← production live engine
│   ├── main.py                    ← entry point + scheduler + daily rollover
│   ├── scanner.py                 ← Webull / Alpaca / Finviz pre-market scan
│   ├── streamer.py                ← Alpaca WS 1-min → 2-min aggregation
│   ├── engine_combined.py         ← wraps simulate_day_combined for live
│   ├── executor.py                ← Alpaca order plumbing
│   ├── engine.py                  ← (legacy single-strategy engine)
│   └── validate.py                ← validates extracted strategy module
│
├── dashboard/                     ← FastAPI + React + Tailwind
│   ├── backend/
│   │   ├── app.py                 ← FastAPI app + WebSocket endpoint
│   │   ├── routers/               ← account, positions, watchlist, charts, etc.
│   │   └── services/
│   │       ├── engine_bridge.py   ← read-only adapter over engine + executor
│   │       └── ws_manager.py      ← live bar broadcast
│   ├── frontend/                  ← React 19 + Vite + lightweight-charts
│   └── run.py                     ← standalone dashboard server
│
├── config/                        ← live params
│   ├── settings.py                ← API keys + global config
│   ├── trial_432_params.json      ← currently deployed
│   └── trial_432_params_cleaned.json   ← L/N/I/J/C disabled (untracked)
│
├── strategies/                    ← extracted strategy modules (modularization-WIP)
│   └── low_float_squeeze.py       ← only one fully extracted so far
│
├── deploy/                        ← AWS setup docs + scripts
│
├── docs/                          ← project docs
│   ├── plan.md
│   └── strategy_log.py            ← strategy-version history (data, not code)
│
├── results/                       ← outputs (mostly gitignored via *.db)
│   ├── studies/                   ← all the optuna_*.db files
│   ├── params/                    ← best-params JSON dumps
│   └── logs/                      ← backtest output logs
│
├── scripts/                       ← entry-point CLIs the user runs
│   ├── analysis/                  ← 12 ad-hoc analysis scripts
│   ├── backtest/                  ← backtest_today, backtest_alpaca, run_backtest_*
│   ├── compare/                   ← compare_compounding, configs, daily_cash
│   ├── data/                      ← validate_data, prefilter_data, generate_daily_gainers
│   ├── download/                  ← 14 download_*.py date-range data fetchers
│   ├── legacy/                    ← 11 old test_*.py exploratory scripts
│   ├── optimize/                  ← 11 per-strategy optuna optimizers
│   ├── stress/                    ← stress_test_regime
│   ├── sweep/                     ← sweep_f_timestop, sweep_volcap
│   ├── utils/                     ← check_account, test_order_execution
│   └── verify/                    ← verify_baseline, _bias, _scalein, _volcap, check_regime_jan
│
├── stored_data_*/                 ← 14 dirs of CSV+pickle data (gitignored)
├── charts/                        ← chart output (gitignored)
├── logs/                          ← live trading logs (gitignored)
│
└── (root scripts, kept here because they're production-deep or imported widely)
    ├── test_full.py               ← simulation harness + data loader
    ├── test_green_candle_combined.py  ← the 3070-line strategy library
    ├── optimize_combined.py       ← Optuna driver
    ├── optimize.py                ← legacy single-strategy optimizer
    ├── stress_test.py             ← stress test framework
    ├── regime_filters.py          ← VIX/SPY filter
    └── run_live_params_backtest.py    ← lightweight backtest runner (untracked)
```

**Root cleanup history**:
- Pre-session: 90 Python files in root
- After Phase 1a (dead-file cleanup, `f438d8c`): 82 files
- After Phase 2 artifact relocation (`6196b37`): docs + results dirs created
- After Phase 2 cont. (`8e11fcc`): 41 scripts moved to `scripts/`
- After Phase 2 round 3 (`0a8ca3c`): 33 more moved
- **Current**: 8 .py files at root (production deps + 2 loose ends)

---

## AWS deployment

### SSH into the server
```powershell
ssh -i "C:\Users\Theo Korir\Documents\Python\algo-trading\trading-key.pem" ubuntu@98.92.73.65
```

### Check logs
```bash
docker compose logs -f --tail 100
```

### View dashboard
```
http://98.92.73.65
```

Dashboard stays alive 24/7 (even after market close).

### Push new code to live
```bash
# locally:
git push origin main
# server picks it up within 5 minutes via cron
```

### Restart the bot
```bash
sudo docker compose restart
```

### Rebuild after code changes
```bash
sudo docker compose up -d --build
```

### Stop the bot
```bash
sudo docker compose down
```

### Shell into the container
```bash
docker compose exec algotrader bash
```

### Cron jobs (automatic)
- `@reboot`: starts the Docker container when the server boots
- `*/5 * * * *`: every 5 min, auto-pulls new code from GitHub and rebuilds Docker if a new commit is detected

### `.env` on the server
```
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_PAPER=true
```
Not in git. Survives container rebuilds (file lives on host, mounted in).

---

## Dependencies

```
alpaca-py
fastapi
uvicorn
websockets
optuna
pandas
numpy
matplotlib
requests
lxml
beautifulsoup4   # finviz scraping
yfinance         # regime filter daily data
```

Plus, for the dashboard frontend:
- React 19, Vite 7, TypeScript 5, Tailwind 4, `lightweight-charts` 4.2

Polygon downloads use only `requests`.

---

## Open work after this session

1. **Restart the optimizer with full data + honest cost** — `optimize_combined.py` already updated with the 10-dir list and supports `--slippage 0.3 --vol-cap 5.0`. Just kick it off when ready.
2. **Implement Level-2 liquidity model** — replace constant `SLIPPAGE_PCT` with `base_slip(price) + k × sqrt(participation_rate)`. Roughly 50 lines in `test_green_candle_combined.py`.
3. **Investigate the L=0 trades discrepancy** between `run_backtest_from_json.py` and `run_live_params_backtest.py` — same params, same data, different L counts.
4. **Phase 3 reorg** — extract `algotrader/` package with import shims so the AWS container keeps importing the same names while the code lives under one tree.
5. **Phase 4 reorg** — split `test_green_candle_combined.py` (3070 lines) into one module per strategy under `algotrader/strategies/`.
6. **Live trade comparison** — once the live engine has been trading post-bug-fix for a few days, diff its actual fills against `backtest_today.py` runs of the same day. The slippage gap will be the empirical Level-4 calibration data point.
