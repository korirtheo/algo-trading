# Algo Trading — Gap-Up Day Trading System

Automated intraday trading system targeting small-cap gap-up stocks. 20+ candidate
strategies tuned via Optuna with regime classification, walk-forward validation,
and execution-aware slippage modeling. Runs on Alpaca paper trading with a
real-time React dashboard.

- **Universe**: small-cap US equities gapping ≥ 8% pre-market, price ≤ $50
- **Timeframe**: 2-minute candles, 9:30 AM → 3:45 PM ET
- **Capital**: $25K–$27.5K starting cash, full-balance sizing with stacked liquidity caps
- **Live**: Alpaca paper (IEX feed), Docker Compose on AWS EC2

---

## Table of contents

1. [Quick start](#quick-start)
2. [Live trading architecture](#live-trading-architecture)
3. [Regime framework](#regime-framework)
4. [Slippage models](#slippage-models)
5. [Backtest](#backtest)
6. [Optimization](#optimization)
7. [Walk-forward validation](#walk-forward-validation)
8. [Strategies](#strategies)
9. [Notable trial configs](#notable-trial-configs)
10. [Repo layout](#repo-layout)
11. [AWS deployment](#aws-deployment)
12. [Dependencies](#dependencies)

---

## Quick start

```powershell
# 1. Install deps
pip install -r requirements.txt

# 2. Set API keys (or edit config/settings.py)
$env:ALPACA_API_KEY  = "..."
$env:ALPACA_API_SECRET = "..."

# 3. Run live (paper)
python -m live.main

# 4. Dashboard at http://localhost:8000
```

Common commands:

| Task | Command |
|---|---|
| Run live (laptop) | `python -m live.main` |
| Scanner only | `python -m live.main --scan-only` |
| Disable dashboard | `python -m live.main --no-dash` |
| Backtest a params JSON | `python scripts/backtest/run_backtest_from_json.py --json config/trial_818_squeeze_extracted.json <data_dir>` |
| Backtest with honest slippage | add `--dynamic-slip` (legacy 2-min model: `tgc.USE_2MIN_SLIPPAGE=True` in code) |
| Single-day backtest (today) | `python scripts/backtest/backtest_today.py` |
| Train a specialist | `python optimize_combined.py --trials 500 --db <name>.db --study <name> --regime-filter squeeze` |
| Walk-forward Optuna | `python scripts/analysis/walk_forward_optuna.py --n-trials 500 --n-startup 150` |
| Slippage sensitivity | `python scripts/analysis/slippage_sensitivity_w3.py` |
| Dashboard standalone | `python -m dashboard.run --port 8000` |
| Frontend dev server | `cd dashboard/frontend && npm run dev` |

---

## Live trading architecture

### Daily schedule (ET)

| Time | Action |
|---|---|
| 7:00 / 7:30 / 8:30 / 9:00 / 9:25 AM | Pre-market scans |
| 9:27 AM | Final scan — watchlist locked (top 20 by gap %) |
| 9:30 AM → 3:45 PM | 2-min bars stream in → strategies fire → orders to Alpaca |
| 3:45 PM | EOD close-all |
| Overnight | Dashboard stays live; main process sleeps until ~7:00 ET next session |

Auto-rolls over each weekday using `Alpaca.get_clock().next_open` — weekends and
holidays handled automatically.

### Components

| File | Role |
|---|---|
| [live/main.py](live/main.py) | Entry point — scanner, engine, streamer, dashboard, rollover |
| [live/scanner.py](live/scanner.py) | Webull + Alpaca + Finviz pre-market scan |
| [live/streamer.py](live/streamer.py) | Alpaca WS 1-min → 2-min bar aggregation |
| [live/engine_combined.py](live/engine_combined.py) | Runs `simulate_day_combined` per bar; emits ENTRY/EXIT/PARTIAL |
| [live/executor.py](live/executor.py) | Alpaca order execution + volume-cap enforcement + PDT gate |
| [live/halt_monitor.py](live/halt_monitor.py) | Intraday halt-resume scanner |

### Scanner sources

1. Webull `topGainers` API (primary — SIP-quality volume + float)
2. Alpaca screener `movers` (fallback)
3. Finviz scrape (float backfill only)

Top 20 by gap % are kept; Alpaca 1-min PM bars (4 AM → 9:30 ET) determine each
candidate's premarket high.

### Halt-Resume Scanner (intraday discovery)

A second discovery channel: when an NMS-listed stock is halted intraday and
resumes, the monitor adds it to the WS stream and arms a dedicated `halt_resume`
strategy. Source data is NASDAQ's public halt log (free).

- **Eligibility**: T1/T2/T12/LUDP/LUDS halts, $1–$50, max float 50M
- **Entry**: first post-resume bar with green body + ≥50K volume
- **Exit**: stop −10%, target1 +25% (50% partial), target2 +50%, trail 5%
  (armed at +10%), 30-min time stop, EOD close

Toggle with `HALT_MONITOR_ENABLED` in [config/settings.py](config/settings.py).
Full design: [docs/halt_resume_scanner_spec.md](docs/halt_resume_scanner_spec.md).

### PDT gate

Per [config/settings.py](config/settings.py): `PDT_EQUITY_FLOOR = $25,000`,
`PDT_DAYTRADES_MAX = 3`. The executor blocks new entries when equity drops below
the floor AND daytrade count reaches 3 in the rolling 5-day window. Prevents
account flag-up while under-capitalized.

---

## Regime framework

Markets cycle through structurally different shapes. The classifier in
[strategies/regime_gate.py](strategies/regime_gate.py) categorizes each day in
two layers:

### Layer 1: trade vs skip (current production)

```
classify_regime(picks) -> "dead" | "normal" | "squeeze"
```

Thresholds (tuned 2026-06): `SQUEEZE_N50_MIN = 4` (≥4 picks with gap > 50%) AND
`SQUEEZE_MAX_GAP_MIN = 150%`. `dead` if fewer than 2 picks gap > 20%. Everything
else is `normal`.

Tuning yielded Sharpe 2.09 on the regime-gated full window vs the prior wide
thresholds.

### Layer 2: shape taxonomy (analysis layer)

[scripts/analysis/squeeze_taxonomy_2021_2026.py](scripts/analysis/squeeze_taxonomy_2021_2026.py)
splits days into 7 buckets based on leader price + count of strong gappers:

| Shape | Definition | Behavior |
|---|---|---|
| **broad-squeeze** | 4+ gappers >50%, leader ≥ $3 | Peaks +19% at 14 min, fades -23% by EOD |
| **liquid-normal** | Leader ≥ $3, decent vol | Gentle gap fade, +9% peak |
| **microcap-thin** | Leader < $1 | Pump-and-dump, +13% peak at 8 min then fade |
| **thin-microcap** | Leader $1-3 | Same shape as microcap-thin (nearly identical) |
| **mega-cap** | Leader ≥ $30 | Stable, +3% peak at 52 min, only -5% fade |
| **corp-action** | Gap > 500% AND leader < $1 | Reverse-split artifacts; SKIP |
| **dead/empty** | < 2 picks gapping > 20% | SKIP |

Empirical findings ([results/temporal_shape_rolling20.png](results/temporal_shape_rolling20.png)):

- 2019: 65% microcap-thin/thin-microcap (pump-and-dump era)
- 2021: 62% liquid-normal (SPAC era)
- 2023: 40% empty (dead year)
- 2025 Q3: 21% broad-squeeze (peak of the "real squeeze" era)
- **2026 Q2: 53% microcap-thin — current dominant regime**

Lag-1 autocorrelation on `n_above_50` is **+0.636** — yesterday's squeeze count
predicts today's. Quarterly drift is real, not noise.

---

## Slippage models

The simulator supports a hierarchy of slippage models, controlled by flags in
[test_green_candle_combined.py](test_green_candle_combined.py).

### Tier 1 — Constant (legacy)

`USE_DYNAMIC_SLIPPAGE = False`: every fill uses `SLIPPAGE_PCT` (default 0.05%).
Reproduces legacy results byte-for-byte.

### Tier 2 — Dynamic Almgren-Chriss (cumulative volume)

`USE_DYNAMIC_SLIPPAGE = True`:
```
slippage = base_spread + K × sqrt(position_$ / cumulative_$_volume)
  base_spread = 0.05 + 0.5 / max(price, 0.1)   # widens for penny stocks
  K           = 3.0                              # impact coefficient
```
Used by historical Optuna runs. Overstates available depth because
"cumulative volume since premarket" isn't actually liquid right now.

### Tier 3 — 2-min single-window

`USE_2MIN_SLIPPAGE = True`:
```
slippage_denominator = close[last_bar] × volume[last_bar]   # 1 × 2-min bar
position_cap         = min(5% cumulative, 15% × last_2min_vol)
```
More honest but can over-penalize entries during isolated calm bars.

### Tier 4 — Multiwindow + volatility adjustment (current best)

`USE_MULTIWINDOW_SLIPPAGE = True`:
```python
v_2min   = last_1_bar  × close
v_6min   = last_3_bars × close
v_10min  = last_5_bars × close

V_eff = max(v_2min, 0.5 × v_6min, 0.25 × v_10min)
range_pct = (last_10min_high - last_10min_low) / price × 100
V_eff_adj = V_eff / (1 + min(range_pct/5, 1.0))   # cap 2x

position_cap = min(
    5% × cumulative_$_vol,                       # sanity ceiling
    8% × v_10min,                                # regime cap
    15% × V_eff_adj,                             # execution cap
)
impact = K × sqrt(position_$ / V_eff_adj)
```

This handles bursty volume correctly: sustained-deep markets get the right
liquidity credit, isolated spikes don't fool the model.

### Empirical W3 walk-forward sensitivity (train 2021-23, test 2024)

| Model | Final equity | Sharpe (% returns) | Wins |
|---|---|---|---|
| Cumulative K=3 (legacy headline) | $4,381,635 | 4.99 | 111/206 |
| 2MIN single-window K=3 | $1,066,468 | 3.69 | 103/206 |
| **MULTIWIN + vol-adj K=3 (honest)** | **$1,176,192** | **3.97** | 103/206 |
| MULTIWIN K=6 (2x stress) | $60,677 | 1.43 | 99/206 |
| MULTIWIN K=9 (3x stress) | $25,029 | 0.84 | 101/206 |

**Honest deployment expectation at realistic K=4-5: $300-700K annualized from
$25K start, Sharpe ~2.5-3.5.** Win-rate is stable across K — the strategy
correctly identifies tradeable days regardless of impact model.

Calibrate K from real fills via `fills_calibration.csv` when ≥50 trades
logged. Replace the K range with a single empirical number.

---

## Backtest

All entry points wrap the same simulator: `simulate_day_combined()` in
[test_green_candle_combined.py](test_green_candle_combined.py).

### Data layout

Each `stored_data_*/` directory:
```
stored_data_X/
├── daily_top_gainers.csv     ← per-day top-20 gappers
├── daily/<TICKER>.csv         ← daily OHLCV
├── intraday/<TICKER>.csv      ← 2-min OHLCV (simulation input)
└── fulltest_picks_*.pkl       ← cached merged picks
```

`test_full.load_picks_for_dir()` builds the picks pickle on demand
(`MIN_PM_VOLUME = 250K`, gap ≥ 2%, price ≤ $50). `test_full.load_all_picks([dirs])`
merges across dirs by date. Pickle is shared across all studies once built.

### Currently available data (2019-2026)

| Year | Dir(s) | Days |
|---|---|---|
| 2019 | `stored_data_2019` | 252 |
| 2020 | `stored_data_2020` | 253 |
| 2021 | `stored_data_2021` | 252 |
| 2022 | `stored_data_2022` | 251 |
| 2023 | `stored_data_2023` | 250 |
| 2024 | `stored_data_jan_mar_2024`, `apr_jun_2024`, `jul_sep_2024`, `oct_dec_2024` | 206 |
| 2025 | `stored_data_jan_mar_2025`, `apr_jun_2025`, `jul_2025`, `oos` | 246 |
| 2026 | `stored_data`, `stored_data_mar_may_2026` | 81 |

### Entry points

| Script | Use case |
|---|---|
| [scripts/backtest/run_backtest_from_json.py](scripts/backtest/run_backtest_from_json.py) | Canonical full backtest with charts |
| [run_live_params_backtest.py](run_live_params_backtest.py) | Lightweight (no charts) for quick checks |
| [scripts/backtest/backtest_today.py](scripts/backtest/backtest_today.py) | Single-day backtest of today's picks |
| [scripts/analysis/regime_switch_backtest.py](scripts/analysis/regime_switch_backtest.py) | Routing-based deployment simulator (squeeze→A, normal→B) |
| [scripts/analysis/slippage_sensitivity_w3.py](scripts/analysis/slippage_sensitivity_w3.py) | Stress test a config across K=3/6/9 + slippage models |

After every full backtest, [stress_test.py](stress_test.py) runs five robustness
checks (top-10%-removed PnL, alternate slippages, Kelly, Monte Carlo, drawdown).

---

## Optimization

Optuna TPE sampler over ~130 dimensions (per-strategy params + enable flags
+ priorities). Default score:
```
score = total_pnl × min(profit_factor, 3.0)
```

A Sharpe-aware objective is available via the `USE_SHARPE_OBJECTIVE` flag in
[optimize_combined.py](optimize_combined.py):
```
score = total_pnl × min(sharpe_pct_returns, 4.0)
```
Recommended for any new run — current PF-based objective rewards
higher-PnL-higher-variance basins that don't deploy well (see W3 sensitivity
above).

Hard constraints: trials with `n_trades < 30` or `pf < 0.5` are rejected with
score −9999.

Studies persist to SQLite (`*.db`), fully resumable. Best params dump to JSON
after every new-best.

```powershell
# Full-window combined Optuna (uses DATA_DIRS list at top of optimize_combined.py)
python optimize_combined.py `
    --trials 2000 --dynamic-slip `
    --db results/studies/run.db --study my_run `
    --params-out results/params/best.json

# Regime specialist (filters days by classify_regime() before training)
python optimize_combined.py --regime-filter squeeze ...
python optimize_combined.py --regime-filter normal  ...

# Inspect current best from existing DB
python optimize_combined.py --dump-best --db results/studies/run.db --study my_run
```

Throughput: ~16–24 trials/hour on a typical laptop with `n_jobs=cpu_count()//2`.

---

## Walk-forward validation

The "best Optuna trial" on the full 2021-2026 window may overfit to a specific
regime vintage. Walk-forward replays the *real-time* deployment pattern:
re-train Optuna on each rolling/expanding window, deploy the resulting params
on the next out-of-sample period.

[scripts/analysis/walk_forward_optuna.py](scripts/analysis/walk_forward_optuna.py)
supports both **expanding** and **2-year sliding** windows:

```
Expanding (default):
  W1: train 2021                  test 2022
  W2: train 2021-22               test 2023
  W3: train 2021-23               test 2024
  W4: train 2021-24               test 2025
  W5: train 2021-25               test 2026

2-year sliding (edit WINDOWS in script):
  W3: train 2022-23               test 2024
  W4: train 2023-24               test 2025
  W5: train 2024-25               test 2026
```

```powershell
# Full walk-forward with sharpe + slippage-aware objective (recommended)
python scripts/analysis/walk_forward_optuna.py `
    --n-trials 500 --n-startup 150 `
    --use-2min-slippage --sharpe-objective
```

Each window's best params + forward backtest are saved as JSON for later
chart generation and routing decisions.

ETA: ~3 days at 500 trials × 5 windows on a single laptop. Each window's DB
resumes cleanly if the process is killed.

---

## Strategies

20 candidates + halt-resume. A subset is enabled per config (set per
`enable_X` in the params JSON). Codes used throughout the codebase:

| Code | Name | Core idea |
|---|---|---|
| H | High Conviction | 35%+ gap + body ≥ 4% + 2nd green + new HOD + volume |
| G | Big Gap Runner | 30%+ gap + 2nd green + new HOD |
| A | Quick Scalp | 15%+ gap + body ≥ 4% + 2nd green + new HOD |
| F | Catch-All | 10%+ gap + 2nd green |
| D | Opening Dip Buy | Gap + spike + dip + VWAP reclaim |
| V | VWAP Reclaim | Sub-VWAP for N candles, then reclaim with volume |
| P | PM High Breakout | Above PM high, pullback, bounce |
| M | Midday Range Break | Morning spike + midday base + breakout |
| R | Multi-Day Runner | Day-1 gap + day-2 pullback + bounce |
| W | Power Hour Breakout | Late-day breakout from consolidation |
| O | Opening Range Breakout | First N candles' range break with volume |
| B | Red-to-Green | Red candle 1 + dip + reclaim of open |
| K | First Pullback | Morning run + orderly pullback + bounce |
| C | Micro Flag | Spike + tight base + breakout |
| S | Stuff-and-Break | Multiple HOD rejections then final breakout |
| E | Gap-and-Go RelVol | Extreme PM volume → immediate momentum entry |
| I | PM High Immediate | Breaks PM high within first few candles |
| J | VWAP + PM Breakout | Near-VWAP + PM high break combo |
| N | HOD Reclaim | Old HOD reclaim after pullback |
| L | Low Float Squeeze | Float ≤ 15M + 30%+ gap + HOD break + volume surge |
| HALT | Halt-Resume | First post-halt-resume bar runner (intraday discovery) |

`STRAT_PRIORITY` resolves conflicts when multiple strategies fire on the same
bar — lowest priority number wins (single-position-at-a-time engine).

---

## Notable trial configs

| Config | Trained on | IS PF | OOS notes | Strategies | Recommended for |
|---|---|---|---|---|---|
| `trial_6_extracted.json` | 2024-25 | ~1.4 | Generalist baseline, robust across years | H,G,A,F,V,M,R,B,J,I | Conservative live default |
| `trial_326_squeeze_extracted.json` | 2021-26 (squeeze days) | 2.09 | **Best on 2026 squeeze days** ($5,844/day) | H,A,F,V,P,R,K,C,S,J,I,N | Live broad-squeeze days |
| `trial_541_squeeze_extracted.json` | 2021-26 (squeeze days) | 2.73 | Won 2019 OOS (+$2,056) | similar to #326 | Mixed |
| `trial_587_squeeze_extracted.json` | 2021-26 (squeeze days) | 2.91 | Loses 2026 to #326 (overfit signal) | H,G,A,F,V,P,C,I,J,N | Reference only |
| **`trial_818_squeeze_extracted.json`** | 2021-26 (squeeze days) | 2.81 | **Largest OOS aggregate ($1.7M across 2019/20/26)** | G,P,M,W,B,I,J,L | Current best squeeze |
| `trial_571_normal_extracted.json` | 2021-26 (normal days) | 2.08 | Wins 2026 normal-day Sharpe (3.12) | G,I,L | Live normal days |
| `trial_635_extracted.json` | 2024-25 IS | — | Won 2020 OOS, FAILS 2026 (vintage-locked) | — | Reference only |

The current live config is set via `LIVE_PARAMS_PATH` env var (defaults to
`config/trial_6_extracted.json` per the runbook in [docs/aws_migration.md](docs/aws_migration.md)).

**Routing recommendation for live deployment** (from [scripts/analysis/regime_switch_backtest.py](scripts/analysis/regime_switch_backtest.py)):

```
shape = classify_shape(today_picks, yesterday_features)
if shape in ("dead", "corp-action"): SKIP
elif shape == "broad-squeeze":       use trial_818 (or trial_326 for conservatism)
elif shape in ("microcap-thin", "thin-microcap"): use trial_6 (no specialist yet)
elif shape == "liquid-normal":       use trial_571 (or trial_6 generalist)
elif shape == "mega-cap":            use trial_6 (only 36 days/7yr, too rare to specialize)
```

ROUTED strategy on 2026 OOS: $462,535 final from $25K — vs static-#818
($274K), #571-only ($230K), or #6-only ($341K).

---

## Repo layout

```
algo-trading/
├── live/                          ← production live engine
│   ├── main.py                    ← entry point + scheduler
│   ├── scanner.py                 ← pre-market scan
│   ├── streamer.py                ← Alpaca WS bar aggregation
│   ├── engine_combined.py         ← per-bar simulation wrapper
│   ├── halt_monitor.py            ← intraday halt-resume scanner
│   └── executor.py                ← Alpaca order plumbing + PDT gate
│
├── dashboard/                     ← FastAPI + React
│   ├── backend/                   ← routers + engine bridge + slippage panel
│   ├── frontend/                  ← React 19 + Vite + lightweight-charts
│   └── run.py                     ← standalone dashboard server
│
├── strategies/                    ← extracted strategy modules
│   ├── low_float_squeeze.py
│   ├── halt_resume.py
│   └── regime_gate.py             ← classify_regime() + tuned thresholds
│
├── config/                        ← settings + live params
│   ├── settings.py                ← keys, MIN_PM_VOLUME, PDT, etc.
│   ├── trial_6_extracted.json     ← generalist baseline
│   ├── trial_326_squeeze_extracted.json
│   ├── trial_541_squeeze_extracted.json
│   ├── trial_587_squeeze_extracted.json
│   ├── trial_818_squeeze_extracted.json  ← current squeeze best
│   ├── trial_571_normal_extracted.json   ← current normal best
│   └── trial_432_params.json      ← BASELINE used by all OOS scripts
│
├── scripts/
│   ├── backtest/                  ← canonical backtest entry points
│   ├── download/                  ← Alpaca data downloaders (2019-2026)
│   ├── analysis/                  ← regime/taxonomy/walk-forward/sensitivity
│   ├── compare/  data/  stress/  sweep/  verify/  utils/  legacy/
│
├── docs/                          ← project docs + specs
│   ├── aws_migration.md           ← AWS account migration runbook
│   ├── oracle_cloud_migration.md  ← Oracle Always Free alternative
│   ├── halt_resume_scanner_spec.md
│   └── ...
│
├── results/                       ← Optuna DBs, params JSON, backtest logs (gitignored)
│   ├── walk_forward/              ← walk-forward W1-W5 outputs
│   └── ...
├── stored_data_*/                 ← intraday + daily CSVs (gitignored)
├── charts/                        ← chart output (gitignored)
├── logs/                          ← live logs (gitignored)
├── deploy/                        ← AWS setup scripts + docs
│
└── (root)                         ← production deps imported widely
    ├── test_full.py               ← simulation harness + data loader
    ├── test_green_candle_combined.py  ← strategy library (~3K lines)
    ├── optimize_combined.py       ← Optuna driver + objective
    ├── stress_test.py             ← stress-test framework
    └── regime_filters.py          ← VIX/SPY filter (separate from regime_gate.py)
```

---

## AWS deployment

The bot runs on EC2 Ubuntu via Docker Compose with hardened SSH (port 2222).

**Current production host: 54.172.65.25 (Elastic IP).** Migration runbook:
[docs/aws_migration.md](docs/aws_migration.md).

```bash
# SSH in (port 2222, not 22)
ssh -p 2222 -i trading-key-v2.pem ubuntu@54.172.65.25

# Logs
cd ~/algo-trading && sudo docker compose logs -f --tail 100

# Restart
sudo docker compose restart

# Rebuild after code changes
sudo docker compose up -d --build

# Stop
sudo docker compose down
```

API keys live in `/home/ubuntu/algo-trading/.env`:

```
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_PAPER=true
LIVE_PARAMS_PATH=config/trial_6_extracted.json
```

Cron:
- `@reboot` — start container on boot
- `*/5 * * * *` — pull + rebuild if new commit on `main`
  (only safe to enable after local fixes are pushed)

### Security group inbound rules

| Port | Source | Purpose |
|---|---|---|
| 2222/tcp | 0.0.0.0/0 | SSH (key-only auth — open IP is fine) |
| 80/tcp | 0.0.0.0/0 | Dashboard |
| 8000/tcp | 0.0.0.0/0 | Optional direct dashboard |

Billing alarm set at $0.01 via AWS Budgets. CloudWatch billing alarms enabled.

---

## Dependencies

Python (see [requirements.txt](requirements.txt)):

```
alpaca-py, fastapi, uvicorn, websockets,
optuna, pandas, numpy, matplotlib, scipy,
requests, lxml, beautifulsoup4, yfinance, sqlalchemy
```

Frontend ([dashboard/frontend/package.json](dashboard/frontend/package.json)):
React 19, Vite 7, TypeScript 5, Tailwind 4, `lightweight-charts` 4.2.

---

## Recent methodology findings (2026-06)

Quick reference to validation results — see project memory for full notes.

- **Regime drift is real.** Last 30 days = 56.7% microcap-thin vs all-time
  baseline 21.1%. Routing matters; static-best configs misfire on shifted regime.
- **TPE drifts toward fragile basins.** #818 (PF 2.81) > #587 (PF 2.91) on OOS
  despite lower IS PF. Higher-PF Optuna trials systematically underperform
  lower-PF trials on truly held-out years.
- **W3 walk-forward (train 2021-23, test 2024)**: $1.18M honest projection
  (Sharpe 3.97 on % returns) with multiwindow K=3 slippage. Headline cumulative
  number was $4.38M.
- **Re-entry opportunity is large.** 65.3% of 2024 ticker-days had a tradeable
  >=10% pullback + >=5% recovery setup. Median ideal-reentry PnL: 16%. Only
  14.7% break the first-leg high. Requires a new "range-reversion" strategy
  (not currently implemented).
- **Slippage K=3 is realistic for top-gainer picks.** Median pm $vol on Oct-Dec
  2024 leaders = $28.9M; 5% participation cap rarely binds. Above ~$10M equity
  it does start binding — capacity ceiling.
