# Algo Trading — Gap-Up Day Trading System

Automated intraday trading system targeting small-cap gap-up stocks. ~20 candidate
strategies tuned with Optuna, running on Alpaca paper trading with a real-time
React dashboard.

- **Universe**: small-cap US equities gapping ≥ 8% pre-market, price ≤ $50
- **Timeframe**: 2-minute candles, 9:30 AM → 3:45 PM ET
- **Capital**: $10K–$25K starting cash, full-balance sizing with volume cap
- **Live**: Alpaca paper (IEX feed), deployed on AWS EC2 via Docker Compose

---

## Table of contents

1. [Quick start](#quick-start)
2. [Live trading architecture](#live-trading-architecture)
3. [Backtest](#backtest)
4. [Optimization](#optimization)
5. [Strategies](#strategies)
6. [Repo layout](#repo-layout)
7. [AWS deployment](#aws-deployment)
8. [Dependencies](#dependencies)

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
| Backtest a params JSON | `python scripts/backtest/run_backtest_from_json.py --json config/trial_432_params.json <data_dir> [...]` |
| Backtest with honest slippage | add `--dynamic-slip` to either backtest runner |
| Single-day backtest (today) | `python scripts/backtest/backtest_today.py` |
| Run optimizer | `python optimize_combined.py --trials 500 --db <name>.db --study <name>` |
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

The engine auto-rolls over each weekday using `Alpaca.get_clock().next_open`,
so weekends and holidays are handled automatically.

### Components

| File | Role |
|---|---|
| [live/main.py](live/main.py) | Entry point — scanner, engine, streamer, dashboard, rollover |
| [live/scanner.py](live/scanner.py) | Webull + Alpaca + Finviz pre-market scan |
| [live/streamer.py](live/streamer.py) | Alpaca WS 1-min → 2-min bar aggregation |
| [live/engine_combined.py](live/engine_combined.py) | Runs `simulate_day_combined` per bar; emits ENTRY/EXIT/PARTIAL |
| [live/executor.py](live/executor.py) | Alpaca order execution + volume-cap enforcement |
| [live/halt_monitor.py](live/halt_monitor.py) | Intraday halt-resume scanner (see below) |

### Scanner sources

1. Webull `topGainers` API (primary — SIP-quality volume + float)
2. Alpaca screener `movers` (fallback)
3. Finviz scrape (float backfill only)

Top 20 by gap % are kept; Alpaca 1-min PM bars (4 AM → 9:30 ET) determine each
candidate's premarket high.

### Halt-Resume Scanner (intraday discovery)

A second discovery channel: when an NMS-listed stock is halted intraday and
resumes, the monitor adds it to the WS stream and arms a dedicated
`halt_resume` strategy. Source data is NASDAQ's public halt log (free).

- **Eligibility**: T1/T2/T12/LUDP/LUDS halts, $1–$50, max float 50M
- **Entry**: first post-resume bar with green body + ≥50K volume
- **Exit**: stop −10%, target1 +25% (50% partial), target2 +50%, trail 5%
  (armed at +10%), 30-min time stop, EOD close

Toggle with `HALT_MONITOR_ENABLED` in [config/settings.py](config/settings.py) (or env var). Full design: [docs/halt_resume_scanner_spec.md](docs/halt_resume_scanner_spec.md).

---

## Backtest

All entry points wrap the same simulator: `simulate_day_combined()` in
[test_green_candle_combined.py](test_green_candle_combined.py).

### Data layout

Each `stored_data_*/` directory contains:

```
stored_data_X/
├── daily_top_gainers.csv     ← per-day top-20 gappers
├── daily/<TICKER>.csv         ← daily OHLCV
├── intraday/<TICKER>.csv      ← 2-min OHLCV (simulation input)
└── fulltest_picks_*.pkl       ← cached merged picks
```

`test_full.load_picks_for_dir()` builds the picks pickle on demand (PM volume
≥ 250K, gap ≥ 2%, price ≤ $50). `test_full.load_all_picks([dirs])` merges
across dirs by date.

### Entry points

| Script | Use case |
|---|---|
| [scripts/backtest/run_backtest_from_json.py](scripts/backtest/run_backtest_from_json.py) | Canonical full backtest with charts |
| [run_live_params_backtest.py](run_live_params_backtest.py) | Lightweight runner (no charts) for quick checks |
| [scripts/backtest/backtest_today.py](scripts/backtest/backtest_today.py) | Single-day backtest of today's picks |
| [scripts/backtest/blind_test_2022.py](scripts/backtest/blind_test_2022.py) | OOS test on 2022 data |

### Slippage modes

The simulator supports two slippage models, controlled by `USE_DYNAMIC_SLIPPAGE`:

- **Constant** (default): every fill uses `SLIPPAGE_PCT` (default 0.05%).
  Reproduces legacy results byte-for-byte.
- **Dynamic** (`--dynamic-slip`): liquidity-aware Almgren-Chriss model:

  ```
  slippage = base_spread + K × sqrt(position_$ / cumulative_$_volume)
    base_spread = 0.05 + 0.5 / max(price, 0.1)        # widens for penny stocks
    K           = 3.0                                   # impact coefficient
  ```

  Realistic for large positions where impact matters. Always recommended for
  honest backtests of compounded equity curves.

After every full backtest, [stress_test.py](stress_test.py) runs five robustness
checks (top-10%-removed PnL, alternate slippages, Kelly, Monte Carlo, drawdown).

---

## Optimization

Optuna TPE sampler over ~130 dimensions (per-strategy params + enable flags
+ priorities). Score: `total_pnl × min(profit_factor, 3.0)`.

Hard constraints: trials with `n_trades < 30` or `profit_factor < 0.5` are
rejected with score −9999.

Studies persist to SQLite (`*.db`). Trials are checkpointed on every completion,
so the optimizer is fully resumable. Best params are dumped to JSON after every
new-best so you can inspect without disturbing the run.

```powershell
# Run with honest slippage
python optimize_combined.py `
    --trials 500 --slippage 0.3 --vol-cap 5.0 `
    --db results/studies/run.db --study my_run `
    --params-out results/params/best.json

# Inspect current best
python optimize_combined.py --dump-best --db results/studies/run.db --study my_run
```

Throughput: ~16–24 trials/hour on a typical laptop with `n_jobs=cpu_count()//2`.

---

## Strategies

20 candidates; a subset is enabled in any given config (set per `enable_X`
in the params JSON). Codes used throughout the codebase:

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

## Repo layout

```
algo-trading/
├── live/                       ← production live engine
│   ├── main.py                 ← entry point + scheduler
│   ├── scanner.py              ← pre-market scan
│   ├── streamer.py             ← Alpaca WS bar aggregation
│   ├── engine_combined.py      ← per-bar simulation wrapper
│   ├── halt_monitor.py         ← intraday halt-resume scanner
│   └── executor.py             ← Alpaca order plumbing
│
├── dashboard/                  ← FastAPI + React
│   ├── backend/                ← routers + engine bridge
│   ├── frontend/               ← React 19 + Vite + lightweight-charts
│   └── run.py                  ← standalone dashboard server
│
├── strategies/                 ← extracted strategy modules
│   ├── low_float_squeeze.py
│   └── halt_resume.py
│
├── config/                     ← settings + live params
│   ├── settings.py
│   └── trial_432_params.json   ← currently deployed
│
├── scripts/                    ← CLIs grouped by purpose
│   ├── backtest/  download/  optimize/  analysis/
│   ├── compare/  data/  stress/  sweep/  verify/  utils/  legacy/
│
├── docs/                       ← project docs + specs
├── results/                    ← Optuna DBs, params JSON, backtest logs (gitignored)
├── stored_data_*/              ← intraday + daily CSVs (gitignored)
├── charts/                     ← chart output (gitignored)
├── logs/                       ← live logs (gitignored)
├── deploy/                     ← AWS setup scripts + docs
│
└── (root)                      ← production deps imported widely
    ├── test_full.py            ← simulation harness + data loader
    ├── test_green_candle_combined.py  ← strategy library (~3K lines)
    ├── optimize_combined.py    ← Optuna driver
    ├── stress_test.py          ← stress-test framework
    └── regime_filters.py       ← VIX/SPY filter
```

---

## AWS deployment

The bot runs on an EC2 Ubuntu instance via Docker Compose. The host has a cron
job that pulls + rebuilds every 5 minutes when a new commit lands on `main`.

```bash
# SSH in
ssh -i trading-key.pem ubuntu@<host>

# Logs
docker compose logs -f --tail 100

# Restart
sudo docker compose restart

# Rebuild after code changes
sudo docker compose up -d --build

# Stop
sudo docker compose down
```

API keys live in `/home/ubuntu/.env` on the host (mounted into the container,
survives rebuilds, not in git):

```
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_PAPER=true
```

Cron:
- `@reboot` — start container on boot
- `*/5 * * * *` — pull + rebuild if new commit on `main`

---

## Dependencies

Python (see [requirements.txt](requirements.txt)):

```
alpaca-py, fastapi, uvicorn, websockets,
optuna, pandas, numpy, matplotlib,
requests, lxml, beautifulsoup4, yfinance
```

Frontend ([dashboard/frontend/package.json](dashboard/frontend/package.json)):
React 19, Vite 7, TypeScript 5, Tailwind 4, `lightweight-charts` 4.2.
