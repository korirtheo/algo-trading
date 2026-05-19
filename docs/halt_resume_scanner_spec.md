# Halt-Resume Scanner — Spec

Adds a second discovery channel running alongside the pre-market scan: when an
NMS-listed stock is halted intraday and then resumes, immediately add it to the
streamer's WS subscription and arm a dedicated entry strategy. Catches the
high-EV "halt → resume runner" pattern that the pre-market scan misses entirely.

Maintains the existing repo conventions: production live code under `live/`,
strategy modules under `strategies/`, CLI scripts under `scripts/`, docs here.
No changes to existing files except clearly-marked integration hooks.

---

## Table of contents

1. [Motivation & scope](#motivation--scope)
2. [Architecture](#architecture)
3. [File map](#file-map)
4. [Data flow](#data-flow)
5. [Strategy logic](#strategy-logic)
6. [Backtest plan](#backtest-plan)
7. [Dashboard integration](#dashboard-integration)
8. [Implementation order](#implementation-order)
9. [Testing checklist](#testing-checklist)

---

## Motivation & scope

The current live engine ([live/main.py](../live/main.py)) locks its watchlist at
9:27 AM ET and streams 2-min bars for those 20 tickers only. Stocks that develop
momentum *after* the open are invisible. The biggest single source of intraday
discovery alpha is the **halt-resume runner pattern**:

- A stock is halted intraday by NASDAQ/NYSE because news is pending.
- During the halt (5-15 min), order imbalance builds.
- On resumption, the stock prints a wide gap relative to the pre-halt price.
- The first ~5 min after resumption frequently runs 10-50% on volume.

Frequency: ~1-3 plays per week across the small-cap universe under $50.
EV per play: large (target +25-50%, stop -10%, typical hit rate ~50-60%).

**In scope**:
- Live monitoring of halt-resume events
- Dynamic addition of resumed tickers to the WS bar stream
- A dedicated `halt_resume` strategy with its own entry/exit rules
- Standalone backtest tooling against historical halt data
- Dashboard panel showing recent halts/resumes

**Out of scope** (defer to later spec):
- General intraday top-mover rescan
- Sympathy/sector tracking
- Short side
- Multi-day swing holds

---

## Architecture

```
                  ┌─────────────────────────┐
                  │  live/halt_monitor.py   │  (NEW)
                  │  subscribes to news API │
                  │  emits resume events    │
                  └────────────┬────────────┘
                               │ on_resume(ticker, halt_reason, ts)
                               ▼
        ┌──────────────────────────────────────────────────┐
        │  live/main.py  (MODIFIED — wire the monitor in)  │
        │                                                  │
        │  on_resume(ticker, ...):                         │
        │      streamer.add_symbol(ticker)                 │
        │      engine.on_intraday_addition(ticker, ...)    │
        └──────────────────────────────────────────────────┘
            │                          │
            ▼                          ▼
  live/streamer.py            live/engine_combined.py
  (MODIFIED)                  (MODIFIED)
  + add_symbol(t)             + on_intraday_addition(t, src)
  + remove_symbol(t)             - inits state for new ticker
  + dynamic resubscribe         - arms halt_resume strategy

                               on each bar:
                               ▼
                       strategies/halt_resume.py  (NEW)
                       check_signal() → fire entry
                       check_exit()    → fire exit
```

The existing `simulate_day_combined` flow is untouched. The halt-resume
strategy lives outside it as a parallel evaluator on the engine side — same
pattern as the existing `RECOVERED` position handling in
[live/main.py:39](../live/main.py#L39).

---

## File map

Aligns with the established repo layout (preserves the [Phase 2 reorg](../README.md#repo-layout)).

### New files

| Path | Role |
|---|---|
| `live/halt_monitor.py` | Subscribes to Alpaca news WS stream; filters for halt/resume events; emits `on_resume(ticker, reason, ts)` callback |
| `strategies/halt_resume.py` | The strategy itself: `create_state`, `check_signal`, `check_exit` (same shape as the existing `strategies/low_float_squeeze.py`) |
| `scripts/download/download_halt_history.py` | Fetches historical halt data (NASDAQ public halt log scraper, free) for backtest |
| `scripts/backtest/backtest_halt_resume.py` | Standalone backtest of the halt-resume strategy on a date range |
| `dashboard/backend/routers/halts.py` | `/api/halts/today` endpoint returning today's halt events |
| `dashboard/frontend/src/components/HaltMonitor.tsx` | UI panel showing recent halts/resumes with their entry+exit markers |
| `docs/halt_resume_scanner_spec.md` | This document |

### Modified files (integration hooks only)

| Path | Change |
|---|---|
| `live/streamer.py` | Add `add_symbol(symbol)` and `remove_symbol(symbol)` methods that resubscribe the WS in place |
| `live/engine_combined.py` | Add `on_intraday_addition(ticker, source="halt_resume")` method that initializes per-ticker state for a mid-day discovery |
| `live/main.py` | Spawn the halt_monitor in a thread; wire its callback to the engine + streamer |
| `dashboard/backend/services/engine_bridge.py` | Add `get_halt_events()` returning the engine's recent halt-resume history |
| `dashboard/backend/app.py` | Register the new halts router |
| `dashboard/frontend/src/App.tsx` | Mount the HaltMonitor component below Diagnostics |
| `dashboard/frontend/src/api/client.ts` | Add `halts: () => fetchJSON<HaltEvent[]>('/api/halts/today')` |
| `config/settings.py` | New flags: `HALT_MONITOR_ENABLED`, `HALT_MIN_PRICE`, `HALT_MAX_PRICE`, `HALT_MAX_FLOAT`, `HALT_REASONS_TRADED` |
| `README.md` | New section under [Live trading architecture](../README.md#live-trading-architecture) |

No moves of existing files; only additions and small modifications. Phase 1-2
reorg structure is preserved.

---

## Data flow

### Live (real-time)

```
1. live/main.py starts → spawns halt_monitor in a background thread
2. halt_monitor connects to Alpaca news WS, subscribes to news topic
3. Stream emits events of types:
   - "halt"  (NASDAQ halt notification)
   - "resumption"  (NASDAQ resume notification)
4. halt_monitor filters: keep only halts with status="resumed" AND ticker
   passes price/float/reason filters
5. halt_monitor calls on_resume(ticker, reason, resume_ts) on the main thread
6. main.py handler:
   a. streamer.add_symbol(ticker) — resubscribes WS bars for ticker
   b. engine.on_intraday_addition(ticker) — inits state with halt-resume
      strategy enabled
7. Next 2-min bar for that ticker arrives → engine.on_bar()
8. Inside engine.on_bar, if ticker has source="halt_resume":
   call strategies.halt_resume.check_signal()
9. If signal fires → executor.buy() → position open
10. Subsequent bars → check_exit() → stop/target/trail/time
11. After exit → ticker can stay in WS or get unsubscribed (config flag)
```

### Backtest (offline)

```
1. scripts/download/download_halt_history.py:
   - Scrapes the NASDAQ public halt log (free, daily CSV at
     https://www.nasdaqtrader.com/trader.aspx?id=tradehalts)
   - Output: data/halts_YYYY.csv with columns
     (date, time_halted, time_resumed, ticker, reason, halt_price, resume_price)
2. scripts/backtest/backtest_halt_resume.py:
   - For each halt event in date range:
     - Load 2-min intraday bars for ticker on that date
     - Find bars after resume_ts
     - Apply strategies/halt_resume.check_signal()
     - Track equity, fills (with dynamic slippage)
   - Report per-event outcome, total PnL, WR, drawdown
3. Cross-validate against the existing 2-yr OOS windows we have
```

---

## Strategy logic

### `strategies/halt_resume.py`

Same module shape as `strategies/low_float_squeeze.py`:

```python
DEFAULT_PARAMS = {
    "min_resume_volume": 50_000,    # first post-resume candle vol >= this
    "max_entry_bars_after_resume": 5,  # only enter within 10 min of resume
    "min_price": 1.0,
    "max_price": 50.0,
    "max_float": 50_000_000,         # halt-resumes work best on smaller floats
    "min_resume_gap_pct": 0.0,       # resume_price vs halt_price (0 = any direction)

    # Position management
    "stop_pct": 10.0,
    "partial_sell_pct": 50.0,
    "target1_pct": 25.0,
    "target2_pct": 50.0,
    "trail_pct": 5.0,
    "trail_activate_pct": 10.0,
    "time_limit_minutes": 30,

    # Reasons to TRADE (NASDAQ halt codes)
    "tradable_reasons": ["T1", "T2", "T12", "LUDP", "LUDS"],
    # T1=news pending, T2=news released, T12=other regulatory,
    # LUDP/LUDS=volatility halt up/down. M1-M9 (market-wide) excluded.
}


def is_eligible(halt_event, params=None):
    """Filter halt events to ones we want to trade."""
    p = params or DEFAULT_PARAMS
    return (
        halt_event.reason in p["tradable_reasons"]
        and p["min_price"] <= halt_event.resume_price <= p["max_price"]
        and (halt_event.float_shares is None or halt_event.float_shares <= p["max_float"])
    )


def create_state(ticker, halt_event, params=None):
    """Init per-ticker state at resume."""
    p = params or DEFAULT_PARAMS
    return {
        "ticker": ticker,
        "strategy": "HALT",
        "halt_reason": halt_event.reason,
        "resume_ts": halt_event.resume_ts,
        "resume_price": halt_event.resume_price,
        "halt_price": halt_event.halt_price,
        "bars_since_resume": 0,
        "signal": False, "signal_price": None,
        "entry_price": None, "shares": 0, "position_cost": 0.0,
        "exit_price": None, "exit_reason": None, "pnl": 0.0,
        "highest_since_entry": 0.0,
        "trailing_active": False,
        "partial_taken": False,
        "partial_proceeds": 0.0,
        "done": False,
    }


def check_signal(state, c_open, c_high, c_low, c_close, c_vol, params=None):
    """Fire entry on first post-resume bar with min volume."""
    p = params or DEFAULT_PARAMS
    if state["done"] or state["entry_price"] is not None:
        return False
    state["bars_since_resume"] += 1
    if state["bars_since_resume"] > p["max_entry_bars_after_resume"]:
        state["done"] = True
        return False
    # First bar after resume with sufficient volume → fire
    if c_vol >= p["min_resume_volume"] and c_close > c_open:
        state["signal"] = True
        state["signal_price"] = c_close
        return True
    return False


def check_exit(state, c_high, c_low, c_close, minutes_in_trade, minutes_to_close,
               slippage_pct=0.05, eod_exit_minutes=15, params=None):
    """Exit on stop/target/trail/time. Same shape as L strategy."""
    # ... mirror low_float_squeeze.check_exit
```

### Why this design

- **Liquidity-aware entry**: requires meaningful first-bar volume; rejects "ghost
  resumes" where the resume goes back to quiet trading.
- **Tight time window**: only first 5 bars (10 min) after resume. Past that,
  the easy money is gone and we're in noise.
- **Halt reason filter**: excludes market-wide halts (M1-M9) which trade
  differently. Only stock-specific halts (T1, T2, T12, LUDP, LUDS).
- **Cap at $50 price**: same universe as the rest of the system.
- **Wider target than the other strategies (+25% / +50%)**: halt-resume runs are
  bigger than typical morning gappers — match the setup's payoff profile.
- **Tighter stop than reward (-10%)**: 2.5:1 reward/risk minimum on partial,
  5:1 on full target.

---

## Backtest plan

### Free historical data source

NASDAQ publishes a public halt log at
`https://www.nasdaqtrader.com/dynamic/symdir/tradehalts.txt` — daily CSV with
all halts including resume times and resume prices. **Free, no API key**, ~5
years of history. Format:

```
DATE|TIME_HALTED|ISO|SYMBOL|REASON|TIME_RESUMED|RESUME_PRICE
2024-03-15|10:23:45|...|XYZ|T1|10:38:22|3.45
```

The downloader (`scripts/download/download_halt_history.py`) fetches and
normalizes into `data/halts_YYYY.csv`. Then the backtest script joins each halt
event with the relevant `stored_data_*/intraday/{ticker}.csv` for the 2-min
bars around the resume timestamp.

### Backtest cross-validation

Run on the same windows we already have intraday data for:
- 2022 (251 days) — blind regime test
- Mar-Dec 2024 (206 days) — momentum regime
- 2025 (246 days) — mixed regime
- 2026 OOS slices

Expected outcome:
- ~1-3 halt-resume plays per week × ~50 weeks/year = 50-150 events/year
- ~50-60% WR × +30% avg winner × $50K position = significant added EV
- Should ADD to total PnL of the existing strategy mix, not redistribute

### Validation invariants

Same checks as the main backtest stress tests:
- Sharpe > 1 on the halt-resume contribution alone
- PF > 1.3
- Max drawdown of halt-resume slice < 25%
- Monte Carlo > 80% profitable

If halt-resume passes its own stress test independently, deploy it in live.
If it fails, the strategy is too noisy at our scale and needs tighter filters.

---

## Dashboard integration

A new "Halt Monitor" panel under Diagnostics on the dashboard, showing:

| Time | Ticker | Reason | Halt Price | Resume Price | Our Action | Result |
|---|---|---|---|---|---|---|
| 10:23 | XYZ | T1 | $3.20 | $3.45 | BOUGHT @$3.48 | +$1,250 (TARGET) |
| 11:55 | ABC | LUDP | $1.10 | $1.45 | SKIPPED (max_price) | — |
| 13:02 | DEF | T2 | $7.80 | $9.15 | BOUGHT @$9.21 | open: +12% |

Reuses the existing dashboard pattern (FastAPI router → engine_bridge method →
React component with `usePolling`).

---

## Implementation order

Plan: 1 working day (~6-8 hours) for a working live + backtest deployment.

1. **(30 min)** Spec doc — this file. **(in progress)**
2. **(45 min)** `live/halt_monitor.py` — Alpaca news WS subscriber + halt event parser
3. **(30 min)** Modify `live/streamer.py` — `add_symbol` / `remove_symbol` methods
4. **(45 min)** `strategies/halt_resume.py` — strategy logic
5. **(45 min)** Modify `live/engine_combined.py` — `on_intraday_addition` hook
6. **(30 min)** Modify `live/main.py` — wire monitor + handler
7. **(60 min)** `scripts/download/download_halt_history.py` — NASDAQ scraper
8. **(90 min)** `scripts/backtest/backtest_halt_resume.py` — standalone backtest
9. **(60 min)** Backtest the 2022 + 2024-26 windows, validate stress tests
10. **(30 min)** Dashboard router + frontend component
11. **(30 min)** README update with new section

Total: ~8 hours. Can be split across two work sessions; the live monitor (1-6)
and the backtest (7-9) and the dashboard (10) are independent.

---

## Testing checklist

### Unit-level (each module)

- [ ] `halt_monitor.py` — replay a recorded halt event JSON, verify callback fires
- [ ] `streamer.add_symbol` — start a dummy stream, add a symbol mid-run, verify the WS sees the new subscription
- [ ] `halt_resume.check_signal` — feed it synthetic bars matching the expected fire conditions, verify signal=True
- [ ] `halt_resume.check_exit` — synthetic bars hitting each exit condition (stop, target1, target2, trail, time, EOD)

### Integration

- [ ] Spawn `live/main.py` with `HALT_MONITOR_ENABLED=True`, no other strategies, on a known-halt day from history
- [ ] Verify the engine adds the ticker to bar_data, fires the strategy, places a paper order on Alpaca
- [ ] Verify the dashboard's Halt Monitor panel shows the event

### Backtest

- [ ] `scripts/download/download_halt_history.py 2022 2026` produces clean `data/halts_*.csv`
- [ ] `scripts/backtest/backtest_halt_resume.py --year 2024` produces non-zero events, plausible PnL
- [ ] Stress tests pass on the halt-resume P&L stream independently
- [ ] Adding halt-resume to the main 2024+2025 backtest INCREASES PnL vs the main run alone (additive edge)

### Repo conformance

- [ ] No code added to root — every new file is in `live/`, `strategies/`,
      `scripts/<subdir>/`, or `dashboard/<subtree>/`
- [ ] Every new script under `scripts/` has the 2-line sys.path injection
      (same pattern as Phase 2 reorg)
- [ ] All new tests + assets routed through `results/logs/` or `results/params/`
      (gitignored), not root
- [ ] README's [Quick links cheatsheet](../README.md#quick-links--cheatsheet)
      gets the new commands (`backtest halt-resume`, `download halt history`)
- [ ] README's [Repo layout](../README.md#repo-layout) tree updated with the new
      files

---

## Open questions for review before implementation

1. **Float lookup at resume time** — halt-resume eligibility includes `max_float`.
   The existing FLOAT_DATA dict is loaded once at startup from `float_data.json`.
   New halted tickers (especially small caps with recent IPOs) may not be in the
   dict. Options: (a) skip the float filter if data missing, (b) fetch from
   Webull realtime, (c) skip the ticker. Recommend (a) — be permissive when in
   doubt; the price + reason filters are already restrictive.

2. **Position concurrency** — current engine is "one position at a time". If
   the engine is already long a morning gapper when a halt-resume fires, do
   we ignore? Open a 2nd position? Recommend ignore for now (matches existing
   single-position semantics); upgrade to multi-position in a future spec.

3. **Unsubscribe after exit** — once we've exited the halt-resume position,
   should the ticker stay in WS for the rest of the day or get removed?
   Recommend remove after EOD (frees stream slots for other discoveries).

4. **Backtest data lag** — Alpaca's news API has limited history (~30 days).
   NASDAQ public halt log has ~5 years. NASDAQ log is the right source for
   backtest; Alpaca news is for live.
