# Intraday Top-Gainers Discovery — Plan

**Status**: planning, not yet implemented. Background while W21b finishes.
**Owner**: queued after current W21b deploy stabilizes.
**Origin**: idea from user 2026-06-23 — "find stocks that weren't in pre-market but become top gainers through the day."

## The opportunity

Current state:
- Pre-market scanner catches stocks already gapping before 9:30 ET. Median 8 candidates/day, never hits TOP_N=20 cap.
- ~22% of recent (2024-2026) trading days produce zero G+L signals — most because the watchlist is genuinely flat that day.
- **But what about stocks that DON'T gap pre-market and start running DURING the day?** They're invisible to our current pipeline.

These are a structurally different signal class:
- News catalyst hits at 10:30 ET → stock pops 30% in 20 minutes
- Stock breaks out of multi-day consolidation at 1:15 ET
- Halt-resume runners (T1/T12 reopen)
- Sympathy plays after a sector-mate goes vertical

None of these are in our current alpha. All are real edges in microcap pump trading.

## Why this is non-trivial

| Constraint | Implication |
|---|---|
| **Historical data doesn't exist for us** | We have NO "top gainers at 11:30 ET on 2024-08-15" snapshots. Can't backtest without reconstruction. |
| **Reconstruction is possible but expensive** | For each historical day, for each minute, compute cumulative return from open across ALL symbols → rank → store. Requires multi-symbol minute bars beyond our pre-market watchlist. |
| **Live data has rate limits** | Alpaca's screener API exists but has poll-rate caps. Polygon offers snapshots but costs $$. |
| **Engine architecture is per-day watchlist** | Currently `engine_combined.initialize_watchlist(candidates)` locks the watchlist at 9:30 ET. Adding tickers mid-day requires the halt-resume-style add_symbol pattern (already partially built). |

## Phased approach

### Phase 0: Reconstruct historical top-gainers from minute bars

Use existing minute bar data (we have at least 2-min bars for our scanner universe) to compute, for each historical day and each minute timestamp, what the "intraday top gainers" list looks like.

For each backtest day D:
1. Load minute bars for all symbols that traded that day
2. For each minute T from 9:30 to 16:00:
   - For each symbol, compute `cum_return(T) = (price(T) - open) / open * 100`
   - Filter: `volume(0..T)` >= some threshold (need real liquidity)
   - Filter: `price >= $1` (avoid sub-pennies)
   - Rank by `cum_return(T)` descending
3. Capture: for each minute, the top 10 symbols + their cum_return + volume

Persist as `stored_data*/intraday_gainers_v1.pkl` — `{date: {minute: [{ticker, cum_return, volume, ...}, ...]}}`.

Estimated size: 252 trading days × 195 minutes × ~10 symbols/snapshot × ~50 bytes = ~50 MB per year. Manageable.

**Open question**: where do we get all-symbol minute bars for historical days? Our `stored_data*/fulltest_picks_*.pkl` only has bars for pre-market watchlist members. We'd need a broader minute-bar dataset, OR generate snapshots from Alpaca's historical API.

Cheap fallback: for each day, only look at TICKERS THAT WERE EVER A TOP-GAINER on similar days. Start narrow.

### Phase 1: Live discovery monitor (no trading)

Standalone module `live/intraday_gainers_monitor.py` that:
1. Polls Alpaca screener / Polygon snapshots every 60-120 seconds during RTH (9:30-16:00 ET)
2. Filters: cum_return >= 20% from open, volume >= 500K shares, price >= $1
3. Logs each new entrant to a JSON file with timestamp
4. (optional) Hooks into news cache: tag entrant with current-day news count + catalyst flag

Run for 1-2 weeks alongside W21b live. Collect data. Don't trade.

**Output**: real intraday discovery log we can analyze afterward to see WHEN stocks tend to surface, WHAT news they have, WHETHER they continue running.

### Phase 2: Backtest with reconstructed data + new strategy

Strategy `intraday_discovery_runner` (working name):
- Eligibility: ticker enters top-10 intraday gainers between 10:00-14:00 ET
- Entry trigger: 2-min bar after discovery breaks high of prior 2 bars + volume ≥ 1.5× avg
- Exit: same hierarchy as W21b — target/stop/trail/time

Backtest on Phase 0 reconstructed data. Check:
- Hit rate vs total discoveries
- Multi-year robustness
- Overlap/orthogonality with G+L (do these days overlap with G's signal days?)

### Phase 3: News catalyst overlay

Layer news cache on top:
- News-confirmed: ticker has fresh news in last 24h (filter `news_cache/master.json`)
- Catalyst-confirmed: news isn't just scanner roundup (use `news_filter._is_scanner`)

Test: does filtering discoveries by news-confirmation improve WR? Or does it just reduce sample size?

### Phase 4: Live integration

If Phase 2/3 show positive forward edge:
1. Add `intraday_discovery_monitor` to live engine startup
2. When new discovery detected → call `engine.add_intraday_ticker(symbol)` (mirror halt-resume pattern)
3. Engine subscribes to bars for that ticker, runs the new strategy
4. Position-management: don't enter if already in G/L position (single-position-at-a-time invariant preserved)

## Open design questions

1. **Top-gainers definition**: by % gain from open? By $-volume? By unusual volume vs average? **Probably composite score.**
2. **Discovery cadence**: every minute? every 5 minutes? **Suggest 2-minute (matches our existing bar cadence).**
3. **Lookback for "did it just become" a gainer**: 5 minutes? 30 minutes? **Probably 15 minutes — fresh movers, not all-day grinders.**
4. **Time window**: trade discoveries between 10:00-14:00 ET? Avoid first 30 min (G's territory) and last hour (low-time-to-target).
5. **Position size**: same 30% equity cap as W21b? **Yes for safety.**
6. **Concurrency**: allow holding G+L position AND intraday discovery position? **Initially NO — keep single-position invariant.** Later: explore multi-position if capacity-bound.

## Risks / why this might fail

- Reconstructed historical top-gainers might be SURVIVORSHIP-BIASED if we only look at tickers that survived to be in our dataset. Need to pull broader universe.
- Phase 1 live data collection is mandatory before Phase 2 — we'd otherwise be backtesting on a fantasy of what intraday gainers looked like.
- Multi-day continuation (MDC) and Patient G both LOOKED promising and both lost money. This signal class might do the same — "broader filter catches setups that lack G's edge."
- Live API rate limits could make the discovery polling impossible at sufficient frequency.

## What we WON'T do here

- Won't try to predict WHICH stocks will become top gainers (that's a different problem — momentum factor prediction)
- Won't replace G+L. This is ADDITIVE if it works.
- Won't optimize aggressively before phase 1 produces real data — we've burned enough cycles on backtests with reconstructed-but-not-real signals.

## Estimated effort

| Phase | Effort | Compute | Dependencies |
|---|---|---|---|
| 0 (historical reconstruct) | 1-2 days code + 4-8h compute | High disk I/O, all-symbol minute bars | New data acquisition |
| 1 (live monitor, no trade) | 4-6h code | Minimal (just API polls) | Alpaca/Polygon access |
| 2 (backtest new strategy) | 1 day code + ~6h Optuna runs | Moderate | Phase 0 complete |
| 3 (news overlay) | 4h code + re-run Phase 2 | Low | Phase 2 complete |
| 4 (live integration) | 1-2 days code + careful testing | Live engine changes | All prior |

Total realistic: **~1-2 weeks** of focused work.

Related: [[multi-strategy-investigation-closed]], [[mdc-multiday-continuation-dead]], [[w21b-511-deployed]].
