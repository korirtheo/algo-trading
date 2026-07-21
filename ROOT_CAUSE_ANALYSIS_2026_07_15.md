# Root Cause Analysis: 2026-07-15 Trading & Backtest Discrepancy

## Question 1: Why is there a 15-minute bar delay?

### Streaming Architecture
✓ **We ARE using WebSockets** (`StockDataStream` from Alpaca SDK)  
✓ Not REST polling as the primary source (REST polling is fallback only, every 2 minutes)

### The Real Issue: IEX Feed Latency

```
09:30:00 ET - Subscribed to 20 watchlist symbols via WebSocket
09:30:00 ET - Intraday halt-resume tickers START arriving (LEDS, NXTC, LHAI, etc.)
09:44:00 ET - FIRST watchlist ticker bar arrives (ELVA, VIVS)
             = 14 minute delay
```

**Why the delay on watchlist tickers?**

1. **IEX feed quality varies by ticker** - Alpaca IEX feed doesn't carry all tickers equally
2. **Low-volume/micro-cap liquidity** - Our watchlist is gap-up penny stocks with low volume
3. **Real-time data vs quote delay** - IEX is a quote feed, not the full SIP (consolidated tape)
4. **Feed architecture** - Alpaca's IEX WebSocket may batch or coalesce updates

### Evidence from 7/15
- Halt-resume tickers (on SIP) arrived at 9:30-9:31 (no delay)
- Watchlist tickers (on IEX) arrived at 9:44-9:47 (14-17 minute delay)

### Why This Breaks G Strategy

**G (Gap continuation) requires:**
- First 2 bars after 9:30 to be green candles
- Volume spike confirmation
- Gap continuity

**With 14-minute delay:**
- First G bar arrives at 9:44
- By then, 14 minutes of price action is already complete
- The "gap continuity check" is looking at stale data
- Window is closed

### Solution Options

**Option A: Switch to SIP feed** (production-grade)
- Pros: No latency, complete data, handles all tickers equally
- Cons: Higher cost
- **Recommended**: Yes, this is industry standard

**Option B: Delay strategy triggering** (workaround)
- Shift 9:30 reference to 9:44 when bars actually arrive
- Update G strategy logic: if first bar at 9:44, count that as "market open"
- **Risk**: Timing-based hacks are fragile

**Option C: Add REST polling at 9:30-9:44** (quick fix)
- Poll REST for bars at market open to fill the gap
- Then switch to WS
- **Risk**: Inconsistent bar sources, deduplication complexity

**Recommendation**: Upgrade to SIP feed. The 14-minute latency makes G/L strategies non-viable.

---

## Question 2: Why Wasn't SOBR in the Watchlist?

### Scanner Results at 9:27 AM (Final Scan)

Webull returned **26 gainers >= 8.0%** gap, scanner filtered to TOP 20:

**Top 20 by gap % (from logs):**
1. VIVS (76.2%)
2. ELVA (55.3%)
3. KUST (49.5%)
4. ERNA (41.4%)
5. TGHL (29.3%)
6. VTAK (28.0%)
7. TRT (19.9%)
8. SGLY (15.4%)
9. SHMD (12.6%)
10. STKS (12.2%)
11. VEEE (12.1%)
12. GNTA (12.0%)
13. NVVE (11.3%)
14. GCTK (10.8%)
15. QNC (10.7%)
16. TAOP (10.6%)
17. CCEL (10.0%)
18. GEVO (9.8%)
19. MSC (9.7%)
20. PMN (9.4%)

**SOBR was NOT in the Webull top 30 gainers at 9:27 AM.** It either:
- Gapped up after 9:27 AM scan
- Wasn't visible on Webull scanner at that time
- Had different gap calculation

### Why Backtest Included SOBR

Backtest manually added SOBR to check what would have happened. Alpaca had data for it, so it appeared to work in backtest.

### Verdict: System Working As Designed

- Scanner takes top N from Webull
- SOBR didn't make the cut at 9:27 scan time
- Not a bug, just market rankings

---

## Question 3: Why Was the Backtest Wrong? (The Key Issue)

### The Problem

The initial backtest script (`backtest_2026_07_15.py`) showed **7 trades** but live only had **1 trade**.

```python
# WRONG - backtest_2026_07_15.py
def check_v3_entry(bars, params):
    """Custom V3 logic - NOT what live engine uses!"""
    if last['Close'] <= last['Open']:
        return False
    body_pct = (last['Close'] - last['Open']) / last['Open'] * 100
    if body_pct < params.get('v3_min_body_pct', 1.0):  # <- arbitrary
        return False
    # ... more arbitrary checks
    return True
```

### Root Causes

#### 1. **Wrong Strategy Implementation**
- Backtest used CUSTOM V3 logic
- Live engine uses `test_green_candle_combined.py` which has 12 strategies (G, L, A, F, D, V, P, M, R, W, O, B, K, C, E, I, J, N, L)
- These are fundamentally different algorithms

#### 2. **Missing Combined Engine Logic**
- Live: G and L run FIRST, then V3 overlay
- Live: Position slot management (one position at a time)
- Live: Strategy priority system (`STRAT_PRIORITY`)
- Backtest: Naive "first signal wins" logic

#### 3. **Bar Timing Mismatch**
- Backtest used: All bars from 9:30 onward (48-67 bars per ticker)
- Live: Bars started at 9:44 (11-19 bars per ticker)
- Signal timing completely different

#### 4. **Parameter Mismatch**
- `v3_min_body_pct: 1.0%` (backtest default) vs actual params
- `v3_min_volume_ratio: 1.5x` (backtest) vs actual
- None of the 20 params in v3_overlay were properly loaded

### Why 7 Fake Trades Appeared

With a 1.0% green candle body requirement and 1.5x volume over 10-bar average, many bars would qualify:

```
VIVS: 67 bars available
  - Plenty of candidates where Close > Open and volume decent
  - First entry at bar ~5-10 (artificial signal)
  - Then fake exits and fake re-entries

SOBR: 49 bars (manually added to backtest)
  - +$1,927 profit (luck - cherry picked data)
```

The backtest was essentially finding "any green candle with volume" and calling it a win.

### How to Fix

**The ONLY way to get accurate backtest:**

```python
# RIGHT way - use the actual live engine
from live.engine_combined import CombinedEngine
from test_green_candle_combined import simulate_day_combined

# Feed bars to the SAME code that live uses
engine = CombinedEngine(mock_executor)
for bar_timestamp in sorted(all_bars.keys()):
    for ticker in WATCHLIST:
        if ticker has bar at bar_timestamp:
            engine.on_bar(ticker, bar_dict)
    # This calls the REAL combined strategy logic

# Now backtest results === live results
```

---

## Why This Matters

### The Backtest Was a "Fencepost Error"
- Looked plausible (7 tickers, realistic PnL)
- Actually tested nothing (custom logic ≠ real logic)
- Could have led to false confidence in strategy

### What Actually Happened on 7/15

**Live's Reality:**
1. Bar delay (9:30 → 9:44)
2. G/L found no signals in degraded data
3. V3 overlay found TGHL green candle at 9:48
4. Traded TGHL: -$330.59
5. Day ended

**Backtest's Fantasy:**
1. Had all bars from 9:30
2. Artificial V3 logic found 7 "signals"
3. Showed -$15,906 (misaligned with live)
4. Gave false confidence in system

---

## Recommendations

### 1. **Fix Streaming Delay**
- [ ] Evaluate SIP feed upgrade cost/benefit
- [ ] Or implement Option B (shift G strategy window)

### 2. **Lock Backtest to Real Engine**
- [ ] Delete `backtest_2026_07_15.py`
- [ ] Require all backtests use `CombinedEngine` directly
- [ ] Add test: "backtest result must match live within X% PnL"

### 3. **Investigate G/L Strategy**
- [ ] Why 0 G/L signals on 7/15?
- [ ] Gap threshold too high? (MIN_GAP_PCT)
- [ ] Does G work at all with IEX feed delays?

### 4. **Add Monitoring**
- [ ] Alert if first bar arrives after 9:35 AM
- [ ] Alert if G+L=0 signals by 10:00 AM
- [ ] Track feed latency per ticker

---

## Summary Table

| Aspect | Backtest | Live | Issue |
|--------|----------|------|-------|
| Bar source | Downloaded + resampled | WebSocket (delayed) | Delay not replicated |
| First bar time | 9:30 AM | 9:44 AM | Timing mismatch |
| Bars per ticker | 48-67 | 11-19 | Data quantity diff |
| Strategy logic | Custom V3 check | Full combined engine | Logic mismatch |
| Trades generated | 7 (fake) | 1 (real) | Completely wrong |
| Root cause | Fake logic | Real engine | Not comparable |

**The 7 "trades" weren't trades at all—they were artifacts of using the wrong algorithm.**
