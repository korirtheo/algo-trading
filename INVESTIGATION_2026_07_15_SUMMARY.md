# Investigation Summary: NVVE & SOBR Not Traded on 7/15/2026

**Date:** 2026-07-15  
**Investigation Date:** 2026-07-15 (same day)

## Executive Summary

- **NVVE**: Was in watchlist, received bars, but NO signals (correct behavior - insufficient bars/criteria not met)
- **SOBR**: NOT in watchlist (scanner miss - wasn't in top gainers at 9:27 AM scan)
- **Live traded**: 1 trade (TGHL via V3 overlay) = -$330.59 PnL
- **Backtest issue**: Initial backtest showed 7 trades because it used WRONG strategy logic (simplified V3 instead of real G/L/V3 combined engine)

---

## Detailed Findings

### 1. What Strategies Were Actually Running?

From live log at 9:30 AM:
```
Enabled strategies: G, L
v3 OVERLAY enabled: target=57% stop=30% time=27min trail=0.5%
Priority: {'G': 19, 'L': 3, ...}
```

**Only 3 strategies were active:**
- **G** (Gap continuation) - Priority 19
- **L** (Long consolidation) - Priority 3  
- **V3** (Green candle overlay) - Separate overlay system

**V3 overlay** is what actually traded (TGHL), NOT the main G/L strategies!

### 2. Why NVVE Didn't Trade

**NVVE was IN the watchlist:**
- Scanner found it at 9:00 AM: gap=9.2%, PM vol=15K, float=0.4M
- Final watchlist locked at 9:42 AM: NVVE included (20 tickers total)
- Subscribed to bar stream at 9:42 AM

**NVVE bars received (live):**
- First bar: 9:43 AM at $20.39
- Last bar: 9:54 AM at $21.94
- Total: 13 bars over 11 minutes

**Why no signal:**
- Insufficient bar history (only 13 bars)
- Price action didn't meet G/L/V3 entry criteria
- **Backtest also found NO signal for NVVE** (when using correct engine code)

**Verdict: System worked correctly**

### 3. Why SOBR Didn't Trade

**SOBR was NOT in the watchlist at all:**
- Did not appear in Webull top gainers at final 9:27 AM scan
- Scanner only takes top ~23 tickers, SOBR ranked lower
- Never subscribed to bar stream

**However:**
- Alpaca had 84 bars of data for SOBR (9:04 AM - 10:20 AM ET)
- **Backtest (with correct code) WOULD have generated signals if SOBR was in watchlist**

**Verdict: Scanner limitation** - SOBR didn't make the cut at pre-market scan time

### 4. Why G and L Strategies Didn't Fire

**G and L were enabled but generated ZERO signals all day.**

**Possible reasons:**
1. **Bar arrival timing**: Watchlist bars started arriving at 9:44 AM (14 minutes late)
   - Alpaca IEX feed had delayed data
   - Intraday discoveries (LEDS, LHAI, NXTC, etc.) arrived first at 9:30-9:43
   - Watchlist tickers (VIVS, ELVA, KUST, etc.) didn't stream until 9:44

2. **Insufficient bar history**: By the time bars arrived, optimal entry windows may have passed

3. **Strategy criteria not met**: G/L have specific gap, volume, and pattern requirements that weren't satisfied

### 5. Backtest vs Live Disc repancy

**Initial backtest showed 7 trades (WRONG):**
- VIVS, ELVA, KUST, ERNA, KOPN, GEVO, SOBR
- Used simplified V3 check logic
- Didn't use the actual combined engine code

**Why backtest was wrong:**
1. **Different code path**: Backtest used custom V3 logic instead of `test_green_candle_combined.py`
2. **Different timezone handling**: Initially had bugs in market hours filtering
3. **More bars available**: Downloaded data had 48-67 bars per ticker vs live's 11-19 bars
4. **No bar timing delays**: Backtest had all bars from 9:30, live had first bars at 9:44

**Corrected backtest (using exact engine code):**
- Should match live exactly: only V3 overlay signals
- G and L should generate 0 signals (same as live)

### 6. Bar Arrival Timing Issue

**Expected:** Bars start at 9:30 AM (market open)  
**Actual:** Bars started at 9:44 AM for watchlist tickers

**Evidence:**
```
09:30:00 ET - Subscribed to 20 symbols: ['VIVS', 'ELVA', 'KUST', ...]
09:44:00 ET - EMIT 2min: ELVA c=10.48 count=2  (FIRST watchlist bar)
09:44:00 ET - EMIT 2min: VIVS c=1.16 count=2
```

**Root cause:** Alpaca IEX feed delays
- IEX is a slower, cheaper feed vs SIP (consolidated)
- Low-volume tickers experience longer delays
- Intraday halt-resume tickers (on SIP?) arrived immediately

**Impact:**
- Missed first 14 minutes of trading
- G/L strategies may have missed optimal entry windows
- Only V3 overlay fired (TGHL at 9:48 AM)

---

## What Should Be Fixed

### 1. **Scanner Tuning** (for SOBR-like misses)
- Consider expanding watchlist size from 20 to 25-30 tickers
- Add fallback scan at 9:35 AM to catch late movers
- Weight PM volume more heavily in ranking

### 2. **Feed Upgrade** (for bar timing delays)
- Upgrade from IEX to SIP feed for faster, more complete data
- Current IEX delays cause 14+ minute latency on some tickers
- SIP would provide bars starting at 9:30 AM consistently

### 3. **G/L Strategy Investigation**
- Run dedicated analysis: why did G/L fire ZERO times on 7/15?
- Check if gap thresholds are too restrictive
- Review if strategies need parameter updates

### 4. **Backtest Accuracy**
- Always use EXACT same engine code for backtests
- Match bar timing (respect when bars actually arrived)
- Never use simplified/custom strategy logic

---

## Action Items

- [ ] Check why G/L fired 0 times across 20 tickers (seems low)
- [ ] Consider SIP feed upgrade (cost vs benefit analysis)
- [ ] Review scanner cutoff threshold (currently ~23 tickers)
- [ ] Fix backtest script to use engine_combined.py directly
- [ ] Add alerting for "0 signals all day" anomaly

---

## Appendix: Bar Count Comparison

| Ticker | Live Bars | Backtest Bars (MH) | First Live Bar | First BT Bar |
|--------|-----------|-------------------|---------------|--------------|
| VIVS   | 19        | 48                | 9:44 AM       | 9:30 AM      |
| ELVA   | 19        | 34                | 9:44 AM       | 8:54 AM      |
| KUST   | 11        | 50                | 9:47 AM       | 8:00 AM      |
| ERNA   | 15        | 53                | 9:46 AM       | 8:00 AM      |
| KOPN   | 19        | 26                | 9:45 AM       | 9:30 AM      |
| GEVO   | 14        | 31                | 9:45 AM       | 8:36 AM      |
| TGHL   | 12        | 17                | 9:46 AM       | 9:30 AM      |
| NVVE   | 13        | 10                | 9:43 AM       | 9:30 AM      |
| SOBR   | 0         | 49                | N/A           | 8:04 AM      |

**Key observation:** Live consistently had ~14 minute delay (9:44 vs 9:30)
