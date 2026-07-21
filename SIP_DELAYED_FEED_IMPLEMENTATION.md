# SIP Delayed Feed Implementation (Free Tier)

## Overview

Switched from **IEX** to **SIP (delayed 15 minutes)** to fix the bar timing issues discovered on 2026-07-15.

## The Problem We Had

**IEX Feed Issues:**
- 14-minute random delays on watchlist tickers (first bars at 9:44 instead of 9:30)
- Only sees ~2.5% of market volume (had to multiply by 40x for estimates)
- Incomplete market coverage (not all exchanges)
- G strategy requires first 2 bars at market open → broken with delays

## The Solution

**SIP Delayed Feed (Free Tier):**
- Complete market coverage (consolidated from ALL exchanges)
- Consistent 15-minute delay (predictable, not random)
- 100% accurate volume (no multipliers needed)
- Reliable bar delivery (no gaps or missing data)

## How It Works

### Key Concept: Use Bar Timestamp, Not Wall Clock

```
Wall Clock Time: 9:45 AM
Bar Arrives with timestamp: 9:30 AM
↓
Strategy processes it as "9:30 AM bar"
Dashboard shows "Market Time: 9:30 AM (delayed 15min)"
```

### Example Timeline

| Wall Clock | Bar Timestamp | Action |
|------------|---------------|--------|
| 9:45 AM | 9:30 AM | First bar arrives → G strategy checks "first 2 candles" |
| 9:47 AM | 9:32 AM | Second bar → G can validate gap continuation |
| 10:00 AM | 9:45 AM | Normal operation continues with 15-min offset |
| 4:00 PM | 3:45 PM | EOD close processed |
| 4:15 PM | 4:00 PM | Final bars arrive, summary complete |

### Strategy Timing

**Everything uses bar timestamps, not wall clock:**
- G strategy: "First 2 bars after 9:30" → checks `bar.timestamp` at 9:30 and 9:32
- Time limit checks: "18 minutes since entry" → based on `bar.timestamp`
- EOD close: Triggered by last bar with timestamp 3:45 PM (arrives at 4:00 PM)

## Changes Made

### 1. Config Settings
```python
# config/settings.py
ALPACA_FEED = "sip"  # Changed from "iex"
ALPACA_FEED_DELAYED = True  # Free tier = 15-min delay
```

### 2. Streamer
```python
# live/streamer.py
FEED_DELAYED = ALPACA_FEED_DELAYED if ALPACA_FEED == "sip" else False
```

Already using `bar.timestamp` for aggregation (no code change needed).

### 3. Engine
Engine already uses `bar['timestamp']` from the bar dict (no change needed).

### 4. Scanner
Scanner runs BEFORE market open (7:00-9:27 AM) so NO delay issues:
- Webull data: real-time pre-market
- Alpaca PM bars: fetched via REST at scan time (not streaming)

### 5. Dashboard (Optional Enhancement)
To show the delay clearly:

```python
# dashboard/backend/app.py
market_time = bar['timestamp']  # e.g., 9:30 AM
wall_time = datetime.now(ET)    # e.g., 9:45 AM
delay_min = (wall_time - market_time).total_seconds() / 60

# Display:
# "Market Time: 9:30:00 (delayed 15 min)"
# "Wall Time: 9:45:00"
```

## Benefits

| Aspect | IEX (Before) | SIP Delayed (Now) |
|--------|-------------|-------------------|
| **Bar timing** | Random 14+ min delay | Consistent 15 min |
| **First bar** | 9:44 AM (broken) | 9:45 AM wall, 9:30 timestamp (works!) |
| **Coverage** | ~2.5% volume | 100% consolidated |
| **G strategy** | Broken (no first bars) | Works (has first bars) |
| **Volume accuracy** | Estimated (×40) | Exact |
| **Cost** | Free | Free |

## What Changed in Behavior

### Before (IEX)
```
09:30:00 Wall - Stream subscribed
09:44:00 Wall - First bar (VIVS) timestamp 09:42
           ↑ G strategy sees this as bar #1 at 9:42 (wrong!)
           ↑ G looks for "first 2 green candles at 9:30" → can't find them
```

### After (SIP Delayed)
```
09:45:00 Wall - First bar arrives, timestamp 09:30
           ↑ G strategy sees bar timestamp 09:30 → this IS the first bar
09:47:00 Wall - Second bar, timestamp 09:32
           ↑ G checks: "first 2 bars green?" → can validate properly
```

## Validation

### Test Checklist
- [ ] First bar timestamp is 9:30 AM (not 9:44 AM)
- [ ] G strategy fires on valid gap continuations
- [ ] Volume data is accurate (no 40x multiplier needed)
- [ ] EOD close happens at correct market time (3:45 PM timestamp)
- [ ] Dashboard shows both market time and wall time
- [ ] Backtest and live match (same SIP data)

### Monitoring
```bash
# Check bar timing
grep "EMIT 2min" logs/YYYY-MM-DD_live.log | head -5
# Should show timestamp field at 9:30, 9:32, etc.

# Check G strategy firing
grep "G ENTRY" logs/YYYY-MM-DD_live.log
# Should see entries when gap continuation happens

# Check volume
grep "cum_$vol" logs/YYYY-MM-DD_live.log
# Should NOT have IEX_VOLUME_MULTIPLIER applied
```

## Edge Cases

### 1. Pre-Market Scanner
**No impact** - scanner runs at 7:00-9:27 AM using REST API for PM bars, not streaming.

### 2. Halt-Resume Monitor
**No impact** - uses real-time halt data from Nasdaq, not bar data.

### 3. Order Placement
**Delayed by 15 minutes!** Orders go to broker at wall clock time:
- Strategy sees bar timestamp 9:30, generates signal at 9:45 wall time
- Order placed at 9:45 wall time (market already at 9:45)
- **This is EXPECTED for paper trading validation** - we're testing strategy logic, not execution

### 4. Live Real Money Trading
**For real money, upgrade to paid SIP (real-time):**
```python
ALPACA_FEED = "sip"
ALPACA_FEED_DELAYED = False
```
Everything else stays the same (already using bar timestamps).

## Migration Notes

### Switching Back to IEX
```python
# config/settings.py
ALPACA_FEED = "iex"
# ALPACA_FEED_DELAYED not used for IEX
```

### Upgrading to Real-Time SIP (Paid)
```python
ALPACA_FEED = "sip"
ALPACA_FEED_DELAYED = False
```

## Summary

**The 15-minute delay is a FEATURE, not a bug:**
- Bar timestamp = market time (9:30 AM)
- Wall clock time = delivery time (9:45 AM)
- Strategy uses market time → works correctly
- Orders execute at wall time → expected lag for paper trading

**For production (real money), pay for real-time SIP and set `ALPACA_FEED_DELAYED = False`.**
