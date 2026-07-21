# Deployment: Daily State Persistence - 2026-07-16

## Problem Fixed

**Issue:** On mid-day crash/restart, system could re-buy tickers that were already evaluated and rejected:
- Ticker ABC watched at 9:35, marked "done" (timed out or failed conditions)
- Crash at 10:00, restart at 10:15
- Engine re-runs with fresh state, ABC passes conditions again → duplicate entry
- No memory of exit prices, done strategies, or signal times

**Impact:** Could enter positions that should be blocked, violate time limits, lose track of re-entry floor

## Solution Deployed

**Daily State Manager** ([live/daily_state.py](live/daily_state.py)):
- Persists to `logs/{date}_daily_state.json`
- Tracks per `(ticker, strategy)`:
  - **Exit prices** - Re-entry floor (won't re-buy if price ≤ last_exit × 1.01)
  - **Done strategies** - Strategies that timed out (won't re-fire after restart)
  - **Signal times** - First signal timestamp in ET (enforces time limits)
  - **Last states** - Prevents re-evaluation of already-processed tickers
- All timestamps in ET (respects "G shouldn't fire after 10:30" rules)
- Resets automatically each new trading day
- Survives crashes/restarts

## Changes Made

### New File
- **live/daily_state.py** (180 lines)
  - `DailyStateManager` class
  - Methods: `record_exit()`, `mark_done()`, `is_done()`, `record_signal_time()`, `check_time_limit()`
  - Loads state on init, saves atomically on every change

### Modified Files

**live/engine_combined.py:**
- Line 196-202: Initialize `DailyStateManager`, reference `exit_prices` from it
- Line 431-435: Check if strategy is done before entering
- Line 437: Record signal time on first entry attempt
- Line 440: Use `daily_state.get_exit_price()` instead of in-memory dict
- Line 1262-1265: Record exit, mark done if TIME exit
- Line 1265: Use `daily_state.record_exit()` instead of dict assignment

**live/main.py:**
- Line 164-165: Update `daily_state` on recovery

**README.md:**
- Added Daily State Manager section
- Updated Components table
- Updated re-entry status (currently disabled, marginal impact)

## How It Works

### Normal Flow (No Crash)
1. 9:35 - G signals on ABCD, enters @ $5.00
2. 10:15 - G times out, exits @ $5.10
3. `daily_state.json` saves:
   ```json
   {
     "exit_prices": {"ABCD|G": 5.10},
     "done_strategies": ["ABCD|G"],
     "first_signal_times": {"ABCD|G": "2026-07-16T09:35:00-04:00"}
   }
   ```
4. 11:00 - ABCD moves to $5.50, G conditions met again
5. Engine checks: `is_done("ABCD", "G")` → **True** → skips entry ✓

### Crash Recovery Flow
1. 9:35 - G signals on ABCD, enters @ $5.00
2. 10:00 - **CRASH** (state file last saved at 9:35)
3. 10:15 - **RESTART**
4. Engine loads `daily_state.json`:
   - `first_signal_times["ABCD|G"]` = 09:35 ET
   - `done_strategies` empty (didn't time out yet)
5. 10:30 - Check time limit: now=10:30, signal=09:35, elapsed=55min
6. If time_limit=60min → still valid, continues monitoring
7. 11:00 - Time limit reached → marks done, won't re-enter

### Re-Entry Floor (When Enabled)
1. 9:35 - G enters ABCD @ $5.00
2. 9:45 - G exits ABCD @ $5.20 (TARGET)
3. `exit_prices["ABCD|G"]` = 5.20 saved to disk
4. 10:00 - ABCD drops to $5.10, G signals again
5. Check: `5.10 > 5.20 * 1.01` → **False** → skips re-entry ✓
6. 10:30 - ABCD rises to $5.30, G signals
7. Check: `5.30 > 5.20 * 1.01 = 5.25` → **True** → allows re-entry ✓

## Files to Deploy

1. `live/daily_state.py` (NEW)
2. `live/engine_combined.py` (MODIFIED)
3. `live/main.py` (MODIFIED)
4. `README.md` (MODIFIED)

## Deployment Commands

```bash
# From local machine
cd c:/Users/Theo\ Korir/Documents/Python/algo-trading

# Copy files to AWS
scp -P 2222 -i trading-key-v2.pem live/daily_state.py ubuntu@54.172.65.25:/home/ubuntu/algo-trading/live/
scp -P 2222 -i trading-key-v2.pem live/engine_combined.py ubuntu@54.172.65.25:/home/ubuntu/algo-trading/live/
scp -P 2222 -i trading-key-v2.pem live/main.py ubuntu@54.172.65.25:/home/ubuntu/algo-trading/live/
scp -P 2222 -i trading-key-v2.pem README.md ubuntu@54.172.65.25:/home/ubuntu/algo-trading/

# Copy to container and restart
ssh -p 2222 -i trading-key-v2.pem ubuntu@54.172.65.25 << 'EOF'
docker cp /home/ubuntu/algo-trading/live/daily_state.py algotrader:/app/live/
docker cp /home/ubuntu/algo-trading/live/engine_combined.py algotrader:/app/live/
docker cp /home/ubuntu/algo-trading/live/main.py algotrader:/app/live/
docker cp /home/ubuntu/algo-trading/README.md algotrader:/app/
docker restart algotrader
EOF
```

## Verification

After deployment, check logs:
```bash
ssh -p 2222 -i trading-key-v2.pem ubuntu@54.172.65.25
docker logs algotrader --tail 50 | grep -E "DailyState|STRATEGY DONE"
```

Should see:
```
[DailyState] Loaded state: 2 exits, 1 done, 3 states, 2 signal times
```

Check state file exists:
```bash
docker exec algotrader ls -la /app/logs/*_daily_state.json
```

## Backtest Impact

Re-entry analysis (2024-2026, 440 days):
- **WITH re-entry:** 1,107 trades → $63.9M (2557x)
- **WITHOUT re-entry:** 752 trades → $63.4M (2534x)
- **Difference:** +$574K (+0.9%)
- **Decision:** Disabled due to marginal impact

This deployment includes re-entry **disabled** (`ENABLE_REENTRY = False`).

## Rollback (if needed)

```bash
ssh -p 2222 -i trading-key-v2.pem ubuntu@54.172.65.25
docker exec -it algotrader bash

# Inside container:
cd /app/live
rm daily_state.py
git checkout engine_combined.py main.py
exit

# Restart
docker restart algotrader
```

## Testing Crash Recovery

To test the daily state persistence:

1. Let system run for 1-2 hours
2. Check state file: `docker exec algotrader cat /app/logs/2026-07-16_daily_state.json`
3. Restart container: `docker restart algotrader`
4. Check logs: Should see "Loaded state: X exits, Y done, Z states"
5. Verify no duplicate entries on tickers already processed
