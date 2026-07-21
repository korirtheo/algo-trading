# Manual Deployment Steps - 2026-07-16

## Changes Ready to Deploy

1. **Re-entry DISABLED** - `ENABLE_REENTRY = False` (marginal +0.9% impact over 2.5 years)
2. **Dashboard improvements:**
   - Date navigation (← / → buttons)
   - Entry/Exit times (HH:MM:SS)
   - Deployed amount (position size)
   - Wins/losses stats in header
   - New API endpoint: `GET /trades/{date}`

## Option 1: Run Deployment Script (Easiest)

```bash
cd c:/Users/Theo\ Korir/Documents/Python/algo-trading
bash DEPLOY_DASHBOARD_2026-07-16.sh
```

## Option 2: Manual Commands

### Step 1: Copy Files to AWS

```bash
# Core strategy file (re-entry disabled)
scp -P 2222 test_green_candle_combined.py ubuntu@54.172.65.25:/home/ubuntu/algo-trading/

# Dashboard backend (new /trades/{date} endpoint)
scp -P 2222 -r dashboard/backend ubuntu@54.172.65.25:/home/ubuntu/algo-trading/dashboard/

# Dashboard frontend (built React app with improvements)
scp -P 2222 -r dashboard/frontend/dist ubuntu@54.172.65.25:/home/ubuntu/algo-trading/dashboard/frontend/
```

### Step 2: Copy to Docker Container

```bash
ssh -p 2222 ubuntu@54.172.65.25

# Inside AWS:
docker cp /home/ubuntu/algo-trading/test_green_candle_combined.py algotrader:/app/
docker cp /home/ubuntu/algo-trading/dashboard/backend/. algotrader:/app/dashboard/backend/
docker cp /home/ubuntu/algo-trading/dashboard/frontend/dist/. algotrader:/app/dashboard/frontend/dist/
```

### Step 3: Restart Dashboard

```bash
# Still inside AWS:
docker exec algotrader supervisorctl restart dashboard
```

### Step 4: Verify

Visit: http://54.172.65.25/

Check:
- Trade log has ← / → buttons
- Times shown for entries/exits
- Deployed amount column present
- Wins/losses shown in header (bold green/red)

## Re-Entry Analysis Summary

**Full 2024-2026 backtest (440 trading days):**
- WITH re-entry: 1,107 trades → $63.9M (2557x)
- WITHOUT re-entry: 752 trades → $63.4M (2534x)
- **Difference: +$574K (+0.9%)**

**Decision:** Disabled due to marginal impact

**Re-entry stats:**
- 355 re-entries (all G strategy)
- 94.6% win rate
- $574K total profit
- But only +0.9% improvement over 2.5 years

## Files Modified

### Core Strategy
- `test_green_candle_combined.py` - Line 168: `ENABLE_REENTRY = False`

### Dashboard Backend
- `dashboard/backend/routers/trades.py` - New `/trades/{date}` endpoint
- `dashboard/backend/services/engine_bridge.py` - Added shares, deployed_amount to trades

### Dashboard Frontend
- `dashboard/frontend/src/components/TradeLog.tsx` - Date navigation, enhanced display
- `dashboard/frontend/src/components/Watchlist.tsx` - (no changes after revert)

## Rollback (if needed)

```bash
ssh -p 2222 ubuntu@54.172.65.25
docker exec -it algotrader bash

# Inside container:
cd /app
# Edit test_green_candle_combined.py line 168
sed -i 's/ENABLE_REENTRY = False/ENABLE_REENTRY = True/' test_green_candle_combined.py

# Restart dashboard
supervisorctl restart dashboard
```
