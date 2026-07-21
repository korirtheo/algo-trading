## Comprehensive Fix: SQLite Persistence + Timezone Consistency + Dashboard

### Problems Fixed

1. **Trade log lost on restart** - JSON files not reliable, trades disappear
2. **Strategies column always empty** - Dashboard couldn't access engine state
3. **"Done" strategies reset** - System re-evaluates tickers that already timed out
4. **Mixed timezones** - UTC/EAT/GMT confusion, should always be ET
5. **Date navigation broken** - Wrong API endpoint
6. **Exit prices not persistent** - Re-entry floor lost on crash

### Solution: SQLite Database

**File:** `live/persistence_db.py` (NEW - 350 lines)

- **Database:** `logs/trading.db` (survives restarts, mounted volume)
- **Tables:**
  - `trades` - All completed trades with ET timestamps
  - `daily_state` - Exit prices, done strategies, signal times per (ticker, strategy, date)
  - `position_state` - Open positions with full metadata

**Benefits:**
- ✅ ACID transactions (atomic, no corruption)
- ✅ Indexed queries (fast lookups)
- ✅ Survives any restart/crash
- ✅ All timestamps in ET
- ✅ Thread-safe with context managers

### Integration Points

#### 1. Replace JSON Files in engine_combined.py

**OLD:**
```python
self.exit_prices = {}  # Lost on restart
self.trades_today = _load_today_trades()  # From JSON
self.position_state = PositionStateManager()  # Separate JSON
```

**NEW:**
```python
from live.persistence_db import TradingDatabase

self.db = TradingDatabase()
self.trades_today = self.db.get_trades_today()
```

#### 2. Save Trades to DB

**OLD:**
```python
self.trades_today.append(trade)
_append_trade(trade)  # Writes to JSON
```

**NEW:**
```python
self.trades_today.append(trade)
self.db.save_trade(
    trade_date=datetime.now(ET).date().isoformat(),
    ticker=trade["ticker"],
    strategy=trade["strategy"],
    entry_price=trade["entry_price"],
    exit_price=trade["exit_price"],
    shares=trade["shares"],
    pnl=trade["pnl"],
    pnl_pct=trade["pnl_pct"],
    reason=trade["reason"],
    entry_time=trade["entry_time"],
    exit_time=trade["exit_time"]
)
```

#### 3. Track Daily State

**Check if done:**
```python
if self.db.is_done(ticker, strategy):
    log.debug(f"SIGNAL {ticker} ({strategy}) skipped — marked DONE")
    continue
```

**Mark done on TIME exit:**
```python
if exit_reason and "TIME" in exit_reason:
    self.db.mark_done(ticker, strategy)
```

**Record exit price:**
```python
self.db.record_exit(ticker, strategy, exit_price)
```

#### 4. Update Backend API

**dashboard/backend/routers/trades.py:**
```python
from live.persistence_db import TradingDatabase

db = TradingDatabase()

@router.get("/trades/{date}")
async def get_trades_by_date(date: str):
    trades = db.get_trades_by_date(date)
    return {"trades": trades, "date": date, "found": len(trades) > 0}
```

### Files to Modify

1. **live/persistence_db.py** (NEW)
2. **live/engine_combined.py** - Replace exit_prices, trades, position state with DB
3. **live/main.py** - Use DB for recovery
4. **dashboard/backend/routers/trades.py** - Query DB for historical trades
5. **dashboard/backend/services/engine_bridge.py** - Use DB for trades
6. **dashboard/frontend/src/api/client.ts** - Already fixed
7. **dashboard/frontend/src/components/TradeLog.tsx** - Already fixed
8. **dashboard/frontend/src/components/Watchlist.tsx** - Already fixed

### Timezone Fixes

**Everywhere timestamps are used:**

1. **Import ET:**
```python
from pytz import timezone
ET = timezone("US/Eastern")
```

2. **Create timestamps:**
```python
now = datetime.now(ET)  # NOT datetime.now()
```

3. **Parse timestamps:**
```python
if dt.tzinfo is None:
    dt = ET.localize(dt)
else:
    dt = dt.astimezone(ET)
```

4. **Display to user:**
```python
# Dashboard should show "12:30:45 PM ET" not "16:30:45 UTC"
formatted = dt.strftime("%I:%M:%S %p ET")
```

### Migration Plan

**Step 1: Deploy Database**
```bash
scp -P 2222 -i trading-key-v2.pem live/persistence_db.py ubuntu@54.172.65.25:/home/ubuntu/algo-trading/live/
```

**Step 2: Migrate Existing Data (Optional)**
```python
# One-time script to import old JSON trades into SQLite
import json
import glob
from live.persistence_db import TradingDatabase

db = TradingDatabase()
for file in glob.glob("logs/*_trades.json"):
    date = file.split("/")[-1].split("_")[0]
    with open(file) as f:
        trades = json.load(f)
        for t in trades:
            db.save_trade(date, ...)
```

**Step 3: Update Engine**
- Integrate DB calls into engine_combined.py
- Test locally first

**Step 4: Deploy All Changes**
```bash
# Copy updated files
scp -P 2222 -i trading-key-v2.pem live/*.py ubuntu@54.172.65.25:/home/ubuntu/algo-trading/live/
scp -P 2222 -i trading-key-v2.pem dashboard/backend/routers/trades.py ubuntu@54.172.65.25:/home/ubuntu/algo-trading/dashboard/backend/routers/
scp -P 2222 -i trading-key-v2.pem -r dashboard/frontend/dist ubuntu@54.172.65.25:/home/ubuntu/algo-trading/dashboard/frontend/

# Copy to container
ssh -p 2222 -i trading-key-v2.pem ubuntu@54.172.65.25 << 'EOF'
docker cp /home/ubuntu/algo-trading/live/. algotrader:/app/live/
docker cp /home/ubuntu/algo-trading/dashboard/backend/routers/trades.py algotrader:/app/dashboard/backend/routers/
docker cp /home/ubuntu/algo-trading/dashboard/frontend/dist/. algotrader:/app/dashboard/frontend/dist/
docker restart algotrader
EOF
```

**Step 5: Verify**
```bash
ssh -p 2222 -i trading-key-v2.pem ubuntu@54.172.65.25

# Check database exists
docker exec algotrader ls -la /app/logs/trading.db

# Check tables
docker exec algotrader sqlite3 /app/logs/trading.db "SELECT COUNT(*) FROM trades"

# Check trades persist
curl http://54.172.65.25/api/trades/today | python -m json.tool
```

### Dashboard Improvements Already Made

1. ✅ Date navigation (← / → buttons)
2. ✅ Entry/Exit times shown
3. ✅ Deployed amount column
4. ✅ Wins/losses stats
5. ✅ Strategy tags show WATCHING (bold + outline)

### Testing Plan

1. **Trade persistence:** Make a trade, restart container, verify trade still in log
2. **Date navigation:** Click ←  → buttons, verify historical trades load
3. **Done strategies:** Exit via TIME, restart, verify won't re-enter
4. **Timezone:** Check all timestamps show ET, not UTC
5. **Strategy column:** During market hours (9:30-4:00 ET), verify badges appear

### Rollback

If issues occur:
```bash
# Revert to old code
ssh -p 2222 -i trading-key-v2.pem ubuntu@54.172.65.25
docker exec -it algotrader bash
cd /app/live
git checkout engine_combined.py main.py
exit
docker restart algotrader
```

Database file persists, can migrate data later.

###Should I proceed with integrating the database into the codebase?

This will replace all JSON file persistence with SQLite.
