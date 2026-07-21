# Comprehensive Database Integration Plan

## New Tables & Methods Created

### 1. Watchlist (save_watchlist, get_watchlist_by_date)
**When:** At market open when candidates locked
**Where:** `live/main.py` line ~315 after scanner completes
**What:** Save all 20 candidates with gap%, pm_volume, float, etc.

### 2. Signals (log_signal)
**When:** Every entry signal (taken or rejected)
**Where:** `live/engine_combined.py` on_bar() entry detection
**Actions:**
- `TAKEN` - Order placed successfully
- `REJECTED` - Conditions failed (already_in_position, done, reentry_floor)
- `SKIPPED` - Vol cap, PDT, etc.

### 3. Order Events (log_order_event)
**When:** Every order lifecycle event
**Where:** 
- `live/executor.py` when order placed
- `live/engine_combined.py` _on_buy_fill, _on_sell_fill callbacks
**Events:** placed, partial_fill, fill, canceled, rejected
**Includes:** Slippage in basis points

### 4. Account Snapshots (save_account_snapshot)
**When:**
- Market open (after recovery)
- Market close (EOD)
- After significant events (recovery, large P&L swing)
**Where:** `live/main.py`
**Types:** market_open, market_close, intraday, recovery

### 5. System Events (log_system_event)
**When:** Key system lifecycle events
**Where:** Throughout `live/main.py` and `engine_combined.py`
**Events:**
- startup - Engine initialized
- shutdown - Clean exit
- crash - Unhandled exception
- recovery - Orphan positions restored
- reconnect - WebSocket reconnected
- error - Significant errors

### 6. Bar Summaries (save_bar_summary)
**When:** End of day for each ticker
**Where:** `live/engine_combined.py` at market close
**What:** OHLCV + VWAP + bar count

## Integration Steps

### Step 1: Main.py - Startup & Watchlist
```python
# At startup
db.log_system_event('startup', 'info', f'Engine started with {len(candidates)} candidates')

# After scanner
db.save_watchlist(candidates, scan_time=datetime.now(ET))

# After recovery
account = executor.get_account()
db.save_account_snapshot(
    'recovery',
    cash=account.cash,
    equity=account.equity,
    buying_power=account.buying_power,
    portfolio_value=account.portfolio_value,
    positions_count=len(recovered)
)
```

### Step 2: Engine - Signal Logging
```python
# In on_bar() when entry signal detected
if ticker in self.active_positions:
    self.db.log_signal(ticker, strategy, entry_price, 'REJECTED', 'already_in_position', gap_pct=st.get('gap_pct'))
    continue

if self.db.is_done(ticker, strategy):
    self.db.log_signal(ticker, strategy, entry_price, 'REJECTED', 'done', gap_pct=st.get('gap_pct'))
    continue

last_exit = self.db.get_exit_price(ticker, strategy)
if last_exit and entry_price <= last_exit * 1.01:
    self.db.log_signal(ticker, strategy, entry_price, 'REJECTED', 'reentry_floor', gap_pct=st.get('gap_pct'))
    continue

# Signal taken
order = self.executor.buy(...)
if order:
    self.db.log_signal(ticker, strategy, entry_price, 'TAKEN', None, order.id, gap_pct=st.get('gap_pct'))
```

### Step 3: Executor - Order Events
```python
# In buy()
self.db.log_order_event(
    order_id=str(order.id),
    ticker=ticker,
    strategy=strategy,
    side='buy',
    event_type='placed',
    signal_price=signal_price,
    qty=shares,
    cum_dollar_vol=cumulative_dollar_volume
)

# In fill callbacks
self.db.log_order_event(
    order_id=order_id,
    ticker=ticker,
    strategy=strategy,
    side=side,
    event_type='fill' if fully_filled else 'partial_fill',
    signal_price=signal_price,
    fill_price=fill_avg,
    filled_qty=filled,
    slip_bp=slippage_bp,
    status=status
)
```

### Step 4: Engine - Bar Summaries (EOD)
```python
def save_bar_summaries(self):
    """Called at market close to save daily summaries."""
    for ticker, bars in self.bar_data.items():
        if not bars:
            continue
        
        open_price = bars[0]['Open']
        high = max(b['High'] for b in bars)
        low = min(b['Low'] for b in bars)
        close = bars[-1]['Close']
        volume = sum(b['Volume'] for b in bars)
        
        # Calculate VWAP
        total_pv = sum(b['Close'] * b['Volume'] for b in bars)
        vwap = total_pv / volume if volume > 0 else None
        
        self.db.save_bar_summary(
            ticker, open_price, high, low, close, volume, vwap,
            len(bars), bars[0]['timestamp'], bars[-1]['timestamp']
        )
```

### Step 5: Main.py - EOD Snapshot
```python
# At 3:45 PM close-all
account = executor.get_account()
db.save_account_snapshot(
    'market_close',
    cash=account.cash,
    equity=account.equity,
    buying_power=account.buying_power,
    portfolio_value=account.portfolio_value,
    daily_pnl=engine.daily_pnl,
    trades_count=len(engine.trades_today),
    positions_count=len(engine.active_positions)
)

# Save bar summaries
engine.save_bar_summaries()

# System event
db.log_system_event('shutdown', 'info', 'Market close - engine stopping for day')
```

## Benefits

1. **Debugging** - Complete audit trail of every signal and order
2. **Performance Analysis** - Slippage trends, win rate by strategy, equity curve
3. **Pick Quality** - Which tickers were scanned but never traded
4. **Reliability** - Crash/recovery frequency, downtime impact
5. **Post-Trade** - "Why didn't I enter at $5 when it went to $8?"
6. **Compliance** - Full order history with timestamps

## Queries You Can Now Run

```sql
-- Signal conversion rate
SELECT 
    action,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
FROM signals
WHERE date = '2026-07-16'
GROUP BY action;

-- Average slippage by strategy
SELECT 
    strategy,
    side,
    COUNT(*) as fills,
    ROUND(AVG(slip_bp), 2) as avg_slip_bp
FROM order_events
WHERE date = '2026-07-16' AND event_type = 'fill'
GROUP BY strategy, side;

-- Rejected signals (why didn't we trade)
SELECT 
    ticker,
    strategy,
    reason,
    COUNT(*) as times_rejected
FROM signals
WHERE date = '2026-07-16' AND action = 'REJECTED'
GROUP BY ticker, strategy, reason
ORDER BY times_rejected DESC;

-- Equity curve
SELECT date, equity, daily_pnl
FROM account_snapshots
WHERE snapshot_type = 'market_close'
ORDER BY date;

-- System reliability
SELECT 
    DATE(timestamp) as date,
    event_type,
    COUNT(*) as occurrences
FROM system_events
WHERE event_type IN ('crash', 'recovery', 'reconnect')
GROUP BY date, event_type
ORDER BY date DESC;
```

## File Modifications Needed

1. `live/engine_combined.py` - Add signal logging in on_bar()
2. `live/executor.py` - Add order event logging
3. `live/main.py` - Add watchlist, snapshots, system events
4. Deploy to AWS

Ready to implement?
