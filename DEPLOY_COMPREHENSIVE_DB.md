# Deploy Comprehensive Database Tracking

## What's Been Done

✅ **persistence_db.py** - Extended with 6 new tables:
1. `watchlist` - Daily candidates (20 tickers scanned)
2. `signals` - All entry signals (taken/rejected with reasons)
3. `order_events` - Full order lifecycle + slippage tracking
4. `account_snapshots` - Equity curve (open/close/recovery)
5. `system_events` - Crashes, restarts, recoveries
6. `bar_summaries` - Daily OHLCV per ticker

✅ **engine_combined.py** - Signal logging added:
- REJECTED: already_in_position, done, reentry_floor
- TAKEN: Order placed successfully

## What's Left to Integrate

### 1. Order Event Logging (executor.py)
Add after order placement and in fill callbacks:
```python
# In buy() after order created
self.db.log_order_event(order_id, ticker, strategy, 'buy', 'placed', 
                        signal_price=current_price, qty=shares, 
                        cum_dollar_vol=cumulative_dollar_volume)

# In fill reconciliation
self.db.log_order_event(order_id, ticker, strategy, side, 'fill',
                        signal_price=signal_price, fill_price=fill_avg,
                        filled_qty=filled_qty, slip_bp=slippage_bp, status=status)
```

### 2. Watchlist Logging (main.py)
After scanner completes:
```python
engine.db.save_watchlist(candidates, scan_time=datetime.now(ET))
```

### 3. Account Snapshots (main.py)
At key moments:
```python
# After recovery
account = executor.get_account()
engine.db.save_account_snapshot('recovery', account.cash, account.equity,
                               account.buying_power, account.portfolio_value,
                               positions_count=len(recovered))

# At market close (3:45 PM)
engine.db.save_account_snapshot('market_close', account.cash, account.equity,
                               account.buying_power, account.portfolio_value,
                               daily_pnl=engine.daily_pnl, trades_count=len(engine.trades_today),
                               positions_count=len(engine.active_positions))
```

### 4. System Events (main.py)
```python
# At startup
engine.db.log_system_event('startup', 'info', f'Engine started with {len(candidates)} candidates')

# At shutdown
engine.db.log_system_event('shutdown', 'info', 'Market close - clean shutdown')

# On crash/error
engine.db.log_system_event('crash', 'critical', f'Unhandled exception: {e}', details=traceback.format_exc())
```

### 5. Bar Summaries (engine_combined.py)
Add method and call at EOD:
```python
def save_bar_summaries(self):
    """Save daily bar summaries for all tickers."""
    for ticker, bars in self.bar_data.items():
        if not bars:
            continue
        open_price = bars[0]['Open']
        high = max(b['High'] for b in bars)
        low = min(b['Low'] for b in bars)
        close = bars[-1]['Close']
        volume = sum(b['Volume'] for b in bars)
        total_pv = sum(b['Close'] * b['Volume'] for b in bars)
        vwap = total_pv / volume if volume > 0 else None
        
        self.db.save_bar_summary(ticker, open_price, high, low, close, volume, vwap,
                                len(bars), bars[0]['timestamp'], bars[-1]['timestamp'])
    log.info(f"Bar summaries saved for {len(self.bar_data)} tickers")
```

## Priority for Immediate Deployment

**High Priority (Deploy Now):**
1. ✅ Signal logging - Already done, ready to capture
2. Watchlist logging - Simple, captures daily picks
3. Account snapshots - Critical for equity curve

**Medium Priority (Next):**
4. Order events - More complex integration with executor
5. Bar summaries - End of day only

**Low Priority (Later):**
6. System events - Nice to have, not critical

## Quick Deploy (Minimal Integration)

Just deploy what's done:
1. Copy updated `persistence_db.py` to AWS
2. Copy updated `engine_combined.py` to AWS  
3. Restart container

**Result:** Signal logging starts immediately, captures why entries rejected

## Full Deploy (All Features)

Complete remaining integrations:
1. Add watchlist logging to main.py (5 lines)
2. Add account snapshots to main.py (10 lines)
3. Add bar summaries method to engine (15 lines)
4. Add order events to executor (20 lines)
5. Add system events to main.py (10 lines)

**Estimated time:** 30 minutes to integrate, test, deploy

## Queries You Can Run After Deployment

```sql
-- Why weren't we entering signals?
SELECT reason, COUNT(*) as count
FROM signals
WHERE date='2026-07-16' AND action='REJECTED'
GROUP BY reason;

-- Signal conversion rate
SELECT action, COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as pct
FROM signals
WHERE date='2026-07-16'
GROUP BY action;

-- Today's watchlist
SELECT ticker, gap_pct, pm_volume, float_shares
FROM watchlist
WHERE date='2026-07-16'
ORDER BY gap_pct DESC;
```

Want to deploy what's ready now, or complete the full integration first?
