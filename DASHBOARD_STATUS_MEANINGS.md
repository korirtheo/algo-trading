# Dashboard Status Meanings

## Watchlist Statuses

### "watching"
- **Meaning:** Ticker is in today's watchlist, strategies are monitoring for entry signals
- **What's happening:** Bar data is streaming in, strategies are evaluating entry conditions
- **No action taken yet**

### "signal"
- **Meaning:** At least one strategy has generated an entry signal
- **What's happening:** Strategy conditions met, but entry not yet filled
- **May be waiting for order fill or evaluating against other candidates**

### "in_position"
- **Meaning:** Currently holding shares in this ticker
- **What's happening:** Position is open, monitoring for exit conditions (stop/target/trail/time)
- **Active trade in progress**

### "done"
- **Meaning:** All strategies have finished evaluating this ticker for today
- **What's happening:** Either:
  - Trade completed (entered and exited)
  - Strategy timed out without entry
  - Conditions no longer met (missed window)
- **No further action will be taken today**

## Strategy Tags

### Colors & Opacity
- **Bright/Solid + White outline:** Strategy FIRED (entered position)
- **Semi-transparent:** Strategy is WATCHING (conditions not met yet)
- **Very dim (30% opacity):** Strategy is DONE (timed out or finished)

### Strategy Codes
- **G** - Green (main gap strategy)
- **L** - Low float specialist
- **H** - High conviction gap
- **A** - Aggressive entry
- **V** - VWAP reclaim
- **V3** - Overlay strategy (re-entry after G/L)
- **RECOVERED** - Position recovered after crash/restart

### How to Read
If you see **"watching"** status with a semi-transparent **G** tag:
- G strategy is monitoring the ticker
- Entry conditions not met yet
- Waiting for 2nd green candle + new high

If you see **"signal"** status with a bright **G** tag + outline:
- G strategy fired an entry signal
- Order placed or filling
- About to enter position

If you see **"in_position"** status:
- Currently holding shares
- Check Positions card for entry price, P&L, shares
- System monitoring for exit (stop/target/trail)
