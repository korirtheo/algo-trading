# Algo Trading — Gap-Up Day Trading System

Automated intraday day trading system targeting small-cap gap-up stocks.
20 strategies optimized via Optuna TPE, running live on Alpaca paper trading
with a real-time React dashboard.

- **Universe**: Small-cap stocks gapping up 10%+ with 250K+ premarket volume
- **Timeframe**: 2-minute candles, 9:30 AM – 3:55 PM ET
- **Capital**: $25,000 starting, 100% balance sizing
- **Data**: 576 trading days (2022 + 2024–2026)
- **Live**: Alpaca paper trading (IEX feed)

---

## AWS Deployment

### SSH into the server
```powershell
ssh -i "C:\Users\Theo Korir\Downloads\trading-key.pem" ubuntu@98.92.73.65
```

### Check if the bot is running
```bash
tail -f ~/trading.log
```

### Restart the bot (outside market hours only)
```bash
pkill -f live.main
cd ~/algo-trading
source venv/bin/activate
nohup python -m live.main >> ~/trading.log 2>&1 &
```

### View dashboard
```
http://98.92.73.65:8000
```

### Deploy code changes
1. Push from local machine: `git push`
2. AWS pulls automatically every 5 minutes via cron
3. SSH in and restart the bot (see above)

---

## Local Development

### Run live trading
```bash
python -m live.main
```

### Run backtest
```bash
python test_green_candle_combined.py --no-charts
```

### Run Optuna optimization
```bash
python optimize_combined.py --trials 2000
```

### Blind test (out-of-sample)
```bash
python blind_test_2022.py
```

### Frontend dev server (instant live reload)
```bash
cd dashboard/frontend && npm run dev
# Open http://localhost:5173 (proxies API to port 8000)
```

### Build frontend for production
```bash
cd dashboard/frontend && npm run build
```

---

## Project Structure

```
live/
  main.py              # Entry point — scans, waits, trades, dashboard
  scanner.py           # Alpaca movers endpoint → gap-up candidates
  streamer.py          # WebSocket 1min bars → 2min aggregation
  engine_combined.py   # 20-strategy combined engine
  executor.py          # Alpaca order execution (buy/sell/close)

config/
  settings.py          # API keys + config (NOT in git)
  trial_432_params.json  # Current live params (v8 trial #653)

dashboard/
  backend/
    app.py             # FastAPI server + WebSocket endpoint
    routers/           # API endpoints (account, positions, watchlist, charts)
    services/
      engine_bridge.py # Reads engine state for dashboard
      ws_manager.py    # WebSocket broadcast manager
  frontend/
    src/components/    # React components (Chart, Watchlist, Positions, etc.)

optimize_combined.py   # Optuna optimizer (v9, 20 strategies, 2000 trials)
test_green_candle_combined.py  # Full backtester
blind_test_2022.py     # Out-of-sample backtest on 2022 data
```

---

## Trading Schedule

| Time (ET) | Action |
|-----------|--------|
| 9:00 AM | First premarket scan (Alpaca movers) |
| 9:25 AM | Rescan |
| 9:27 AM | Final scan — watchlist locked |
| 9:30 AM | Bar stream starts, strategies active |
| 3:45 PM | EOD close all positions |
| 4:00 PM | Daily summary + shutdown |

---

## Strategies (20 total)

| Code | Name | Description |
|------|------|-------------|
| H | High Conviction | Big gap + strong candle 2 + volume confirm |
| G | Big Gap Runner | 30%+ gap with candle 2 follow-through |
| A | Quick Scalp | 15%+ gap, strong body, fast target |
| F | Catch-All | 10%+ gap, any green candle 2 |
| D | Opening Dip Buy | Dip after open, VWAP reclaim entry |
| V | VWAP Reclaim | Price reclaims VWAP after pullback |
| P | PM High Breakout | Breakout above premarket high + pullback + bounce |
| M | Midday Range Break | Breakout of midday consolidation range |
| R | Multi-Day Runner | Continuation from prior day strength |
| W | Power Hour Breakout | Late-day breakout 3–4 PM |
| L | Low Float Squeeze | Very low float + extreme volume |
| O | Opening Range Breakout | Breakout above first 5-min range |
| B | Red-to-Green (R2G) | Red candle 1 stock reverses above open |
| K | First Pullback Buy | Pullback to rising VWAP after morning run |
| C | Micro Flag | Tight 3–5 candle consolidation breakout |
| S | Stuff-and-Break | Multiple HOD rejections then final breakout |
| E | Gap-and-Go RelVol | Extreme relative volume continuation |
| I | Immediate PM Breakout | Instant breakout above PM high at open |
| J | VWAP + PM Breakout | VWAP reclaim + PM high breakout combo |
| N | HOD Reclaim | Reclaim of high-of-day after rejection |

---

## Optuna Optimization (v9)

- **Study**: `combined_v9_20strats_2022_2026`
- **DB**: `optuna_combined_v9.db`
- **Data**: 576 days (2022-01-03 to 2026-02-27)
- **Objective**: `total_pnl × min(profit_factor, 3.0)`
- **Workers**: 4 parallel (SQLite RDBStorage)
- **Best trial**: #9 — score 149,374,350 | strategies: A, F, D, V, P, R, B, K, C, N

---

## Backtest Results

### In-Sample (2024–2026, 325 days)
| Metric | Value |
|--------|-------|
| Starting Capital | $25,000 |
| Profit Factor | ~1.80 |
| Best Trial Score | 149,374,350 |

### Out-of-Sample (2022, blind test)
- Tested on 250 days of unseen 2022 data
- Strategies O, J, D flagged as overfitted (profitable in-sample, losing OOS)

---

## Dependencies

```
alpaca-py
fastapi
uvicorn
websockets
optuna
pandas
numpy
pytz
```
