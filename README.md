# Algo Trading — Gap-Up Day Trading System

Automated intraday trading system targeting small-cap gap-up stocks.
20 strategies optimized via Optuna, running live on Alpaca paper trading
with a real-time React dashboard on AWS EC2.

- **Universe**: Small-cap stocks gapping up 8%+ pre-market (Alpaca movers + Finviz top 20)
- **Timeframe**: 2-minute candles, 9:30 AM – 3:45 PM ET
- **Capital**: $25,000 starting, 100% balance sizing
- **Data**: 576 trading days (2022 + 2024–2026)
- **Live**: Alpaca paper trading (IEX feed), AWS EC2 Ubuntu

---

## AWS Deployment

### SSH into the server
```powershell
ssh -i "C:\Users\Theo Korir\Documents\Python\algo-trading\trading-key.pem" ubuntu@98.92.73.65
```

### Check logs
```bash
cd ~/algo-trading
tail -f logs/trading.log &
```
> Use `&` so Ctrl+C doesn't kill the bot.

### View dashboard
```
http://98.92.73.65:8000
```
Dashboard stays alive 24/7 (even after market close).

### Restart the bot manually (if needed)
```bash
cd ~/algo-trading
pkill -f live.main; sleep 2
nohup venv/bin/python -m live.main >> logs/trading.log 2>&1 &
```

### Cron jobs (automatic — no manual intervention needed)
The bot runs itself via two cron jobs:
- **`@reboot`**: Starts the bot when the server boots
- **`*/5 * * * *`**: Every 5 minutes — auto-pulls new code from GitHub, restarts if new commit detected, ensures bot is always running

To deploy new code: just `git push` from your local machine. The server picks it up within 5 minutes.

---

## Trading Schedule

| Time (ET) | Action |
|-----------|--------|
| 7:00 AM | First pre-market scan (Alpaca movers + Finviz top 20) |
| 7:30 AM | Second pre-market scan |
| 9:00 AM | Third scan (more PM data available) |
| 9:25 AM | Rescan |
| 9:27 AM | Final scan — watchlist locked (top 20 by gap%) |
| 9:30 AM | Market open — bar stream starts, strategies active |
| 3:45 PM | EOD close all positions |
| 4:00 PM | Daily summary, dashboard stays alive |

If the bot restarts mid-day, it scans immediately and recovers any open positions.

---

## Local Development

### Run backtest
```bash
python test_green_candle_combined.py --no-charts
```

### Run Optuna optimization
```bash
python optimize_combined.py --trials 2000
```

### Run live trading locally
```bash
python -m live.main
```

### Frontend dev server
```bash
cd dashboard/frontend && npm run dev
# Opens http://localhost:5173 (proxies API to port 8000)
```

### Build frontend for production
```bash
cd dashboard/frontend && npm run build
```

---

## Project Structure

```
live/
  main.py              # Entry point — scans, trades, dashboard (24/7)
  scanner.py           # Alpaca movers + Finviz screener → gap-up candidates
  streamer.py          # WebSocket 1min bars → 2min aggregation
  engine_combined.py   # 20-strategy combined engine
  executor.py          # Alpaca order execution (buy/sell/close)

config/
  settings.py          # API keys + config (NOT in git)
  trial_432_params.json  # Current live params (v8 trial #432)

dashboard/
  backend/
    app.py             # FastAPI server + WebSocket endpoint
    routers/           # API endpoints (account, positions, watchlist, charts)
    services/
      engine_bridge.py # Reads engine state for dashboard
      ws_manager.py    # WebSocket broadcast manager
  frontend/
    src/components/    # React components (Chart, Watchlist, Positions, etc.)

optimize_combined.py   # Optuna optimizer (20 strategies)
test_green_candle_combined.py  # Full backtester
download_swing_data.py # Daily bars downloader for swing analysis
```

---

## Strategies (20 total)

12 currently enabled (trial #432): **H, A, D, M, W, O, B, C, I, J, N, L**

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

## Optuna Optimization

### v8 (2024–2026, 325 days)
- **700 trials** → best #653 (score 376.8M)
- Live engine uses trial #432 (same strategy set)

### v9 (2022–2026, 576 days)
- **1,436+ trials** → best #9 (score 149.4M)
- Strategies: A, F, D, V, P, R, B, K, C, N
- Harder test (includes 2022 bear market data)

---

## Scanner Sources

1. **Alpaca Movers API** — top 10 pre-market gainers (free tier limit)
2. **Finviz Screener** — top 20 by % change, price under $50, sorted by pre-market change
3. Combined and deduplicated, filtered by gap% >= 8%, capped at top 20

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
requests
lxml
```
