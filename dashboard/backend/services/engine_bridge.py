"""
Engine Bridge: Reads state from the live CombinedEngine for the dashboard.
Provides a clean interface between the trading engine and the dashboard API.
"""
import logging
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def _fetch_alpaca_bars(symbol: str) -> list:
    """Fetch today's 2-minute bars from Alpaca for chart display."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED

        now = datetime.now(ET)
        today = now.date()
        market_open = datetime.combine(today, dt_time(9, 30), tzinfo=ET)
        # Don't request future data
        end = min(now, datetime.combine(today, dt_time(16, 0), tzinfo=ET))

        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(2, TimeFrameUnit.Minute),
            start=market_open,
            end=end,
            adjustment="raw",
            feed=ALPACA_FEED,
        )
        bars_resp = client.get_stock_bars(req)
        if bars_resp.df.empty:
            return []
        df = bars_resp.df.reset_index()
        result = []
        for _, row in df.iterrows():
            t = row["timestamp"]
            if hasattr(t, "to_pydatetime"):
                t = t.to_pydatetime()
            result.append({
                "time": t.isoformat() if hasattr(t, "isoformat") else str(t),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        return result
    except Exception as e:
        log.warning(f"Alpaca bar fetch failed for {symbol}: {e}")
        return []


class EngineBridge:
    """Reads engine and executor state for dashboard display."""

    def __init__(self, engine=None, executor=None, scanner_candidates=None):
        self.engine = engine
        self.executor = executor
        self.scanner_candidates = scanner_candidates or []

    def get_account(self):
        """Get account info from Alpaca."""
        if not self.executor:
            return {"cash": 0, "buying_power": 0, "equity": 0, "portfolio_value": 0,
                    "daily_pnl": 0, "status": "inactive", "pdt": False, "daytrade_count": 0}
        try:
            acct = self.executor.get_account()
            return {
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "equity": float(acct.equity),
                "portfolio_value": float(acct.portfolio_value),
                "daily_pnl": self.engine.daily_pnl if self.engine else 0,
                "status": str(acct.status),
                "pdt": acct.pattern_day_trader,
                "daytrade_count": acct.daytrade_count,
            }
        except Exception as e:
            log.error(f"Error getting account: {e}")
            return {"cash": 0, "buying_power": 0, "equity": 0, "daily_pnl": 0, "error": str(e)}

    def get_positions(self):
        """Get current positions from Alpaca + engine state."""
        positions = []
        if not self.executor:
            return positions
        try:
            alpaca_positions = self.executor.client.get_all_positions()
            for pos in alpaca_positions:
                ticker = pos.symbol
                engine_state = {}
                if self.engine and ticker in self.engine.last_states:
                    engine_state = self.engine.last_states[ticker]
                entry_info = {}
                if self.engine and ticker in self.engine.position_entry:
                    entry_info = self.engine.position_entry[ticker]

                positions.append({
                    "ticker": ticker,
                    "qty": float(pos.qty),
                    "avg_entry": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pl),
                    "unrealized_pnl_pct": float(pos.unrealized_plpc) * 100,
                    "strategy": entry_info.get("strategy", engine_state.get("strategy", "?")),
                    "entry_time": str(entry_info.get("entry_time", "")),
                    "gap_pct": engine_state.get("gap_pct", 0),
                })
        except Exception as e:
            log.error(f"Error getting positions: {e}")
        return positions

    def get_watchlist(self):
        """Get today's scanner candidates."""
        watchlist = []
        for cand in self.scanner_candidates:
            ticker = cand["ticker"]

            # Get ALL sub-states for this ticker
            sub_states = []
            if self.engine and hasattr(self.engine, 'all_states') and ticker in self.engine.all_states:
                sub_states = self.engine.all_states[ticker]
            elif self.engine and ticker in self.engine.last_states:
                sub_states = [self.engine.last_states[ticker]]

            # Determine status by merging all sub-states
            status = "watching"
            any_in_position = any(st.get("entry_price") is not None for st in sub_states)
            any_signal = any(st.get("signal") for st in sub_states)
            all_done = all(st.get("done", False) for st in sub_states) if sub_states else False
            if any_in_position:
                status = "in_position"
            elif any_signal:
                status = "signal"
            elif all_done:
                status = "done"

            # Last price from engine bar data
            last_price = None
            change_pct = None
            if self.engine:
                bars = self.engine.bar_data.get(ticker, [])
                if bars:
                    last_price = bars[-1].get("Close")
                    prev_close = cand.get("prev_close", 0)
                    if last_price and prev_close:
                        change_pct = (last_price / prev_close - 1) * 100

            # Merge eligible strategies across all sub-states
            eligible_strategies = []
            for code in "HGAFDVPMRWOBKCEIJNL":
                key = code.lower()
                best_status = None
                for st in sub_states:
                    eligible = st.get(f"{key}_eligible", False)
                    if not eligible:
                        continue
                    fired = st.get("strategy") == code and st.get("entry_price") is not None
                    is_done = st.get("done", False)
                    if fired:
                        best_status = "fired"
                    elif not is_done and best_status != "fired":
                        best_status = "active"
                    elif is_done and best_status not in ("fired", "active"):
                        best_status = "done"
                if best_status:
                    eligible_strategies.append({"code": code, "status": best_status})

            watchlist.append({
                "ticker": ticker,
                "gap_pct": cand["gap_pct"],
                "pm_volume": cand["pm_volume"],
                "premarket_high": cand["premarket_high"],
                "prev_close": cand.get("prev_close", 0),
                "float_shares": cand.get("float_shares"),
                "status": status,
                "strategy": next((st.get("strategy", "") for st in sub_states if st.get("strategy")), ""),
                "eligible_strategies": eligible_strategies,
                "candle_count": max((st.get("candle_count", 0) for st in sub_states), default=0),
                "last_price": last_price,
                "change_pct": change_pct,
            })
        return watchlist

    def get_trades_today(self):
        """Get completed trades for today."""
        if not self.engine:
            return []
        trades = []
        for t in self.engine.trades_today:
            trades.append({
                "ticker": t["ticker"],
                "strategy": t.get("strategy", "?"),
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "pnl": t["pnl"],
                "pnl_pct": ((t["exit_price"] / t["entry_price"] - 1) * 100) if t["entry_price"] > 0 else 0,
                "reason": t["reason"],
                "entry_time": str(t.get("entry_time", "")),
                "exit_time": str(t.get("exit_time", "")),
            })
        return trades

    def get_strategy_stats(self):
        """Aggregate today's trades by strategy."""
        if not self.engine:
            return {}
        stats = {}
        for t in self.engine.trades_today:
            s = t.get("strategy", "?")
            if s not in stats:
                stats[s] = {"trades": 0, "wins": 0, "pnl": 0.0, "best": 0.0, "worst": 0.0}
            stats[s]["trades"] += 1
            stats[s]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                stats[s]["wins"] += 1
            stats[s]["best"] = max(stats[s]["best"], t["pnl"])
            stats[s]["worst"] = min(stats[s]["worst"], t["pnl"])

        # Add win rate
        for s in stats:
            stats[s]["win_rate"] = (stats[s]["wins"] / stats[s]["trades"] * 100) if stats[s]["trades"] > 0 else 0
        return stats

    def get_chart_data(self, symbol):
        """Get OHLCV bar data + trade markers for a symbol."""
        bars = []
        markers = []

        # Try to get bars from engine's live buffer
        engine_bars = []
        if self.engine and symbol in self.engine.bar_data:
            for bar in self.engine.bar_data[symbol]:
                engine_bars.append({
                    "time": bar["timestamp"].isoformat() if hasattr(bar["timestamp"], "isoformat") else str(bar["timestamp"]),
                    "open": bar["Open"],
                    "high": bar["High"],
                    "low": bar["Low"],
                    "close": bar["Close"],
                    "volume": bar["Volume"],
                })

        # Always fetch full-day bars from Alpaca for context
        alpaca_bars = _fetch_alpaca_bars(symbol)

        if alpaca_bars:
            # Merge: use Alpaca as base, overlay engine bars (more up-to-date) by time key
            engine_by_time = {b["time"][:19]: b for b in engine_bars}
            merged = {}
            for b in alpaca_bars:
                key = b["time"][:19]
                merged[key] = b
            for b in engine_bars:
                key = b["time"][:19]
                merged[key] = b
            bars = sorted(merged.values(), key=lambda x: x["time"])
        else:
            bars = engine_bars

        # Add trade markers
        if self.engine:
            for t in self.engine.trades_today:
                if t["ticker"] == symbol:
                    if t.get("entry_time"):
                        markers.append({
                            "time": str(t["entry_time"]),
                            "type": "buy",
                            "price": t["entry_price"],
                            "strategy": t.get("strategy", "?"),
                            "text": f"BUY {t.get('strategy', '?')}",
                        })
                    if t.get("exit_time"):
                        markers.append({
                            "time": str(t["exit_time"]),
                            "type": "sell",
                            "price": t["exit_price"],
                            "strategy": t.get("strategy", "?"),
                            "pnl": t["pnl"],
                            "reason": t["reason"],
                            "text": f"SELL ${t['pnl']:+,.0f} ({t['reason']})",
                        })

        return {"bars": bars, "markers": markers, "symbol": symbol}

    def get_diagnostics(self):
        """Per-ticker diagnostics: merge ALL sub-states (main, l_only, o_only, b_only, e_only)."""
        result = []
        for cand in self.scanner_candidates:
            ticker = cand["ticker"]

            # Get ALL sub-states for this ticker (not just the last one)
            sub_states = []
            if self.engine and hasattr(self.engine, 'all_states') and ticker in self.engine.all_states:
                sub_states = self.engine.all_states[ticker]
            elif self.engine and ticker in self.engine.last_states:
                sub_states = [self.engine.last_states[ticker]]

            bars = self.engine.bar_data.get(ticker, []) if self.engine else []

            # Merge eligibility/status across all sub-states
            max_candles = 0
            market_open = 0
            any_traded = False
            active_strategy = ""
            strategies = []

            for code in "HGAFDVPMRWOBKCEIJNL":
                key = code.lower()
                best_status = "not_eligible"
                for st in sub_states:
                    max_candles = max(max_candles, st.get("candle_count", 0))
                    if not market_open:
                        market_open = st.get("market_open", 0)
                    fired = st.get("strategy") == code and st.get("entry_price") is not None
                    eligible = st.get(f"{key}_eligible", False)
                    is_done = st.get("done", False)
                    if fired:
                        best_status = "fired"
                        any_traded = True
                        active_strategy = code
                    elif eligible and best_status != "fired":
                        best_status = "timed_out" if is_done else "watching"
                strategies.append({"code": code, "status": best_status})

            candle_count = max_candles or len(bars)
            if not market_open and bars:
                market_open = bars[0]["Open"]
            ticker_done = all(st.get("done", False) for st in sub_states) if sub_states else False

            premarket_high = cand.get("premarket_high", 0)
            pm_high_ratio = ((premarket_high / market_open - 1) * 100) if market_open else 0
            last_price = bars[-1]["Close"] if bars else None
            prev_close = cand.get("prev_close", 0)
            change_pct = ((last_price / prev_close - 1) * 100) if (last_price and prev_close) else None

            result.append({
                "ticker": ticker,
                "gap_pct": cand["gap_pct"],
                "candle_count": candle_count,
                "premarket_high": premarket_high,
                "market_open": round(market_open, 3) if market_open else 0,
                "pm_high_pct_above_open": round(pm_high_ratio, 1),
                "last_price": last_price,
                "change_pct": round(change_pct, 1) if change_pct is not None else None,
                "strategies": strategies,
                "traded": any_traded,
                "active_strategy": active_strategy,
                "done": ticker_done,
            })
        return result

    def get_engine_summary(self):
        """Get overall engine state summary."""
        if not self.engine:
            return {
                "active_position": None,
                "daily_pnl": 0,
                "trades_count": 0,
                "wins": 0,
                "losses": 0,
                "candidates_count": len(self.scanner_candidates),
                "tracking_count": 0,
            }
        return {
            "active_position": self.engine.active_position,
            "daily_pnl": self.engine.daily_pnl,
            "trades_count": len(self.engine.trades_today),
            "wins": sum(1 for t in self.engine.trades_today if t["pnl"] > 0),
            "losses": sum(1 for t in self.engine.trades_today if t["pnl"] <= 0),
            "candidates_count": len(self.scanner_candidates),
            "tracking_count": len(self.engine.last_states) if self.engine else 0,
        }
