"""
Engine Bridge: Reads state from the live CombinedEngine for the dashboard.
Provides a clean interface between the trading engine and the dashboard API.
"""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class EngineBridge:
    """Reads engine and executor state for dashboard display."""

    def __init__(self, engine=None, executor=None, scanner_candidates=None):
        self.engine = engine
        self.executor = executor
        self.scanner_candidates = scanner_candidates or []

    def get_account(self):
        """Get account info from Alpaca."""
        if not self.executor:
            return {"cash": 0, "buying_power": 0, "equity": 0, "daily_pnl": 0}
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
            alpaca_positions = self.executor.trading_client.get_all_positions()
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
            engine_state = {}
            if self.engine and ticker in self.engine.last_states:
                engine_state = self.engine.last_states[ticker]

            # Determine status
            status = "watching"
            if engine_state.get("done"):
                status = "done"
            elif engine_state.get("entry_price") is not None:
                status = "in_position"
            elif engine_state.get("signal"):
                status = "signal"

            watchlist.append({
                "ticker": ticker,
                "gap_pct": cand["gap_pct"],
                "pm_volume": cand["pm_volume"],
                "premarket_high": cand["premarket_high"],
                "prev_close": cand.get("prev_close", 0),
                "float_shares": cand.get("float_shares"),
                "status": status,
                "strategy": engine_state.get("strategy", ""),
                "candle_count": engine_state.get("candle_count", 0),
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

        if self.engine and symbol in self.engine.bar_data:
            for bar in self.engine.bar_data[symbol]:
                bars.append({
                    "time": bar["timestamp"].isoformat() if hasattr(bar["timestamp"], "isoformat") else str(bar["timestamp"]),
                    "open": bar["Open"],
                    "high": bar["High"],
                    "low": bar["Low"],
                    "close": bar["Close"],
                    "volume": bar["Volume"],
                })

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

    def get_engine_summary(self):
        """Get overall engine state summary."""
        if not self.engine:
            return {}
        return {
            "active_position": self.engine.active_position,
            "daily_pnl": self.engine.daily_pnl,
            "trades_count": len(self.engine.trades_today),
            "wins": sum(1 for t in self.engine.trades_today if t["pnl"] > 0),
            "losses": sum(1 for t in self.engine.trades_today if t["pnl"] <= 0),
            "candidates_count": len(self.scanner_candidates),
            "tracking_count": len(self.engine.last_states) if self.engine else 0,
        }
