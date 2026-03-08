"""
Strategy Engine: Feeds real-time bars to strategy logic, manages state and exits.

Receives 2-min bars from streamer, runs L strategy signal detection,
triggers entries/exits via executor.
"""
import logging
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from strategies.low_float_squeeze import (
    DEFAULT_PARAMS, create_state, check_signal, check_exit,
    compute_vwap, is_eligible,
)
from config.settings import SLIPPAGE_PCT, EOD_EXIT_MINUTES, FLOAT_DATA

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class StrategyEngine:
    def __init__(self, executor, params=None):
        """
        Args:
            executor: OrderExecutor instance for placing orders
            params: L strategy params dict (uses defaults if None)
        """
        self.executor = executor
        self.params = params or DEFAULT_PARAMS
        self.states = {}  # ticker -> state dict
        self.bar_history = {}  # ticker -> list of bar dicts (for VWAP)
        self.active_position = None  # ticker currently in position (one at a time)
        self.daily_pnl = 0.0
        self.trades_today = []

    def initialize_watchlist(self, candidates):
        """Set up state tracking for scanner candidates.

        Args:
            candidates: list of dicts from scanner
                [{"ticker", "gap_pct", "pm_volume", "premarket_high", "float_shares"}, ...]
        """
        self.states.clear()
        self.bar_history.clear()
        self.active_position = None
        self.daily_pnl = 0.0
        self.trades_today = []

        for cand in candidates:
            ticker = cand["ticker"]
            float_shares = cand.get("float_shares") or FLOAT_DATA.get(ticker)
            if float_shares is None:
                continue
            if not is_eligible(ticker, cand["gap_pct"], float_shares, self.params):
                continue

            state = create_state(
                ticker=ticker,
                gap_pct=cand["gap_pct"],
                float_shares=float_shares,
                premarket_high=cand["premarket_high"],
                pm_volume=cand["pm_volume"],
                params=self.params,
            )
            self.states[ticker] = state
            self.bar_history[ticker] = {
                "highs": [], "lows": [], "closes": [], "volumes": [],
            }

        log.info(f"Initialized {len(self.states)} candidates for L strategy")
        for ticker, st in self.states.items():
            log.info(f"  {ticker}: gap={st['gap_pct']:.1f}%, float={st['float_shares']/1e6:.1f}M, "
                     f"targets=+{st['target1_pct']:.0f}%/+{st['target2_pct']:.0f}%")

    def on_bar(self, symbol, bar):
        """Process a completed 2-min bar.

        Args:
            symbol: ticker symbol
            bar: dict with Open, High, Low, Close, Volume, timestamp
        """
        if symbol not in self.states:
            return

        st = self.states[symbol]
        if st["done"]:
            return

        c_open = float(bar["Open"])
        c_high = float(bar["High"])
        c_low = float(bar["Low"])
        c_close = float(bar["Close"])
        c_vol = float(bar["Volume"])
        ts = bar["timestamp"]

        # Update bar history for VWAP
        hist = self.bar_history[symbol]
        hist["highs"].append(c_high)
        hist["lows"].append(c_low)
        hist["closes"].append(c_close)
        hist["volumes"].append(c_vol)

        # Compute current VWAP
        vwap_arr = compute_vwap(hist["highs"], hist["lows"], hist["closes"], hist["volumes"])
        vwap_value = vwap_arr[-1] if len(vwap_arr) > 0 else None

        try:
            ts_et = ts.astimezone(ET)
        except Exception:
            ts_et = datetime.now(ET)

        minutes_to_close = (16 * 60) - (ts_et.hour * 60 + ts_et.minute)

        # --- IN POSITION: check exits ---
        if st["entry_price"] is not None:
            entry_et = st["entry_time"]
            try:
                entry_et = entry_et.astimezone(ET)
            except Exception:
                pass
            minutes_in_trade = (ts_et.hour * 60 + ts_et.minute) - (entry_et.hour * 60 + entry_et.minute)

            should_exit, exit_price, exit_reason = check_exit(
                st, c_high, c_low, c_close,
                minutes_in_trade, minutes_to_close,
                slippage_pct=SLIPPAGE_PCT,
                eod_exit_minutes=EOD_EXIT_MINUTES,
                params=self.params,
            )

            if should_exit:
                if exit_reason == "PARTIAL":
                    # Handle partial sell
                    partial_shares = st["shares"] * (self.params["partial_sell_pct"] / 100)
                    order = self.executor.sell(symbol, shares=partial_shares, reason="PARTIAL")
                    if order:
                        st["partial_proceeds"] += partial_shares * exit_price * (1 - SLIPPAGE_PCT / 100)
                        st["shares"] -= partial_shares
                        log.info(f"PARTIAL SELL {symbol}: {partial_shares:.4f} shares @ ${exit_price:.2f}")
                else:
                    # Full exit
                    order = self.executor.sell(symbol, reason=exit_reason)
                    if order:
                        sell_price = exit_price * (1 - SLIPPAGE_PCT / 100)
                        proceeds = st["shares"] * sell_price + st["partial_proceeds"]
                        pnl = proceeds - st["position_cost"]
                        st["pnl"] = pnl
                        st["exit_price"] = exit_price
                        st["exit_time"] = ts
                        st["exit_reason"] = exit_reason
                        st["entry_price"] = None
                        st["shares"] = 0
                        st["done"] = True
                        self.active_position = None
                        self.daily_pnl += pnl

                        self.trades_today.append({
                            "ticker": symbol,
                            "entry_price": st.get("_orig_entry_price", 0),
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "reason": exit_reason,
                            "entry_time": st["entry_time"],
                            "exit_time": ts,
                        })
                        log.info(f"EXIT {symbol} ({exit_reason}): PnL=${pnl:+,.2f} "
                                 f"entry=${st.get('_orig_entry_price', 0):.2f} -> exit=${exit_price:.2f}")
            return

        # --- NOT IN POSITION: check for signal ---
        if self.active_position is not None:
            # Already in a position with another ticker — still track state
            check_signal(st, c_open, c_high, c_low, c_close, c_vol, vwap_value, self.params)
            return

        # Check for entry signal
        fired = check_signal(st, c_open, c_high, c_low, c_close, c_vol, vwap_value, self.params)

        if fired:
            # Get available capital
            cash = self.executor.get_buying_power()
            if cash < 100:
                log.warning(f"Skip {symbol}: insufficient buying power ${cash:.2f}")
                return

            if minutes_to_close <= EOD_EXIT_MINUTES:
                log.info(f"Skip {symbol}: too close to EOD ({minutes_to_close}m left)")
                return

            # Calculate cumulative volume for vol cap
            cum_vol = sum(hist["volumes"])

            trade_size = cash  # Full balance sizing
            fill_price = st["signal_price"]

            log.info(f"SIGNAL {symbol}: gap={st['gap_pct']:.1f}%, price=${fill_price:.2f}, "
                     f"candle={st['candle_count']}, float={st['float_shares']/1e6:.1f}M")

            order = self.executor.buy(symbol, trade_size, fill_price, cumulative_volume=cum_vol)
            if order:
                entry_price = fill_price * (1 + SLIPPAGE_PCT / 100)
                st["entry_price"] = entry_price
                st["_orig_entry_price"] = entry_price
                st["entry_time"] = ts
                st["position_cost"] = trade_size
                st["shares"] = trade_size / entry_price
                st["_orig_shares"] = st["shares"]
                st["highest_since_entry"] = entry_price
                self.active_position = symbol

                log.info(f"ENTRY {symbol}: {st['shares']:.2f} shares @ ${entry_price:.2f} "
                         f"(${trade_size:,.0f}) | "
                         f"stop=${entry_price * (1 - self.params['stop_pct']/100):.2f} | "
                         f"tgt1=+{st['target1_pct']:.0f}% tgt2=+{st['target2_pct']:.0f}%")

    def eod_close(self):
        """Force close all positions at end of day."""
        if self.active_position:
            self.executor.sell(self.active_position, reason="EOD_CLOSE")
            self.active_position = None
        self.executor.close_all_positions(reason="EOD_CLOSE")

    def get_summary(self):
        """Return daily summary."""
        return {
            "trades": len(self.trades_today),
            "daily_pnl": self.daily_pnl,
            "wins": sum(1 for t in self.trades_today if t["pnl"] > 0),
            "losses": sum(1 for t in self.trades_today if t["pnl"] <= 0),
            "trade_details": self.trades_today,
        }
