"""
Combined Strategy Engine: Uses all 12 strategies from Optuna trial 432.

Instead of extracting each strategy individually, this engine uses the
existing test_green_candle_combined.py module directly — loading trial params
via set_strategy_params() from optimize_combined.py.

For live trading, it:
1. Loads trial params into the backtest module globals
2. Receives real-time 2-min bars and builds per-ticker DataFrames
3. At each bar, runs simulate_day_combined() on the accumulated bars
4. Detects new entries/exits by comparing state changes
"""
import json
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import SLIPPAGE_PCT, VOL_CAP_PCT, ET_TZ

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config", "trial_432_params.json")


def load_trial_params(path=None):
    """Load trial params from JSON and apply to tgc module."""
    path = path or PARAMS_PATH
    with open(path) as f:
        params = json.load(f)
    set_strategy_params(params)
    log.info("Loaded %d params from %s", len(params), os.path.basename(path))

    # Log enabled strategies
    enabled = []
    for s in "HGAFDVPMRWOBKCEIJNL":
        gap_attr = f"{s}_MIN_GAP_PCT" if s != "R" else "R_DAY1_MIN_GAP"
        gap_val = getattr(tgc, gap_attr, 9999)
        if gap_val < 9000:
            enabled.append(s)
    log.info("Enabled strategies: %s", ", ".join(enabled))
    log.info("Priority: %s", tgc.STRAT_PRIORITY)
    return params


class CombinedEngine:
    """Runs all 12 strategies using the backtest's simulate_day_combined()."""

    def __init__(self, executor, params_path=None):
        self.executor = executor
        self.params = load_trial_params(params_path)
        self.bar_data = {}      # ticker -> list of (timestamp, OHLCV dict)
        self.picks = []         # list of pick dicts (scanner output)
        self.last_states = {}   # ticker -> last known state from simulate
        self.active_position = None  # ticker currently in position
        self.position_entry = {}     # ticker -> {entry_price, shares, cost}
        self.daily_pnl = 0.0
        self.trades_today = []

    def initialize_watchlist(self, candidates):
        """Set up from scanner candidates.

        Args:
            candidates: list of dicts with ticker, gap_pct, pm_volume,
                        premarket_high, prev_close, float_shares
        """
        self.bar_data.clear()
        self.last_states.clear()
        self.active_position = None
        self.position_entry.clear()
        self.daily_pnl = 0.0
        self.trades_today = []

        self.picks = []
        for cand in candidates:
            ticker = cand["ticker"]
            self.bar_data[ticker] = []
            self.picks.append({
                "ticker": ticker,
                "gap_pct": cand["gap_pct"],
                "market_open": None,  # Will be set from first bar
                "premarket_high": cand["premarket_high"],
                "prev_close": cand["prev_close"],
                "pm_volume": cand["pm_volume"],
                "market_hour_candles": None,  # Built incrementally
            })

        log.info("Initialized %d candidates for combined strategy", len(self.picks))

    def on_bar(self, symbol, bar):
        """Process a completed 2-min bar.

        Appends to the ticker's bar history, rebuilds the DataFrame,
        and runs the full simulation to detect state changes.
        """
        if symbol not in self.bar_data:
            return

        ts = bar["timestamp"]
        self.bar_data[symbol].append({
            "timestamp": ts,
            "Open": bar["Open"],
            "High": bar["High"],
            "Low": bar["Low"],
            "Close": bar["Close"],
            "Volume": bar["Volume"],
        })

        # Rebuild DataFrames for all tickers with data
        picks_with_data = []
        for pick in self.picks:
            ticker = pick["ticker"]
            bars = self.bar_data.get(ticker, [])
            if not bars:
                continue

            df = pd.DataFrame(bars)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()

            pick_copy = dict(pick)
            pick_copy["market_hour_candles"] = df
            if pick_copy["market_open"] is None and len(df) > 0:
                pick_copy["market_open"] = float(df.iloc[0]["Open"])
            picks_with_data.append(pick_copy)

        if not picks_with_data:
            return

        # Run the backtest simulation on accumulated bars
        cash = self.executor.get_buying_power()
        states, _, _, _ = tgc.simulate_day_combined(picks_with_data, cash)

        # Detect state changes
        for st in states:
            ticker = st["ticker"]
            prev = self.last_states.get(ticker)

            # New entry detected
            if st.get("entry_price") is not None and (prev is None or prev.get("entry_price") is None):
                if self.active_position is None:
                    entry_price = st["entry_price"]
                    strategy = st.get("strategy", "?")
                    trade_size = st.get("position_cost", cash)

                    log.info("SIGNAL %s (strategy %s): price=$%.2f, gap=%.1f%%",
                             ticker, strategy, entry_price, st.get("gap_pct", 0))

                    order = self.executor.buy(ticker, trade_size, entry_price,
                                             cumulative_volume=self._cum_vol(ticker))
                    if order:
                        self.active_position = ticker
                        self.position_entry[ticker] = {
                            "entry_price": entry_price,
                            "shares": st.get("shares", 0),
                            "cost": trade_size,
                            "strategy": strategy,
                            "entry_time": ts,
                        }
                        log.info("ENTRY %s (%s): %.2f shares @ $%.2f ($%s)",
                                 ticker, strategy, st.get("shares", 0), entry_price,
                                 format(trade_size, ",.0f"))

            # Exit detected
            if st.get("exit_price") is not None and (prev is None or prev.get("exit_price") is None):
                if ticker == self.active_position:
                    exit_price = st["exit_price"]
                    exit_reason = st.get("exit_reason", "UNKNOWN")
                    pnl = st.get("pnl", 0)

                    order = self.executor.sell(ticker, reason=exit_reason)
                    if order:
                        self.active_position = None
                        self.daily_pnl += pnl
                        entry_info = self.position_entry.get(ticker, {})

                        self.trades_today.append({
                            "ticker": ticker,
                            "strategy": entry_info.get("strategy", "?"),
                            "entry_price": entry_info.get("entry_price", 0),
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "reason": exit_reason,
                            "entry_time": entry_info.get("entry_time"),
                            "exit_time": ts,
                        })
                        log.info("EXIT %s (%s): PnL=$%s | $%.2f -> $%.2f",
                                 ticker, exit_reason, format(pnl, "+,.2f"),
                                 entry_info.get("entry_price", 0), exit_price)

            # Partial sell detected
            if (prev is not None
                and st.get("shares", 0) < prev.get("shares", 0)
                and st.get("entry_price") is not None
                and prev.get("entry_price") is not None):
                sold_shares = prev["shares"] - st["shares"]
                if sold_shares > 0.001 and ticker == self.active_position:
                    log.info("PARTIAL SELL %s: %.2f shares", ticker, sold_shares)
                    self.executor.sell(ticker, shares=sold_shares, reason="PARTIAL")

            self.last_states[ticker] = dict(st)

    def _cum_vol(self, ticker):
        """Get cumulative volume for a ticker."""
        bars = self.bar_data.get(ticker, [])
        return sum(b["Volume"] for b in bars)

    def eod_close(self):
        """Force close all positions."""
        if self.active_position:
            self.executor.sell(self.active_position, reason="EOD_CLOSE")
            self.active_position = None
        self.executor.close_all_positions(reason="EOD_CLOSE")

    def get_summary(self):
        return {
            "trades": len(self.trades_today),
            "daily_pnl": self.daily_pnl,
            "wins": sum(1 for t in self.trades_today if t["pnl"] > 0),
            "losses": sum(1 for t in self.trades_today if t["pnl"] <= 0),
            "trade_details": self.trades_today,
        }
