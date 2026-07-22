import logging
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

import test_green_candle_combined as tgc
from config.settings import SLIPPAGE_PCT, EOD_EXIT_MINUTES, FLOAT_DATA

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class StrategyEngine:
    def __init__(self, executor, params=None):
        self.executor = executor
        self.params = params
        self.states = {}
        self.bar_history = {}
        self.active_position = None
        self.daily_pnl = 0.0
        self.trades_today = []

    def initialize_watchlist(self, candidates):
        self.states.clear()
        self.bar_history.clear()
        self.active_position = None
        self.daily_pnl = 0.0
        self.trades_today = []

        for cand in candidates:
            ticker = cand["ticker"]
            self.states[ticker] = cand
            self.bar_history[ticker] = {
                "highs": [],
                "lows": [],
                "closes": [],
                "volumes": [],
            }

        log.info(f"Initialized {len(self.states)} candidates for L strategy")

    def on_bar(self, symbol, bar):
        if symbol not in self.states:
            return

        st = self.states[symbol]
        if st.get("done"):
            return

        # Pass the bar to the backtesting engine
        states, _, _, _ = tgc.simulate_day_combined(
            [st],
            self.executor.get_buying_power(),
            False,
            params=self.params,
            real_time_bar=bar,
        )

        # Update state
        self.states[symbol] = states[0]

        # Handle trades
        if states[0].get("exit_reason") and not st.get("exit_reason"):
            self.executor.sell(symbol, reason=states[0]["exit_reason"])
            self.active_position = None
        elif states[0].get("entry_price") and not st.get("entry_price"):
            self.executor.buy(
                symbol, self.executor.get_buying_power(), states[0]["entry_price"]
            )
            self.active_position = symbol

    def eod_close(self):
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
