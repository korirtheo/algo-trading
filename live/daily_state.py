"""
Daily state persistence for crash recovery.

Tracks:
- exit_prices: (ticker, strategy) -> exit_price (for re-entry floor)
- done_strategies: set of (ticker, strategy) that are done for the day
- last_states: ticker -> last known state from simulate (prevents re-evaluation)
- first_signal_times: (ticker, strategy) -> timestamp when strategy first signaled

This ensures that on restart:
1. We don't re-buy tickers that were already rejected/done
2. We respect strategy time limits (G shouldn't fire after 10:30 if it already timed out)
3. We maintain re-entry floor (price > last exit)
4. All timestamps are in ET
"""
import json
import os
from datetime import datetime
from pytz import timezone

ET = timezone("US/Eastern")


class DailyStateManager:
    """Persists intraday state to survive crashes/restarts."""

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.state_file = self._get_state_file()
        self.exit_prices = {}  # (ticker, strategy) -> exit_price
        self.done_strategies = set()  # {(ticker, strategy)}
        self.last_states = {}  # ticker -> state dict
        self.first_signal_times = {}  # (ticker, strategy) -> ET timestamp
        self._load()

    def _get_state_file(self):
        """Get today's state file path."""
        today = datetime.now(ET).strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{today}_daily_state.json")

    def _load(self):
        """Load state from disk if exists."""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            # Restore exit_prices (convert string keys back to tuples)
            self.exit_prices = {
                tuple(k.split("|")): float(v)
                for k, v in data.get("exit_prices", {}).items()
            }

            # Restore done_strategies (convert string keys back to tuples)
            self.done_strategies = {
                tuple(k.split("|"))
                for k in data.get("done_strategies", [])
            }

            # Restore last_states
            self.last_states = data.get("last_states", {})

            # Restore first_signal_times
            self.first_signal_times = {
                tuple(k.split("|")): v
                for k, v in data.get("first_signal_times", {}).items()
            }

            print(f"[DailyState] Loaded state: {len(self.exit_prices)} exits, "
                  f"{len(self.done_strategies)} done, {len(self.last_states)} states, "
                  f"{len(self.first_signal_times)} signal times")

        except Exception as e:
            print(f"[DailyState] Failed to load state: {e}")

    def _save(self):
        """Save state to disk."""
        try:
            # Convert tuples to strings for JSON serialization
            data = {
                "exit_prices": {
                    f"{ticker}|{strategy}": price
                    for (ticker, strategy), price in self.exit_prices.items()
                },
                "done_strategies": [
                    f"{ticker}|{strategy}"
                    for ticker, strategy in self.done_strategies
                ],
                "last_states": self.last_states,
                "first_signal_times": {
                    f"{ticker}|{strategy}": ts
                    for (ticker, strategy), ts in self.first_signal_times.items()
                },
                "saved_at": datetime.now(ET).isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            print(f"[DailyState] Failed to save state: {e}")

    def record_exit(self, ticker, strategy, exit_price):
        """Record an exit price for re-entry floor tracking."""
        self.exit_prices[(ticker, strategy)] = exit_price
        self._save()

    def mark_done(self, ticker, strategy):
        """Mark a strategy as done for the day (timed out or finished)."""
        self.done_strategies.add((ticker, strategy))
        self._save()

    def is_done(self, ticker, strategy):
        """Check if strategy is done for the day."""
        return (ticker, strategy) in self.done_strategies

    def update_last_state(self, ticker, state):
        """Update last known state for a ticker."""
        self.last_states[ticker] = state
        self._save()

    def record_signal_time(self, ticker, strategy, timestamp):
        """Record first signal time for time limit enforcement."""
        key = (ticker, strategy)
        if key not in self.first_signal_times:
            # Convert to ET if not already
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            if timestamp.tzinfo is None:
                timestamp = ET.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(ET)

            self.first_signal_times[key] = timestamp.isoformat()
            self._save()

    def get_signal_time(self, ticker, strategy):
        """Get first signal time for time limit check."""
        key = (ticker, strategy)
        ts = self.first_signal_times.get(key)
        if ts:
            return datetime.fromisoformat(ts)
        return None

    def check_time_limit(self, ticker, strategy, time_limit_min):
        """Check if strategy has exceeded its time window."""
        signal_time = self.get_signal_time(ticker, strategy)
        if not signal_time:
            return False  # No signal yet, not expired

        now = datetime.now(ET)
        elapsed_min = (now - signal_time).total_seconds() / 60
        return elapsed_min > time_limit_min

    def get_exit_price(self, ticker, strategy):
        """Get exit price for re-entry floor check."""
        return self.exit_prices.get((ticker, strategy))

    def reset_for_new_day(self):
        """Clear state for a new trading day."""
        self.exit_prices.clear()
        self.done_strategies.clear()
        self.last_states.clear()
        self.first_signal_times.clear()
        # State file path will update automatically via _get_state_file()
        self.state_file = self._get_state_file()
        self._save()

    def get_summary(self):
        """Get summary of current state."""
        return {
            "exit_prices": len(self.exit_prices),
            "done_strategies": len(self.done_strategies),
            "last_states": len(self.last_states),
            "first_signal_times": len(self.first_signal_times),
        }
