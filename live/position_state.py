"""
Persistent position state manager - ensures positions survive crashes.

Maintains a JSON file tracking all open positions with full metadata needed for
monitoring (entry price, peak, strategy, stop/target levels). On restart, can
reconcile Alpaca positions and resume monitoring.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)


class PositionStateManager:
    """Manages persistent position state across crashes/restarts."""

    def __init__(self, state_file: str = "logs/position_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.positions: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load position state from disk."""
        if not self.state_file.exists():
            log.info("No position state file found - starting fresh")
            return

        try:
            with open(self.state_file) as f:
                self.positions = json.load(f)
            log.info(f"Loaded {len(self.positions)} positions from state file")
            for ticker, state in self.positions.items():
                log.info(f"  {ticker}: {state.get('strategy')} entry=${state.get('entry_price'):.2f} "
                        f"shares={state.get('shares')} peak=${state.get('peak_price', 0):.2f}")
        except Exception as e:
            log.error(f"Failed to load position state: {e}")
            self.positions = {}

    def _save(self):
        """Save position state to disk."""
        try:
            with self._lock:
                # Write to temp file first, then atomic rename
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    # Convert any datetime/Timestamp objects to ISO strings
                    json.dump(self.positions, f, indent=2, default=str)
                temp_file.replace(self.state_file)
        except Exception as e:
            log.error(f"Failed to save position state: {e}")

    def add_position(self, ticker: str, order_id: str, strategy: str,
                    entry_price: float, shares: int, cost: float,
                    stop_price: float, target_price: float,
                    trail_pct: float, time_limit_min: int,
                    entry_time: Optional[str] = None):
        """Record a new position entry."""
        if entry_time is None:
            entry_time = datetime.utcnow().isoformat()

        self.positions[ticker] = {
            "order_id": order_id,
            "ticker": ticker,
            "strategy": strategy,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "shares": shares,
            "cost": cost,
            "peak_price": entry_price,  # Initialize peak at entry
            "stop_price": stop_price,
            "target_price": target_price,
            "trail_pct": trail_pct,
            "time_limit_min": time_limit_min,
            "last_updated": datetime.utcnow().isoformat(),
        }
        self._save()
        log.info(f"Position state saved: {ticker} ({strategy}) @ ${entry_price:.2f}")

    def update_peak(self, ticker: str, new_peak: float):
        """Update peak price for trailing stop."""
        if ticker not in self.positions:
            return

        self.positions[ticker]["peak_price"] = new_peak
        self.positions[ticker]["last_updated"] = datetime.utcnow().isoformat()
        self._save()

    def remove_position(self, ticker: str):
        """Remove position on exit."""
        if ticker in self.positions:
            del self.positions[ticker]
            self._save()
            log.info(f"Position state removed: {ticker}")

    def get_position(self, ticker: str) -> Optional[dict]:
        """Get position state."""
        return self.positions.get(ticker)

    def get_all_positions(self) -> Dict[str, dict]:
        """Get all position states."""
        return self.positions.copy()

    def reconcile_with_alpaca(self, alpaca_positions) -> Dict[str, dict]:
        """
        Reconcile internal state with Alpaca positions.

        Returns dict of recovered positions that need monitoring setup:
        {
            "ticker": {
                "strategy": "RECOVERED",  # or actual strategy if in state
                "entry_price": ...,
                "shares": ...,
                "has_metadata": True/False,  # whether we have full monitoring data
                ...
            }
        }
        """
        recovered = {}
        alpaca_tickers = {p.symbol for p in alpaca_positions}

        # Find positions in Alpaca but not being monitored
        for alpaca_pos in alpaca_positions:
            ticker = alpaca_pos.symbol

            # Check if we have state for this position
            if ticker in self.positions:
                # We have full metadata - can resume normal monitoring
                state = self.positions[ticker]
                log.info(f"Reconcile {ticker}: Found in state, strategy={state['strategy']}, "
                        f"entry=${state['entry_price']:.2f}, peak=${state.get('peak_price', 0):.2f}")
                recovered[ticker] = {
                    **state,
                    "has_metadata": True,
                    "alpaca_qty": int(alpaca_pos.qty),
                    "alpaca_avg_price": float(alpaca_pos.avg_entry_price),
                    "current_price": float(alpaca_pos.current_price),
                    "unrealized_pl": float(alpaca_pos.unrealized_pl),
                }
            else:
                # Orphaned position - no internal state
                # Create defensive recovery state
                entry_price = float(alpaca_pos.avg_entry_price)
                current_price = float(alpaca_pos.current_price)
                shares = int(alpaca_pos.qty)

                log.warning(f"Reconcile {ticker}: ORPHANED position found! "
                           f"{shares} shares @ ${entry_price:.2f}, current=${current_price:.2f}")

                # Defensive defaults: tight stop, conservative trail
                recovered[ticker] = {
                    "ticker": ticker,
                    "strategy": "RECOVERED",
                    "entry_price": entry_price,
                    "entry_time": datetime.utcnow().isoformat(),  # Conservative - no time stop
                    "shares": shares,
                    "cost": entry_price * shares,
                    "peak_price": max(entry_price, current_price),  # Start tracking from now
                    "stop_price": entry_price * 0.95,  # 5% stop loss
                    "target_price": entry_price * 1.10,  # 10% take profit
                    "trail_pct": 1.0,  # 1% trail
                    "time_limit_min": 999,  # Effectively no time limit
                    "last_updated": datetime.utcnow().isoformat(),
                    "has_metadata": False,
                    "alpaca_qty": shares,
                    "alpaca_avg_price": entry_price,
                    "current_price": current_price,
                    "unrealized_pl": float(alpaca_pos.unrealized_pl),
                }

                # Save this recovered position to state
                self.positions[ticker] = {
                    k: v for k, v in recovered[ticker].items()
                    if k not in ["has_metadata", "alpaca_qty", "alpaca_avg_price",
                                "current_price", "unrealized_pl"]
                }
                self._save()

        # Clean up state for positions that are no longer in Alpaca
        for ticker in list(self.positions.keys()):
            if ticker not in alpaca_tickers:
                log.warning(f"Position {ticker} in state file but not in Alpaca - removing")
                del self.positions[ticker]

        if self.positions and not recovered:
            log.info("Reconcile: All positions in state file match Alpaca")

        return recovered
