"""
Central configuration for live trading and backtesting.
"""
import os
import json

# --- Alpaca API ---
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "PKZGRFUVCPRAAZT34RPCCED47X")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "ALfFBJStYK5KGGYfN5AE34Lvz46DLxQJrWVYAnNCeFKi")
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"  # env override
ALPACA_FEED = "iex"  # "sip" for full market data, "iex" for free tier

# --- Risk Management ---
SLIPPAGE_PCT = 0.05
VOL_CAP_PCT = 5.0            # Max % of traded volume to take
EOD_EXIT_MINUTES = 15         # Close all positions 15 min before market close
MAX_PRICE = 50.0              # Skip stocks above this price

# --- Scanner ---
MIN_GAP_PCT = 8.0             # Lowest min_gap across all strategies (V=8%, O=8%)
TOP_N = 20                    # Max candidates per day
MIN_PM_VOLUME = 250_000       # Minimum premarket volume

# --- Float Data ---
FLOAT_DATA = {}
_float_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "float_data.json")
if os.path.exists(_float_path):
    with open(_float_path) as _f:
        _raw = json.load(_f)
    for _tk, _v in _raw.items():
        if isinstance(_v, dict) and _v.get("floatShares"):
            FLOAT_DATA[_tk] = _v["floatShares"]
