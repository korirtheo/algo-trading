"""
Central configuration for live trading and backtesting.
"""
import os
import json

# --- Alpaca API ---
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "PKIPXFIETM7H4BAGQ64FQV3IWJ")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "25RY682kuN9EBcFr6SFxgbSpFjdZkn613PLKPy1TdYzG")
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"  # env override
ALPACA_FEED = "iex"  # "sip" for full market data, "iex" for free tier

# --- Risk Management ---
SLIPPAGE_PCT = 0.05
VOL_CAP_PCT = 5.0            # Max % of traded volume to take
EOD_EXIT_MINUTES = 15         # Close all positions 15 min before market close
MAX_PRICE = 50.0              # Skip stocks above this price

# --- LIVE SAFETY RAILS (broker-side enforcement, even if engine dies) ---
# Bracket orders attach a STOP and a TAKE-PROFIT leg to every buy. Alpaca
# enforces these regardless of whether the Python engine is alive.
#
# The bracket stop is a SAFETY NET — read from the strategy's Optuna-tuned
# X_STOP_PCT and BUFFERED so the engine's tighter intra-bar logic fires
# first. Bracket only triggers if the engine fails to call sell() (network
# drop, crash, halt-resume gap, etc.).
#
# Resolved per-trade as:
#   strategy_stop = tgc.{STRATEGY}_STOP_PCT  (e.g. tgc.G_STOP_PCT = 2.0)
#   bracket_stop  = min(strategy_stop * BUFFER_MULT, MAX_STOP_PCT)
#   if strategy_stop == 0 (trail-only strategy):  use DEFAULT_STOP_PCT
LIVE_BRACKET_ORDERS = True
LIVE_BRACKET_BUFFER_MULT = 1.5         # bracket is 50% wider than strategy stop
LIVE_BRACKET_DEFAULT_STOP_PCT = 12.0   # fallback for trail-only strategies (H, etc.)
LIVE_BRACKET_MAX_STOP_PCT = 25.0       # absolute ceiling — never wider than this
LIVE_BRACKET_TARGET_PCT = 50.0         # generous so strategy target/trail fires first
# Hard cap on position size: never more than this fraction of total EQUITY
# per single trade. Equity-basis (not literal cash) is correct because:
#   1. backtest's cash_box[0] accumulates wins — matches equity, not cash
#   2. on margin accounts, cash can be near-zero while equity is healthy
#   3. drawdowns eat cash first; cash-basis would lock the bot out
# Prevents the bot from putting 100%+ of equity (via buying power) into
# one microcap when position_cost arrives misconfigured.
LIVE_MAX_POSITION_PCT_OF_CASH = 30.0  # cap at 30% of EQUITY per trade (name kept for import-stability)

# --- PDT (Pattern Day Trader) — NO-OP'd 2026-06-17 ---
# FINRA abolished the $25K minimum and the 4-day-trade-in-5-days counter
# effective 2026-06-04, replacing the PDT framework with risk-based
# intraday margin under amended Rule 4210. The local gate is no-op'd by
# setting the floor to 0 (executor._pdt_blocked returns False when
# equity >= threshold, and any positive equity now satisfies that).
#
# Code path kept in executor.py for reversibility: if Alpaca enforces
# internally during their 18-month transition (until 2027-10-20) and we
# get rejected orders, flip PDT_EQUITY_FLOOR back to 25_000.0.
#
# Refs:
#   FINRA — https://www.finra.org/investors/insights/intraday-margin-requirements
#   SEC SR-FINRA-2025-017 (effective 2026-06-04)
PDT_EQUITY_FLOOR = 0.0         # no-op — was 25_000.0 pre-2026-06-04
PDT_EQUITY_BUFFER = 0.0
PDT_DAYTRADES_MAX = 3          # unused while floor=0; dead code in executor

# --- Scanner ---
MIN_GAP_PCT = 8.0             # Lowest min_gap across all strategies (V=8%, O=8%)
TOP_N = 20                    # Max candidates per day
MIN_PM_VOLUME = 250_000       # Minimum premarket volume

# --- Halt-Resume Scanner (intraday discovery channel) ---
HALT_MONITOR_ENABLED = os.environ.get("HALT_MONITOR_ENABLED", "true").lower() == "true"
HALT_MIN_PRICE = 1.0
HALT_MAX_PRICE = 50.0
HALT_MAX_FLOAT = 50_000_000
# NASDAQ halt reason codes to trade. T1=news pending, T2=news released,
# T12=other regulatory, LUDP/LUDS=volatility halt up/down. Market-wide
# (M1-M9) and operational (O1-O3) halts excluded.
HALT_REASONS_TRADED = ["T1", "T2", "T12", "LUDP", "LUDS"]
HALT_POLL_INTERVAL_SECS = 30

# --- Float Data ---
FLOAT_DATA = {}
_float_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "float_data.json")
if os.path.exists(_float_path):
    with open(_float_path) as _f:
        _raw = json.load(_f)
    for _tk, _v in _raw.items():
        if isinstance(_v, dict) and _v.get("floatShares"):
            FLOAT_DATA[_tk] = _v["floatShares"]
