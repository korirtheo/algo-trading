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
from strategies import halt_resume as hr
from config.settings import FLOAT_DATA

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# LIVE_PARAMS_PATH env var lets us switch the live config (e.g. between
# trial_432, trial_6_extracted, or a fresh optimizer dump) without code
# changes. Path is resolved relative to the project root if not absolute.
_env_params = os.environ.get("LIVE_PARAMS_PATH")
if _env_params:
    PARAMS_PATH = _env_params if os.path.isabs(_env_params) \
        else os.path.join(_PROJECT_ROOT, _env_params)
else:
    PARAMS_PATH = os.path.join(_PROJECT_ROOT, "config", "trial_432_params.json")


def load_trial_params(path=None):
    """Load trial params from JSON and apply to tgc module.

    Supports two on-disk formats:
      1. Flat dict of param name -> value (legacy trial_432_params.json style)
      2. Extracted-trial dict with top-level metadata + "params" subkey
         (e.g. trial_124_microcap_pump_extracted.json). For this format we
         use the params subkey and merge over the trial_432 baseline so any
         params Optuna didn't touch fall back to a known-good default.
    """
    path = path or PARAMS_PATH
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
        baseline_path = os.path.join(_PROJECT_ROOT, "config", "trial_432_params.json")
        with open(baseline_path) as bf:
            baseline = json.load(bf)
        params = dict(baseline)
        params.update(raw["params"])
        log.info("Loaded extracted trial #%s (study=%s): %d tuned params merged over %d baseline keys",
                 raw.get("trial_number", "?"), raw.get("study", "?"),
                 len(raw["params"]), len(baseline))
    else:
        params = raw
    set_strategy_params(params)
    log.info("Loaded %d params from %s", len(params), os.path.basename(path))

    # Live-only: zero the volume caps when IEX feed under-counts cum_$vol.
    # Without this, sizing binds at ~$100 on hot gappers because the IEX feed
    # only sees ~0.1% of market volume. The 30% equity cap in the executor
    # remains binding for true risk control. See config/settings.py for
    # the LIVE_DISABLE_VOL_CAPS rationale + removal trigger.
    try:
        from config.settings import LIVE_DISABLE_VOL_CAPS
    except ImportError:
        LIVE_DISABLE_VOL_CAPS = False
    if LIVE_DISABLE_VOL_CAPS:
        tgc.MAX_2MIN_PARTICIPATION = 0.0
        tgc.MAX_REGIME_PARTICIPATION = 0.0
        tgc.VOL_CAP_PCT = 0.0
        log.warning(
            "LIVE_DISABLE_VOL_CAPS=True: zeroed MAX_2MIN_PARTICIPATION, "
            "MAX_REGIME_PARTICIPATION, VOL_CAP_PCT. Sizing falls back to "
            "30%% equity cap in executor.py. Restore by setting "
            "LIVE_DISABLE_VOL_CAPS=False once SIP feed is active."
        )

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


TRADE_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def _trade_log_path():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    return os.path.join(TRADE_LOG_DIR, f"{today}_trades.json")


def _load_today_trades():
    path = _trade_log_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                trades = json.load(f)
            log.info("Restored %d trades from %s", len(trades), os.path.basename(path))
            return trades
        except Exception as e:
            log.warning("Could not load trade log: %s", e)
    return []


def _append_trade(trade):
    os.makedirs(TRADE_LOG_DIR, exist_ok=True)
    path = _trade_log_path()
    trades = _load_today_trades()
    trades.append({k: str(v) if hasattr(v, 'isoformat') else v for k, v in trade.items()})
    with open(path, "w") as f:
        json.dump(trades, f, indent=2, default=str)


class CombinedEngine:
    """Runs all 12 strategies using the backtest's simulate_day_combined()."""

    def __init__(self, executor, params_path=None):
        self.executor = executor
        self.params = load_trial_params(params_path)
        self.bar_data = {}      # ticker -> list of (timestamp, OHLCV dict)
        self.picks = []         # list of pick dicts (scanner output)
        self.last_states = {}   # ticker -> last known state from simulate (for entry/exit detection)
        self.all_states = {}    # ticker -> list of ALL sub-states (main, l_only, o_only, b_only, e_only)
        self.active_position = None  # ticker currently in position
        self.position_entry = {}     # ticker -> {entry_price, shares, cost}
        self.daily_pnl = 0.0
        self.trades_today = _load_today_trades()
        self.daily_pnl = sum(t.get("pnl", 0) for t in self.trades_today if isinstance(t.get("pnl"), (int, float)))
        # Halt-resume scanner state: tickers added mid-day by the halt monitor.
        # These run a SEPARATE evaluator (strategies.halt_resume) and never
        # touch simulate_day_combined.
        self.halt_states = {}        # ticker -> state dict from hr.create_state

    def initialize_watchlist(self, candidates):
        """Set up from scanner candidates.

        Args:
            candidates: list of dicts with ticker, gap_pct, pm_volume,
                        premarket_high, prev_close, float_shares
        """
        self.bar_data.clear()
        self.last_states.clear()
        self.all_states.clear()
        self.active_position = None
        self.position_entry.clear()
        self.daily_pnl = 0.0
        self.trades_today = []
        self.halt_states.clear()

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
                "float_shares": cand.get("float_shares"),  # needed for L strategy
                "market_hour_candles": None,  # Built incrementally
            })

        log.info("Initialized %d candidates for combined strategy", len(self.picks))

    def on_intraday_addition(self, ticker, halt_event, source="halt_resume"):
        """Register a ticker discovered mid-day (e.g. via halt-resume scanner).

        Initializes per-ticker bar tracking and a halt-resume strategy state.
        Does NOT add to self.picks — the halt-resume path runs separately from
        simulate_day_combined.

        Args:
            ticker: symbol string
            halt_event: object exposing .reason, .resume_dt, .resume_price,
                        .halt_price (see live.halt_monitor.HaltEvent)
            source: tag for logs/diagnostics. Currently only "halt_resume".
        """
        ticker = ticker.upper()
        if ticker in self.halt_states:
            log.debug("on_intraday_addition: %s already tracked (skip)", ticker)
            return False

        float_shares = FLOAT_DATA.get(ticker)  # None if unknown — permissive
        if not hr.is_eligible(halt_event, float_shares=float_shares):
            log.info("on_intraday_addition: %s ineligible (reason=%s resume=%s float=%s)",
                     ticker, halt_event.reason, halt_event.resume_price, float_shares)
            return False

        self.bar_data.setdefault(ticker, [])
        self.halt_states[ticker] = hr.create_state(ticker, halt_event,
                                                   float_shares=float_shares)
        log.info("INTRADAY-ADD %s (source=%s): reason=%s resume=$%.3f float=%s",
                 ticker, source, halt_event.reason,
                 halt_event.resume_price or 0.0,
                 (f"{float_shares/1e6:.1f}M" if float_shares else "N/A"))
        return True

    def on_bar(self, symbol, bar):
        """Process a completed 2-min bar.

        Appends to the ticker's bar history, rebuilds the DataFrame,
        and runs the full simulation to detect state changes.
        """
        # Halt-resume path runs separately from the simulate_day_combined flow.
        if symbol in self.halt_states:
            self._on_bar_halt(symbol, bar)
            return

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
            df = df[~df.index.duplicated(keep="last")]

            pick_copy = dict(pick)
            pick_copy["market_hour_candles"] = df
            if pick_copy["market_open"] is None and len(df) > 0:
                pick_copy["market_open"] = float(df.iloc[0]["Open"])
            picks_with_data.append(pick_copy)

        if not picks_with_data:
            return

        # Run the backtest simulation on accumulated bars
        cash = self.executor.get_buying_power()
        log.debug("on_bar %s | cash=$%.0f | %d picks with data", symbol, cash, len(picks_with_data))
        states, _, _, _ = tgc.simulate_day_combined(picks_with_data, cash, is_live=True)

        # Log any state that has an entry (for diagnostics)
        for st in states:
            if st.get("entry_price") is not None:
                log.debug("  sim-entry: %s strat=%s entry=$%.3f cost=$%.0f",
                          st["ticker"], st.get("strategy"), st.get("entry_price"), st.get("position_cost", 0))

        # Store all sub-states per ticker for diagnostics
        per_ticker = {}
        for st in states:
            per_ticker.setdefault(st["ticker"], []).append(dict(st))
        self.all_states = per_ticker

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
                    cum_dollar = self._cum_dollar_vol(ticker)

                    log.info("SIGNAL %s (strategy %s): price=$%.3f gap=%.1f%% cost=$%.0f cum_$vol=$%.0f",
                             ticker, strategy, entry_price, st.get("gap_pct", 0), trade_size, cum_dollar)

                    order = self.executor.buy(ticker, trade_size, entry_price,
                                             cumulative_dollar_volume=cum_dollar,
                                             strategy=strategy)
                    if order:
                        self.active_position = ticker
                        self.position_entry[ticker] = {
                            "entry_price": entry_price,
                            "shares": st.get("shares", 0),
                            "cost": trade_size,
                            "strategy": strategy,
                            "entry_time": ts,
                        }
                        log.info("ENTRY %s (%s): %.2f shares @ $%.3f ($%s)",
                                 ticker, strategy, st.get("shares", 0), entry_price,
                                 format(trade_size, ",.0f"))
                    else:
                        log.warning("BUY REJECTED %s: order returned None (vol_cap or executor error)", ticker)
                elif self.active_position is not None:
                    log.debug("SIGNAL %s skipped — already in position %s", ticker, self.active_position)

            # Exit detected
            if st.get("exit_price") is not None and (prev is None or prev.get("exit_price") is None):
                if ticker == self.active_position:
                    exit_price = st["exit_price"]
                    exit_reason = st.get("exit_reason", "UNKNOWN")
                    pnl = st.get("pnl", 0)

                    entry_info = self.position_entry.get(ticker, {})
                    order = self.executor.sell(
                        ticker, reason=exit_reason,
                        signal_price=exit_price,
                        cumulative_dollar_volume=self._cum_dollar_vol(ticker),
                        strategy=entry_info.get("strategy", "?"),
                    )
                    if order:
                        self.active_position = None
                        self.daily_pnl += pnl
                        entry_info = self.position_entry.get(ticker, {})

                        trade = {
                            "ticker": ticker,
                            "strategy": entry_info.get("strategy", "?"),
                            "entry_price": entry_info.get("entry_price", 0),
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "reason": exit_reason,
                            "entry_time": entry_info.get("entry_time"),
                            "exit_time": ts,
                        }
                        self.trades_today.append(trade)
                        _append_trade(trade)
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
                    pinfo = self.position_entry.get(ticker, {})
                    self.executor.sell(
                        ticker, shares=sold_shares, reason="PARTIAL",
                        signal_price=st.get("partial_price") or st.get("close"),
                        strategy=pinfo.get("strategy", "?"),
                    )

            self.last_states[ticker] = dict(st)

    def _on_bar_halt(self, symbol, bar):
        """Halt-resume strategy bar handler. Runs independently of
        simulate_day_combined. Mirrors the entry/exit/partial bookkeeping in
        on_bar so the same dashboard + trade-log surfaces apply.
        """
        ts = bar["timestamp"]
        self.bar_data.setdefault(symbol, []).append({
            "timestamp": ts,
            "Open": bar["Open"],
            "High": bar["High"],
            "Low": bar["Low"],
            "Close": bar["Close"],
            "Volume": bar["Volume"],
        })

        state = self.halt_states.get(symbol)
        if state is None or state.get("done"):
            return

        c_open = float(bar["Open"])
        c_high = float(bar["High"])
        c_low = float(bar["Low"])
        c_close = float(bar["Close"])
        c_vol = float(bar["Volume"])

        # ----- Entry path -----
        if state["entry_price"] is None:
            # Per spec open-question (2): single-position semantics — skip if
            # already in a (non-halt) trade.
            if self.active_position is not None and self.active_position != symbol:
                log.debug("HALT %s: signal suppressed — already long %s",
                          symbol, self.active_position)
                return

            fired = hr.check_signal(state, c_open, c_high, c_low, c_close, c_vol)
            if not fired:
                return

            cash = self.executor.get_buying_power()
            cum_dollar = self._cum_dollar_vol(symbol)
            entry_price = state["signal_price"]
            log.info("HALT-SIGNAL %s: price=$%.3f reason=%s cum_$vol=$%.0f",
                     symbol, entry_price, state["halt_reason"], cum_dollar)
            order = self.executor.buy(symbol, cash, entry_price,
                                       cumulative_dollar_volume=cum_dollar,
                                       strategy="HALT")
            if order is None:
                log.warning("HALT BUY REJECTED %s (vol_cap or executor error)", symbol)
                state["done"] = True
                return

            shares = getattr(order, "qty", None)
            shares = float(shares) if shares is not None else (cash / entry_price)
            state["entry_price"] = entry_price
            state["entry_time"] = ts
            state["shares"] = shares
            state["position_cost"] = shares * entry_price
            state["highest_since_entry"] = c_high
            self.active_position = symbol
            self.position_entry[symbol] = {
                "entry_price": entry_price,
                "shares": shares,
                "cost": state["position_cost"],
                "strategy": "HALT",
                "entry_time": ts,
            }
            log.info("HALT-ENTRY %s: %.2f shares @ $%.3f ($%s)",
                     symbol, shares, entry_price, format(state["position_cost"], ",.0f"))
            return

        # ----- Exit path -----
        entry_time = state.get("entry_time") or ts
        try:
            mins_in = max(0, int((ts - entry_time).total_seconds() // 60))
        except Exception:
            mins_in = 0

        # Minutes to 4:00 PM ET — used by EOD branch
        try:
            ts_et = ts.astimezone(ET) if hasattr(ts, "astimezone") else ts
            close_dt = datetime.combine(ts_et.date(),
                                        datetime.strptime("16:00", "%H:%M").time(),
                                        tzinfo=ET)
            mins_to_close = max(0, int((close_dt - ts_et).total_seconds() // 60))
        except Exception:
            mins_to_close = 999

        should_exit, exit_price, reason = hr.check_exit(
            state, c_high, c_low, c_close, mins_in, mins_to_close,
        )
        if not should_exit:
            return

        if reason == "PARTIAL":
            sell_shares = state["shares"] * (hr.DEFAULT_PARAMS["partial_sell_pct"] / 100.0)
            log.info("HALT-PARTIAL %s: %.2f shares @ $%.3f",
                     symbol, sell_shares, exit_price)
            self.executor.sell(symbol, shares=sell_shares, reason="HALT_PARTIAL",
                               signal_price=exit_price, strategy="HALT")
            state["shares"] = max(0.0, state["shares"] - sell_shares)
            state["partial_proceeds"] = sell_shares * exit_price
            return

        # Full exit
        log.info("HALT-EXIT %s (%s): @ $%.3f", symbol, reason, exit_price)
        order = self.executor.sell(symbol, reason=f"HALT_{reason}",
                                    signal_price=exit_price, strategy="HALT")
        if order is not None or True:  # always finalize state even if executor was noop
            entry = state["entry_price"] or 0.0
            pnl = (exit_price - entry) * state["shares"] + state.get("partial_proceeds", 0.0) - state.get("position_cost", 0.0)
            # Recompute pnl cleanly: partial_proceeds are gross. Final pnl is
            # (partial_proceeds + remaining_shares*exit_price) - original_cost.
            try:
                orig_cost = state.get("position_cost", entry * (state["shares"] + 0))
                # state["shares"] is now post-partial; reconstruct original
                if state.get("partial_taken"):
                    partial_pct = hr.DEFAULT_PARAMS["partial_sell_pct"] / 100.0
                    orig_shares = state["shares"] / max(1.0 - partial_pct, 1e-9)
                else:
                    orig_shares = state["shares"]
                orig_cost = entry * orig_shares
                proceeds = state.get("partial_proceeds", 0.0) + state["shares"] * exit_price
                pnl = proceeds - orig_cost
            except Exception:
                pass

            state["exit_price"] = exit_price
            state["exit_time"] = ts
            state["exit_reason"] = reason
            state["pnl"] = pnl
            state["done"] = True
            self.daily_pnl += pnl
            if self.active_position == symbol:
                self.active_position = None
            trade = {
                "ticker": symbol,
                "strategy": "HALT",
                "entry_price": entry,
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": f"HALT_{reason}",
                "entry_time": state.get("entry_time"),
                "exit_time": ts,
            }
            self.trades_today.append(trade)
            _append_trade(trade)
            log.info("HALT-CLOSED %s: PnL=$%s | $%.2f -> $%.2f",
                     symbol, format(pnl, "+,.2f"), entry, exit_price)

    def _cum_vol(self, ticker):
        """Cumulative SHARE volume for a ticker (engine-local — undercounts
        after mid-day restart)."""
        bars = self.bar_data.get(ticker, [])
        return sum(b["Volume"] for b in bars)

    def _cum_dollar_vol(self, ticker):
        """Cumulative DOLLAR volume since 9:30 ET today, fetched live from
        Alpaca. Replaces the engine-local estimate so the vol-cap is accurate
        regardless of when the engine started (mid-day restarts, halt-resume
        tickers added intraday, etc.).

        Uses **SIP** feed (consolidated tape across all US exchanges) even
        though the live stream is IEX-only — IEX is ~2-3% of total US volume,
        so vol-cap built on IEX would clamp positions to ~3% of what it
        should. SIP historical bars are available on the free tier; only the
        real-time stream is paid.

        Falls back to engine-local sum(close*volume) if the REST call fails.
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import time as dt_time, timedelta
            from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
            now_et = datetime.now(ET)
            market_open = datetime.combine(now_et.date(), dt_time(9, 30), tzinfo=ET)
            # Free-tier SIP forbids querying the last ~15 min of "recent SIP".
            # Clamp the end of the SIP window to now - 16 min and add the
            # engine-local IEX bars for that tail. Slightly understates true
            # cumulative dollar volume by the IEX/SIP ratio over the last
            # 16 min, but is correct to within a few percent.
            sip_cutoff = now_et - timedelta(minutes=16)
            sip_dvol = 0.0
            if sip_cutoff > market_open:
                req = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Minute,
                    start=market_open,
                    end=sip_cutoff,
                    adjustment="raw",
                    feed="sip",
                )
                resp = client.get_stock_bars(req)
                if not resp.df.empty:
                    df = resp.df.reset_index()
                    sip_dvol = float((df["close"] * df["volume"]).sum())
            # Add engine-local IEX bars that arrived AFTER the SIP cutoff.
            tail_dvol = 0.0
            for b in self.bar_data.get(ticker, []):
                ts = b["timestamp"]
                ts_et = ts.astimezone(ET) if hasattr(ts, "astimezone") else ts
                if ts_et >= sip_cutoff:
                    tail_dvol += float(b["Close"]) * float(b["Volume"])
            return sip_dvol + tail_dvol
        except Exception as e:
            log.warning(f"_cum_dollar_vol({ticker}) REST failed: {e}; "
                        f"falling back to local")
            bars = self.bar_data.get(ticker, [])
            return float(sum(float(b["Close"]) * float(b["Volume"]) for b in bars))

    def eod_close(self):
        """Force close all positions."""
        if self.active_position:
            self.executor.sell(self.active_position, reason="EOD_CLOSE")
            self.active_position = None
        # Mark any unfinished halt-resume states as done so they don't fire
        # entries on the next session if the process keeps running.
        for st in self.halt_states.values():
            st["done"] = True
        self.executor.close_all_positions(reason="EOD_CLOSE")
        self._log_eod_diagnostics()

    def _log_eod_diagnostics(self):
        """Log final simulation state for every ticker to diagnose why signals didn't fire."""
        if not self.picks:
            return
        picks_with_data = []
        for pick in self.picks:
            ticker = pick["ticker"]
            bars = self.bar_data.get(ticker, [])
            if not bars:
                log.info("EOD-DIAG %s: 0 bars received", ticker)
                continue
            df = pd.DataFrame(bars)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            pick_copy = dict(pick)
            pick_copy["market_hour_candles"] = df
            if pick_copy["market_open"] is None:
                pick_copy["market_open"] = float(df.iloc[0]["Open"])
            picks_with_data.append(pick_copy)

        if not picks_with_data:
            return

        cash = self.executor.get_buying_power()
        states, _, _, _ = tgc.simulate_day_combined(picks_with_data, cash, is_live=True)
        log.info("=== EOD DIAGNOSTICS (%d tickers, %d bars avg) ===",
                 len(picks_with_data),
                 sum(len(self.bar_data.get(p["ticker"], [])) for p in picks_with_data) // max(len(picks_with_data), 1))
        for st in states:
            tk = st["ticker"]
            candles = st.get("candle_count", 0)
            gap = st.get("gap_pct", 0)
            entry = st.get("entry_price")
            strategy = st.get("strategy", "none")
            if entry:
                log.info("  EOD-DIAG %s: TRADED strat=%s candles=%d gap=%.1f%%",
                         tk, strategy, candles, gap)
            else:
                # Log which strategies were eligible
                eligible = []
                for s in "HGAFDVPMRWOBKCEIJNL":
                    if st.get(f"{s.lower()}_eligible", False):
                        eligible.append(s)
                log.info("  EOD-DIAG %s: NO SIGNAL | candles=%d gap=%.1f%% eligible=%s pm_high=%.3f open=%.3f",
                         tk, candles, gap, eligible or "none",
                         pick.get("premarket_high", 0) if (pick := next((p for p in self.picks if p["ticker"] == tk), {})) else 0,
                         st.get("market_open", 0))

    def get_summary(self):
        return {
            "trades": len(self.trades_today),
            "daily_pnl": self.daily_pnl,
            "wins": sum(1 for t in self.trades_today if t["pnl"] > 0),
            "losses": sum(1 for t in self.trades_today if t["pnl"] <= 0),
            "trade_details": self.trades_today,
        }
