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
import threading
import time
from collections import defaultdict
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
BARS_LOG_DIR = os.path.join(TRADE_LOG_DIR, "bars")

# Ensure logging directories exist at module load time
os.makedirs(TRADE_LOG_DIR, exist_ok=True)
os.makedirs(BARS_LOG_DIR, exist_ok=True)


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

    def __init__(self, executor, params_path=None, fill_stream=None):
        self.executor = executor
        self.params = load_trial_params(params_path)
        # v3 overlay: read from the config's v3_overlay section
        self.v3_params = {}
        self.v3_active_trade = None
        self._v3_allocated_today = 0.0
        try:
            _vp = params_path or PARAMS_PATH
            with open(_vp) as _f:
                _raw = json.load(_f)
            if isinstance(_raw, dict) and "v3_overlay" in _raw:
                self.v3_params = _raw["v3_overlay"]
                if self.v3_params.get("enabled", False):
                    log.info("v3 OVERLAY enabled: target=%.0f%% stop=%.0f%% time=%dmin trail=%.1f%%",
                             self.v3_params.get("target_pct", 0),
                             self.v3_params.get("stop_pct", 0),
                             self.v3_params.get("time_limit_min", 0),
                             self.v3_params.get("trail_pct", 0))
        except Exception as e:
            log.debug("No v3 overlay in config: %s", e)
        self.bar_data = {}      # ticker -> list of (timestamp, OHLCV dict)
        self.picks = []         # list of pick dicts (scanner output)
        self.last_states = {}   # ticker -> last known state from simulate (for entry/exit detection)
        self.all_states = {}    # ticker -> list of ALL sub-states (main, l_only, o_only, b_only, e_only)
        self.active_positions = set()  # set of tickers currently in position
        self.position_entry = {}     # ticker -> {entry_price, shares, cost}
        self.daily_pnl = 0.0
        self.trades_today = _load_today_trades()
        self.daily_pnl = sum(t.get("pnl", 0) for t in self.trades_today if isinstance(t.get("pnl"), (int, float)))
        # Halt-resume scanner state: tickers added mid-day by the halt monitor.
        # These run a SEPARATE evaluator (strategies.halt_resume) and never
        # touch simulate_day_combined.
        self.halt_states = {}        # ticker -> state dict from hr.create_state

        # Pending orders awaiting fill notification via TradingStream.
        # Replaces the 15s polling that caused 85 duplicate fills on GITS 2026-06-23.
        # Format: order_id -> {ticker, strategy, signal_price, signal_time, side}
        self.pending_orders = {}
        self._pending_lock = threading.Lock()
        self.fill_stream = fill_stream  # FillStream instance, or None for legacy polling

    def initialize_watchlist(self, candidates):
        """Set up from scanner candidates.

        Args:
            candidates: list of dicts with ticker, gap_pct, pm_volume,
                        premarket_high, prev_close, float_shares
        """
        self.bar_data.clear()
        self.last_states.clear()
        self.all_states.clear()
        self.active_positions.clear()
        self.position_entry.clear()
        self.daily_pnl = 0.0
        self.trades_today = []
        self.halt_states.clear()
        self.v3_active_trade = None
        self._v3_allocated_today = 0.0

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

        # FIX 2026-06-22: detect orphan positions at startup.
        # If the engine was restarted while a position was open at Alpaca,
        # the engine has no internal record of it and no strategy will
        # manage exits. Close any orphan immediately at startup to avoid
        # uncontrolled positions (the SAGT pattern observed today).
        try:
            existing = self.executor.client.get_all_positions()
            for p in existing:
                qty = float(p.qty)
                if qty < 1: continue
                log.warning("ORPHAN POSITION at startup: %s qty=%.0f avg=$%.4f — closing",
                            p.symbol, qty, float(p.avg_entry_price))
                try:
                    self.executor._cancel_bracket_legs_for_ticker(p.symbol)
                    self.executor.client.close_position(p.symbol)
                except Exception as e:
                    log.error("ORPHAN close failed for %s: %s", p.symbol, e)
        except Exception as e:
            log.warning("Orphan-position check failed: %s", e)

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

    def _log_bar_to_csv(self, symbol, bar):
        """Append each 2-min bar to a per-ticker CSV for backtest-vs-live audit.

        Output path: logs/bars/intraday/<YYYY-MM-DD>/<symbol>.csv
        Captures the EXACT 2-min bars the bot's engine processed after aggregating
        from live 1-min bars, so a follow-up backtest can run on the same data
        and produce an apples-to-apples comparison (no source-mismatch artifact
        from SIP-vs-IEX or resample boundary differences).
        """
        import csv
        try:
            today = datetime.now(ET).strftime("%Y-%m-%d")
            bars_date_dir = os.path.join(BARS_LOG_DIR, "intraday", today)
            os.makedirs(bars_date_dir, exist_ok=True)
            path = os.path.join(bars_date_dir, f"{symbol}.csv")
            new_file = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["timestamp", "Open", "High", "Low", "Close", "Volume"])
                w.writerow([
                    bar["timestamp"], bar["Open"], bar["High"],
                    bar["Low"], bar["Close"], bar["Volume"],
                ])
        except Exception as e:
            log.warning("Failed to log bar for %s: %s", symbol, e)

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

        # Log raw bar for backtest comparison (no perf impact: 1 CSV append).
        self._log_bar_to_csv(symbol, bar)

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

        # FIX 2026-06-24: comprehensive per-bar reconciliation with Alpaca.
        # Fixes 4 bugs from 2026-06-24 CCXI incident:
        #   1) pop key ordering — was popping None after clearing active
        #   2) KeyError silently swallowed when position_entry missing the ticker dict
        #   3) active_position stuck forever if Alpaca says no-position but internal empty
        #   4) no fallback to rehydrate position_entry from Alpaca when stream miss the fill
        if symbol in self.active_positions:
            self._reconcile_position(symbol, ts)

        # --- v3 overlay exit check ---
        if (self.v3_params.get("enabled", False)
                and symbol in self.active_positions
                and self.position_entry.get(symbol, {}).get("strategy") == "V3"):
            self._manage_v3_exit(symbol, bar)

        # Store all sub-states per ticker for diagnostics
        per_ticker = {}
        for st in states:
            per_ticker.setdefault(st["ticker"], []).append(dict(st))
        self.all_states = per_ticker

        # Detect state changes
        for st in states:
            ticker = st["ticker"]
            prev = self.last_states.get(ticker)

            # FIX 2026-06-22: when a ticker is already active, only respond
            # to state changes from the SAME strategy that owns the position.
            # The simulator returns one state PER STRATEGY per ticker (G state,
            # L state, etc.). Without this guard, an irrelevant strategy's
            # state (e.g. L's parallel evaluation of SAGT) triggers phantom
            # partial-sells when its shares value differs from prev. The
            # SAGT 30% phantom sell at 14:35 today is the symptom.
            if (ticker in self.active_positions
                    and self.position_entry.get(ticker, {}).get("strategy")
                    and st.get("strategy")
                    and st["strategy"] != self.position_entry[ticker]["strategy"]):
                continue

            # New entry detected
            if st.get("entry_price") is not None and (prev is None or prev.get("entry_price") is None):
                if ticker in self.active_positions:
                    log.debug("SIGNAL %s skipped — already in this ticker", ticker)
                    continue
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
                    # 2026-06-23 FIX: previously polled for 15s and gave up — caused
                    # 85 untracked duplicate fills on GITS today (microcap fills took
                    # >15s; each "give up" triggered a NEW signal).
                    # New: claim the slot IMMEDIATELY (active_positions + pending_orders),
                    # register fill callback with TradingStream, return control.
                    # Stream callback resolves position_entry with actual filled qty/avg
                    # when the fill arrives — could be 2s or 30s, no longer matters.
                    self.active_positions.add(ticker)
                    # 2026-06-24 FIX: bootstrap position_entry IMMEDIATELY (using
                    # signal_price as entry estimate, shares=0 placeholder). The
                    # CCXI 2026-06-24 incident showed that the TradingStream fill
                    # callback can silently miss, leaving position_entry empty for
                    # the entire trade. Without strategy/entry_time, TIME_STOP can't
                    # fire, and RECONCILE can't rehydrate (no internal record to update).
                    # With this bootstrap, RECONCILE will see internal_qty=0 < Alpaca
                    # actual_qty, hit the "rehydrate" path, and self-heal on the next bar.
                    if ticker not in self.position_entry:
                        self.position_entry[ticker] = {
                            "entry_price": entry_price,
                            "shares": 0,  # placeholder — RECONCILE/stream fills this in
                            "cost": trade_size,
                            "strategy": strategy,
                            "entry_time": ts,
                        }
                    with self._pending_lock:
                        self.pending_orders[str(order.id)] = {
                            "ticker": ticker, "strategy": strategy,
                            "signal_price": entry_price, "signal_time": ts,
                            "side": "buy", "requested_cost": trade_size,
                        }
                    if self.fill_stream is not None:
                        self.fill_stream.register(order.id, self._on_buy_fill)
                        log.info("BUY %s: order %s placed, awaiting TradingStream fill notification",
                                 ticker, order.id)
                        # Set safety-net polling: if stream misses for >60s, force-poll
                        self._schedule_safety_poll(order.id, ticker, "buy", entry_price, ts, strategy)
                    else:
                        # Legacy path — no stream, poll inline (kept for backward compat)
                        self._poll_buy_fill_inline(order, ticker, strategy, entry_price, ts)
                else:
                    log.warning("BUY REJECTED %s: order returned None (vol_cap or executor error)", ticker)

            # Exit detected
            if st.get("exit_price") is not None and (prev is None or prev.get("exit_price") is None):
                if ticker in self.active_positions:
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
                        # 2026-06-23: stream-based sell tracking — don't clear
                        # active_position until fill confirmed. Prevents phantom
                        # "I sold" state when actually order was rejected/unfilled.
                        with self._pending_lock:
                            self.pending_orders[str(order.id)] = {
                                "ticker": ticker, "strategy": entry_info.get("strategy", "?"),
                                "signal_price": exit_price, "signal_time": ts,
                                "side": "sell", "exit_reason": exit_reason,
                                "expected_pnl": pnl,
                                "pre_sell_shares": entry_info.get("shares", 0),
                                "is_partial": False,
                            }
                        if self.fill_stream is not None:
                            self.fill_stream.register(order.id, self._on_sell_fill)
                            log.info("SELL %s (%s): order %s placed, awaiting stream fill",
                                     ticker, exit_reason, order.id)
                            self._schedule_sell_safety_poll(order.id, ticker)
                        else:
                            # Legacy synchronous behavior
                            self.active_positions.discard(ticker)
                            self.daily_pnl += pnl
                            entry_info_l = self.position_entry.get(ticker, {})
                            trade = {
                                "ticker": ticker,
                                "strategy": entry_info_l.get("strategy", "?"),
                                "entry_price": entry_info_l.get("entry_price", 0),
                                "exit_price": exit_price,
                                "pnl": pnl,
                                "reason": exit_reason,
                                "entry_time": entry_info_l.get("entry_time"),
                                "exit_time": ts,
                            }
                            self.trades_today.append(trade)
                            _append_trade(trade)
                            log.info("EXIT %s (%s) [LEGACY]: PnL=$%s | $%.2f -> $%.2f",
                                     ticker, exit_reason, format(pnl, "+,.2f"),
                                     entry_info_l.get("entry_price", 0), exit_price)

            # Partial sell detected
            # FIX 2026-06-22: previously used the simulator's `st.shares` diff
            # directly, which was based on the phantom cash/entry_price size.
            # Now compute a FRACTION sold from the simulator state and apply
            # that fraction to the ACTUAL Alpaca position size.
            if (prev is not None
                and st.get("shares", 0) < prev.get("shares", 0)
                and st.get("entry_price") is not None
                and prev.get("entry_price") is not None
                and ticker in self.active_positions):
                pinfo = self.position_entry.get(ticker, {})
                actual_shares = float(pinfo.get("shares", 0))
                if actual_shares >= 1:
                    sim_frac_sold = (prev["shares"] - st["shares"]) / max(prev["shares"], 1e-9)
                    if sim_frac_sold >= 0.01:
                        sell_qty = int(actual_shares * sim_frac_sold)
                        if sell_qty >= 1:
                            log.info("PARTIAL SELL %s: %d shares (%.0f%% of actual %.0f)",
                                     ticker, sell_qty, sim_frac_sold * 100, actual_shares)
                            sell_order = self.executor.sell(
                                ticker, shares=sell_qty, reason="PARTIAL",
                                signal_price=st.get("partial_price") or st.get("close"),
                                strategy=pinfo.get("strategy", "?"),
                            )
                            # FIX 2026-06-22: also poll the sell for terminal status.
                            # Critical — if the sell only partially fills, our
                            # position_entry update would over-decrement and
                            # subsequent PARTIAL SELLs would target a non-existent
                            # block of shares. Wait briefly for terminal state.
                            actual_sold = sell_qty  # fallback assumption
                            if sell_order is not None:
                                import time as _time
                                term = {"filled", "canceled", "rejected", "expired"}
                                deadline = _time.time() + 10.0
                                while _time.time() < deadline:
                                    try:
                                        o = self.executor.client.get_order_by_id(sell_order.id)
                                        sst = o.status.value if hasattr(o.status, 'value') else str(o.status)
                                        if o.filled_qty:
                                            actual_sold = float(o.filled_qty)
                                        if sst in term:
                                            break
                                    except Exception:
                                        pass
                                    _time.sleep(0.3)
                            self.position_entry[ticker]["shares"] = actual_shares - actual_sold

            self.last_states[ticker] = dict(st)

        # --- v3 overlay entry check ---
        if (self.v3_params.get("enabled", False)
                and picks_with_data):
            self._check_v3_entry(picks_with_data)

    # ------------------------------------------------------------------ #
    #  Reconciliation helper                                            #
    # ------------------------------------------------------------------ #

    def _reconcile_position(self, ticker, current_ts):
        """Reconcile internal state for one ticker against Alpaca."""
        try:
            actual_pos = self.executor.client.get_open_position(ticker)
            actual_qty = float(actual_pos.qty)
            internal = self.position_entry.get(ticker, {})
            internal_qty = float(internal.get("shares", 0))
            if abs(actual_qty - internal_qty) >= 1.0:
                log.warning("RECONCILE %s: internal=%.0f, Alpaca=%.0f → fixing",
                            ticker, internal_qty, actual_qty)
                if actual_qty < 1:
                    log.info("RECONCILE %s: position closed externally — clearing", ticker)
                    self.position_entry.pop(ticker, None)
                    self.active_positions.discard(ticker)
                else:
                    if ticker not in self.position_entry or not internal:
                        try:
                            actual_avg = float(getattr(actual_pos, "avg_entry_price", 0)) \
                                         or float(actual_pos.cost_basis) / max(actual_qty, 1)
                        except (AttributeError, ValueError, ZeroDivisionError, TypeError):
                            actual_avg = 0.0
                        with self._pending_lock:
                            pending_match = next(
                                (p for p in self.pending_orders.values()
                                 if p.get("ticker") == ticker and p.get("side") == "buy"),
                                None,
                            )
                        strategy = (pending_match or {}).get("strategy", "G")
                        entry_time = (pending_match or {}).get("signal_time", current_ts)
                        self.position_entry[ticker] = {
                            "entry_price": actual_avg, "shares": actual_qty,
                            "cost": actual_qty * actual_avg,
                            "strategy": strategy, "entry_time": entry_time,
                        }
                        log.info("RECONCILE %s: rehydrated from Alpaca — strategy=%s "
                                 "qty=%.0f avg=$%.3f entry_time=%s",
                                 ticker, strategy, actual_qty, actual_avg, entry_time)
                    else:
                        self.position_entry[ticker]["shares"] = actual_qty
        except Exception:
            with self._pending_lock:
                has_pending_buy = any(
                    p.get("ticker") == ticker and p.get("side") == "buy"
                    for p in self.pending_orders.values()
                )
            if has_pending_buy:
                log.debug("RECONCILE %s: no Alpaca position yet but pending buy exists — holding slot",
                          ticker)
            else:
                log.warning("RECONCILE %s: no position at Alpaca → clearing", ticker)
                self.position_entry.pop(ticker, None)
                self.active_positions.discard(ticker)

    # ------------------------------------------------------------------ #
    #  v3 OVERLAY (R-O any-green after G hold expires)                  #
    #  Entry: first green candle after G's last hold on the same ticker  #
    #  Exit: target=57%, stop=30%, time=27min, trail=0.5% act=0%        #
    # ------------------------------------------------------------------ #

    def _v3_candidate_for_ticker(self, mh, day_open, g_holds_list):
        """Find first any-green candle after G hold expires.

        Args:
            mh: DataFrame of market-hour candles
            day_open: first candle's Open price
            g_holds_list: list of (entry_time, exit_time) for G trades on this ticker
        Returns:
            (entry_timestamp, entry_price, remaining_bars) or None
        """
        if mh is None or len(mh) < 2:
            return None
        bar0_red = float(mh.iloc[0]["Close"]) <= day_open
        scan = 1
        if not bar0_red and g_holds_list:
            ge = max(x for _, x in g_holds_list)
            ns = None
            for i in range(1, len(mh)):
                if mh.index[i] > ge:
                    ns = i
                    break
            if ns is None:
                return None
            scan = ns
        for i in range(scan, len(mh)):
            if float(mh.iloc[i]["Close"]) > day_open:
                ba = mh.iloc[i + 1:]
                if len(ba) == 0:
                    return None
                return (mh.index[i], float(mh.iloc[i]["Close"]), ba)
        return None

    def _manage_v3_exit(self, symbol, bar):
        """Check v3 active trade for exit conditions on each bar.

        Exits: target hit, stop hit, trailing stop, time limit.
        Passes correct v3 bracket params so the safety net has the right values.
        """
        ticker = symbol
        entry_info = self.position_entry.get(ticker, {})
        entry_price = entry_info.get("entry_price")
        if entry_price is None or entry_price <= 0:
            return

        entry_time = entry_info.get("entry_time")
        if entry_time is None:
            return

        # Convert entry_time to naive datetime for comparison
        if hasattr(entry_time, "to_pydatetime"):
            et_dt = entry_time.to_pydatetime().replace(tzinfo=None)
        elif isinstance(entry_time, str):
            et_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00")).replace(tzinfo=None)
        else:
            et_dt = entry_time if not hasattr(entry_time, "tzinfo") or entry_time.tzinfo is None \
                else entry_time.replace(tzinfo=None)

        current = float(bar["Close"])
        high = float(bar["High"])
        low = float(bar["Low"])
        bar_ts = bar["timestamp"]
        if hasattr(bar_ts, "to_pydatetime"):
            bt_dt = bar_ts.to_pydatetime().replace(tzinfo=None)
        elif isinstance(bar_ts, str):
            bt_dt = datetime.fromisoformat(bar_ts.replace("Z", "+00:00")).replace(tzinfo=None)
        else:
            bt_dt = bar_ts if not hasattr(bar_ts, "tzinfo") or bar_ts.tzinfo is None \
                else bar_ts.replace(tzinfo=None)

        tp = self.v3_params.get("target_pct", 57.0)
        sp = self.v3_params.get("stop_pct", 30.0)
        tl = self.v3_params.get("time_limit_min", 27)
        trp = self.v3_params.get("trail_pct", 0.5)
        tap = self.v3_params.get("trail_activate_pct", 0.0)

        target_price = entry_price * (1 + tp / 100)
        stop_price = entry_price * (1 - sp / 100)

        # Trail tracking
        v3t = self.v3_active_trade
        peak = entry_price
        trailing_stop = None
        if v3t:
            peak = v3t.get("peak", entry_price)
            trailing_stop = v3t.get("trailing_stop")

        # Update peak if current high exceeds it
        if high > peak:
            peak = high
            if (peak / entry_price - 1) * 100 >= tap:
                trailing_stop = peak * (1 - trp / 100)

        self.v3_active_trade = {"peak": peak, "trailing_stop": trailing_stop,
                                "entry_price": entry_price, "entry_time": entry_time}

        # Check exit conditions
        exit_reason = None
        if high >= target_price:
            exit_reason = "V3_TARGET"
            exit_price = target_price
        elif low <= stop_price:
            exit_reason = "V3_STOP"
            exit_price = stop_price
        elif trailing_stop is not None and low <= trailing_stop:
            exit_reason = "V3_TRAIL"
            exit_price = trailing_stop
        elif tl > 0:
            elapsed = (bt_dt - et_dt).total_seconds() / 60
            if elapsed >= tl:
                exit_reason = "V3_TIME"
                exit_price = current

        if exit_reason:
            log.info("V3 EXIT %s (%s): entry=$%.3f exit=$%.3f", ticker, exit_reason, entry_price, exit_price)
            order = self.executor.sell(
                ticker, reason=exit_reason,
                signal_price=exit_price,
                cumulative_dollar_volume=self._cum_dollar_vol(ticker),
                strategy="V3",
            )
            if order:
                with self._pending_lock:
                    self.pending_orders[str(order.id)] = {
                        "ticker": ticker, "strategy": "V3",
                        "signal_price": exit_price, "signal_time": bar["timestamp"],
                        "side": "sell", "exit_reason": exit_reason,
                        "pre_sell_shares": entry_info.get("shares", 0),
                        "is_partial": False,
                    }
                if self.fill_stream is not None:
                    self.fill_stream.register(order.id, self._on_sell_fill)
                    log.info("V3 SELL %s (%s): order %s placed, awaiting stream fill",
                             ticker, exit_reason, order.id)
                else:
                    # Legacy synchronous exit
                    self.active_positions.discard(ticker)
                    pnl = (exit_price - entry_price) * entry_info.get("shares", 0)
                    self.daily_pnl += pnl
                    trade = {
                        "ticker": ticker, "strategy": "V3",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "pnl": pnl, "reason": exit_reason,
                        "entry_time": entry_time, "exit_time": bar["timestamp"],
                    }
                    self.trades_today.append(trade)
                    _append_trade(trade)
            else:
                log.warning("V3 SELL REJECTED %s: order returned None", ticker)

    def _check_v3_entry(self, picks_with_data):
        """Check for v3 overlay entry: first any-green after G holds expire.

        Scans all picks, picks the first eligible ticker with a v3 candidate.
        Enforces aggregate 30% equity cap across ALL v3 positions combined.
        Skips tickers already in active_positions (same-ticker guard).
        Passes v3 bracket params so the safety net uses v3's own stop/target.
        """
        # Build G hold end times from the current day's simulated states
        g_holds = defaultdict(list)
        for ticker, st_list in self.all_states.items():
            for st in st_list:
                if st.get("strategy") == "G" and st.get("entry_time") and st.get("exit_time"):
                    g_holds[ticker].append((st["entry_time"], st["exit_time"]))

        # --- AGGREGATE CAP: all v3 positions combined ≤ 30% of equity ---
        try:
            acct = self.executor.get_account()
            equity = max(0, float(acct.equity))
        except Exception:
            equity = 0
        v3_budget = equity * 0.30  # 30% of total equity for ALL v3 entries
        remaining_budget = v3_budget - getattr(self, "_v3_allocated_today", 0)
        if remaining_budget <= 0:
            return  # aggregate cap reached

        # Use cash (floored at 0) for position sizing base
        raw_cash = self.executor.get_buying_power()
        cash = max(0, raw_cash)
        pos_pct = self.v3_params.get("position_pct", 30.0)
        if cash <= 0:
            log.debug("V3: no cash available (cash=$%.0f, equity=$%.0f)", raw_cash, equity)
            return

        for pick in picks_with_data:
            mh = pick.get("market_hour_candles")
            if mh is None or len(mh) < 2:
                continue
            ticker = pick["ticker"]
            day_open = float(mh.iloc[0]["Open"])
            ghol = g_holds.get(ticker, [])

            cand = self._v3_candidate_for_ticker(mh, day_open, ghol)
            if cand is None:
                continue

            ets, fp, ba = cand
            # Cap both by per-position PCT and remaining aggregate budget
            trade_size = min(cash * (pos_pct / 100), remaining_budget)
            if trade_size < 50:
                log.debug("V3 %s: trade_size $%.0f too small after cap", ticker, trade_size)
                continue
            cum_dvol = self._cum_dollar_vol(ticker)

            # Use v3 bracket params so the safety net stops at v3's own values
            v3_stop = self.v3_params.get("stop_pct", 30.0)
            v3_target = self.v3_params.get("target_pct", 57.0)

            order = self.executor.buy(
                ticker, trade_size, fp,
                cumulative_dollar_volume=cum_dvol,
                strategy="V3",
                bracket_stop_pct=v3_stop,
                bracket_target_pct=v3_target,
            )
            if order:
                # Track aggregate allocation and recompute remaining budget
                self._v3_allocated_today = getattr(self, "_v3_allocated_today", 0) + trade_size
                remaining_budget = v3_budget - self._v3_allocated_today
                self.active_positions.add(ticker)
                if ticker not in self.position_entry:
                    self.position_entry[ticker] = {
                        "entry_price": fp,
                        "shares": 0,
                        "cost": trade_size,
                        "strategy": "V3",
                        "entry_time": ets,
                    }
                with self._pending_lock:
                    self.pending_orders[str(order.id)] = {
                        "ticker": ticker, "strategy": "V3",
                        "signal_price": fp, "signal_time": ets,
                        "side": "buy", "requested_cost": trade_size,
                    }
                self.v3_active_trade = {
                    "peak": fp, "trailing_stop": None,
                    "entry_price": fp, "entry_time": ets,
                }
                if self.fill_stream is not None:
                    self.fill_stream.register(order.id, self._on_buy_fill)
                    log.info("V3 ENTRY %s: price=$%.3f size=$%.0f order=%s",
                             ticker, fp, trade_size, order.id)
                    self._schedule_safety_poll(order.id, ticker, "buy", fp, ets, "V3")
                else:
                    self._poll_buy_fill_inline(order, ticker, "V3", fp, ets)
                return  # one v3 entry per ticker per day
            else:
                log.debug("V3 ENTRY %s rejected by executor (vol cap or equity cap)", ticker)

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
            # Skip if already in this ticker (multi-position allowed across tickers)
            if symbol in self.active_positions:
                log.debug("HALT %s: signal suppressed — already in this ticker", symbol)
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
            self.active_positions.add(symbol)
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
            if symbol in self.active_positions:
                self.active_positions.discard(symbol)
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

    # Events that complete an order's lifecycle. Pop pending only on these.
    # Non-terminal events (new, pending_new, held, accepted, partial_fill) arrive
    # BEFORE 'fill' and would otherwise wipe out the pending entry.
    _BUY_TERMINAL_EVENTS = {"fill", "canceled", "rejected", "expired", "done_for_day"}

    # ----- Stream-based fill handling (2026-06-23, fixed 2026-06-24) -----
    def _on_buy_fill(self, event_type, order):
        """Called by FillStream when a buy order changes status.

        On 'fill': records position_entry with actual filled qty + avg price, pops pending.
        On 'partial_fill': records partial entry but keeps pending (more fills may come).
        On 'canceled'/'rejected'/'expired': clears active_position, pops pending.
        On other events ('new', 'pending_new', 'held', etc.): peeks pending without popping.

        FIX 2026-06-24: previously popped on EVERY event. First non-fill event
        ("pending_new" or "new") destroyed the pending entry before the actual
        "fill" arrived, causing the 100+ RECONCILE bug chain on CCXI today.
        """
        oid = str(getattr(order, "id", ""))
        is_terminal = event_type in self._BUY_TERMINAL_EVENTS
        with self._pending_lock:
            if is_terminal:
                pending = self.pending_orders.pop(oid, None)
            else:
                pending = self.pending_orders.get(oid, None)
        if pending is None:
            # Already resolved (popped by an earlier terminal event), or never registered.
            log.debug(f"BUY fill event {event_type} for unknown order {oid}")
            return
        ticker = pending["ticker"]
        strategy = pending["strategy"]

        if event_type in ("fill", "partial_fill"):
            try:
                actual_shares = float(order.filled_qty) if order.filled_qty else 0.0
                actual_avg = float(order.filled_avg_price) if order.filled_avg_price else pending["signal_price"]
            except (TypeError, ValueError):
                actual_shares = 0.0; actual_avg = pending["signal_price"]
            if actual_shares < 1:
                log.warning("BUY %s STREAM-FILL: 0 shares filled (event=%s)", ticker, event_type)
                # Don't clear active_positions yet on partial_fill — may complete later
                if event_type != "partial_fill":
                    if ticker in self.active_positions: self.active_positions.discard(ticker)
                return
            actual_cost = actual_shares * actual_avg
            self.position_entry[ticker] = {
                "entry_price": actual_avg,
                "shares": actual_shares,
                "cost": actual_cost,
                "strategy": strategy,
                "entry_time": pending["signal_time"],
            }
            log.info("ENTRY %s (%s) [STREAM]: %.2f shares @ $%.3f ($%s) — order=%s event=%s",
                     ticker, strategy, actual_shares, actual_avg,
                     format(actual_cost, ",.0f"), oid, event_type)
        elif event_type in ("canceled", "rejected", "expired", "done_for_day"):
            # No fill — clear from active_positions so engine can react to next signal
            log.warning("BUY %s [STREAM]: no fill (event=%s) — clearing active_positions", ticker, event_type)
            if ticker in self.active_positions and ticker not in self.position_entry:
                self.active_positions.discard(ticker)
        else:
            log.debug(f"BUY {ticker} STREAM event={event_type} order={oid}")

    def _on_sell_fill(self, event_type, order):
        """Called by FillStream when a sell order changes status.

        On 'fill': clears active_position (if full sell) or decrements shares
                   (if partial), records trade, updates daily_pnl.
        On 'partial_fill': decrements shares only, keeps position open + pending.
        On 'canceled'/'rejected'/'expired': position is STILL OPEN — don't
                   clear active_position (engine will retry exit next bar).
        Same FIX 2026-06-24 as _on_buy_fill: only pop on terminal events.
        """
        oid = str(getattr(order, "id", ""))
        is_terminal = event_type in self._BUY_TERMINAL_EVENTS  # same terminal set
        with self._pending_lock:
            if is_terminal:
                pending = self.pending_orders.pop(oid, None)
            else:
                pending = self.pending_orders.get(oid, None)
        if pending is None:
            log.debug(f"SELL fill event {event_type} for unknown order {oid}")
            return
        ticker = pending["ticker"]
        strategy = pending["strategy"]
        exit_reason = pending.get("exit_reason", "UNKNOWN")
        is_partial = pending.get("is_partial", False)

        if event_type in ("fill", "partial_fill"):
            try:
                actual_sold = float(order.filled_qty) if order.filled_qty else 0.0
                actual_avg = float(order.filled_avg_price) if order.filled_avg_price else pending["signal_price"]
            except (TypeError, ValueError):
                actual_sold = 0.0; actual_avg = pending["signal_price"]
            if actual_sold < 1:
                log.warning("SELL %s STREAM-FILL: 0 shares filled (event=%s)", ticker, event_type)
                return
            pinfo = self.position_entry.get(ticker, {})
            pre_shares = float(pinfo.get("shares", 0))
            remaining = max(0.0, pre_shares - actual_sold)
            pinfo["shares"] = remaining
            self.position_entry[ticker] = pinfo

            if is_partial and event_type == "partial_fill":
                # Still some shares pending in this partial sell — don't clear active
                log.info("PARTIAL SELL %s [STREAM]: %.0f shares @ $%.3f (partial event, still pending)",
                         ticker, actual_sold, actual_avg)
                return

            # Position fully closed by this fill (full sell, or partial sell that completed all of its target qty)
            if remaining < 1 and not is_partial:
                # Full exit
                entry_price = float(pinfo.get("entry_price", 0))
                # Compute realized PnL from actual fills (true vs expected)
                pnl = (actual_avg - entry_price) * actual_sold
                self.active_positions.discard(ticker)
                self.daily_pnl += pnl
                trade = {
                    "ticker": ticker, "strategy": strategy,
                    "entry_price": entry_price,
                    "exit_price": actual_avg,
                    "pnl": pnl, "reason": exit_reason,
                    "entry_time": pinfo.get("entry_time"),
                    "exit_time": pending["signal_time"],
                }
                self.trades_today.append(trade)
                _append_trade(trade)
                log.info("EXIT %s (%s) [STREAM]: PnL=$%s | $%.2f -> $%.3f  (sold %.0f shares)",
                         ticker, exit_reason, format(pnl, "+,.2f"),
                         entry_price, actual_avg, actual_sold)
            else:
                # Partial sell completed — log but stay in position
                log.info("PARTIAL SELL %s [STREAM]: %.0f shares @ $%.3f sold, %.0f remaining",
                         ticker, actual_sold, actual_avg, remaining)
        elif event_type in ("canceled", "rejected", "expired"):
            log.warning("SELL %s [STREAM]: NO FILL (event=%s) — position STILL OPEN, engine will retry exit",
                        ticker, event_type)
        else:
            log.debug(f"SELL {ticker} STREAM event={event_type} order={oid}")

    def _schedule_sell_safety_poll(self, order_id, ticker):
        """Safety net for sell orders — same logic as buy, polls after 60s."""
        def _poll():
            time.sleep(60.0)
            with self._pending_lock:
                still_pending = str(order_id) in self.pending_orders
            if not still_pending: return
            log.warning("SAFETY-POLL SELL %s order %s: stream silent 60s, polling Alpaca", ticker, order_id)
            try:
                o = self.executor.client.get_order_by_id(order_id)
                status = o.status.value if hasattr(o.status, "value") else str(o.status)
                status = status.lower().replace("orderstatus.", "")
                if status == "filled" or (o.filled_qty and float(o.filled_qty) > 0):
                    self._on_sell_fill("fill", o)
                elif status in ("canceled", "rejected", "expired"):
                    self._on_sell_fill(status, o)
                else:
                    log.warning("SAFETY-POLL SELL %s order %s: still %s after 60s — abandoning callback",
                                ticker, order_id, status)
                    with self._pending_lock:
                        self.pending_orders.pop(str(order_id), None)
            except Exception as e:
                log.error(f"SAFETY-POLL SELL {ticker} order {order_id} failed: {e}")
        threading.Thread(target=_poll, daemon=True, name=f"sell-safety-{str(order_id)[:8]}").start()

    def _schedule_safety_poll(self, order_id, ticker, side, signal_price, signal_time, strategy):
        """Background safety net: if TradingStream misses an event (stream disconnect,
        callback bug, etc.), poll Alpaca after 60s to force-resolve the pending order.
        """
        def _poll():
            time.sleep(60.0)
            with self._pending_lock:
                still_pending = str(order_id) in self.pending_orders
            if not still_pending:
                return  # stream already handled it
            log.warning("SAFETY-POLL %s order %s: stream didn't notify in 60s, polling Alpaca", ticker, order_id)
            try:
                o = self.executor.client.get_order_by_id(order_id)
                status = o.status.value if hasattr(o.status, "value") else str(o.status)
                status = status.lower().replace("orderstatus.", "")
                if status == "filled" or (o.filled_qty and float(o.filled_qty) > 0):
                    # Manually invoke the same handler as stream would
                    self._on_buy_fill("fill", o)
                elif status in ("canceled", "rejected", "expired"):
                    self._on_buy_fill(status, o)
                else:
                    log.warning("SAFETY-POLL %s order %s: status=%s after 60s, still pending — abandoning",
                                ticker, order_id, status)
                    with self._pending_lock:
                        self.pending_orders.pop(str(order_id), None)
                    if ticker in self.active_positions and ticker not in self.position_entry:
                        self.active_positions.discard(ticker)
            except Exception as e:
                log.error(f"SAFETY-POLL {ticker} order {order_id} failed: {e}")
        threading.Thread(target=_poll, daemon=True, name=f"safety-poll-{order_id[:8]}").start()

    def _poll_buy_fill_inline(self, order, ticker, strategy, entry_price, ts):
        """LEGACY synchronous polling — used only when fill_stream is None.
        Kept for backward compat / unit tests."""
        actual_shares = 0.0
        actual_avg = entry_price
        terminal_states = {"filled", "canceled", "rejected", "expired"}
        deadline = time.time() + 30.0  # extended from 15s
        last_status = None
        while time.time() < deadline:
            try:
                o = self.executor.client.get_order_by_id(order.id)
                fst = o.status.value if hasattr(o.status, 'value') else str(o.status)
                last_status = fst
                if o.filled_qty: actual_shares = float(o.filled_qty)
                if o.filled_avg_price: actual_avg = float(o.filled_avg_price)
                if fst in terminal_states: break
            except Exception:
                pass
            time.sleep(0.3)
        else:
            log.warning("BUY %s: order %s still %s after 30s polling — reconciliation",
                        ticker, order.id, last_status)
        if actual_shares < 1:
            log.warning("BUY %s LEGACY-POLL: 0 shares filled in 30s — not recording position", ticker)
            if ticker in self.active_positions: self.active_positions.discard(ticker)
        else:
            actual_cost = actual_shares * actual_avg
            self.position_entry[ticker] = {
                "entry_price": actual_avg, "shares": actual_shares, "cost": actual_cost,
                "strategy": strategy, "entry_time": ts,
            }
            log.info("ENTRY %s (%s) [LEGACY-POLL]: %.2f shares @ $%.3f ($%s)",
                     ticker, strategy, actual_shares, actual_avg, format(actual_cost, ",.0f"))

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
        for ticker in list(self.active_positions):
            self.executor.sell(ticker, reason="EOD_CLOSE")
        self.active_positions.clear()
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
