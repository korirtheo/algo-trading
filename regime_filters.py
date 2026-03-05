"""
Regime Filters — Market-aware trading filters to reduce drawdowns.

Filters:
  1. VIX Filter: Skip trading when VIX > threshold (high fear)
  2. SPY Trend Filter: Only trade when SPY is above its 20-day SMA
  3. Adaptive Sizing: Cut position size 50% if recent win rate < 45%

Downloads SPY and VIX data via yfinance, caches to CSV.

Usage:
  from regime_filters import RegimeFilter
  rf = RegimeFilter(vix_threshold=30, spy_ma_period=20)
  rf.load_data("2025-08-01", "2026-02-28")
  should_trade, size_mult = rf.check(date_str, recent_daily_pnls)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

CACHE_DIR = "regime_data"


def _download_and_cache(ticker, start, end, filename):
    """Download daily data from yfinance and cache to CSV."""
    cache_path = os.path.join(CACHE_DIR, filename)

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Check if we have data covering the requested range
        if len(df) > 0:
            first = df.index.min().strftime("%Y-%m-%d")
            last = df.index.max().strftime("%Y-%m-%d")
            if first <= start and last >= end:
                return df

    import yfinance as yf
    # Download with extra buffer for MA calculation
    buffer_start = (pd.Timestamp(start) - timedelta(days=60)).strftime("%Y-%m-%d")
    print(f"  Downloading {ticker} data ({buffer_start} to {end})...")
    data = yf.download(ticker, start=buffer_start, end=end, progress=False)

    if data.empty:
        print(f"  WARNING: No data for {ticker}")
        return pd.DataFrame()

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    os.makedirs(CACHE_DIR, exist_ok=True)
    data.to_csv(cache_path)
    print(f"  Cached {len(data)} rows to {cache_path}")
    return data


class RegimeFilter:
    """Market regime filter for backtesting."""

    def __init__(self, vix_threshold=30.0, spy_ma_period=20,
                 adaptive_lookback=5, adaptive_wr_threshold=0.45,
                 adaptive_size_mult=0.50,
                 enable_vix=True, enable_spy_trend=True, enable_adaptive=True):
        self.vix_threshold = vix_threshold
        self.spy_ma_period = spy_ma_period
        self.adaptive_lookback = adaptive_lookback
        self.adaptive_wr_threshold = adaptive_wr_threshold
        self.adaptive_size_mult = adaptive_size_mult
        self.enable_vix = enable_vix
        self.enable_spy_trend = enable_spy_trend
        self.enable_adaptive = enable_adaptive

        self.vix_data = None
        self.spy_data = None
        self.spy_sma = None
        self._loaded = False

    def load_data(self, start_date, end_date):
        """Download/load VIX and SPY data."""
        print("Loading regime filter data...")

        if self.enable_vix:
            self.vix_data = _download_and_cache("^VIX", start_date, end_date, "vix_daily.csv")
            if not self.vix_data.empty:
                print(f"  VIX: {len(self.vix_data)} days, range: "
                      f"{self.vix_data['Close'].min():.1f} - {self.vix_data['Close'].max():.1f}")

        if self.enable_spy_trend:
            self.spy_data = _download_and_cache("SPY", start_date, end_date, "spy_daily.csv")
            if not self.spy_data.empty:
                self.spy_sma = self.spy_data["Close"].rolling(self.spy_ma_period).mean()
                print(f"  SPY: {len(self.spy_data)} days, "
                      f"SMA({self.spy_ma_period}) computed")

        self._loaded = True

    def _get_vix_close(self, date_str):
        """Get VIX close for a date (or most recent prior)."""
        if self.vix_data is None or self.vix_data.empty:
            return None
        target = pd.Timestamp(date_str)
        # Find most recent date <= target
        valid = self.vix_data.index[self.vix_data.index.tz_localize(None) <= target
                                     if self.vix_data.index.tz is not None
                                     else self.vix_data.index <= target]
        if len(valid) == 0:
            return None
        return float(self.vix_data.loc[valid[-1], "Close"])

    def _get_spy_above_sma(self, date_str):
        """Check if SPY is above its SMA on the given date."""
        if self.spy_data is None or self.spy_data.empty or self.spy_sma is None:
            return None
        target = pd.Timestamp(date_str)
        valid_idx = self.spy_data.index[
            self.spy_data.index.tz_localize(None) <= target
            if self.spy_data.index.tz is not None
            else self.spy_data.index <= target
        ]
        if len(valid_idx) == 0:
            return None
        last_date = valid_idx[-1]
        spy_close = float(self.spy_data.loc[last_date, "Close"])
        sma_val = self.spy_sma.loc[last_date]
        if pd.isna(sma_val):
            return None
        return spy_close > float(sma_val)

    def check(self, date_str, recent_daily_pnls=None):
        """
        Check regime filters for a given date.

        Args:
            date_str: "YYYY-MM-DD"
            recent_daily_pnls: list of recent daily P&L values (for adaptive sizing)

        Returns:
            (should_trade: bool, size_multiplier: float, reasons: dict)
        """
        should_trade = True
        size_mult = 1.0
        reasons = {}

        # 1. VIX filter
        if self.enable_vix:
            vix = self._get_vix_close(date_str)
            if vix is not None:
                reasons["vix"] = vix
                if vix > self.vix_threshold:
                    should_trade = False
                    reasons["vix_blocked"] = True

        # 2. SPY trend filter
        if self.enable_spy_trend:
            above_sma = self._get_spy_above_sma(date_str)
            if above_sma is not None:
                reasons["spy_above_sma"] = above_sma
                if not above_sma:
                    should_trade = False
                    reasons["spy_blocked"] = True

        # 3. Adaptive sizing (doesn't block trading, just reduces size)
        if self.enable_adaptive and recent_daily_pnls is not None:
            if len(recent_daily_pnls) >= self.adaptive_lookback:
                last_n = recent_daily_pnls[-self.adaptive_lookback:]
                wins = sum(1 for p in last_n if p > 0)
                wr = wins / len(last_n)
                reasons["recent_wr"] = wr
                if wr < self.adaptive_wr_threshold:
                    size_mult = self.adaptive_size_mult
                    reasons["size_reduced"] = True

        return should_trade, size_mult, reasons


def run_regime_backtest(daily_picks, params, regime_filter,
                        simulate_fn=None, slippage_pct=None):
    """
    Run backtest with regime filters applied.

    Args:
        daily_picks: dict of date -> picks
        params: strategy parameters dict
        regime_filter: RegimeFilter instance (already loaded)
        simulate_fn: simulation function (default: optimize.simulate_day_fast)
        slippage_pct: override slippage if needed

    Returns:
        list of (date, day_pnl, regime_info) tuples
    """
    if simulate_fn is None:
        from optimize import simulate_day_fast
        simulate_fn = simulate_day_fast

    import optimize
    original_slippage = optimize.SLIPPAGE_PCT
    if slippage_pct is not None:
        optimize.SLIPPAGE_PCT = slippage_pct

    results = []
    recent_pnls = []

    for date_str in sorted(daily_picks.keys()):
        picks = daily_picks[date_str]

        # Check regime
        should_trade, size_mult, regime_info = regime_filter.check(
            date_str, recent_pnls
        )

        if not should_trade:
            # Skip this day entirely
            results.append((date_str, 0.0, {**regime_info, "skipped": True}))
            recent_pnls.append(0.0)
            continue

        # Apply size multiplier
        adjusted_params = dict(params)
        if size_mult < 1.0:
            adjusted_params["trade_pct"] = params["trade_pct"] * size_mult

        day_pnl_result = simulate_fn(picks, adjusted_params)
        # Handle both old (scalar) and new (tuple) return formats
        day_pnl = day_pnl_result[0] if isinstance(day_pnl_result, tuple) else day_pnl_result
        results.append((date_str, day_pnl, {**regime_info, "skipped": False,
                                              "size_mult": size_mult}))
        recent_pnls.append(day_pnl)

    if slippage_pct is not None:
        optimize.SLIPPAGE_PCT = original_slippage

    return results


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick test: show regime state for each trading day."""
    from optimize import load_all_picks

    DATA_DIRS = ["stored_data", "stored_data_oos"]
    daily_picks = load_all_picks(DATA_DIRS)
    dates = sorted(daily_picks.keys())
    print(f"Trading days: {dates[0]} to {dates[-1]} ({len(dates)} days)\n")

    rf = RegimeFilter(vix_threshold=30.0, spy_ma_period=20)
    rf.load_data(dates[0], dates[-1])

    print(f"\n{'Date':<14} {'VIX':>6} {'SPY>SMA':>8} {'Trade?':>7}")
    print("-" * 40)

    skipped = 0
    for d in dates:
        should_trade, size_mult, info = rf.check(d)
        vix = info.get("vix", "N/A")
        spy = "Yes" if info.get("spy_above_sma") else "No" if info.get("spy_above_sma") is not None else "N/A"
        trade = "YES" if should_trade else "SKIP"
        vix_str = f"{vix:.1f}" if isinstance(vix, float) else vix
        if not should_trade:
            skipped += 1
        print(f"  {d:<14} {vix_str:>6} {spy:>8} {trade:>7}")

    print(f"\nDays skipped: {skipped}/{len(dates)} ({skipped/len(dates)*100:.1f}%)")
