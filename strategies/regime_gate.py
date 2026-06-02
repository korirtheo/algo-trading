"""
Regime gate: multi-signal classification of each trading day's character,
computable at 9:25 AM ET from data the engine already has.

Six signals are evaluated (mirroring the small-cap momentum-trader framework):

  S1  Gap-up density        — count of universe gappers >= 20% pre-market
  S2  Squeeze count         — count of universe gappers >= 50% pre-market
  S3  Top-end magnitude     — max gap %, captures single-name extremes
  S4  Universe volume       — median pre-market volume across picks
  S5  Yesterday's follow-thru — % of yesterday's picks that closed UP yesterday
                                (proxy for recent regime quality)
  S6  Recent setup WR        — rolling 5-day fired-signal WR, if available

Three regimes emerge:
  DEAD     - low-density, low-vol, fading market: skip trading entirely
  SQUEEZE  - extreme density / single huge mover: enable L overlay, hold longer
  NORMAL   - default: deploy the baseline config (e.g. trial #6)

The classifier returns the regime label PLUS the raw features dict so callers
can log everything for later inspection.
"""
from __future__ import annotations

from typing import Iterable, Optional

# --- Thresholds (tuned to 2021-2026 universe-density stats; configurable) ---
DEAD_N20_MAX = 2          # under this many >=20% gappers AND ...
DEAD_MAX_GAP_MAX = 30.0   # ... no single name >= this gap → DEAD
DEAD_MED_PM_VOL_MAX = 500_000  # ... and weak overall PM volume

SQUEEZE_N50_MIN = 4       # this many >= 50% gappers triggers SQUEEZE
                          # (tuned 2026-06-01: 4 beat 3 on Sharpe by +9%)
SQUEEZE_MAX_GAP_MIN = 150.0  # any single >=150% triggers SQUEEZE
                              # (tuned 2026-06-01: 150 beat 100 on Sharpe)

FOLLOWTHRU_BAD_MAX = 30.0    # <30% of yesterday's picks held their gain → soft DEAD
FOLLOWTHRU_GOOD_MIN = 60.0   # >60% held → confirms NORMAL/SQUEEZE


def _gap_pct(p) -> float:
    if isinstance(p, dict):
        return float(p.get("gap_pct", 0.0))
    return float(getattr(p, "gap_pct", 0.0))


def _pm_volume(p) -> float:
    if isinstance(p, dict):
        return float(p.get("pm_volume", p.get("volume", 0.0)))
    return float(getattr(p, "pm_volume", getattr(p, "volume", 0.0)))


def compute_features(
    picks: Iterable,
    *,
    yesterday_followthru_pct: Optional[float] = None,
    recent_setup_wr: Optional[float] = None,
) -> dict:
    """Compute every signal from today's pick list + optional context.

    Args:
        picks: iterable of pick dicts (or rows) with `gap_pct` and `pm_volume`
        yesterday_followthru_pct: % of yesterday's picks that closed up
                                  (caller computes from daily bars)
        recent_setup_wr: rolling 5-day fired-signal WR (caller computes)
    """
    picks_list = list(picks)
    if not picks_list:
        return {
            "n_picks": 0, "n_above_20": 0, "n_above_50": 0, "max_gap": 0.0,
            "med_pm_volume": 0.0,
            "yesterday_followthru_pct": yesterday_followthru_pct,
            "recent_setup_wr": recent_setup_wr,
        }

    gaps = [_gap_pct(p) for p in picks_list]
    vols = [_pm_volume(p) for p in picks_list]
    vols_sorted = sorted(vols)
    med_vol = vols_sorted[len(vols_sorted) // 2] if vols_sorted else 0.0

    return {
        "n_picks": len(picks_list),
        "n_above_20": sum(1 for g in gaps if g >= 20.0),
        "n_above_50": sum(1 for g in gaps if g >= 50.0),
        "max_gap": max(gaps),
        "med_pm_volume": med_vol,
        "yesterday_followthru_pct": yesterday_followthru_pct,
        "recent_setup_wr": recent_setup_wr,
    }


def classify_regime(
    picks: Iterable,
    *,
    yesterday_followthru_pct: Optional[float] = None,
    recent_setup_wr: Optional[float] = None,
    rolling_5d_features: Optional[dict] = None,
    w_today: float = 1.0,
) -> str:
    """Multi-signal regime classifier with optional rolling-history smoothing.

    Args:
        picks: today's pre-market scan picks
        rolling_5d_features: dict with keys n_above_20, n_above_50, max_gap,
                              med_pm_volume — rolling 5-day means up to yesterday.
                              When provided, today's features are blended with
                              the history per `w_today`.
        w_today: weight on today's features vs the 5-day rolling average.
                 1.0 = ignore history (current behavior).
                 0.0 = classify entirely on history.
                 0.5 = equal blend.

    Returns: "dead" | "normal" | "squeeze"
    """
    f_today = compute_features(picks,
                                yesterday_followthru_pct=yesterday_followthru_pct,
                                recent_setup_wr=recent_setup_wr)

    if f_today["n_picks"] == 0:
        return "dead"

    # Blend today vs rolling-history if we have both
    if rolling_5d_features is not None and 0.0 <= w_today < 1.0:
        f = {}
        for k in ("n_above_20", "n_above_50", "max_gap", "med_pm_volume"):
            today_v = f_today[k]
            hist_v = rolling_5d_features.get(k, today_v)
            f[k] = w_today * today_v + (1.0 - w_today) * hist_v
    else:
        f = f_today

    # SQUEEZE if any extreme signal fires
    if f["n_above_50"] >= SQUEEZE_N50_MIN or f["max_gap"] > SQUEEZE_MAX_GAP_MIN:
        return "squeeze"

    # DEAD if every universe-activity signal is weak
    cond_low_density = f["n_above_20"] < DEAD_N20_MAX and f["max_gap"] < DEAD_MAX_GAP_MAX
    cond_low_volume = f["med_pm_volume"] < DEAD_MED_PM_VOL_MAX
    cond_bad_followthru = (yesterday_followthru_pct is not None
                            and yesterday_followthru_pct < FOLLOWTHRU_BAD_MAX)
    cond_bad_recent_wr = (recent_setup_wr is not None and recent_setup_wr < 40.0)

    if cond_low_density and (cond_low_volume or cond_bad_followthru or cond_bad_recent_wr):
        return "dead"

    return "normal"


def regime_features(picks: Iterable, **kwargs) -> dict:
    """Backwards-compatible wrapper — returns features + regime label."""
    f = compute_features(picks, **kwargs)
    f["regime"] = classify_regime(picks, **kwargs)
    return f


# ---------------------------------------------------------------------------
# Helpers for the optional signals — keep them here so callers can use them
# without re-implementing the math.
# ---------------------------------------------------------------------------

def compute_yesterday_followthru(yesterday_picks: Iterable,
                                  yesterday_close_prices: dict) -> Optional[float]:
    """Given yesterday's picks and their close prices, what % closed UP?

    Args:
        yesterday_picks: list of pick dicts from yesterday's scan
        yesterday_close_prices: {ticker: yesterday_close_price}

    Returns:
        Percentage 0-100, or None if we can't compute (no data)
    """
    yesterday_picks = list(yesterday_picks)
    if not yesterday_picks:
        return None
    n_held = 0
    n_total = 0
    for p in yesterday_picks:
        ticker = p.get("ticker") if isinstance(p, dict) else getattr(p, "ticker", None)
        if not ticker:
            continue
        market_open = float(p.get("market_open", 0)) if isinstance(p, dict) \
            else float(getattr(p, "market_open", 0))
        close = yesterday_close_prices.get(ticker)
        if not market_open or close is None:
            continue
        n_total += 1
        if close > market_open:
            n_held += 1
    if n_total == 0:
        return None
    return 100.0 * n_held / n_total
