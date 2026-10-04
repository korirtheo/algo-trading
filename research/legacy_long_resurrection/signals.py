"""Recovered signal definitions for the first legacy-long resurrection batch.

These functions preserve the *idea* of the old strategies while returning a
close-confirmed Signal only. Execution is handled separately by causal_execution.
No legacy PnL, sizing, or Optuna result is imported.
"""
import numpy as np
from .causal_execution import Signal


def cumulative_vwap(bars):
    h=bars["High"].to_numpy(float); l=bars["Low"].to_numpy(float)
    c=bars["Close"].to_numpy(float); v=bars["Volume"].to_numpy(float)
    cv=np.cumsum(v); cv=np.where(cv == 0, np.nan, cv)
    return np.cumsum(((h+l+c)/3.0)*v)/cv


def vwap_reclaim(bars, *, min_below_candles=2, min_below_pct=0.0,
                 vol_spike_ratio=1.0, max_entry_candle=120):
    vw=cumulative_vwap(bars); c=bars["Close"].to_numpy(float); v=bars["Volume"].to_numpy(float)
    below=0; depth=0.0
    for i in range(2, min(len(bars), max_entry_candle)):
        if np.isfinite(vw[i]) and c[i] < vw[i]:
            below += 1; depth=max(depth, (vw[i]-c[i])/vw[i]*100.0); continue
        if below >= min_below_candles and depth >= min_below_pct and np.isfinite(vw[i]):
            avg=float(np.mean(v[max(0,i-10):i])) if i else 0.0
            if avg > 0 and v[i]/avg >= vol_spike_ratio:
                return Signal("V", i, bars.index[i], c[i], "VWAP_RECLAIM")
        below=0; depth=0.0
    return None


def opening_dip_buy(bars, *, spike_window=5, min_spike_pct=3.0, dip_pct=5.0,
                    entry_mode="vwap", max_entry_candle=60):
    if len(bars) <= spike_window: return None
    h=bars["High"].to_numpy(float); l=bars["Low"].to_numpy(float)
    c=bars["Close"].to_numpy(float); o=bars["Open"].to_numpy(float)
    vw=cumulative_vwap(bars)
    spike_high=float(np.max(h[:spike_window]))
    if (spike_high/o[0]-1)*100 < min_spike_pct: return None
    dip_level=spike_high*(1-dip_pct/100); pulled=False; below_vwap=False
    for i in range(spike_window, min(len(bars), max_entry_candle)):
        if not pulled:
            if l[i] <= dip_level: pulled=True
            continue
        if np.isfinite(vw[i]) and c[i] < vw[i]: below_vwap=True
        hit=(entry_mode=="vwap" and below_vwap and c[i] > vw[i])
        if entry_mode!="vwap" and i>=5: hit=c[i] > np.max(c[i-5:i])
        if hit: return Signal("D", i, bars.index[i], c[i], "DIP_RECLAIM")
    return None


def pm_high_pullback(bars, pm_high, *, confirm_above=1, confirm_window=2,
                     pullback_pct=1.0, pullback_timeout=5, max_entry_candle=120):
    c=bars["Close"].to_numpy(float); l=bars["Low"].to_numpy(float)
    recent=[]; breakout=None
    for i in range(min(len(bars), max_entry_candle)):
        recent.append(bool(c[i] > pm_high)); recent=recent[-confirm_window:]
        if sum(recent) >= confirm_above: breakout=i; break
    if breakout is None: return None
    zone=pm_high*(1+pullback_pct/100); pulled=False; waited=0
    for i in range(breakout+1, min(len(bars), max_entry_candle)):
        waited += 1
        if not pulled and l[i] <= zone: pulled=True
        if pulled and c[i] > pm_high:
            return Signal("P", i, bars.index[i], c[i], "PMH_PULLBACK_BOUNCE")
        if not pulled and waited >= pullback_timeout:
            if c[i] > pm_high: return Signal("P", i, bars.index[i], c[i], "PMH_TIMEOUT_HOLD")
            return None
    return None


def red_to_green(bars, *, min_dip_pct=1.0, max_dip_pct=15.0,
                 earliest_candle=3, latest_candle=60, vol_surge_mult=1.5,
                 min_body_pct=0.5, require_above_vwap=False):
    if len(bars)<2: return None
    o=bars["Open"].to_numpy(float); h=bars["High"].to_numpy(float)
    l=bars["Low"].to_numpy(float); c=bars["Close"].to_numpy(float); v=bars["Volume"].to_numpy(float)
    vw=cumulative_vwap(bars); day_open=o[0]; deepest=0.0; went_red=False
    for i in range(min(len(bars), latest_candle)):
        if l[i] < day_open:
            went_red=True; deepest=max(deepest,(day_open-l[i])/day_open*100)
        count=i+1
        if count < earliest_candle: continue
        if not went_red or deepest < min_dip_pct: continue
        if deepest > max_dip_pct: return None
        if c[i] <= day_open: continue
        if o[i] <= 0 or (c[i]/o[i]-1)*100 < min_body_pct: continue
        prior=v[max(0,i-5):i]; avg=float(np.mean(prior)) if len(prior) else 0.0
        if avg > 0 and v[i] < avg*vol_surge_mult: continue
        if require_above_vwap and (not np.isfinite(vw[i]) or c[i] < vw[i]): continue
        return Signal("B", i, bars.index[i], c[i], "RED_TO_GREEN")
    return None
