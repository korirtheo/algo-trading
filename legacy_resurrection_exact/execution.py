"""Canonical 2026-10-04 execution patch for the legacy-long resurrection study.

The first exploratory harness omitted the volatility adjustment applied to the
effective 2/6/10-minute liquidity denominator.  This module installs the exact
current standard without modifying the exploratory artifact, preserving audit
history.

V_eff = max(DV2, .5*DV6, .25*DV10)
V_eff_adj = V_eff / (1 + min(recent_10m_range_pct / 5, 1))
The 15% execution cap and square-root impact both use V_eff_adj.
"""
import pandas as pd

VOL_FACTOR_SCALE=5.0

def install(rb):
    def liquidity_at(bars, i, price):
        upto=bars.iloc[:i+1]
        cumulative=float((upto["Close"].astype(float)*upto["Volume"].astype(float)).sum())
        dv2=rb._window_dvol(bars,i,2,price)
        dv6=rb._window_dvol(bars,i,6,price)
        dv10=rb._window_dvol(bars,i,10,price)
        eff=max(dv2,0.5*dv6,0.25*dv10)

        ts=bars.index[i]
        start=ts-pd.Timedelta(minutes=10)
        recent=bars.loc[(bars.index>start)&(bars.index<=ts)]
        if eff>0 and not recent.empty and price>0:
            hi=float(recent["High"].astype(float).max())
            lo=float(recent["Low"].astype(float).min())
            if hi>0 and lo>0:
                range_pct=(hi-lo)/float(price)*100.0
                vol_factor=1.0+min(range_pct/VOL_FACTOR_SCALE,1.0)
                eff/=vol_factor
        return cumulative,dv10,eff

    rb.liquidity_at=liquidity_at
    return rb
