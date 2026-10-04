"""Execution-aware baseline evaluator for recovered legacy long strategies.

Purpose: establish a CLEAN, UNOPTIMIZED baseline.  Legacy performance numbers are
never imported.  Signals are close-confirmed and market entries execute at the
next available bar open.

Execution standard (2026-10-04):
  notional <= min(30% current equity, $100k,
                  5% cumulative dvol, 8% trailing-10m dvol,
                  15% effective 2/6/10m liquidity)
  effective liquidity = max(DV2, 0.5*DV6, 0.25*DV10)
  adverse impact % = 0.05 + 0.5/max(price,0.1)
                     + 3*sqrt(notional/effective_liquidity)
No leverage; concurrent positions are allowed subject to free cash.
"""
from __future__ import annotations
import argparse, heapq, json, math, os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import test_full
from research.legacy_long_resurrection.causal_execution import next_bar_open_fill, assert_causal_fill
from research.legacy_long_resurrection.signals import (
    opening_dip_buy, vwap_reclaim, pm_high_pullback, red_to_green,
)

START_EQUITY=25_000.0
EQUITY_FRACTION=0.30
HARD_CAP=100_000.0
CUM_DVOL_CAP=0.05
DV10_CAP=0.08
EFF_LIQ_CAP=0.15

DEFAULTS={
 "D":{"min_gap":15.0,"signal":dict(spike_window=10,min_spike_pct=5.0,dip_pct=8.0,entry_mode="vwap",max_entry_candle=30),
      "exit":dict(target=6.0,stop=5.0,trail=3.0,trail_activate=3.0,minutes=30)},
 "V":{"min_gap":15.0,"signal":dict(min_below_candles=5,min_below_pct=1.0,vol_spike_ratio=1.5,max_entry_candle=60),
      "exit":dict(target=6.0,stop=5.0,trail=3.0,trail_activate=3.0,minutes=30)},
 "P":{"min_gap":15.0,"signal":dict(confirm_above=2,confirm_window=4,pullback_pct=4.0,pullback_timeout=24,max_entry_candle=60),
      "exit":dict(target=8.0,stop=5.0,trail=None,trail_activate=None,minutes=30)},
 "B":{"min_gap":10.0,"signal":dict(min_dip_pct=1.0,max_dip_pct=15.0,earliest_candle=3,latest_candle=60,
      vol_surge_mult=1.5,min_body_pct=0.5,require_above_vwap=False),
      "exit":dict(target=8.0,stop=5.0,trail=3.0,trail_activate=4.0,minutes=60)},
}


def _window_dvol(bars, i, minutes, price_hint=None):
    ts=bars.index[i]; start=ts-pd.Timedelta(minutes=minutes)
    x=bars.loc[(bars.index>start)&(bars.index<=ts)]
    if x.empty: return 0.0
    px=(x["Close"].astype(float)*x["Volume"].astype(float)).sum()
    return float(px)


def liquidity_at(bars, i, price):
    upto=bars.iloc[:i+1]
    cumulative=float((upto["Close"].astype(float)*upto["Volume"].astype(float)).sum())
    dv2=_window_dvol(bars,i,2,price); dv6=_window_dvol(bars,i,6,price); dv10=_window_dvol(bars,i,10,price)
    eff=max(dv2,0.5*dv6,0.25*dv10)
    return cumulative,dv10,eff


def max_notional(equity, free_cash, bars, i, price):
    cum,dv10,eff=liquidity_at(bars,i,price)
    cap=min(EQUITY_FRACTION*equity,HARD_CAP,free_cash,
            CUM_DVOL_CAP*cum,DV10_CAP*dv10,EFF_LIQ_CAP*eff)
    return max(0.0,float(cap)),eff


def impact_pct(price,notional,eff):
    if notional<=0 or eff<=0: return float("inf")
    return 0.05+0.5/max(float(price),0.1)+3.0*math.sqrt(notional/eff)


def adverse_price(raw,side,notional,eff):
    pct=impact_pct(raw,notional,eff)/100.0
    return raw*(1+pct if side=="buy" else 1-pct),pct*100


@dataclass
class Candidate:
    strategy:str; ticker:str; date:str; gap_pct:float; pm_high:float
    bars:pd.DataFrame; signal:object; fill_i:int; fill_ts:object; raw_fill:float


@dataclass
class Trade:
    strategy:str; ticker:str; date:str; gap_pct:float
    signal_time:object; fill_time:object; exit_time:object
    signal_price:float; raw_fill:float; entry:float; exit:float
    notional:float; pnl:float; return_pct:float; exit_reason:str
    entry_slip_pct:float; exit_slip_pct:float


def build_candidates(all_dates,daily_picks):
    out=[]
    for d in all_dates:
        for p in daily_picks.get(d,[]):
            b=p["market_hour_candles"].sort_index()
            if len(b)<3: continue
            for s,cfg in DEFAULTS.items():
                if float(p["gap_pct"])<cfg["min_gap"]: continue
                if s=="D": sig=opening_dip_buy(b,**cfg["signal"])
                elif s=="V": sig=vwap_reclaim(b,**cfg["signal"])
                elif s=="P": sig=pm_high_pullback(b,float(p["premarket_high"]),**cfg["signal"])
                else: sig=red_to_green(b,**cfg["signal"])
                if sig is None: continue
                f=next_bar_open_fill(sig,b,0.0)
                if f is None: continue
                assert_causal_fill(f)
                out.append(Candidate(s,p["ticker"],d,float(p["gap_pct"]),float(p["premarket_high"]),
                                     b,sig,f.fill_bar_index,f.fill_timestamp,f.raw_fill_price))
    out.sort(key=lambda x:(x.fill_ts,x.strategy,x.ticker))
    return out


def simulate_exit(c:Candidate,entry,notional):
    cfg=DEFAULTS[c.strategy]["exit"]; b=c.bars
    stop=entry*(1-cfg["stop"]/100); target=entry*(1+cfg["target"]/100)
    highest=entry; trailing=False
    deadline=c.fill_ts+pd.Timedelta(minutes=cfg["minutes"])
    raw_exit=float(b.iloc[-1]["Close"]); exit_i=len(b)-1; reason="EOD"
    for i in range(c.fill_i,len(b)):
        row=b.iloc[i]; hi=float(row["High"]); lo=float(row["Low"]); cl=float(row["Close"])
        ts=b.index[i]
        highest=max(highest,hi)
        # Conservative OHLC ambiguity: when stop and target/trail are both touchable,
        # adverse exit wins.
        if trailing and cfg["trail"] is not None:
            trail_stop=highest*(1-cfg["trail"]/100)
            if lo<=trail_stop:
                raw_exit=trail_stop; exit_i=i; reason="TRAIL"; break
        else:
            if lo<=stop:
                raw_exit=stop; exit_i=i; reason="STOP"; break
        if (not trailing and cfg["trail"] is not None and cfg["trail_activate"] is not None
                and hi>=entry*(1+cfg["trail_activate"]/100)):
            trailing=True
            trail_stop=highest*(1-cfg["trail"]/100)
            if lo<=trail_stop:
                raw_exit=trail_stop; exit_i=i; reason="TRAIL"; break
        if hi>=target:
            raw_exit=target; exit_i=i; reason="TARGET"; break
        if ts>=deadline:
            raw_exit=cl; exit_i=i; reason="TIME"; break
    _,_,eff=liquidity_at(b,exit_i,raw_exit)
    exit_price,exit_slip=adverse_price(raw_exit,"sell",notional,eff)
    return b.index[exit_i],exit_price,exit_slip,reason


def replay(candidates,start=START_EQUITY):
    cash=start; realized=0.0; open_heap=[]; seq=0; trades=[]
    # heap item: exit timestamp, seq, proceeds, Trade
    def close_through(ts):
        nonlocal cash,realized
        while open_heap and open_heap[0][0] <= ts:
            _,_,proceeds,t=heapq.heappop(open_heap)
            cash += proceeds; realized += t.pnl; trades.append(t)
    for c in candidates:
        close_through(c.fill_ts)
        equity=start+realized
        cap,eff=max_notional(equity,cash,c.bars,c.fill_i,c.raw_fill)
        if cap<50 or eff<=0: continue
        entry,entry_slip=adverse_price(c.raw_fill,"buy",cap,eff)
        # Recompute size once using slipped price but preserve dollar notional.
        shares=cap/entry
        exit_ts,exit_price,exit_slip,reason=simulate_exit(c,entry,cap)
        proceeds=shares*exit_price
        pnl=proceeds-cap
        ret=(exit_price/entry-1)*100
        t=Trade(c.strategy,c.ticker,c.date,c.gap_pct,c.signal.timestamp,c.fill_ts,exit_ts,
                float(c.signal.reference_price),c.raw_fill,entry,exit_price,cap,pnl,ret,reason,
                entry_slip,exit_slip)
        cash-=cap; seq+=1; heapq.heappush(open_heap,(exit_ts,seq,proceeds,t))
    close_through(pd.Timestamp.max.tz_localize("UTC"))
    return trades,start+realized


def stats(trades,start):
    if not trades:return dict(trades=0,pnl=0,pf=0,wr=0,mean=0,median=0,max_dd=0,end_equity=start)
    pnl=np.array([t.pnl for t in trades]); rets=np.array([t.return_pct for t in trades])
    gw=float(pnl[pnl>0].sum()); gl=float(-pnl[pnl<=0].sum()); pf=gw/gl if gl>0 else 99.0
    eq=start; peak=start; maxdd=0.0
    for t in sorted(trades,key=lambda x:x.exit_time):
        eq+=t.pnl; peak=max(peak,eq); maxdd=min(maxdd,(eq/peak-1)*100)
    return dict(trades=len(trades),pnl=float(pnl.sum()),pf=pf,wr=float((pnl>0).mean()*100),
                mean=float(rets.mean()),median=float(np.median(rets)),max_dd=maxdd,end_equity=eq)


def grouped(trades):
    d={}
    for key,fn in [
      ("strategy",lambda t:t.strategy),
      ("year",lambda t:t.date[:4]),
      ("quarter",lambda t:str(pd.Period(t.date,freq="Q"))),
      ("month",lambda t:t.date[:7])]:
        groups=defaultdict(list)
        for t in trades: groups[fn(t)].append(t)
        d[key]={k:stats(v,START_EQUITY) for k,v in sorted(groups.items())}
    return d


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="stored_data_combined")
    ap.add_argument("--out",default="results/legacy_long_resurrection_baseline_20261004")
    a=ap.parse_args()
    # Preserve old universe builder but loosen only enough to avoid excluding a recovered
    # strategy before its own gap filter.  No optimizer is run.
    test_full.MIN_GAP_PCT=5.0
    test_full.MIN_PM_VOLUME=250_000
    all_dates,picks=test_full.load_all_picks([a.data])
    cands=build_candidates(all_dates,picks)
    trades,end=replay(cands)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    payload={
      "study":"legacy_long_resurrection_clean_unoptimized_baseline",
      "coverage":{"first":all_dates[0] if all_dates else None,"last":all_dates[-1] if all_dates else None,
                  "days":len(all_dates),"candidates":len(cands)},
      "execution":"30% equity/$100k + 5% cumulative/8% 10m/15% effective 2-6-10m caps; impact=0.05+0.5/max(price,.1)+3*sqrt(notional/effective_liquidity)%; next-bar-open execution",
      "defaults":DEFAULTS,"overall":stats(trades,START_EQUITY),"groups":grouped(trades),
      "trades":[t.__dict__ | {"signal_time":str(t.signal_time),"fill_time":str(t.fill_time),"exit_time":str(t.exit_time)} for t in trades],
    }
    (out/"summary.json").write_text(json.dumps(payload,indent=2,default=str))
    pd.DataFrame(payload["trades"]).to_csv(out/"trades.csv",index=False)
    print(json.dumps({k:payload[k] for k in ["coverage","overall"]},indent=2))
    print(json.dumps(payload["groups"]["strategy"],indent=2))


if __name__=="__main__": main()
