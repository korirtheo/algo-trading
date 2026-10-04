"""Execution-aware evaluator for second-batch recovered legacy longs.

Signals: O/K/C/S/N/X.  Signals are close-confirmed; entries are next-bar open.
Execution uses the canonical volatility-adjusted 2/6/10-minute liquidity model
installed by legacy_resurrection_exact.execution.

Cash released by partial exits becomes available at the actual partial timestamp.
Unrealized PnL is intentionally not borrowed against (no leverage).
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd

import research.legacy_long_resurrection.run_baseline as rb
from legacy_resurrection_exact.execution import install
from research.legacy_long_resurrection.causal_execution import next_bar_open_fill, assert_causal_fill
from legacy_resurrection_next.signals2 import (
    opening_range_breakout, first_pullback, micro_flag,
    stuff_and_break, hod_reclaim, range_reversion,
)

install(rb)

DEFAULTS={
 "O":{"min_gap":10.0,"signal":dict(range_candles=5,breakout_vol_mult=1.5,max_entry_candle=30),
      "exit":dict(target1=8.0,target2=15.0,stop=0.0,partial=.50,trail=2.0,trail_activate=3.0,minutes=60)},
 "K":{"min_gap":10.0,"signal":dict(min_run_pct=5.0,run_window=15,pullback_pct=3.0,pullback_vol_ratio=.5,bounce_vol_mult=1.5,max_entry_candle=45),
      "exit":dict(target1=8.0,target2=15.0,stop=5.0,partial=.50,trail=2.0,trail_activate=3.0,minutes=60)},
 "C":{"min_gap":10.0,"signal":dict(min_spike_pct=5.0,min_base_candles=3,max_base_candles=8,max_base_range_pct=3.0,breakout_vol_mult=1.5,max_entry_candle=60),
      "exit":dict(target1=8.0,target2=15.0,stop=4.0,partial=.50,trail=2.0,trail_activate=3.0,minutes=60)},
 "S":{"min_gap":10.0,"signal":dict(min_hod_tests=2,hod_tolerance_pct=.5,rejection_pct=1.0,breakout_vol_mult=1.5,max_entry_candle=90),
      "exit":dict(target1=8.0,target2=15.0,stop=4.0,partial=.50,trail=2.0,trail_activate=3.0,minutes=90)},
 "N":{"min_gap":10.0,"signal":dict(min_hod_age=10,pullback_from_hod_pct=3.0,max_entry_candle=120),
      "exit":dict(target1=8.0,target2=15.0,stop=5.0,partial=.50,trail=2.0,trail_activate=3.0,minutes=90)},
 "X":{"min_gap":5.0,"signal":dict(min_first_leg_gain_pct=5.0,first_leg_window_bars=15,min_pullback_pct=10.0,min_recovery_pct=3.0,min_bars_since_peak=5,target_pct_of_peak=92.0,stop_pct_below_trough=2.0,min_entry_room_pct=4.0,min_vol_vs_avg=1.0,vol_avg_bars=5,max_entry_hhmm="14:30",require_green=True),
      "exit":dict(partial=0.0,trail=5.0,trail_activate=5.0,minutes=60)},
}


@dataclass
class Candidate:
    strategy:str; ticker:str; date:str; gap_pct:float; bars:pd.DataFrame
    recovered:object; fill_i:int; fill_ts:object; raw_fill:float


@dataclass
class Trade:
    strategy:str; ticker:str; date:str; gap_pct:float
    signal_time:object; fill_time:object; exit_time:object
    signal_price:float; raw_fill:float; entry:float; exit:float
    notional:float; pnl:float; return_pct:float; exit_reason:str
    entry_slip_pct:float; exit_slip_pct:float; partial_taken:bool


def _detect(s,b,cfg):
    p=cfg["signal"]
    if s=="O": return opening_range_breakout(b,**p)
    if s=="K": return first_pullback(b,**p)
    if s=="C": return micro_flag(b,**p)
    if s=="S": return stuff_and_break(b,**p)
    if s=="N": return hod_reclaim(b,**p)
    return range_reversion(b,**p)


def build_candidates(dates,picks,configs=None):
    configs=configs or DEFAULTS
    out=[]
    for d in dates:
        for p in picks.get(d,[]):
            b=p["market_hour_candles"].sort_index()
            if len(b)<3: continue
            for s,cfg in configs.items():
                if float(p["gap_pct"])<cfg["min_gap"]: continue
                rec=_detect(s,b,cfg)
                if rec is None: continue
                f=next_bar_open_fill(rec.signal,b,0.0)
                if f is None: continue
                assert_causal_fill(f)
                out.append(Candidate(s,p["ticker"],d,float(p["gap_pct"]),b,rec,
                                     f.fill_bar_index,f.fill_timestamp,f.raw_fill_price))
    out.sort(key=lambda x:(x.fill_ts,x.strategy,x.ticker))
    return out


def _sell(raw,shares,bars,i):
    notional=max(0.0,raw*shares)
    _,_,eff=rb.liquidity_at(bars,i,raw)
    px,sl=rb.adverse_price(raw,"sell",notional,eff)
    return px,sl,shares*px


def plan_trade(c,entry,cap):
    b=c.bars; cfg=DEFAULTS[c.strategy]["exit"]; meta=c.recovered.meta
    orig_shares=cap/entry; remaining=orig_shares
    partial_taken=False; partial_proceeds=0.0; partial_slip=0.0
    cashflows=[]
    highest=entry; trailing=False
    deadline=c.fill_ts+pd.Timedelta(minutes=cfg["minutes"])
    final_raw=float(b.iloc[-1].Close); final_i=len(b)-1; reason="EOD"

    if c.strategy=="X":
        hard=float(meta["structural_stop"]); target=float(meta["structural_target"])
    else:
        hard=(float(meta["structural_stop"]) if c.strategy=="O" and cfg["stop"]==0
              else entry*(1-cfg["stop"]/100))
        target1=entry*(1+cfg["target1"]/100); target2=entry*(1+cfg["target2"]/100)

    for i in range(c.fill_i,len(b)):
        r=b.iloc[i]; hi=float(r.High); lo=float(r.Low); cl=float(r.Close); ts=b.index[i]
        highest=max(highest,hi)

        if trailing:
            trail_stop=highest*(1-cfg["trail"]/100)
            if lo<=trail_stop:
                final_raw=trail_stop; final_i=i; reason="TRAIL"; break
        elif hard>0 and lo<=hard:
            final_raw=hard; final_i=i; reason="STOP"; break

        if not trailing and hi>=entry*(1+cfg["trail_activate"]/100):
            trailing=True
            trail_stop=highest*(1-cfg["trail"]/100)
            if lo<=trail_stop:
                final_raw=trail_stop; final_i=i; reason="TRAIL"; break

        if c.strategy=="X":
            if hi>=target:
                final_raw=target; final_i=i; reason="TARGET"; break
        else:
            if (not partial_taken) and cfg["partial"]>0 and hi>=target1:
                sell_sh=orig_shares*cfg["partial"]
                sell_sh=min(sell_sh,remaining)
                px,sl,proceeds=_sell(target1,sell_sh,b,i)
                remaining-=sell_sh; partial_taken=True
                partial_proceeds+=proceeds; partial_slip=sl
                cost=cap*(sell_sh/orig_shares)
                cashflows.append((ts,proceeds,proceeds-cost,False))
            if remaining>1e-12 and hi>=target2:
                final_raw=target2; final_i=i; reason="TARGET"; break

        if ts>=deadline:
            final_raw=cl; final_i=i; reason="TIME"; break

    px,final_sl,final_proceeds=_sell(final_raw,remaining,b,final_i)
    cost_remaining=cap*(remaining/orig_shares)
    cashflows.append((b.index[final_i],final_proceeds,final_proceeds-cost_remaining,True))
    pnl=partial_proceeds+final_proceeds-cap
    weighted_exit=(partial_proceeds+final_proceeds)/orig_shares if orig_shares>0 else px
    trade=Trade(c.strategy,c.ticker,c.date,c.gap_pct,c.recovered.signal.timestamp,c.fill_ts,
                b.index[final_i],float(c.recovered.signal.reference_price),c.raw_fill,
                entry,weighted_exit,cap,pnl,pnl/cap*100,reason,0.0,
                max(partial_slip,final_sl),partial_taken)
    return trade,cashflows


def replay(candidates,start=rb.START_EQUITY):
    cash=float(start); realized=0.0; events=[]; trades=[]; seq=0
    def process(ts):
        nonlocal cash,realized
        while events and events[0][0]<=ts:
            _,_,proceeds,pnl_piece,is_final,t=heapq.heappop(events)
            cash+=proceeds; realized+=pnl_piece
            if is_final: trades.append(t)

    for c in candidates:
        process(c.fill_ts)
        equity=start+realized
        cap,eff=rb.max_notional(equity,cash,c.bars,c.fill_i,c.raw_fill)
        if cap<50 or eff<=0: continue
        entry,entry_slip=rb.adverse_price(c.raw_fill,"buy",cap,eff)
        t,flows=plan_trade(c,entry,cap)
        t.entry_slip_pct=entry_slip
        cash-=cap
        for ts,proceeds,pnl_piece,is_final in flows:
            seq+=1
            heapq.heappush(events,(ts,seq,proceeds,pnl_piece,is_final,t))
    if candidates:
        z=pd.Timestamp.max.tz_localize("UTC")
        process(z)
    return trades,start+realized


def stats(trades,start=rb.START_EQUITY):
    return rb.stats(trades,start)


def grouped(trades):
    d={}
    for key,fn in [
        ("strategy",lambda t:t.strategy),
        ("year",lambda t:t.date[:4]),
        ("quarter",lambda t:str(pd.Period(t.date,freq="Q"))),
        ("month",lambda t:t.date[:7])]:
        g=defaultdict(list)
        for t in trades:g[fn(t)].append(t)
        d[key]={k:stats(v) for k,v in sorted(g.items())}
    return d
