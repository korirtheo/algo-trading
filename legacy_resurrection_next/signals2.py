"""Strictly causal recovered signals: second legacy-long batch.

Recovered from test_green_candle_combined.py @ fc985c67b2.
All functions return only after the confirming bar CLOSE is known.  The caller
must execute on the next bar (never at this close).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from research.legacy_long_resurrection.causal_execution import Signal


@dataclass(frozen=True)
class RecoveredSignal:
    signal: Signal
    meta: dict[str, Any] = field(default_factory=dict)


def opening_range_breakout(bars, *, range_candles=5, breakout_vol_mult=1.5,
                           max_entry_candle=30):
    if len(bars) <= range_candles:
        return None
    rh=-np.inf; rl=np.inf; vol_sum=0.0
    for i in range(min(range_candles,len(bars))):
        r=bars.iloc[i]
        rh=max(rh,float(r.High)); rl=min(rl,float(r.Low)); vol_sum+=float(r.Volume)
    avg=vol_sum/range_candles
    for i in range(range_candles,min(len(bars),max_entry_candle)):
        r=bars.iloc[i]
        if float(r.Close)>rh and (avg<=0 or float(r.Volume)>=breakout_vol_mult*avg):
            s=Signal("O",i,bars.index[i],float(r.Close),"OPENING_RANGE_BREAK")
            return RecoveredSignal(s,{"structural_stop":rl,"range_high":rh})
    return None


def first_pullback(bars, *, min_run_pct=5.0, run_window=15, pullback_pct=3.0,
                   pullback_vol_ratio=.5, bounce_vol_mult=1.5,
                   max_entry_candle=45):
    if len(bars)<run_window+2:
        return None
    day_open=float(bars.iloc[0].Open)
    run_high=0.0; run_vol=0.0
    for i in range(min(run_window,len(bars))):
        r=bars.iloc[i]
        run_high=max(run_high,float(r.High)); run_vol+=float(r.Volume)
    if day_open<=0 or (run_high/day_open-1)*100<min_run_pct:
        return None
    avg_run=run_vol/max(run_window,1)
    pb_low=999999.0; pb_vol=0.0; pb_n=0; armed=False
    for i in range(run_window,min(len(bars),max_entry_candle)):
        r=bars.iloc[i]; vol=float(r.Volume)
        if not armed:
            pb_low=min(pb_low,float(r.Low)); pb_vol+=vol; pb_n+=1
            dd=(run_high-pb_low)/run_high*100 if run_high>0 else 0
            if dd>=pullback_pct:
                avg_pb=pb_vol/max(pb_n,1)
                if avg_run<=0 or avg_pb<=pullback_vol_ratio*avg_run:
                    armed=True
            continue
        avg_pb=pb_vol/max(pb_n,1)
        if float(r.Close)>pb_low*(1+pullback_pct/100) and (avg_pb<=0 or vol>=bounce_vol_mult*avg_pb):
            s=Signal("K",i,bars.index[i],float(r.Close),"FIRST_PULLBACK_BOUNCE")
            return RecoveredSignal(s,{"run_high":run_high,"pullback_low":pb_low})
    return None


def micro_flag(bars, *, min_spike_pct=5.0, min_base_candles=3,
               max_base_candles=8, max_base_range_pct=3.0,
               breakout_vol_mult=1.5, max_entry_candle=60):
    if len(bars)<3:
        return None
    day_open=float(bars.iloc[0].Open)
    first_vol=float(bars.iloc[0].Volume)
    spike=False; spike_high=0.0
    base_n=0; base_high=0.0; base_low=999999.0
    for i in range(min(len(bars),max_entry_candle)):
        r=bars.iloc[i]; hi=float(r.High); lo=float(r.Low); cl=float(r.Close); vol=float(r.Volume)
        if not spike:
            spike_high=max(spike_high,hi)
            if day_open>0 and (spike_high/day_open-1)*100>=min_spike_pct:
                spike=True; base_n=0; base_high=0.0; base_low=999999.0
            continue
        if base_n<max_base_candles:
            base_n+=1; base_high=max(base_high,hi); base_low=min(base_low,lo)
            if base_low>0:
                br=(base_high-base_low)/base_low*100
                if br>max_base_range_pct:
                    # Faithful reset from legacy state machine.
                    spike=False; spike_high=hi; base_n=0; base_high=0.0; base_low=999999.0
            continue
        if cl>base_high:
            if base_n>=min_base_candles and vol>=breakout_vol_mult*(first_vol or vol):
                s=Signal("C",i,bars.index[i],cl,"MICRO_FLAG_BREAK")
                return RecoveredSignal(s,{"base_high":base_high,"base_low":base_low})
            return None
    return None


def stuff_and_break(bars, *, min_hod_tests=2, hod_tolerance_pct=.5,
                    rejection_pct=1.0, breakout_vol_mult=1.5,
                    max_entry_candle=90):
    if len(bars)<3:
        return None
    hod=0.0; tests=0; rejected=False; first_vol=float(bars.iloc[0].Volume)
    for i in range(min(len(bars),max_entry_candle)):
        r=bars.iloc[i]; hi=float(r.High); lo=float(r.Low); cl=float(r.Close); vol=float(r.Volume)
        if hi>hod:
            old=hod; hod=hi
            if tests>=min_hod_tests and old>0 and vol>=breakout_vol_mult*(first_vol or vol):
                s=Signal("S",i,bars.index[i],cl,"STUFF_AND_BREAK")
                return RecoveredSignal(s,{"prior_hod":old,"hod_tests":tests})
        elif hod>0:
            dist=(hod-hi)/hod*100
            if dist<=hod_tolerance_pct and not rejected:
                rejected=True
            if rejected:
                pb=(hod-lo)/hod*100
                if pb>=rejection_pct:
                    tests+=1; rejected=False
    return None


def hod_reclaim(bars, *, min_hod_age=10, pullback_from_hod_pct=3.0,
                max_entry_candle=120):
    hod=0.0; hod_i=0; pullback=False
    for i in range(min(len(bars),max_entry_candle)):
        r=bars.iloc[i]; hi=float(r.High); lo=float(r.Low); cl=float(r.Close)
        if hi>hod:
            if hod>0 and pullback and ((i+1)-hod_i)>=min_hod_age:
                s=Signal("N",i,bars.index[i],cl,"HOD_RECLAIM")
                return RecoveredSignal(s,{"prior_hod":hod,"hod_bar":hod_i})
            hod=hi; hod_i=i+1; pullback=False
        elif hod>0 and (hod-lo)/hod*100>=pullback_from_hod_pct:
            pullback=True
    return None


def range_reversion(bars, *, min_first_leg_gain_pct=5.0,
                    first_leg_window_bars=15, min_pullback_pct=10.0,
                    min_recovery_pct=3.0, min_bars_since_peak=5,
                    target_pct_of_peak=92.0, stop_pct_below_trough=2.0,
                    min_entry_room_pct=4.0, min_vol_vs_avg=1.0,
                    vol_avg_bars=5, max_entry_hhmm="14:30",
                    require_green=True):
    if len(bars)<=first_leg_window_bars:
        return None
    op=float(bars.iloc[0].Open)
    peak=0.0; peak_i=0
    vols=[]
    for i in range(min(first_leg_window_bars,len(bars))):
        r=bars.iloc[i]; vols.append(float(r.Volume))
        if float(r.High)>peak:
            peak=float(r.High); peak_i=i+1
    if op<=0 or peak<=0 or (peak-op)/op*100<min_first_leg_gain_pct:
        return None

    trough=0.0
    for i in range(first_leg_window_bars,len(bars)):
        r=bars.iloc[i]; hi=float(r.High); lo=float(r.Low); cl=float(r.Close); oo=float(r.Open); vol=float(r.Volume)
        vols.append(vol)
        if len(vols)>vol_avg_bars+1:
            vols.pop(0)
        if trough==0 or lo<trough:
            trough=lo
        try:
            ts=bars.index[i].tz_convert("America/New_York") if bars.index[i].tzinfo else bars.index[i]
            hhmm=f"{ts.hour:02d}:{ts.minute:02d}"
        except Exception:
            hhmm="00:00"
        if hhmm>max_entry_hhmm:
            break
        pullback=(peak-trough)/peak*100
        recovery=(cl-trough)/trough*100 if trough>0 else 0
        bars_since=(i+1)-peak_i
        green=(not require_green) or cl>oo
        prior=vols[:-1]
        avg=float(np.mean(prior)) if prior else 0.0
        vol_ok=avg<=0 or vol>=avg*min_vol_vs_avg
        target=peak*target_pct_of_peak/100
        room=(target-cl)/cl*100 if cl>0 else -999
        if (pullback>=min_pullback_pct and bars_since>=min_bars_since_peak
                and recovery>=min_recovery_pct and green and vol_ok
                and room>=min_entry_room_pct):
            stop=trough*(1-stop_pct_below_trough/100)
            s=Signal("X",i,bars.index[i],cl,"RANGE_REVERSION_SECOND_LEG")
            return RecoveredSignal(s,{"structural_target":target,"structural_stop":stop,
                                      "first_leg_peak":peak,"trough":trough})
    return None
