"""High-payoff VWAP reclaim research.

Goal: determine whether V can become a genuine second-leg runner strategy rather
than a ~3% central-tendency scalp. Selection uses 2025 ONLY. 2026 remains
untouched validation.

This study changes BOTH entry context and exit architecture:
- entry: washout depth/duration, reclaim RVOL/body, PMH/HOD room, local pivot,
  optional reclaim+hold confirmation, and pre-reclaim acceleration
- exit: larger target or state/trend exit; structural invalidation below VWAP/base
Objective explicitly rewards mean + median net return and trade count while
requiring PF and penalizing DD. No 2026 metric enters selection.
"""
from __future__ import annotations
import copy, json, math
from pathlib import Path
import numpy as np, pandas as pd
import optuna
from optuna.samplers import TPESampler
import test_full
import research.legacy_long_resurrection.run_baseline as rb
from research.legacy_long_resurrection.causal_execution import Signal

SEED=20261004
N_TRIALS=600
OUT=Path("results/vwap_reclaim_runner_research_20261004")

def v2_signal(b,p,cfg):
    vw=rb.vwap_reclaim.__globals__["cumulative_vwap"](b)
    close=b.Close.to_numpy(float); op=b.Open.to_numpy(float); hi=b.High.to_numpy(float)
    lo=b.Low.to_numpy(float); vol=b.Volume.to_numpy(float)
    below=0; maxdepth=0.; trough=1e99
    for i in range(2,min(len(b),cfg["latest"])):
        if np.isfinite(vw[i]) and close[i] < vw[i]:
            below+=1; maxdepth=max(maxdepth,(vw[i]-close[i])/vw[i]*100); trough=min(trough,lo[i]); continue
        if below < cfg["below_bars"] or maxdepth < cfg["depth"] or not np.isfinite(vw[i]):
            below=0; maxdepth=0.; trough=1e99; continue
        avgv=float(np.mean(vol[max(0,i-10):i])); rvol=vol[i]/avgv if avgv>0 else 0
        body=(close[i]/op[i]-1)*100 if op[i]>0 else 0
        prior_hod=float(np.max(hi[:i])) if i else hi[i]
        room=(prior_hod/close[i]-1)*100 if close[i]>0 else -99
        pmroom=(float(p["premarket_high"])/close[i]-1)*100 if close[i]>0 else -99
        # local pivot: reclaim bar must also clear recent closes when requested.
        pivot=float(np.max(close[max(0,i-cfg["pivot_bars"]):i])) if i else close[i]
        accel=(close[i]/close[max(0,i-cfg["accel_bars"])]-1)*100 if i>=cfg["accel_bars"] else 0
        ok=(rvol>=cfg["rvol"] and body>=cfg["body"] and room>=cfg["hod_room"]
            and pmroom>=cfg["pmh_room"] and accel>=cfg["accel"]
            and (cfg["pivot_bars"]==0 or close[i]>pivot))
        if ok:
            j=i+cfg["hold_bars"]
            if j>=len(b) or j>=cfg["latest"]: return None
            if cfg["hold_bars"] and min(close[i:j+1]) < vw[i]: return None
            return Signal("V2",j,b.index[j],float(close[j]),"VWAP_SECOND_LEG_RECLAIM")
        below=0; maxdepth=0.; trough=1e99
    return None

def cfg(tr):
    return dict(
      min_gap=tr.suggest_int("min_gap",10,40,step=2),
      below_bars=tr.suggest_int("below_bars",2,12),
      depth=tr.suggest_float("depth",1,10,step=1),
      rvol=tr.suggest_float("rvol",1.0,6.0,step=.5),
      body=tr.suggest_float("body",0,5,step=.5),
      hod_room=tr.suggest_float("hod_room",-5,20,step=2.5),
      pmh_room=tr.suggest_float("pmh_room",-10,20,step=2.5),
      pivot_bars=tr.suggest_categorical("pivot_bars",[0,2,3,5,8]),
      accel_bars=tr.suggest_categorical("accel_bars",[2,3,5]),
      accel=tr.suggest_float("accel",-2,8,step=1),
      hold_bars=tr.suggest_int("hold_bars",0,2),
      latest=tr.suggest_int("latest",30,120,step=10),
      target=tr.suggest_float("target",10,35,step=2.5),
      stop=tr.suggest_float("stop",3,12,step=1),
      trail=tr.suggest_float("trail",3,12,step=1),
      trail_activate=tr.suggest_float("trail_activate",5,20,step=2.5),
      minutes=tr.suggest_int("minutes",30,150,step=15))

def candidates(dates,picks,c):
    out=[]
    for d in dates:
      for p in picks.get(d,[]):
        if p["gap_pct"]<c["min_gap"]: continue
        b=p["market_hour_candles"].sort_index()
        s=v2_signal(b,p,c)
        if s is None: continue
        f=rb.next_bar_open_fill(s,b,0)
        if f is None: continue
        out.append(rb.Candidate("V2",p["ticker"],d,p["gap_pct"],p["premarket_high"],b,s,f.fill_bar_index,f.fill_timestamp,f.raw_fill_price))
    out.sort(key=lambda x:(x.fill_ts,x.ticker)); return out

def evaluate(dates,picks,c):
    old=rb.DEFAULTS
    rb.DEFAULTS={"V2":{"min_gap":c["min_gap"],"signal":{},"exit":{k:c[k] for k in ["target","stop","trail","trail_activate","minutes"]}}}
    try:
      cs=candidates(dates,picks,c); ts,_=rb.replay(cs,rb.START_EQUITY); return rb.stats(ts,rb.START_EQUITY),len(cs),ts
    finally: rb.DEFAULTS=old

def score(m):
    n=m["trades"]
    if n<35: return -1e9+n
    # Explicitly demand economics. Below 4% median or 5% mean receives a large penalty.
    quality=2500*min(m["pf"],3)+1800*m["mean"]+2200*m["median"]
    count=4000*math.log1p(n)
    dd=700*abs(m["max_dd"])
    floor_pen=12000*max(0,4-m["median"])+9000*max(0,5-m["mean"])
    return quality+count-dd-floor_pen

def main():
    test_full.MIN_GAP_PCT=5; test_full.MIN_PM_VOLUME=250_000
    dates,picks=test_full.load_all_picks(["stored_data_combined"])
    dev=[d for d in dates if d.startswith("2025-")]; val=[d for d in dates if d.startswith("2026-")]
    study=optuna.create_study(direction="maximize",sampler=TPESampler(seed=SEED,n_startup_trials=120,multivariate=True),study_name="V2_runner_2025")
    def obj(t):
      c=cfg(t); m,n,_=evaluate(dev,picks,c)
      for k,v in m.items():
        if isinstance(v,(int,float)):t.set_user_attr(k,float(v))
      return score(m)
    study.optimize(obj,n_trials=N_TRIALS,show_progress_bar=False)
    ranked=sorted(study.trials,key=lambda t:t.value if t.value is not None else -1e99,reverse=True)[:30]
    rows=[]
    for rank,t in enumerate(ranked,1):
      class F:
        def __init__(self,p):self.p=p
        suggest_int=lambda self,n,*a,**k:self.p[n]
        suggest_float=lambda self,n,*a,**k:self.p[n]
        suggest_categorical=lambda self,n,*a,**k:self.p[n]
      c=cfg(F(t.params)); dm,_,_=evaluate(dev,picks,c); vm,_,vts=evaluate(val,picks,c)
      rows.append(dict(rank=rank,params=c,development=dm,validation=vm,
                       validation_trades=[x.__dict__|{"signal_time":str(x.signal_time),"fill_time":str(x.fill_time),"exit_time":str(x.exit_time)} for x in vts]))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"top30_with_untouched_2026.json").write_text(json.dumps(rows,indent=2,default=str))
    # Selection diagnostics: do NOT select on these; report robustness across top dev candidates.
    good=[x for x in rows if x["validation"]["mean"]>=5 and x["validation"]["median"]>=4 and x["validation"]["pf"]>1 and x["validation"]["trades"]>=10]
    print(json.dumps({"coverage":[dates[0],dates[-1]],"best_dev":rows[0],"top30_high_payoff_2026_count":len(good),
                      "high_payoff_2026":[{k:x[k] for k in ["rank","params","development","validation"]} for x in good]},indent=2,default=str))

if __name__=="__main__": main()
