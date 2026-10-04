"""Corrected causal resurrection of legacy G (Big Gap Runner) and A (Quick Scalp).

The legacy implementation confirmed candle 2 and then filled at candle 1 close.
This file rebuilds the ideas only:
  G: large gap + second green + new high
  A: lower gap + strong first body + second green + new high
Signal occurs only after candle 2 is complete. Fill is the NEXT bar open.

2025 is selection. 2026 is untouched validation. Dynamic execution comes from
run_baseline.py. G and A are researched independently; no priority masking.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import optuna
from optuna.samplers import TPESampler
import test_full
import research.legacy_long_resurrection.run_baseline as rb
from research.legacy_long_resurrection.causal_execution import Signal

SEED=20261005; N_TRIALS=700
OUT=Path("results/ga_corrected_resurrection_20261005")

def signal(b,gap,c,strat):
    if len(b)<3 or gap<c["min_gap"]: return None
    x=b.iloc[:2]
    o1,c1,h1=float(x.iloc[0].Open),float(x.iloc[0].Close),float(x.iloc[0].High)
    o2,c2,h2=float(x.iloc[1].Open),float(x.iloc[1].Close),float(x.iloc[1].High)
    if o1<=0:return None
    body=(c1/o1-1)*100
    if body<c["min_body"]: return None
    if c2<=o2 or h2<=h1:return None
    # Optional causal volume confirmation: candle 2 volume versus candle 1.
    if float(x.iloc[1].Volume) < float(x.iloc[0].Volume)*c["vol_ratio"]:return None
    return Signal(strat,1,b.index[1],c2,"CANDLE2_CONFIRMED")

def cfg(t,strat):
    if strat=="G":
      ming=t.suggest_int("min_gap",20,50,step=5); body=t.suggest_float("min_body",0,4,step=1)
    else:
      ming=t.suggest_int("min_gap",10,30,step=5); body=t.suggest_float("min_body",2,8,step=1)
    return dict(min_gap=ming,min_body=body,
      vol_ratio=t.suggest_float("vol_ratio",0,1.5,step=.25),
      target=t.suggest_float("target",8,30,step=2),
      stop=t.suggest_float("stop",3,10,step=1),
      trail=t.suggest_float("trail",1,8,step=1),
      trail_activate=t.suggest_float("trail_activate",3,15,step=2),
      minutes=t.suggest_int("minutes",10,90,step=10))

def build(dates,picks,c,strat):
    out=[]
    for d in dates:
      for p in picks.get(d,[]):
        b=p["market_hour_candles"].sort_index()
        s=signal(b,float(p["gap_pct"]),c,strat)
        if s is None:continue
        f=rb.next_bar_open_fill(s,b,0)
        if f is None:continue
        rb.assert_causal_fill(f)
        out.append(rb.Candidate(strat,p["ticker"],d,float(p["gap_pct"]),float(p["premarket_high"]),
                   b,s,f.fill_bar_index,f.fill_timestamp,f.raw_fill_price))
    out.sort(key=lambda x:(x.fill_ts,x.ticker));return out

def eval_(dates,picks,c,strat):
    old=rb.DEFAULTS
    rb.DEFAULTS={strat:{"min_gap":c["min_gap"],"signal":{},"exit":{
      k:c[k] for k in ["target","stop","trail","trail_activate","minutes"]}}}
    try:
      cs=build(dates,picks,c,strat);ts,_=rb.replay(cs,rb.START_EQUITY)
      return rb.stats(ts,rb.START_EQUITY)
    finally:rb.DEFAULTS=old

def score(m):
    n=m["trades"]
    if n<50:return -1e9+n
    return (m["pnl"]+1800*m["mean"]+2200*m["median"]+1800*min(m["pf"],3)
            +3500*math.log1p(n)-650*abs(m["max_dd"])
            -6000*max(0,4-m["mean"])-7000*max(0,3.5-m["median"]))

def run_one(strat,dev,val,picks):
    st=optuna.create_study(direction="maximize",sampler=TPESampler(seed=SEED+(0 if strat=="G" else 1),
      n_startup_trials=140,multivariate=True),study_name=f"{strat}_corrected")
    def obj(t):
      c=cfg(t,strat);m=eval_(dev,picks,c,strat)
      for k,v in m.items():
        if isinstance(v,(int,float)):t.set_user_attr(k,float(v))
      return score(m)
    st.optimize(obj,n_trials=N_TRIALS,show_progress_bar=False)
    ranked=sorted(st.trials,key=lambda t:t.value if t.value is not None else -1e99,reverse=True)[:30]
    rows=[]
    for rank,t in enumerate(ranked,1):
      class F:
        def __init__(self,p):self.p=p
        suggest_int=lambda self,n,*a,**k:self.p[n]
        suggest_float=lambda self,n,*a,**k:self.p[n]
      c=cfg(F(t.params),strat)
      rows.append({"rank":rank,"params":c,"development":eval_(dev,picks,c,strat),"validation":eval_(val,picks,c,strat)})
    return {"best_dev":rows[0],"top30":rows,
      "high_payoff_validation_count":sum(x["validation"]["trades"]>=10 and x["validation"]["pf"]>1 and x["validation"]["mean"]>=5 and x["validation"]["median"]>=4 for x in rows)}

def main():
    test_full.MIN_GAP_PCT=5;test_full.MIN_PM_VOLUME=250_000
    dates,picks=test_full.load_all_picks(["stored_data_combined"])
    dev=[d for d in dates if d.startswith("2025-")];val=[d for d in dates if d.startswith("2026-")]
    res={"coverage":[dates[0],dates[-1]],"selection":"2025","validation":"2026 untouched","trials_per_strategy":N_TRIALS}
    for s in ["G","A"]:res[s]=run_one(s,dev,val,picks)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(res,indent=2,default=str))
    print(json.dumps({k:(v if k not in ["G","A"] else {"best_dev":v["best_dev"],"high_payoff_validation_count":v["high_payoff_validation_count"]}) for k,v in res.items()},indent=2,default=str))
if __name__=="__main__":main()
