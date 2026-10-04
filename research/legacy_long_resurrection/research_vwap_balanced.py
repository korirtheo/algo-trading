"""Balanced VWAP reclaim research: preserve frequency, improve payoff via exits.

2025 is selection only; 2026 is untouched validation. This study deliberately
avoids the over-filtering failure of V2. It keeps the original causal VWAP-reclaim
family, permits at most one hold-confirmation bar, and searches risk/runner exits.

Execution stays identical to run_baseline.py: next-bar-open entry, current
liquidity caps, dynamic adverse impact, shared cash.
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
from research.legacy_long_resurrection.signals import vwap_reclaim, cumulative_vwap

SEED=20261005
N_TRIALS=800
OUT=Path("results/vwap_reclaim_balanced_research_20261005")

def cfg(t):
    return dict(
      min_gap=t.suggest_int("min_gap",8,30,step=2),
      below=t.suggest_int("below",2,9),
      depth=t.suggest_float("depth",0,6,step=.5),
      rvol=t.suggest_float("rvol",1,4,step=.5),
      latest=t.suggest_int("latest",40,120,step=10),
      hold=t.suggest_int("hold",0,1),
      target=t.suggest_float("target",8,25,step=1),
      stop=t.suggest_float("stop",2,8,step=.5),
      trail=t.suggest_float("trail",1,5,step=.5),
      trail_activate=t.suggest_float("trail_activate",3,12,step=1),
      minutes=t.suggest_int("minutes",30,120,step=15))

def build(dates,picks,c):
    out=[]
    for d in dates:
      for p in picks.get(d,[]):
        if float(p["gap_pct"])<c["min_gap"]: continue
        b=p["market_hour_candles"].sort_index()
        if len(b)<4: continue
        s=vwap_reclaim(b,min_below_candles=c["below"],min_below_pct=c["depth"],
                       vol_spike_ratio=c["rvol"],max_entry_candle=c["latest"])
        if s is None: continue
        if c["hold"]:
          j=s.bar_index+1
          if j>=len(b): continue
          vw=cumulative_vwap(b)
          if not np.isfinite(vw[j]) or float(b.iloc[j]["Close"])<=vw[j]: continue
          s=Signal("V3",j,b.index[j],float(b.iloc[j]["Close"]),"VWAP_RECLAIM_HOLD")
        else:
          s=Signal("V3",s.bar_index,s.timestamp,s.reference_price,s.reason)
        f=rb.next_bar_open_fill(s,b,0.0)
        if f is None: continue
        rb.assert_causal_fill(f)
        out.append(rb.Candidate("V3",p["ticker"],d,float(p["gap_pct"]),float(p["premarket_high"]),
                    b,s,f.fill_bar_index,f.fill_timestamp,f.raw_fill_price))
    out.sort(key=lambda x:(x.fill_ts,x.ticker))
    return out

def evaluate(dates,picks,c):
    old=rb.DEFAULTS
    rb.DEFAULTS={"V3":{"min_gap":c["min_gap"],"signal":{},"exit":{
      "target":c["target"],"stop":c["stop"],"trail":c["trail"],
      "trail_activate":c["trail_activate"],"minutes":c["minutes"]}}}
    try:
      cs=build(dates,picks,c); ts,_=rb.replay(cs,rb.START_EQUITY)
      return rb.stats(ts,rb.START_EQUITY),ts
    finally:
      rb.DEFAULTS=old

def score(m):
    n=m["trades"]
    if n<55: return -1e9+n
    # Frequency-preserving economic objective. PnL remains relevant, but central
    # tendency gets much more weight than in the original V search.
    quality=1400*m["mean"]+1800*m["median"]+1800*min(m["pf"],3)
    freq=3500*math.log1p(n)
    dd=650*abs(m["max_dd"])
    floors=5000*max(0,4-m["mean"])+6000*max(0,3.5-m["median"])
    return m["pnl"]+quality+freq-dd-floors

def main():
    test_full.MIN_GAP_PCT=5; test_full.MIN_PM_VOLUME=250_000
    dates,picks=test_full.load_all_picks(["stored_data_combined"])
    dev=[d for d in dates if d.startswith("2025-")]
    val=[d for d in dates if d.startswith("2026-")]
    study=optuna.create_study(direction="maximize",
      sampler=TPESampler(seed=SEED,n_startup_trials=160,multivariate=True),
      study_name="V3_balanced_2025")
    def obj(t):
      c=cfg(t); m,_=evaluate(dev,picks,c)
      for k,v in m.items():
        if isinstance(v,(int,float)): t.set_user_attr(k,float(v))
      return score(m)
    study.optimize(obj,n_trials=N_TRIALS,show_progress_bar=False)
    ranked=sorted(study.trials,key=lambda t:t.value if t.value is not None else -1e99,reverse=True)[:30]
    rows=[]
    for rank,t in enumerate(ranked,1):
      class F:
        def __init__(self,p): self.p=p
        suggest_int=lambda self,n,*a,**k:self.p[n]
        suggest_float=lambda self,n,*a,**k:self.p[n]
      c=cfg(F(t.params)); dm,_=evaluate(dev,picks,c); vm,vts=evaluate(val,picks,c)
      rows.append(dict(rank=rank,params=c,development=dm,validation=vm))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"top30.json").write_text(json.dumps(rows,indent=2,default=str))
    # Robustness summaries only; never used to select params.
    gates={
      "high_payoff": [x for x in rows if x["validation"]["trades"]>=10 and x["validation"]["pf"]>1 and x["validation"]["mean"]>=5 and x["validation"]["median"]>=4],
      "better_than_original_central": [x for x in rows if x["validation"]["trades"]>=10 and x["validation"]["pf"]>1 and x["validation"]["mean"]>3.338 and x["validation"]["median"]>3.17],
    }
    summary={"coverage":[dates[0],dates[-1]],"trials":N_TRIALS,"best_dev":rows[0],
      "high_payoff_count":len(gates["high_payoff"]),
      "better_than_original_central_count":len(gates["better_than_original_central"]),
      "high_payoff":[{k:x[k] for k in ["rank","params","development","validation"]} for x in gates["high_payoff"][:10]],
      "better_than_original_central":[{k:x[k] for k in ["rank","params","development","validation"]} for x in gates["better_than_original_central"][:10]]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=="__main__": main()
