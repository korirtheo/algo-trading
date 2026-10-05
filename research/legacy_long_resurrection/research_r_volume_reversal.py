"""Causal resurrection of legacy R — Volume Spike Reversal / second impulse.

Original mechanism preserved: gapping stock, prior HOD established, meaningful pullback,
then a green expansion candle on abnormal prior-relative volume while still below HOD.
Signal is close-confirmed; fill is strictly next-bar open. 2025 selects; 2026 validates.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, optuna
from optuna.samplers import TPESampler
import test_full
import research.legacy_long_resurrection.run_baseline as rb
from research.legacy_long_resurrection.causal_execution import Signal
SEED=20261005; N_TRIALS=800
OUT=Path("results/r_volume_spike_reversal_20261005")

def cfg(t):
 return dict(min_gap=t.suggest_int("min_gap",10,40,step=5),
  scan_start=t.suggest_int("scan_start",20,100,step=10),
  scan_end=t.suggest_int("scan_end",100,200,step=10),
  pullback=t.suggest_float("pullback",2,12,step=1),
  vol_mult=t.suggest_float("vol_mult",1,4,step=.5),
  min_body=t.suggest_float("min_body",.5,5,step=.5),
  target=t.suggest_float("target",10,35,step=2.5),
  stop=t.suggest_float("stop",3,10,step=1),
  trail=t.suggest_float("trail",1,8,step=1),
  trail_activate=t.suggest_float("trail_activate",4,18,step=2),
  minutes=t.suggest_int("minutes",30,150,step=15))

def sig(b,gap,c):
 if gap<c["min_gap"] or len(b)<c["scan_start"]+20:return None
 o=b.Open.astype(float).to_numpy(); cl=b.Close.astype(float).to_numpy()
 h=b.High.astype(float).to_numpy(); v=b.Volume.astype(float).to_numpy()
 end=min(len(b),c["scan_end"])
 for i in range(c["scan_start"],end):
  if cl[i]<=o[i] or o[i]<=0:continue
  if (cl[i]/o[i]-1)*100<c["min_body"]:continue
  ph=float(np.max(h[:i]))
  if ph<=0 or (ph-cl[i])/ph*100<c["pullback"]:continue
  av=float(np.mean(v[i-20:i]))
  if av<=0 or v[i]/av<c["vol_mult"]:continue
  return Signal("R",i,b.index[i],cl[i],"VOLUME_SPIKE_REVERSAL")
 return None

def build(ds,picks,c):
 out=[]
 for d in ds:
  for p in picks.get(d,[]):
   b=p["market_hour_candles"].sort_index();s=sig(b,float(p["gap_pct"]),c)
   if s is None:continue
   f=rb.next_bar_open_fill(s,b,0)
   if f is None:continue
   rb.assert_causal_fill(f)
   out.append(rb.Candidate("R",p["ticker"],d,float(p["gap_pct"]),float(p["premarket_high"]),b,s,f.fill_bar_index,f.fill_timestamp,f.raw_fill_price))
 out.sort(key=lambda x:(x.fill_ts,x.ticker));return out

def ev(ds,picks,c):
 old=rb.DEFAULTS; rb.DEFAULTS={"R":{"min_gap":c["min_gap"],"signal":{},"exit":{k:c[k] for k in ["target","stop","trail","trail_activate","minutes"]}}}
 try:return rb.stats(rb.replay(build(ds,picks,c),rb.START_EQUITY)[0],rb.START_EQUITY)
 finally:rb.DEFAULTS=old

def score(m):
 n=m["trades"]
 if n<35:return -1e9+n
 return m["pnl"]+1900*m["mean"]+2300*m["median"]+1800*min(m["pf"],3)+3500*math.log1p(n)-650*abs(m["max_dd"])-7000*max(0,5-m["mean"])-8000*max(0,4-m["median"])

def main():
 test_full.MIN_GAP_PCT=5;test_full.MIN_PM_VOLUME=250000
 dates,picks=test_full.load_all_picks(["stored_data_combined"])
 dev=[d for d in dates if d.startswith("2025-")];val=[d for d in dates if d.startswith("2026-")]
 st=optuna.create_study(direction="maximize",sampler=TPESampler(seed=SEED,n_startup_trials=160,multivariate=True),study_name="R_corrected")
 def obj(t):
  c=cfg(t)
  if c["scan_start"]>=c["scan_end"]:return -1e9
  m=ev(dev,picks,c)
  for k,v in m.items():
   if isinstance(v,(int,float)):t.set_user_attr(k,float(v))
  return score(m)
 st.optimize(obj,n_trials=N_TRIALS,show_progress_bar=False)
 ranked=sorted(st.trials,key=lambda t:t.value if t.value is not None else -1e99,reverse=True)[:30];rows=[]
 for rank,t in enumerate(ranked,1):
  class F:
   def __init__(self,p):self.p=p
   suggest_int=lambda self,n,*a,**k:self.p[n]
   suggest_float=lambda self,n,*a,**k:self.p[n]
  c=cfg(F(t.params));rows.append({"rank":rank,"params":c,"development":ev(dev,picks,c),"validation":ev(val,picks,c)})
 hp=[x for x in rows if x["validation"]["trades"]>=10 and x["validation"]["pf"]>1 and x["validation"]["mean"]>=5 and x["validation"]["median"]>=4]
 res={"coverage":[dates[0],dates[-1]],"trials":N_TRIALS,"best_dev":rows[0],"high_payoff_validation_count":len(hp),"high_payoff":hp[:10],"top30":rows}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(res,indent=2,default=str));print(json.dumps({k:v for k,v in res.items() if k!="top30"},indent=2,default=str))
if __name__=="__main__":main()
