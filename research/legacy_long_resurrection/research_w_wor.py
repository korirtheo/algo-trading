"""Joint independent causal recovery: W Power Hour Breakout + WOR Weak-Open Reclaim.

Each strategy is selected independently on 2025 and validated untouched on 2026.
All signals are close-confirmed and fill strictly at next-bar open.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, optuna
from optuna.samplers import TPESampler
import test_full
import research.legacy_long_resurrection.run_baseline as rb
from research.legacy_long_resurrection.causal_execution import Signal
SEED=20261005; N=700; OUT=Path("results/w_wor_resurrection_20261005")

def vwap(b):
 h=b.High.astype(float).to_numpy();l=b.Low.astype(float).to_numpy();c=b.Close.astype(float).to_numpy();v=b.Volume.astype(float).to_numpy()
 cv=np.cumsum(v);cv[cv==0]=1e-9;return np.cumsum(((h+l+c)/3)*v)/cv

def cfg(t,s):
 if s=="W":
  return dict(min_gap=t.suggest_int("min_gap",5,25,step=5),morning_run=t.suggest_int("morning_run",3,15,step=2),
   consol_start=t.suggest_int("consol_start",20,60,step=10),earliest=t.suggest_int("earliest",130,165,step=5),
   latest=t.suggest_int("latest",165,180,step=5),max_range=t.suggest_int("max_range",8,25),
   max_vwap_dev=t.suggest_int("max_vwap_dev",3,12),vol_surge=t.suggest_float("vol_surge",.5,3,step=.5),
   morning_vol=t.suggest_float("morning_vol",.1,.6,step=.1),max_hod_breaks=t.suggest_int("max_hod_breaks",2,8),
   target=t.suggest_float("target",8,25,step=1),stop=t.suggest_float("stop",2,8,step=1),trail=t.suggest_float("trail",1,5,step=1),trail_activate=t.suggest_float("trail_activate",2,10,step=1),minutes=60)
 return dict(min_gap=t.suggest_int("min_gap",0,30,step=5),open_below=t.suggest_float("open_below",.5,5,step=.5),
  ext_pm=t.suggest_float("ext_pm",0,12,step=1),entry_candle=t.suggest_int("entry_candle",3,12),
  target=t.suggest_float("target",10,35,step=2.5),stop=t.suggest_float("stop",3,10,step=1),trail=t.suggest_float("trail",1,7,step=1),trail_activate=t.suggest_float("trail_activate",2,14,step=2),minutes=t.suggest_int("minutes",30,120,step=15))

def sigW(b,gap,c):
 if gap<c["min_gap"] or len(b)<=c["latest"] or c["consol_start"]>=c["earliest"]:return None
 h=b.High.astype(float).to_numpy();l=b.Low.astype(float).to_numpy();cl=b.Close.astype(float).to_numpy();o=b.Open.astype(float).to_numpy();vol=b.Volume.astype(float).to_numpy();vw=vwap(b)
 if o[0]<=0 or (max(h[:30])/o[0]-1)*100<c["morning_run"]:return None
 ch=max(h[c["consol_start"]:c["earliest"]]);lo=min(l[c["consol_start"]:c["earliest"]])
 if ch<=0 or (ch-lo)/ch*100>c["max_range"]:return None
 sl=slice(c["consol_start"],c["earliest"])
 if np.max(np.abs(cl[sl]-vw[sl])/np.maximum(vw[sl],1e-9)*100)>c["max_vwap_dev"]:return None
 hod=h[0];br=0
 for x in h[1:c["earliest"]]:
  if x>hod:br+=1;hod=x
 if br>c["max_hod_breaks"]:return None
 av=np.mean(vol[sl]);mv=max(vol[:30])
 for i in range(c["earliest"],min(c["latest"]+1,len(b))):
  if cl[i]>ch and (av<=0 or vol[i]>=av*c["vol_surge"]) and (mv<=0 or vol[i]>=mv*c["morning_vol"]) and cl[i]>=vw[i]:
   return Signal("W",i,b.index[i],cl[i],"POWER_HOUR_BREAKOUT")
 return None

def sigWOR(b,gap,pm,c):
 if gap<c["min_gap"] or pm<=0 or len(b)<=c["entry_candle"]:return None
 vw=vwap(b);o=float(b.iloc[0].Open)
 if vw[0]<=0 or (o/vw[0]-1)*100>=-c["open_below"]:return None
 i=c["entry_candle"]-1; mx=float(b.High.iloc[:i+1].max())
 if (mx/pm-1)*100<c["ext_pm"]:return None
 return Signal("WOR",i,b.index[i],float(b.iloc[i].Close),"WEAK_OPEN_RECLAIM")

def build(ds,picks,c,s):
 out=[]
 for d in ds:
  for p in picks.get(d,[]):
   b=p["market_hour_candles"].sort_index();gap=float(p["gap_pct"]);pm=float(p["premarket_high"])
   z=sigW(b,gap,c) if s=="W" else sigWOR(b,gap,pm,c)
   if z is None:continue
   f=rb.next_bar_open_fill(z,b,0)
   if f is None:continue
   rb.assert_causal_fill(f);out.append(rb.Candidate(s,p["ticker"],d,gap,pm,b,z,f.fill_bar_index,f.fill_timestamp,f.raw_fill_price))
 out.sort(key=lambda x:(x.fill_ts,x.ticker));return out

def ev(ds,picks,c,s):
 old=rb.DEFAULTS;rb.DEFAULTS={s:{"min_gap":c["min_gap"],"signal":{},"exit":{k:c[k] for k in ["target","stop","trail","trail_activate","minutes"]}}}
 try:return rb.stats(rb.replay(build(ds,picks,c,s),rb.START_EQUITY)[0],rb.START_EQUITY)
 finally:rb.DEFAULTS=old

def score(m):
 n=m["trades"]
 if n<25:return -1e9+n
 return m["pnl"]+1900*m["mean"]+2300*m["median"]+1800*min(m["pf"],3)+3500*math.log1p(n)-650*abs(m["max_dd"])-7000*max(0,5-m["mean"])-8000*max(0,4-m["median"])

def one(s,dev,val,picks):
 st=optuna.create_study(direction="maximize",sampler=TPESampler(seed=SEED+(s=="WOR"),n_startup_trials=140,multivariate=True),study_name=s)
 def obj(t):
  c=cfg(t,s)
  if s=="W" and c["earliest"]>=c["latest"]:return -1e9
  m=ev(dev,picks,c,s)
  for k,v in m.items():
   if isinstance(v,(int,float)):t.set_user_attr(k,float(v))
  return score(m)
 st.optimize(obj,n_trials=N,show_progress_bar=False)
 rr=sorted(st.trials,key=lambda t:t.value if t.value is not None else -1e99,reverse=True)[:30];rows=[]
 for rank,t in enumerate(rr,1):
  class F:
   def __init__(self,p):self.p=p
   suggest_int=lambda self,n,*a,**k:self.p[n]
   suggest_float=lambda self,n,*a,**k:self.p[n]
  c=cfg(F(t.params),s);rows.append({"rank":rank,"params":c,"development":ev(dev,picks,c,s),"validation":ev(val,picks,c,s)})
 return {"best_dev":rows[0],"high_payoff_validation_count":sum(x["validation"]["trades"]>=10 and x["validation"]["pf"]>1 and x["validation"]["mean"]>=5 and x["validation"]["median"]>=4 for x in rows),"top30":rows}

def main():
 test_full.MIN_GAP_PCT=0;test_full.MIN_PM_VOLUME=250000
 dates,picks=test_full.load_all_picks(["stored_data_combined"]);dev=[d for d in dates if d.startswith("2025-")];val=[d for d in dates if d.startswith("2026-")]
 res={"coverage":[dates[0],dates[-1]],"trials_each":N}
 for s in ["W","WOR"]:res[s]=one(s,dev,val,picks)
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(res,indent=2,default=str));print(json.dumps({k:(v if k not in ["W","WOR"] else {"best_dev":v["best_dev"],"high_payoff_validation_count":v["high_payoff_validation_count"]}) for k,v in res.items()},indent=2,default=str))
if __name__=="__main__":main()
