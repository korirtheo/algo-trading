"""Deep causal economic-state research for recovered G and R.

This is deliberately NOT another raw parameter sweep.
- Build every eligible causal state for G/R from 2024-2026 archive.
- Attach forward path labels only for model training/evaluation (never features).
- Train entry-state models on 2024 only.
- Select score threshold + exit policy on 2025 only under the real execution engine.
- Replay the frozen 2025-selected policy on 2026 with no 2026 mathematical input.

Core labels mirror the successful RTH competing-risk idea, inverted for longs:
+10/+15% MFE before -8% adverse excursion, plus continuous MFE/MAE diagnostics.
"""
from __future__ import annotations
import json, math, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import test_full
import research.legacy_long_resurrection.run_baseline as rb
from research.legacy_long_resurrection.causal_execution import Signal

warnings.filterwarnings("ignore")
SEED=20261005; N_POLICY_TRIALS=400
OUT=Path("results/gr_economic_state_engine_20261005")
OUT.mkdir(parents=True,exist_ok=True)
def progress(msg, **payload):
 print(json.dumps({"phase":msg, **payload}, default=str), flush=True)
def checkpoint(name, obj):
 p=OUT/f"{name}.json"; p.write_text(json.dumps(obj,indent=2,default=str))
 progress("checkpoint_saved", file=str(p))
FEATURES=[
 "gap","price","body","range_pct","close_pos","upper_wick","lower_wick",
 "vol_ratio1","vol_ratio5","cum_dvol_m","vwap_dist","open_ext","pm_dist",
 "hod_break","green_streak","ret1","ret2","ret5","accel","efficiency",
 "pullback_hod","bounce5","time_since_hod","bar_index"
]

def _vwap(b):
 h=b.High.astype(float).to_numpy();l=b.Low.astype(float).to_numpy();c=b.Close.astype(float).to_numpy();v=b.Volume.astype(float).to_numpy()
 cv=np.cumsum(v);cv[cv==0]=1e-9
 return np.cumsum(((h+l+c)/3)*v)/cv

def _ret(a,b):
 return (a/b-1)*100 if b and b>0 else 0.0

def feat(b,i,gap,pm):
 o=b.Open.astype(float).to_numpy();h=b.High.astype(float).to_numpy();l=b.Low.astype(float).to_numpy();c=b.Close.astype(float).to_numpy();v=b.Volume.astype(float).to_numpy()
 vw=_vwap(b); px=c[i]; op=o[i]
 body=_ret(px,op); rng=(h[i]-l[i])/max(op,1e-9)*100
 cp=(px-l[i])/max(h[i]-l[i],1e-9)
 uw=(h[i]-max(px,op))/max(op,1e-9)*100; lw=(min(px,op)-l[i])/max(op,1e-9)*100
 vr1=v[i]/max(v[i-1],1) if i>0 else 1
 av5=np.mean(v[max(0,i-5):i]) if i>0 else v[i]; vr5=v[i]/max(av5,1)
 cum=float(np.sum(c[:i+1]*v[:i+1]))/1e6
 ret1=_ret(c[i],c[i-1]) if i>=1 else 0; ret2=_ret(c[i],c[i-2]) if i>=2 else ret1;ret5=_ret(c[i],c[i-5]) if i>=5 else ret2
 prev_ret=_ret(c[i-1],c[i-2]) if i>=2 else 0;acc=ret1-prev_ret
 path=np.diff(c[:i+1])/np.maximum(c[:i],1e-9) if i>=1 else np.array([0.])
 eff=abs(_ret(c[i],o[0]))/(np.sum(np.abs(path))*100+1e-9)
 prior_hod=float(np.max(h[:i])) if i>0 else h[i];hod_break=_ret(h[i],prior_hod)
 pull=max(0,_ret(prior_hod,px))
 lo5=float(np.min(l[max(0,i-5):i+1])); bounce=_ret(px,lo5)
 hod_i=int(np.argmax(h[:i+1])); tsh=i-hod_i
 streak=0
 for j in range(i,-1,-1):
  if c[j]>o[j]:streak+=1
  else:break
 return dict(gap=gap,price=px,body=body,range_pct=rng,close_pos=cp,upper_wick=uw,lower_wick=lw,
  vol_ratio1=vr1,vol_ratio5=vr5,cum_dvol_m=cum,vwap_dist=_ret(px,vw[i]),open_ext=_ret(px,o[0]),
  pm_dist=_ret(px,pm) if pm>0 else 0,hod_break=hod_break,green_streak=streak,ret1=ret1,ret2=ret2,ret5=ret5,
  accel=acc,efficiency=eff,pullback_hod=pull,bounce5=bounce,time_since_hod=tsh,bar_index=i)

def labels(b,signal_i,horizon_min=150):
 fi=signal_i+1
 if fi>=len(b):return None
 entry=float(b.iloc[fi].Open)
 if entry<=0:return None
 end_ts=b.index[fi]+pd.Timedelta(minutes=horizon_min)
 sub=b.iloc[fi:][b.index[fi:]<=end_ts]
 if sub.empty:return None
 hs=sub.High.astype(float).to_numpy();ls=sub.Low.astype(float).to_numpy()
 mfe=float((np.max(hs)/entry-1)*100); mae=float((np.min(ls)/entry-1)*100)
 t_mfe=int(np.argmax(hs)); t_mae=int(np.argmin(ls))
 def before(up,down):
  ui=np.where(hs>=entry*(1+up/100))[0];di=np.where(ls<=entry*(1-down/100))[0]
  if len(ui)==0:return 0
  return int(len(di)==0 or ui[0]<di[0])
 return dict(mfe=mfe,mae=mae,time_mfe=t_mfe,time_mae=t_mae,
  up10_before_dn8=before(10,8),up15_before_dn8=before(15,8),up20_before_dn10=before(20,10))

def states_for_pick(p,d):
 b=p["market_hour_candles"].sort_index();n=len(b)
 if n<5:return []
 gap=float(p["gap_pct"]);pm=float(p.get("premarket_high",0) or 0)
 o=b.Open.astype(float).to_numpy();h=b.High.astype(float).to_numpy();c=b.Close.astype(float).to_numpy();v=b.Volume.astype(float).to_numpy()
 out=[]
 # G surface: opening continuation states, not just the legacy candle-2 point.
 if gap>=10:
  for i in range(1,min(n-1,8)):
   if c[i]<=o[i]:continue
   streak=0
   for j in range(i,-1,-1):
    if c[j]>o[j]:streak+=1
    else:break
   if streak<2:continue
   if h[i]<=np.max(h[:i]):continue
   f=feat(b,i,gap,pm);lab=labels(b,i,120)
   if lab:out.append(dict(strategy="G",date=d,ticker=p["ticker"],i=i,bars=b,pm=pm,**f,**lab))
 # R surface: pullback from established HOD + green re-acceleration with >=1x recent volume.
 if gap>=10 and n>=25:
  for i in range(20,min(n-1,180)):
   if c[i]<=o[i] or o[i]<=0:continue
   ph=float(np.max(h[:i]))
   if ph<=0 or (ph-c[i])/ph*100<2:continue
   av=float(np.mean(v[i-20:i]))
   if av<=0 or v[i]/av<1.0:continue
   if (c[i]/o[i]-1)*100<.5:continue
   f=feat(b,i,gap,pm);lab=labels(b,i,150)
   if lab:out.append(dict(strategy="R",date=d,ticker=p["ticker"],i=i,bars=b,pm=pm,**f,**lab))
 return out

def build_surface(dates,picks):
 rows=[]
 for di,d in enumerate(dates,1):
  for p in picks.get(d,[]):rows.extend(states_for_pick(p,d))
  if di % 25 == 0 or di == len(dates):
   progress("surface_progress", days_done=di, days_total=len(dates), states=len(rows))
 return rows

def X(rows):return pd.DataFrame([{k:r[k] for k in FEATURES} for r in rows]).replace([np.inf,-np.inf],0).fillna(0)

def train_models(train_rows,hold_rows):
 xt=X(train_rows);xh=X(hold_rows)
 y10=np.array([r["up10_before_dn8"] for r in train_rows]);y15=np.array([r["up15_before_dn8"] for r in train_rows])
 m10=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.06,l2_regularization=1,random_state=SEED).fit(xt,y10)
 m15=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.06,l2_regularization=1,random_state=SEED+1).fit(xt,y15)
 mm=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,learning_rate=.06,l2_regularization=1,random_state=SEED+2).fit(xt,[r["mfe"] for r in train_rows])
 ma=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,learning_rate=.06,l2_regularization=1,random_state=SEED+3).fit(xt,[-r["mae"] for r in train_rows])
 def pred(rows):
  xx=X(rows);p10=m10.predict_proba(xx)[:,1];p15=m15.predict_proba(xx)[:,1];pmfe=np.maximum(0,mm.predict(xx));pmae=np.maximum(0,ma.predict(xx))
  # executable-opportunity score: probability-weighted upside less predicted adverse path.
  return .35*p10+.45*p15+.012*pmfe-.012*pmae
 ph=pred(hold_rows)
 auc10=roc_auc_score([r["up10_before_dn8"] for r in hold_rows],m10.predict_proba(xh)[:,1]) if len(set(r["up10_before_dn8"] for r in hold_rows))>1 else None
 auc15=roc_auc_score([r["up15_before_dn8"] for r in hold_rows],m15.predict_proba(xh)[:,1]) if len(set(r["up15_before_dn8"] for r in hold_rows))>1 else None
 # compact causal feature importance on holdout for +15 before -8.
 try:
  pi=permutation_importance(m15,xh,[r["up15_before_dn8"] for r in hold_rows],n_repeats=4,random_state=SEED,scoring="roc_auc")
  imp=sorted(zip(FEATURES,pi.importances_mean),key=lambda z:z[1],reverse=True)[:10]
 except Exception:imp=[]
 return pred,dict(auc10=auc10,auc15=auc15,top_features=imp)

def candidate(r,s):
 b=r["bars"];sig=Signal(s,r["i"],b.index[r["i"]],float(b.iloc[r["i"]].Close),"ECON_STATE")
 f=rb.next_bar_open_fill(sig,b,0)
 if f is None:return None
 rb.assert_causal_fill(f)
 return rb.Candidate(s,r["ticker"],r["date"],float(r["gap"]),float(r["pm"]),b,sig,f.fill_bar_index,f.fill_timestamp,f.raw_fill_price)

def replay_selected(rows,scores,s,thr,ex):
 chosen=[];seen=set()
 order=np.argsort([r["bars"].index[r["i"]].value for r in rows])
 for q in order:
  r=rows[q]
  if scores[q]<thr:continue
  key=(r["date"],r["ticker"])
  if key in seen:continue
  c=candidate(r,s)
  if c:chosen.append(c);seen.add(key)
 chosen.sort(key=lambda c:(c.fill_ts,c.ticker))
 old=rb.DEFAULTS;rb.DEFAULTS={s:{"min_gap":0,"signal":{},"exit":ex}}
 try:
  tr,_=rb.replay(chosen,rb.START_EQUITY);return rb.stats(tr,rb.START_EQUITY),tr
 finally:rb.DEFAULTS=old

def policy_search(sel_rows,sel_scores,s):
 qs=np.unique(np.quantile(sel_scores,np.linspace(.50,.95,19)))
 st=optuna.create_study(direction="maximize",sampler=TPESampler(seed=SEED+(s=="R"),n_startup_trials=80))
 def obj(t):
  thr=float(qs[t.suggest_int("q_idx",0,len(qs)-1)])
  ex=dict(target=t.suggest_float("target",10,35,step=2.5),stop=t.suggest_float("stop",3,10,step=1),
   trail=t.suggest_float("trail",1,6,step=1),trail_activate=t.suggest_float("trail_activate",4,16,step=2),
   minutes=t.suggest_int("minutes",45,150,step=15))
  m,_=replay_selected(sel_rows,sel_scores,s,thr,ex);n=m["trades"]
  if n<15:return -1e9+n
  return m["pnl"]+1600*m["mean"]+2200*m["median"]+1800*min(m["pf"],3)+2500*math.log1p(n)-700*abs(m["max_dd"])-5000*max(0,4-m["mean"])
def cb(study, trial):
  n=trial.number+1
  if n % 50 == 0 or n == N_POLICY_TRIALS:
   progress("policy_search_progress", strategy=s, trials_done=n, best_value=study.best_value if study.best_trial else None)
 st.optimize(obj,n_trials=N_POLICY_TRIALS,show_progress_bar=False,callbacks=[cb])
 p=st.best_params.copy();thr=float(qs[p.pop("q_idx")]);ex=p
 return thr,ex

def bins(rows):
 out={}
 for f in ["gap","body","vol_ratio1","vol_ratio5","vwap_dist","open_ext","pm_dist","efficiency","pullback_hod","bounce5","time_since_hod","bar_index"]:
  vals=np.array([r[f] for r in rows],float)
  try:cats=pd.qcut(vals,4,duplicates="drop")
  except:continue
  z=pd.DataFrame({"bin":cats,"y":[r["up15_before_dn8"] for r in rows],"mfe":[r["mfe"] for r in rows],"mae":[r["mae"] for r in rows]})
  g=z.groupby("bin",observed=True).agg(n=("y","size"),hit15=("y","mean"),mfe=("mfe","mean"),mae=("mae","mean")).reset_index()
  out[f]=g.astype({"bin":str}).to_dict("records")
 return out

def main():
 test_full.MIN_GAP_PCT=5;test_full.MIN_PM_VOLUME=250000
 progress("load_start")
 dates,picks=test_full.load_all_picks(["stored_data_combined"])
 progress("load_complete", days=len(dates), coverage=[dates[0],dates[-1]])
 progress("surface_build_start")
 surf=build_surface(dates,picks)
 checkpoint("surface_counts", {"coverage":[dates[0],dates[-1]],"total_states":len(surf),"G":sum(r["strategy"]=="G" for r in surf),"R":sum(r["strategy"]=="R" for r in surf)})
 res={"coverage":[dates[0],dates[-1]],"method":"2024 model train -> 2025 policy select -> 2026 replay"}
 for s in ["G","R"]:
  progress("strategy_start", strategy=s)
  rr=[r for r in surf if r["strategy"]==s];tr=[r for r in rr if r["date"].startswith("2024-")];sel=[r for r in rr if r["date"].startswith("2025-")];oos=[r for r in rr if r["date"].startswith("2026-")]
  progress("split_ready", strategy=s, train_2024=len(tr), select_2025=len(sel), oos_2026=len(oos))
  pred,diag=train_models(tr,sel)
  checkpoint(f"{s.lower()}_model_diagnostics", {"states":{"2024":len(tr),"2025":len(sel),"2026":len(oos)},"model":diag})
  progress("model_complete", strategy=s, auc10=diag.get("auc10"), auc15=diag.get("auc15"), top_features=diag.get("top_features"))
  ss=pred(sel);oo=pred(oos)
  progress("policy_search_start", strategy=s, trials=N_POLICY_TRIALS)
  thr,ex=policy_search(sel,ss,s)
  checkpoint(f"{s.lower()}_selected_policy", {"threshold":thr,"exit":ex})
  progress("policy_selected", strategy=s, threshold=thr, exit=ex)
  m25,t25=replay_selected(sel,ss,s,thr,ex);m26,t26=replay_selected(oos,oo,s,thr,ex)
  progress("replay_complete", strategy=s, selection_2025=m25, oos_2026=m26)
  res[s]={"states":{"2024":len(tr),"2025":len(sel),"2026":len(oos)},
   "base_labels":{"2024_hit10":np.mean([r["up10_before_dn8"] for r in tr]),"2024_hit15":np.mean([r["up15_before_dn8"] for r in tr]),
                  "2025_hit10":np.mean([r["up10_before_dn8"] for r in sel]),"2025_hit15":np.mean([r["up15_before_dn8"] for r in sel])},
   "model":diag,"threshold":thr,"exit":ex,"selection_2025":m25,"oos_2026":m26,
   "diagnostic_bins_2024":bins(tr)}
  checkpoint(f"{s.lower()}_complete", res[s])
 checkpoint("summary",res)
 print(json.dumps({k:v if k not in ["G","R"] else {q:v[q] for q in ["states","base_labels","model","threshold","exit","selection_2025","oos_2026"]} for k,v in res.items()},indent=2,default=str))
if __name__=="__main__":main()
