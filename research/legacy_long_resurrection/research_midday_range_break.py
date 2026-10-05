"""Causal recovery study for legacy M — Midday Range Break.

2025 selects parameters; 2026 remains untouched validation.
Signal:
1) morning spike over a fixed completed morning window;
2) fixed midday consolidation window with tight range and volume contraction;
3) first close above the completed consolidation high;
4) fill at NEXT bar open.

The legacy partial-exit mechanics are intentionally not imported. This study tests
whether the entry mechanism itself supports high-payoff economics under the current
execution engine.
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

SEED=20261005; N_TRIALS=800
OUT=Path("results/midday_range_break_resurrection_20261005")

def cfg(t):
    return dict(
      min_gap=t.suggest_int("min_gap",8,30,step=2),
      morning_spike=t.suggest_int("morning_spike",4,18,step=2),
      morning_candles=t.suggest_int("morning_candles",20,45,step=5),
      range_start=t.suggest_int("range_start",40,75,step=5),
      consol_len=t.suggest_int("consol_len",15,45,step=5),
      max_range=t.suggest_int("max_range",3,12),
      vol_ratio=t.suggest_float("vol_ratio",0.3,0.9,step=.1),
      breakout_rvol=t.suggest_float("breakout_rvol",0.8,3.0,step=.2),
      latest=t.suggest_int("latest",100,170,step=10),
      target=t.suggest_float("target",10,35,step=2.5),
      stop=t.suggest_float("stop",3,10,step=1),
      trail=t.suggest_float("trail",2,8,step=1),
      trail_activate=t.suggest_float("trail_activate",5,18,step=1),
      minutes=t.suggest_int("minutes",30,150,step=15))

def m_signal(b,gap,c):
    if gap<c["min_gap"] or len(b)<30:return None
    o=b["Open"].astype(float).to_numpy();h=b["High"].astype(float).to_numpy()
    l=b["Low"].astype(float).to_numpy();cl=b["Close"].astype(float).to_numpy()
    v=b["Volume"].astype(float).to_numpy(); n=len(b)
    rs=c["range_start"]; ce=rs+c["consol_len"]
    if ce>=n-1 or c["morning_candles"]>rs:return None
    op=o[0]
    if op<=0:return None
    if (np.max(h[:c["morning_candles"]])/op-1)*100<c["morning_spike"]:return None
    ch=float(np.max(h[rs:ce])); lo=float(np.min(l[rs:ce]))
    if ch<=0:return None
    if (ch-lo)/ch*100>c["max_range"]:return None
    mav=float(np.mean(v[:min(30,n)])); cav=float(np.mean(v[rs:ce]))
    if mav>0 and cav/mav>c["vol_ratio"]:return None
    end=min(n,c["latest"])
    for i in range(ce,end):
        prev=max(ce,i-10)
        av=float(np.mean(v[prev:i])) if i>prev else 0
        if cl[i]>ch and (av<=0 or v[i]>=av*c["breakout_rvol"]):
            return Signal("M",i,b.index[i],cl[i],"MIDDAY_RANGE_BREAK")
    return None

def build(dates,picks,c):
    out=[]
    for d in dates:
      for p in picks.get(d,[]):
        b=p["market_hour_candles"].sort_index()
        s=m_signal(b,float(p["gap_pct"]),c)
        if s is None:continue
        f=rb.next_bar_open_fill(s,b,0)
        if f is None:continue
        rb.assert_causal_fill(f)
        out.append(rb.Candidate("M",p["ticker"],d,float(p["gap_pct"]),float(p["premarket_high"]),
                  b,s,f.fill_bar_index,f.fill_timestamp,f.raw_fill_price))
    out.sort(key=lambda x:(x.fill_ts,x.ticker));return out

def evaluate(dates,picks,c):
    old=rb.DEFAULTS
    rb.DEFAULTS={"M":{"min_gap":c["min_gap"],"signal":{},"exit":{
      k:c[k] for k in ["target","stop","trail","trail_activate","minutes"]}}}
    try:
      cs=build(dates,picks,c);ts,_=rb.replay(cs,rb.START_EQUITY)
      return rb.stats(ts,rb.START_EQUITY)
    finally:rb.DEFAULTS=old

def score(m):
    n=m["trades"]
    if n<25:return -1e9+n
    return (m["pnl"]+1800*m["mean"]+2200*m["median"]+1800*min(m["pf"],3)
      +3500*math.log1p(n)-650*abs(m["max_dd"])
      -7000*max(0,5-m["mean"])-8000*max(0,4-m["median"]))

def main():
    test_full.MIN_GAP_PCT=5;test_full.MIN_PM_VOLUME=250_000
    dates,picks=test_full.load_all_picks(["stored_data_combined"])
    dev=[d for d in dates if d.startswith("2025-")];val=[d for d in dates if d.startswith("2026-")]
    st=optuna.create_study(direction="maximize",
      sampler=TPESampler(seed=SEED,n_startup_trials=160,multivariate=True),study_name="M_corrected")
    def obj(t):
      c=cfg(t);m=evaluate(dev,picks,c)
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
      c=cfg(F(t.params));rows.append({"rank":rank,"params":c,
          "development":evaluate(dev,picks,c),"validation":evaluate(val,picks,c)})
    hp=[x for x in rows if x["validation"]["trades"]>=8 and x["validation"]["pf"]>1
        and x["validation"]["mean"]>=5 and x["validation"]["median"]>=4]
    res={"coverage":[dates[0],dates[-1]],"trials":N_TRIALS,"selection":"2025",
         "validation":"2026 untouched","best_dev":rows[0],
         "high_payoff_validation_count":len(hp),"high_payoff":hp[:10],"top30":rows}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(res,indent=2,default=str))
    print(json.dumps({k:v for k,v in res.items() if k!="top30"},indent=2,default=str))
if __name__=="__main__":main()
