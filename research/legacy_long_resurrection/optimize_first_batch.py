"""Causal 2025 development search for the first recovered legacy-long batch.

2024 in the old GitHub archive is materially incomplete, so it is NOT used for
selection.  Parameters are selected on 2025 only.  Jan-Feb 2026 is evaluated
afterward as validation and never enters the objective.
"""
from __future__ import annotations
import copy, json, math
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import test_full
import research.legacy_long_resurrection.run_baseline as rb

N_TRIALS=160
SEED=20261004


def cfg_for(trial,s):
    c=copy.deepcopy(rb.DEFAULTS[s])
    if s=="P":
        c["min_gap"]=trial.suggest_int("min_gap",10,40,step=5)
        c["signal"].update(
          confirm_above=trial.suggest_int("confirm_above",1,4),
          confirm_window=trial.suggest_int("confirm_window",2,8),
          pullback_pct=trial.suggest_float("pullback_pct",1,6,step=.5),
          pullback_timeout=trial.suggest_int("pullback_timeout",5,30,step=5),
          max_entry_candle=trial.suggest_int("max_entry_candle",15,120,step=15))
        c["exit"].update(target=trial.suggest_float("target",3,15,step=1),
                         stop=trial.suggest_float("stop",2,10,step=1),
                         minutes=trial.suggest_int("minutes",10,60,step=10))
    elif s=="D":
        c["min_gap"]=trial.suggest_int("min_gap",8,30,step=2)
        c["signal"].update(
          min_spike_pct=trial.suggest_int("min_spike_pct",3,15),
          spike_window=trial.suggest_categorical("spike_window",[5,10,15,20]),
          dip_pct=trial.suggest_int("dip_pct",5,15),
          entry_mode=trial.suggest_categorical("entry_mode",["vwap","5candle"]),
          max_entry_candle=trial.suggest_int("max_entry_candle",15,60,step=5))
        c["exit"].update(target=trial.suggest_float("target",3,12,step=1),
                         stop=trial.suggest_float("stop",3,10,step=1),
                         trail=trial.suggest_float("trail",2,6,step=1),
                         trail_activate=trial.suggest_float("trail_activate",2,6,step=1),
                         minutes=trial.suggest_int("minutes",15,60,step=5))
    elif s=="V":
        c["min_gap"]=trial.suggest_int("min_gap",8,26,step=2)
        c["signal"].update(
          min_below_candles=trial.suggest_int("min_below_candles",2,15),
          min_below_pct=trial.suggest_int("min_below_pct",0,5),
          vol_spike_ratio=trial.suggest_float("vol_spike_ratio",1,4,step=.5),
          max_entry_candle=trial.suggest_int("max_entry_candle",30,120,step=10))
        c["exit"].update(target=trial.suggest_float("target",3,12,step=1),
                         stop=trial.suggest_float("stop",3,10,step=1),
                         trail=trial.suggest_float("trail",2,6,step=1),
                         trail_activate=trial.suggest_float("trail_activate",2,6,step=1),
                         minutes=trial.suggest_int("minutes",15,60,step=5))
    elif s=="B":
        c["min_gap"]=trial.suggest_int("min_gap",5,40,step=5)
        lo=trial.suggest_float("min_dip_pct",.5,5,step=.5)
        hi=trial.suggest_float("max_dip_pct",5,25,step=2.5)
        c["signal"].update(
          min_dip_pct=lo,max_dip_pct=max(hi,lo+.5),
          earliest_candle=trial.suggest_int("earliest_candle",2,10),
          latest_candle=trial.suggest_int("latest_candle",20,120,step=10),
          vol_surge_mult=trial.suggest_float("vol_surge_mult",.5,4,step=.5),
          min_body_pct=trial.suggest_float("min_body_pct",0,3,step=.5),
          require_above_vwap=trial.suggest_categorical("require_above_vwap",[True,False]))
        c["exit"].update(target=trial.suggest_float("target",3,25,step=1),
                         stop=trial.suggest_float("stop",2,12,step=1),
                         trail=trial.suggest_float("trail",1,8,step=.5),
                         trail_activate=trial.suggest_float("trail_activate",1,10,step=1),
                         minutes=trial.suggest_int("minutes",15,120,step=15))
    return c


def score(m):
    if m["trades"]<30: return -1e9 + m["trades"]
    # Economic score: PnL dominates; PF/central tendency reward quality; DD penalized.
    return (m["pnl"] + 1000*min(m["pf"],3.0)
            + 200*m["mean"] + 100*m["median"] - 100*abs(m["max_dd"]))


def evaluate(s,cfg,dates,picks):
    old=rb.DEFAULTS
    try:
        rb.DEFAULTS={s:cfg}
        cs=rb.build_candidates(dates,picks)
        ts,_=rb.replay(cs,rb.START_EQUITY)
        return rb.stats(ts,rb.START_EQUITY),len(cs)
    finally:
        rb.DEFAULTS=old


def main():
    test_full.MIN_GAP_PCT=5.0; test_full.MIN_PM_VOLUME=250_000
    dates,picks=test_full.load_all_picks(["stored_data_combined"])
    dev=[d for d in dates if d.startswith("2025-")]
    val=[d for d in dates if d.startswith("2026-")]
    out=Path("results/legacy_long_resurrection_baseline_20261004")
    report={"development":"2025 only","validation":"2026-01-01 through archive end",
            "archive_end":dates[-1],"trials_per_strategy":N_TRIALS,"strategies":{}}

    for s in ["P","D","V","B"]:
        def objective(trial):
            cfg=cfg_for(trial,s)
            m,n=evaluate(s,cfg,dev,picks)
            for k,v in m.items():
                if isinstance(v,(int,float)): trial.set_user_attr(k,float(v))
            trial.set_user_attr("candidates",n)
            return score(m)
        study=optuna.create_study(direction="maximize",
            sampler=TPESampler(seed=SEED,n_startup_trials=48,multivariate=True),
            study_name=f"legacy_resurrection_{s}_2025")
        study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=False)

        ranked=sorted([t for t in study.trials if t.value is not None],
                      key=lambda t:t.value,reverse=True)[:20]
        rows=[]
        for rank,t in enumerate(ranked,1):
            class Fixed:
                def __init__(self,p): self.p=p
                def _g(self,n): return self.p[n]
                suggest_int=lambda self,n,*a,**k:self._g(n)
                suggest_float=lambda self,n,*a,**k:self._g(n)
                suggest_categorical=lambda self,n,*a,**k:self._g(n)
            cfg=cfg_for(Fixed(t.params),s)
            vm,vn=evaluate(s,cfg,val,picks)
            rows.append({"rank":rank,"dev_score":t.value,"params":t.params,
                         "dev":{k:t.user_attrs.get(k) for k in ["trades","pnl","pf","wr","mean","median","max_dd"]},
                         "validation":vm,"validation_candidates":vn})
        survivors=[r for r in rows if r["validation"]["pnl"]>0 and r["validation"]["pf"]>1]
        report["strategies"][s]={
          "best_development":rows[0],
          "top20_validation_positive_pf_gt_1":len(survivors),
          "top20":rows,
        }
        print(s,json.dumps(report["strategies"][s]["best_development"],indent=2))
        print("survivors",len(survivors))

    (out/"first_batch_causal_search.json").write_text(json.dumps(report,indent=2,default=str))


if __name__=="__main__": main()
