"""Run clean unoptimized standalone baselines for O/K/C/S/N/X."""
from __future__ import annotations
import json
from pathlib import Path
import test_full
from legacy_resurrection_next import engine as e

OUT=Path("results/legacy_resurrection_second_batch_20261004")

def main():
    test_full.MIN_GAP_PCT=5.0
    test_full.MIN_PM_VOLUME=250_000
    dates,picks=test_full.load_all_picks(["stored_data_combined"])
    cands=e.build_candidates(dates,picks)
    report={"study":"legacy-long second batch clean unoptimized baseline",
            "coverage":{"first":dates[0] if dates else None,"last":dates[-1] if dates else None,"days":len(dates)},
            "execution":"canonical volatility-adjusted 30%/$100k/5%-8%-15% caps + dynamic impact; next-bar-open entries",
            "strategies":{}}
    for s in e.DEFAULTS:
        sc=[x for x in cands if x.strategy==s]
        all_t,_=e.replay(sc)
        v26=[x for x in sc if x.date.startswith("2026-")]
        t26,_=e.replay(v26)
        report["strategies"][s]={
          "candidates":len(sc),
          "all":e.stats(all_t),
          "fresh_2026":e.stats(t26),
          "year":e.grouped(all_t)["year"],
          "quarter":e.grouped(all_t)["quarter"],
        }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"baseline.json").write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2,default=str))

if __name__=="__main__":main()
