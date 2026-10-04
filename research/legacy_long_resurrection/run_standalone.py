"""Standalone viability replay for recovered legacy long strategies.

Each strategy receives a fresh $25k account, so one failed strategy cannot starve
another of capital.  Also runs a fresh-2026 slice independently.
"""
import json
from pathlib import Path
import pandas as pd
import test_full
from research.legacy_long_resurrection.run_baseline import (
    START_EQUITY, DEFAULTS, build_candidates, replay, stats, grouped
)


def main():
    test_full.MIN_GAP_PCT=5.0
    test_full.MIN_PM_VOLUME=250_000
    dates,picks=test_full.load_all_picks(["stored_data_combined"])
    candidates=build_candidates(dates,picks)

    result={
      "coverage":{"first":dates[0] if dates else None,"last":dates[-1] if dates else None,"days":len(dates)},
      "candidate_counts_by_strategy":{},
      "candidate_counts_by_year":{},
      "strategies":{},
    }
    for s in DEFAULTS:
        sc=[x for x in candidates if x.strategy==s]
        result["candidate_counts_by_strategy"][s]=len(sc)
        for x in sc:
            result["candidate_counts_by_year"].setdefault(x.date[:4],0)
            result["candidate_counts_by_year"][x.date[:4]]+=1

        all_trades,_=replay(sc,START_EQUITY)
        t26,_=replay([x for x in sc if x.date.startswith("2026-")],START_EQUITY)
        result["strategies"][s]={
          "all":stats(all_trades,START_EQUITY),
          "fresh_2026":stats(t26,START_EQUITY),
          "year":grouped(all_trades)["year"],
          "quarter":grouped(all_trades)["quarter"],
          "month":grouped(all_trades)["month"],
        }

    out=Path("results/legacy_long_resurrection_baseline_20261004")
    out.mkdir(parents=True,exist_ok=True)
    (out/"standalone_summary.json").write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__":
    main()
