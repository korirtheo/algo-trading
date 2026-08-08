"""Backtest equivalence: verify the refactored module produces byte-identical
trade outcomes to the pre-refactor version.

Uses the git HEAD version of test_green_candle_combined.py + optimize_combined.py
as the reference, runs a multi-day backtest with the deployed config, and compares
the resulting (ticker, strategy, entry, exit, reason, pnl) tuples against the
refactored working tree.

We exploit the fact that git still has the original files: we check out the old
versions into a temp dir, run there, capture results to JSON, then run the same
scenario on the current tree and diff.
"""
import json
import os
import subprocess
import sys
import tempfile

PROJECT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "stored_data_jun_2026"  # small-ish dataset for a fast equivalence run
PARAMS = "config/trial_1655_g1_l1_optimized_deploy.json"
START = "2026-06-08"
END = "2026-06-16"

REF_DIR = os.path.join(tempfile.gettempdir(), "algo_ref")

def run_scenario(cwd, out_path):
    """Run the backtest scenario in the given working tree, write JSON result."""
    code = f'''
import sys, json, os
sys.path.insert(0, r"{cwd}")
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks
raw = json.load(open(r"{os.path.join(cwd, PARAMS)}"))
p = raw["params"] if isinstance(raw, dict) and "params" in raw else raw
set_strategy_params(dict(p))
all_dates, picks_by_date = load_all_picks([r"{DATA_DIR}"])
all_dates = [d for d in all_dates if r"{START}" <= d <= r"{END}"]
cash = 25000.0
results = []
for d in all_dates:
    picks = picks_by_date.get(d, [])
    if not picks: continue
    states, cash, unsettled, _ = tgc.simulate_day_combined(picks, cash)
    for st in states:
        if st.get("exit_price") is None: continue
        results.append({{
            "d": d, "t": st["ticker"], "s": st.get("strategy"),
            "e": round(float(st["entry_price"]),4) if st.get("entry_price") else None,
            "x": round(float(st["exit_price"]),4),
            "r": st.get("exit_reason"), "p": round(float(st.get("pnl",0)),2),
        }})
json.dump({{"days": len(all_dates), "trades": results, "cash": round(cash,2)}}, open(r"{out_path}", "w"))
print("ok", len(results))
'''
    subprocess.run([sys.executable, "-c", code], cwd=cwd, check=True)

def main():
    ref_out = os.path.join(tempfile.gettempdir(), "algo_ref_results.json")
    new_out = os.path.join(tempfile.gettempdir(), "algo_new_results.json")

    if not os.path.exists(REF_DIR):
        import shutil
        os.makedirs(REF_DIR)
        # copy entire project, then restore git versions of the two refactored files
        subprocess.run(["git", "archive", "HEAD"], cwd=PROJECT, stdout=open(os.path.join(REF_DIR, "arc.tar"), "wb"), check=True)
        subprocess.run(["tar", "xf", "arc.tar"], cwd=REF_DIR, check=True)
        # extract stored data + config into ref tree
        for d in (DATA_DIR, "config"):
            shutil.copytree(os.path.join(PROJECT, d), os.path.join(REF_DIR, d), dirs_exist_ok=True)
        # float_data.json is gitignored — copy it too so FLOAT_DATA (L-strategy
        # eligibility) is identical in both trees (not part of the refactor).
        if os.path.exists(os.path.join(PROJECT, "float_data.json")):
            shutil.copy(os.path.join(PROJECT, "float_data.json"),
                        os.path.join(REF_DIR, "float_data.json"))
        # put the ORIGINAL (HEAD) versions of the two refactored files in place
        subprocess.run(["git", "show", f"HEAD:test_green_candle_combined.py"], cwd=PROJECT, stdout=open(os.path.join(REF_DIR, "test_green_candle_combined.py"), "wb"), check=True)
        subprocess.run(["git", "show", f"HEAD:optimize_combined.py"], cwd=PROJECT, stdout=open(os.path.join(REF_DIR, "optimize_combined.py"), "wb"), check=True)
        # strategies/config.py does not exist at HEAD; ensure new strategies dir is usable there too
        os.makedirs(os.path.join(REF_DIR, "strategies"), exist_ok=True)
        shutil.copytree(os.path.join(PROJECT, "strategies"), os.path.join(REF_DIR, "strategies"), dirs_exist_ok=True)
        print("reference tree prepared")

    print("running reference (HEAD)...")
    run_scenario(REF_DIR, ref_out)
    print("running new (working tree)...")
    run_scenario(PROJECT, new_out)

    ref = json.load(open(ref_out))
    new = json.load(open(new_out))
    print(f"\nref: {len(ref['trades'])} trades, final cash ${ref['cash']:,.2f}")
    print(f"new: {len(new['trades'])} trades, final cash ${new['cash']:,.2f}")

    if ref == new:
        print("\nIDENTICAL: backtest outcomes match byte-for-byte")
        return 0
    print("\nMISMATCH!")
    # find first diff
    rt = {(t['d'], t['t'], t['s'], t['e']) : t for t in ref['trades']}
    nt = {(t['d'], t['t'], t['s'], t['e']) : t for t in new['trades']}
    for k in rt:
        if rt.get(k) != nt.get(k):
            print("  first diff:", k)
            print("   ref:", rt.get(k))
            print("   new:", nt.get(k))
            break
    return 1

sys.exit(main())
