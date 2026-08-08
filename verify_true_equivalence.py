"""TRUE equivalence: apply_params vs the ORIGINAL set_strategy_params body from
git HEAD (before the refactor). Loads the original body into a standalone
function and compares all globals across many params dicts."""
import subprocess
import sys
import os
import json
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Extract original set_strategy_params source from git HEAD and exec it
head = subprocess.run(["git", "show", "HEAD:optimize_combined.py"], capture_output=True, text=True, encoding="utf-8").stdout

# find the original function text
src_start = head.index("def set_strategy_params(params):")
src_end = head.index("def suggest_all_params(trial):")
orig_src = head[src_start:src_end].rstrip()
orig_src = orig_src.replace("def set_strategy_params(params):", "def _orig_set_strategy_params(params):", 1)

# exec it in a fresh namespace with a fake 'tgc'? No — it references module 'tgc'.
# We'll exec in this process after importing tgc so 'tgc' resolves.
import test_green_candle_combined as tgc
from strategies.config import apply_params
from optimize_combined import ALL_STRATS

exec(compile(orig_src, "<orig>", "exec"), globals())

def snapshot():
    return {k: getattr(tgc, k) for k in dir(tgc) if k.isupper() and not k.startswith("_")}

def reset():
    global tgc
    tgc = importlib.reload(tgc)
    globals()["tgc"] = tgc

CASES = []
CASES.append(("empty", {}))
CASES.append(("all_disabled", {f"enable_{s}": False for s in "hgafdvpmrwobkcseijnlx"}))
CASES.append(("all_enabled", {f"enable_{s}": True for s in "hgafdvpmrwobkcseijnlx"}))
for cfg in ["config/trial_1655_g1_l1_optimized_deploy.json", "config/trial_432_params.json",
            "config/trial_6_extracted.json", "config/trial_818_squeeze_extracted.json",
            "config/trial_571_normal_extracted.json"]:
    if os.path.exists(cfg):
        raw = json.load(open(cfg))
        p = raw.get("params", raw) if isinstance(raw, dict) and "params" in raw else raw
        CASES.append((os.path.basename(cfg), p))

# an adversarial case with keys that original HARDCODES (h/a/f min gaps, x extras)
adv = {f"enable_{s}": True for s in "hgafdvpmrwobkcseijnlx"}
adv.update({"h_min_gap_pct": 50.0, "a_min_gap_pct": 25.0, "f_min_gap_pct": 20.0,
            "x_min_pullback_pct": 15.0, "x_min_recovery_pct": 5.0, "x_target_pct_of_peak": 80.0,
            "x_min_first_leg_gain_pct": 9.0, "x_time_limit_min": 90, "x_trail_pct": 7.0,
            "x_trail_activate_pct": 6.0, "x_max_entry_hhmm": "13:00", "x_entry_require_green": False,
            "x_first_leg_window_bars": 20, "x_min_bars_since_peak": 8, "x_min_vol_vs_avg": 2.0,
            "x_vol_avg_bars": 10, "x_min_entry_room_pct": 6.0, "x_stop_pct_below_trough": 3.0})
CASES.append(("adversarial_hardcoded", adv))

fails = 0
for name, params in CASES:
    reset()
    _orig_set_strategy_params(dict(params))
    a = snapshot()
    reset()
    apply_params(dict(params), tgc)
    b = snapshot()
    keys = set(a) | set(b)
    diff = {k: (a.get(k), b.get(k)) for k in keys if repr(a.get(k)) != repr(b.get(k))}
    if diff:
        fails += 1
        print(f"[FAIL] {name}: {len(diff)} differ")
        for k, (va, vb) in sorted(diff.items())[:15]:
            print(f"    {k}: orig={va!r} apply={vb!r}")
    else:
        print(f"[OK]   {name}: {len(keys)} globals identical")

sys.exit(1 if fails else 0)
