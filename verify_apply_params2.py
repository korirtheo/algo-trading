"""Extended equivalence checks for apply_params vs set_strategy_params.

Cases:
  1. Empty params dict (the drift case: set_strategy_params({}) resets 21 gaps)
  2. A real deployed config JSON (trial_1655_g1_l1_optimized_deploy.json)
  3. All-disabled params
  4. Baseline trial_432_params.json
"""
import sys
import os
import json
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_green_candle_combined as tgc
from strategies.config import apply_params
from optimize_combined import set_strategy_params

CASES = []
CASES.append(("empty", {}))
CASES.append(("all_disabled", {f"enable_{s}": False for s in "hgafdvpmrwobkcseijnlx"}))
for cfg in [
    "config/trial_1655_g1_l1_optimized_deploy.json",
    "config/trial_432_params.json",
    "config/trial_6_extracted.json",
]:
    if os.path.exists(cfg):
        raw = json.load(open(cfg))
        p = raw.get("params", raw) if isinstance(raw, dict) and "params" in raw else raw
        CASES.append((os.path.basename(cfg), p))

fails = 0
for name, params in CASES:
    set_strategy_params(dict(params))
    a = {k: getattr(tgc, k) for k in dir(tgc) if k.isupper() and not k.startswith("_")}

    tgc = importlib.reload(tgc)
    apply_params(dict(params), tgc)
    b = {k: getattr(tgc, k) for k in dir(tgc) if k.isupper() and not k.startswith("_")}

    keys = set(a) | set(b)
    diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
    if diff:
        fails += 1
        print(f"[FAIL] {name}: {len(diff)} differ")
        for k, (va, vb) in sorted(diff.items())[:10]:
            print(f"    {k}: set={va!r} apply={vb!r}")
    else:
        print(f"[OK]   {name}: {len(keys)} globals identical")

sys.exit(1 if fails else 0)
