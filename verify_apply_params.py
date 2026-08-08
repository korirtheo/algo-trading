"""Equivalence test: strategies.config.apply_params must produce byte-identical
module globals to optimize_combined.set_strategy_params for the same params dict.

Runs in a subprocess-safe way: loads tgc fresh, applies both paths, compares all
uppercase non-underscore globals.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_green_candle_combined as tgc
from strategies.config import apply_params

# Build a representative params dict touching every strategy, some enabled some not
params = {}
ALL = "hgafdvpmrwobkcseijnlx"
enabled = {"g", "l", "h", "a", "f", "v", "p", "m"}
for s in ALL:
    params[f"enable_{s}"] = s in enabled

# Give each strategy distinct params
for s in ALL:
    params[f"priority_{s}"] = 99 if s not in enabled else 5
params["g_target_pct"] = 13.5
params["g_stop_pct"] = 4.0
params["g_trail_pct"] = 1.0
params["g_trail_activate_pct"] = 2.0
params["g_time_limit_min"] = 15
params["g_min_gap_pct"] = 40.0
params["g_require_2nd_green"] = False
params["l_min_gap"] = 35.0
params["l_max_float"] = 8000000
params["l_stop_pct"] = 10.0
params["h_target_pct"] = 20.0
params["d_min_gap"] = 50.0
params["v_min_gap"] = 30.0
params["p_min_gap"] = 25.0
params["x_min_first_leg_gain_pct"] = 8.0
params["min_price"] = 1.5
params["max_modeled_slip_bp"] = 200.0
params["g_participation_cap"] = 0.10
params["w_not_wed"] = 0.2
params["enable_news_filter"] = True
params["min_news_articles"] = 3

# Path A: current set_strategy_params
from optimize_combined import set_strategy_params
set_strategy_params(dict(params))
a = {k: getattr(tgc, k) for k in dir(tgc) if k.isupper() and not k.startswith("_")}

# Path B: reset to defaults then apply_params
# Reload tgc module fresh to clear set_strategy_params effects
import importlib
tgc = importlib.reload(tgc)
apply_params(dict(params), tgc)
b = {k: getattr(tgc, k) for k in dir(tgc) if k.isupper() and not k.startswith("_")}

keys = set(a) | set(b)
diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
if diff:
    print(f"MISMATCH: {len(diff)} globals differ")
    for k, (va, vb) in sorted(diff.items()):
        print(f"  {k}: set_strategy_params={va!r}  apply_params={vb!r}")
    sys.exit(1)
print(f"OK: {len(keys)} globals identical across both paths")
