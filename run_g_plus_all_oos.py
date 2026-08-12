"""Test G + each single strategy on the 2026 OOS window.

For each extra strategy X, runs G+X and reports G's PnL, X's PnL, total.
"""
import pickle
import sys

sys.path.insert(0, r"C:\Users\Theo Korir\Documents\Python\algo-trading")
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params

tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.USE_2MIN_SLIPPAGE = False

data = pickle.load(open(r"stored_data_1min\fulltest_picks_gap2_vol250k.pkl", "rb"))
dates = sorted(x for x in data if "2026-03-02" <= x <= "2026-08-07")

ALL = "hgafdvpmrwobkcseijnlx"


def run(g_on, extras):
    params = {f"enable_{s}": False for s in ALL}
    params["enable_g"] = g_on
    for e in extras:
        params[f"enable_{e}"] = True
    set_strategy_params(params)
    cash = 25000.0
    pnl_by = {}
    n_by = {}
    for dd in dates:
        picks = data.get(dd, [])
        if not picks:
            continue
        states, cash, unsettled, _ = tgc.simulate_day_combined(
            picks, cash, cash < tgc.MARGIN_THRESHOLD, params=None
        )
        cash += unsettled if cash < tgc.MARGIN_THRESHOLD else 0
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                s = st.get("strategy", "?")
                p = st.get("pnl", 0) or 0
                pnl_by[s] = pnl_by.get(s, 0) + p
                n_by[s] = n_by.get(s, 0) + 1
    return pnl_by, n_by


def fmt(pnl_by, n_by):
    parts = []
    for s, p in pnl_by.items():
        parts.append(f"{s.upper()}={n_by.get(s,0)}(${p:,.0f})")
    total = sum(pnl_by.values())
    return " ".join(parts) + f" TOTAL=${total:,.0f}"


# Baseline G only
pb, nb = run(True, [])
print("G only         :", fmt(pb, nb))

# G + each other strategy
for s in ALL:
    if s == "g":
        continue
    pb, nb = run(True, [s])
    print(f"G + {s.upper()}      :", fmt(pb, nb))
