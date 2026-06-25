"""OOS validation of new Optuna bests #818 (squeeze) and #571 (normal).

For each OOS year (2019, 2020, 2026), runs every config on:
  - regime-squeeze days (cleaned of corp-action)
  - regime-normal days
Reports PnL, PnL/day, and wins per config.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from strategies.regime_gate import classify_regime

YEAR_DIRS = {
    "2019": ["stored_data_2019"],
    "2020": ["stored_data_2020"],
    "2026": ["stored_data", "stored_data_mar_may_2026"],
}

CFGS = [
    ("config/trial_6_extracted.json",           "#6   generalist"),
    ("config/trial_326_squeeze_extracted.json", "#326 sq2.09 (2026 winner)"),
    ("config/trial_587_squeeze_extracted.json", "#587 sq2.91"),
    ("config/trial_818_squeeze_extracted.json", "#818 sq2.81 NEW"),
    ("config/trial_571_normal_extracted.json",  "#571 norm2.08 NEW"),
]
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000


def _merged(p):
    with open(p) as f: x = json.load(f)
    if isinstance(x, dict) and "params" in x: x = x["params"]
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(x); return m


def _is_corp_action(picks):
    if not picks: return False
    leader = max(picks, key=lambda p: p.get("gap_pct") or 0)
    return (leader.get("gap_pct", 0) or 0) > 500 and (leader.get("prev_close", 0) or 0) < 1.0


def _run(days, daily_picks, cfg_path):
    set_strategy_params(_merged(cfg_path))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.VOL_CAP_PCT = 5.0
    cash = STARTING_CASH
    pnls = []
    for d in days:
        picks = daily_picks.get(d, [])
        if not picks:
            pnls.append(0); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            _, end_c, unset, _ = tgc.simulate_day_combined(picks, cash, cash_account=is_cash)
        except Exception:
            pnls.append(0); continue
        pnl = end_c - cash; cash = end_c
        if is_cash: cash += unset
        pnls.append(pnl)
    pnls = np.array(pnls)
    return cash, cash - STARTING_CASH, int((pnls > 0).sum()), pnls


def year_section(year, dirs):
    print("=" * 100)
    print(f"YEAR {year}")
    print("=" * 100)
    dirs_present = [d for d in dirs if os.path.exists(d)]
    if not dirs_present:
        print(f"[skip] no data dirs"); return
    dates, dp = load_all_picks(dirs_present)

    sq = [d for d in dates if classify_regime(dp.get(d, [])) == "squeeze"]
    sq_clean = [d for d in sq if not _is_corp_action(dp.get(d, []))]
    nm = [d for d in dates if classify_regime(dp.get(d, [])) == "normal"]
    print(f"\n{len(dates)} trading days  |  squeeze={len(sq_clean)} (after corp filter)  |  normal={len(nm)}")

    for set_label, days in [("SQUEEZE days", sq_clean), ("NORMAL days", nm)]:
        if not days: continue
        print(f"\n--- {set_label} ({len(days)} days) ---")
        print(f"  {'Config':<28} {'Final':>11} {'PnL':>11} {'PnL/day':>10}  Wins")
        for cfg, label in CFGS:
            if not os.path.exists(cfg): continue
            final, total, wins, _ = _run(days, dp, cfg)
            print(f"  {label:<28} ${final:>10,.0f} ${total:>+10,.0f} ${total/len(days):>+9,.0f}  {wins:>4}/{len(days):<3}")


def main():
    for y, dirs in YEAR_DIRS.items():
        year_section(y, dirs)


if __name__ == "__main__":
    main()
