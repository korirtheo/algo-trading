"""Compare #511 vs Trial #5 on FULL OOS (Mar-Jun 2026) by strategy."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock

STARTING_CASH = 25_000

def configure_simulator():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08

def run_backtest_by_strategy(dates, picks_by_date, params, snapshot=None):
    """Run backtest and return strategy breakdown."""
    cash = float(STARTING_CASH)
    strat_stats = {}
    
    for d in dates:
        picks = picks_by_date.get(d, [])
        if not picks:
            continue
        
        cash_account = cash < MARGIN_THRESHOLD
        try:
            states, cash, unsettled, _ = tgc.simulate_day_combined(picks, cash, cash_account, params=snapshot)
        except Exception:
            continue
        
        cash += (unsettled if cash_account else 0)
        
        for st in states:
            if st.get("exit_reason") and st.get("pnl"):
                strat = st.get("strategy", "?")
                if strat not in strat_stats:
                    strat_stats[strat] = {"wins": 0, "losses": 0, "total_pnl": 0, "trades": 0}
                strat_stats[strat]["trades"] += 1
                strat_stats[strat]["total_pnl"] += st.get("pnl", 0)
                if st.get("pnl") > 0:
                    strat_stats[strat]["wins"] += 1
                else:
                    strat_stats[strat]["losses"] += 1
    
    return strat_stats

configure_simulator()
dirs = [d for d in ["stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data"] if Path(d).exists()]
all_dates, picks_by_date = load_all_picks(dirs)
dates = sorted([d for d in all_dates if "2026-03" in d or "2026-04" in d or "2026-05" in d or "2026-06" in d])
print(f"OOS window: {len(dates)} trading days\n")

# #511
with open("config/trial_g511_l626_v3_overlay.json") as f:
    params_511 = dict(json.load(f)["params"])

with _param_lock:
    _oc_set_params(params_511)
    snapshot_511 = _build_param_snapshot()
strats_511 = run_backtest_by_strategy(dates, picks_by_date, params_511, snapshot_511)

# Trial #5
with open("config/trial_g511_l626_v3_overlay.json") as f:
    base_params = dict(json.load(f)["params"])
with open("config/trial_g511_l626_v3_full_no_trail_best.json") as f:
    t5_only = json.load(f)["params"]

params_t5 = dict(base_params)
params_t5.update(t5_only)

with _param_lock:
    _oc_set_params(params_t5)
    snapshot_t5 = _build_param_snapshot()
strats_t5 = run_backtest_by_strategy(dates, picks_by_date, params_t5, snapshot_t5)

print("="*80)
print("FULL OOS (Mar-Jun 2026): #511 vs Trial #5 by Strategy")
print("="*80)

print("\n#511 (WITH TRAILING):")
print(f"  V3 enabled: {params_511.get('v_enabled', 'N/A')}")
for strat in sorted(strats_511.keys()):
    s = strats_511[strat]
    wr = s["wins"] / max(s["trades"], 1) * 100
    print(f"  {strat}: {s['trades']} trades | {s['wins']}W-{s['losses']}L ({wr:.0f}%) | ${s['total_pnl']:+,.0f}")

print("\nTRIAL #5 (NO TRAILING):")
print(f"  V3 enabled: {params_t5.get('v_enabled', 'N/A')}")
for strat in sorted(strats_t5.keys()):
    s = strats_t5[strat]
    wr = s["wins"] / max(s["trades"], 1) * 100
    print(f"  {strat}: {s['trades']} trades | {s['wins']}W-{s['losses']}L ({wr:.0f}%) | ${s['total_pnl']:+,.0f}")

print("\n" + "="*80)
total_511 = sum(s["total_pnl"] for s in strats_511.values())
total_t5 = sum(s["total_pnl"] for s in strats_t5.values())
print(f"#511 TOTAL:      ${total_511:+,.0f} ({sum(s['trades'] for s in strats_511.values())} trades)")
print(f"Trial #5 TOTAL:  ${total_t5:+,.0f} ({sum(s['trades'] for s in strats_t5.values())} trades)")
print(f"Difference:      ${total_t5 - total_511:+,.0f}")

print("\n" + "="*80)
print("V3 PARAMS COMPARISON:")
print("="*80)
print(f"#511 V3:     {[f'{k}={v}' for k,v in params_511.items() if k.startswith('v_')]}")
print(f"Trial #5 V3: {[f'{k}={v}' for k,v in params_t5.items() if k.startswith('v_')]}")
