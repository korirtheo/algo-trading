"""Quick test: trailing impact on July 2026 blind OOS."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock

STARTING_CASH = 25_000
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

def configure_simulator():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08

def disable_adaptive_controls():
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    for s in ALL_STRATS:
        setattr(tgc, f"{s.upper()}_PARTICIPATION_CAP", 0.0)

def disable_all_trailing(params):
    p = dict(params)
    for s in ALL_STRATS:
        p[f"{s}_trail_pct"] = 0.0
        p[f"{s}_trail_activate_pct"] = 0.0
    return p

def run_backtest(dates, picks_by_date, params, snapshot=None):
    cash = float(STARTING_CASH)
    trades = []
    for d in dates:
        picks = picks_by_date.get(d, [])
        if not picks:
            continue
        try:
            states, cash, unsettled, _ = tgc.simulate_day_combined(picks, cash, cash < MARGIN_THRESHOLD, params=snapshot)
            cash += (unsettled if cash < MARGIN_THRESHOLD else 0)
            trades.extend([st.get("pnl", 0) for st in states if st.get("exit_reason") and st.get("pnl")])
        except:
            pass
    
    if not trades:
        return {"n": 0, "pnl": 0, "pf": 0, "wr": 0}
    pnl = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    gross_win = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t <= 0))
    pf = gross_win / max(gross_loss, 1e-9)
    return {"n": len(trades), "pnl": pnl, "pf": pf, "wr": wins/len(trades)*100}

configure_simulator()
dirs = [d for d in ["stored_data"] if Path(d).exists()]
all_dates, picks_by_date = load_all_picks(dirs)
july_dates = sorted([d for d in all_dates if "2026-07" in d])

print("="*60)
print("July 2026 OOS (Blind Forward Test)")
print("="*60 + f"\n{len(july_dates)} trading days\n")

with open("config/trial_g511_l626_v3_overlay.json") as f:
    params = dict(json.load(f)["params"])

print("Testing #511 WITH trailing...")
with _param_lock:
    _oc_set_params(params)
    disable_adaptive_controls()
    snap = _build_param_snapshot()
with_trail = run_backtest(july_dates, picks_by_date, params, snap)

print("Testing #511 WITHOUT trailing...")
no_trail_p = disable_all_trailing(params)
with _param_lock:
    _oc_set_params(no_trail_p)
    disable_adaptive_controls()
    snap = _build_param_snapshot()
no_trail = run_backtest(july_dates, picks_by_date, no_trail_p, snap)

print(f"\nWITH Trailing:    {with_trail['n']} trades | P&L ${with_trail['pnl']:+,.0f} | PF {with_trail['pf']:.2f} | WR {with_trail['wr']:.1f}%")
print(f"WITHOUT Trailing: {no_trail['n']} trades | P&L ${no_trail['pnl']:+,.0f} | PF {no_trail['pf']:.2f} | WR {no_trail['wr']:.1f}%")
print(f"\nTrailing Impact: ${with_trail['pnl'] - no_trail['pnl']:+,.0f}")
