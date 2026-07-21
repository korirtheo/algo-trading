"""Forward test: #511 with trailing disabled vs #511 with trailing on (blind OOS Mar-Jun 2026)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock

STARTING_CASH = 25_000
OOS_DATA_DIRS = ["stored_data_mar_may_2026", "stored_data_jun_2026"]
OOS_DATE_LO = "2026-03-01"
OOS_DATE_HI = "2026-06-30"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

def configure_simulator():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

def disable_adaptive_controls():
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    for s in ALL_STRATS:
        setattr(tgc, f"{s.upper()}_PARTICIPATION_CAP", 0.0)
    tgc.NEWS_FILTER_ENABLED = False

def disable_all_trailing(params):
    """Force all trailing params to 0."""
    p = dict(params)
    for s in ALL_STRATS:
        p[f"{s}_trail_pct"] = 0.0
        p[f"{s}_trail_activate_pct"] = 0.0
    return p

def run_backtest(dates, picks_by_date, params, snapshot=None):
    cash = float(STARTING_CASH)
    all_trades = []
    for d in dates:
        picks = picks_by_date.get(d, [])
        if not picks:
            continue
        cash_account = cash < MARGIN_THRESHOLD
        try:
            states, cash, unsettled, _ = tgc.simulate_day_combined(picks, cash, cash_account, params=snapshot)
        except Exception:
            continue
        effective_cash = cash + (unsettled if cash_account else 0)
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0)
                if pnl:
                    all_trades.append(pnl)
        cash = effective_cash
    
    if not all_trades:
        return {"n": 0, "total_pnl": 0, "pf": 0, "wr": 0, "equity": cash}
    
    total_pnl = sum(all_trades)
    wins = [p for p in all_trades if p > 0]
    losses = [p for p in all_trades if p <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0
    
    return {"n": len(all_trades), "total_pnl": total_pnl, "pf": pf, "wr": wr, "equity": cash}

print("="*80)
print("BLIND OOS FORWARD TEST (Mar-Jun 2026): #511 With vs Without Trailing")
print("="*80)

configure_simulator()
dirs = [d for d in OOS_DATA_DIRS if Path(d).exists()]
all_dates, picks_by_date = load_all_picks(dirs)
dates = sorted([d for d in all_dates if OOS_DATE_LO <= d <= OOS_DATE_HI])
print(f"\nBlind OOS window: {OOS_DATE_LO} to {OOS_DATE_HI} ({len(dates)} days)\n")

# Load #511 config
with open("config/trial_g511_l626_v3_overlay.json") as f:
    deployed_params = dict(json.load(f)["params"])

# Test #511 WITH trailing
print("Testing #511 (trailing ENABLED)...")
with _param_lock:
    _oc_set_params(deployed_params)
    disable_adaptive_controls()
    with_trail_snap = _build_param_snapshot()
with_trail_result = run_backtest(dates, picks_by_date, deployed_params, with_trail_snap)

# Test #511 WITHOUT trailing
print("Testing #511 (trailing DISABLED)...")
no_trail_params = disable_all_trailing(deployed_params)
with _param_lock:
    _oc_set_params(no_trail_params)
    disable_adaptive_controls()
    no_trail_snap = _build_param_snapshot()
no_trail_result = run_backtest(dates, picks_by_date, no_trail_params, no_trail_snap)

print("\n" + "="*80)
print("BLIND OOS RESULTS (Mar-Jun 2026):")
print("="*80)
print(f"\n#511 WITH Trailing:")
print(f"  Trades:    {with_trail_result['n']}")
print(f"  P&L:       ${with_trail_result['total_pnl']:,.2f}")
print(f"  PF:        {with_trail_result['pf']:.3f}")
print(f"  WR:        {with_trail_result['wr']:.1f}%")
print(f"  Equity:    ${with_trail_result['equity']:,.2f}")

print(f"\n#511 WITHOUT Trailing:")
print(f"  Trades:    {no_trail_result['n']}")
print(f"  P&L:       ${no_trail_result['total_pnl']:,.2f}")
print(f"  PF:        {no_trail_result['pf']:.3f}")
print(f"  WR:        {no_trail_result['wr']:.1f}%")
print(f"  Equity:    ${no_trail_result['equity']:,.2f}")

print(f"\nIMPACT OF TRAILING:")
pnl_diff = with_trail_result['total_pnl'] - no_trail_result['total_pnl']
if pnl_diff > 0:
    ratio = pnl_diff / max(abs(no_trail_result['total_pnl']), 1)
    print(f"  ✓ Trailing IMPROVES: +${pnl_diff:+,.2f} ({ratio:.1f}x)")
else:
    ratio = abs(pnl_diff) / max(abs(with_trail_result['total_pnl']), 1)
    print(f"  ✗ Trailing HURTS: ${pnl_diff:+,.2f} ({ratio:.1f}% worse)")
