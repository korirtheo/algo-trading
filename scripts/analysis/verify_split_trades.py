"""Verify G1/G2/L1/L2 exit params are used correctly per trade."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import optuna
import optimize_gl_split as ogs
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params, DATA_DIRS, STARTING_CASH
from test_full import load_all_picks

storage = optuna.storages.RDBStorage(url='postgresql://postgres@127.0.0.1:5432/optuna_gl_split')
study = optuna.load_study(study_name='gl_split_v2', storage=storage)
trial = study.trials[0]
trial_params = dict(trial.params)
trial_params['enable_g'] = True
trial_params['enable_l'] = True
for s in 'vhafdrwobkcsexijn':
    trial_params[f'enable_{s}'] = False

tgc.USE_DYNAMIC_SLIPPAGE = True
tgc.USE_MULTIWINDOW_SLIPPAGE = True
std_params = ogs._map_split_to_standard(trial_params)
set_strategy_params(std_params)
snapshot = ogs._build_split_snapshot(std_params)

# Expected exit params
g1_stop = snapshot['g_stop_pct']; g1_trail = snapshot['g_trail_pct']; g1_target = snapshot['g_target_pct']; g1_time = snapshot['g_time_limit_min']
g2_stop = snapshot['g2_stop_pct']; g2_trail = snapshot['g2_trail_pct']; g2_target = snapshot['g2_target_pct']; g2_time = snapshot['g2_time_limit_min']
l1_stop = snapshot['l_stop_pct']; l1_trail = snapshot['l_trail_pct']; l1_time = snapshot['l_time_limit_min']
l2_stop = snapshot['l2_stop_pct']; l2_trail = snapshot['l2_trail_pct']; l2_time = snapshot['l2_time_limit_min']

print(f'Expected params:')
print(f'  G1: stop={g1_stop}% trail={g1_trail}% target={g1_target}% time={g1_time}min')
print(f'  G2: stop={g2_stop}% trail={g2_trail}% target={g2_target}% time={g2_time}min')
print(f'  L1: stop={l1_stop}% trail={l1_trail}% time={l1_time}min')
print(f'  L2: stop={l2_stop}% trail={l2_trail}% time={l2_time}min')

all_dates, daily_picks = load_all_picks(DATA_DIRS)
all_dates = [d for d in all_dates if '2024-01-01' <= d <= '2026-02-28']

all_trades = []
cash = float(STARTING_CASH)
unsettled = 0.0
for d in all_dates:
    cash += unsettled; unsettled = 0.0
    picks = daily_picks.get(d, [])
    states, cash, unsettled, _ = tgc.simulate_day_combined(picks, cash, params=snapshot)
    for st in states:
        if st['exit_reason'] is not None:
            all_trades.append(st)

def pnl_pct(st):
    entry = st.get('entry_price') or 0
    exit_p = st.get('exit_price') or 0
    return ((exit_p - entry) / entry * 100) if entry else 0

def drop_pct(st):
    entry = st.get('entry_price') or 0
    exit_p = st.get('exit_price') or 0
    return ((entry - exit_p) / entry * 100) if entry else 0

# Split
g1 = [t for t in all_trades if t['strategy'] == 'G' and t.get('entry_trade_seq', 0) == 0]
g2 = [t for t in all_trades if t['strategy'] == 'G' and t.get('entry_trade_seq', 0) > 0]
l1 = [t for t in all_trades if t['strategy'] == 'L' and t.get('entry_trade_seq', 0) == 0]
l2 = [t for t in all_trades if t['strategy'] == 'L' and t.get('entry_trade_seq', 0) > 0]

print(f'\nTrade counts: G1={len(g1)} G2={len(g2)} L1={len(l1)} L2={len(l2)}')

# === G1 vs G2 STOP exits ===
print(f'\n{"="*70}')
print(f'G1 STOP exits (param: g_stop={g1_stop}%)')
g1_stops = sorted([t for t in g1 if t['exit_reason'] == 'STOP'], key=lambda t: -drop_pct(t))[:5]
for st in g1_stops:
    print(f'  {st.get("ticker","?")} seq={st["entry_trade_seq"]} drop={drop_pct(st):.1f}%  entry_params g_stop={st.get("entry_params",{}).get("g_stop_pct","MISSING")}  exit={st["exit_reason"]} ${st.get("pnl",0):.0f}')

print(f'\nG2 STOP exits (param: g2_stop={g2_stop}%)')
g2_stops = sorted([t for t in g2 if t['exit_reason'] == 'STOP'], key=lambda t: -drop_pct(t))[:5]
for st in g2_stops:
    print(f'  {st.get("ticker","?")} seq={st["entry_trade_seq"]} drop={drop_pct(st):.1f}%  entry_params g2_stop={st.get("entry_params",{}).get("g2_stop_pct","MISSING")}  exit={st["exit_reason"]} ${st.get("pnl",0):.0f}')

# === G1 vs G2 TRAIL exits ===
print(f'\n{"="*70}')
print(f'G1 TRAIL exits (param: g_trail={g1_trail}%)')
g1_trails = sorted([t for t in g1 if t['exit_reason'] == 'TRAIL'], key=lambda t: -pnl_pct(t))[:5]
for st in g1_trails:
    print(f'  {st.get("ticker","?")} seq={st["entry_trade_seq"]} gain={pnl_pct(st):+.1f}%  entry_params g_trail={st.get("entry_params",{}).get("g_trail_pct","MISSING")}  ${st.get("pnl",0):.0f}')

print(f'\nG2 TRAIL exits (param: g2_trail={g2_trail}%)')
g2_trails = sorted([t for t in g2 if t['exit_reason'] == 'TRAIL'], key=lambda t: -pnl_pct(t))[:5]
for st in g2_trails:
    print(f'  {st.get("ticker","?")} seq={st["entry_trade_seq"]} gain={pnl_pct(st):+.1f}%  entry_params g2_trail={st.get("entry_params",{}).get("g2_trail_pct","MISSING")}  ${st.get("pnl",0):.0f}')

# === G1 vs G2 TARGET exits ===
print(f'\n{"="*70}')
print(f'G1 TARGET exits (param: g_target={g1_target}%)')
g1_targs = [t for t in g1 if t['exit_reason'] in ('TARGET', 'TARGET2')][:5]
for st in g1_targs:
    print(f'  {st.get("ticker","?")} seq={st["entry_trade_seq"]} gain={pnl_pct(st):+.1f}%  entry_params g_target={st.get("entry_params",{}).get("g_target_pct","MISSING")}  ${st.get("pnl",0):.0f}')

print(f'\nG2 TARGET exits (param: g2_target={g2_target}%)')
g2_targs = [t for t in g2 if t['exit_reason'] in ('TARGET', 'TARGET2')][:5]
for st in g2_targs:
    print(f'  {st.get("ticker","?")} seq={st["entry_trade_seq"]} gain={pnl_pct(st):+.1f}%  entry_params g2_target={st.get("entry_params",{}).get("g2_target_pct","MISSING")}  ${st.get("pnl",0):.0f}')

# === L1 vs L2 TRAIL exits ===
print(f'\n{"="*70}')
print(f'L1 TRAIL exits (param: l_trail={l1_trail}%)')
l1_trails = sorted([t for t in l1 if t['exit_reason'] == 'TRAIL'], key=lambda t: -pnl_pct(t))[:5]
for st in l1_trails:
    print(f'  {st.get("ticker","?")} seq={st["entry_trade_seq"]} gain={pnl_pct(st):+.1f}%  entry_params l_trail={st.get("entry_params",{}).get("l_trail_pct","MISSING")}  ${st.get("pnl",0):.0f}')

print(f'\nL2 TRAIL exits (param: l2_trail={l2_trail}%)')
l2_trails = sorted([t for t in l2 if t['exit_reason'] == 'TRAIL'], key=lambda t: -pnl_pct(t))[:5]
for st in l2_trails:
    print(f'  {st.get("ticker","?")} seq={st["entry_trade_seq"]} gain={pnl_pct(st):+.1f}%  entry_params l2_trail={st.get("entry_params",{}).get("l2_trail_pct","MISSING")}  ${st.get("pnl",0):.0f}')

# === SUMMARY: per-split stats ===
print(f'\n{"="*70}')
print('SUMMARY PER SPLIT:')
for name, trades, stop, trail, target in [
    ('G1', g1, g1_stop, g1_trail, g1_target),
    ('G2', g2, g2_stop, g2_trail, g2_target),
    ('L1', l1, l1_stop, l1_trail, None),
    ('L2', l2, l2_stop, l2_trail, None),
]:
    n = len(trades)
    if n == 0:
        print(f'  {name}: 0 trades')
        continue
    wins = [t for t in trades if t.get('pnl', 0) > 0]
    wr = len(wins) / n * 100
    total = sum(t.get('pnl', 0) for t in trades)
    stops = len([t for t in trades if t['exit_reason'] == 'STOP'])
    trails = len([t for t in trades if t['exit_reason'] == 'TRAIL'])
    targets = len([t for t in trades if t['exit_reason'] in ('TARGET', 'TARGET2')])
    times = len([t for t in trades if t['exit_reason'] == 'TIME_STOP'])
    other = n - stops - trails - targets - times
    # Verify: avg drop for stops should be close to stop param
    stop_trades = [t for t in trades if t['exit_reason'] == 'STOP']
    avg_drop = sum(drop_pct(t) for t in stop_trades) / len(stop_trades) if stop_trades else 0
    print(f'  {name}: {n} trades | WR={wr:.1f}% | PnL=${total:,.0f} | STOP={stops} TRAIL={trails} TARGET={targets} TIME={times} OTHER={other}')
    print(f'       stop_param={stop}% avg_stop_drop={avg_drop:.1f}% | trail_param={trail}%', end='')
    if target:
        print(f' | target_param={target}%', end='')
    print()
