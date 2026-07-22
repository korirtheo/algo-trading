
import optuna, json
optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = 'postgresql://postgres@127.0.0.1:5432/optuna_gl_trail'
study = optuna.load_study(study_name='gl_trail_v3', storage=storage)
t = [x for x in study.trials if x.state.name == 'COMPLETE']
f = [x for x in study.trials if x.state.name == 'FAIL']
print(f'complete={len(t)} fail={len(f)}')
best = study.best_trial
ua = best.user_attrs
print(f'Best trial #{best.number}')
print(f'  Score:  ${best.value:,.0f}')
print(f'  PnL:    ${ua.get("total_pnl",0):,.0f}')
print(f'  PF:     {ua.get("pf",0):.3f}')
print(f'  WR:     {ua.get("wr",0):.1f}%')
print(f'  n:      {ua.get("n")}')
print(f'  Equity: ${ua.get("equity",0):,.0f}')
print()
print('Params:')
params = {}
for k, v in sorted(best.params.items()):
    print(f'  {k}: {v}')
    params[k] = v

with open('config/best_params_gl_trail_v3.json', 'w') as f:
    json.dump({'params': params}, f, indent=2)
