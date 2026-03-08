import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
for db in ['optuna_low_float.db', 'optuna_low_float_tiered.db']:
    storage = f'sqlite:///{db}'
    names = optuna.study.get_all_study_names(storage)
    for name in names:
        study = optuna.load_study(study_name=name, storage=storage)
        print(f'{db} | {name} | {len(study.trials)} trials | best score: {study.best_value:,.0f}')
        best = study.best_trial
        ua = best.user_attrs
        print(f'  Trial #{best.number}')
        print(f'  PnL: ${ua.get("total_pnl", 0):,.0f} | PF: {ua.get("pf", 0)} | WR: {ua.get("wr", 0)}% | Trades: {ua.get("n", 0)}')
        print(f'  Params: {dict(best.params)}')
        print()
