
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = 'postgresql://postgres@127.0.0.1:5432/optuna_gl_trail'
study = optuna.load_study(study_name='gl_trail_v3', storage=storage)

completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)

top_50_trial_numbers = [t.number for t in sorted_trials[:50]]

trial_number_to_find = 509

trial_found = None
for t in completed_trials:
    if t.number == trial_number_to_find:
        trial_found = t
        break

if trial_found:
    print(f"--- Details for Trial #{trial_number_to_find} ---")
    
    if trial_number_to_find in top_50_trial_numbers:
        rank = top_50_trial_numbers.index(trial_number_to_find) + 1
        print(f"This was one of the top 50 trials, ranked at #{rank}.")
    else:
        print("This was one of the random trials.")

    print("\nParameters:")
    for k, v in sorted(trial_found.params.items()):
        print(f"  {k}: {v}")
else:
    print(f"Trial #{trial_number_to_find} not found in the study.")
