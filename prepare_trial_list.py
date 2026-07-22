
import json
import random

def prepare_trial_list():
    # 1a. Get the top 50 trials from the results file
    with open("config/trial_gl_trail_v2_oos_2026_results.json") as f:
        results = json.load(f)

    top_50_trials = [r["trial"] for r in results if r["train_rank"] <= 50]

    # 1b. Get all trial numbers from the wide forward list
    with open("config/trial_gl_trail_wide_forward_list.json") as f:
        all_trials_data = json.load(f)
    all_trials = all_trials_data["trial_numbers"]

    # 2. Get the random 50 trials
    non_top_50_trials = [t for t in all_trials if t not in top_50_trials]
    random_50_trials = random.sample(non_top_50_trials, 50)

    # 3. Combine the lists
    trial_list = top_50_trials + random_50_trials
    
    print(f"Top 50 trials: {top_50_trials}")
    print(f"Random 50 trials: {random_50_trials}")
    print(f"Total trials to run: {len(trial_list)}")
    
    # Save the lists to files to be used later
    with open("top_50_trials.json", "w") as f:
        json.dump(top_50_trials, f)
        
    with open("random_50_trials.json", "w") as f:
        json.dump(random_50_trials, f)
        
    with open("trial_list.json", "w") as f:
        json.dump(trial_list, f)

if __name__ == "__main__":
    prepare_trial_list()
