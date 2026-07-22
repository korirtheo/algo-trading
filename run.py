import argparse
import json
from optimize import create_objective, run_full_backtest, load_all_picks
from forward_test import main as forward_test_main
import optuna


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the configuration file")
    parser.add_argument(
        "mode",
        choices=["live", "backtest", "forward_test", "optimize"],
        help="Execution mode",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers to use"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Loading configuration from {args.config}")
    print(f"Executing in {args.mode} mode with {args.workers} workers")

    if args.mode == "optimize":
        study_name = config["study"]
        with open(f"studies/{study_name}.json") as f:
            study_config = json.load(f)

        storage = "postgresql://postgres@127.0.0.1:5432/optuna_gl_trail"

        daily_picks = load_all_picks(config["data_dirs"])
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(
            create_objective(daily_picks, study_config),
            n_trials=100,
            n_jobs=args.workers,
        )
    elif args.mode == "forward_test":
        forward_test_main(config, args)


if __name__ == "__main__":
    main()
