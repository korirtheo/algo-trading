"""
Optuna Launcher: G+L only, trained on the 1-min data pipeline
================================================================
Spawns N parallel worker processes running optimize_combined.py with
ALLOWED_STRATS=g,l. Each trial tunes G and L together (enable both ON,
others locked OFF). Trains on the 1-min-resampled pipeline
(stored_data_1min) — SEPARATE from all legacy 2-min studies.

Data window default: 2024-01-01 -> 2026-02-28 (matches the live deployed
configs' training regime).

Usage:
  python run_optuna_gl_1min.py              # 8 workers, 2000 total trials
  python run_optuna_gl_1min.py --workers 4  # 4 workers
  python run_optuna_gl_1min.py --trials 500 --startup 100  # quick test
"""

import os
import sys
import subprocess
import argparse
import time

STRATS = "g,l"
STUDY = "gl_1min_v2"
DB = "postgresql://postgres@127.0.0.1:5432/optuna_gl_1min"
PARAMS_OUT = "config/trial_gl_1min_v2_best.json"
DATA_DIRS = "stored_data_1min"


def main():
    parser = argparse.ArgumentParser(description="Launch G+L-only 1-min Optuna study")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--trials", type=int, default=1650,
                        help="Total trials for the STUDY, split across workers (default: 1650 = ~50x dims)")
    parser.add_argument("--startup", type=int, default=330,
                        help="Total startup trials for the STUDY, split across workers (default: 330 = 10x dims)")
    parser.add_argument("--date-start", default="2024-01-01",
                        help="Training window start (default: 2024-01-01)")
    parser.add_argument("--date-end", default="2026-02-28",
                        help="Training window end (default: 2026-02-28)")
    parser.add_argument("--data-dirs", default=DATA_DIRS,
                        help="Comma-separated data dirs (default: stored_data_1min)")
    args = parser.parse_args()

    per_worker_trials = max(1, args.trials // args.workers)
    per_worker_startup = max(1, args.startup // args.workers)

    print("=" * 70)
    print("Optuna Launcher: G+L only, 1-min pipeline")
    print(f"  Strategies:    {STRATS.upper()} (G/L independently enabled, others OFF)")
    print(f"  Workers:       {args.workers}")
    print(f"  Study trials:  {args.trials} total ({per_worker_trials}/worker)")
    print(f"  Study startup: {args.startup} total ({per_worker_startup}/worker)")
    print(f"  Study:         {STUDY}")
    print(f"  DB:            {DB}")
    print(f"  Data dirs:     {args.data_dirs}")
    print(f"  Params out:    {PARAMS_OUT}")
    print("=" * 70)

    # Pre-set slippage flags before launching workers
    import test_green_candle_combined as _tgc
    _tgc.USE_DYNAMIC_SLIPPAGE = True
    _tgc.USE_MULTIWINDOW_SLIPPAGE = True

    base_cmd = [
        sys.executable, "optimize_combined.py",
        "--study-type", "combined",
        "--study", STUDY,
        "--db", DB,
        "--trials", str(per_worker_trials),
        "--startup-trials", str(per_worker_startup),
        "--n-jobs", "1",
        "--dynamic-slip",
        "--params-out", PARAMS_OUT,
        "--date-start", args.date_start,
        "--date-end", args.date_end,
        "--data-dirs", args.data_dirs,
    ]

    worker_env = os.environ.copy()
    worker_env["ALLOWED_STRATS"] = STRATS  # G and L sampled enable/disable independently

    print(f"\nInitializing PostgreSQL schema (1-trial bootstrap)...")
    init_cmd = base_cmd + ["--trials", "1", "--startup-trials", "1"]
    init_proc = subprocess.run(init_cmd, env=worker_env, capture_output=True, text=True)
    if init_proc.returncode != 0:
        print(f"  WARNING: DB init failed (may already be initialized):")
        print(f"  {init_proc.stderr[-300:] if init_proc.stderr else 'no stderr'}")
    else:
        print("  Schema initialized OK")

    print(f"\nLaunching {args.workers} workers...")
    print(f"  Per worker: {per_worker_trials} trials, {per_worker_startup} startup")
    print(f"  ALLOWED_STRATS={STRATS}\n")

    start_time = time.time()
    procs = []
    for i in range(args.workers):
        p = subprocess.Popen(
            base_cmd,
            env=worker_env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        procs.append(p)
        print(f"  Worker {i+1}/{args.workers} started (PID {p.pid})")

    print(f"\nAll {args.workers} workers launched. Waiting for completion...\n")

    failed = 0
    for i, p in enumerate(procs):
        rc = p.wait()
        elapsed = time.time() - start_time
        if rc == 0:
            print(f"  Worker {i+1} completed (elapsed: {elapsed/60:.1f} min)")
        else:
            print(f"  Worker {i+1} FAILED with exit code {rc}")
            failed += 1

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"All workers finished in {total_time/60:.1f} min ({total_time/3600:.1f} hrs)")
    if failed:
        print(f"  WARNING: {failed}/{args.workers} workers failed")
    print(f"{'='*70}")

    if not failed:
        print(f"\nExtracting best params from {DB}...")
        dump_cmd = [
            sys.executable, "optimize_combined.py",
            "--dump-best",
            "--db", DB,
            "--study", STUDY,
            "--params-out", PARAMS_OUT,
        ]
        subprocess.run(dump_cmd, check=True)
        print(f"\nBest params saved to: {PARAMS_OUT}")
    else:
        print("\nSkipping best-param extraction (some workers failed)")
        print(f"Run manually: python optimize_combined.py --dump-best "
              f"--db {DB} --study {STUDY} --params-out {PARAMS_OUT}")


if __name__ == "__main__":
    main()
