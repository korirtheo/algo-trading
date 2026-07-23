"""
Optuna Optimizer Launcher: O, G, L, H, M, A, P, F
===================================================
Spawns 8 parallel worker processes running optimize_combined.py with
FORCE_ENABLE_STRATS=o,g,l,h,m,a,p,f to lock the 8 profitable strategies ON.
Each worker runs 1500 trials with 300 startup trials.

Usage:
  python run_optuna_oglhmafp.py              # 8 workers, 1500 trials each
  python run_optuna_oglhmafp.py --workers 4  # 4 workers
  python run_optuna_oglhmafp.py --trials 500 --startup 100  # quick test
"""

import os
import sys
import subprocess
import argparse
import time

STRATS = "o,g,l,h,m,a,p,f"
STUDY = "oglhmafp_combined"
DB = "postgresql://postgres@127.0.0.1:5432/optuna_oglhmafp"
PARAMS_OUT = "config/trial_oglhmafp_best.json"


def main():
    parser = argparse.ArgumentParser(description="Launch OGLHMAFP Optuna study")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--trials", type=int, default=3000,
                        help="Total trials per worker (default: 3000)")
    parser.add_argument("--startup", type=int, default=1000,
                        help="TPE startup trials (default: 1000)")
    parser.add_argument("--date-start", default="2024-01-01",
                        help="Training window start (default: 2024-01-01)")
    parser.add_argument("--date-end", default="2026-02-28",
                        help="Training window end (default: 2026-02-28)")
    args = parser.parse_args()

    print("=" * 70)
    print("Optuna Launcher: O, G, L, H, M, A, P, F")
    print(f"  Strategies:  {STRATS.upper()}")
    print(f"  Workers:     {args.workers}")
    print(f"  Trials:      {args.trials} per worker")
    print(f"  Startup:     {args.startup}")
    print(f"  Study:       {STUDY}")
    print(f"  DB:          {DB}")
    print(f"  Params out:  {PARAMS_OUT}")
    print("=" * 70)

    # Pre-set tgc slippage flags before launching workers — the objective
    # function's assertion (line 1392) fires before the first trial sets them.
    import test_green_candle_combined as _tgc
    _tgc.USE_DYNAMIC_SLIPPAGE = True
    _tgc.USE_MULTIWINDOW_SLIPPAGE = True

    # Build the command for each worker
    base_cmd = [
        sys.executable, "optimize_combined.py",
        "--study-type", "combined",
        "--study", STUDY,
        "--db", DB,
        "--trials", str(args.trials),
        "--startup-trials", str(args.startup),
        "--n-jobs", "1",        # single-threaded per process
        "--dynamic-slip",       # liquidity-aware slippage
        "--params-out", PARAMS_OUT,
        "--date-start", args.date_start,
        "--date-end", args.date_end,
    ]

    # Set env for each worker — FORCE_ENABLE_STRATS locks the 8 strategies ON
    worker_env = os.environ.copy()
    worker_env["FORCE_ENABLE_STRATS"] = STRATS

    # Step 1: Initialize DB schema with a single trial (avoids race condition)
    print(f"\nInitializing PostgreSQL schema...")
    init_cmd = base_cmd + ["--trials", "1", "--startup-trials", "1"]
    init_proc = subprocess.run(init_cmd, env=worker_env, capture_output=True, text=True)
    if init_proc.returncode != 0:
        print(f"  WARNING: DB init failed (may already be initialized):")
        print(f"  {init_proc.stderr[-200:] if init_proc.stderr else 'no stderr'}")
    else:
        print("  Schema initialized OK")

    # Step 2: Launch all workers
    print(f"\nLaunching {args.workers} workers...")
    print(f"  Command: {' '.join(base_cmd)}")
    print(f"  FORCE_ENABLE_STRATS={STRATS}\n")

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

    # Wait for all workers
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

    # Extract best params
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
        print("Run manually: python optimize_combined.py --dump-best "
              f"--db {DB} --study {STUDY} --params-out {PARAMS_OUT}")


if __name__ == "__main__":
    main()
