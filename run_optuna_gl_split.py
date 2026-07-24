"""
Optuna Optimizer: G+L Split-Param Study (G1/G2/L1/L2)
=====================================================
Tunes G and L with SEPARATE exit params for the 1st trade of the day vs
subsequent trades. Entry conditions are shared between G1/G2 and L1/L2.

OOS trade sequence analysis revealed:
  G: 1st trade WR=92.2%, 2nd+ WR=70.3% (degrades)
  L: 1st trade WR=75.0%, 2nd+ WR=83.8% (improves)
Split params let Optuna optimize each independently.

Usage:
  python run_optuna_gl_split.py              # 8 workers, 3000 total trials
  python run_optuna_gl_split.py --workers 4  # 4 workers
  python run_optuna_gl_split.py --trials 500 --startup 100  # quick test
"""

import os
import sys
import subprocess
import argparse
import time

STUDY = "gl_split_v1"
DB = "postgresql://postgres@127.0.0.1:5432/optuna_oglhmafp"
PARAMS_OUT = "config/trial_gl_split_v1_best.json"
STRATS = "g,l"


def main():
    parser = argparse.ArgumentParser(description="Launch G+L Split-Param Optuna study")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--trials", type=int, default=3000,
                        help="Total trials for the STUDY, split across workers (default: 3000)")
    parser.add_argument("--startup", type=int, default=500,
                        help="Total startup trials for the STUDY, split across workers (default: 500)")
    parser.add_argument("--date-start", default="2024-01-01",
                        help="Training window start (default: 2024-01-01)")
    parser.add_argument("--date-end", default="2026-02-28",
                        help="Training window end (default: 2026-02-28)")
    args = parser.parse_args()

    per_worker_trials = max(1, args.trials // args.workers)
    per_worker_startup = max(1, args.startup // args.workers)

    print("=" * 70)
    print("Optuna Launcher: G+L Split-Param (G1/G2/L1/L2)")
    print(f"  Strategies:    {STRATS.upper()} (locked ON)")
    print(f"  Workers:       {args.workers}")
    print(f"  Study trials:  {args.trials} total ({per_worker_trials}/worker)")
    print(f"  Study startup: {args.startup} total ({per_worker_startup}/worker)")
    print(f"  Study:         {STUDY}")
    print(f"  DB:            {DB}")
    print(f"  Params out:    {PARAMS_OUT}")
    print("=" * 70)

    # Pre-set tgc slippage flags before launching workers
    import test_green_candle_combined as _tgc
    _tgc.USE_DYNAMIC_SLIPPAGE = True
    _tgc.USE_MULTIWINDOW_SLIPPAGE = True

    # Build the command for each worker
    base_cmd = [
        sys.executable, "optimize_gl_split.py",
        "--study", STUDY,
        "--db", DB,
        "--trials", str(per_worker_trials),
        "--startup-trials", str(per_worker_startup),
        "--n-jobs", "1",
        "--dynamic-slip",
        "--params-out", PARAMS_OUT,
        "--date-start", args.date_start,
        "--date-end", args.date_end,
    ]

    # Set env for each worker
    worker_env = os.environ.copy()
    worker_env["ALLOWED_STRATS"] = STRATS

    # Step 1: Launch all workers (DB schema auto-created by Optuna on first connect)
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
            sys.executable, "optimize_gl_split.py",
            "--dump-best",
            "--db", DB,
            "--study", STUDY,
            "--params-out", PARAMS_OUT,
        ]
        subprocess.run(dump_cmd, check=True)
        print(f"\nBest params saved to: {PARAMS_OUT}")
    else:
        print("\nSkipping best-param extraction (some workers failed)")
        print("Run manually: python optimize_gl_split.py --dump-best "
              f"--db {DB} --study {STUDY} --params-out {PARAMS_OUT}")


if __name__ == "__main__":
    main()
