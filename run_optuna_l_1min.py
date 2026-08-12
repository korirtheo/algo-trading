"""Optuna launcher: L (Low Float Squeeze) standalone on the 1-min pipeline.

L is gated by the sim's hard entry floor (L_MIN_ENTRY_CANDLE=3, i.e. never
before 09:32) and never on the same candle G fires. G is NOT in this search
space — this tunes L in isolation so we can see its standalone edge before
ever stacking it with G.

  python run_optuna_l_1min.py --workers 8 --trials 300 --startup 100
"""
import os
import sys
import subprocess
import argparse
import time

STRATS = "l"  # only L fires; L_MIN_ENTRY_CANDLE floor (09:32+) baked into sim
STUDY = "gl_1min_l_standalone_v1"
DB = "postgresql://postgres@127.0.0.1:5432/optuna_gl_1min"
PARAMS_OUT = "config/trial_gl_1min_l_standalone_v1_best.json"
DATA_DIRS = "stored_data_1min"


def main():
    parser = argparse.ArgumentParser(description="Launch L (Low Float Squeeze) 1-min Optuna study")
    parser.add_argument("--workers", type=int, default=8)
    # L tunes ~19 dims. TPE guidance: startup ~10x dims, trials ~30x dims.
    parser.add_argument("--trials", type=int, default=570,
                        help="Total trials (L tunes ~19 dims)")
    parser.add_argument("--startup", type=int, default=190)
    parser.add_argument("--date-start", default="2024-01-01")
    parser.add_argument("--date-end", default="2026-02-28")
    parser.add_argument("--data-dirs", default=DATA_DIRS)
    args = parser.parse_args()

    per_worker_trials = max(1, args.trials // args.workers)
    per_worker_startup = max(1, args.startup // args.workers)

    print("=" * 70)
    print("Optuna Launcher: L (Low Float Squeeze), 1-min pipeline")
    print(f"  Entry gates:  candle >= max(L_EARLIEST_CANDLE, 3) i.e. 09:32+,")
    print(f"                and never the same candle G fires")
    print(f"  Tuned:        L entry + exit (~19 dims)")
    print(f"  Workers:      {args.workers}")
    print(f"  Study trials: {args.trials} total ({per_worker_trials}/worker)")
    print(f"  Study:        {STUDY}")
    print(f"  DB:           {DB}")
    print(f"  Data dirs:    {args.data_dirs}")
    print("=" * 70)

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
    worker_env["ALLOWED_STRATS"] = STRATS
    # Leverage overrides (opt-in): default 1x / 100 = cash-account semantics.
    worker_env.setdefault("MARGIN_MULTIPLIER", os.environ.get("MARGIN_MULTIPLIER", "1.0"))
    worker_env.setdefault("MAX_POSITION_PCT_OF_CASH", os.environ.get("MAX_POSITION_PCT_OF_CASH", "100"))

    print(f"\nInitializing PostgreSQL schema (1-trial bootstrap)...")
    init_cmd = base_cmd + ["--trials", "1", "--startup-trials", "1"]
    init_proc = subprocess.run(init_cmd, env=worker_env, capture_output=True, text=True)
    if init_proc.returncode != 0:
        print(f"  WARNING: DB init failed (may already be initialized):")
        print(f"  {init_proc.stderr[-300:] if init_proc.stderr else 'no stderr'}")
    else:
        print("  Schema initialized OK")

    print(f"\nLaunching {args.workers} workers...")
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
            "--study", STUDY,
            "--db", DB,
            "--params-out", PARAMS_OUT,
        ]
        dump_proc = subprocess.run(dump_cmd, env=worker_env, capture_output=True, text=True)
        print(dump_proc.stdout[-1500:] if dump_proc.stdout else "")
        if dump_proc.stderr:
            print(f"  stderr: {dump_proc.stderr[-400:]}")
        print(f"  Best params -> {PARAMS_OUT}")


if __name__ == "__main__":
    main()
