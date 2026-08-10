"""Optuna launcher: G2 (simplified G) on the 1-min pipeline.

G2 fixes the entry structure to "2nd green candle + >=15% gap, buy on the
second candle (09:31), no retry-fills" and tunes ONLY the exit side
(target, stop, time, trail, trail_activate). No permanent strategy added —
this is a study-scoped experiment driven by env flags.

  python run_optuna_gl_1min_g2.py --workers 8 --trials 200 --startup 40
"""
import os
import sys
import subprocess
import argparse
import time

STRATS = "g"  # only G fires; G2 is the fixed-entry variant of G
STUDY = "gl_1min_v5_g2_2x"
DB = "postgresql://postgres@127.0.0.1:5432/optuna_gl_1min"
PARAMS_OUT = "config/trial_gl_1min_v5_g2_2x_best.json"
DATA_DIRS = "stored_data_1min"


def main():
    parser = argparse.ArgumentParser(description="Launch G2 (simplified G) 1-min Optuna study")
    parser.add_argument("--workers", type=int, default=8)
    # G2 tunes 11 meaningful dims (5 exit + 3 bar gates + conviction + PM gate
    # + PM credit). Optuna TPE guidance: startup ~10x dims (110), trials
    # ~30x dims (330) for solid convergence.
    parser.add_argument("--trials", type=int, default=330,
                        help="Total trials (G2 tunes 11 dims)")
    parser.add_argument("--startup", type=int, default=110)
    parser.add_argument("--date-start", default="2024-01-01")
    parser.add_argument("--date-end", default="2026-02-28")
    parser.add_argument("--data-dirs", default=DATA_DIRS)
    args = parser.parse_args()

    per_worker_trials = max(1, args.trials // args.workers)
    per_worker_startup = max(1, args.startup // args.workers)

    print("=" * 70)
    print("Optuna Launcher: G2 (simplified G), 1-min pipeline")
    print(f"  Entry:        2nd green + >=15% gap, buy candle 2 (09:31), no retry-fill")
    print(f"  Tuned:        exit only (target/stop/time/trail/trail_activate)")
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
    worker_env["G_SIMPLE_ENTRY"] = "1"      # fix entry: 2nd green + 15% gap
    worker_env["G_FIRST_BAR_ONLY"] = "1"    # buy candle 2 only, no retry fills
    # Leverage overrides (opt-in): MARGIN_MULTIPLIER scales size by cash;
    # MAX_POSITION_PCT_OF_CASH caps a position at that % of cash (100 = 1x,
    # 200 = 2x margin). Default 1x / 100 = cash-account semantics.
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
    print(f"  ALLOWED_STRATS={STRATS}  G_SIMPLE_ENTRY=1  G_FIRST_BAR_ONLY=1\n")

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
