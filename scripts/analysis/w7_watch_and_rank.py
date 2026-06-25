"""Watcher: poll Postgres for W7 completion (>=600 trials), then auto-launch
the top-N forward ranking script.

Launch:
  python scripts/analysis/w7_watch_and_rank.py --target 600 --top-n 10 --workers 4
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import time
import subprocess
import psycopg2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--poll-sec", type=int, default=300,
                    help="Poll interval (default 300s = 5 min)")
    args = ap.parse_args()

    print(f"Watcher started — waiting for W7 to reach {args.target} COMPLETE trials")
    print(f"Poll interval: {args.poll_sec}s")

    while True:
        try:
            c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres",
                                  dbname="optuna", connect_timeout=5)
            cur = c.cursor()
            cur.execute("SELECT count(*) FROM trials WHERE state='COMPLETE'")
            done = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM trials WHERE state='RUNNING'")
            running = cur.fetchone()[0]
            c.close()
            print(f"[{time.strftime('%H:%M:%S')}] COMPLETE={done}  RUNNING={running}", flush=True)
            if done >= args.target:
                print(f"\n>>> Target reached. Launching top-{args.top_n} forward ranking...")
                cmd = [
                    "python", "-u",
                    "scripts/analysis/w7_top_n_forward.py",
                    str(args.top_n),
                    "--workers", str(args.workers),
                ]
                print(f"  cmd: {' '.join(cmd)}")
                # Wait for the W7 workers to terminate before running the
                # top-N forward (avoid CPU contention). We do a 60-second
                # grace period for the writeback of forward.json.
                print("  Waiting 60s for W7 workers to write final JSON...")
                time.sleep(60)
                subprocess.run(cmd, check=False)
                break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] poll error: {e}", flush=True)
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
