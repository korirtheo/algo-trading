"""Poll W21b every 90s. When COMPLETE trial count hits TARGET, kill all workers.

This is a background watcher so user doesn't have to manually monitor.
"""
import time
import psutil
import psycopg2

TARGET_TRIALS = 1200
POLL_SECONDS = 90

print(f"WATCHDOG: Will kill W21b workers when COMPLETE trial count reaches {TARGET_TRIALS}")
print(f"Polling every {POLL_SECONDS}s...")

while True:
    try:
        c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w21b')
        cur = c.cursor()
        cur.execute("""SELECT COUNT(*) FROM trials
                       WHERE state='COMPLETE'
                       AND study_id=(SELECT study_id FROM studies WHERE study_name='w21b_target100')""")
        n_complete = cur.fetchone()[0]
        cur.execute("""SELECT COUNT(*) FROM trials
                       WHERE state='RUNNING'
                       AND study_id=(SELECT study_id FROM studies WHERE study_name='w21b_target100')""")
        n_running = cur.fetchone()[0]
        c.close()

        ps = [p for p in psutil.process_iter(['pid','cmdline'])
              if any('target100' in str(c) for c in (p.info.get('cmdline') or []))]

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] complete={n_complete}, running={n_running}, workers={len(ps)}")

        if n_complete >= TARGET_TRIALS:
            print(f"\n*** TARGET HIT ({n_complete} >= {TARGET_TRIALS}). Killing {len(ps)} workers... ***")
            for p in ps:
                try:
                    p.kill()
                    print(f"  killed PID {p.pid}")
                except Exception as e:
                    print(f"  PID {p.pid}: {e}")
            time.sleep(3)
            # Clean any remaining zombie RUNNING trial records
            c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w21b')
            c.autocommit = True
            cur = c.cursor()
            cur.execute("""UPDATE trials SET state='FAIL' WHERE state='RUNNING'
                           AND study_id=(SELECT study_id FROM studies WHERE study_name='w21b_target100')""")
            print(f"Cleaned {cur.rowcount} zombie RUNNING trials")
            c.close()
            print("DONE. W21b stopped at target. Study preserved in optuna_w21b DB for resume.")
            break
    except Exception as e:
        print(f"poll error: {e}")
    time.sleep(POLL_SECONDS)
