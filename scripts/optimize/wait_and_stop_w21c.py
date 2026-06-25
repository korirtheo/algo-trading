"""Poll W21c every 90s. Kill workers when COMPLETE >= TARGET_TRIALS."""
import time
import psutil
import psycopg2

TARGET_TRIALS = 800
POLL_SECONDS = 90

print(f"WATCHDOG: Will kill W21c workers when COMPLETE >= {TARGET_TRIALS}")
print(f"Polling every {POLL_SECONDS}s...")

while True:
    try:
        c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w21c')
        cur = c.cursor()
        cur.execute("""SELECT COUNT(*) FROM trials
                       WHERE state='COMPLETE'
                       AND study_id=(SELECT study_id FROM studies WHERE study_name='w21c_multiwindow_v2')""")
        n_complete = cur.fetchone()[0]
        cur.execute("""SELECT COUNT(*) FROM trials
                       WHERE state='RUNNING'
                       AND study_id=(SELECT study_id FROM studies WHERE study_name='w21c_multiwindow_v2')""")
        n_running = cur.fetchone()[0]
        c.close()

        ps = [p for p in psutil.process_iter(['pid', 'cmdline'])
              if any('optuna_w21c_multiwindow.py' in str(c) for c in (p.info.get('cmdline') or []))]

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
            c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w21c')
            c.autocommit = True
            cur = c.cursor()
            cur.execute("""UPDATE trials SET state='FAIL' WHERE state='RUNNING'
                           AND study_id=(SELECT study_id FROM studies WHERE study_name='w21c_multiwindow_v2')""")
            print(f"Cleaned {cur.rowcount} zombie RUNNING trials")
            c.close()
            print("DONE. W21c stopped at target. Study preserved in optuna_w21c DB for resume.")
            break
    except Exception as e:
        print(f"poll error: {e}")
    time.sleep(POLL_SECONDS)
