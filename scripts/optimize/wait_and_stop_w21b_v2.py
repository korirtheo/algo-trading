"""Poll W21b v2 every 90s. Kill workers when COMPLETE >= TARGET_TRIALS."""
import time
import psutil
import psycopg2

TARGET_TRIALS = 800
POLL_SECONDS = 90

print(f"WATCHDOG: Will kill W21b v2 workers when COMPLETE >= {TARGET_TRIALS}")
print(f"Polling every {POLL_SECONDS}s...")

while True:
    try:
        c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w21b_v2')
        cur = c.cursor()
        cur.execute("""SELECT COUNT(*) FROM trials
                       WHERE state='COMPLETE'
                       AND study_id=(SELECT study_id FROM studies WHERE study_name='w21b_v2_full_2024')""")
        n_complete = cur.fetchone()[0]
        cur.execute("""SELECT COUNT(*) FROM trials
                       WHERE state='RUNNING'
                       AND study_id=(SELECT study_id FROM studies WHERE study_name='w21b_v2_full_2024')""")
        n_running = cur.fetchone()[0]
        c.close()

        ps = [p for p in psutil.process_iter(['pid', 'cmdline'])
              if any('optuna_w21b_v2.py' in str(c) for c in (p.info.get('cmdline') or []))]

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
            c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_w21b_v2')
            c.autocommit = True
            cur = c.cursor()
            cur.execute("""UPDATE trials SET state='FAIL' WHERE state='RUNNING'
                           AND study_id=(SELECT study_id FROM studies WHERE study_name='w21b_v2_full_2024')""")
            print(f"Cleaned {cur.rowcount} zombie RUNNING trials")
            c.close()
            print("DONE. W21b v2 stopped at target.")
            break
    except Exception as e:
        print(f"poll error: {e}")
    time.sleep(POLL_SECONDS)
