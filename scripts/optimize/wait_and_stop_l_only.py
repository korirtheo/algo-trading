"""Watchdog: kill l_only_w21b workers at 600 completes."""
import time, psutil, psycopg2
TARGET, POLL = 600, 90
print(f"WATCHDOG l_only_w21b: kill at {TARGET}")
while True:
    try:
        c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_l_only')
        cur = c.cursor()
        cur.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE' AND study_id=(SELECT study_id FROM studies WHERE study_name='l_only_w21b')")
        n = cur.fetchone()[0]
        c.close()
        ps = [p for p in psutil.process_iter(['pid','cmdline']) if any('optimize_combined.py' in str(c) and 'l_only_w21b' in str(c) for c in (p.info.get('cmdline') or []))]
        print(f"[{time.strftime('%H:%M:%S')}] l_only complete={n} workers={len(ps)}")
        if n >= TARGET:
            print(f"TARGET HIT. Killing {len(ps)} workers...")
            for p in ps:
                try: p.kill(); print(f"  killed {p.pid}")
                except: pass
            time.sleep(3)
            c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_l_only')
            c.autocommit = True
            c.cursor().execute("UPDATE trials SET state='FAIL' WHERE state='RUNNING' AND study_id=(SELECT study_id FROM studies WHERE study_name='l_only_w21b')")
            c.close()
            print("DONE.")
            break
    except Exception as e:
        print(f"poll error: {e}")
    time.sleep(POLL)
