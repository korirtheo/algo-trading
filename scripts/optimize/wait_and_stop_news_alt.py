"""Poll news_alt every 90s. Kill workers when COMPLETE >= 600."""
import time, psutil, psycopg2
TARGET, POLL = 600, 90
print(f"WATCHDOG: kill news_alt workers at {TARGET} completes")
while True:
    try:
        c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_news_alt')
        cur = c.cursor()
        cur.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE' AND study_id=(SELECT study_id FROM studies WHERE study_name='news_alt_1st_green')")
        n = cur.fetchone()[0]
        c.close()
        ps = [p for p in psutil.process_iter(['pid','cmdline']) if any('optuna_news_alt_entry.py' in str(c) for c in (p.info.get('cmdline') or []))]
        print(f"[{time.strftime('%H:%M:%S')}] complete={n} workers={len(ps)}")
        if n >= TARGET:
            print(f"TARGET HIT. Killing {len(ps)} workers...")
            for p in ps:
                try: p.kill(); print(f"  killed {p.pid}")
                except: pass
            time.sleep(3)
            c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_news_alt')
            c.autocommit = True
            c.cursor().execute("UPDATE trials SET state='FAIL' WHERE state='RUNNING' AND study_id=(SELECT study_id FROM studies WHERE study_name='news_alt_1st_green')")
            c.close()
            print("DONE.")
            break
    except Exception as e:
        print(f"poll error: {e}")
    time.sleep(POLL)
