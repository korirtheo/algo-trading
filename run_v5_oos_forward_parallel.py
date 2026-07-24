"""
Parallel 2026 OOS forward test — top 100 best + 100 random from oglhmafp_v5.
Spawns 8 worker subprocesses for ~8x speedup.

Usage: python run_v5_oos_forward_parallel.py
"""
import sys, os, json, random, subprocess, time, tempfile
import numpy as np

STUDY_ID = 44
DB = "postgresql://postgres@127.0.0.1:5432/optuna_oglhmafp"
DATA_DIRS = ["stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data_jul_2026"]
STARTING_CASH = 25_000
N_WORKERS = 8
RESULTS_DIR = "results"

# ── 1. Load trials from PostgreSQL ────────────────────────────────────
print("Loading trials from PostgreSQL...")
import psycopg2
conn = psycopg2.connect(DB)
cur = conn.cursor()

# Top 100 by score
cur.execute("""
    SELECT t.trial_id, t.number, v.value as score
    FROM trials t JOIN trial_values v ON t.trial_id = v.trial_id
    WHERE t.study_id = %s AND t.state = 'COMPLETE'
    ORDER BY v.value DESC LIMIT 100
""", (STUDY_ID,))
top100 = cur.fetchall()
print(f"  Top 100: trial #{top100[0][1]} (score=${top100[0][2]:,.0f}) to #{top100[-1][1]} (score=${top100[-1][2]:,.0f})")

# 100 random (excluding top 100)
cur.execute("""
    SELECT t.trial_id, t.number, v.value as score
    FROM trials t JOIN trial_values v ON t.trial_id = v.trial_id
    WHERE t.study_id = %s AND t.state = 'COMPLETE'
""", (STUDY_ID,))
all_trials = cur.fetchall()
top_ids = {t[0] for t in top100}
remaining = [t for t in all_trials if t[0] not in top_ids]
random.seed(42)
random100 = random.sample(remaining, min(100, len(remaining)))
print(f"  Random 100: from {len(remaining)} non-top trials")

# Load params + user_attrs for all 200 trials
def load_trial(cur, trial_id, number):
    cur.execute("SELECT param_name, param_value FROM trial_params WHERE trial_id = %s", (trial_id,))
    params = {name: val for name, val in cur.fetchall()}
    cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id = %s", (trial_id,))
    for key, val in cur.fetchall():
        if key.startswith("enable_"):
            params[key] = val == "true" if isinstance(val, str) else val
    for k, v in params.items():
        if isinstance(v, float) and v == int(v):
            params[k] = int(v)
    cur.execute("SELECT value FROM trial_values WHERE trial_id = %s", (trial_id,))
    row = cur.fetchone()
    is_score = row[0] if row else 0
    return {"trial_id": trial_id, "number": number, "is_score": is_score, "params": params}

print("Loading params for 200 trials...")
trials_data = []
for tid, num, score in top100:
    trials_data.append(("top", load_trial(cur, tid, num)))
for tid, num, score in random100:
    trials_data.append(("random", load_trial(cur, tid, num)))
conn.close()

# ── 2. Split into chunks for parallel workers ────────────────────────
chunks = [[] for _ in range(N_WORKERS)]
for i, item in enumerate(trials_data):
    chunks[i % N_WORKERS].append(item)

print(f"\nSplit {len(trials_data)} trials across {N_WORKERS} workers:")
for i, c in enumerate(chunks):
    top_n = sum(1 for g, _ in c if g == "top")
    print(f"  Worker {i+1}: {len(c)} trials ({top_n} top, {len(c)-top_n} random)")

# ── 3. Write chunk files and launch workers ───────────────────────────
worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_oos_worker.py")
# Worker script is standalone _oos_worker.py (not overwritten here)

# Write chunk files and launch workers
print(f"\nLaunching {N_WORKERS} workers...")
os.makedirs(RESULTS_DIR, exist_ok=True)
start_time = time.time()

procs = []
chunk_files = []
for i in range(N_WORKERS):
    chunk_file = os.path.join(RESULTS_DIR, f"_chunk_{i}.json")
    out_file = os.path.join(RESULTS_DIR, f"_oos_result_{i}.json")
    chunk_files.append((out_file,))

    with open(chunk_file, "w") as f:
        json.dump(chunks[i], f)

    p = subprocess.Popen(
        [sys.executable, worker_script, chunk_file, out_file, str(i+1)],
        stdout=sys.stdout, stderr=sys.stderr,
    )
    procs.append(p)
    print(f"  Worker {i+1}/{N_WORKERS} started (PID {p.pid})")

# Wait for all workers
print(f"\nWaiting for all workers...\n")
for i, p in enumerate(procs):
    rc = p.wait()
    elapsed = time.time() - start_time
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"  Worker {i+1} {status} (elapsed: {elapsed/60:.1f} min)")

# ── 4. Collect and aggregate results ──────────────────────────────────
print(f"\nCollecting results...")
results = []
for out_file, in chunk_files:
    if os.path.exists(out_file):
        with open(out_file) as f:
            results.extend(json.load(f))

# Clean up chunk files
for i in range(N_WORKERS):
    chunk_file = os.path.join(RESULTS_DIR, f"_chunk_{i}.json")
    out_file = os.path.join(RESULTS_DIR, f"_oos_result_{i}.json")
    if os.path.exists(chunk_file): os.remove(chunk_file)
    if os.path.exists(out_file): os.remove(out_file)

elapsed = time.time() - start_time
print(f"All {len(results)} backtests done in {elapsed/60:.1f} min")

# Save results IMMEDIATELY (before summary, in case of encoding errors)
top_results = [r for r in results if r["group"] == "top"]
rand_results = [r for r in results if r["group"] == "random"]
out = {
    "study": "oglhmafp_v5",
    "starting_cash": STARTING_CASH,
    "elapsed_min": elapsed / 60,
    "top100": top_results,
    "random100": rand_results,
}
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "v5_oos_forward_2026.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"Results saved to {RESULTS_DIR}/v5_oos_forward_2026.json")

# ── 5. Summary ────────────────────────────────────────────────────────
top_results = [r for r in results if r["group"] == "top"]
rand_results = [r for r in results if r["group"] == "random"]

def stats(label, subset):
    pnls = [r["oos_pnl"] for r in subset]
    sharpes = [r["oos_sharpe"] for r in subset]
    trades = [r["oos_trades"] for r in subset]
    wrs = [r["oos_wr"] for r in subset]
    pfs = [r["oos_pf"] for r in subset if r["oos_pf"] != float("inf")]
    dds = [r["oos_max_dd"] for r in subset]
    winners = sum(1 for p in pnls if p > 0)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Trials:          {len(subset)}")
    print(f"  Winners:         {winners}/{len(subset)} ({100*winners/len(subset):.0f}%)")
    print(f"  OOS PnL:         mean=${np.mean(pnls):>+10,.0f}  median=${np.median(pnls):>+10,.0f}")
    print(f"                   best=${max(pnls):>+10,.0f}  worst=${min(pnls):>+10,.0f}")
    print(f"  OOS Sharpe:      mean={np.mean(sharpes):.2f}  median={np.median(sharpes):.2f}")
    print(f"  OOS WR:          mean={np.mean(wrs):.1f}%  median={np.median(wrs):.1f}%")
    print(f"  OOS PF:          mean={np.mean(pfs):.2f}  median={np.median(pfs):.2f}")
    print(f"  OOS Trades:      mean={np.mean(trades):.0f}  median={np.median(trades):.0f}")
    print(f"  OOS MaxDD:       mean={np.mean(dds):.1f}%  median={np.median(dds):.1f}%")

if top_results:
    stats("TOP 100 (by IS score)", top_results)
if rand_results:
    stats("RANDOM 100", rand_results)

# ── 6. IS vs OOS correlation ──────────────────────────────────────────
if len(top_results) > 1:
    top_is = [r["is_score"] for r in top_results]
    top_oos = [r["oos_pnl"] for r in top_results]
    corr = np.corrcoef(top_is, top_oos)[0, 1]

    print(f"\n{'='*70}")
    print(f"  IS vs OOS CORRELATION (top 100)")
    print(f"{'='*70}")
    print(f"  Pearson r = {corr:.4f}")
    if corr < 0.3:
        print(f"  -> Weak correlation - IS score is a poor OOS predictor")
    elif corr < 0.6:
        print(f"  -> Moderate correlation - some IS->OOS transfer")
    else:
        print(f"  -> Strong correlation - IS generalizes well")

# ── 7. Strategy combo distribution ────────────────────────────────────
from collections import Counter
print(f"\n{'='*70}")
print(f"  STRATEGY COMBOS (top 100)")
print(f"{'='*70}")
combos = Counter(tuple(sorted(r["enabled"])) for r in top_results)
for combo, cnt in combos.most_common(10):
    subset = [r for r in top_results if tuple(sorted(r["enabled"])) == combo]
    avg_oos = np.mean([r["oos_pnl"] for r in subset])
    print(f"  {list(combo):<30} {cnt:>3} trials  avg OOS: ${avg_oos:>+10,.0f}")

print(f"\n{'='*70}")
print(f"  STRATEGY COMBOS (random 100)")
print(f"{'='*70}")
combos_r = Counter(tuple(sorted(r["enabled"])) for r in rand_results)
for combo, cnt in combos_r.most_common(10):
    subset = [r for r in rand_results if tuple(sorted(r["enabled"])) == combo]
    avg_oos = np.mean([r["oos_pnl"] for r in subset])
    print(f"  {list(combo):<30} {cnt:>3} trials  avg OOS: ${avg_oos:>+10,.0f}")

# ── 8. Top 10 best OOS ────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TOP 10 BEST OOS (from all {len(results)})")
print(f"{'='*70}")
by_oos = sorted(results, key=lambda r: -r["oos_pnl"])[:10]
for r in by_oos:
    print(f"  #{r['trial']:<5} ({r['group']:<6}) IS=${r['is_score']:>+12,.0f}  "
          f"OOS=${r['oos_pnl']:>+10,.0f}  Sharpe={r['oos_sharpe']:.2f}  "
          f"WR={r['oos_wr']:.0f}%  Trades={r['oos_trades']:<4}  "
          f"strats={r['enabled']}")

# ── 9. Top 10 worst OOS ──────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  TOP 10 WORST OOS (from all {len(results)})")
print(f"{'='*70}")
by_oos_worst = sorted(results, key=lambda r: r["oos_pnl"])[:10]
for r in by_oos_worst:
    print(f"  #{r['trial']:<5} ({r['group']:<6}) IS=${r['is_score']:>+12,.0f}  "
          f"OOS=${r['oos_pnl']:>+10,.0f}  Sharpe={r['oos_sharpe']:.2f}  "
          f"WR={r['oos_wr']:.0f}%  Trades={r['oos_trades']:<4}  "
          f"strats={r['enabled']}")

# Done - results already saved earlier
