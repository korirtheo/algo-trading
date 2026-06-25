#!/bin/bash
# Auto-launch the X-enabled W4 microcap-pump run AFTER the no-X run finishes.
#
# Monitors the no-X DB for completion (500 trials COMPLETE+PRUNED),
# then kicks off the same W4 setup but WITHOUT --no-x (X strategy in search space).

set -e

cd "$(dirname "$0")/../.."

NOX_DB="results/wf_pf_microcap_pump_noX/W4_train_2021_2022_2023_2024_test_2025.db"
echo "[$(date)] Watching $NOX_DB for completion (500 trials COMPLETE+PRUNED)..."

while true; do
    if [[ -f "$NOX_DB" ]]; then
        DONE=$(python -c "
import sqlite3, sys
try:
    conn = sqlite3.connect('$NOX_DB')
    cur = conn.cursor()
    cur.execute('SELECT state, COUNT(*) FROM trials GROUP BY state')
    rows = dict(cur.fetchall())
    print(rows.get('COMPLETE', 0) + rows.get('PRUNED', 0))
except Exception:
    print(0)
")
        if [[ "$DONE" -ge 500 ]]; then
            echo "[$(date)] no-X run complete ($DONE/500). Launching X-enabled run..."
            break
        fi
        echo "[$(date)] no-X progress: $DONE/500 — waiting..."
    fi
    sleep 300  # check every 5 min
done

# Launch X-enabled W4 — same flags but no --no-x
mkdir -p results/wf_pf_microcap_pump_X
python scripts/analysis/walk_forward_optuna.py \
    --n-trials 500 --n-startup 150 --only-window 4 \
    --use-multiwindow-slippage \
    --shape-filter microcap-thin,thin-microcap \
    --outdir results/wf_pf_microcap_pump_X \
    2>&1 | tee results/wf_pf_microcap_pump_X/W4_run.log
