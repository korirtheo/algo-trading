#!/usr/bin/env bash
# G+L Trail Optuna Study — v3 (unconditional trail params, no g_use_trail)
WORKERS=${1:-4}
echo "Starting G+L Trail v3 study with $WORKERS workers..."
python optimize_combined.py \
    --study-type gl_trail \
    --db "postgresql://postgres@127.0.0.1:5432/optuna_gl_trail" \
    --study "gl_trail_v3" \
    --trials 600 \
    --startup-trials 200 \
    --workers $WORKERS \
    --params-out "config/trial_gl_trail_best.json"
