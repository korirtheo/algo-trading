#!/bin/bash

# Run main process
echo "Starting main process..."
python3 scripts/optimize/optuna_g511_l626_v3_full_no_trail.py --trials 600 --startup-trials 200 > /tmp/optuna_g511_l626_v3_full_no_trail_main.log 2>&1 &
MAIN_PID=$!

# Wait for main to load data
sleep 150

# Start 8 workers
echo "Starting 8 workers..."
for i in {1..8}; do
  python3 scripts/optimize/optuna_g511_l626_v3_full_no_trail.py --trials 600 --startup-trials 200 --worker > /tmp/optuna_g511_l626_v3_full_no_trail_worker_$i.log 2>&1 &
done

# Wait for all
wait

echo "Done"
