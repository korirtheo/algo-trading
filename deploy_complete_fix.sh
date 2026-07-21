#!/bin/bash
# Complete deployment: SQLite persistence + timezone fixes + dashboard improvements
# 2026-07-16

set -e

echo "============================================================"
echo "Deploying: SQLite Persistence + Timezone + Dashboard Fixes"
echo "============================================================"

PEM_KEY="trading-key-v2.pem"
AWS_HOST="ubuntu@54.172.65.25"
PORT="2222"

echo ""
echo "Step 1: Copying files to AWS..."
scp -P $PORT -i $PEM_KEY live/persistence_db.py $AWS_HOST:/home/ubuntu/algo-trading/live/
scp -P $PORT -i $PEM_KEY live/engine_combined.py $AWS_HOST:/home/ubuntu/algo-trading/live/
scp -P $PORT -i $PEM_KEY dashboard/backend/routers/trades.py $AWS_HOST:/home/ubuntu/algo-trading/dashboard/backend/routers/
scp -P $PORT -i $PEM_KEY -r dashboard/frontend/dist $AWS_HOST:/home/ubuntu/algo-trading/dashboard/frontend/

echo ""
echo "Step 2: Copying to Docker container..."
ssh -p $PORT -i $PEM_KEY $AWS_HOST << 'EOF'
docker cp /home/ubuntu/algo-trading/live/persistence_db.py algotrader:/app/live/
docker cp /home/ubuntu/algo-trading/live/engine_combined.py algotrader:/app/live/
docker cp /home/ubuntu/algo-trading/dashboard/backend/routers/trades.py algotrader:/app/dashboard/backend/routers/
docker cp /home/ubuntu/algo-trading/dashboard/frontend/dist/. algotrader:/app/dashboard/frontend/dist/
EOF

echo ""
echo "Step 3: Restarting container..."
ssh -p $PORT -i $PEM_KEY $AWS_HOST "docker restart algotrader"

echo ""
echo "Waiting for container to start..."
sleep 10

echo ""
echo "Step 4: Verifying deployment..."
ssh -p $PORT -i $PEM_KEY $AWS_HOST << 'EOF'
echo "Checking database file..."
docker exec algotrader ls -lh /app/logs/trading.db

echo ""
echo "Checking recent logs..."
docker logs algotrader 2>&1 | tail -10
EOF

echo ""
echo "============================================================"
echo "Deployment Complete!"
echo "============================================================"
echo ""
echo "Changes deployed:"
echo "  ✅ SQLite persistence (logs/trading.db)"
echo "  ✅ All timestamps in ET timezone"
echo "  ✅ Trade log date navigation fixed"
echo "  ✅ Done strategies persist across restarts"
echo "  ✅ Exit prices persist (re-entry floor)"
echo "  ✅ Dashboard improvements (times, deployed amount, stats)"
echo ""
echo "Dashboard: http://54.172.65.25/"
echo ""
echo "Test:"
echo "  - Make a trade, restart container, verify trade still visible"
echo "  - Click ← → on trade log, verify historical dates load"
echo "  - Check all timestamps show ET not UTC"
echo ""
