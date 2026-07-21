#!/bin/bash
# Deployment script for dashboard improvements + re-entry disabled
# Run from your machine that has SSH access to AWS

set -e

echo "============================================================"
echo "Deploying Dashboard Improvements + Re-entry Disabled"
echo "============================================================"

# 1. Copy updated files to AWS
echo ""
echo "Step 1: Copying files to AWS..."
scp -P 2222 test_green_candle_combined.py ubuntu@54.172.65.25:/home/ubuntu/algo-trading/
scp -P 2222 -r dashboard/backend ubuntu@54.172.65.25:/home/ubuntu/algo-trading/dashboard/
scp -P 2222 -r dashboard/frontend/dist ubuntu@54.172.65.25:/home/ubuntu/algo-trading/dashboard/frontend/

# 2. Copy to Docker container
echo ""
echo "Step 2: Copying to Docker container..."
ssh -p 2222 ubuntu@54.172.65.25 << 'EOF'
docker cp /home/ubuntu/algo-trading/test_green_candle_combined.py algotrader:/app/
docker cp /home/ubuntu/algo-trading/dashboard/backend/. algotrader:/app/dashboard/backend/
docker cp /home/ubuntu/algo-trading/dashboard/frontend/dist/. algotrader:/app/dashboard/frontend/dist/
EOF

# 3. Restart dashboard in container
echo ""
echo "Step 3: Restarting dashboard..."
ssh -p 2222 ubuntu@54.172.65.25 "docker exec algotrader supervisorctl restart dashboard"

echo ""
echo "============================================================"
echo "Deployment Complete!"
echo "============================================================"
echo ""
echo "Changes deployed:"
echo "  ✓ Re-entry DISABLED (marginal +0.9% impact)"
echo "  ✓ Trade log date navigation (← / → buttons)"
echo "  ✓ Entry/Exit times displayed"
echo "  ✓ Deployed amount (position size) shown"
echo "  ✓ Wins/losses stats in header"
echo ""
echo "Dashboard: http://54.172.65.25/"
echo ""
