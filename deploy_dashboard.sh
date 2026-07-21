#!/bin/bash
# Deploy dashboard updates to AWS
# Usage: ./deploy_dashboard.sh

set -e

SERVER="ubuntu@54.172.65.25"
KEY="trading-key-v2.pem"
CONTAINER="algotrader"

echo "=== Deploying Dashboard to AWS ==="

# 1. Copy files to host
echo "1. Copying files to host..."
scp -P 2222 -i "$KEY" -r dashboard/frontend/dist "$SERVER:~/algo-trading/dashboard/frontend/"

# 2. Copy files into running container
echo "2. Copying files into container..."
ssh -p 2222 -i "$KEY" "$SERVER" "docker cp ~/algo-trading/dashboard/frontend/dist/index.html $CONTAINER:/app/dashboard/frontend/dist/index.html"

# Get the current bundle name from index.html
BUNDLE_JS=$(grep -oP 'index-[^.]+\.js' dashboard/frontend/dist/index.html | head -1)
BUNDLE_CSS=$(grep -oP 'index-[^.]+\.css' dashboard/frontend/dist/index.html | head -1)

echo "   Copying $BUNDLE_JS..."
ssh -p 2222 -i "$KEY" "$SERVER" "docker cp ~/algo-trading/dashboard/frontend/dist/assets/$BUNDLE_JS $CONTAINER:/app/dashboard/frontend/dist/assets/"

echo "   Copying $BUNDLE_CSS..."
ssh -p 2222 -i "$KEY" "$SERVER" "docker cp ~/algo-trading/dashboard/frontend/dist/assets/$BUNDLE_CSS $CONTAINER:/app/dashboard/frontend/dist/assets/"

# 3. Verify deployment
echo "3. Verifying deployment..."
ssh -p 2222 -i "$KEY" "$SERVER" "docker exec $CONTAINER ls -lh /app/dashboard/frontend/dist/assets/$BUNDLE_JS" | grep -q "$BUNDLE_JS" && echo "   ✓ JS bundle deployed"
ssh -p 2222 -i "$KEY" "$SERVER" "docker exec $CONTAINER ls -lh /app/dashboard/frontend/dist/assets/$BUNDLE_CSS" | grep -q "$BUNDLE_CSS" && echo "   ✓ CSS bundle deployed"

echo ""
echo "✓ Dashboard deployment complete!"
echo ""
echo "Refresh your browser to see changes (Ctrl+R or Cmd+R)"
echo "No cache clear needed - new bundle names force browser reload"
