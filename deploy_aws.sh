#!/bin/bash
# Quick deployment script for AWS instance
# Pulls latest code, rebuilds Docker image, restarts service

set -e

echo "==========================================="
echo "Deploying to AWS"
echo "==========================================="

# Pull latest code
echo ""
echo "→ Pulling latest code from git..."
git pull origin main

# Rebuild Docker image
echo ""
echo "→ Rebuilding Docker image..."
docker-compose build

# Restart service
echo ""
echo "→ Restarting service..."
docker-compose down
docker-compose up -d

# Show status
echo ""
echo "→ Service status:"
docker-compose ps

# Show logs
echo ""
echo "→ Tailing logs (Ctrl+C to exit)..."
docker-compose logs -f --tail=100
