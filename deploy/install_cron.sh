#!/bin/bash
# Install the post-close reconcile cron on the AWS host (run as root / sudo).
# Deploys the cron file to /etc/cron.d/ so it runs inside the algotrader container.
#
# Usage (on AWS, after git pull):
#   sudo bash deploy/install_cron.sh
set -euo pipefail

CRON_SRC="$(dirname "$0")/cron_post_close_reconcile.txt"
CRON_DEST="/etc/cron.d/algo-reconcile"

if [ ! -f "$CRON_SRC" ]; then
  echo "ERROR: $CRON_SRC not found (run from repo root: sudo bash deploy/install_cron.sh)"
  exit 1
fi

# Verify the container is up before wiring the cron
if ! docker ps --format '{{.Names}}' | grep -q '^algotrader$'; then
  echo "WARNING: algotrader container not running — cron will fail until it is."
fi

cp "$CRON_SRC" "$CRON_DEST"
chmod 644 "$CRON_DEST"
echo "Installed cron: $CRON_DEST"
cat "$CRON_DEST"

# Sanity check: cron.d files must not have world/group-writable perms (Debian checks this)
chown root:root "$CRON_DEST"

echo ""
echo "Verify with:"
echo "  grep -v '^#' $CRON_DEST | crontab -  # dry parse (as any user)"
echo "  docker exec algotrader python /app/scripts/reconcile/post_close_reconcile.py --dry-run --date <today>"
