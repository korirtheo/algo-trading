#!/bin/bash
# Engine watchdog: restarts the algotrader container if the 2-min bar
# aggregation pipeline goes silent during market hours.
#
# Symptom we're guarding against (observed 2026-06-22):
#   Container was "healthy" but on_2min_bar callback never fired all day,
#   resulting in zero entries despite 5+ qualifying signals.
#
# Run via cron every 2 minutes during market hours:
#   */2 9-16 * * 1-5 /home/ubuntu/engine_watchdog.sh >> /var/log/watchdog.log 2>&1
#
# Logic:
#   1. Skip if outside US market hours (9:30 ET to 16:00 ET) — convert to UTC
#   2. Get the most recent "EMIT 2min" log line timestamp from docker
#   3. If older than MAX_SILENCE_SEC (300 = 5 min), restart container
#   4. Otherwise no-op

set -e

CONTAINER="algotrader"
MAX_SILENCE_SEC=300       # 5 minutes
COOLDOWN_FILE="/tmp/algotrader_watchdog_cooldown"
COOLDOWN_SEC=600          # don't restart more than once per 10 min

log() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] WATCHDOG: $*"
}

# --- 1. Market-hours check (ET) ---
# Get current ET time. AWS instance is UTC; ET in summer = UTC-4 (EDT).
# Quick check using TZ:
ET_HOUR=$(TZ='America/New_York' date +%H)
ET_MIN=$(TZ='America/New_York' date +%M)
ET_DOW=$(TZ='America/New_York' date +%u)  # 1=Mon..7=Sun

# Skip weekends
if [ "$ET_DOW" -gt 5 ]; then
    exit 0
fi

# Market open 9:30 ET, close 16:00 ET
# Convert HH:MM to minutes-since-midnight for comparison
NOW_MIN=$((10#$ET_HOUR * 60 + 10#$ET_MIN))
MARKET_OPEN_MIN=$((9*60 + 30))    # 570
MARKET_CLOSE_MIN=$((16*60))       # 960

# Only watch during market hours + 5 min warmup buffer after open
if [ "$NOW_MIN" -lt $((MARKET_OPEN_MIN + 5)) ] || [ "$NOW_MIN" -gt "$MARKET_CLOSE_MIN" ]; then
    exit 0
fi

# --- 2. Cooldown check (avoid restart loops) ---
if [ -f "$COOLDOWN_FILE" ]; then
    LAST_RESTART=$(cat "$COOLDOWN_FILE")
    AGE=$(( $(date +%s) - LAST_RESTART ))
    if [ "$AGE" -lt "$COOLDOWN_SEC" ]; then
        log "skip (cooldown — last restart ${AGE}s ago, cooldown ${COOLDOWN_SEC}s)"
        exit 0
    fi
fi

# --- 3. Check container is even running ---
STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "missing")
if [ "$STATUS" != "running" ]; then
    log "container status=$STATUS — restarting"
    date +%s > "$COOLDOWN_FILE"
    docker restart "$CONTAINER" || log "restart failed"
    exit 0
fi

# --- 4. Get last EMIT 2min log timestamp ---
# Last 1000 log lines should easily cover ~5 min of activity.
LAST_EMIT=$(docker logs --tail 2000 "$CONTAINER" 2>&1 | grep "EMIT 2min" | tail -1 || true)

if [ -z "$LAST_EMIT" ]; then
    # No EMIT in recent logs — could be startup; check uptime
    STARTED_AT=$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER")
    STARTED_TS=$(date -d "$STARTED_AT" +%s)
    UPTIME=$(( $(date +%s) - STARTED_TS ))
    if [ "$UPTIME" -lt 300 ]; then
        log "no EMIT 2min yet but uptime ${UPTIME}s — within warmup, skip"
        exit 0
    fi
    log "no EMIT 2min in last 2000 lines, uptime ${UPTIME}s — RESTARTING"
    date +%s > "$COOLDOWN_FILE"
    docker restart "$CONTAINER"
    exit 0
fi

# Parse timestamp from log line. Engine logs use ET timestamps like:
#   14:25:00 ET [INFO] live.streamer: EMIT 2min: HQ c=40.16 count=1
EMIT_TIME_ET=$(echo "$LAST_EMIT" | grep -oE '^[0-9]{2}:[0-9]{2}:[0-9]{2} ET' | head -1 | awk '{print $1}')

if [ -z "$EMIT_TIME_ET" ]; then
    log "could not parse timestamp from: $LAST_EMIT"
    exit 0
fi

# Convert HH:MM:SS to ET seconds since midnight
EMIT_H=$(echo "$EMIT_TIME_ET" | cut -d: -f1)
EMIT_M=$(echo "$EMIT_TIME_ET" | cut -d: -f2)
EMIT_S=$(echo "$EMIT_TIME_ET" | cut -d: -f3)
EMIT_SEC=$((10#$EMIT_H * 3600 + 10#$EMIT_M * 60 + 10#$EMIT_S))
NOW_SEC=$((10#$ET_HOUR * 3600 + 10#$ET_MIN * 60 + $(TZ='America/New_York' date +%S | sed 's/^0*//')))

AGE=$((NOW_SEC - EMIT_SEC))
# Handle midnight wrap (shouldn't happen during market hours but defensive)
if [ "$AGE" -lt 0 ]; then
    AGE=0
fi

if [ "$AGE" -ge "$MAX_SILENCE_SEC" ]; then
    log "last EMIT 2min ${AGE}s ago (>${MAX_SILENCE_SEC}s threshold) — RESTARTING container"
    date +%s > "$COOLDOWN_FILE"
    docker restart "$CONTAINER"
else
    # Healthy — quiet exit, no log
    exit 0
fi
