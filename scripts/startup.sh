#!/bin/bash
# Farnsworth Auto-Startup Script
# Run this after RunPod reset or server restart

echo "========================================"
echo "FARNSWORTH STARTUP SEQUENCE"
echo "========================================"

cd /workspace/Farnsworth

# Load environment
export $(grep -v '^#' .env | xargs)
export COQUI_TOS_AGREED=1

# Kill any existing processes
pkill -f "python.*farnsworth" 2>/dev/null
pkill -f "uvicorn.*server" 2>/dev/null
sleep 2

echo "[1/4] Starting main Farnsworth server..."
nohup python3 -m farnsworth.web.server > /tmp/farnsworth.log 2>&1 &
sleep 10

# Verify server started
if curl -s localhost:8080/health | grep -q "healthy"; then
    echo "      Server: ONLINE"
else
    echo "      Server: FAILED - check /tmp/farnsworth.log"
    exit 1
fi

echo "[2/4] Checking API status..."
STATUS=$(curl -s localhost:8080/api/status)
echo "      $STATUS" | head -c 100

echo "[3/4] Checking social media manager..."
SOCIAL=$(curl -s localhost:8080/api/social/status)
echo "      $SOCIAL"

echo "[4/4] Checking X OAuth2 status..."
if echo "$SOCIAL" | grep -q '"x_configured":true'; then
    echo "      X Posting: ENABLED"
else
    echo "      X Posting: NEEDS AUTH - visit https://ai.farnsworth.cloud/x/auth"
fi

echo ""
echo "========================================"
echo "FARNSWORTH STARTUP COMPLETE"
echo "========================================"
echo ""
echo "Web Interface: https://ai.farnsworth.cloud"
echo "Health Check:  curl localhost:8080/health"
echo "Logs:          tail -f /tmp/farnsworth.log"
echo ""
