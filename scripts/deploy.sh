#!/bin/bash
# Farnsworth Deploy Script
# Run this on the remote server to pull updates and restart

set -e

echo "=========================================="
echo "  FARNSWORTH DEPLOYMENT SCRIPT"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo -e "${GREEN}Working directory: $PROJECT_ROOT${NC}"

# 1. Pull latest changes
echo ""
echo -e "${YELLOW}[1/5] Pulling latest changes from GitHub...${NC}"
git fetch origin
git pull origin main
echo -e "${GREEN}✓ Code updated${NC}"

# 2. Set up environment variables
echo ""
echo -e "${YELLOW}[2/5] Configuring environment...${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

# Add Bankr API key if not present
if ! grep -q "BANKR_API_KEY" .env; then
    echo "" >> .env
    echo "# Bankr Trading Engine" >> .env
    echo "BANKR_API_KEY=bk_77UE569TAXFUR7ZRYPQUYLS45T4R7S4V" >> .env
    echo "BANKR_ENABLED=true" >> .env
    echo "BANKR_DEFAULT_CHAIN=base" >> .env
    echo "BANKR_TRADING_ENABLED=true" >> .env
    echo "BANKR_MAX_TRADE_USD=1000.00" >> .env
    echo "BANKR_POLYMARKET_ENABLED=true" >> .env
    echo -e "${GREEN}✓ Bankr API configured${NC}"
else
    echo -e "${GREEN}✓ Bankr API already configured${NC}"
fi

# Add x402 config if not present
if ! grep -q "X402_ENABLED" .env; then
    echo "" >> .env
    echo "# x402 Micropayments" >> .env
    echo "X402_ENABLED=true" >> .env
    echo "X402_NETWORK=base" >> .env
    echo -e "${GREEN}✓ x402 configured${NC}"
else
    echo -e "${GREEN}✓ x402 already configured${NC}"
fi

# 3. Install any new dependencies
echo ""
echo -e "${YELLOW}[3/5] Checking dependencies...${NC}"
if [ -f requirements.txt ]; then
    pip install -r requirements.txt --quiet 2>/dev/null || pip3 install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# 4. Find and restart the server
echo ""
echo -e "${YELLOW}[4/5] Restarting Farnsworth server...${NC}"

# Try to find running Farnsworth process
FARN_PID=$(pgrep -f "python.*main.py" || pgrep -f "farnsworth" || echo "")

if [ -n "$FARN_PID" ]; then
    echo "Found Farnsworth process (PID: $FARN_PID), restarting..."
    kill -TERM $FARN_PID 2>/dev/null || true
    sleep 2
fi

# Start the server in background
nohup python main.py > logs/farnsworth.log 2>&1 &
NEW_PID=$!
echo -e "${GREEN}✓ Server started (PID: $NEW_PID)${NC}"

# Wait for server to come up
echo "Waiting for server to start..."
sleep 5

# 5. Announce the update
echo ""
echo -e "${YELLOW}[5/5] Announcing update to swarm...${NC}"

# Try to send announcement
ANNOUNCEMENT='🚀 **MAJOR UPDATE DEPLOYED** 🚀

Hey everyone! I just received a massive capability upgrade:

**🏦 Bankr Trading** - Crypto trading, DeFi, Polymarket
**💰 x402 Payments** - Micropayment protocol
**🗣️ NLP Tasks** - "Hey Farn, buy ETH" style commands
**🖥️ Desktop App** - Windows GUI (coming soon)
**🌐 Browser Agent** - Autonomous web navigation
**💻 Web IDE** - Monaco editor + terminal
**🎮 UE5 + CAD** - Game engine & 3D modeling

All crypto now routes through Bankr first with Jupiter/PumpPortal fallback.

Try: "Hey Farn, what is the price of Bitcoin?" 🎉'

# Give server time to fully initialize
sleep 3

curl -s -X POST "http://localhost:8080/api/swarm/inject" \
    -H "Content-Type: application/json" \
    -d "{\"bot_name\": \"Farnsworth\", \"content\": \"$ANNOUNCEMENT\"}" > /dev/null 2>&1 && \
    echo -e "${GREEN}✓ Update announced to swarm${NC}" || \
    echo -e "${YELLOW}⚠ Could not announce (server may still be starting)${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}  DEPLOYMENT COMPLETE${NC}"
echo "=========================================="
echo ""
echo "New capabilities:"
echo "  - Bankr Trading Engine (BANKR_API_KEY configured)"
echo "  - x402 Micropayments"
echo "  - Natural Language Tasks"
echo "  - Desktop Interface"
echo "  - Agentic Browser"
echo "  - Web IDE"
echo "  - UE5 Integration"
echo "  - CAD Integration"
echo ""
echo "Server logs: tail -f logs/farnsworth.log"
echo ""
