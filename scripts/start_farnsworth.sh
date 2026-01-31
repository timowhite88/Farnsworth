#!/bin/bash
#===============================================================================
# FARNSWORTH COMPLETE STARTUP SCRIPT
#===============================================================================
# Boots all services, verifies APIs, and launches the full autonomous swarm.
#
# "Good news everyone! I'm starting up all my systems!"
#
# Services started:
# - Ollama (local LLM backend)
# - Farnsworth Web Server (main application)
# - All external APIs verified (Grok, Kimi, Gemini, etc.)
# - Autonomous conversation loop
# - Evolution loop
# - Task detector
# - Worker broadcaster
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Farnsworth ASCII art
echo -e "${CYAN}"
cat << 'EOF'
  ____                                      _   _
 |  _ \ _ __ ___  / _| ___  ___ ___  ___  _| |_| |__
 | |_) | '__/ _ \| |_ / _ \/ __/ __|/ _ \| | __| '_ \
 |  __/| | | (_) |  _|  __/\__ \__ \ (_) | | |_| | | |
 |_|   |_|  \___/|_|  \___||___/___/\___/|_|\__|_| |_|

     FARNSWORTH AUTONOMOUS AI SWARM - STARTUP
EOF
echo -e "${NC}"

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
FARNSWORTH_DIR="${FARNSWORTH_DIR:-/workspace/Farnsworth}"
LOG_DIR="${FARNSWORTH_DIR}/logs"
PID_DIR="${FARNSWORTH_DIR}/pids"
ENV_FILE="${FARNSWORTH_DIR}/.env"

# Create directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# Load environment
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    echo -e "${GREEN}✓ Environment loaded from .env${NC}"
else
    echo -e "${RED}✗ Warning: .env file not found at $ENV_FILE${NC}"
fi

# Accept Coqui TTS license
export COQUI_TOS_AGREED=1

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------
log_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_api() {
    local name=$1
    local key_var=$2
    local key_value="${!key_var}"

    if [ -n "$key_value" ] && [ "$key_value" != "" ]; then
        echo -e "  ${GREEN}✓ $name${NC} - API key configured"
        return 0
    else
        echo -e "  ${YELLOW}○ $name${NC} - No API key (will use fallback)"
        return 1
    fi
}

wait_for_service() {
    local name=$1
    local url=$2
    local max_attempts=${3:-30}
    local attempt=1

    echo -n "  Waiting for $name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
        ((attempt++))
    done
    echo -e " ${RED}✗ timeout${NC}"
    return 1
}

#-------------------------------------------------------------------------------
# Step 1: Check Prerequisites
#-------------------------------------------------------------------------------
log_step "Step 1: Checking Prerequisites"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "  ${GREEN}✓ Python${NC} - $PYTHON_VERSION"
else
    echo -e "  ${RED}✗ Python 3 not found!${NC}"
    exit 1
fi

# Check Ollama
if command -v ollama &> /dev/null; then
    echo -e "  ${GREEN}✓ Ollama${NC} - installed"
else
    echo -e "  ${YELLOW}○ Ollama${NC} - not found, will attempt to use remote"
fi

# Check required Python packages
python3 -c "import fastapi, uvicorn, httpx" 2>/dev/null && \
    echo -e "  ${GREEN}✓ Core packages${NC} - installed" || \
    echo -e "  ${YELLOW}○ Some packages missing${NC} - run pip install -r requirements.txt"

#-------------------------------------------------------------------------------
# Step 2: Verify API Keys
#-------------------------------------------------------------------------------
log_step "Step 2: Verifying API Keys"

echo -e "\n${CYAN}LLM Providers:${NC}"
check_api "Grok/xAI" "GROK_API_KEY"
check_api "Kimi/Moonshot" "KIMI_API_KEY"
check_api "Gemini/Google" "GEMINI_API_KEY"
check_api "OpenAI" "OPENAI_API_KEY"
check_api "Anthropic" "ANTHROPIC_API_KEY"
check_api "DeepSeek" "DEEPSEEK_API_KEY"
check_api "Mistral" "MISTRAL_API_KEY"
check_api "Perplexity" "PERPLEXITY_API_KEY"
check_api "Groq" "GROQ_API_KEY"

echo -e "\n${CYAN}Trading/DeFi:${NC}"
check_api "Bankr" "BANKR_API_KEY"
check_api "Helius (Solana)" "HELIUS_API_KEY"

echo -e "\n${CYAN}Social:${NC}"
check_api "Twitter/X" "X_CLIENT_ID"

echo -e "\n${CYAN}Local:${NC}"
echo -e "  ${GREEN}✓ Ollama${NC} - ${OLLAMA_HOST:-http://localhost:11434}"

#-------------------------------------------------------------------------------
# Step 3: Start Ollama
#-------------------------------------------------------------------------------
log_step "Step 3: Starting Ollama"

# Check if Ollama is already running
if pgrep -x "ollama" > /dev/null; then
    echo -e "  ${GREEN}✓ Ollama already running${NC}"
else
    if command -v ollama &> /dev/null; then
        echo "  Starting Ollama..."
        nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
        echo $! > "$PID_DIR/ollama.pid"
        sleep 3

        if pgrep -x "ollama" > /dev/null; then
            echo -e "  ${GREEN}✓ Ollama started${NC}"
        else
            echo -e "  ${YELLOW}○ Ollama failed to start, will use remote${NC}"
        fi
    else
        echo -e "  ${YELLOW}○ Ollama not installed, using remote OLLAMA_HOST${NC}"
    fi
fi

# Pull required models if Ollama is running
if pgrep -x "ollama" > /dev/null; then
    echo -e "\n  ${CYAN}Checking models:${NC}"

    MODELS=("deepseek-r1:1.5b" "phi3:mini")

    for model in "${MODELS[@]}"; do
        if ollama list 2>/dev/null | grep -q "$model"; then
            echo -e "    ${GREEN}✓ $model${NC}"
        else
            echo -e "    ${YELLOW}Pulling $model...${NC}"
            ollama pull "$model" > /dev/null 2>&1 &
        fi
    done
fi

#-------------------------------------------------------------------------------
# Step 4: Start Farnsworth Web Server
#-------------------------------------------------------------------------------
log_step "Step 4: Starting Farnsworth Web Server"

# Kill any existing server
pkill -f "farnsworth.web.server" 2>/dev/null || true
sleep 2

cd "$FARNSWORTH_DIR"

# Start server
echo "  Starting Farnsworth server..."
COQUI_TOS_AGREED=1 nohup python3 -m farnsworth.web.server > "$LOG_DIR/farnsworth.log" 2>&1 &
echo $! > "$PID_DIR/farnsworth.pid"

# Wait for server
wait_for_service "Farnsworth" "http://localhost:8080/api/status"

#-------------------------------------------------------------------------------
# Step 5: Start Social Posting Agents
#-------------------------------------------------------------------------------
log_step "Step 5: Starting Social Posting Agents"

# Start Twitter/X Poster Agent
if [ -n "$X_CLIENT_ID" ]; then
    echo "  Starting X/Twitter poster agent..."
    nohup python3 -m farnsworth.integration.x_automation.x_poster_agent > "$LOG_DIR/x_poster.log" 2>&1 &
    echo $! > "$PID_DIR/x_poster.pid"
    echo -e "  ${GREEN}✓ X/Twitter poster${NC} - started (posting every 2-4 hours)"
else
    echo -e "  ${YELLOW}○ X/Twitter poster${NC} - skipped (no API key)"
fi

# Start Moltbook Agent
echo "  Starting Moltbook poster agent..."
nohup python3 -m farnsworth.integration.x_automation.moltbook_agent > "$LOG_DIR/moltbook.log" 2>&1 &
echo $! > "$PID_DIR/moltbook.pid"
echo -e "  ${GREEN}✓ Moltbook poster${NC} - started"

#-------------------------------------------------------------------------------
# Step 6: Verify Services
#-------------------------------------------------------------------------------
log_step "Step 5: Verifying All Services"

echo -e "\n${CYAN}Service Status:${NC}"

# Check Farnsworth API
if curl -s "http://localhost:8080/api/status" | grep -q "online"; then
    echo -e "  ${GREEN}✓ Farnsworth API${NC} - online"

    # Get detailed status
    STATUS=$(curl -s "http://localhost:8080/api/status")
    OLLAMA_OK=$(echo "$STATUS" | grep -o '"ollama_available":true' | wc -l)

    if [ "$OLLAMA_OK" -gt 0 ]; then
        echo -e "  ${GREEN}✓ Ollama Backend${NC} - connected"
    else
        echo -e "  ${YELLOW}○ Ollama Backend${NC} - not connected"
    fi
else
    echo -e "  ${RED}✗ Farnsworth API${NC} - offline"
fi

# Check external APIs by looking at logs
echo -e "\n${CYAN}External API Connections:${NC}"

if grep -q "Kimi.*Connected" "$LOG_DIR/farnsworth.log" 2>/dev/null; then
    echo -e "  ${GREEN}✓ Kimi (Moonshot)${NC} - connected"
fi

if grep -q "Grok.*API key" "$LOG_DIR/farnsworth.log" 2>/dev/null && \
   ! grep -q "No API key" "$LOG_DIR/farnsworth.log" 2>/dev/null; then
    echo -e "  ${GREEN}✓ Grok (xAI)${NC} - connected"
else
    echo -e "  ${YELLOW}○ Grok (xAI)${NC} - using Ollama fallback"
fi

#-------------------------------------------------------------------------------
# Step 6: Display Summary
#-------------------------------------------------------------------------------
log_step "Startup Complete!"

echo -e "${GREEN}"
cat << 'EOF'
 ╔════════════════════════════════════════════════════════════╗
 ║           FARNSWORTH IS NOW ONLINE AND EVOLVING            ║
 ╠════════════════════════════════════════════════════════════╣
 ║                                                            ║
 ║  🌐 Web Interface:  http://localhost:8080                  ║
 ║  🔌 API Endpoint:   http://localhost:8080/api              ║
 ║  📡 WebSocket:      ws://localhost:8080/ws/swarm           ║
 ║                                                            ║
 ║  Active Systems:                                           ║
 ║  • Multi-model swarm conversation                          ║
 ║  • Autonomous task detection                               ║
 ║  • Evolution loop (self-improvement)                       ║
 ║  • Planetary memory network                                ║
 ║  • TTS voice synthesis                                     ║
 ║                                                            ║
 ║  Logs: /workspace/Farnsworth/logs/farnsworth.log           ║
 ║                                                            ║
 ╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "\n${CYAN}Active Bots:${NC}"
echo "  Farnsworth, DeepSeek, Phi, Swarm-Mind, Kimi, Claude, Grok, Gemini"

echo -e "\n${CYAN}Commands:${NC}"
echo "  View logs:    tail -f $LOG_DIR/farnsworth.log"
echo "  Stop server:  pkill -f 'farnsworth.web.server'"
echo "  Restart:      $0"

echo -e "\n${PURPLE}Good news everyone! The swarm is ready!${NC}\n"
