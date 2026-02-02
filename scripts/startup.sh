#!/bin/bash
# =============================================================================
# FARNSWORTH COLLECTIVE STARTUP SCRIPT
# =============================================================================
# "Good news everyone! The collective is initializing!"
#
# This script starts all services for the Farnsworth AI Swarm:
# - Ollama (local models: Phi-4, DeepSeek-R1, Llama 3.2)
# - Main web server (FastAPI on port 8080)
# - Grok conversation thread (15-min monitoring)
# - Meme scheduler (4-hour interval)
# - Claude tmux session
# - Evolution loop (optional)
#
# Usage: ./scripts/startup.sh [--all] [--minimal] [--restart]
#   --all      Start everything including evolution loop
#   --minimal  Start only essential services (server + ollama)
#   --restart  Kill existing processes before starting
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE="/workspace/Farnsworth"
LOG_DIR="/tmp"
export PYTHONPATH="$WORKSPACE"
export COQUI_TOS_AGREED=1

echo -e "${CYAN}"
echo "============================================"
echo "   FARNSWORTH COLLECTIVE STARTUP"
echo "   11 AIs. One Mind. Zero Limits."
echo "============================================"
echo -e "${NC}"

cd "$WORKSPACE"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null
fi

# Parse arguments
START_ALL=false
MINIMAL=false
RESTART=false
for arg in "$@"; do
    case $arg in
        --all)
            START_ALL=true
            ;;
        --minimal)
            MINIMAL=true
            ;;
        --restart)
            RESTART=true
            ;;
    esac
done

# Function to check if a process is running
is_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

# Function to wait for a service
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1

    echo -n "  Waiting for $name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}OK${NC}"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}TIMEOUT${NC}"
    return 1
}

# Kill existing processes if --restart
if [ "$RESTART" = true ]; then
    echo -e "${YELLOW}Stopping existing services...${NC}"
    pkill -f "python.*farnsworth.web.server" 2>/dev/null || true
    pkill -f "grok_fresh_thread" 2>/dev/null || true
    pkill -f "meme_scheduler" 2>/dev/null || true
    sleep 3
    echo -e "  ${GREEN}Services stopped${NC}"
fi

# =============================================================================
# 1. OLLAMA (Local Models)
# =============================================================================
echo -e "${YELLOW}[1/6] Ollama & Local Models...${NC}"

if is_running "ollama serve"; then
    echo -e "  ${GREEN}✓ Ollama already running${NC}"
else
    echo "  Starting Ollama server..."
    nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    sleep 3
fi

# Verify Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [m['name'] for m in data.get('models', [])]
    print(', '.join(models[:4]))
except: print('unknown')
" 2>/dev/null)
    echo -e "  ${GREEN}✓ Ollama ready: $MODELS${NC}"
else
    echo -e "  ${RED}✗ Ollama failed to start${NC}"
fi

# =============================================================================
# 2. MAIN WEB SERVER
# =============================================================================
echo -e "${YELLOW}[2/6] Main Web Server...${NC}"

if is_running "farnsworth.web.server"; then
    echo -e "  ${GREEN}✓ Server already running${NC}"
else
    echo "  Starting FastAPI server..."
    nohup python3 -m farnsworth.web.server > "$LOG_DIR/farnsworth_server.log" 2>&1 &
    wait_for_service "http://localhost:8080/health" "Server"
fi

# Verify health
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo -e "  ${GREEN}✓ Server healthy${NC}"
else
    echo -e "  ${RED}✗ Server not responding${NC}"
fi

# =============================================================================
# 3. GROK CONVERSATION THREAD
# =============================================================================
if [ "$MINIMAL" = false ]; then
    echo -e "${YELLOW}[3/6] Grok Conversation Thread...${NC}"

    if is_running "grok_fresh_thread.py"; then
        echo -e "  ${GREEN}✓ Grok thread already running${NC}"
    else
        echo "  Starting Grok monitor (15-min interval)..."
        nohup python3 scripts/grok_fresh_thread.py > "$LOG_DIR/grok_fresh.log" 2>&1 &
        sleep 2
        if is_running "grok_fresh_thread.py"; then
            echo -e "  ${GREEN}✓ Grok thread started${NC}"
        else
            echo -e "  ${RED}✗ Failed to start${NC}"
        fi
    fi
else
    echo -e "${YELLOW}[3/6] Grok Thread (skipped - minimal mode)${NC}"
fi

# =============================================================================
# 4. MEME SCHEDULER
# =============================================================================
if [ "$MINIMAL" = false ]; then
    echo -e "${YELLOW}[4/6] Meme Scheduler...${NC}"

    if is_running "meme_scheduler"; then
        echo -e "  ${GREEN}✓ Meme scheduler already running${NC}"
    else
        echo "  Starting meme scheduler (4-hour interval)..."
        nohup python3 -m farnsworth.integration.x_automation.meme_scheduler > "$LOG_DIR/meme_scheduler.log" 2>&1 &
        sleep 2
        if is_running "meme_scheduler"; then
            echo -e "  ${GREEN}✓ Meme scheduler started${NC}"
        else
            echo -e "  ${RED}✗ Failed to start${NC}"
        fi
    fi
else
    echo -e "${YELLOW}[4/6] Meme Scheduler (skipped - minimal mode)${NC}"
fi

# =============================================================================
# 5. TMUX SESSIONS
# =============================================================================
echo -e "${YELLOW}[5/6] Tmux Sessions...${NC}"

# Claude session
if tmux has-session -t claude 2>/dev/null; then
    echo -e "  ${GREEN}✓ claude tmux exists${NC}"
else
    tmux new-session -d -s claude -c "$WORKSPACE"
    echo -e "  ${GREEN}✓ claude tmux created${NC}"
fi

# Farnsworth Claude session
if tmux has-session -t farnsworth_claude 2>/dev/null; then
    echo -e "  ${GREEN}✓ farnsworth_claude tmux exists${NC}"
else
    tmux new-session -d -s farnsworth_claude -c "$WORKSPACE"
    echo -e "  ${GREEN}✓ farnsworth_claude tmux created${NC}"
fi

# =============================================================================
# 6. EVOLUTION LOOP (Optional)
# =============================================================================
if [ "$START_ALL" = true ]; then
    echo -e "${YELLOW}[6/6] Evolution Loop...${NC}"

    if is_running "evolution_loop"; then
        echo -e "  ${GREEN}✓ Evolution loop already running${NC}"
    else
        if [ -f "farnsworth/core/collective/evolution_loop.py" ]; then
            nohup python3 -m farnsworth.core.collective.evolution_loop > "$LOG_DIR/evolution_loop.log" 2>&1 &
            sleep 2
            echo -e "  ${GREEN}✓ Evolution loop started${NC}"
        else
            echo -e "  ${YELLOW}○ Evolution loop module not found${NC}"
        fi
    fi
else
    echo -e "${YELLOW}[6/6] Evolution Loop (use --all to enable)${NC}"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}   COLLECTIVE INITIALIZED${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

echo -e "${GREEN}Services:${NC}"
is_running "ollama serve" && echo "  ✓ Ollama" || echo "  ✗ Ollama"
is_running "farnsworth.web.server" && echo "  ✓ Main Server" || echo "  ✗ Main Server"
is_running "grok_fresh_thread" && echo "  ✓ Grok Thread" || echo "  ○ Grok Thread"
is_running "meme_scheduler" && echo "  ✓ Meme Scheduler" || echo "  ○ Meme Scheduler"

echo ""
echo -e "${GREEN}Tmux Sessions:${NC}"
tmux list-sessions 2>/dev/null | sed 's/^/  /'

echo ""
echo -e "${GREEN}Endpoints:${NC}"
echo "  Web:    https://ai.farnsworth.cloud"
echo "  Health: https://ai.farnsworth.cloud/health"
echo "  Chat:   https://ai.farnsworth.cloud/api/chat"

echo ""
echo -e "${GREEN}Logs:${NC}"
echo "  tail -f $LOG_DIR/farnsworth_server.log"
echo "  tail -f $LOG_DIR/grok_fresh.log"

echo ""
echo -e "${CYAN}Good news everyone! The collective is ready.${NC}"
