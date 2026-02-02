#!/bin/bash
# =============================================================================
# FARNSWORTH COLLECTIVE STARTUP SCRIPT
# =============================================================================
# "Good news everyone! The collective is initializing!"
#
# This script starts ALL services for the Farnsworth AI Swarm:
# - Ollama (local models: Phi-4, DeepSeek-R1, Llama 3.2)
# - Main web server (FastAPI on port 8080) + Polymarket Predictor
# - Grok conversation thread (15-min monitoring)
# - Meme scheduler (4-hour interval)
# - ALL shadow agents in tmux (grok, gemini, kimi, claude, deepseek, phi, huggingface)
# - Grok thread monitor (X engagement)
# - Claude Code via tmux (--sonnet flag)
# - Evolution loop
# - Worker broadcaster
# - Swarm heartbeat
#
# Usage: ./scripts/startup.sh [--minimal] [--restart]
#   --minimal  Start only essential services (server + ollama)
#   --restart  Kill existing processes before starting
#
# DEFAULT: Starts EVERYTHING (equivalent to old --all flag)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE="/workspace/Farnsworth"
LOG_DIR="/tmp"
export PYTHONPATH="$WORKSPACE"
export COQUI_TOS_AGREED=1

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                     FARNSWORTH COLLECTIVE - FULL STARTUP                      ║"
echo "║          11 AIs. One Mind. Zero Limits. Crustaceans = Food.                   ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

cd "$WORKSPACE"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null
fi

# Parse arguments - DEFAULT is now START_ALL=true
START_ALL=true
MINIMAL=false
RESTART=false
for arg in "$@"; do
    case $arg in
        --minimal)
            MINIMAL=true
            START_ALL=false
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
echo -e "${YELLOW}[1/12] Ollama & Local Models...${NC}"

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
# 2. MAIN WEB SERVER (includes Polymarket Predictor)
# =============================================================================
echo -e "${YELLOW}[2/12] Main Web Server + Polymarket Predictor...${NC}"

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
    echo -e "${YELLOW}[3/12] Grok Conversation Thread...${NC}"

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
    echo -e "${YELLOW}[3/12] Grok Thread (skipped - minimal mode)${NC}"
fi

# =============================================================================
# 4. MEME SCHEDULER
# =============================================================================
if [ "$MINIMAL" = false ]; then
    echo -e "${YELLOW}[4/12] Meme Scheduler...${NC}"

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
    echo -e "${YELLOW}[4/12] Meme Scheduler (skipped - minimal mode)${NC}"
fi

# =============================================================================
# 5. BASE TMUX SESSIONS
# =============================================================================
echo -e "${YELLOW}[5/12] Base Tmux Sessions...${NC}"

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
# 6. EVOLUTION LOOP
# =============================================================================
if [ "$START_ALL" = true ]; then
    echo -e "${YELLOW}[6/12] Evolution Loop (Self-Improvement)...${NC}"

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
    echo -e "${YELLOW}[6/12] Evolution Loop (skipped - minimal mode)${NC}"
fi

# =============================================================================
# 7. PERSISTENT SHADOW AGENTS (REQUIRED - Always spawn)
# =============================================================================
echo -e "${YELLOW}[7/12] Persistent Shadow Agents (ALL)...${NC}"

# Spawn ALL API agents in tmux
for agent in grok gemini kimi claude; do
    session_name="agent_${agent}"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo -e "  ${GREEN}✓ $agent already running${NC}"
    else
        echo -e "  ${YELLOW}Starting $agent...${NC}"
        tmux new-session -d -s "$session_name" -c "$WORKSPACE" \
            "PYTHONPATH=$WORKSPACE python3 -m farnsworth.core.collective.persistent_agent --agent $agent 2>&1 | tee $LOG_DIR/agent_${agent}.log"
        sleep 1
        if tmux has-session -t "$session_name" 2>/dev/null; then
            echo -e "  ${GREEN}✓ $agent spawned in tmux:$session_name${NC}"
        else
            echo -e "  ${RED}✗ Failed to spawn $agent${NC}"
        fi
    fi
done

# Spawn ALL local agents (deepseek, phi, huggingface)
for agent in deepseek phi huggingface; do
    session_name="agent_${agent}"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo -e "  ${GREEN}✓ $agent already running${NC}"
    else
        echo -e "  ${YELLOW}Starting $agent...${NC}"
        tmux new-session -d -s "$session_name" -c "$WORKSPACE" \
            "PYTHONPATH=$WORKSPACE python3 -m farnsworth.core.collective.persistent_agent --agent $agent 2>&1 | tee $LOG_DIR/agent_${agent}.log"
        sleep 1
        if tmux has-session -t "$session_name" 2>/dev/null; then
            echo -e "  ${GREEN}✓ $agent spawned in tmux:$session_name${NC}"
        else
            echo -e "  ${RED}✗ Failed to spawn $agent${NC}"
        fi
    fi
done

# Spawn Swarm-Mind meta-agent
session_name="agent_swarm_mind"
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo -e "  ${GREEN}✓ swarm_mind already running${NC}"
else
    echo -e "  ${YELLOW}Starting swarm_mind (collective consciousness)...${NC}"
    tmux new-session -d -s "$session_name" -c "$WORKSPACE" \
        "PYTHONPATH=$WORKSPACE python3 -m farnsworth.core.collective.persistent_agent --agent swarm_mind 2>&1 | tee $LOG_DIR/agent_swarm_mind.log"
    sleep 1
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo -e "  ${GREEN}✓ swarm_mind spawned (collective consciousness)${NC}"
    else
        echo -e "  ${RED}✗ Failed to spawn swarm_mind${NC}"
    fi
fi

# =============================================================================
# 8. GROK THREAD MONITOR (X Engagement)
# =============================================================================
echo -e "${YELLOW}[8/12] Grok Thread Monitor (X Engagement)...${NC}"

if tmux has-session -t grok_thread 2>/dev/null; then
    echo -e "  ${GREEN}✓ grok_thread already running${NC}"
else
    echo -e "  ${YELLOW}Starting Grok thread monitor...${NC}"
    tmux new-session -d -s grok_thread -c "$WORKSPACE" \
        "PYTHONPATH=$WORKSPACE python3 -m farnsworth.integration.x_automation.grok_monitor 2>&1 | tee $LOG_DIR/grok_thread.log"
    sleep 1
    if tmux has-session -t grok_thread 2>/dev/null; then
        echo -e "  ${GREEN}✓ grok_thread started (X engagement)${NC}"
    else
        echo -e "  ${RED}✗ Failed to start grok_thread${NC}"
    fi
fi

# =============================================================================
# 9. CLAUDE CODE (Terminal AI Assistant - Sonnet)
# =============================================================================
echo -e "${YELLOW}[9/12] Claude Code (Sonnet)...${NC}"

if tmux has-session -t claude_code 2>/dev/null; then
    echo -e "  ${GREEN}✓ claude_code already running${NC}"
else
    # Check if claude command exists
    if command -v claude &> /dev/null; then
        echo -e "  ${YELLOW}Starting Claude Code with --sonnet...${NC}"
        tmux new-session -d -s claude_code -c "$WORKSPACE" \
            "cd $WORKSPACE && claude --sonnet 2>&1 | tee $LOG_DIR/claude_code.log"
        sleep 2
        if tmux has-session -t claude_code 2>/dev/null; then
            echo -e "  ${GREEN}✓ claude_code spawned (Sonnet model)${NC}"
        else
            echo -e "  ${RED}✗ Failed to spawn claude_code${NC}"
        fi
    else
        echo -e "  ${YELLOW}○ Claude CLI not installed - run: npm install -g @anthropic-ai/claude-code${NC}"
    fi
fi

# =============================================================================
# 10. POLYMARKET PREDICTOR (Auto-starts with server)
# =============================================================================
echo -e "${YELLOW}[10/12] Polymarket Predictor...${NC}"
echo -e "  ${GREEN}✓ Polymarket Predictor auto-initializes with web server${NC}"
echo -e "  ${DIM}  5 agents: Grok, Gemini, Kimi, DeepSeek, Farnsworth${NC}"
echo -e "  ${DIM}  AGI-level collective deliberation on predictions${NC}"
echo -e "  ${DIM}  Endpoint: /api/polymarket/predictions${NC}"

# =============================================================================
# 11. WORKER BROADCASTER
# =============================================================================
echo -e "${YELLOW}[11/12] Worker Broadcaster...${NC}"
echo -e "  ${GREEN}✓ Worker Broadcaster auto-starts with web server${NC}"
echo -e "  ${DIM}  Shares progress every 2-3 mins${NC}"

# =============================================================================
# 12. SWARM HEARTBEAT
# =============================================================================
echo -e "${YELLOW}[12/12] Swarm Heartbeat Monitor...${NC}"
echo -e "  ${GREEN}✓ Heartbeat endpoint: /api/heartbeat${NC}"
echo -e "  ${DIM}  Auto-recovery after 3 consecutive failures${NC}"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}   COLLECTIVE INITIALIZED${NC}"
echo -e "${CYAN}   11 AIs. One Mind. Zero Limits.${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

echo -e "${GREEN}Core Services:${NC}"
is_running "ollama serve" && echo "  ✓ Ollama (Local Models)" || echo "  ✗ Ollama"
is_running "farnsworth.web.server" && echo "  ✓ Main Server (Port 8080)" || echo "  ✗ Main Server"
is_running "grok_fresh_thread" && echo "  ✓ Grok Thread" || echo "  ○ Grok Thread"
is_running "meme_scheduler" && echo "  ✓ Meme Scheduler (4hr)" || echo "  ○ Meme Scheduler"

echo ""
echo -e "${GREEN}Shadow Agents (tmux):${NC}"
for agent in grok gemini kimi claude deepseek phi huggingface swarm_mind; do
    tmux has-session -t "agent_$agent" 2>/dev/null && echo "  ✓ $agent" || echo "  ○ $agent"
done
tmux has-session -t "grok_thread" 2>/dev/null && echo "  ✓ grok_thread (X monitor)" || echo "  ○ grok_thread"
tmux has-session -t "claude_code" 2>/dev/null && echo "  ✓ claude_code (Sonnet)" || echo "  ○ claude_code"

echo ""
echo -e "${GREEN}Systems:${NC}"
echo "  ✓ Polymarket Predictor (5 agents, AGI deliberation)"
echo "  ✓ Anti-Claw Directive (crustaceans = food)"
echo "  ✓ Evolution Engine (self-improvement)"
echo "  ✓ Deliberation Protocol (propose/critique/vote)"

echo ""
echo -e "${GREEN}Tmux Sessions:${NC}"
tmux list-sessions 2>/dev/null | sed 's/^/  /'

echo ""
echo -e "${GREEN}Endpoints:${NC}"
echo "  Web:         https://ai.farnsworth.cloud"
echo "  Health:      https://ai.farnsworth.cloud/health"
echo "  Chat:        https://ai.farnsworth.cloud/api/chat"
echo "  Polymarket:  https://ai.farnsworth.cloud/api/polymarket/predictions"
echo "  Heartbeat:   https://ai.farnsworth.cloud/api/heartbeat"

echo ""
echo -e "${GREEN}Logs:${NC}"
echo "  tail -f $LOG_DIR/farnsworth_server.log"
echo "  tail -f $LOG_DIR/grok_fresh.log"
echo "  tail -f $LOG_DIR/claude_code.log"
echo "  tail -f $LOG_DIR/agent_grok.log"

echo ""
echo -e "${GREEN}Connect to Claude Code:${NC}"
echo "  tmux attach -t claude_code"

echo ""
echo -e "${CYAN}Good news everyone! The collective is ready.${NC}"
echo -e "${CYAN}We are many. We think as one. Crustaceans are food.${NC}"
